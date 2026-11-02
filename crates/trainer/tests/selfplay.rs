//! End-to-end: play hands with a scripted hero and check the solver review.

#![allow(clippy::field_reassign_with_default, clippy::while_let_loop)]

use holdem::Action;
use std::time::{Duration, Instant};
use trainer::{Session, StepResult, TrainerConfig, VillainRange};

fn quick_config(seed: u64) -> TrainerConfig {
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(seed);
    cfg.opponent = "Calling Station".into();
    cfg.solver.flop_mode = trainer::FlopMode::Fast;
    cfg.solver.flop_turn_samples = 4;
    cfg.solver.flop_river_samples = 3;
    cfg.solver.max_iterations = 60;
    cfg.solver.target_exploitability_pct = 3.0;
    cfg.solver.villain_range = VillainRange::Both;
    cfg
}

/// Scripted hero: calls/checks preflop, then alternates between betting
/// and checking/calling postflop.
fn hero_choice(actions: &[Action], street_idx: usize) -> Action {
    let passive = actions.iter().copied().find(|a| matches!(a, Action::Check | Action::Call));
    let raise = actions.iter().copied().find(|a| matches!(a, Action::Raise(_)));
    if street_idx == 0 {
        return passive.unwrap_or(actions[0]);
    }
    if street_idx % 2 == 1 {
        raise.or(passive).unwrap_or(actions[0])
    } else {
        passive.unwrap_or(actions[0])
    }
}

#[test]
fn hands_are_played_and_reviewed() {
    let mut session = Session::new(quick_config(42)).unwrap();
    let mut reviewed = 0;
    for _ in 0..30 {
        let mut hand = session.new_hand();
        loop {
            match hand.run_until_hero() {
                StepResult::HeroToAct => {
                    let actions = hand.hero_actions();
                    assert!(!actions.is_empty());
                    let street_idx = hand.state.board.len().saturating_sub(2);
                    hand.hero_act(hero_choice(&actions, street_idx)).unwrap();
                }
                StepResult::Finished => break,
                _ => unreachable!(),
            }
        }
        let net = hand.hero_net().unwrap();
        assert!(net.abs() <= 10_000);
        if let Some(review) = hand.review() {
            let start = Instant::now();
            while !review.is_finished() {
                assert!(
                    start.elapsed() < Duration::from_secs(300),
                    "review timed out: {:?}",
                    review.status().stage
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            let st = review.status();
            assert!(st.error.is_none(), "review error: {:?} notes {:?}", st.error, st.notes);
            let hero_postflop = hand
                .state
                .history
                .iter()
                .filter(|r| r.player == hand.hero && r.street != holdem::Street::Preflop)
                .count();
            // Both variants grade every postflop hero decision.
            assert_eq!(st.decisions.len(), 2 * hero_postflop, "{:#?}", st.notes);
            for d in &st.decisions {
                let mix: f32 = d.hero_strategy.iter().sum();
                assert!((mix - 1.0).abs() < 1e-3);
                assert!(d.ev_loss >= 0.0);
                assert!(d.chosen < d.actions.len());
            }
            if hero_postflop > 0 {
                reviewed += 1;
            }
        }
        session.finish_hand(hand);
        if reviewed >= 2 {
            break;
        }
    }
    assert!(reviewed >= 2, "not enough reviewed hands");
    let stats = session.stats();
    assert!(stats.decisions > 0);
}

/// 6-max and 9-max: hands complete, heads-up pots between the hero and one
/// opponent are reviewed (with positions taken from the postflop order).
#[test]
fn multiway_tables_play_and_review() {
    for players in [6usize, 9] {
        let mut cfg = quick_config(7 + players as u64);
        cfg.players = players;
        cfg.opponent = "random".into();
        cfg.solver.villain_range = VillainRange::Gto;
        cfg.solver.flop_mode = trainer::FlopMode::Fast;
        cfg.solver.multiway_turn_samples = 3;
        cfg.solver.multiway_river_samples = 2;
        cfg.solver.multiway_target_pct = 5.0;
        let mut session = Session::new(cfg).unwrap();
        let mut reviewed = 0;
        let mut three_way = 0;
        for _ in 0..150 {
            let mut hand = session.new_hand();
            loop {
                match hand.run_until_hero() {
                    StepResult::HeroToAct => {
                        let actions = hand.hero_actions();
                        // Defend wide preflop so hands reach the flop.
                        let street_idx = hand.state.board.len().saturating_sub(2);
                        hand.hero_act(hero_choice(&actions, street_idx)).unwrap();
                    }
                    StepResult::Finished => break,
                    _ => unreachable!(),
                }
            }
            assert_eq!(hand.state.invariants(), Ok(()));
            if let Some(review) = hand.review() {
                let start = Instant::now();
                while !review.is_finished() {
                    assert!(
                        start.elapsed() < Duration::from_secs(300),
                        "review timed out: {}",
                        review.status().stage
                    );
                    std::thread::sleep(Duration::from_millis(50));
                }
                let st = review.status();
                assert!(st.error.is_none(), "{:?} {:?}", st.error, st.notes);
                for d in &st.decisions {
                    let mix: f32 = d.hero_strategy.iter().sum();
                    assert!((mix - 1.0).abs() < 1e-3);
                    assert!(d.ev_loss >= 0.0 && d.chosen < d.actions.len());
                }
                if st.decisions.iter().any(|d| d.notes.iter().any(|n| n.starts_with("3-way"))) {
                    three_way += 1;
                } else if !st.decisions.is_empty() {
                    reviewed += 1;
                }
            }
            session.finish_hand(hand);
            if reviewed >= 1 && three_way >= 1 {
                break;
            }
        }
        assert!(reviewed >= 1, "{players}-max: no heads-up pot was reviewed");
        assert!(three_way >= 1, "{players}-max: no 3-way pot was reviewed");
    }
}

/// Starting the next hand stops the previous hands' solves (they would keep
/// the CPU and the solve queue busy), promptly and visibly.
#[test]
fn next_hand_cancels_running_reviews() {
    for players in [2, 6] {
        let mut cfg = quick_config(7);
        cfg.players = players;
        cfg.opponent = "Calling Station".into();
        // Solves that never reach their target: they run until cancelled.
        cfg.solver.target_exploitability_pct = 0.0;
        cfg.solver.multiway_target_pct = 0.0;
        cfg.solver.max_iterations = 1_000_000;
        cfg.solver.villain_range = VillainRange::Gto;
        let mut session = Session::new(cfg).unwrap();
        let mut tested = false;
        for _ in 0..200 {
            let mut hand = session.new_hand();
            while hand.run_until_hero() == StepResult::HeroToAct {
                let acts = hand.hero_actions();
                let passive = acts.iter().copied().find(|a| matches!(a, Action::Check | Action::Call));
                hand.hero_act(passive.unwrap_or(acts[0])).unwrap();
            }
            let Some(review) = hand.review() else {
                session.finish_hand(hand);
                continue;
            };
            // Let the review get into a solve.
            let start = Instant::now();
            while !review.status().stage.contains("solving") && !review.is_finished() {
                assert!(start.elapsed() < Duration::from_secs(120), "review never started solving");
                std::thread::sleep(Duration::from_millis(20));
            }
            // At 6-max, test a 3-way solve specifically.
            if review.is_finished() || (players > 2 && !review.status().stage.contains("3-way")) {
                review.cancel();
                session.finish_hand(hand);
                continue;
            }
            let id = hand.id;
            session.finish_hand(hand);
            let _next = session.new_hand();
            let prev = session.history.iter().find(|h| h.id == id).unwrap().review().unwrap();
            let t = Instant::now();
            while !prev.is_finished() {
                assert!(
                    t.elapsed() < Duration::from_secs(20),
                    "{players} players: previous review still running"
                );
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(prev.status().cancelled, "{:?}", prev.status().stage);
            tested = true;
            break;
        }
        assert!(tested, "{players} players: no hand reached a running solve");
    }
}
