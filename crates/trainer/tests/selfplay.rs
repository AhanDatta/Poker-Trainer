//! End-to-end: play hands with a scripted hero and check the solver review.

#![allow(clippy::field_reassign_with_default, clippy::while_let_loop)]

use holdem::Action;
use std::time::{Duration, Instant};
use trainer::{Session, StepResult, TrainerConfig, VillainRange};

fn quick_config(seed: u64) -> TrainerConfig {
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(seed);
    cfg.opponent = "Calling Station".into();
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
