//! Saved sessions and the frequency score, end to end: play reviewed hands,
//! save them, continue the session from disk.

#![allow(clippy::field_reassign_with_default)]

use holdem::{Action, Street};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use trainer::save::{self, SessionStore};
use trainer::{LiveHand, Session, StepResult, TrainerConfig, Variant};

fn quick_config(seed: u64) -> TrainerConfig {
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(seed);
    cfg.opponent = "Calling Station".into();
    cfg.solver.flop_mode = trainer::FlopMode::Fast;
    cfg.solver.flop_turn_samples = 3;
    cfg.solver.flop_river_samples = 2;
    cfg.solver.max_iterations = 40;
    cfg.solver.target_exploitability_pct = 5.0;
    cfg
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("poker-trainer-it-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Plays one hand with a hero that bets when it can on odd streets, and
/// waits for its review.
fn play(session: &mut Session) -> LiveHand {
    let mut hand = session.new_hand();
    loop {
        match hand.run_until_hero() {
            StepResult::HeroToAct => {
                let acts = hand.hero_actions();
                let roll = hand.current_roll().expect("a roll at every hero decision");
                assert!(roll < 100);
                let raise = acts.iter().copied().find(|a| matches!(a, Action::Raise(_)));
                let passive = acts.iter().copied().find(|a| matches!(a, Action::Check | Action::Call));
                let pick = if hand.state.street != Street::Preflop && roll < 40 {
                    raise.or(passive)
                } else {
                    passive
                };
                hand.hero_act(pick.unwrap_or(acts[0])).unwrap();
            }
            StepResult::Finished => break,
            _ => unreachable!(),
        }
    }
    assert_eq!(hand.current_roll(), None);
    if let Some(r) = hand.review() {
        let t = Instant::now();
        while !r.is_finished() {
            assert!(t.elapsed() < Duration::from_secs(300), "review timed out");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    hand
}

/// Stats recomputed from scratch, as the session did before caching.
fn naive_stats(s: &Session) -> (usize, i64, usize, f64) {
    let mut out = (0, 0, 0, 0.0);
    for h in &s.history {
        out.0 += 1;
        out.1 += h.hero_net().unwrap();
        if let Some(r) = h.review() {
            let st = r.status();
            let gto = st.decisions.iter().any(|d| d.variant == Variant::Gto);
            for d in st.decisions.iter().filter(|d| !gto || d.variant == Variant::Gto) {
                out.2 += 1;
                out.3 += d.ev_loss_pct as f64;
            }
        }
    }
    out
}

#[test]
fn sessions_save_and_continue() {
    let root = scratch("continue");
    let mut session = Session::new(quick_config(5)).unwrap();
    session.start_saving(root.clone());
    let mut graded = 0;
    for _ in 0..40 {
        let hand = play(&mut session);
        if let Some(r) = hand.review() {
            let st = r.status();
            for d in &st.decisions {
                // Spot info: the decision's place in the hand and its categories.
                assert_eq!(d.spot.players, 2);
                assert_eq!(d.spot.categories.len(), d.actions.len());
                let rec = &hand.state.history[d.spot.history_index];
                assert_eq!(rec.player, hand.hero);
                assert_eq!(rec.street, d.street);
                assert_eq!(rec.action, d.actions[d.chosen]);
                let raises_before = hand.state.history[..d.spot.history_index]
                    .iter()
                    .filter(|r| r.street == d.street && matches!(r.action, Action::Raise(_)))
                    .count();
                assert_eq!(d.spot.facing as usize, raises_before);
            }
            graded += st.decisions.len();
        }
        session.finish_hand(hand);
        session.refresh();
        let n = naive_stats(&session);
        let s = session.stats();
        assert_eq!((s.hands, s.net_chips, s.decisions), (n.0, n.1, n.2));
        assert!((s.total_loss_pct - n.3).abs() < 1e-6);
        if graded >= 6 && session.history.len() >= 8 {
            break;
        }
    }
    assert!(graded >= 6, "too few graded decisions");
    let report = session.report().expect("a frequency score").clone();
    assert!(report.decisions > 0 && report.rows.iter().all(|r| r.decisions > 0));

    // Everything finished is on disk.
    let id = session.save_id().expect("saved").to_string();
    let list = save::list_sessions(&root);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, id);
    assert_eq!(list[0].summary.hands, session.history.len());
    assert_eq!(list[0].summary.score, Some(report.calibration.score()));

    // Continue it.
    let loaded = SessionStore::open(&root, &id).unwrap().load().unwrap();
    assert_eq!(loaded.skipped, 0);
    let mut resumed = Session::resume(loaded, root.clone(), &quick_config(6)).unwrap();
    assert_eq!(resumed.history.len(), session.history.len());
    let (a, b) = (session.stats(), resumed.stats());
    assert_eq!(
        (a.hands, a.net_chips, a.decisions, &a.verdicts),
        (b.hands, b.net_chips, b.decisions, &b.verdicts)
    );
    assert_eq!(resumed.report(), Some(&report), "same decisions, same score");
    for (x, y) in session.history.iter().zip(&resumed.history) {
        assert_eq!(x.id, y.id);
        assert_eq!(x.log, y.log);
        assert_eq!(x.hero_net(), y.hero_net());
        for i in 0..8 {
            assert_eq!(x.roll(i), y.roll(i), "rolls are saved");
        }
        match (x.review(), y.review()) {
            (None, None) => {}
            (Some(rx), Some(ry)) => {
                let (sx, sy) = (rx.status(), ry.status());
                assert!(sy.finished);
                assert_eq!(sx.decisions.len(), sy.decisions.len());
                for (dx, dy) in sx.decisions.iter().zip(&sy.decisions) {
                    assert_eq!(dx.hero_strategy, dy.hero_strategy);
                    assert_eq!(dx.hero_ev, dy.hero_ev);
                    assert_eq!(dx.spot, dy.spot);
                    assert_eq!((dx.chosen, dx.verdict, &dx.labels), (dy.chosen, dy.verdict, &dy.labels));
                    // Grids are saved to 1/10000.
                    for (cx, cy) in dx.grid.iter().zip(&dy.grid) {
                        match (cx, cy) {
                            (None, None) => {}
                            (Some(cx), Some(cy)) => {
                                assert_eq!(cx.weight, cy.weight);
                                for (p, q) in cx.strategy.iter().zip(&cy.strategy) {
                                    assert!((p - q).abs() <= 6e-5, "{p} vs {q}");
                                }
                            }
                            _ => panic!("grid cell lost"),
                        }
                    }
                }
            }
            _ => panic!("review lost"),
        }
    }

    // New hands continue the numbering and are appended.
    let before = resumed.history.len();
    let last_id = resumed.history.last().unwrap().id;
    let hand = play(&mut resumed);
    assert_eq!(hand.id, last_id + 1);
    resumed.finish_hand(hand);
    resumed.flush(Duration::from_secs(5));
    let again = SessionStore::open(&root, &id).unwrap().load().unwrap();
    assert_eq!(again.hands.len(), before + 1);
    assert_eq!(save::list_sessions(&root)[0].summary.hands, before + 1);

    save::delete_session(&root, &id).unwrap();
    assert!(save::list_sessions(&root).is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

/// Quitting while a review runs saves the hand with the review cut short.
#[test]
fn flush_saves_unfinished_reviews() {
    let root = scratch("flush");
    let mut cfg = quick_config(8);
    cfg.solver.max_iterations = 400;
    cfg.solver.target_exploitability_pct = 0.01;
    let mut session = Session::new(cfg).unwrap();
    session.start_saving(root.clone());
    // Play until a hand reaches the flop; don't wait for its review.
    for _ in 0..30 {
        let mut hand = session.new_hand();
        while hand.run_until_hero() == StepResult::HeroToAct {
            let acts = hand.hero_actions();
            let passive = acts.iter().copied().find(|a| matches!(a, Action::Check | Action::Call));
            hand.hero_act(passive.unwrap_or(acts[0])).unwrap();
        }
        let reviewed = hand.review().is_some();
        session.finish_hand(hand);
        if reviewed {
            break;
        }
    }
    let n = session.history.len();
    session.flush(Duration::from_secs(10));
    let id = session.save_id().unwrap().to_string();
    let loaded = SessionStore::open(&root, &id).unwrap().load().unwrap();
    assert_eq!(loaded.hands.len(), n);
    let last = loaded.hands.last().unwrap().review.as_ref().expect("review saved");
    assert!(last.cancelled || last.error.is_none());
    let resumed = Session::resume(loaded, root.clone(), &TrainerConfig::default()).unwrap();
    assert!(resumed.history.iter().all(|h| h.review().is_none_or(|r| r.is_finished())));
    assert_eq!(resumed.stats().reviews_pending, 0);
    std::fs::remove_dir_all(root).unwrap();
}
