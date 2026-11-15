//! Preflop grades end to end: every hero preflop decision at a charted spot
//! is graded when the hand ends, counted in the session's own stats, saved,
//! and graded again when an old save without grades is loaded.

#![allow(clippy::field_reassign_with_default)]

use holdem::{Action, HandClass, Street};
use std::path::PathBuf;
use trainer::preflop::PreAction;
use trainer::preflop_grade::verdict_of;
use trainer::save::{self, SessionStore};
use trainer::{LiveHand, Session, StepResult, TrainerConfig};

fn config(seed: u64) -> TrainerConfig {
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(seed);
    cfg.players = 6;
    // No solver review: these tests are about preflop.
    cfg.solver.enabled = false;
    cfg
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("poker-trainer-pg-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Plays a hand where the hero picks preflop actions by its roll (so it
/// folds, calls and raises, on and off chart), and checks or calls after.
fn play(session: &mut Session) -> LiveHand {
    let mut hand = session.new_hand();
    loop {
        match hand.run_until_hero() {
            StepResult::HeroToAct => {
                let acts = hand.hero_actions();
                let roll = hand.current_roll().unwrap();
                let fold = acts.iter().copied().find(|a| *a == Action::Fold);
                let raise = acts.iter().copied().find(|a| matches!(a, Action::Raise(_)));
                let passive = acts.iter().copied().find(|a| matches!(a, Action::Check | Action::Call));
                let pick = match (hand.state.street == Street::Preflop, roll) {
                    (true, 0..=39) => fold.or(passive),
                    (true, 40..=69) => passive,
                    (true, _) => raise.or(passive),
                    (false, _) => passive,
                };
                hand.hero_act(pick.unwrap_or(acts[0])).unwrap();
            }
            StepResult::Finished => break,
            _ => unreachable!(),
        }
    }
    hand
}

#[test]
fn every_hero_preflop_decision_is_graded_against_the_charts() {
    let mut session = Session::new(config(11)).unwrap();
    let charts = session.charts().clone();
    let (mut graded, mut in_chart, mut ended_preflop) = (0, 0, 0);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..120 {
        let hand = play(&mut session);
        let grades = hand.preflop_grades();
        let hero_decisions: Vec<usize> =
            (0..hand.preflop.len()).filter(|&i| hand.preflop[i].seat == hand.hero).collect();
        assert_eq!(grades.iter().map(|g| g.decision).collect::<Vec<_>>(), hero_decisions);
        let hole = hand.state.hole[hand.hero];
        let class = HandClass::of_cards(hole[0], hole[1]).grid_index();
        for g in grades {
            let d = &hand.preflop[g.decision];
            assert_eq!((g.class, g.chosen), (class, d.action));
            let (freqs, from_chart) = charts.freqs(&d.key, class, &d.avail);
            assert_eq!(g.chart.is_some(), from_chart, "{}", d.key);
            assert_eq!(g.verdict, from_chart.then(|| verdict_of(&freqs, d.action)));
            if let Some(v) = g.verdict {
                graded += 1;
                in_chart += usize::from(matches!(v, trainer::Verdict::Best | trainer::Verdict::Good));
                seen.insert(v);
            }
        }
        if hand.state.board.is_empty() && !grades.is_empty() {
            ended_preflop += 1;
        }
        session.finish_hand(hand);
    }
    session.refresh();
    let st = session.stats();
    assert_eq!(st.preflop_decisions, graded);
    assert_eq!(st.preflop_verdicts.values().sum::<usize>(), graded);
    assert_eq!(st.decisions, 0, "preflop grades don't count as solver-graded decisions");
    assert!(graded > 100 && ended_preflop > 20, "{graded} graded, {ended_preflop} ended preflop");
    // The hero's random choices land both on and off the charts.
    assert!(seen.contains(&trainer::Verdict::Best) && seen.contains(&trainer::Verdict::Blunder), "{seen:?}");
    // Accuracy: the share graded Best or Good.
    assert!(seen.contains(&trainer::Verdict::Good), "{seen:?}");
    assert!((st.preflop_accuracy() - in_chart as f64 / graded as f64).abs() < 1e-12);
}

#[test]
fn grades_are_saved_and_old_saves_are_graded_on_load() {
    let root = scratch("saves");
    let mut session = Session::new(config(12)).unwrap();
    session.start_saving(root.clone());
    for _ in 0..12 {
        let hand = play(&mut session);
        session.finish_hand(hand);
        session.refresh();
    }
    session.flush(std::time::Duration::from_secs(1));
    let id = session.save_id().unwrap().to_string();
    let loaded = SessionStore::open(&root, &id).unwrap().load().unwrap();
    assert_eq!(loaded.hands.len(), session.history.len());
    for (saved, played) in loaded.hands.iter().zip(&session.history) {
        assert_eq!(saved.preflop_grades.as_deref(), Some(played.preflop_grades()));
    }
    // Continuing the session keeps the grades and the stats.
    let resumed = Session::resume(loaded.clone(), root.clone(), &config(12)).unwrap();
    for (a, b) in resumed.history.iter().zip(&session.history) {
        assert_eq!(a.preflop_grades(), b.preflop_grades());
    }
    assert_eq!(resumed.stats().preflop_decisions, session.stats().preflop_decisions);
    // A save from before preflop grading: the field is missing, and the
    // hand is graded when loaded.
    let mut old = serde_json::to_value(&loaded.hands[0]).unwrap();
    old.as_object_mut().unwrap().remove("preflop_grades");
    let old: save::SavedHand = serde_json::from_value(old).unwrap();
    assert_eq!(old.preflop_grades, None);
    let hand = LiveHand::from_saved(old, session.config.menu.clone(), session.charts().clone());
    assert_eq!(hand.preflop_grades(), session.history[0].preflop_grades());
    // Saved grades are kept as they were, whatever charts load them.
    let none = std::sync::Arc::new(trainer::preflop::ChartSet { charts: Default::default(), players: 6 });
    let graded = loaded
        .hands
        .iter()
        .find(|h| h.preflop_grades.as_ref().is_some_and(|g| g.iter().any(|g| g.verdict.is_some())))
        .unwrap();
    let kept = LiveHand::from_saved(graded.clone(), session.config.menu.clone(), none.clone());
    assert_eq!(Some(kept.preflop_grades()), graded.preflop_grades.as_deref());
    // ... while an old save is graded with the charts at hand (none here).
    let mut old = graded.clone();
    old.preflop_grades = None;
    let regraded = LiveHand::from_saved(old, session.config.menu.clone(), none);
    assert!(
        !regraded.preflop_grades().is_empty()
            && regraded.preflop_grades().iter().all(|g| g.verdict.is_none())
    );
    let _ = std::fs::remove_dir_all(&root);
    // The hero's actions are recorded as abstract preflop actions.
    assert!(session.history.iter().flat_map(|h| h.preflop_grades()).any(|g| g.chosen == PreAction::Fold));
}

/// An old save of a session whose table size changed: continuing it grades
/// each hand with the charts of its own table size.
#[test]
fn old_saves_are_graded_with_their_own_table_size() {
    let root = scratch("sizes");
    let mut session = Session::new(config(13)).unwrap();
    session.start_saving(root.clone());
    for i in 0..30 {
        if i == 15 {
            session.set_players(9).unwrap();
        }
        let hand = play(&mut session);
        session.finish_hand(hand);
        session.refresh();
    }
    session.flush(std::time::Duration::from_secs(1));
    let id = session.save_id().unwrap().to_string();
    let mut loaded = SessionStore::open(&root, &id).unwrap().load().unwrap();
    let sizes: std::collections::BTreeSet<usize> =
        loaded.hands.iter().map(|h| h.state.num_players()).collect();
    assert_eq!(sizes.len(), 2, "both table sizes are in the save");
    for h in &mut loaded.hands {
        h.preflop_grades = None;
    }
    let resumed = Session::resume(loaded, root.clone(), &config(13)).unwrap();
    assert_eq!(resumed.history.len(), session.history.len());
    for (a, b) in resumed.history.iter().zip(&session.history) {
        assert_eq!(a.preflop_grades(), b.preflop_grades(), "hand {}", a.id);
    }
    let _ = std::fs::remove_dir_all(&root);
}
