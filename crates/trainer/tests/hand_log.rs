//! The hand log: a hand that ends in an all-in runout (on a dealt card, not
//! an action) logs its showdown and winners like any other.

#![allow(clippy::field_reassign_with_default)]

use holdem::Action;
use trainer::{Session, StepResult, TrainerConfig};

#[test]
fn all_in_runouts_log_the_showdown_and_winners() {
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(21);
    cfg.players = 2;
    cfg.opponent = "Calling Station".into();
    cfg.solver.enabled = false;
    let mut session = Session::new(cfg).unwrap();
    for _ in 0..200 {
        let mut hand = session.new_hand();
        // The hero moves all-in whenever it can.
        while hand.run_until_hero() == StepResult::HeroToAct {
            let acts = hand.hero_actions();
            let jam = acts.iter().copied().rfind(|a| matches!(a, Action::Raise(_)));
            let call = acts.iter().copied().find(|a| matches!(a, Action::Call | Action::Check));
            hand.hero_act(jam.or(call).unwrap_or(acts[0])).unwrap();
        }
        let live = hand.state.players_in_hand();
        let ran_out =
            live.len() == 2 && hand.state.board.len() == 5 && live.iter().all(|&p| hand.state.is_all_in(p));
        if !ran_out {
            session.finish_hand(hand);
            continue;
        }
        let shows: Vec<&str> =
            hand.log.iter().map(|e| e.text.as_str()).filter(|t| t.contains(" shows ")).collect();
        let wins: Vec<&str> =
            hand.log.iter().map(|e| e.text.as_str()).filter(|t| t.contains(" wins ")).collect();
        assert_eq!(shows.len(), 2, "{:#?}", hand.log);
        assert!((1..=2).contains(&wins.len()), "{:#?}", hand.log);
        // Logged once, after the river.
        let river = hand.log.iter().position(|e| e.text.starts_with("River")).expect("a river line");
        assert!(hand.log.iter().position(|e| e.text.contains(" shows ")).unwrap() > river);
        assert!(!hand.preflop_grades().is_empty());
        return;
    }
    panic!("no all-in runout in 200 hands");
}
