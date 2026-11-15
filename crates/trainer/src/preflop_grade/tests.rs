use super::*;
use crate::preflop::{Availability, ChartSet};
use holdem::Action;
use proptest::prelude::*;
use std::collections::HashMap;

fn cards(s: &str) -> [Card; 2] {
    let v: Vec<Card> = s.split_whitespace().map(|c| c.parse().unwrap()).collect();
    [v[0], v[1]]
}

/// Fold, limp, raise to 2.5bb or all-in: a first-in decision.
fn first_in() -> Availability {
    Availability {
        actions: [
            Some(Action::Fold),
            Some(Action::Call),
            Some(Action::Raise(250)),
            Some(Action::Raise(10_000)),
        ],
    }
}

fn decision(seat: usize, position: &str, key: &str, action: PreAction) -> PreDecision {
    PreDecision { seat, position: position.into(), key: key.into(), avail: first_in(), action }
}

#[test]
fn verdicts_follow_the_chart_frequency_of_the_chosen_action() {
    let f = |fold: f32, call: f32, raise: f32| [fold, call, raise, 1.0 - fold - call - raise];
    use PreAction::*;
    // The chart's main action, or one played at least half the time: Best.
    assert_eq!(verdict_of(&f(0.0, 0.0, 1.0), Raise), Verdict::Best);
    assert_eq!(verdict_of(&f(0.5, 0.0, 0.5), Fold), Verdict::Best);
    assert_eq!(verdict_of(&f(0.4, 0.3, 0.3), Fold), Verdict::Best);
    // Mixed in less often: by frequency.
    assert_eq!(verdict_of(&f(0.6, 0.0, 0.4), Raise), Verdict::Good);
    assert_eq!(verdict_of(&f(0.85, 0.0, 0.15), Raise), Verdict::Good);
    assert_eq!(verdict_of(&f(0.86, 0.0, 0.14), Raise), Verdict::Inaccuracy);
    assert_eq!(verdict_of(&f(0.95, 0.0, 0.05), Raise), Verdict::Inaccuracy);
    assert_eq!(verdict_of(&f(0.96, 0.0, 0.04), Raise), Verdict::Mistake);
    assert_eq!(verdict_of(&f(0.99, 0.0, 0.01), Raise), Verdict::Mistake);
    assert_eq!(verdict_of(&f(0.995, 0.0, 0.005), Raise), Verdict::Blunder);
    // Never played: Blunder, even next to an action the chart never plays either.
    assert_eq!(verdict_of(&f(1.0, 0.0, 0.0), Raise), Verdict::Blunder);
    assert_eq!(verdict_of(&f(1.0, 0.0, 0.0), AllIn), Verdict::Blunder);
}

#[test]
fn hero_decisions_are_graded_against_the_built_in_charts() {
    let charts = ChartSet::builtin(6);
    // UTG (seat 3) first in: aces raise, 72o folds.
    let raise_aa = [decision(3, "UTG", ">UTG", PreAction::Raise)];
    let g = grade(&charts, &raise_aa, 3, cards("Ah As"));
    assert_eq!(g.len(), 1);
    assert_eq!((g[0].decision, g[0].class, g[0].chosen), (0, 0, PreAction::Raise));
    assert_eq!(g[0].verdict, Some(Verdict::Best));
    assert_eq!(g[0].chart_action(), Some(PreAction::Raise));
    assert!(g[0].chosen_freq().unwrap() > 0.99);
    let raise_72 = grade(&charts, &raise_aa, 3, cards("7h 2c"));
    assert_eq!(raise_72[0].verdict, Some(Verdict::Blunder));
    assert_eq!(raise_72[0].chart_action(), Some(PreAction::Fold));
    let fold_72 = grade(&charts, &[decision(3, "UTG", ">UTG", PreAction::Fold)], 3, cards("7h 2c"));
    assert_eq!(fold_72[0].verdict, Some(Verdict::Best));
    // Only the hero's decisions, with their index in the hand.
    let line = [
        decision(3, "UTG", ">UTG", PreAction::Fold),
        decision(4, "HJ", ">HJ", PreAction::Raise),
        decision(0, "BTN", "HJ:R>BTN", PreAction::Fold),
    ];
    let g = grade(&charts, &line, 4, cards("Kd Kc"));
    assert_eq!(g.iter().map(|g| g.decision).collect::<Vec<_>>(), vec![1]);
    assert_eq!(g[0].verdict, Some(Verdict::Best));
    assert!(grade(&charts, &line, 2, cards("Kd Kc")).is_empty());
}

#[test]
fn spots_without_a_chart_are_not_graded() {
    let empty = ChartSet { charts: HashMap::new(), players: 6 };
    let g = grade(&empty, &[decision(3, "UTG", ">UTG", PreAction::Raise)], 3, cards("Ah As"));
    assert_eq!((g[0].chart, g[0].verdict, g[0].chosen_freq(), g[0].chart_action()), (None, None, None, None));
    // A chart set covers some spots and not others.
    let charts = ChartSet::builtin(6);
    let odd = decision(3, "UTG", "UTG:C HJ:C CO:C BTN:C SB:C>BB", PreAction::Passive);
    assert_eq!(grade(&charts, &[odd], 3, cards("Ah As"))[0].verdict, None);
}

/// Grades survive a save (JSON) unchanged.
#[test]
fn grades_round_trip_through_json() {
    let charts = ChartSet::builtin(6);
    let g = grade(&charts, &[decision(3, "UTG", ">UTG", PreAction::Passive)], 3, cards("Jh Td"));
    let text = serde_json::to_string(&g).unwrap();
    let back: Vec<PreflopGrade> = serde_json::from_str(&text).unwrap();
    assert_eq!(back, g);
}

proptest! {
    /// Choosing an action the chart plays more never grades worse, and any
    /// frequencies (even nonsense ones) give a verdict without panicking.
    #[test]
    fn more_frequent_actions_never_grade_worse(
        w in proptest::array::uniform4(0u32..1000),
        chosen in 0usize..4,
        boost in 0u32..1000,
    ) {
        let total: u32 = w.iter().sum::<u32>().max(1);
        let f: Freqs = std::array::from_fn(|a| w[a] as f32 / total as f32);
        let mut more = w;
        more[chosen] += boost;
        let total_more: u32 = more.iter().sum::<u32>().max(1);
        let g: Freqs = std::array::from_fn(|a| more[a] as f32 / total_more as f32);
        let a = PreAction::ALL[chosen];
        prop_assert!(verdict_of(&g, a) <= verdict_of(&f, a), "{:?} {:?} -> {:?} {:?}", f, g, verdict_of(&f, a), verdict_of(&g, a));
    }

    #[test]
    fn odd_frequencies_never_panic(f in proptest::array::uniform4(proptest::num::f32::ANY), chosen in 0usize..4) {
        let _ = verdict_of(&f, PreAction::ALL[chosen]);
    }
}
