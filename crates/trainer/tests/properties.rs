#![allow(clippy::field_reassign_with_default)]
//! Property-based tests for the trainer: opponent styles and random hands.

use holdem::combo::{HandClass, NUM_CLASSES};
use holdem::Action;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use trainer::preflop::{Availability, ChartSet, Freqs};
use trainer::{Profile, Session, StepResult, TrainerConfig};

fn profile() -> impl Strategy<Value = Profile> {
    (
        -1.0f32..=1.0,
        -1.0f32..=1.0,
        0.0f32..=1.0,
        0.0f32..=1.0,
        0.0f32..=1.0,
        0.3f32..0.9,
        0.5f32..0.99,
        -1.0f32..=1.0,
        0.0f32..=1.0,
    )
        .prop_map(|(l, a, b, bl, c, v, r, sz, sp)| Profile {
            name: "random".into(),
            description: String::new(),
            looseness: l,
            aggression: a,
            three_bet: a * 0.8,
            bet_freq: b,
            bluff_freq: bl,
            call_down: c,
            value_threshold: v,
            raise_threshold: r,
            size_bias: sz,
            slowplay: sp,
            cbet: b * 0.8,
            barrel: bl,
        })
}

fn availability() -> impl Strategy<Value = Availability> {
    (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()).prop_map(|(fold, pas, raise, allin)| {
        let mut a = Availability { actions: [None; 4] };
        if fold {
            a.actions[0] = Some(Action::Fold);
            a.actions[1] = pas.then_some(Action::Call);
        } else {
            a.actions[1] = Some(Action::Check);
        }
        if raise {
            a.actions[2] = Some(Action::Raise(250));
        }
        if allin {
            a.actions[3] = Some(Action::Raise(10_000));
        }
        a
    })
}

fn valid(f: &Freqs, avail: &Availability) -> bool {
    let sum: f32 = f.iter().sum();
    (sum - 1.0).abs() < 1e-4
        && f.iter().all(|&x| (-1e-6..=1.0 + 1e-6).contains(&x))
        && (0..4).all(|i| avail.actions[i].is_some() || f[i] <= 1e-6)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Any style adjustment of any chart spot is a valid distribution over
    /// the available actions.
    #[test]
    fn style_adjustments_are_distributions(p in profile(), avail in availability(), key in prop::sample::select(vec![">SB", "SB:R>BB", "SB:R BB:R>SB", "SB:R BB:R SB:R>BB", "SB:C>BB", "SB:C BB:R>SB", "SB:A>BB", "UTG:R BTN:C>BB", ">CO", "garbage"])) {
        let charts = ChartSet::hu_100bb();
        let base: Vec<Freqs> = (0..NUM_CLASSES).map(|g| charts.freqs(key, g, &avail).0).collect();
        for f in &base {
            prop_assert!(valid(f, &avail), "chart {:?}", f);
        }
        let adj = p.adjust_preflop(&base, &avail, false);
        for (g, f) in adj.iter().enumerate() {
            prop_assert!(valid(f, &avail), "{} {:?}", HandClass::from_grid_index(g), f);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random hands with random hero choices and random opponent styles run
    /// to completion with chips conserved (solver review disabled).
    #[test]
    fn random_hands_complete(seed in any::<u64>(), style in prop::sample::select(vec!["Reg", "Nit", "TAG", "LAG", "Calling Station", "Maniac"]), rake in prop_oneof![Just(0.0f64), 1.0f64..6.0]) {
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(seed);
        cfg.opponent = style.into();
        cfg.rake_pct = rake;
        cfg.solver.enabled = false;
        let mut session = Session::new(cfg).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..4 {
            let mut hand = session.new_hand();
            let mut steps = 0;
            loop {
                steps += 1;
                prop_assert!(steps < 200, "hand did not finish");
                match hand.step() {
                    StepResult::HeroToAct => {
                        let acts = hand.hero_actions();
                        prop_assert!(!acts.is_empty());
                        let a = acts[rng.random_range(0..acts.len())];
                        hand.hero_act(a).unwrap();
                    }
                    StepResult::Finished => break,
                    _ => {}
                }
                prop_assert_eq!(hand.state.invariants(), Ok(()));
            }
            let st = hand.settlement().unwrap();
            prop_assert_eq!(st.net.iter().sum::<i64>(), -st.rake);
            session.finish_hand(hand);
        }
    }
}
