use super::*;
use crate::realize::Realization;
use crate::solver::Options;
use crate::spec::GameSpec;

fn small_solve(deals: u64) -> Solver {
    let mut spec = GameSpec::new(3).with_stack(30.0);
    spec.rules.max_cold_callers = 1;
    let mut solver = Solver::new(Tree::build(&spec).unwrap(), Realization::neutral(), Options::default());
    solver.run(deals);
    solver
}

fn class(s: &str) -> usize {
    s.parse::<HandClass>().unwrap().grid_index()
}

#[test]
fn charts_cover_every_decision_with_distributions() {
    let solver = small_solve(200_000);
    let charts = to_charts(&solver, 50);
    assert_eq!(charts.players, 3);
    assert_eq!(charts.charts.len(), solver.tree().num_decisions);
    for d in solver.tree().decisions() {
        let chart = charts.get(&d.key).unwrap();
        for c in 0..NUM_CLASSES {
            let f = chart.class_freqs(c);
            assert!((f.iter().sum::<f32>() - 1.0).abs() < 1e-4, "{} {c}: {f:?}", d.key);
            for a in PreAction::ALL {
                assert!(f[a.index()] >= 0.0 && (d.avail.has(a) || f[a.index()] == 0.0), "{} {a:?}", d.key);
            }
        }
    }
    // The first decision sees every hand often; its chart is the average.
    let root = solver.tree().find(">BTN").unwrap();
    assert_eq!(charts.get(">BTN").unwrap().class_freqs(class("AKs")), solver.average(root, class("AKs")).0);
}

#[test]
fn hands_without_samples_use_the_fallback_rules() {
    let solver = small_solve(1000);
    let charts = to_charts(&solver, u64::MAX);
    for d in solver.tree().decisions() {
        for c in [class("AA"), class("72o"), class("T9s")] {
            let expected = d.avail.remap(fallback_freqs(&d.key, c, 3));
            assert_eq!(charts.get(&d.key).unwrap().class_freqs(c), expected, "{}", d.key);
        }
    }
}

/// Dropping the spots that are all fallback changes nothing the trainer
/// plays, and keeps every spot with a solved hand.
#[test]
fn solved_charts_drop_only_fallback_spots() {
    let solver = small_solve(20_000);
    let min = 50;
    let full = to_charts(&solver, min);
    let solved = solved_charts(&solver, min);
    assert!(solved.charts.len() < full.charts.len(), "some spots are never solved after 20k deals");
    assert!(!solved.charts.is_empty());
    assert_eq!(solved.players, 3);
    for d in solver.tree().decisions() {
        let any_solved = (0..NUM_CLASSES).any(|c| solver.average(d, c).1 >= min);
        assert_eq!(solved.get(&d.key).is_some(), any_solved, "{}", d.key);
        if let Some(chart) = solved.get(&d.key) {
            assert_eq!(chart.freqs, full.get(&d.key).unwrap().freqs);
        }
        for c in 0..NUM_CLASSES {
            assert_eq!(
                solved.freqs(&d.key, c, &d.avail).0,
                full.freqs(&d.key, c, &d.avail).0,
                "{} {c}",
                d.key
            );
        }
    }
    // A spot whose best-sampled class has exactly `min` samples is solved.
    let d = solver.tree().decisions().nth(5).unwrap();
    let most = (0..NUM_CLASSES).map(|c| solver.average(d, c).1).max().unwrap();
    assert!(most > 0);
    assert!(solved_charts(&solver, most).get(&d.key).is_some());
    assert!(solved_charts(&solver, most + 1).get(&d.key).is_none());
    // Nothing solved: no spots; everything solved: every spot.
    assert!(solved_charts(&solver, u64::MAX).charts.is_empty());
    assert_eq!(solved_charts(&small_solve(200_000), 1).charts.len(), solver.tree().num_decisions);
}

/// Charts survive the trainer's text format (4 decimals per frequency).
#[test]
fn charts_round_trip_through_the_text_format() {
    let charts = to_charts(&small_solve(200_000), 50);
    let parsed = ChartSet::parse(&charts.to_text("test"), 3).unwrap();
    assert_eq!(parsed.charts.len(), charts.charts.len());
    for (key, chart) in &charts.charts {
        let back = parsed.get(key).unwrap();
        for c in 0..NUM_CLASSES {
            for a in 0..4 {
                assert!((chart.class_freqs(c)[a] - back.class_freqs(c)[a]).abs() < 2e-4, "{key} {c} {a}");
            }
        }
    }
}

fn uniform_chart(freqs: Freqs) -> Chart {
    Chart { freqs: vec![freqs; NUM_CLASSES] }
}

#[test]
fn comparison_measures_distance_and_reach() {
    let tree = Tree::build(&GameSpec::new(2)).unwrap();
    let set = |open: Freqs, defend: Freqs| ChartSet {
        charts: [(">SB".to_string(), uniform_chart(open)), ("SB:R>BB".to_string(), uniform_chart(defend))]
            .into(),
        players: 2,
    };
    let reference = set([0.4, 0.0, 0.6, 0.0], [0.5, 0.3, 0.2, 0.0]);
    // Identical charts are at distance 0; the defense is reached 60% of the time.
    let same = compare(&tree, &reference, &reference);
    assert_eq!(same.iter().map(|d| d.key.as_str()).collect::<Vec<_>>(), [">SB", "SB:R>BB"]);
    assert!((same[0].reach - 1.0).abs() < 1e-6 && (same[1].reach - 0.6).abs() < 1e-6);
    assert!(same.iter().all(|d| d.distance.abs() < 1e-6));
    assert_eq!(mean_distance(&same), 0.0);
    // A model that opens 10 points less and never 3-bets.
    let model = set([0.5, 0.0, 0.5, 0.0], [0.5, 0.5, 0.0, 0.0]);
    let diff = compare(&tree, &model, &reference);
    assert!((diff[0].distance - 0.1).abs() < 1e-6 && (diff[1].distance - 0.2).abs() < 1e-6);
    assert!((diff[0].model[2] - 0.5).abs() < 1e-6 && (diff[0].reference[2] - 0.6).abs() < 1e-6);
    assert!((mean_distance(&diff) - (0.1 + 0.6 * 0.2) / 1.6).abs() < 1e-6);
    // Spots the reference lacks, or never reaches, are left out.
    let folds = set([1.0, 0.0, 0.0, 0.0], [0.5, 0.3, 0.2, 0.0]);
    assert_eq!(compare(&tree, &model, &folds).len(), 1);
    assert_eq!(mean_distance(&[]), 0.0);
}

/// The range that reaches a spot is the one the reference plays there:
/// distances only count the hands that get there.
#[test]
fn comparison_weights_hands_by_the_reference_range() {
    let tree = Tree::build(&GameSpec::new(2)).unwrap();
    // The reference opens only aces; facing a 3-bet it always calls.
    let mut open = uniform_chart([1.0, 0.0, 0.0, 0.0]);
    open.freqs[class("AA")] = [0.0, 0.0, 1.0, 0.0];
    let charts = |vs_3bet: Chart| ChartSet {
        charts: [
            (">SB".to_string(), open.clone()),
            ("SB:R>BB".to_string(), uniform_chart([0.0, 0.0, 1.0, 0.0])),
            ("SB:R BB:R>SB".to_string(), vs_3bet),
        ]
        .into(),
        players: 2,
    };
    let reference = charts(uniform_chart([0.0, 1.0, 0.0, 0.0]));
    // The model differs facing the 3-bet for every hand but aces.
    let mut vs_3bet = uniform_chart([1.0, 0.0, 0.0, 0.0]);
    vs_3bet.freqs[class("AA")] = [0.0, 1.0, 0.0, 0.0];
    let diff = compare(&tree, &charts(vs_3bet), &reference);
    let spot = diff.iter().find(|d| d.key == "SB:R BB:R>SB").unwrap();
    assert!(spot.distance.abs() < 1e-6, "only aces get there, and they play the same");
    assert!((spot.reach - 6.0 / 1326.0).abs() < 1e-6);
}

/// A player's range at a later spot doesn't depend on which other branch
/// was looked at before.
#[test]
fn comparison_keeps_each_branch_its_own_ranges() {
    let mut spec = GameSpec::new(2);
    spec.rules.allin_after_raises = 1;
    let tree = Tree::build(&spec).unwrap();
    // The small blind opens everything; the big blind 3-bets or jams.
    // Facing the 3-bet only aces continue (they jam).
    let mut vs_3bet = uniform_chart([1.0, 0.0, 0.0, 0.0]);
    vs_3bet.freqs[class("AA")] = [0.0, 0.0, 0.0, 1.0];
    let charts = |vs_jam: Chart| ChartSet {
        charts: [
            (">SB".to_string(), uniform_chart([0.0, 0.0, 1.0, 0.0])),
            ("SB:R>BB".to_string(), uniform_chart([0.0, 0.0, 0.5, 0.5])),
            ("SB:R BB:R>SB".to_string(), vs_3bet.clone()),
            ("SB:R BB:A>SB".to_string(), vs_jam),
        ]
        .into(),
        players: 2,
    };
    // Facing the jam the reference calls with everything, the model only
    // with aces. The whole opening range is there, so every hand but aces
    // counts as different.
    let mut only_aces = uniform_chart([1.0, 0.0, 0.0, 0.0]);
    only_aces.freqs[class("AA")] = [0.0, 1.0, 0.0, 0.0];
    let diff = compare(&tree, &charts(only_aces), &charts(uniform_chart([0.0, 1.0, 0.0, 0.0])));
    let spot = diff.iter().find(|d| d.key == "SB:R BB:A>SB").unwrap();
    assert!(
        (spot.distance - (1.0 - 6.0 / 1326.0)).abs() < 1e-5 && (spot.reach - 0.5).abs() < 1e-6,
        "{spot:?}"
    );
}

#[test]
fn first_in_frequencies_are_range_weighted() {
    let tree = Tree::build(&GameSpec::new(3)).unwrap();
    let mut open = uniform_chart([1.0, 0.0, 0.0, 0.0]);
    open.freqs[class("AA")] = [0.0, 0.0, 1.0, 0.0];
    open.freqs[class("AKs")] = [0.5, 0.0, 0.5, 0.0];
    let charts = ChartSet { charts: [(">BTN".to_string(), open)].into(), players: 3 };
    let freqs = first_in_frequencies(&tree, &charts);
    assert_eq!(freqs.len(), 1, "only spots with a chart, and only first-in ones");
    assert!((freqs[">BTN"][2] - (6.0 + 0.5 * 4.0) / 1326.0).abs() < 1e-6);
}

fn units(f: Freqs) -> [i64; 4] {
    f.map(|x| (x * 100.0).round() as i64)
}

#[test]
fn rounding_drops_noise_and_keeps_whole_percents() {
    // The 0.005% all-in is noise; the rest scales up and rounds to 100.
    let r = round_freqs([0.5, 0.2345, 0.26545, 0.00005], MIN_MIX);
    assert_eq!(units(r), [50, 23, 27, 0]);
    assert_eq!(r[3], 0.0);
    // Thirds: largest remainders, still 100 in total.
    let third = 1.0 / 3.0;
    assert_eq!(units(round_freqs([third, third, third, 0.0], MIN_MIX)).iter().sum::<i64>(), 100);
    // Pure strategies and 0s stay; a 1.9% mix goes to the others.
    assert_eq!(round_freqs([0.0, 0.0, 1.0, 0.0], MIN_MIX), [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(units(round_freqs([0.019, 0.0, 0.981, 0.0], MIN_MIX)), [0, 0, 100, 0]);
    assert_eq!(units(round_freqs([0.02, 0.0, 0.98, 0.0], MIN_MIX)), [2, 0, 98, 0]);
    // A threshold above every action keeps the largest.
    assert_eq!(round_freqs([0.3, 0.3, 0.4, 0.0], 0.5), [0.0, 0.0, 1.0, 0.0]);
    // Not a distribution: unchanged, never a panic.
    for bad in [[f32::NAN, 0.5, 0.5, 0.0], [-0.1, 0.6, 0.5, 0.0], [0.0; 4], [f32::INFINITY, 0.0, 0.0, 0.0]] {
        let r = round_freqs(bad, MIN_MIX);
        assert!(r.iter().zip(&bad).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())), "{bad:?} -> {r:?}");
    }
    let _ = round_freqs([0.5, 0.5, 0.0, 0.0], f32::NAN);
    let _ = round_freqs([0.5, 0.5, 0.0, 0.0], -1.0);
}

#[test]
fn rounding_charts_rounds_every_hand() {
    let mut charts = to_charts(&small_solve(20_000), 50);
    round_charts(&mut charts, MIN_MIX);
    for (key, chart) in &charts.charts {
        for f in &chart.freqs {
            assert_eq!(units(*f).iter().sum::<i64>(), 100, "{key}");
            assert!(f.iter().all(|&x| x == 0.0 || x >= MIN_MIX - 1e-6), "{key}: {f:?}");
        }
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]
    /// Rounded strategies are whole percents adding up to 100, keep 0s at
    /// 0 (unavailable actions stay unavailable), drop only mixes below the
    /// threshold, move nothing far, and rounding again changes nothing.
    #[test]
    fn rounded_strategies_are_close_whole_percents(
        w in proptest::array::uniform4(0u32..1000),
        zero in proptest::array::uniform4(proptest::bool::weighted(0.3)),
    ) {
        let w: [f32; 4] = std::array::from_fn(|a| if zero[a] { 0.0 } else { w[a] as f32 });
        let sum: f32 = w.iter().sum();
        proptest::prop_assume!(sum > 0.0);
        let f = w.map(|x| x / sum);
        let r = round_freqs(f, MIN_MIX);
        let u = units(r);
        proptest::prop_assert_eq!(u.iter().sum::<i64>(), 100);
        for a in 0..4 {
            proptest::prop_assert!((r[a] * 100.0 - u[a] as f32).abs() < 1e-4);
            if f[a] == 0.0 || f[a] < MIN_MIX {
                proptest::prop_assert_eq!(r[a], 0.0);
            } else {
                proptest::prop_assert!(r[a] >= MIN_MIX - 1e-6);
            }
            proptest::prop_assert!((r[a] - f[a]).abs() <= 3.0 * MIN_MIX + 0.011, "{:?} -> {:?}", f, r);
        }
        proptest::prop_assert_eq!(round_freqs(r, MIN_MIX), r);
    }
}
