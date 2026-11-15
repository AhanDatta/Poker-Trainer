use super::*;
use crate::spec::GameSpec;
use trainer::preflop::ChartSet;

fn small_job(deals: u64) -> Job {
    let mut spec = GameSpec::new(3).with_stack(30.0).with_rake(0.05, 3.0);
    spec.rules.max_cold_callers = 1;
    Job {
        spec,
        factors: Realization::positional(1.05, 0.8, 0.9),
        factors_name: "flat test factors".into(),
        deals,
        threads: 1,
        seed: 3,
    }
}

#[test]
fn solving_writes_the_solved_spots_with_a_header() {
    // Enough deals for the first decisions to pass the 1000-update minimum.
    let job = small_job(600_000);
    let mut epochs = Vec::new();
    let out = solve(&job, |p| epochs.push(*p)).unwrap();
    assert_eq!(epochs.len(), EPOCHS as usize);
    assert!(epochs.windows(2).all(|w| w[0].deals < w[1].deals && w[0].seconds <= w[1].seconds));
    assert_eq!(epochs.last().unwrap().deals, job.deals);
    // Only solved spots are written; the first decision always is.
    assert!(out.charts.charts.len() <= out.decisions);
    assert!(out.charts.get(">BTN").is_some());
    assert_eq!(out.charts.players, 3);
    let header = header(&job, &out);
    for needle in [
        "solve-preflop",
        "3 players",
        "30bb",
        "rake 5% capped at 3bb",
        "flat test factors",
        "seed 3",
        "1000 regret updates",
        "BTN",
        "SB",
        "BB",
    ] {
        assert!(header.contains(needle), "{needle:?} not in\n{header}");
    }
    // Whole percents, as the file writes them.
    for chart in out.charts.charts.values() {
        for f in &chart.freqs {
            assert!(f.iter().all(|x| (x * 100.0 - (x * 100.0).round()).abs() < 1e-4), "{f:?}");
        }
    }
    assert!(header.contains("whole percents"));
    // The file reads back as charts.
    let parsed = ChartSet::parse(&out.charts.to_text(&header), 3).unwrap();
    assert_eq!(parsed.charts.len(), out.charts.charts.len());
}

#[test]
fn one_thread_is_reproducible() {
    let a = solve(&small_job(20_000), |_| ()).unwrap();
    let b = solve(&small_job(20_000), |_| ()).unwrap();
    assert_eq!(a.charts.to_text(""), b.charts.to_text(""));
    assert_eq!(a.diagnosis.ev, b.diagnosis.ev);
    // The gain sums over a hash map, whose order changes the last bit.
    for (x, y) in a.diagnosis.deviation_gain.iter().zip(&b.diagnosis.deviation_gain) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn bad_games_are_errors() {
    let mut job = small_job(1000);
    job.spec.players = 1;
    assert!(solve(&job, |_| ()).is_err());
    let mut job = small_job(1000);
    job.spec.rules.max_nodes = 10;
    assert!(solve(&job, |_| ()).unwrap_err().contains("nodes"));
    // No deals: every spot uses the fallback rules, so no spot is written.
    let out = solve(&small_job(0), |_| ()).unwrap();
    assert!(out.charts.charts.is_empty());
    assert!(out.decisions > 0);
}

/// The file holds exactly the hands [`Trust::CHARTS`] trusts, rounded: the
/// same solve done by hand (one thread is reproducible) gives the same text.
#[test]
fn charts_are_the_trusted_hands_rounded() {
    use crate::charts::{round_charts, solved_charts, Trust, MIN_MIX};
    use crate::solver::{Options, Solver};
    use crate::tree::Tree;
    let job = small_job(600_000);
    let out = solve(&job, |_| ()).unwrap();
    let options = Options { threads: 1, seed: job.seed, ..Options::default() };
    let mut solver = Solver::new(Tree::build(&job.spec).unwrap(), job.factors.clone(), options);
    for epoch in 1..=EPOCHS as u64 {
        solver.run(job.deals * epoch / EPOCHS as u64 - job.deals * (epoch - 1) / EPOCHS as u64);
    }
    let mut expected = solved_charts(&solver, Trust::CHARTS);
    round_charts(&mut expected, MIN_MIX);
    assert_eq!(out.charts.to_text(""), expected.to_text(""));
    // A looser rule would write more spots.
    assert!(
        solved_charts(&solver, Trust { updates: 0, ..Trust::CHARTS }).charts.len() > expected.charts.len()
    );
}
