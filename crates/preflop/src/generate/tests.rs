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
fn solving_writes_charts_for_every_decision_with_a_header() {
    let job = small_job(100_000);
    let mut epochs = Vec::new();
    let out = solve(&job, |p| epochs.push(*p)).unwrap();
    assert_eq!(epochs.len(), EPOCHS as usize);
    assert!(epochs.windows(2).all(|w| w[0].deals < w[1].deals && w[0].seconds <= w[1].seconds));
    assert_eq!(epochs.last().unwrap().deals, job.deals);
    assert_eq!(out.charts.charts.len(), out.decisions);
    assert_eq!(out.charts.players, 3);
    let header = header(&job, &out);
    for needle in [
        "solve-preflop",
        "3 players",
        "30bb",
        "rake 5% capped at 3bb",
        "flat test factors",
        "seed 3",
        "BTN",
        "SB",
        "BB",
    ] {
        assert!(header.contains(needle), "{needle:?} not in\n{header}");
    }
    // The file reads back as charts.
    let parsed = ChartSet::parse(&out.charts.to_text(&header), 3).unwrap();
    assert_eq!(parsed.charts.len(), out.decisions);
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
    // No deals: every spot uses the fallback rules, but it still works.
    let out = solve(&small_job(0), |_| ()).unwrap();
    assert_eq!(out.charts.charts.len(), out.decisions);
}
