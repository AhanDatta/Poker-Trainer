use super::*;
use trainer::preflop::ChartSet;

fn settings(out: &str) -> Settings<'_> {
    Settings {
        players: 3,
        stack_bb: 30.0,
        rake_pct: 5.0,
        rake_cap_bb: 3.0,
        million_deals: 0.02,
        threads: 1,
        seed: 7,
        cold_callers: 1,
        limp: None,
        factors: None,
        out,
    }
}

/// A path in the temp directory that no other test uses.
fn temp_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("poker-trainer-solve-preflop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name).to_string_lossy().into_owned()
}

#[test]
fn defaults_use_the_built_in_factors() {
    let s = Settings { threads: 0, million_deals: 0.0015, ..settings("x") };
    let job = job(&s).unwrap();
    assert_eq!(job.factors, Realization::builtin());
    assert_eq!(job.factors_name, Realization::BUILTIN_NAME);
    assert!(job.threads >= 1);
    assert_eq!(job.deals, 1500);
    assert_eq!(job.seed, 7);
    assert_eq!(job.spec.players, 3);
    assert_eq!(job.spec.rules.max_cold_callers, 1);
    assert_eq!(job.spec.rules.limp, preflop::Limp::Never);
    assert!((job.spec.rake_rate - 0.05).abs() < 1e-12);
    assert!((job.spec.rake_cap_bb - 3.0).abs() < 1e-12);
    assert!((job.spec.stack_bb - 30.0).abs() < 1e-12);
}

#[test]
fn a_factors_file_is_read() {
    let path = temp_path("factors.txt");
    let factors = Realization::positional(1.1, 0.7, 0.9);
    std::fs::write(&path, factors.to_text("test")).unwrap();
    let s = Settings { factors: Some(&path), ..settings("x") };
    let job = job(&s).unwrap();
    assert_eq!(job.factors, factors);
    assert_eq!(job.factors_name, path);
}

#[test]
fn bad_settings_are_errors() {
    let base = || settings("x");
    for deals in [-1.0, f64::NAN, f64::INFINITY, MAX_MILLION_DEALS * 2.0] {
        let err = job(&Settings { million_deals: deals, ..base() }).unwrap_err().to_string();
        assert!(err.contains("--deals"), "{err}");
    }
    for players in [0, 1, 10, usize::MAX] {
        assert!(job(&Settings { players, ..base() }).is_err(), "{players} players");
    }
    for stack in [0.0, 1.0, -5.0, f64::NAN, 5000.0] {
        assert!(job(&Settings { stack_bb: stack, ..base() }).is_err(), "stack {stack}");
    }
    for (rake, cap) in [(100.0, 3.0), (-1.0, 3.0), (f64::NAN, 3.0), (5.0, -1.0), (5.0, f64::INFINITY)] {
        assert!(
            job(&Settings { rake_pct: rake, rake_cap_bb: cap, ..base() }).is_err(),
            "rake {rake} cap {cap}"
        );
    }
    let missing = temp_path("no-such-factors.txt");
    let err = job(&Settings { factors: Some(&missing), ..base() }).unwrap_err();
    assert!(format!("{err:#}").contains("reading"), "{err:#}");
    let damaged = temp_path("damaged-factors.txt");
    std::fs::write(&damaged, "includes_rake false\n[raised caller ip headsup]\n1 2 x\n").unwrap();
    let err = job(&Settings { factors: Some(&damaged), ..base() }).unwrap_err().to_string();
    assert!(err.contains("damaged-factors.txt") && err.contains("line 3"), "{err}");
}

#[test]
fn running_writes_a_chart_file_the_trainer_reads() {
    let out = temp_path("charts_3max.txt");
    // Enough deals for the first decisions to be trusted, so the file has spots.
    run(&Settings { million_deals: 0.6, ..settings(&out) }).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.starts_with('#'));
    for needle in ["solve-preflop", "3 players", "30bb", "seed 7", "built-in"] {
        assert!(text.contains(needle), "{needle:?} not in the header");
    }
    let charts = ChartSet::parse(&text, 3).unwrap();
    assert_eq!(charts.players, 3);
    assert!(!charts.charts.is_empty());
    // A folder that doesn't exist is an error, not a panic.
    let bad = temp_path("no-such-dir/charts.txt");
    assert!(run(&settings(&bad)).is_err());
}

#[test]
fn heads_up_lets_the_small_blind_limp_by_default() {
    use preflop::Limp;
    let limp = |players, limp| job(&Settings { players, limp, ..settings("x") }).map(|j| j.spec.rules.limp);
    assert_eq!(limp(2, None).unwrap(), Limp::SmallBlind);
    assert_eq!(limp(6, None).unwrap(), Limp::Never);
    assert_eq!(limp(2, Some("never")).unwrap(), Limp::Never);
    assert_eq!(limp(9, Some("sb")).unwrap(), Limp::SmallBlind);
    assert!(limp(6, Some("always")).unwrap_err().to_string().contains("--limp"));
}
