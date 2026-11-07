//! Benchmarks three-way flop solves on chart ranges (CO opens, BTN and BB
//! call), as study mode and the reviews run them.
//!
//! Usage: cargo run --release -p trainer --example multibench -- [board] [iterations]
//! Env: TURNS / RIVERS (sampled cards, default 8 / 4), EXACT=1 (every card),
//! EXACT_REMOVAL=1 (card removal between opponents), COMPARE=1 (both, judged
//! in the exact game), RANGES="r0|r1|r2" (instead of the chart ranges),
//! GPU=1 (solve on the GPU; needs `--features gpu` and f32 storage, which it
//! implies; FAST=1 for double-single leaf sums), F32=1 (f32 storage on the
//! CPU too, to compare like with like).

use holdem::card::parse_cards;
use solver::multi::{MultiConfig, MultiSolver, MultiState};
use solver::{ChanceMode, SolveOptions};
use std::sync::Arc;
use std::time::Instant;
use trainer::config::SolverSettings;
use trainer::fmt::BB;
use trainer::preflop::ChartSet;
use trainer::study::{range_combos, PreflopNav, StudySettings};
use trainer::TrainerConfig;

/// DCFR="alpha,beta,gamma[,flags]" (flags: r = restart the average at
/// powers of 4, u = unweighted average).
fn dcfr_from_env() -> solver::DcfrParams {
    let Ok(d) = std::env::var("DCFR") else { return solver::multi::multiway_dcfr() };
    let v: Vec<&str> = d.split(',').collect();
    solver::DcfrParams {
        alpha: v[0].parse().unwrap(),
        beta: if v[1] == "clip" { None } else { Some(v[1].parse().unwrap()) },
        gamma: v[2].parse().unwrap(),
        restart_average: v.get(3).is_some_and(|x| x.contains('r')),
        unweighted_average: v.get(3).is_some_and(|x| x.contains('u')),
    }
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let board = std::env::args().nth(1).unwrap_or_else(|| "Js9s4d".into());
    let iters: u32 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(100);
    let cfg = TrainerConfig { players: 6, ..TrainerConfig::default() };
    let settings = Arc::new(StudySettings::from_config(&cfg, Arc::new(ChartSet::builtin(6))));
    // UTG, HJ fold; CO raises; BTN calls; SB folds; BB calls.
    let mut nav = PreflopNav::new(settings);
    for want in ["fold", "fold", "raise", "call", "fold", "call"] {
        let node = nav.node().expect("preflop decision");
        let i = node
            .labels
            .iter()
            .position(|l| l.to_lowercase().starts_with(want))
            .unwrap_or_else(|| panic!("{want} not in {:?}", node.labels));
        nav.act(i).unwrap();
    }
    let mut spot = nav.spot().expect("3-way flop");
    // RANGES="r0|r1|r2" replaces the chart ranges (e.g. wide limped pots).
    if let Ok(r) = std::env::var("RANGES") {
        for (i, text) in r.split('|').enumerate() {
            spot.ranges[i] = holdem::Range::parse(text).expect("range").into_weights();
        }
    }
    let board = parse_cards(&board).unwrap();
    for (n, r) in spot.names.iter().zip(&spot.ranges) {
        println!("{n}: {:.0} combos", range_combos(r, &board));
    }
    let s = SolverSettings::default();
    let chance = if std::env::var("EXACT").is_ok() {
        ChanceMode::Exact
    } else {
        ChanceMode::Sampled {
            turn: env("TURNS", s.multiway_turn_samples),
            river: env("RIVERS", s.multiway_river_samples),
            seed: 1,
        }
    };
    let mc = MultiConfig {
        board: board.clone(),
        ranges: [spot.ranges[0].clone(), spot.ranges[1].clone(), spot.ranges[2].clone()],
        root: MultiState::street_start(
            holdem::Street::from_board_len(board.len()),
            spot.pot,
            [spot.stacks[0], spot.stacks[1], spot.stacks[2]],
            [false; 3],
            BB,
        ),
        menu: cfg.menu.postflop.clone(),
        future_menu: Some(s.multiway_future_menu.clone()),
        big_blind: BB,
        chance,
        prune: s.prune.max(0.002),
        keep: Default::default(),
        max_memory_bytes: 12 << 30,
        rake_rate: 0.0,
        rake_cap: 0.0,
        exact_removal: std::env::var("EXACT_REMOVAL").is_ok(),
        storage: if std::env::var("GPU").is_ok() || std::env::var("F32").is_ok() {
            solver::StorageMode::F32
        } else {
            solver::StorageMode::Auto
        },
        dcfr: dcfr_from_env(),
    };
    if std::env::var("COMPARE").is_ok() {
        compare(mc, iters);
        return;
    }
    let t0 = Instant::now();
    let mut solver = match MultiSolver::new(mc) {
        Ok(x) => x,
        Err(e) => {
            println!("build failed: {e}");
            return;
        }
    };
    println!(
        "tree: {} strategy cells (~{:.2} GB), built in {:.2?}",
        solver.cells(),
        solver.cells() as f64 * 8.0 / 1e9,
        t0.elapsed()
    );
    if std::env::var("NOBATCH").is_ok() {
        solver.set_batch_river(false);
    }
    let (by_bet, runout) = solver.showdown_counts();
    println!("showdowns: {by_bet} after betting, {runout} in all-in runouts");
    if std::env::var("GPU").is_ok() {
        use_gpu(&mut solver);
    }
    if iters == 0 {
        return;
    }
    let t1 = Instant::now();
    let opts = SolveOptions {
        max_iterations: iters,
        target_exploitability_pct: env("TARGET", 0.0),
        check_every: env("CHECK", 20),
    };
    let p = solver.solve(&opts, |p| {
        println!(
            "  it {:4}: convergence {:.3}% pot ({:.2?})",
            p.iteration,
            p.exploitability_pct,
            t1.elapsed()
        );
        true
    });
    let el = t1.elapsed();
    println!(
        "done: {} iterations in {:.2?} ({:.1} ms/iter), {:.3}% pot",
        p.iteration,
        el,
        el.as_secs_f64() * 1e3 / p.iteration as f64,
        p.exploitability_pct
    );
    if let Some(e) = solver.gpu_error() {
        println!("GPU failed, solve restarted on the CPU: {e}");
    }
}

/// `GPU=1`: solve on the GPU (f64 leaf sums, bit-identical to the CPU;
/// with `FAST=1`, double-single).
#[cfg(feature = "gpu")]
fn use_gpu(solver: &mut MultiSolver) {
    use solver::gpu::GpuPrecision;
    let t = Instant::now();
    let precision = if std::env::var("FAST").is_ok() { GpuPrecision::Fast } else { GpuPrecision::Exact };
    match solver.use_gpu(precision) {
        Ok(()) => println!("on the GPU (setup {:.2?})", t.elapsed()),
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn use_gpu(_: &mut MultiSolver) {
    eprintln!("GPU=1 needs a build with --features gpu");
    std::process::exit(1);
}

/// Solves with independent opponents and with exact card removal, then
/// judges both strategies in the exact game.
fn compare(mc: MultiConfig, iters: u32) {
    let pot = mc.root.pot as f64;
    let opts = SolveOptions { max_iterations: iters, target_exploitability_pct: 0.0, check_every: iters };
    let mut approx = MultiSolver::new(MultiConfig { exact_removal: false, ..mc.clone() }).unwrap();
    let t = Instant::now();
    approx.solve(&opts, |_| true);
    println!("independent: {iters} iterations in {:.2?}", t.elapsed());
    let mut exact = MultiSolver::new(MultiConfig { exact_removal: true, ..mc }).unwrap();
    let t = Instant::now();
    exact.solve(&opts, |_| true);
    println!("exact:       {iters} iterations in {:.2?}", t.elapsed());
    let pct = |c: f64| 100.0 * c / pot;
    let (ca, ce) = (approx.convergence(), exact.convergence());
    approx.set_exact_removal(true);
    let ca_exact = approx.convergence();
    println!("convergence (avg best-response gain, % pot):");
    println!("  independent strategy, judged in its own model: {:.3}", pct(ca));
    println!("  independent strategy, judged in the exact game: {:.3}", pct(ca_exact));
    println!("  exact strategy, exact game:                     {:.3}", pct(ce));
    let va = approx.values();
    let ve = exact.values();
    println!("values in the exact game (chips): independent strategy {va:.1?}, exact strategy {ve:.1?}");
    let (ra, re) = (approx.root_range_strategy(), exact.root_range_strategy());
    println!(
        "root strategy (first to act): independent {:.1?}% vs exact {:.1?}%",
        ra.iter().map(|x| x * 100.0).collect::<Vec<_>>(),
        re.iter().map(|x| x * 100.0).collect::<Vec<_>>()
    );
}
