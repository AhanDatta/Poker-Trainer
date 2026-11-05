//! Benchmarks realistic heads-up single-raised-pot solves.
//!
//! Usage: cargo run --release -p solver --example bench -- [river|turn|flop-fast|flop-exact|flop-exact-lean] [board]
//! Env: ITERS (max iterations), TARGET (% pot), NOISO=1, STORAGE=f32|compressed|auto, NOBATCH=1,
//! GPU=1|fast|exact (with --features gpu)

use holdem::card::parse_cards;
use holdem::{PostflopMenu, Range, Street, StreetSizes};
use solver::*;
use std::time::Instant;

const SB_OPEN: &str = "22+, A2+, K2+, Q2+, J2+, T2s+, T5o+, 92s+, 96o+, 82s+, 85o+, 72s+, 75o+, 62s+, 64o+, 52s+, 54o, 42s+, 32s";
const BB_CALL: &str = "22-99, A2s-A9s, A2o-AJo, K2s-KJs, K5o-KJo, Q2s-QJs, Q8o-QJo, J5s-JTs, J8o-JTo, T6s-T9s, T8o-T9o, 96s-98s, 98o, 85s+, 87o, 74s+, 76o, 63s+, 65o, 53s+, 43s";

/// Simplified sizes for streets after the flop inside flop solves.
fn future(raise: &str, max_raises: u32) -> PostflopMenu {
    PostflopMenu {
        turn: StreetSizes::new(&["66%"], &[raise]),
        river: StreetSizes::new(&["75%"], &[raise]),
        max_raises,
        ..PostflopMenu::default()
    }
}

/// Peak working set (Windows has no `/proc`): `K32GetProcessMemoryInfo`
/// from kernel32, declared here so the example needs no extra crate.
#[cfg(windows)]
fn peak_rss_gb() -> f64 {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut c = ProcessMemoryCounters::default();
    let size = std::mem::size_of::<ProcessMemoryCounters>() as u32;
    c.cb = size;
    // SAFETY: `c` is a correctly sized, writable PROCESS_MEMORY_COUNTERS and
    // the pseudo-handle from GetCurrentProcess needs no closing.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, size) };
    if ok != 0 {
        c.peak_working_set_size as f64 / 1e9
    } else {
        0.0
    }
}

#[cfg(not(windows))]
fn peak_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1e6)
        .unwrap_or(0.0)
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn run(name: &str, board: &str, chance: ChanceMode, future_menu: Option<PostflopMenu>) {
    let board = parse_cards(board).unwrap();
    let street = Street::from_board_len(board.len());
    let mut cfg = SolveConfig::new(
        board,
        [Range::parse(BB_CALL).unwrap().into_weights(), Range::parse(SB_OPEN).unwrap().into_weights()],
        BettingState::street_start(street, 500, [9750, 9750], 100),
        PostflopMenu::default(),
    );
    cfg.future_menu = future_menu;
    cfg.chance = chance;
    cfg.max_memory_bytes = 14 << 30;
    cfg.isomorphism = std::env::var("NOISO").is_err();
    cfg.storage = match std::env::var("STORAGE").as_deref() {
        Ok("f32") => StorageMode::F32,
        Ok("compressed") => StorageMode::Compressed,
        _ => StorageMode::Auto,
    };
    if let Ok(d) = std::env::var("DCFR") {
        // e.g. "1.5,0,2" or "1.5,clip,2"
        let v: Vec<&str> = d.split(',').collect();
        cfg.dcfr = DcfrParams {
            alpha: v[0].parse().unwrap(),
            beta: if v[1] == "clip" { None } else { Some(v[1].parse().unwrap()) },
            gamma: v[2].parse().unwrap(),
            restart_average: v.get(3).is_some_and(|x| x.contains('r')),
            unweighted_average: v.get(3).is_some_and(|x| x.contains('u')),
        };
    }
    let est = Solver::estimate(&cfg).unwrap();
    println!(
        "== {name}: {} player nodes, {} chance, {} terminal, ~{:.2} GB ({})",
        est.player_nodes,
        est.chance_nodes,
        est.terminal_nodes,
        est.memory_bytes() as f64 / 1e9,
        if est.compressed { "compressed" } else { "f32" }
    );
    let t0 = Instant::now();
    let mut s = Solver::new(cfg).unwrap();
    if std::env::var("NOBATCH").is_ok() {
        s.set_batch_river(false);
    }
    println!(
        "   hands {} vs {}, iso pairs {}, build {:.2?}",
        s.hands().len(0),
        s.hands().len(1),
        s.isomorphic_pairs(),
        t0.elapsed()
    );
    if std::env::var("GPU").is_ok() {
        use_gpu(&mut s);
    }
    let t1 = Instant::now();
    let iters: u32 = env("ITERS", 1000);
    let opts = SolveOptions {
        max_iterations: iters,
        target_exploitability_pct: env("TARGET", 0.3),
        check_every: env("CHECK", 10u32).min(iters),
    };
    let p = s.solve(&opts, |p| {
        println!("   it {:4}: expl {:.3}% pot  ({:.2?})", p.iteration, p.exploitability_pct, t1.elapsed());
        true
    });
    let el = t1.elapsed();
    println!(
        "   done: {} iterations, {:.3}% pot, {:.2?} ({:.2} ms/iter), peak RSS {:.2} GB",
        p.iteration,
        p.exploitability_pct,
        el,
        el.as_secs_f64() * 1000.0 / p.iteration as f64,
        peak_rss_gb()
    );
    #[cfg(feature = "gpu")]
    if let Some(e) = s.gpu_error() {
        println!("   GPU failed, solve restarted on the CPU: {e}");
    }
    let root = s.query(&[]).unwrap();
    let rs = root.range_strategy();
    println!(
        "   root (OOP) strategy: {:?}",
        root.actions.iter().zip(rs).map(|(a, f)| format!("{a:?} {:.1}%", f * 100.0)).collect::<Vec<_>>()
    );
}

/// `GPU=1` (or `GPU=fast`): solve on the GPU with double-single terminal
/// sums; `GPU=exact`: with f64 sums, bit-identical to the CPU. Needs
/// `--features gpu`.
#[cfg(feature = "gpu")]
fn use_gpu(s: &mut Solver) {
    let precision = match std::env::var("GPU").as_deref() {
        Ok("exact") => solver::gpu::GpuPrecision::Exact,
        _ => solver::gpu::GpuPrecision::Fast,
    };
    let t = Instant::now();
    match s.use_gpu(precision) {
        Ok(()) => println!("   on the GPU, {precision:?} sums (setup {:.2?})", t.elapsed()),
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn use_gpu(_: &mut Solver) {
    eprintln!("GPU=1 needs a build with --features gpu");
    std::process::exit(1);
}

fn main() {
    let what = std::env::args().nth(1).unwrap_or_else(|| "turn".into());
    let flop = std::env::args().nth(2).unwrap_or_else(|| "Ks7d2c".into());
    match what.as_str() {
        "river" => run("river", "Ks7d2c9h4s", ChanceMode::Exact, None),
        "turn" => run("turn", "Ks7d2c9h", ChanceMode::Exact, None),
        "flop-fast" => run(
            "flop sampled 8x6, simplified turn/river",
            &flop,
            ChanceMode::Sampled { turn: 8, river: 6, seed: 1 },
            Some(future("3x", 2)),
        ),
        "flop-exact" => {
            run("flop exact, turn/river 1 size + 3x raise", &flop, ChanceMode::Exact, Some(future("3x", 2)))
        }
        "flop-exact-lean" => run(
            "flop exact, turn/river 1 size + all-in raise",
            &flop,
            ChanceMode::Exact,
            Some(future("allin", 2)),
        ),
        other => eprintln!("unknown benchmark {other}"),
    }
}
