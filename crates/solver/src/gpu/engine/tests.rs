//! GPU traversals and solves against the CPU solver. Tests skip when there
//! is no NVIDIA device.
//!
//! Tolerances: a single traversal from the same storage differs from the
//! CPU only by the f32 rounding of terminal values (the f64 sums run in a
//! different order), so values and updated regrets must agree to 1e-5 of
//! their scale. Full solves use the tolerances of the CPU-vs-CPU test
//! `batched_river_matches_recursion` (whose two paths also differ by f32
//! rounding): after 5 iterations 5e-5 in strategy and 0.25 chips in EV, and
//! exploitability within 1e-4 relative.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::cfr::{self, Discount, Mode};
use crate::gpu::HostStorage;
use crate::gpu::testutil::{
    allin_turn_config, arb_river_config, arb_turn_config, compressed, device, iso_turn_config, mid_street_config,
    river_config, sampled_flop_config, turn_config,
};
use crate::gpu::{GpuPrecision, Scratch};
use crate::{Action, DcfrParams, NodeInfo, SolveConfig, SolveOptions, Solver, Step};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn engine_for(s: &mut Solver, gpu: Gpu, precision: GpuPrecision) -> Engine {
    let plan = Plan::build(&s.ctx()).unwrap();
    Engine::new(gpu, plan, s.store.gpu_view(), precision).unwrap()
}

/// Random regrets (some exactly zero, many negative) and strategy sums
/// (some all-zero), so every branch of regret matching and normalization
/// is exercised. 16-bit storage gets random raw values and scales (some
/// scales zero).
fn randomize_storage(s: &mut Solver, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    match s.store.gpu_view() {
        HostStorage::F32 { regrets, sums } => {
            for x in regrets.iter_mut() {
                *x = if rng.random::<f64>() < 0.2 { 0.0 } else { rng.random_range(-50.0f32..50.0) };
            }
            for x in sums.iter_mut() {
                *x = if rng.random::<f64>() < 0.3 { 0.0 } else { rng.random_range(0.0f32..20.0) };
            }
        }
        HostStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => {
            for x in regrets.iter_mut() {
                *x = if rng.random::<f64>() < 0.2 { 0 } else { rng.random() };
            }
            for x in sums.iter_mut() {
                *x = if rng.random::<f64>() < 0.3 { 0 } else { rng.random() };
            }
            for x in regret_scale.iter_mut() {
                *x = if rng.random::<f64>() < 0.1 { 0.0 } else { rng.random_range(0.0f32..50.0) };
            }
            for x in strategy_scale.iter_mut() {
                *x = if rng.random::<f64>() < 0.1 { 0.0 } else { rng.random_range(0.0f32..20.0) };
            }
        }
    }
}

/// Every decision node's regrets and strategy sums, decoded by the CPU's own
/// storage code (so 16-bit storage compares the same way as f32).
fn storage_copy(s: &mut Solver) -> (Vec<f32>, Vec<f32>) {
    let len = s.tree.storage_len as usize;
    let (mut regrets, mut sums) = (vec![0f32; len], vec![0f32; len]);
    for node in &s.tree.nodes {
        let crate::tree::Kind::Player(q) = node.kind else { continue };
        let b = node.block(s.hands.len(q as usize));
        // SAFETY: no traversal is running; blocks don't overlap.
        unsafe {
            s.store.load_regrets(b, &mut regrets[b.offset..b.offset + b.len]);
            s.store.load_strat(b, &mut sums[b.offset..b.offset + b.len]);
        }
    }
    (regrets, sums)
}

/// `max |a - b|` and `max |b|`.
fn diff_and_scale(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let diff = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let scale = b.iter().fold(0f32, |m, y| m.max(y.abs()));
    (diff, scale)
}

fn assert_relative(a: &[f32], b: &[f32], tol: f32, what: &str) {
    let (diff, scale) = diff_and_scale(a, b);
    assert!(a.iter().all(|x| x.is_finite()), "{what}: non-finite GPU value");
    assert!(diff <= tol * scale.max(1e-30), "{what}: max diff {diff:e} vs scale {scale:e}");
}

fn assert_same_node(a: &NodeInfo, b: &NodeInfo, strat_tol: f32, ev_tol: f32, what: &str) {
    assert_eq!(a.actions, b.actions, "{what}: actions");
    assert_eq!(a.combos, b.combos, "{what}: combos");
    let (ds, _) = diff_and_scale(&a.strategy, &b.strategy);
    let (de, _) = diff_and_scale(&a.ev, &b.ev);
    assert!(ds <= strat_tol, "{what}: strategy differs by {ds:e}");
    assert!(de <= ev_tol, "{what}: EV differs by {de} chips");
}

fn opts(iterations: u32) -> SolveOptions {
    SolveOptions { max_iterations: iterations, target_exploitability_pct: 0.0, check_every: iterations }
}

/// CPU and GPU solvers of the same spot after `iterations` iterations.
fn solve_both(cfg: SolveConfig, iterations: u32, precision: GpuPrecision) -> Option<(Solver, Solver)> {
    let gpu = device()?;
    drop(gpu);
    let mut cpu = Solver::new(cfg.clone()).unwrap();
    let mut on_gpu = Solver::new(cfg).unwrap();
    on_gpu.use_gpu(precision).unwrap();
    let pc = cpu.solve(&opts(iterations), |_| true);
    let pg = on_gpu.solve(&opts(iterations), |_| true);
    assert!(
        on_gpu.uses_gpu() && on_gpu.gpu_error().is_none(),
        "GPU solve fell back: {:?}",
        on_gpu.gpu_error()
    );
    assert_eq!(pc.iteration, pg.iteration);
    let tol = 1e-4 * (1.0 + pc.exploitability.abs());
    assert!(
        (pc.exploitability - pg.exploitability).abs() < tol,
        "exploitability: cpu {} vs gpu {}",
        pc.exploitability,
        pg.exploitability
    );
    Some((cpu, on_gpu))
}

/// Compares the root, the node after a check, and the first river node
/// after check-check for each card in `rivers` (on turn spots; a
/// non-canonical card goes through the isomorphism).
fn compare_nodes(cpu: &Solver, gpu: &Solver, strat_tol: f32, ev_tol: f32, rivers: &[&str]) {
    let check = Step::Action(Action::Check);
    let mut paths: Vec<Vec<Step>> = vec![vec![], vec![check], vec![Step::Action(Action::Fold)]];
    for card in rivers {
        paths.push(vec![check, check, Step::Card(card.parse().unwrap())]);
    }
    for path in &paths {
        if let (Ok(a), Ok(b)) = (gpu.query(path), cpu.query(path)) {
            assert_same_node(&a, &b, strat_tol, ev_tol, &format!("{path:?}"));
        } else if path.len() > 2 {
            panic!("river path {path:?} not found");
        }
    }
}

/// Spots with chance nodes, the river cards to compare after check-check.
/// Includes 16-bit storage variants.
fn chance_spots() -> Vec<(SolveConfig, &'static [&'static str])> {
    vec![
        (turn_config(), &["Ks", "2h"]),
        (iso_turn_config(), &["Kc", "Kh", "3s"]),
        (allin_turn_config(), &["Ah"]),
        (sampled_flop_config(), &[]),
        (compressed(turn_config()), &["Ks", "2h"]),
        (compressed(iso_turn_config()), &["Kc", "Kh", "3s"]),
        (compressed(sampled_flop_config()), &[]),
    ]
}

/// River spots, including one with 16-bit storage.
fn river_spots() -> [SolveConfig; 4] {
    [river_config(), mid_street_config(false), mid_street_config(true), compressed(mid_street_config(true))]
}

#[test]
fn traversal_values_match_cpu_from_random_storage() {
    let spots = river_spots().into_iter().chain(chance_spots().into_iter().map(|(c, _)| c));
    for ((k, cfg), precision) in
        spots.enumerate().flat_map(|spot| [(spot.clone(), GpuPrecision::Exact), (spot, GpuPrecision::Fast)])
    {
        let Some(gpu) = device() else { return };
        let mut s = Solver::new(cfg).unwrap();
        randomize_storage(&mut s, 11 + k as u64);
        let mut engine = engine_for(&mut s, gpu, precision);
        for p in 0..2 {
            for mode in [Mode::Average, Mode::BestResponse] {
                let mut cpu = vec![0f32; s.hands.len(p)];
                cfr::traverse(&s.ctx(), 0, p, &s.hands.weights[p ^ 1], mode, &mut cpu);
                let got = engine.root_hand_values(p, mode).unwrap();
                assert_relative(
                    &got,
                    &cpu,
                    1e-5,
                    &format!("config {k}, {precision:?}, player {p}, {mode:?}"),
                );
            }
        }
        let cpu = cfr::root_values(&s.ctx(), Mode::Average);
        let got = engine.root_values(Mode::Average).unwrap();
        for p in 0..2 {
            assert!(
                (cpu[p] - got[p]).abs() <= 1e-5 * (1.0 + cpu[p].abs()),
                "root value {p}: {cpu:?} vs {got:?}"
            );
        }
    }
}

#[test]
fn one_cfr_iteration_matches_cpu_from_random_storage() {
    for t in [1u32, 2, 4, 7] {
        let spots = [
            river_config(),
            mid_street_config(true),
            turn_config(),
            iso_turn_config(),
            sampled_flop_config(),
            compressed(mid_street_config(true)),
            compressed(turn_config()),
            compressed(sampled_flop_config()),
        ];
        for (k, cfg) in spots.into_iter().enumerate() {
            let Some(gpu) = device() else { return };
            let mut cpu = Solver::new(cfg.clone()).unwrap();
            let mut on_gpu = Solver::new(cfg).unwrap();
            randomize_storage(&mut cpu, 100 + t as u64);
            randomize_storage(&mut on_gpu, 100 + t as u64);
            let mut engine = engine_for(&mut on_gpu, gpu, GpuPrecision::Exact);
            let mut params = DcfrParams::default();
            for weighted in [false, true] {
                params.unweighted_average = !weighted;
                let d = Discount::new(t, &params);
                let ctx = cpu.ctx();
                for p in 0..2 {
                    let mut out = vec![0f32; cpu.hands.len(p)];
                    cfr::traverse(&ctx, 0, p, &cpu.hands.weights[p ^ 1], Mode::Cfr(d), &mut out);
                }
                engine.cfr_iteration(d).unwrap();
                engine.download(on_gpu.store.gpu_view()).unwrap();
                let (cr, cm) = storage_copy(&mut cpu);
                let (gr, gm) = storage_copy(&mut on_gpu);
                let what = format!("t {t}, config {k}, weighted {weighted}");
                assert_relative(&gr, &cr, 1e-5, &format!("{what}: regrets"));
                assert_relative(&gm, &cm, 1e-5, &format!("{what}: strategy sums"));
            }
        }
    }
}

#[test]
fn solves_match_cpu() {
    for cfg in river_spots() {
        let Some((cpu, gpu)) = solve_both(cfg.clone(), 5, GpuPrecision::Exact) else { return };
        compare_nodes(&cpu, &gpu, 5e-5, 0.25, &[]);
    }
    for (cfg, rivers) in chance_spots() {
        let Some((cpu, gpu)) = solve_both(cfg.clone(), 5, GpuPrecision::Exact) else { return };
        compare_nodes(&cpu, &gpu, 5e-5, 0.25, rivers);
    }
}

/// With the CPU's batched river evaluator switched off (every terminal
/// one at a time), the GPU follows that path instead.
#[test]
fn matches_cpu_without_river_batching() {
    let Some(_) = device() else { return };
    let mut cpu = Solver::new(turn_config()).unwrap();
    let mut gpu = Solver::new(turn_config()).unwrap();
    cpu.set_batch_river(false);
    gpu.set_batch_river(false);
    gpu.use_gpu(GpuPrecision::Exact).unwrap();
    let pc = cpu.solve(&opts(4), |_| true);
    let pg = gpu.solve(&opts(4), |_| true);
    assert!(gpu.uses_gpu() && gpu.gpu_error().is_none());
    assert!((pc.exploitability - pg.exploitability).abs() < 1e-4 * (1.0 + pc.exploitability.abs()));
    compare_nodes(&cpu, &gpu, 5e-5, 0.25, &["Ks"]);
}

#[test]
fn continues_a_cpu_solve() {
    let Some(_) = device() else { return };
    let mut reference = Solver::new(river_config()).unwrap();
    reference.solve(&opts(6), |_| true);
    let mut s = Solver::new(river_config()).unwrap();
    s.solve(&opts(3), |_| true);
    s.use_gpu(GpuPrecision::Exact).unwrap();
    s.solve(&opts(6), |_| true);
    assert!(s.uses_gpu() && s.gpu_error().is_none());
    assert_eq!(s.iterations(), 6);
    compare_nodes(&reference, &s, 5e-5, 0.25, &[]);
}

#[test]
fn accepts_compressed_storage() {
    let Some(_) = device() else { return };
    let mut s = Solver::new(compressed(turn_config())).unwrap();
    assert!(s.is_compressed());
    s.use_gpu(GpuPrecision::Exact).unwrap();
    let p = s.solve(&opts(2), |_| true);
    assert_eq!(p.iteration, 2);
    assert!(s.uses_gpu() && s.gpu_error().is_none());
}

#[test]
fn cancelled_gpu_solve_returns_and_stays_queryable() {
    let Some(_) = device() else { return };
    let mut s = Solver::new(river_config()).unwrap();
    s.use_gpu(GpuPrecision::Exact).unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let p = s.solve_cancellable(&opts(10), &cancel, |_| true);
    assert!(p.finished);
    assert_eq!(s.iterations(), 0);
    let root = s.query(&[]).unwrap();
    assert!(root.strategy.iter().all(|x| x.is_finite()));
    // Stopping from the progress callback.
    let p = s
        .solve(&SolveOptions { max_iterations: 50, target_exploitability_pct: 0.0, check_every: 5 }, |_| {
            false
        });
    assert_eq!(p.iteration, 5);
    assert!(s.uses_gpu());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Random spots: values from random storage and short solves agree.
    #[test]
    fn random_spots_match_cpu(cfg in arb_river_config(), seed in any::<u64>()) {
        let Ok(mut s) = Solver::new(cfg.clone()) else { return Ok(()) };
        let Some(gpu) = device() else { return Ok(()) };
        randomize_storage(&mut s, seed);
        let mut engine = engine_for(&mut s, gpu, GpuPrecision::Exact);
        for p in 0..2 {
            for mode in [Mode::Average, Mode::BestResponse] {
                let mut cpu = vec![0f32; s.hands.len(p)];
                cfr::traverse(&s.ctx(), 0, p, &s.hands.weights[p ^ 1], mode, &mut cpu);
                let got = engine.root_hand_values(p, mode).unwrap();
                assert_relative(&got, &cpu, 1e-5, &format!("player {p}, {mode:?}"));
            }
        }
        drop(engine);
        let Some((cpu, gpu)) = solve_both(cfg, 5, GpuPrecision::Exact) else { return Ok(()) };
        // EVs scale with the pot: keep the reference test's 0.25 chips per
        // ~2500 chips of EV.
        let root = cpu.query(&[]).unwrap();
        let ev_scale = root.ev.iter().fold(0f32, |m, x| m.max(x.abs()));
        compare_nodes(&cpu, &gpu, 5e-5, 0.25 * (1.0 + ev_scale / 2500.0), &[]);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Random turn spots: values from random storage and short solves agree.
    #[test]
    fn random_turn_spots_match_cpu(cfg in arb_turn_config(), seed in any::<u64>()) {
        let Ok(mut s) = Solver::new(cfg.clone()) else { return Ok(()) };
        let Some(gpu) = device() else { return Ok(()) };
        randomize_storage(&mut s, seed);
        let mut engine = engine_for(&mut s, gpu, GpuPrecision::Exact);
        for p in 0..2 {
            for mode in [Mode::Average, Mode::BestResponse] {
                let mut cpu = vec![0f32; s.hands.len(p)];
                cfr::traverse(&s.ctx(), 0, p, &s.hands.weights[p ^ 1], mode, &mut cpu);
                let got = engine.root_hand_values(p, mode).unwrap();
                assert_relative(&got, &cpu, 1e-5, &format!("player {p}, {mode:?}"));
            }
        }
        drop(engine);
        let Some((cpu, gpu)) = solve_both(cfg, 4, GpuPrecision::Exact) else { return Ok(()) };
        let root = cpu.query(&[]).unwrap();
        let ev_scale = root.ev.iter().fold(0f32, |m, x| m.max(x.abs()));
        compare_nodes(&cpu, &gpu, 5e-5, 0.25 * (1.0 + ev_scale / 2500.0), &[]);
    }
}

/// Every decision node, chance node and one-at-a-time terminal, evaluated
/// by the CPU from the reach the GPU computed for it, has the GPU's values.
/// Localizes a mismatch to a node kind (batched river showdowns are left
/// out: the CPU only ever evaluates those inside `river::run`, which the
/// decision-node checks cover).
#[test]
fn every_node_matches_cpu_given_gpu_reach() {
    for cfg in [
        turn_config(),
        iso_turn_config(),
        allin_turn_config(),
        sampled_flop_config(),
        compressed(turn_config()),
        compressed(sampled_flop_config()),
    ] {
        let Some(gpu) = device() else { return };
        let mut s = Solver::new(cfg).unwrap();
        randomize_storage(&mut s, 101);
        let mut engine = engine_for(&mut s, gpu, GpuPrecision::Exact);
        for mode in [Mode::Average, Mode::BestResponse] {
            for p in 0..2 {
                engine.traverse(p, mode).unwrap();
                let values = engine.gpu.stream.clone_dtoh(&engine.values).unwrap();
                let reach = engine.gpu.stream.clone_dtoh(&engine.reach).unwrap();
                let pp = &engine.plan.players[p];
                let (n, nq) = (pp.hands as usize, pp.opp_hands as usize);
                let ctx = s.ctx();
                let (mut checked, mut fused_checked) = (0, 0);
                for (id, node) in s.tree.nodes.iter().enumerate() {
                    let terminal =
                        matches!(node.kind, crate::tree::Kind::Fold(_) | crate::tree::Kind::Showdown);
                    if terminal && engine.plan.scalar[id] == 0 {
                        continue;
                    }
                    let slot = pp.slot[id] as usize;
                    let r = &reach[slot * nq..(slot + 1) * nq];
                    if r.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let mut cpu = vec![0f32; n];
                    cfr::traverse(&ctx, id, p, r, mode, &mut cpu);
                    let what = format!("{mode:?}, player {p}, node {id} ({:?} {:?})", node.kind, node.street);
                    let gpu_row = |i: usize| {
                        let r = engine.plan.row[i] as usize;
                        &values[r * n..(r + 1) * n]
                    };
                    // A fused node has no row of its own: its parent sees the
                    // sum of its children's rows.
                    let gpu: Vec<f32> = if pp.fused[id] == 1 {
                        let mut sum = vec![0f32; n];
                        for c in node.children() {
                            for (x, &v) in sum.iter_mut().zip(gpu_row(c)) {
                                *x += v;
                            }
                        }
                        fused_checked += 1;
                        sum
                    } else {
                        gpu_row(id).to_vec()
                    };
                    assert_relative(&gpu, &cpu, 1e-5, &what);
                    checked += 1;
                }
                assert!(checked > 100, "only {checked} nodes checked");
                assert!(fused_checked > 10, "only {fused_checked} fused nodes checked");
            }
        }
    }
}

/// Fast (double-single) and exact (f64) terminal values on identical reach
/// (the traversal kernels upstream of the terminals are f32 and the same
/// for both): `(values, bit-identical, max |diff| / row scale)`.
fn fast_vs_exact_terminals(cfg: SolveConfig, seed: u64) -> Option<(usize, usize, f32)> {
    let (exact_gpu, fast_gpu) = (device()?, device()?);
    let mut s = Solver::new(cfg).unwrap();
    randomize_storage(&mut s, seed);
    let mut exact = engine_for(&mut s, exact_gpu, GpuPrecision::Exact);
    let mut fast = engine_for(&mut s, fast_gpu, GpuPrecision::Fast);
    let (mut total, mut identical, mut worst) = (0usize, 0usize, 0f32);
    for mode in [Mode::Average, Mode::BestResponse] {
        for p in 0..2 {
            exact.traverse(p, mode).unwrap();
            fast.traverse(p, mode).unwrap();
            let ve = exact.gpu.stream.clone_dtoh(&exact.values).unwrap();
            let vf = fast.gpu.stream.clone_dtoh(&fast.values).unwrap();
            let n = exact.plan.players[p].hands as usize;
            for (id, node) in s.tree.nodes.iter().enumerate() {
                if !matches!(node.kind, crate::tree::Kind::Fold(_) | crate::tree::Kind::Showdown) {
                    continue;
                }
                let (e, f) = (&ve[id * n..(id + 1) * n], &vf[id * n..(id + 1) * n]);
                let scale = e.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
                for (a, b) in e.iter().zip(f) {
                    total += 1;
                    if a.to_bits() == b.to_bits() {
                        identical += 1;
                    } else {
                        worst = worst.max((a - b).abs() / scale);
                    }
                }
            }
        }
    }
    Some((total, identical, worst))
}

#[test]
fn fast_terminals_match_exact_terminals() {
    let spots = [
        river_config(),
        mid_street_config(true),
        turn_config(),
        iso_turn_config(),
        allin_turn_config(),
        sampled_flop_config(),
    ];
    for (k, cfg) in spots.into_iter().enumerate() {
        let Some((total, identical, worst)) = fast_vs_exact_terminals(cfg, 300 + k as u64) else { return };
        // Double-single carries ~2^-44 relative precision, so an f32 result
        // can only differ from the exact one when the exact value sits
        // within that distance of an f32 rounding boundary: rarely, and by
        // one rounding step. (Measured: >= 99.997% identical, worst 7e-8.)
        // Losing precision anywhere (e.g. plain f32 sums) fails both checks.
        assert!(identical * 10_000 >= total * 9_999, "spot {k}: only {identical} of {total} identical");
        assert!(worst <= 2.4e-7, "spot {k}: a value differs by {worst:e} of its row's scale");
    }
}

/// Fast-mode solves after 5 iterations, before CFR has amplified rounding
/// differences: the CPU-vs-CPU standard of `batched_river_matches_recursion`.
#[test]
fn fast_short_solves_match_cpu() {
    let spots = river_spots().into_iter().map(|c| (c, &[][..])).chain(chance_spots());
    for (cfg, rivers) in spots {
        let Some((cpu, gpu)) = solve_both(cfg, 5, GpuPrecision::Fast) else { return };
        compare_nodes(&cpu, &gpu, 5e-5, 0.25, rivers);
    }
}

/// Fast-mode solves to convergence. Per-hand strategies can't be compared
/// here: CFR amplifies any rounding difference near indifference points, and
/// the CPU's own batched and recursive paths (which differ only by rounding)
/// end up 0.18 apart in a hand's strategy and 28 chips in EV on these spots
/// after 60 iterations. What must agree is convergence and the range-level
/// strategy, by the standard of `compressed_storage_is_close`.
#[test]
fn fast_converged_solves_match_cpu() {
    for cfg in [turn_config(), iso_turn_config(), sampled_flop_config(), compressed(turn_config())] {
        let Some(_) = device() else { return };
        let opts = SolveOptions { max_iterations: 400, target_exploitability_pct: 0.3, check_every: 20 };
        let mut cpu = Solver::new(cfg.clone()).unwrap();
        let mut fast = Solver::new(cfg).unwrap();
        fast.use_gpu(GpuPrecision::Fast).unwrap();
        let pc = cpu.solve(&opts, |_| true);
        let pf = fast.solve(&opts, |_| true);
        assert!(fast.uses_gpu() && fast.gpu_error().is_none());
        assert!(pc.exploitability_pct <= 0.3 && pf.exploitability_pct <= 0.3, "{pc:?} {pf:?}");
        let (ic, ifast) = (f64::from(pc.iteration), f64::from(pf.iteration));
        assert!((ic - ifast).abs() <= 0.2 * ic, "iterations to target: cpu {ic}, gpu {ifast}");
        let check = Step::Action(Action::Check);
        for path in [vec![], vec![check]] {
            let (a, b) =
                (fast.query(&path).unwrap().range_strategy(), cpu.query(&path).unwrap().range_strategy());
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 0.03, "{path:?}: {a:?} vs {b:?}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Random turn spots in fast mode: traversal values match the CPU to
    /// 1e-5, short solves to the 5-iteration standard.
    #[test]
    fn random_turn_spots_match_cpu_fast(cfg in arb_turn_config(), seed in any::<u64>()) {
        let Ok(mut s) = Solver::new(cfg.clone()) else { return Ok(()) };
        let Some(gpu) = device() else { return Ok(()) };
        randomize_storage(&mut s, seed);
        let mut engine = engine_for(&mut s, gpu, GpuPrecision::Fast);
        for p in 0..2 {
            for mode in [Mode::Average, Mode::BestResponse] {
                let mut cpu = vec![0f32; s.hands.len(p)];
                cfr::traverse(&s.ctx(), 0, p, &s.hands.weights[p ^ 1], mode, &mut cpu);
                let got = engine.root_hand_values(p, mode).unwrap();
                assert_relative(&got, &cpu, 1e-5, &format!("player {p}, {mode:?}"));
            }
        }
        drop(engine);
        let Some((cpu, gpu)) = solve_both(cfg, 4, GpuPrecision::Fast) else { return Ok(()) };
        let root = cpu.query(&[]).unwrap();
        let ev_scale = root.ev.iter().fold(0f32, |m, x| m.max(x.abs()));
        compare_nodes(&cpu, &gpu, 5e-5, 0.25 * (1.0 + ev_scale / 2500.0), &[]);
    }
}

/// Every regret and strategy sum (and 16-bit scale), as raw bits.
fn storage_bits(s: &mut Solver) -> Vec<u32> {
    match s.store.gpu_view() {
        HostStorage::F32 { regrets, sums } => regrets.iter().chain(sums.iter()).map(|x| x.to_bits()).collect(),
        HostStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => regrets
            .iter()
            .map(|&x| u32::from(x as u16))
            .chain(sums.iter().map(|&x| u32::from(x)))
            .chain(regret_scale.iter().chain(strategy_scale.iter()).map(|x| x.to_bits()))
            .collect(),
    }
}

/// A solver of `cfg` on the GPU with its tree split as `scratch` says, from
/// storage randomized with `seed` (zero storage without one).
fn gpu_solver(cfg: SolveConfig, precision: GpuPrecision, scratch: Scratch, seed: Option<u64>) -> Solver {
    let mut s = Solver::new(cfg).unwrap();
    if let Some(seed) = seed {
        randomize_storage(&mut s, seed);
    }
    s.use_gpu_with(device().unwrap(), precision, scratch).unwrap();
    s
}

fn segments(s: &Solver) -> u32 {
    s.gpu.as_ref().unwrap().plan.num_segments
}

/// Splitting must not change a single bit: every group runs the same
/// kernels on the same inputs, only in other value rows and reach slots.
fn assert_split_matches_whole(cfg: &SolveConfig, precision: GpuPrecision, scratch: Scratch, seed: Option<u64>, iterations: u32) {
    let mut whole = gpu_solver(cfg.clone(), precision, Scratch::Unlimited, seed);
    let mut split = gpu_solver(cfg.clone(), precision, scratch, seed);
    assert_eq!(segments(&whole), 1);
    assert!(segments(&split) > 1, "{scratch:?}: tree not split");
    let pw = whole.solve(&opts(iterations), |_| true);
    let ps = split.solve(&opts(iterations), |_| true);
    assert!(whole.uses_gpu() && whole.gpu_error().is_none(), "{:?}", whole.gpu_error());
    assert!(split.uses_gpu() && split.gpu_error().is_none(), "{:?}", split.gpu_error());
    assert_eq!(pw.iteration, ps.iteration);
    assert_eq!(
        pw.exploitability.to_bits(),
        ps.exploitability.to_bits(),
        "{precision:?}, {scratch:?}: exploitability {} vs {}",
        pw.exploitability,
        ps.exploitability
    );
    let (bw, bs) = (storage_bits(&mut whole), storage_bits(&mut split));
    let differing = bw.iter().zip(&bs).filter(|(a, b)| a != b).count();
    assert!(bw.len() == bs.len() && differing == 0, "{precision:?}, {scratch:?}: {differing} storage values differ");
}

#[test]
fn split_solves_match_whole_solves_bit_for_bit() {
    let Some(_) = device() else { return };
    for (cfg, _) in chance_spots() {
        let nodes = Solver::new(cfg.clone()).unwrap().tree.nodes.len() as u32;
        for precision in [GpuPrecision::Exact, GpuPrecision::Fast] {
            for max_nodes in [1, nodes / 4] {
                assert_split_matches_whole(&cfg, precision, Scratch::ForceGroups { max_nodes }, None, 6);
            }
        }
    }
}

/// From random storage: dead subtrees, zero scales and all-zero sums inside
/// the groups.
#[test]
fn split_solves_match_whole_solves_from_random_storage() {
    let Some(_) = device() else { return };
    for (k, (cfg, _)) in chance_spots().into_iter().enumerate() {
        let precision = if k % 2 == 0 { GpuPrecision::Exact } else { GpuPrecision::Fast };
        assert_split_matches_whole(&cfg, precision, Scratch::ForceGroups { max_nodes: 1 }, Some(k as u64), 2);
    }
}

/// A memory budget below the whole tree's needs splits it, with the same
/// results; one too small for any chunk leaves the solver on the CPU.
#[test]
fn budget_split_solves_match_whole_solves() {
    let Some(_) = device() else { return };
    for cfg in [turn_config(), compressed(sampled_flop_config())] {
        let s = Solver::new(cfg.clone()).unwrap();
        let whole = Plan::build(&s.ctx()).unwrap();
        let [p0, p1] = &whole.players;
        let scratch = scratch_bytes(
            [u64::from(p0.num_slots), u64::from(p1.num_slots)],
            u64::from(whole.num_rows),
            [u64::from(p0.hands), u64::from(p1.hands)],
        );
        let budget = whole.device_bytes() + scratch - 1;
        assert_split_matches_whole(&cfg, GpuPrecision::Fast, Scratch::Budget(budget), None, 4);

        let mut tiny = Solver::new(cfg).unwrap();
        let err = tiny.use_gpu_with(device().unwrap(), GpuPrecision::Fast, Scratch::Budget(whole.device_bytes()));
        assert!(matches!(err, Err(GpuError::OutOfMemory { .. })), "{err:?}");
        assert!(!tiny.uses_gpu());
        assert_eq!(tiny.solve(&opts(2), |_| true).iteration, 2);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn random_split_solves_match_whole_solves(
        cfg in arb_turn_config(),
        max_nodes in 1u32..3000,
        seed in any::<u64>(),
        exact in any::<bool>(),
    ) {
        let Some(_) = device() else { return Ok(()) };
        let Ok(s) = Solver::new(cfg.clone()) else { return Ok(()) };
        let scratch = Scratch::ForceGroups { max_nodes };
        if Plan::build_with(&s.ctx(), scratch).unwrap().num_segments == 1 {
            return Ok(());
        }
        let precision = if exact { GpuPrecision::Exact } else { GpuPrecision::Fast };
        assert_split_matches_whole(&cfg, precision, scratch, Some(seed), 2);
    }
}

fn with_backend(mut cfg: SolveConfig, backend: crate::Backend) -> SolveConfig {
    cfg.backend = backend;
    cfg
}

/// `Backend::Gpu` runs each solve on the GPU (bit for bit as an explicit
/// `use_gpu` with the default precision) and frees it afterwards.
#[test]
fn gpu_backend_matches_explicit_use_gpu() {
    let Some(_) = device() else { return };
    for cfg in [turn_config(), compressed(sampled_flop_config())] {
        let mut explicit = Solver::new(cfg.clone()).unwrap();
        explicit.use_gpu(GpuPrecision::default()).unwrap();
        let pe = explicit.solve(&opts(6), |_| true);
        assert!(explicit.uses_gpu(), "explicit use_gpu stays on");
        let mut backed = Solver::new(with_backend(cfg, crate::Backend::Gpu)).unwrap();
        assert!(!backed.uses_gpu(), "the GPU is taken only while solving");
        let pb = backed.solve(&opts(6), |_| true);
        assert!(!backed.uses_gpu() && backed.gpu_error().is_none(), "{:?}", backed.gpu_error());
        assert_eq!(backed.solved_on(), crate::SolvedOn::GpuFast, "the tag outlives the engine");
        assert_eq!(explicit.solved_on(), crate::SolvedOn::GpuFast);
        assert_eq!(pe.exploitability.to_bits(), pb.exploitability.to_bits());
        assert!(storage_bits(&mut explicit) == storage_bits(&mut backed), "storage differs");
    }
}

/// A GPU-backed solve continued by a second `solve` call (the GPU taken
/// again, the storage uploaded again) ends where one call would.
#[test]
fn gpu_backend_continues_across_solve_calls() {
    let Some(_) = device() else { return };
    let cfg = with_backend(turn_config(), crate::Backend::Gpu);
    let mut once = Solver::new(cfg.clone()).unwrap();
    once.solve(&opts(6), |_| true);
    let mut twice = Solver::new(cfg).unwrap();
    twice.solve(&opts(3), |_| true);
    twice.solve(&opts(6), |_| true);
    assert_eq!(twice.iterations(), 6);
    assert!(storage_bits(&mut once) == storage_bits(&mut twice), "storage differs");
}

/// `Backend::Auto` solves turn roots on the GPU (same bits as `Gpu`).
#[test]
fn auto_backend_uses_the_gpu_off_the_river() {
    let Some(_) = device() else { return };
    let mut gpu = Solver::new(with_backend(turn_config(), crate::Backend::Gpu)).unwrap();
    let pg = gpu.solve(&opts(6), |_| true);
    let mut auto = Solver::new(with_backend(turn_config(), crate::Backend::Auto)).unwrap();
    let pa = auto.solve(&opts(6), |_| true);
    assert!(auto.gpu_error().is_none());
    assert_eq!(pg.exploitability.to_bits(), pa.exploitability.to_bits());
    assert!(storage_bits(&mut gpu) == storage_bits(&mut auto), "storage differs");
}

/// The exact path is bit-identical to the CPU: every regret and strategy sum
/// (raw bits, scales included) and the exploitability, after 5 iterations,
/// on every test spot (f32 and 16-bit storage, batched and one-at-a-time
/// terminals, isomorphic cards, all-in runouts, rake).
#[test]
fn exact_solves_match_cpu_bit_for_bit() {
    let Some(_) = device() else { return };
    let mut no_batching = Solver::new(turn_config()).unwrap();
    no_batching.set_batch_river(false);
    let spots = river_spots().into_iter().chain(chance_spots().into_iter().map(|(c, _)| c));
    for (k, cfg) in spots.enumerate() {
        let mut cpu = Solver::new(cfg.clone()).unwrap();
        let mut gpu = Solver::new(cfg).unwrap();
        gpu.use_gpu(GpuPrecision::Exact).unwrap();
        let pc = cpu.solve(&opts(5), |_| true);
        let pg = gpu.solve(&opts(5), |_| true);
        assert!(gpu.gpu_error().is_none(), "spot {k}: {:?}", gpu.gpu_error());
        assert_eq!((cpu.solved_on(), gpu.solved_on()), (crate::SolvedOn::Cpu, crate::SolvedOn::GpuExact));
        let (bc, bg) = (storage_bits(&mut cpu), storage_bits(&mut gpu));
        let differing = bc.iter().zip(&bg).filter(|(a, b)| a != b).count();
        assert!(differing == 0, "spot {k}: {differing} of {} storage values differ", bc.len());
        assert_eq!(pc.exploitability.to_bits(), pg.exploitability.to_bits(), "spot {k}: exploitability");
    }
    let mut cpu = Solver::new(turn_config()).unwrap();
    cpu.set_batch_river(false);
    no_batching.use_gpu(GpuPrecision::Exact).unwrap();
    cpu.solve(&opts(5), |_| true);
    no_batching.solve(&opts(5), |_| true);
    assert!(storage_bits(&mut cpu) == storage_bits(&mut no_batching), "without river batching");
}

/// Fusing opponent nodes into their parents' updates changes no bit:
/// fused and unfused plans give identical solves, in both precisions, with
/// f32 and 16-bit storage, whole and split.
#[test]
fn fused_and_unfused_solves_agree_bit_for_bit() {
    let Some(_) = device() else { return };
    for (cfg, _) in chance_spots() {
        for precision in [GpuPrecision::Exact, GpuPrecision::Fast] {
            for scratch in [Scratch::Unlimited, Scratch::ForceGroups { max_nodes: 1 }] {
                let mut runs = Vec::new();
                for fused in [true, false] {
                    let mut s = Solver::new(cfg.clone()).unwrap();
                    let plan = Plan::build_with(&s.ctx(), scratch).unwrap();
                    let plan = if fused { plan } else { plan.unfused() };
                    s.gpu = Some(Box::new(Engine::new(device().unwrap(), plan, s.store.gpu_view(), precision).unwrap()));
                    let p = s.solve(&opts(4), |_| true);
                    assert!(s.gpu_error().is_none(), "{:?}", s.gpu_error());
                    runs.push((p.exploitability.to_bits(), storage_bits(&mut s)));
                }
                assert!(runs[0] == runs[1], "{precision:?}, {scratch:?}: fused and unfused solves differ");
            }
        }
    }
}

