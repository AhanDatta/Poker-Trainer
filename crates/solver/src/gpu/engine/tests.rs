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
use crate::gpu::testutil::{
    allin_turn_config, arb_river_config, arb_turn_config, device, iso_turn_config, mid_street_config,
    river_config, sampled_flop_config, turn_config,
};
use crate::gpu::GpuPrecision;
use crate::{Action, DcfrParams, NodeInfo, SolveConfig, SolveOptions, Solver, Step, StorageMode};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn engine_for(s: &mut Solver, gpu: Gpu, precision: GpuPrecision) -> Engine {
    let plan = Plan::build(&s.ctx()).unwrap();
    let (r, m) = s.store.f32_buffers_mut().unwrap();
    Engine::new(gpu, plan, r, m, precision).unwrap()
}

/// Random regrets (some exactly zero, many negative) and strategy sums
/// (some all-zero), so every branch of regret matching and normalization
/// is exercised.
fn randomize_storage(s: &mut Solver, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (r, m) = s.store.f32_buffers_mut().unwrap();
    for x in r.iter_mut() {
        *x = if rng.random::<f64>() < 0.2 { 0.0 } else { rng.random_range(-50.0f32..50.0) };
    }
    for x in m.iter_mut() {
        *x = if rng.random::<f64>() < 0.3 { 0.0 } else { rng.random_range(0.0f32..20.0) };
    }
}

fn storage_copy(s: &mut Solver) -> (Vec<f32>, Vec<f32>) {
    let (r, m) = s.store.f32_buffers_mut().unwrap();
    (r.to_vec(), m.to_vec())
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
fn chance_spots() -> Vec<(SolveConfig, &'static [&'static str])> {
    vec![
        (turn_config(), &["Ks", "2h"]),
        (iso_turn_config(), &["Kc", "Kh", "3s"]),
        (allin_turn_config(), &["Ah"]),
        (sampled_flop_config(), &[]),
    ]
}

#[test]
fn traversal_values_match_cpu_from_random_storage() {
    let river_spots = [river_config(), mid_street_config(false), mid_street_config(true)];
    let spots = river_spots.into_iter().chain(chance_spots().into_iter().map(|(c, _)| c));
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
                let (r, m) = on_gpu.store.f32_buffers_mut().unwrap();
                engine.download(r, m).unwrap();
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
    for cfg in [river_config(), mid_street_config(false), mid_street_config(true)] {
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
fn declines_compressed_storage_and_stays_on_cpu() {
    let mut cfg = turn_config();
    cfg.storage = StorageMode::Compressed;
    let mut s = Solver::new(cfg).unwrap();
    assert!(matches!(s.use_gpu(GpuPrecision::Exact), Err(GpuError::InvalidInput(_))));
    assert!(!s.uses_gpu());
    let p = s.solve(&opts(2), |_| true);
    assert_eq!(p.iteration, 2);
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
    for cfg in [turn_config(), iso_turn_config(), allin_turn_config(), sampled_flop_config()] {
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
                let mut checked = 0;
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
                    assert_relative(&values[id * n..(id + 1) * n], &cpu, 1e-5, &what);
                    checked += 1;
                }
                assert!(checked > 100, "only {checked} nodes checked");
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
    let river_spots = [river_config(), mid_street_config(false), mid_street_config(true)];
    let spots = river_spots.into_iter().map(|c| (c, &[][..])).chain(chance_spots());
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
    for cfg in [turn_config(), iso_turn_config(), sampled_flop_config()] {
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
