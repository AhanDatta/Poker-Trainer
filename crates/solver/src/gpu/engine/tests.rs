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
use crate::gpu::testutil::{arb_river_config, device, mid_street_config, river_config, turn_config};
use crate::{Action, DcfrParams, NodeInfo, SolveConfig, SolveOptions, Solver, Step};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn engine_for(s: &mut Solver, gpu: Gpu) -> Engine {
    let plan = Plan::build(&s.ctx()).unwrap();
    let (r, m) = s.store.f32_buffers_mut().unwrap();
    Engine::new(gpu, plan, r, m).unwrap()
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
fn solve_both(cfg: SolveConfig, iterations: u32) -> Option<(Solver, Solver)> {
    let gpu = device()?;
    drop(gpu);
    let mut cpu = Solver::new(cfg.clone()).unwrap();
    let mut on_gpu = Solver::new(cfg).unwrap();
    on_gpu.use_gpu().unwrap();
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

fn compare_nodes(cpu: &Solver, gpu: &Solver, strat_tol: f32, ev_tol: f32) {
    let paths: [&[Step]; 3] = [&[], &[Step::Action(Action::Check)], &[Step::Action(Action::Fold)]];
    for path in paths {
        if let (Ok(a), Ok(b)) = (gpu.query(path), cpu.query(path)) {
            assert_same_node(&a, &b, strat_tol, ev_tol, &format!("{path:?}"));
        }
    }
}

#[test]
fn traversal_values_match_cpu_from_random_storage() {
    for (k, cfg) in
        [river_config(), mid_street_config(false), mid_street_config(true)].into_iter().enumerate()
    {
        let Some(gpu) = device() else { return };
        let mut s = Solver::new(cfg).unwrap();
        randomize_storage(&mut s, 11 + k as u64);
        let mut engine = engine_for(&mut s, gpu);
        for p in 0..2 {
            for mode in [Mode::Average, Mode::BestResponse] {
                let mut cpu = vec![0f32; s.hands.len(p)];
                cfr::traverse(&s.ctx(), 0, p, &s.hands.weights[p ^ 1], mode, &mut cpu);
                let got = engine.root_hand_values(p, mode).unwrap();
                assert_relative(&got, &cpu, 1e-5, &format!("config {k}, player {p}, {mode:?}"));
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
        for (k, cfg) in [river_config(), mid_street_config(true)].into_iter().enumerate() {
            let Some(gpu) = device() else { return };
            let mut cpu = Solver::new(cfg.clone()).unwrap();
            let mut on_gpu = Solver::new(cfg).unwrap();
            randomize_storage(&mut cpu, 100 + t as u64);
            randomize_storage(&mut on_gpu, 100 + t as u64);
            let mut engine = engine_for(&mut on_gpu, gpu);
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
        let Some((cpu, gpu)) = solve_both(cfg.clone(), 5) else { return };
        compare_nodes(&cpu, &gpu, 5e-5, 0.25);
    }
}

#[test]
fn continues_a_cpu_solve() {
    let Some(_) = device() else { return };
    let mut reference = Solver::new(river_config()).unwrap();
    reference.solve(&opts(6), |_| true);
    let mut s = Solver::new(river_config()).unwrap();
    s.solve(&opts(3), |_| true);
    s.use_gpu().unwrap();
    s.solve(&opts(6), |_| true);
    assert!(s.uses_gpu() && s.gpu_error().is_none());
    assert_eq!(s.iterations(), 6);
    compare_nodes(&reference, &s, 5e-5, 0.25);
}

#[test]
fn declines_trees_with_chance_nodes_and_stays_on_cpu() {
    let mut s = Solver::new(turn_config()).unwrap();
    assert!(matches!(s.use_gpu(), Err(GpuError::InvalidInput(_))));
    assert!(!s.uses_gpu());
    let p = s.solve(&opts(2), |_| true);
    assert_eq!(p.iteration, 2);
}

#[test]
fn cancelled_gpu_solve_returns_and_stays_queryable() {
    let Some(_) = device() else { return };
    let mut s = Solver::new(river_config()).unwrap();
    s.use_gpu().unwrap();
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
        let mut engine = engine_for(&mut s, gpu);
        for p in 0..2 {
            for mode in [Mode::Average, Mode::BestResponse] {
                let mut cpu = vec![0f32; s.hands.len(p)];
                cfr::traverse(&s.ctx(), 0, p, &s.hands.weights[p ^ 1], mode, &mut cpu);
                let got = engine.root_hand_values(p, mode).unwrap();
                assert_relative(&got, &cpu, 1e-5, &format!("player {p}, {mode:?}"));
            }
        }
        drop(engine);
        let Some((cpu, gpu)) = solve_both(cfg, 5) else { return Ok(()) };
        // EVs scale with the pot: keep the reference test's 0.25 chips per
        // ~2500 chips of EV.
        let root = cpu.query(&[]).unwrap();
        let ev_scale = root.ev.iter().fold(0f32, |m, x| m.max(x.abs()));
        compare_nodes(&cpu, &gpu, 5e-5, 0.25 * (1.0 + ev_scale / 2500.0));
    }
}
