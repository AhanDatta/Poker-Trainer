//! 3-way GPU traversals and solves against the CPU solver
//! (`multi::cfr::traverse`), with both terminal models. Everything here is
//! bit for bit (`GpuPrecision::Exact`): the kernels do the CPU's f32
//! operations in its order, its independent-removal terminal sums in f64
//! (whose summation order differs, which hasn't changed a single bit in any
//! test), and its exact-removal sweeps in its own f64 order. The fast
//! precision and a brute force are in `tests/exact.rs`. Tests skip without
//! an NVIDIA device.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::cfr::Discount;
use crate::gpu::multi::testutil::{arb_multi_config, sampled, spots, RANGES};
use crate::gpu::testutil::device;
use crate::gpu::GpuPrecision;
use crate::multi::cfr::traverse;
use crate::multi::opponents;
use crate::multi::tests::config;
use crate::multi::{multiway_dcfr, MultiConfig, MultiSolver};
use crate::{DcfrParams, SolveOptions, StorageMode};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod backend;
mod leaves;
mod shared;
mod split;

/// The test spots with independent opponents.
fn independent_spots() -> Vec<(&'static str, MultiConfig)> {
    spots()
        .into_iter()
        .map(|(name, mut cfg)| {
            cfg.exact_removal = false;
            (name, cfg)
        })
        .collect()
}

/// A terminal model: exact removal on or off, river batching on or off
/// (batching only takes effect with exact removal).
#[derive(Clone, Copy, Debug)]
struct Model {
    exact: bool,
    batch: bool,
}

const MODELS: [Model; 3] = [
    Model { exact: false, batch: true },
    Model { exact: true, batch: true },
    Model { exact: true, batch: false },
];

/// A solver for `cfg` under `model`.
fn solver(cfg: &MultiConfig, model: Model) -> MultiSolver {
    let mut cfg = cfg.clone();
    cfg.exact_removal = model.exact;
    let mut m = MultiSolver::new(cfg).unwrap();
    m.set_batch_river(model.batch);
    m
}

/// Every test spot under every model.
fn variants() -> Vec<(String, MultiConfig, Model)> {
    let mut out = Vec::new();
    for (name, cfg) in spots() {
        for model in MODELS {
            out.push((format!("{name} ({model:?})"), cfg.clone(), model));
        }
    }
    out
}

fn engine_for(m: &mut MultiSolver, gpu: Gpu) -> MultiEngine {
    engine_with(m, gpu, GpuPrecision::Exact)
}

fn engine_with(m: &mut MultiSolver, gpu: Gpu, precision: GpuPrecision) -> MultiEngine {
    let plan = MultiPlan::build(&m.ctx()).unwrap();
    MultiEngine::new(gpu, plan, m.store_mut().gpu_view(), precision).unwrap()
}

/// Random regrets (some exactly zero, many negative) and strategy sums
/// (some all-zero), so every branch of regret matching and normalization
/// is exercised.
fn randomize_storage(m: &mut MultiSolver, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let HostStorage::F32 { regrets, sums } = m.store_mut().gpu_view() else { panic!("f32 storage expected") };
    for x in regrets.iter_mut() {
        *x = if rng.random::<f64>() < 0.2 { 0.0 } else { rng.random_range(-50.0f32..50.0) };
    }
    for x in sums.iter_mut() {
        *x = if rng.random::<f64>() < 0.3 { 0.0 } else { rng.random_range(0.0f32..20.0) };
    }
}

/// Every regret and strategy sum, as raw bits.
fn storage_bits(m: &mut MultiSolver) -> Vec<u32> {
    match m.store_mut().gpu_view() {
        HostStorage::F32 { regrets, sums } => regrets.iter().chain(sums.iter()).map(|x| x.to_bits()).collect(),
        HostStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => regrets
            .iter()
            .map(|&x| x as u16 as u32)
            .chain(sums.iter().map(|&x| u32::from(x)))
            .chain(regret_scale.iter().chain(strategy_scale.iter()).map(|x| x.to_bits()))
            .collect(),
    }
}

/// `(differing values, total)` between two equally long bit vectors.
fn differing(a: &[u32], b: &[u32]) -> (usize, usize) {
    assert_eq!(a.len(), b.len());
    (a.iter().zip(b).filter(|(x, y)| x != y).count(), a.len())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The CPU traversal of player `p` from the root.
fn cpu_root_values(m: &MultiSolver, p: usize, mode: Mode) -> Vec<f32> {
    let ctx = m.ctx();
    let (q, r) = opponents(p);
    let w = &ctx.hands.weights;
    let mut out = vec![0f32; ctx.hands.len(p)];
    traverse(&ctx, 0, p, &w[q], &w[r], mode, false, &mut out);
    out
}

/// One CFR iteration on the CPU, as `MultiSolver::solve` runs it.
fn cpu_iteration(m: &MultiSolver, d: Discount) {
    for p in 0..N {
        cpu_root_values(m, p, Mode::Cfr(d));
    }
}

fn opts(iterations: u32, check_every: u32) -> SolveOptions {
    SolveOptions { max_iterations: iterations, target_exploitability_pct: 0.0, check_every }
}

/// Root values in every mode, and one CFR iteration (regrets and strategy
/// sums), from random storage: bit for bit on every spot.
#[test]
fn traversals_match_cpu_bit_for_bit_from_random_storage() {
    for (k, (name, cfg, model)) in variants().into_iter().enumerate() {
        let Some(gpu) = device() else { return };
        let mut cpu = solver(&cfg, model);
        let mut on_gpu = solver(&cfg, model);
        randomize_storage(&mut cpu, 7 + k as u64);
        randomize_storage(&mut on_gpu, 7 + k as u64);
        let mut engine = engine_for(&mut on_gpu, gpu);
        for p in 0..N {
            for mode in [Mode::Average, Mode::BestResponse] {
                let want = cpu_root_values(&cpu, p, mode);
                let got = engine.root_hand_values(p, mode).unwrap();
                let (diff, total) = differing(&bits(&got), &bits(&want));
                assert!(diff == 0, "{name}, player {p}, {mode:?}: {diff} of {total} values differ");
            }
        }
        for (t, weighted) in [(1u32, true), (3, false), (8, true)] {
            let params = DcfrParams { unweighted_average: !weighted, ..multiway_dcfr() };
            let d = Discount::new(t, &params);
            for p in 0..N {
                // A CFR traversal's root values too, before its updates land.
                let want = cpu_root_values(&cpu, p, Mode::Cfr(d));
                let got = engine.root_hand_values(p, Mode::Cfr(d)).unwrap();
                assert!(bits(&got) == bits(&want), "{name}, t {t}, player {p}: CFR root values");
            }
            engine.download(on_gpu.store_mut().gpu_view()).unwrap();
            let (diff, total) = differing(&storage_bits(&mut on_gpu), &storage_bits(&mut cpu));
            assert!(diff == 0, "{name}, t {t}, weighted {weighted}: {diff} of {total} storage values differ");
        }
        let (want, got) = (cpu.convergence(), engine.convergence().unwrap());
        assert_eq!(want.to_bits(), got.to_bits(), "{name}: convergence {want} vs {got}");
    }
}

/// `cfr_iteration` runs the three traversals in the CPU's order.
#[test]
fn cfr_iterations_match_cpu_bit_for_bit() {
    let (name, cfg) = independent_spots().into_iter().find(|s| s.0 == "sampled turn").unwrap();
    for model in MODELS {
        let Some(gpu) = device() else { return };
        let mut cpu = solver(&cfg, model);
        let mut on_gpu = solver(&cfg, model);
        let mut engine = engine_for(&mut on_gpu, gpu);
        for t in 1..=4 {
            let d = Discount::new(t, &multiway_dcfr());
            cpu_iteration(&cpu, d);
            engine.cfr_iteration(d).unwrap();
        }
        engine.download(on_gpu.store_mut().gpu_view()).unwrap();
        let (diff, total) = differing(&storage_bits(&mut on_gpu), &storage_bits(&mut cpu));
        assert!(diff == 0, "{name} ({model:?}): {diff} of {total} storage values differ");
    }
}

/// A GPU solve and a CPU solve of the same spot, `iterations` long with a
/// convergence check every `check_every`: same progress reports, same
/// storage, bit for bit.
fn assert_solves_agree(cfg: MultiConfig, model: Model, iterations: u32, check_every: u32, what: &str) {
    let mut cpu = solver(&cfg, model);
    let mut on_gpu = solver(&cfg, model);
    on_gpu.use_gpu(GpuPrecision::Exact).unwrap();
    let (mut seen_cpu, mut seen_gpu) = (Vec::new(), Vec::new());
    let pc = cpu.solve(&opts(iterations, check_every), |p| {
        seen_cpu.push((p.iteration, p.exploitability.to_bits()));
        true
    });
    let pg = on_gpu.solve(&opts(iterations, check_every), |p| {
        seen_gpu.push((p.iteration, p.exploitability.to_bits()));
        true
    });
    assert!(on_gpu.uses_gpu() && on_gpu.gpu_error().is_none(), "{what}: fell back: {:?}", on_gpu.gpu_error());
    assert_eq!(seen_cpu, seen_gpu, "{what}: progress reports");
    assert_eq!((pc.iteration, pc.exploitability.to_bits()), (pg.iteration, pg.exploitability.to_bits()), "{what}");
    let (diff, total) = differing(&storage_bits(&mut on_gpu), &storage_bits(&mut cpu));
    assert!(diff == 0, "{what}: {diff} of {total} storage values differ");
    // The downloaded storage answers queries like the CPU's.
    assert_eq!(cpu.query(&[]).unwrap().strategy, on_gpu.query(&[]).unwrap().strategy, "{what}: root strategy");
}

#[test]
fn short_solves_match_cpu_bit_for_bit() {
    for (name, cfg, model) in variants() {
        let Some(_) = device() else { return };
        assert_solves_agree(cfg, model, 5, 2, &name);
    }
}

/// `three_way_turn_spot` solved to 1% of the pot on both, with each
/// terminal model: the same iterations, checks and strategies.
#[test]
fn converged_solve_matches_cpu_bit_for_bit() {
    let cfg = sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 12, 1);
    for model in [MODELS[0], MODELS[1]] {
        let Some(_) = device() else { return };
        let mut cpu = solver(&cfg, model);
        let mut on_gpu = solver(&cfg, model);
        on_gpu.use_gpu(GpuPrecision::Exact).unwrap();
        let opts = SolveOptions { max_iterations: 150, target_exploitability_pct: 1.0, check_every: 25 };
        let pc = cpu.solve(&opts, |_| true);
        let pg = on_gpu.solve(&opts, |_| true);
        assert!(on_gpu.gpu_error().is_none(), "{:?}", on_gpu.gpu_error());
        assert!(pc.exploitability_pct < 5.0, "{model:?}: the CPU solve didn't converge: {pc:?}");
        assert_eq!((pc.iteration, pc.exploitability.to_bits()), (pg.iteration, pg.exploitability.to_bits()));
        assert!(storage_bits(&mut cpu) == storage_bits(&mut on_gpu), "{model:?}: storage differs");
        assert_eq!(cpu.values().map(f64::to_bits), on_gpu.values().map(f64::to_bits));
    }
}

/// A GPU solve continues a CPU solve (and the reverse) exactly where it
/// left off.
#[test]
fn continues_a_cpu_solve_bit_for_bit() {
    let Some(_) = device() else { return };
    let cfg = independent_spots().into_iter().find(|s| s.0 == "all-in turn").unwrap().1;
    let mut cpu = solver(&cfg, MODELS[1]);
    let mut mixed = solver(&cfg, MODELS[1]);
    cpu.solve(&opts(6, 3), |_| true);
    mixed.solve(&opts(3, 3), |_| true);
    mixed.use_gpu(GpuPrecision::Exact).unwrap();
    mixed.solve(&opts(5, 3), |_| true);
    mixed.release_gpu();
    let p = mixed.solve(&opts(6, 3), |_| true);
    assert!(!mixed.uses_gpu() && p.iteration == 6);
    assert!(storage_bits(&mut cpu) == storage_bits(&mut mixed), "storage differs");
}

/// Switching the terminal model or river batching drops the engine (its
/// plan is stale), and the solver goes on on the CPU.
#[test]
fn model_changes_release_the_engine() {
    let Some(_) = device() else { return };
    let (_, base) = independent_spots().into_iter().find(|s| s.0 == "river").unwrap();
    let mut m = MultiSolver::new(base).unwrap();
    m.use_gpu(GpuPrecision::Exact).unwrap();
    m.set_exact_removal(true);
    assert!(!m.uses_gpu());
    m.use_gpu(GpuPrecision::Exact).unwrap();
    m.set_batch_river(false);
    assert!(!m.uses_gpu());
    let p = m.solve(&opts(2, 2), |_| true);
    assert!(p.exploitability.is_finite() && p.iteration == 2);
}

/// `cfg` with 16-bit storage.
fn compressed(mut cfg: MultiConfig) -> MultiConfig {
    cfg.storage = StorageMode::Compressed;
    cfg
}

/// 16-bit storage (the CPU's encoding and saturation, per-node scales):
/// 5-iteration solves with their progress reports, bit for bit, on every
/// spot under every model.
#[test]
fn compressed_storage_solves_match_cpu_bit_for_bit() {
    for (name, cfg, model) in variants() {
        let Some(_) = device() else { return };
        assert_solves_agree(compressed(cfg), model, 5, 2, &format!("{name}, 16-bit"));
    }
}

/// 16-bit storage to 1% of the pot, both models: bit for bit.
#[test]
fn compressed_converged_solve_matches_cpu_bit_for_bit() {
    let cfg = compressed(sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 12, 1));
    for model in [MODELS[0], MODELS[1]] {
        let Some(_) = device() else { return };
        let mut cpu = solver(&cfg, model);
        let mut on_gpu = solver(&cfg, model);
        on_gpu.use_gpu(GpuPrecision::Exact).unwrap();
        let opts = SolveOptions { max_iterations: 150, target_exploitability_pct: 1.0, check_every: 25 };
        let pc = cpu.solve(&opts, |_| true);
        let pg = on_gpu.solve(&opts, |_| true);
        assert!(on_gpu.gpu_error().is_none(), "{:?}", on_gpu.gpu_error());
        assert!(pc.exploitability_pct < 5.0, "{model:?}: the CPU solve didn't converge: {pc:?}");
        assert_eq!((pc.iteration, pc.exploitability.to_bits()), (pg.iteration, pg.exploitability.to_bits()));
        assert!(storage_bits(&mut cpu) == storage_bits(&mut on_gpu), "{model:?}: storage differs");
    }
}

/// Guards against vacuous property tests: most generated spots are
/// solvable and fit the GPU's current limits.
#[test]
fn generated_spots_mostly_run_on_the_gpu() {
    use proptest::strategy::ValueTree;
    let Some(_) = device() else { return };
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let strategy = arb_multi_config();
    let total = 32;
    let ok = (0..total)
        .filter(|_| {
            let (cfg, _) = strategy.new_tree(&mut runner).unwrap().current();
            MultiSolver::new(cfg).is_ok_and(|mut m| m.use_gpu(GpuPrecision::Exact).is_ok())
        })
        .count();
    assert!(ok * 4 >= total * 3, "only {ok} of {total} generated spots run on the GPU");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Random spots with 16-bit storage: 3 iterations bit for bit.
    #[test]
    fn random_compressed_spots_match_cpu_bit_for_bit((cfg, batch) in arb_multi_config()) {
        let Some(_) = device() else { return Ok(()) };
        let cfg = compressed(cfg);
        if MultiSolver::new(cfg.clone()).is_err() {
            return Ok(());
        }
        let model = Model { exact: cfg.exact_removal, batch };
        let mut cpu = solver(&cfg, model);
        let mut on_gpu = solver(&cfg, model);
        on_gpu.use_gpu(GpuPrecision::Exact).unwrap();
        let pc = cpu.solve(&opts(3, 3), |_| true);
        let pg = on_gpu.solve(&opts(3, 3), |_| true);
        prop_assert!(on_gpu.gpu_error().is_none());
        prop_assert_eq!(pc.exploitability.to_bits(), pg.exploitability.to_bits());
        let (diff, total) = differing(&storage_bits(&mut on_gpu), &storage_bits(&mut cpu));
        prop_assert!(diff == 0, "{} of {} storage values differ", diff, total);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Random 3-way spots (rivers, turns and flops with sampled runouts,
    /// short stacks, a player folded at the root, rake, either terminal
    /// model, river batching on or off), from random storage: 3 iterations
    /// bit for bit.
    #[test]
    fn random_spots_match_cpu_bit_for_bit((cfg, batch) in arb_multi_config(), seed in any::<u64>()) {
        let Some(_) = device() else { return Ok(()) };
        if MultiSolver::new(cfg.clone()).is_err() {
            return Ok(());
        }
        let model = Model { exact: cfg.exact_removal, batch };
        let mut cpu = solver(&cfg, model);
        let mut on_gpu = solver(&cfg, model);
        randomize_storage(&mut cpu, seed);
        randomize_storage(&mut on_gpu, seed);
        on_gpu.use_gpu(GpuPrecision::Exact).unwrap();
        let pc = cpu.solve(&opts(3, 3), |_| true);
        let pg = on_gpu.solve(&opts(3, 3), |_| true);
        prop_assert!(on_gpu.gpu_error().is_none());
        prop_assert_eq!(pc.exploitability.to_bits(), pg.exploitability.to_bits());
        let (diff, total) = differing(&storage_bits(&mut on_gpu), &storage_bits(&mut cpu));
        prop_assert!(diff == 0, "{} of {} storage values differ", diff, total);
    }
}
