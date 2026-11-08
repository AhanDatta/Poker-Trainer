//! 3-way trees split into groups of subtrees give the same bits as whole
//! trees.

use super::*;
use crate::gpu::Scratch;

/// A solver of `cfg` under `model` on the GPU with its tree split as
/// `scratch` says, from storage randomized with `seed` (zero storage
/// without one).
fn gpu_solver(cfg: &MultiConfig, model: Model, precision: GpuPrecision, scratch: Scratch, seed: Option<u64>) -> MultiSolver {
    let mut m = solver(cfg, model);
    if let Some(seed) = seed {
        randomize_storage(&mut m, seed);
    }
    m.use_gpu_with(device().unwrap(), precision, scratch).unwrap();
    m
}

fn segments(m: &mut MultiSolver) -> u32 {
    m.gpu_engine().unwrap().plan.num_segments
}

/// Splitting must not change a single bit: every group runs the same
/// kernels on the same inputs, only in other value rows and reach slots.
fn assert_split_matches_whole(
    cfg: &MultiConfig,
    model: Model,
    precision: GpuPrecision,
    scratch: Scratch,
    seed: Option<u64>,
    iterations: u32,
    what: &str,
) {
    let mut whole = gpu_solver(cfg, model, precision, Scratch::Unlimited, seed);
    let mut split = gpu_solver(cfg, model, precision, scratch, seed);
    assert_eq!(segments(&mut whole), 1);
    assert!(segments(&mut split) > 1, "{what}, {scratch:?}: tree not split");
    let pw = whole.solve(&opts(iterations, 2), |_| true);
    let ps = split.solve(&opts(iterations, 2), |_| true);
    assert!(whole.uses_gpu() && whole.gpu_error().is_none(), "{:?}", whole.gpu_error());
    assert!(split.uses_gpu() && split.gpu_error().is_none(), "{:?}", split.gpu_error());
    assert_eq!(
        pw.exploitability.to_bits(),
        ps.exploitability.to_bits(),
        "{what}, {precision:?}, {scratch:?}: convergence {} vs {}",
        pw.exploitability,
        ps.exploitability
    );
    let (diff, total) = differing(&storage_bits(&mut whole), &storage_bits(&mut split));
    assert!(diff == 0, "{what}, {precision:?}, {scratch:?}: {diff} of {total} storage values differ");
}

/// The spots with chunks (chance nodes on the root street).
fn chance_variants() -> Vec<(String, MultiConfig, Model)> {
    variants().into_iter().filter(|(name, _, _)| !name.starts_with("river")).collect()
}

#[test]
fn split_solves_match_whole_solves_bit_for_bit() {
    for (name, cfg, model) in chance_variants() {
        let Some(_) = device() else { return };
        let nodes = MultiSolver::new(cfg.clone()).unwrap().ctx().tree.nodes.len() as u32;
        for precision in [GpuPrecision::Exact, GpuPrecision::Fast] {
            // (One chunk per group: the random-storage test below.)
            let max_nodes = nodes / 4;
            assert_split_matches_whole(&cfg, model, precision, Scratch::ForceGroups { max_nodes }, None, 4, &name);
        }
    }
}

/// From random storage (dead subtrees inside the groups), and with 16-bit
/// storage.
#[test]
fn split_solves_match_whole_solves_from_random_storage_and_16_bit() {
    for (k, (name, cfg, model)) in chance_variants().into_iter().enumerate() {
        let Some(_) = device() else { return };
        let precision = if k % 2 == 0 { GpuPrecision::Exact } else { GpuPrecision::Fast };
        assert_split_matches_whole(&cfg, model, precision, Scratch::ForceGroups { max_nodes: 1 }, Some(k as u64), 2, &name);
        let mut cfg16 = cfg.clone();
        cfg16.storage = StorageMode::Compressed;
        assert_split_matches_whole(&cfg16, model, precision, Scratch::ForceGroups { max_nodes: 1 }, None, 3, &name);
    }
}

/// A split exact-precision solve is still the CPU's, bit for bit.
#[test]
fn split_solves_match_cpu_bit_for_bit() {
    let Some(_) = device() else { return };
    let cfg = sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 12, 1);
    for model in [MODELS[0], MODELS[1]] {
        let mut cpu = solver(&cfg, model);
        let mut split = gpu_solver(&cfg, model, GpuPrecision::Exact, Scratch::ForceGroups { max_nodes: 50 }, None);
        assert!(segments(&mut split) > 2);
        let pc = cpu.solve(&opts(6, 3), |_| true);
        let ps = split.solve(&opts(6, 3), |_| true);
        assert_eq!(pc.exploitability.to_bits(), ps.exploitability.to_bits(), "{model:?}");
        assert!(storage_bits(&mut cpu) == storage_bits(&mut split), "{model:?}: storage differs");
    }
}

/// A memory budget below the whole tree's needs splits it, with the same
/// results; one too small for any group leaves the solver on the CPU.
#[test]
fn budget_split_solves_match_whole_solves() {
    let Some(_) = device() else { return };
    let cfg = sampled(config("Ks7d2c", crate::gpu::multi::testutil::NARROW, [false; N]), 2, 2, 5);
    let m = solver(&cfg, MODELS[1]);
    let fixed = MultiEngine::fixed_bytes();
    let whole = MultiPlan::build_with(&m.ctx(), Scratch::Unlimited, fixed).unwrap();
    // The whole tree's split-dependent scratch, as `MultiPlan::segment` counts it.
    let hands = whole.players.iter().map(|pp| u64::from(pp.hands)).max().unwrap();
    let values = u64::from(whole.num_rows) * hands * 4;
    // A budget that only leaves room for half the value rows forces a split.
    let budget = fixed + values / 2;
    let split = MultiPlan::build_with(&m.ctx(), Scratch::Budget(budget), fixed).unwrap();
    assert!(split.num_segments > 1, "a budget of {budget} bytes didn't split the tree");
    assert_split_matches_whole(&cfg, MODELS[1], GpuPrecision::Exact, Scratch::Budget(budget), None, 4, "budget");
    let tiny = MultiPlan::build_with(&m.ctx(), Scratch::Budget(fixed), fixed);
    assert!(matches!(tiny, Err(GpuError::OutOfMemory { .. })), "{:?}", tiny.map(|p| p.num_segments));
}
