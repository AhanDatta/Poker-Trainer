//! Trees split into groups of subtrees give the same bits as whole trees.

use super::*;

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
