//! The fast path (double-single terminal sums) against the exact path and the CPU.

use super::*;

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
