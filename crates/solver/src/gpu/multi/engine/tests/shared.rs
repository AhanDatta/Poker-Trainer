//! Trees with `MultiConfig::share_runouts` (one shared deck for every chance
//! node, so many more leaves per board) on the GPU: bit for bit with the
//! CPU at the exact precision, within the fast precision's bounds, and split
//! like any other tree.

use super::*;
use crate::gpu::multi::testutil::NARROW;
use crate::gpu::Scratch;
use crate::multi::MultiState;
use crate::ChanceMode;
use holdem::Street;

fn shared(mut cfg: MultiConfig) -> MultiConfig {
    cfg.share_runouts = true;
    cfg
}

/// The sampled test spots, plus a short-stack flop (all-in runouts on the
/// flop and the turn), all with a shared deck.
fn shared_spots() -> Vec<(String, MultiConfig)> {
    let mut flop = sampled(config("Ks7d2c", NARROW, [false; N]), 3, 2, 5);
    flop.root = MultiState::street_start(Street::Flop, 900, [600, 2000, 3000], [false; N], 100);
    let mut out: Vec<(String, MultiConfig)> = spots()
        .into_iter()
        .filter(|(_, cfg)| cfg.chance != ChanceMode::Exact)
        .map(|(name, cfg)| (name.to_string(), shared(cfg)))
        .collect();
    out.push(("short-stack flop".into(), shared(flop)));
    out
}

#[test]
fn shared_deck_solves_match_cpu_bit_for_bit() {
    for (name, cfg) in shared_spots() {
        for model in MODELS {
            let Some(_) = device() else { return };
            assert_solves_agree(cfg.clone(), model, 5, 2, &format!("{name} ({model:?})"));
        }
    }
}

/// As `fast_short_solves_match_cpu`: exploitability within 3% and the
/// root's range strategy within 0.01 after 5 iterations.
#[test]
fn shared_deck_fast_solves_match_cpu() {
    for (name, cfg) in shared_spots() {
        for model in MODELS {
            let Some(_) = device() else { return };
            let mut cpu = solver(&cfg, model);
            let mut fast = solver(&cfg, model);
            fast.use_gpu(GpuPrecision::Fast).unwrap();
            let pc = cpu.solve(&opts(5, 5), |_| true);
            let pf = fast.solve(&opts(5, 5), |_| true);
            assert!(fast.uses_gpu() && fast.gpu_error().is_none(), "{name}: {:?}", fast.gpu_error());
            let rel = (pc.exploitability - pf.exploitability).abs() / pc.exploitability.abs().max(1e-9);
            assert!(rel <= 0.03, "{name} ({model:?}): exploitability {} vs {}", pc.exploitability, pf.exploitability);
            let (a, b) = (cpu.query(&[]).unwrap().range_strategy(), fast.query(&[]).unwrap().range_strategy());
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() <= 0.01, "{name} ({model:?}): range strategy {a:?} vs {b:?}");
            }
        }
    }
}

/// A shared tree split into groups of chunks gives the whole tree's bits.
#[test]
fn shared_deck_split_solves_match_cpu_bit_for_bit() {
    let Some(_) = device() else { return };
    let (_, cfg) = shared_spots().pop().unwrap();
    for model in [MODELS[0], MODELS[1]] {
        let mut cpu = solver(&cfg, model);
        let mut split = solver(&cfg, model);
        split.use_gpu_with(device().unwrap(), GpuPrecision::Exact, Scratch::ForceGroups { max_nodes: 1 }).unwrap();
        assert!(split.gpu_engine().unwrap().plan.num_segments > 2, "{model:?}: not split");
        let pc = cpu.solve(&opts(4, 2), |_| true);
        let ps = split.solve(&opts(4, 2), |_| true);
        assert!(split.gpu_error().is_none(), "{:?}", split.gpu_error());
        assert_eq!(pc.exploitability.to_bits(), ps.exploitability.to_bits(), "{model:?}");
        let (diff, total) = differing(&storage_bits(&mut split), &storage_bits(&mut cpu));
        assert!(diff == 0, "{model:?}: {diff} of {total} storage values differ");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Random spots with a shared deck, from random storage: 3 iterations
    /// bit for bit.
    #[test]
    fn random_shared_spots_match_cpu_bit_for_bit((cfg, batch) in arb_multi_config(), seed in any::<u64>()) {
        let Some(_) = device() else { return Ok(()) };
        let cfg = shared(cfg);
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
