//! Solves that take the GPU through `SolveConfig::backend`.

use super::*;

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
