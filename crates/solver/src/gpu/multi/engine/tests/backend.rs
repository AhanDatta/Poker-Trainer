//! 3-way solves that take the GPU through `MultiConfig::backend`.

use super::*;

fn with_backend(mut cfg: MultiConfig, backend: crate::Backend) -> MultiConfig {
    cfg.backend = backend;
    cfg
}

/// `Backend::Gpu` runs each solve on the GPU (bit for bit as an explicit
/// `use_gpu` with the default precision) and frees it afterwards; the tag
/// says where it ran.
#[test]
fn gpu_backend_matches_explicit_use_gpu() {
    let turn = sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 6, 1);
    for (cfg, model) in [(turn.clone(), MODELS[0]), (turn.clone(), MODELS[1]), (compressed(turn), MODELS[1])] {
        let Some(_) = device() else { return };
        let mut explicit = solver(&cfg, model);
        explicit.use_gpu(GpuPrecision::default()).unwrap();
        let pe = explicit.solve(&opts(6, 3), |_| true);
        assert!(explicit.uses_gpu(), "explicit use_gpu stays on");
        let mut backed = solver(&with_backend(cfg, crate::Backend::Gpu), model);
        assert!(!backed.uses_gpu(), "the GPU is taken only while solving");
        let pb = backed.solve(&opts(6, 3), |_| true);
        assert!(!backed.uses_gpu() && backed.gpu_error().is_none(), "{:?}", backed.gpu_error());
        assert_eq!(backed.solved_on(), crate::SolvedOn::GpuFast, "the tag outlives the engine");
        assert_eq!(explicit.solved_on(), crate::SolvedOn::GpuFast);
        assert_eq!(pe.exploitability.to_bits(), pb.exploitability.to_bits(), "{model:?}");
        assert!(storage_bits(&mut explicit) == storage_bits(&mut backed), "{model:?}: storage differs");
    }
}

/// A GPU-backed solve continued by a second `solve` call (the GPU taken
/// again, the storage uploaded again) ends where one call would.
#[test]
fn gpu_backend_continues_across_solve_calls() {
    let Some(_) = device() else { return };
    let cfg = with_backend(sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 6, 1), crate::Backend::Gpu);
    let mut once = solver(&cfg, MODELS[1]);
    once.solve(&opts(6, 3), |_| true);
    let mut twice = solver(&cfg, MODELS[1]);
    twice.solve(&opts(3, 3), |_| true);
    twice.solve(&opts(6, 3), |_| true);
    assert_eq!(twice.iterations(), 6);
    assert!(storage_bits(&mut once) == storage_bits(&mut twice), "storage differs");
}

/// `Backend::Auto` solves turn roots on the GPU (same bits as `Gpu`) and
/// river roots on the CPU (same bits as the CPU).
#[test]
fn auto_backend_uses_the_gpu_off_the_river() {
    let Some(_) = device() else { return };
    let turn = sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 6, 1);
    let mut gpu = solver(&with_backend(turn.clone(), crate::Backend::Gpu), MODELS[1]);
    let pg = gpu.solve(&opts(6, 3), |_| true);
    let mut auto = solver(&with_backend(turn, crate::Backend::Auto), MODELS[1]);
    let pa = auto.solve(&opts(6, 3), |_| true);
    assert!(auto.gpu_error().is_none() && auto.solved_on() == crate::SolvedOn::GpuFast);
    assert_eq!(pg.exploitability.to_bits(), pa.exploitability.to_bits());
    assert!(storage_bits(&mut gpu) == storage_bits(&mut auto), "storage differs");

    let river = config("Jh8d4c2s9s", RANGES, [false; N]);
    let mut cpu = solver(&river, MODELS[1]);
    let pc = cpu.solve(&opts(6, 3), |_| true);
    let mut auto = solver(&with_backend(river, crate::Backend::Auto), MODELS[1]);
    let pa = auto.solve(&opts(6, 3), |_| true);
    assert_eq!(auto.solved_on(), crate::SolvedOn::Cpu);
    assert!(auto.gpu_error().is_none());
    assert_eq!(pc.exploitability.to_bits(), pa.exploitability.to_bits());
    assert!(storage_bits(&mut cpu) == storage_bits(&mut auto), "storage differs");
}
