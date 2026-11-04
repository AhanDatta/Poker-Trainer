//! GPU kernels against their CPU twins. Tests that need a device skip (with
//! a message) when there is no NVIDIA driver or device; host-side validation
//! tests always run.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::cfr::regret_match_into;
use proptest::prelude::*;

/// The device, or `None` (test skipped) on a machine without one. Any other
/// failure to initialize is a test failure.
fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e @ (GpuError::DriverMissing | GpuError::NoDevice | GpuError::DriverTooOld { .. })) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
        Err(e) => panic!("GPU initialization failed: {e}"),
    }
}

fn cpu_regret_match<R: Copy + Into<f32>>(regrets: &[R], na: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; na * n];
    regret_match_into(regrets, na, n, &mut out);
    out
}

/// Exact equality of every value (`==`, so +0 and -0 count as equal).
fn assert_identical(gpu: &[f32], cpu: &[f32], what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: lengths differ");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(g == c, "{what}: value {i} differs: gpu {g:e} vs cpu {c:e}");
    }
}

// ---- Host-side validation (no device needed) ----

#[test]
fn block_shape_accepts_valid_blocks() {
    assert_eq!(BlockShape::new(3, 5, 15).unwrap(), BlockShape { actions: 3, hands: 5 });
    assert_eq!(BlockShape::new(1, 0, 0).unwrap(), BlockShape { actions: 1, hands: 0 });
    assert_eq!(
        BlockShape::new(1, u32::MAX as usize, u32::MAX as usize).unwrap(),
        BlockShape { actions: 1, hands: u32::MAX }
    );
}

#[test]
fn block_shape_rejects_bad_blocks() {
    let bad = [
        (0, 5, 0),                                     // no actions
        (3, 5, 14),                                    // length mismatch
        (3, 5, 16),                                    // length mismatch
        (usize::MAX, 2, 0),                            // product overflows usize
        (2, u32::MAX as usize, 2 * u32::MAX as usize), // product exceeds u32
    ];
    for (na, n, len) in bad {
        assert!(
            matches!(BlockShape::new(na, n, len), Err(GpuError::InvalidInput(_))),
            "({na}, {n}, {len}) accepted"
        );
    }
}

#[test]
fn launch_covers_every_hand_exactly_once() {
    for hands in [1u32, 255, 256, 257, 1000, 1326] {
        let cfg = BlockShape { actions: 2, hands }.per_hand_launch();
        let threads = cfg.grid_dim.0 * cfg.block_dim.0;
        assert!(threads >= hands && threads - hands < BLOCK_THREADS, "hands {hands}: {threads} threads");
    }
}

proptest! {
    #[test]
    fn block_shape_never_panics(na in any::<usize>(), n in any::<usize>(), len in any::<usize>()) {
        if let Ok(s) = BlockShape::new(na, n, len) {
            prop_assert_eq!(s.actions as usize * s.hands as usize, len);
            prop_assert!(s.actions >= 1);
        }
    }
}

// ---- Kernels (need a device) ----

#[test]
fn reports_device() {
    let Some(g) = gpu() else { return };
    eprintln!(
        "GPU: {} (sm_{}{}), bounds checked: {BOUNDS_CHECKED}",
        g.name(),
        g.compute_capability().0,
        g.compute_capability().1
    );
    assert!(g.compute_capability().0 >= 5);
}

#[test]
fn regret_match_edge_cases() {
    let Some(g) = gpu() else { return };
    let sub = f32::from_bits(1); // smallest subnormal: 1/sub overflows
    let cases: Vec<(&str, usize, Vec<f32>)> = vec![
        ("all negative -> uniform", 3, vec![-1.0, -2.0, -0.5, -3.0, -0.0, -1e-30]),
        ("all zero -> uniform", 2, vec![0.0; 8]),
        ("subnormal sum -> uniform (recip guard)", 2, vec![sub, 0.0, 0.0, -1.0]),
        ("small subnormal sum still invertible", 2, vec![f32::from_bits(0x0040_0000), 0.0, 1.0, 1.0]),
        ("sum overflows to inf -> uniform", 2, vec![f32::MAX, f32::MAX]),
        ("one action", 1, vec![5.0, -5.0, 0.0, 1e-40]),
        ("mixed", 4, vec![1.0, -1.0, 2.0, 0.0, 3.0, 4.0, -2.0, 1e30]),
    ];
    for (what, na, regrets) in cases {
        let n = regrets.len() / na;
        let got = g.regret_match(&regrets, na, n).unwrap();
        assert!(got.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)), "{what}: {got:?}");
        assert_identical(&got, &cpu_regret_match(&regrets, na, n), what);
    }
}

#[test]
fn regret_match_straddles_block_boundaries() {
    let Some(g) = gpu() else { return };
    for n in [1usize, 255, 256, 257, 511, 513, 1326] {
        let na = 3;
        let regrets: Vec<f32> = (0..na * n).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
        let got = g.regret_match(&regrets, na, n).unwrap();
        assert_identical(&got, &cpu_regret_match(&regrets, na, n), &format!("n = {n}"));
    }
}

#[test]
fn regret_match_empty_and_invalid_inputs() {
    let Some(g) = gpu() else { return };
    assert!(g.regret_match::<f32>(&[], 3, 0).unwrap().is_empty());
    assert!(matches!(g.regret_match::<f32>(&[], 0, 0), Err(GpuError::InvalidInput(_))));
    assert!(matches!(g.regret_match(&[1.0f32; 5], 2, 3), Err(GpuError::InvalidInput(_))));
    // The device is still usable after rejected inputs.
    assert_eq!(g.regret_match(&[1.0f32, 3.0], 2, 1).unwrap(), vec![0.25, 0.75]);
}

#[test]
fn regret_match_compressed_extremes() {
    let Some(g) = gpu() else { return };
    let regrets: Vec<i16> = vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MAX, i16::MIN, i16::MIN];
    let got = g.regret_match(&regrets, 2, 4).unwrap();
    assert_identical(&got, &cpu_regret_match(&regrets, 2, 4), "i16 extremes");
}

/// Finite `f32` regrets with the awkward values over-represented: zeros of
/// both signs, subnormals, and magnitudes up to `f32::MAX`.
fn regret_value() -> impl Strategy<Value = f32> + Clone {
    prop_oneof![
        4 => -100.0f32..100.0,
        1 => Just(0.0f32),
        1 => Just(-0.0f32),
        1 => prop::num::f32::SUBNORMAL | prop::num::f32::POSITIVE | prop::num::f32::NEGATIVE,
        1 => prop::num::f32::NORMAL | prop::num::f32::POSITIVE | prop::num::f32::NEGATIVE,
    ]
}

fn block<T: std::fmt::Debug>(
    value: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (usize, usize, Vec<T>)> {
    (1usize..=8, 0usize..=1100)
        .prop_flat_map(move |(na, n)| (Just(na), Just(n), prop::collection::vec(value.clone(), na * n)))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn regret_match_f32_matches_cpu((na, n, regrets) in block(regret_value())) {
        let Some(g) = gpu() else { return Ok(()) };
        let got = g.regret_match(&regrets, na, n).unwrap();
        let cpu = cpu_regret_match(&regrets, na, n);
        prop_assert_eq!(got.len(), cpu.len());
        for (i, (x, y)) in got.iter().zip(&cpu).enumerate() {
            prop_assert!(x == y, "value {} differs: gpu {:e} vs cpu {:e}", i, x, y);
        }
    }

    #[test]
    fn regret_match_i16_matches_cpu((na, n, regrets) in block(any::<i16>())) {
        let Some(g) = gpu() else { return Ok(()) };
        let got = g.regret_match(&regrets, na, n).unwrap();
        let cpu = cpu_regret_match(&regrets, na, n);
        prop_assert_eq!(got.len(), cpu.len());
        for (i, (x, y)) in got.iter().zip(&cpu).enumerate() {
            prop_assert!(x == y, "value {} differs: gpu {:e} vs cpu {:e}", i, x, y);
        }
    }
}
