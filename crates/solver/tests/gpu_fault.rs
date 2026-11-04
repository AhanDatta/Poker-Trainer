//! The GPU safety net, end to end: an out-of-bounds access inside a kernel
//! must come back as an error, not a crash or a silent read. This file is
//! its own test process because the fault leaves the CUDA context unusable,
//! which would break any other GPU test sharing the process.
#![cfg(feature = "gpu")]

use solver::gpu::{Gpu, GpuError, BOUNDS_CHECKED};

#[test]
fn out_of_bounds_access_becomes_an_error() {
    if !BOUNDS_CHECKED {
        eprintln!("skipping: kernels built without bounds assertions (release build)");
        return;
    }
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e @ (GpuError::DriverMissing | GpuError::NoDevice | GpuError::DriverTooOld { .. })) => {
            eprintln!("skipping GPU test: {e}");
            return;
        }
        Err(e) => panic!("GPU initialization failed: {e}"),
    };
    match gpu.selftest_out_of_bounds() {
        Err(GpuError::Driver(e)) => eprintln!("fault reported as expected: {e}"),
        other => panic!("out-of-bounds read was not caught: {other:?}"),
    }
}
