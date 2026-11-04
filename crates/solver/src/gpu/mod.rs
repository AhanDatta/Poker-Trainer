//! Optional CUDA backend (the `gpu` feature). Experimental: the solver doesn't
//! use it yet; this module grows kernel by kernel, each one tested against
//! its CPU twin.
//!
//! Safety rules (see `cuda/README.md`): nothing here panics. Every fallible
//! operation returns [`GpuError`], kernel inputs are validated on the host
//! before launch, and the two cases in which cudarc itself would panic (no
//! driver library, a driver older than the bindings) are checked first.
//! The lints below keep it that way.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use cudarc::driver::sys::{self, CUdevice_attribute, CUresult};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
mod tests;

/// Kernel images built by `build.rs` from `cuda/kernels/*.cu`.
static REGRET_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/regret.fatbin"));
static SELFTEST_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/selftest.fatbin"));

/// Whether the kernels were built with device bounds assertions
/// (`CCCL_ENABLE_DEVICE_ASSERTIONS`): on in debug/test builds, off in release
/// unless `SOLVER_GPU_CHECKED=1`.
pub const BOUNDS_CHECKED: bool = matches!(env!("SOLVER_GPU_CHECKED_BUILD").as_bytes(), b"1");

/// Oldest driver API the bindings (`cuda-13000`) may call into: 13.0.
const MIN_DRIVER_VERSION: i32 = 13_000;

/// Threads per block for one-thread-per-hand kernels.
const BLOCK_THREADS: u32 = 256;

/// Why the GPU backend can't be used or an operation failed.
#[derive(Debug)]
pub enum GpuError {
    /// No NVIDIA driver library on this machine.
    DriverMissing,
    /// The driver is older than the CUDA version the bindings target.
    DriverTooOld { found: i32, required: i32 },
    /// The driver works but reports no CUDA device.
    NoDevice,
    /// A CUDA driver call failed (including faults inside a kernel, which
    /// surface at the next synchronization and leave the context unusable).
    Driver(DriverError),
    /// Inputs rejected on the host before any launch.
    InvalidInput(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DriverMissing => write!(f, "no NVIDIA driver found"),
            GpuError::DriverTooOld { found, required } => write!(
                f,
                "NVIDIA driver supports CUDA {}.{}, need {}.{} or newer",
                found / 1000,
                found % 1000 / 10,
                required / 1000,
                required % 1000 / 10
            ),
            GpuError::NoDevice => write!(f, "no CUDA device found"),
            GpuError::Driver(e) => write!(f, "CUDA error: {e}"),
            GpuError::InvalidInput(s) => write!(f, "invalid GPU input: {s}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::Driver(e)
    }
}

fn invalid(msg: impl Into<String>) -> GpuError {
    GpuError::InvalidInput(msg.into())
}

/// Shape of one decision node's action-major storage block, validated for
/// the kernels: at least one action, `actions * hands == len`, and every
/// index fits in `u32` (the kernels' index type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockShape {
    pub actions: u32,
    pub hands: u32,
}

impl BlockShape {
    pub(crate) fn new(actions: usize, hands: usize, len: usize) -> Result<BlockShape, GpuError> {
        if actions == 0 {
            return Err(invalid("a decision node needs at least one action"));
        }
        let cells = actions
            .checked_mul(hands)
            .ok_or_else(|| invalid(format!("{actions} actions x {hands} hands overflows")))?;
        if cells != len {
            return Err(invalid(format!("block of {actions} x {hands} needs {cells} values, got {len}")));
        }
        if u32::try_from(cells).is_err() {
            return Err(invalid(format!("block of {cells} values exceeds the kernels' u32 indexing")));
        }
        // Both fit because their product (with actions >= 1) does.
        let actions = u32::try_from(actions).map_err(|_| invalid("too many actions"))?;
        let hands = u32::try_from(hands).map_err(|_| invalid("too many hands"))?;
        Ok(BlockShape { actions, hands })
    }

    /// One thread per hand.
    fn per_hand_launch(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.hands.div_ceil(BLOCK_THREADS), 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i16 {}
}

/// Element types a regret block is stored in (`f32`, or `i16` in compressed
/// storage). Sealed: each one needs a matching kernel instantiation.
pub trait RegretValue: DeviceRepr + Copy + sealed::Sealed {
    #[doc(hidden)]
    fn regret_match_kernel(gpu: &Gpu) -> &CudaFunction;
}

impl RegretValue for f32 {
    fn regret_match_kernel(gpu: &Gpu) -> &CudaFunction {
        &gpu.kernels.regret_match_f32
    }
}

impl RegretValue for i16 {
    fn regret_match_kernel(gpu: &Gpu) -> &CudaFunction {
        &gpu.kernels.regret_match_i16
    }
}

struct Kernels {
    regret_match_f32: CudaFunction,
    regret_match_i16: CudaFunction,
    selftest_out_of_bounds: CudaFunction,
}

/// A CUDA device with the solver's kernels loaded.
pub struct Gpu {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    name: String,
    compute_capability: (i32, i32),
}

impl fmt::Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu")
            .field("name", &self.name)
            .field("compute_capability", &self.compute_capability)
            .finish()
    }
}

impl Gpu {
    /// Opens device 0 and loads the kernels.
    pub fn new() -> Result<Gpu, GpuError> {
        check_driver()?;
        if CudaContext::device_count()? < 1 {
            return Err(GpuError::NoDevice);
        }
        let ctx = CudaContext::new(0)?;
        let name = ctx.name()?;
        let compute_capability = (
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        );
        // A GPU without code for its architecture in the images fails here
        // with an error (not a panic).
        let regret = ctx.load_module(Ptx::from_binary(REGRET_IMAGE.to_vec()))?;
        let selftest = ctx.load_module(Ptx::from_binary(SELFTEST_IMAGE.to_vec()))?;
        let kernels = Kernels {
            regret_match_f32: regret.load_function("regret_match_f32")?,
            regret_match_i16: regret.load_function("regret_match_i16")?,
            selftest_out_of_bounds: selftest.load_function("selftest_out_of_bounds")?,
        };
        Ok(Gpu { stream: ctx.default_stream(), kernels, name, compute_capability })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// `(major, minor)`, e.g. `(8, 9)` for the RTX 40 series.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Regret matching of one node (`regrets` action-major, `num_actions *
    /// num_hands` values): the GPU twin of `cfr::regret_match_into`, with
    /// identical results.
    pub fn regret_match<R: RegretValue>(
        &self,
        regrets: &[R],
        num_actions: usize,
        num_hands: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let shape = BlockShape::new(num_actions, num_hands, regrets.len())?;
        if shape.hands == 0 {
            return Ok(Vec::new());
        }
        let d_regrets = self.stream.clone_htod(regrets)?;
        let mut d_strategy = self.stream.alloc_zeros::<f32>(regrets.len())?;
        let mut launch = self.stream.launch_builder(R::regret_match_kernel(self));
        launch.arg(&d_regrets).arg(&mut d_strategy).arg(&shape.actions).arg(&shape.hands);
        // SAFETY: the arguments match `regret_match_{f32,i16}(const R*, float*,
        // u32 actions, u32 hands)`, and both buffers hold `actions * hands`
        // elements (checked by `BlockShape`), which is all the kernel touches.
        unsafe { launch.launch(shape.per_hand_launch()) }?;
        Ok(self.stream.clone_dtoh(&d_strategy)?)
    }

    /// Launches a kernel that reads one element past the end of its input,
    /// to prove that checked builds turn that into an error. Leaves the
    /// context unusable afterwards; only for `tests/gpu_fault.rs`.
    #[doc(hidden)]
    pub fn selftest_out_of_bounds(&self) -> Result<f32, GpuError> {
        let data = self.stream.clone_htod(&[1.0f32, 2.0, 3.0])?;
        let mut out = self.stream.alloc_zeros::<f32>(1)?;
        let len: u32 = 3;
        let mut launch = self.stream.launch_builder(&self.kernels.selftest_out_of_bounds);
        launch.arg(&data).arg(&len).arg(&mut out);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: deliberately not safe: the kernel reads `data[3]`. In
        // checked builds the bounds assertion traps before the read.
        unsafe { launch.launch(cfg) }?;
        let v = self.stream.clone_dtoh(&out)?;
        v.first().copied().ok_or_else(|| invalid("empty selftest output"))
    }
}

/// The checks that keep cudarc from panicking: its driver loader panics when
/// the library is missing, and calling an entry point newer than the
/// installed driver panics on the missing symbol.
fn check_driver() -> Result<(), GpuError> {
    // SAFETY: only tries to open the driver library.
    if !unsafe { sys::is_culib_present() } {
        return Err(GpuError::DriverMissing);
    }
    let mut version = 0i32;
    // SAFETY: the library is present, `cuDriverGetVersion` exists in every
    // driver, may be called before `cuInit`, and only writes one int.
    let status = unsafe { sys::cuDriverGetVersion(&mut version) };
    if status != CUresult::CUDA_SUCCESS {
        return Err(GpuError::Driver(DriverError(status)));
    }
    if version < MIN_DRIVER_VERSION {
        return Err(GpuError::DriverTooOld { found: version, required: MIN_DRIVER_VERSION });
    }
    Ok(())
}
