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

mod device;
mod engine;
mod multi;
mod plan;
mod segments;
mod shared;
mod tables;
#[cfg(test)]
mod testutil;

pub(crate) use engine::{Engine, HostStorage};
pub(crate) use multi::engine::MultiEngine;
pub(crate) use multi::plan::MultiPlan;

/// Operations of the double-single test kernel (`Op` in
/// double_float_check.cu).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DoubleFloatOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    AddF32 = 3,
    RoundToF32 = 4,
}

/// How the GPU accumulates terminal sums (showdowns and folds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuPrecision {
    /// Double-single arithmetic (pairs of f32, about 44 bits) on the GPU's
    /// fast f32 units. Agrees with `Exact` except for rare one-rounding-step
    /// differences in terminal values, which CFR then carries forward.
    #[default]
    Fast,
    /// f64, in the CPU's operation order: bit-identical to the CPU solver in
    /// every test so far, and slower (this GPU runs f64 at 1/64 of its f32
    /// rate).
    Exact,
}
pub(crate) use plan::{Plan, Scratch};

/// Kernel images built by `build.rs` from `cuda/kernels/*.cu`.
static REGRET_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/regret.fatbin"));
static SELFTEST_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/selftest.fatbin"));
static TRAVERSE_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/traverse.fatbin"));
static TRAVERSE_COMPRESSED_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/traverse_compressed.fatbin"));
static TERMINAL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/terminal.fatbin"));
static MULTI_TRAVERSE_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/multi_traverse.fatbin"));
static MULTI_LEAVES_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/multi_leaves.fatbin"));
static MULTI_TRAVERSE_COMPRESSED_IMAGE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/multi_traverse_compressed.fatbin"));
static DOUBLE_FLOAT_CHECK_IMAGE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/double_float_check.fatbin"));

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
    /// The solve needs more device memory than is free (bytes; a margin
    /// is kept for the display and the CUDA runtime).
    OutOfMemory { needed: u64, available: u64 },
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
            GpuError::OutOfMemory { needed, available } => write!(
                f,
                "the solve needs {:.2} GB of GPU memory, {:.2} GB is available",
                *needed as f64 / 1e9,
                *available as f64 / 1e9
            ),
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
    begin_traversal: CudaFunction,
    reset_alive: CudaFunction,
    opponent_split: CudaFunction,
    own_update: CudaFunction,
    opponent_sum: CudaFunction,
    chance_split: CudaFunction,
    chance_sum: CudaFunction,
    opponent_split_compressed: CudaFunction,
    own_update_compressed: CudaFunction,
    /// Terminal kernels per precision: `[exact, fast]`.
    fold_values: [CudaFunction; 2],
    showdown_values: [CudaFunction; 2],
    /// Three-player traversal kernels (multi_traverse.cu).
    multi_opponent_split: CudaFunction,
    multi_chance_split: CudaFunction,
    multi_own_update: CudaFunction,
    multi_opponent_sum: CudaFunction,
    multi_chance_sum: CudaFunction,
    /// The 16-bit storage steps (multi_traverse_compressed.cu).
    multi_opponent_split_compressed: CudaFunction,
    multi_own_update_compressed: CudaFunction,
    /// Three-player leaves (multi_leaves.cu), `[model][precision]`:
    /// `[independent, exact removal][exact, fast]`.
    multi_showdowns: [[CudaFunction; 2]; 2],
    multi_fixed: [[CudaFunction; 2]; 2],
    double_float_check: CudaFunction,
    double_float_sum: CudaFunction,
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
        let traverse = ctx.load_module(Ptx::from_binary(TRAVERSE_IMAGE.to_vec()))?;
        let traverse_compressed = ctx.load_module(Ptx::from_binary(TRAVERSE_COMPRESSED_IMAGE.to_vec()))?;
        let terminal = ctx.load_module(Ptx::from_binary(TERMINAL_IMAGE.to_vec()))?;
        let double_float = ctx.load_module(Ptx::from_binary(DOUBLE_FLOAT_CHECK_IMAGE.to_vec()))?;
        let multi_traverse = ctx.load_module(Ptx::from_binary(MULTI_TRAVERSE_IMAGE.to_vec()))?;
        let multi_leaves = ctx.load_module(Ptx::from_binary(MULTI_LEAVES_IMAGE.to_vec()))?;
        let multi_compressed = ctx.load_module(Ptx::from_binary(MULTI_TRAVERSE_COMPRESSED_IMAGE.to_vec()))?;
        let leaf = |name: &str| multi_leaves.load_function(name);
        let kernels = Kernels {
            regret_match_f32: regret.load_function("regret_match_f32")?,
            regret_match_i16: regret.load_function("regret_match_i16")?,
            selftest_out_of_bounds: selftest.load_function("selftest_out_of_bounds")?,
            begin_traversal: traverse.load_function("begin_traversal")?,
            reset_alive: traverse.load_function("reset_alive")?,
            opponent_split: traverse.load_function("opponent_split")?,
            own_update: traverse.load_function("own_update")?,
            opponent_sum: traverse.load_function("opponent_sum")?,
            chance_split: traverse.load_function("chance_split")?,
            chance_sum: traverse.load_function("chance_sum")?,
            opponent_split_compressed: traverse_compressed.load_function("opponent_split_compressed")?,
            own_update_compressed: traverse_compressed.load_function("own_update_compressed")?,
            fold_values: [terminal.load_function("fold_values_exact")?, terminal.load_function("fold_values_fast")?],
            showdown_values: [
                terminal.load_function("showdown_values_exact")?,
                terminal.load_function("showdown_values_fast")?,
            ],
            multi_opponent_split: multi_traverse.load_function("multi_opponent_split")?,
            multi_chance_split: multi_traverse.load_function("multi_chance_split")?,
            multi_own_update: multi_traverse.load_function("multi_own_update")?,
            multi_opponent_sum: multi_traverse.load_function("multi_opponent_sum")?,
            multi_chance_sum: multi_traverse.load_function("multi_chance_sum")?,
            multi_opponent_split_compressed: multi_compressed.load_function("multi_opponent_split_compressed")?,
            multi_own_update_compressed: multi_compressed.load_function("multi_own_update_compressed")?,
            multi_showdowns: [
                [leaf("multi_showdowns_independent_exact")?, leaf("multi_showdowns_independent_fast")?],
                [leaf("multi_showdowns_exact_exact")?, leaf("multi_showdowns_exact_fast")?],
            ],
            multi_fixed: [
                [leaf("multi_fixed_independent_exact")?, leaf("multi_fixed_independent_fast")?],
                [leaf("multi_fixed_exact_exact")?, leaf("multi_fixed_exact_fast")?],
            ],
            double_float_check: double_float.load_function("double_float_check")?,
            double_float_sum: double_float.load_function("double_float_sum")?,
        };
        // The leaf kernels use no shared memory and stream per-lane state:
        // all of the on-chip memory as L1.
        for f in kernels.multi_showdowns.iter().chain(&kernels.multi_fixed).flatten() {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                0,
            )?;
        }
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

    /// `op(x[i], y[i])` in double-single arithmetic, read back exactly as
    /// f64 (tests of `double_float.cuh`).
    #[doc(hidden)]
    pub fn double_float_check(&self, x: &[f64], y: &[f64], op: DoubleFloatOp) -> Result<Vec<f64>, GpuError> {
        if x.len() != y.len() {
            return Err(invalid("operand lengths differ"));
        }
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let len = u32::try_from(x.len()).map_err(|_| invalid("too many operands"))?;
        let dx = self.stream.clone_htod(x)?;
        let dy = self.stream.clone_htod(y)?;
        let mut out = self.stream.alloc_zeros::<f64>(x.len())?;
        let op = op as u32;
        let mut launch = self.stream.launch_builder(&self.kernels.double_float_check);
        launch.arg(&dx).arg(&dy).arg(&mut out).arg(&len).arg(&op);
        let cfg = LaunchConfig::for_num_elems(len);
        // SAFETY: arguments match `double_float_check(const double*, const
        // double*, double*, u32 len, u32 op)`; all buffers hold `len` values.
        unsafe { launch.launch(cfg) }?;
        Ok(self.stream.clone_dtoh(&out)?)
    }

    /// The sequential double-single sum of `values`, read back exactly.
    #[doc(hidden)]
    pub fn double_float_sum(&self, values: &[f32]) -> Result<f64, GpuError> {
        let len = u32::try_from(values.len()).map_err(|_| invalid("too many values"))?;
        let padded: &[f32] = if values.is_empty() { &[0.0] } else { values };
        let dv = self.stream.clone_htod(padded)?;
        let mut out = self.stream.alloc_zeros::<f64>(1)?;
        let mut launch = self.stream.launch_builder(&self.kernels.double_float_sum);
        launch.arg(&dv).arg(&len).arg(&mut out);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: arguments match `double_float_sum(const float*, u32, double*)`;
        // the input holds `len` values (at least one element allocated).
        unsafe { launch.launch(cfg) }?;
        let v = self.stream.clone_dtoh(&out)?;
        v.first().copied().ok_or_else(|| invalid("empty sum output"))
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
