//! Device plumbing shared by the heads-up and three-player engines:
//! buffers with their kernel-visible lengths, argument pushing, launch
//! grids, traversal modes, the regret/strategy storage, and memory checks.

use super::shared::{MODE_AVERAGE, MODE_BEST_RESPONSE, MODE_CFR};
use super::tables::Tables;
use super::{invalid, Gpu, GpuError};
use crate::cfr::Mode;
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg, ValidAsZeroBits};

/// The solver's storage as the engine uploads and downloads it.
pub(crate) enum HostStorage<'a> {
    F32 { regrets: &'a mut [f32], sums: &'a mut [f32] },
    /// 16-bit values with one f32 scale per decision node.
    Compressed { regrets: &'a mut [i16], sums: &'a mut [u16], regret_scale: &'a mut [f32], strategy_scale: &'a mut [f32] },
}

/// The storage on the device.
pub(crate) enum DeviceStorage {
    F32 { regrets: CudaSlice<f32>, sums: CudaSlice<f32> },
    Compressed { regrets: CudaSlice<i16>, sums: CudaSlice<u16>, regret_scale: CudaSlice<f32>, strategy_scale: CudaSlice<f32> },
}

impl HostStorage<'_> {
    /// Device bytes of this storage.
    pub(crate) fn bytes(&self) -> u64 {
        match self {
            HostStorage::F32 { regrets, sums } => 4 * (regrets.len() + sums.len()) as u64,
            HostStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => {
                2 * (regrets.len() + sums.len()) as u64 + 4 * (regret_scale.len() + strategy_scale.len()) as u64
            }
        }
    }

    /// Checks the lengths against a tree's `storage_len` values and
    /// `scales` decision nodes; returns the device bytes needed.
    pub(crate) fn check(&self, storage_len: u32, scales: u32) -> Result<u64, GpuError> {
        let len = storage_len as usize;
        let scales = scales as usize;
        let ok = match self {
            HostStorage::F32 { regrets, sums } => regrets.len() == len && sums.len() == len,
            HostStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => {
                regrets.len() == len && sums.len() == len && regret_scale.len() == scales && strategy_scale.len() == scales
            }
        };
        if !ok {
            return Err(invalid("storage doesn't match the tree"));
        }
        Ok(self.bytes())
    }
}


impl DeviceStorage {
    /// Uploads `host`'s storage.
    pub(crate) fn upload(gpu: &Gpu, host: HostStorage<'_>) -> Result<DeviceStorage, GpuError> {
        Ok(match host {
            HostStorage::F32 { regrets, sums } => {
                DeviceStorage::F32 { regrets: gpu.upload(regrets)?.data, sums: gpu.upload(sums)?.data }
            }
            HostStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => DeviceStorage::Compressed {
                regrets: gpu.upload(regrets)?.data,
                sums: gpu.upload(sums)?.data,
                regret_scale: gpu.upload(regret_scale)?.data,
                strategy_scale: gpu.upload(strategy_scale)?.data,
            },
        })
    }

    /// Copies the regrets and strategy sums (and scales) back into host
    /// storage of the same kind (lengths checked by the caller).
    pub(crate) fn download(&self, gpu: &Gpu, host: HostStorage<'_>) -> Result<(), GpuError> {
        let stream = &gpu.stream;
        /// `dst.len()` elements of `src` into `dst` (slicing to exactly
        /// `dst.len()` keeps cudarc's length assertion from ever firing).
        fn copy<T: DeviceRepr>(
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            src: &CudaSlice<T>,
            dst: &mut [T],
        ) -> Result<(), GpuError> {
            if dst.is_empty() {
                return Ok(());
            }
            let view = src.try_slice(..dst.len()).ok_or_else(|| invalid("device storage too short"))?;
            Ok(stream.memcpy_dtoh(&view, dst)?)
        }
        match (self, host) {
            (DeviceStorage::F32 { regrets, sums }, HostStorage::F32 { regrets: r, sums: s }) => {
                copy(stream, regrets, r)?;
                copy(stream, sums, s)?;
            }
            (
                DeviceStorage::Compressed { regrets, sums, regret_scale, strategy_scale },
                HostStorage::Compressed { regrets: r, sums: s, regret_scale: rs, strategy_scale: ss },
            ) => {
                copy(stream, regrets, r)?;
                copy(stream, sums, s)?;
                copy(stream, regret_scale, rs)?;
                copy(stream, strategy_scale, ss)?;
            }
            _ => return Err(invalid("host and device storage kinds differ")),
        }
        stream.synchronize()?;
        Ok(())
    }
}

/// Fails with `OutOfMemory` unless `needed` bytes fit in the device's free
/// memory minus [`MEMORY_MARGIN`].
pub(crate) fn check_memory(gpu: &Gpu, needed: u64) -> Result<(), GpuError> {
    let (free, _total) = gpu.stream.context().mem_get_info()?;
    let available = (free as u64).saturating_sub(MEMORY_MARGIN);
    if needed > available {
        return Err(GpuError::OutOfMemory { needed, available });
    }
    Ok(())
}

/// Launches `begin_traversal`: slot 0 of `reach` gets `weights` and is the
/// only live slot of `alive`.
pub(crate) fn begin_traversal(
    gpu: &Gpu,
    alive: &mut CudaSlice<u32>,
    alive_len: u32,
    reach: &mut CudaSlice<f32>,
    reach_len: u32,
    weights: &Buffer<f32>,
) -> Result<(), GpuError> {
    let mut l = gpu.stream.launch_builder(&gpu.kernels.begin_traversal);
    push_mut(&mut l, alive, &alive_len);
    push_mut(&mut l, reach, &reach_len);
    push(&mut l, weights);
    let cfg = LaunchConfig {
        grid_dim: (alive_len.max(weights.len).div_ceil(THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: arguments match `begin_traversal`; the callers' buffers hold at
    // least their logical lengths, and `weights` fits in slot 0 of `reach`
    // (the plans validate the slot layout).
    unsafe { l.launch(cfg) }?;
    Ok(())
}

/// Threads per block, as the kernels assume (`kThreads` in terminal.cu;
/// the traversal kernels take any size).
pub(crate) const THREADS: u32 = 256;
/// Most blocks along grid y (hardware limit 65535); kernels stride beyond.
pub(crate) const MAX_GRID_Y: u32 = 65_535;
/// Most blocks for the one-block-per-terminal kernels; they stride beyond.
pub(crate) const MAX_TERMINAL_BLOCKS: u32 = 4096;

/// A device buffer and the logical length the kernels see. Allocations are
/// never empty (an empty list gets one unused element).
pub(crate) struct Buffer<T> {
    pub data: CudaSlice<T>,
    pub len: u32,
}

/// Pushes a read-only buffer as the kernel's `(pointer, length)` pair.
pub(crate) fn push<'a, T>(l: &mut LaunchArgs<'a>, b: &'a Buffer<T>) {
    l.arg(&b.data).arg(&b.len);
}

/// Pushes a buffer the kernel writes, with the logical length `len`.
pub(crate) fn push_mut<'a, T>(l: &mut LaunchArgs<'a>, data: &'a mut CudaSlice<T>, len: &'a u32) {
    l.arg(data).arg(len);
}

impl Gpu {
    pub(crate) fn upload<T: DeviceRepr + Default>(&self, v: &[T]) -> Result<Buffer<T>, GpuError> {
        let len = u32::try_from(v.len()).map_err(|_| invalid("buffer exceeds u32 indexing"))?;
        let data = if v.is_empty() {
            self.stream.clone_htod(&[T::default()])?
        } else {
            self.stream.clone_htod(v)?
        };
        Ok(Buffer { data, len })
    }

    pub(crate) fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.alloc_zeros::<T>(len.max(1))?)
    }
}

/// The kernels' view of a traversal mode.
#[derive(Clone, Copy)]
pub(crate) struct ModeArgs {
    pub mode: u32,
    pub positive_discount: f32,
    pub negative_discount: f32,
    pub strategy_discount: f32,
    pub reach_weighted: u32,
}

impl ModeArgs {
    pub(crate) fn new(mode: Mode) -> ModeArgs {
        match mode {
            Mode::Cfr(d) => ModeArgs {
                mode: MODE_CFR,
                positive_discount: d.pos,
                negative_discount: d.neg,
                strategy_discount: d.strat,
                reach_weighted: u32::from(d.reach_weighted),
            },
            Mode::BestResponse => ModeArgs::plain(MODE_BEST_RESPONSE),
            Mode::Average => ModeArgs::plain(MODE_AVERAGE),
        }
    }

    fn plain(mode: u32) -> ModeArgs {
        ModeArgs {
            mode,
            positive_discount: 0.0,
            negative_discount: 0.0,
            strategy_discount: 0.0,
            reach_weighted: 0,
        }
    }
}

/// Grid for kernels with hands along x and a level's nodes along y.
pub(crate) fn per_hand_grid(hands: u32, nodes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (hands.div_ceil(THREADS).max(1), nodes.clamp(1, MAX_GRID_Y), 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Grid for one-block-per-node kernels (16-bit storage updates).
pub(crate) fn per_node_grid(nodes: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (nodes.clamp(1, 1 << 20), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

/// Grid for one-block-per-terminal kernels.
pub(crate) fn per_terminal_grid(terminals: u32, threads: u32, shared_mem_bytes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (terminals.clamp(1, MAX_TERMINAL_BLOCKS), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    }
}

/// Device memory left free for the display and the CUDA runtime.
pub(crate) const MEMORY_MARGIN: u64 = 512 << 20;


/// A traverser's showdown lookup against one opponent on the device
/// (`Tables`), pushed as the kernels' run of eight `(pointer, length)` pairs.
pub(crate) struct DeviceTables {
    opp_off: Buffer<u32>,
    opp_len: Buffer<u32>,
    opp_order: Buffer<u32>,
    card_base: Buffer<u32>,
    card_off: Buffer<u32>,
    card_pos: Buffer<u32>,
    ranges: Buffer<u32>,
    ranks: Buffer<u32>,
}

impl DeviceTables {
    pub(crate) fn upload(gpu: &Gpu, t: &Tables) -> Result<DeviceTables, GpuError> {
        Ok(DeviceTables {
            opp_off: gpu.upload(&t.opp_off)?,
            opp_len: gpu.upload(&t.opp_len)?,
            opp_order: gpu.upload(&t.opp_order)?,
            card_base: gpu.upload(&t.card_base)?,
            card_off: gpu.upload(&t.card_off)?,
            card_pos: gpu.upload(&t.card_pos)?,
            ranges: gpu.upload(&t.ranges)?,
            ranks: gpu.upload(&t.ranks)?,
        })
    }

    /// Pushes the tables in the kernels' order (`ShowdownTables`).
    pub(crate) fn push<'a>(&'a self, l: &mut LaunchArgs<'a>) {
        for b in [
            &self.opp_off,
            &self.opp_len,
            &self.opp_order,
            &self.card_base,
            &self.card_off,
            &self.card_pos,
            &self.ranges,
            &self.ranks,
        ] {
            push(l, b);
        }
    }
}
