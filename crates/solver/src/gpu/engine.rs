//! Device side of a GPU solve: the plan and the regret/strategy storage are
//! uploaded once; each traversal is a fixed sequence of level-by-level
//! kernel launches on one stream (see `cuda/kernels/traverse.cu`).

#[cfg(test)]
mod tests;

use super::plan::{
    showdown_shared_bytes, Plan, PlayerPlan, MODE_AVERAGE, MODE_BEST_RESPONSE, MODE_CFR,
    SHOWDOWN_SHARED_LIMIT,
};
use super::{invalid, Gpu, GpuError};
use crate::cfr::{Discount, Mode};
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg, ValidAsZeroBits};

/// Threads per block, as the kernels assume (`kThreads` in terminal.cu).
const THREADS: u32 = 256;
/// Most blocks along grid y (hardware limit 65535); kernels stride beyond.
const MAX_GRID_Y: u32 = 65_535;
/// Most blocks for the one-block-per-terminal kernels; they stride beyond.
const MAX_TERMINAL_BLOCKS: u32 = 4096;

/// A device buffer and the logical length the kernels see. Allocations are
/// never empty (an empty list gets one unused element).
struct Buffer<T> {
    data: CudaSlice<T>,
    len: u32,
}

/// Pushes a read-only buffer as the kernel's `(pointer, length)` pair.
fn push<'a, T>(l: &mut LaunchArgs<'a>, b: &'a Buffer<T>) {
    l.arg(&b.data).arg(&b.len);
}

/// Pushes a buffer the kernel writes, with the logical length `len`.
fn push_mut<'a, T>(l: &mut LaunchArgs<'a>, data: &'a mut CudaSlice<T>, len: &'a u32) {
    l.arg(data).arg(len);
}

impl Gpu {
    fn upload<T: DeviceRepr + Default>(&self, v: &[T]) -> Result<Buffer<T>, GpuError> {
        let len = u32::try_from(v.len()).map_err(|_| invalid("buffer exceeds u32 indexing"))?;
        let data = if v.is_empty() {
            self.stream.clone_htod(&[T::default()])?
        } else {
            self.stream.clone_htod(v)?
        };
        Ok(Buffer { data, len })
    }

    fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>, GpuError> {
        Ok(self.stream.alloc_zeros::<T>(len.max(1))?)
    }
}

/// One traverser's plan on the device.
struct DevicePlayer {
    slots: Buffer<u32>,
    child_slots: Buffer<u32>,
    opp_nodes: Buffer<u32>,
    own_nodes: Buffer<u32>,
    folds: Buffer<u32>,
    fold_payoff: Buffer<f64>,
    showdowns: Buffer<u32>,
    showdown_table: Buffer<u32>,
    win: Buffer<f64>,
    lose: Buffer<f64>,
    tie: Buffer<f64>,
    my_cards: Buffer<u32>,
    same: Buffer<u32>,
    opp_card_off: Buffer<u32>,
    opp_card_hands: Buffer<u32>,
    opp_weights: Buffer<f32>,
    opp_off: Buffer<u32>,
    opp_len: Buffer<u32>,
    opp_order: Buffer<u32>,
    card_base: Buffer<u32>,
    card_off: Buffer<u32>,
    card_pos: Buffer<u32>,
    ranges: Buffer<u32>,
    ranks: Buffer<u32>,
}

impl DevicePlayer {
    fn upload(gpu: &Gpu, pp: &PlayerPlan) -> Result<DevicePlayer, GpuError> {
        let t = &pp.tables;
        Ok(DevicePlayer {
            slots: gpu.upload(&pp.slot)?,
            child_slots: gpu.upload(&pp.child_slot)?,
            opp_nodes: gpu.upload(&pp.opp_nodes)?,
            own_nodes: gpu.upload(&pp.own_nodes)?,
            folds: gpu.upload(&pp.folds)?,
            fold_payoff: gpu.upload(&pp.fold_payoff)?,
            showdowns: gpu.upload(&pp.showdowns)?,
            showdown_table: gpu.upload(&pp.showdown_table)?,
            win: gpu.upload(&pp.win)?,
            lose: gpu.upload(&pp.lose)?,
            tie: gpu.upload(&pp.tie)?,
            my_cards: gpu.upload(&pp.my_cards)?,
            same: gpu.upload(&pp.same)?,
            opp_card_off: gpu.upload(&pp.opp_card_off)?,
            opp_card_hands: gpu.upload(&pp.opp_card_hands)?,
            opp_weights: gpu.upload(&pp.opp_weights)?,
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
}

/// The kernels' view of a traversal mode.
#[derive(Clone, Copy)]
struct ModeArgs {
    mode: u32,
    positive_discount: f32,
    negative_discount: f32,
    strategy_discount: f32,
    reach_weighted: u32,
}

impl ModeArgs {
    fn new(mode: Mode) -> ModeArgs {
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
fn per_hand_grid(hands: u32, nodes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (hands.div_ceil(THREADS).max(1), nodes.clamp(1, MAX_GRID_Y), 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Grid for one-block-per-terminal kernels.
fn per_terminal_grid(terminals: u32, shared_mem_bytes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (terminals.clamp(1, MAX_TERMINAL_BLOCKS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes,
    }
}

/// A game tree with its regrets and strategy sums on the GPU.
pub(crate) struct Engine {
    gpu: Gpu,
    plan: Plan,
    actions: Buffer<u32>,
    first_child: Buffer<u32>,
    offsets: Buffer<u32>,
    players: [DevicePlayer; 2],
    regrets: CudaSlice<f32>,
    sums: CudaSlice<f32>,
    /// Opponent reach slots (sized for the larger of the two traversals).
    reach: CudaSlice<f32>,
    /// One row of values per node (sized for the larger hand count).
    values: CudaSlice<f32>,
    alive: CudaSlice<u32>,
}

impl Engine {
    /// Uploads a validated plan and the current `f32` storage.
    pub(crate) fn new(gpu: Gpu, plan: Plan, regrets: &[f32], sums: &[f32]) -> Result<Engine, GpuError> {
        plan.validate()?;
        let len = plan.storage_len as usize;
        if regrets.len() != len || sums.len() != len {
            return Err(invalid(format!(
                "storage has {} / {} values, the tree needs {len}",
                regrets.len(),
                sums.len()
            )));
        }
        let reach_len = plan.players.iter().map(|pp| pp.num_slots as usize * pp.opp_hands as usize).max();
        let values_len = plan.players.iter().map(|pp| plan.num_nodes as usize * pp.hands as usize).max();
        let slots_len = plan.players.iter().map(|pp| pp.num_slots as usize).max();
        let engine = Engine {
            actions: gpu.upload(&plan.actions)?,
            first_child: gpu.upload(&plan.first_child)?,
            offsets: gpu.upload(&plan.offset)?,
            players: [
                DevicePlayer::upload(&gpu, &plan.players[0])?,
                DevicePlayer::upload(&gpu, &plan.players[1])?,
            ],
            regrets: gpu.upload(regrets)?.data,
            sums: gpu.upload(sums)?.data,
            reach: gpu.zeros(reach_len.unwrap_or(1))?,
            values: gpu.zeros(values_len.unwrap_or(1))?,
            alive: gpu.zeros(slots_len.unwrap_or(1))?,
            gpu,
            plan,
        };
        engine.gpu.stream.synchronize()?;
        Ok(engine)
    }

    /// Copies the regrets and strategy sums back into host storage.
    pub(crate) fn download(&self, regrets: &mut [f32], sums: &mut [f32]) -> Result<(), GpuError> {
        let len = self.plan.storage_len as usize;
        if regrets.len() != len || sums.len() != len {
            return Err(invalid("host storage doesn't match the tree"));
        }
        let stream = &self.gpu.stream;
        let src = self.regrets.try_slice(..len).ok_or_else(|| invalid("device regrets too short"))?;
        stream.memcpy_dtoh(&src, regrets)?;
        let src = self.sums.try_slice(..len).ok_or_else(|| invalid("device strategy sums too short"))?;
        stream.memcpy_dtoh(&src, sums)?;
        stream.synchronize()?;
        Ok(())
    }

    /// One DCFR iteration: player 0's update, then player 1's (which sees
    /// player 0's new regrets), as on the CPU.
    pub(crate) fn cfr_iteration(&mut self, d: Discount) -> Result<(), GpuError> {
        for p in 0..2 {
            self.traverse(p, Mode::Cfr(d))?;
        }
        self.gpu.stream.synchronize()?;
        Ok(())
    }

    /// Values of player `p` at the root (one per hand) under `mode`.
    pub(crate) fn root_hand_values(&mut self, p: usize, mode: Mode) -> Result<Vec<f32>, GpuError> {
        self.traverse(p, mode)?;
        let n = self.plan.players.get(p).ok_or_else(|| invalid("player out of range"))?.hands as usize;
        let row = self.values.try_slice(..n).ok_or_else(|| invalid("value buffer too short"))?;
        let v = self.gpu.stream.clone_dtoh(&row)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }

    /// Expected payoff of each player at the root (CPU: `cfr::root_values`).
    pub(crate) fn root_values(&mut self, mode: Mode) -> Result<[f64; 2], GpuError> {
        let mut res = [0f64; 2];
        for (p, r) in res.iter_mut().enumerate() {
            let v = self.root_hand_values(p, mode)?;
            let w = &self.plan.players.get(p ^ 1).ok_or_else(|| invalid("player out of range"))?.opp_weights;
            *r = v.iter().zip(w).map(|(&x, &w)| f64::from(x) * f64::from(w)).sum();
        }
        Ok(res)
    }

    /// Exploitability in chips (CPU: `cfr::exploitability`).
    pub(crate) fn exploitability(&mut self, raked: bool) -> Result<f64, GpuError> {
        let br = self.root_values(Mode::BestResponse)?;
        Ok(if raked {
            let ev = self.root_values(Mode::Average)?;
            ((br[0] - ev[0]) + (br[1] - ev[1])) / 2.0
        } else {
            (br[0] + br[1]) / 2.0
        })
    }

    /// Enqueues one traversal by player `p`; its root values end up in row 0
    /// of the value buffer.
    fn traverse(&mut self, p: usize, mode: Mode) -> Result<(), GpuError> {
        let Engine { gpu, plan, actions, first_child, offsets, players, regrets, sums, reach, values, alive } =
            self;
        let (Some(pp), Some(dp)) = (plan.players.get(p), players.get(p)) else {
            return Err(invalid("player out of range"));
        };
        let m = ModeArgs::new(mode);
        let (hands, opp_hands) = (pp.hands, pp.opp_hands);
        // `Plan::validate` checked these fit in u32; checked again for free.
        let reach_len =
            pp.num_slots.checked_mul(opp_hands).ok_or_else(|| invalid("reach buffer overflow"))?;
        let values_len = plan.num_nodes.checked_mul(hands).ok_or_else(|| invalid("value buffer overflow"))?;
        let alive_len = pp.num_slots;
        let kernels = &gpu.kernels;
        let stream = &gpu.stream;

        // 1. Root reach and live flags.
        {
            let mut l = stream.launch_builder(&kernels.begin_traversal);
            push_mut(&mut l, alive, &alive_len);
            push_mut(&mut l, reach, &reach_len);
            push(&mut l, &dp.opp_weights);
            let cfg = LaunchConfig {
                grid_dim: (alive_len.max(opp_hands).div_ceil(THREADS).max(1), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: arguments match `begin_traversal`; every buffer holds at
            // least its logical length (allocated for the larger traversal).
            unsafe { l.launch(cfg) }?;
        }

        // 2. Top-down: the opponent's strategy splits its reach.
        for lv in pp.levels.iter().filter(|lv| lv.opp_count > 0) {
            let mut l = stream.launch_builder(&kernels.opponent_split);
            push(&mut l, &dp.opp_nodes);
            l.arg(&lv.opp_start).arg(&lv.opp_count);
            push(&mut l, actions);
            push(&mut l, offsets);
            push(&mut l, &dp.slots);
            push(&mut l, &dp.child_slots);
            l.arg(&*regrets).arg(&plan.storage_len);
            push_mut(&mut l, sums, &plan.storage_len);
            push_mut(&mut l, reach, &reach_len);
            push_mut(&mut l, alive, &alive_len);
            l.arg(&opp_hands).arg(&m.mode).arg(&m.strategy_discount).arg(&m.reach_weighted);
            // SAFETY: arguments match `opponent_split`; every index the
            // kernel reads comes from the validated plan.
            unsafe { l.launch(per_hand_grid(opp_hands, lv.opp_count)) }?;
        }

        // 3. Terminals.
        if dp.folds.len > 0 {
            let mut l = stream.launch_builder(&kernels.fold_values);
            push(&mut l, &dp.folds);
            push(&mut l, &dp.fold_payoff);
            push(&mut l, &dp.slots);
            l.arg(&*alive).arg(&alive_len);
            l.arg(&*reach).arg(&reach_len);
            push_mut(&mut l, values, &values_len);
            push(&mut l, &dp.my_cards);
            push(&mut l, &dp.same);
            push(&mut l, &dp.opp_card_off);
            push(&mut l, &dp.opp_card_hands);
            l.arg(&hands).arg(&opp_hands);
            // SAFETY: arguments match `fold_values`; indices are validated.
            unsafe { l.launch(per_terminal_grid(dp.folds.len, 0)) }?;
        }
        if dp.showdowns.len > 0 {
            let shared = showdown_shared_bytes(pp.tables.max_opp);
            if shared > u64::from(SHOWDOWN_SHARED_LIMIT) {
                return Err(invalid("showdown lists too long for shared memory"));
            }
            let shared = shared as u32;
            let mut l = stream.launch_builder(&kernels.showdown_values);
            push(&mut l, &dp.showdowns);
            push(&mut l, &dp.showdown_table);
            push(&mut l, &dp.win);
            push(&mut l, &dp.lose);
            push(&mut l, &dp.tie);
            push(&mut l, &dp.slots);
            l.arg(&*alive).arg(&alive_len);
            l.arg(&*reach).arg(&reach_len);
            push_mut(&mut l, values, &values_len);
            push(&mut l, &dp.my_cards);
            push(&mut l, &dp.same);
            push(&mut l, &dp.opp_off);
            push(&mut l, &dp.opp_len);
            push(&mut l, &dp.opp_order);
            push(&mut l, &dp.card_base);
            push(&mut l, &dp.card_off);
            push(&mut l, &dp.card_pos);
            push(&mut l, &dp.ranges);
            push(&mut l, &dp.ranks);
            l.arg(&hands).arg(&opp_hands).arg(&pp.tables.max_opp);
            // SAFETY: arguments match `showdown_values`; indices are
            // validated and the dynamic shared memory is sized for `max_opp`.
            unsafe { l.launch(per_terminal_grid(dp.showdowns.len, shared)) }?;
        }

        // 4. Bottom-up: values and the traverser's regret updates.
        for lv in pp.levels.iter().rev() {
            if lv.own_count > 0 {
                let mut l = stream.launch_builder(&kernels.own_update);
                push(&mut l, &dp.own_nodes);
                l.arg(&lv.own_start).arg(&lv.own_count);
                push(&mut l, actions);
                push(&mut l, offsets);
                push(&mut l, first_child);
                push(&mut l, &dp.slots);
                l.arg(&*alive).arg(&alive_len);
                push_mut(&mut l, regrets, &plan.storage_len);
                l.arg(&*sums).arg(&plan.storage_len);
                push_mut(&mut l, values, &values_len);
                l.arg(&hands).arg(&m.mode).arg(&m.positive_discount).arg(&m.negative_discount);
                // SAFETY: arguments match `own_update`; indices are validated.
                unsafe { l.launch(per_hand_grid(hands, lv.own_count)) }?;
            }
            if lv.opp_count > 0 {
                let mut l = stream.launch_builder(&kernels.opponent_sum);
                push(&mut l, &dp.opp_nodes);
                l.arg(&lv.opp_start).arg(&lv.opp_count);
                push(&mut l, actions);
                push(&mut l, first_child);
                push(&mut l, &dp.slots);
                l.arg(&*alive).arg(&alive_len);
                push_mut(&mut l, values, &values_len);
                l.arg(&hands);
                // SAFETY: arguments match `opponent_sum`; indices are validated.
                unsafe { l.launch(per_hand_grid(hands, lv.opp_count)) }?;
            }
        }
        Ok(())
    }
}
