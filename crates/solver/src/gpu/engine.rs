//! Device side of a GPU solve: the plan and the regret/strategy storage are
//! uploaded once; each traversal is a fixed sequence of level-by-level
//! kernel launches on one stream (see `cuda/kernels/traverse.cu`).

#[cfg(test)]
mod tests;

pub(crate) use super::device::HostStorage;
use super::device::{
    begin_traversal, check_memory, per_hand_grid, per_node_grid, per_terminal_grid, push, push_mut, Buffer, DeviceStorage, DeviceTables, ModeArgs,
    MAX_TERMINAL_BLOCKS, MEMORY_MARGIN, THREADS,
};
use super::plan::{scratch_bytes, showdown_shared_bytes, LevelRange, Plan, PlayerPlan, Segment, SHOWDOWN_SHARED_LIMIT};
use super::{invalid, Gpu, GpuError, GpuPrecision};
use crate::cfr::{Discount, Mode};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

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
    showdown_fold: Buffer<u32>,
    showdown_fold_payoff: Buffer<f64>,
    my_cards: Buffer<u32>,
    same: Buffer<u32>,
    opp_card_off: Buffer<u32>,
    opp_card_hands: Buffer<u32>,
    chance_nodes: Buffer<u32>,
    fused: Buffer<u32>,
    opp_cards: Buffer<u32>,
    perm: Buffer<u32>,
    opp_weights: Buffer<f32>,
    tables: DeviceTables,
}

impl DevicePlayer {
    fn upload(gpu: &Gpu, pp: &PlayerPlan) -> Result<DevicePlayer, GpuError> {
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
            showdown_fold: gpu.upload(&pp.showdown_fold)?,
            showdown_fold_payoff: gpu.upload(&pp.showdown_fold_payoff)?,
            my_cards: gpu.upload(&pp.my_cards)?,
            same: gpu.upload(&pp.same)?,
            opp_card_off: gpu.upload(&pp.opp_card_off)?,
            opp_card_hands: gpu.upload(&pp.opp_card_hands)?,
            chance_nodes: gpu.upload(&pp.chance_nodes)?,
            fused: gpu.upload(&pp.fused)?,
            opp_cards: gpu.upload(&pp.opp_cards)?,
            perm: gpu.upload(&pp.perm)?,
            opp_weights: gpu.upload(&pp.opp_weights)?,
            tables: DeviceTables::upload(gpu, &pp.tables)?,
        })
    }
}

/// The per-node arrays of the plan on the device.
struct DeviceTree {
    actions: Buffer<u32>,
    first_child: Buffer<u32>,
    offsets: Buffer<u32>,
    cards: Buffer<u32>,
    chance_factor: Buffer<f32>,
    iso_start: Buffer<u32>,
    iso_count: Buffer<u32>,
    iso_child: Buffer<u32>,
    iso_pair: Buffer<u32>,
    scalar: Buffer<u32>,
    /// Per node: its value row.
    rows: Buffer<u32>,
}

impl DeviceTree {
    fn upload(gpu: &Gpu, plan: &Plan) -> Result<DeviceTree, GpuError> {
        Ok(DeviceTree {
            actions: gpu.upload(&plan.actions)?,
            first_child: gpu.upload(&plan.first_child)?,
            offsets: gpu.upload(&plan.offset)?,
            cards: gpu.upload(&plan.card)?,
            chance_factor: gpu.upload(&plan.chance_factor)?,
            iso_start: gpu.upload(&plan.iso_start)?,
            iso_count: gpu.upload(&plan.iso_count)?,
            iso_child: gpu.upload(&plan.iso_child)?,
            iso_pair: gpu.upload(&plan.iso_pair)?,
            scalar: gpu.upload(&plan.scalar)?,
            rows: gpu.upload(&plan.row)?,
        })
    }
}

/// A game tree with its regrets and strategy sums on the GPU.
pub(crate) struct Engine {
    gpu: Gpu,
    plan: Plan,
    tree: DeviceTree,
    players: [DevicePlayer; 2],
    storage: DeviceStorage,
    /// Per node: scale slot in 16-bit storage.
    scale_slots: Buffer<u32>,
    /// Opponent reach slots (sized for the larger of the two traversals).
    reach: CudaSlice<f32>,
    /// Value rows (`Plan::num_rows`, sized for the larger hand count).
    values: CudaSlice<f32>,
    alive: CudaSlice<u32>,
    /// 1 for f64 terminal sums, 0 for double-single (picks the terminal kernels).
    exact: u32,
}

impl Engine {
    /// How terminal sums are accumulated.
    pub(crate) fn precision(&self) -> GpuPrecision {
        if self.exact != 0 {
            GpuPrecision::Exact
        } else {
            GpuPrecision::Fast
        }
    }

    /// Device bytes a plan and its scratch buffers may take next to `host`'s
    /// storage: the free memory minus the storage and a margin for the
    /// display and the CUDA runtime (`Scratch::Budget` for `Plan::build_with`).
    pub(crate) fn scratch_budget(gpu: &Gpu, host: &HostStorage<'_>) -> Result<u64, GpuError> {
        let (free, _total) = gpu.stream.context().mem_get_info()?;
        Ok((free as u64).saturating_sub(MEMORY_MARGIN).saturating_sub(host.bytes()))
    }

    /// Uploads a validated plan and the current storage.
    pub(crate) fn new(
        gpu: Gpu,
        plan: Plan,
        host: HostStorage<'_>,
        precision: GpuPrecision,
    ) -> Result<Engine, GpuError> {
        plan.validate()?;
        let storage_bytes = host.check(plan.storage_len, plan.num_scale_slots)?;
        let [p0, p1] = &plan.players;
        let reach_len =
            (p0.num_slots as usize * p0.opp_hands as usize).max(p1.num_slots as usize * p1.opp_hands as usize);
        let values_len = plan.num_rows as usize * p0.hands.max(p1.hands) as usize;
        let slots_len = p0.num_slots.max(p1.num_slots) as usize;
        let scratch = scratch_bytes(
            [u64::from(p0.num_slots), u64::from(p1.num_slots)],
            u64::from(plan.num_rows),
            [u64::from(p0.hands), u64::from(p1.hands)],
        );
        check_memory(&gpu, plan.device_bytes() + storage_bytes + scratch)?;
        let engine = Engine {
            tree: DeviceTree::upload(&gpu, &plan)?,
            players: [
                DevicePlayer::upload(&gpu, &plan.players[0])?,
                DevicePlayer::upload(&gpu, &plan.players[1])?,
            ],
            storage: DeviceStorage::upload(&gpu, host)?,
            scale_slots: gpu.upload(&plan.scale_slot)?,
            reach: gpu.zeros(reach_len)?,
            values: gpu.zeros(values_len)?,
            alive: gpu.zeros(slots_len)?,
            exact: u32::from(precision == GpuPrecision::Exact),
            gpu,
            plan,
        };
        engine.gpu.stream.synchronize()?;
        Ok(engine)
    }

    /// Copies the regrets and strategy sums (and scales) back into host
    /// storage of the same kind.
    pub(crate) fn download(&self, host: HostStorage<'_>) -> Result<(), GpuError> {
        host.check(self.plan.storage_len, self.plan.num_scale_slots)?;
        self.storage.download(&self.gpu, host)
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
        let start = *self.plan.row.first().ok_or_else(|| invalid("empty plan"))? as usize * n;
        let row = self.values.try_slice(start..start + n).ok_or_else(|| invalid("value buffer too short"))?;
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

    /// Enqueues one traversal by player `p`; its root values end up in the
    /// root's value row. The segments run as `plan::schedule` lays out: the
    /// top segment top-down and its terminals, then each group in full (its
    /// reach slots' alive flags cleared first), then the top bottom-up.
    fn traverse(&mut self, p: usize, mode: Mode) -> Result<(), GpuError> {
        let Engine { gpu, plan, tree, players, storage, scale_slots, reach, values, alive, exact } = self;
        let plan: &Plan = plan;
        let (Some(pp), Some(dp)) = (plan.players.get(p), players.get(p)) else {
            return Err(invalid("player out of range"));
        };
        // `Plan::validate` checked these fit in u32; checked again for free.
        let reach_len = pp.num_slots.checked_mul(pp.opp_hands).ok_or_else(|| invalid("reach buffer overflow"))?;
        let values_len = plan.num_rows.checked_mul(pp.hands).ok_or_else(|| invalid("value buffer overflow"))?;
        let mut pass = Pass {
            gpu,
            plan,
            tree,
            pp,
            dp,
            storage,
            scale_slots,
            reach,
            values,
            alive,
            exact: *exact,
            m: ModeArgs::new(mode),
            reach_len,
            values_len,
            alive_len: pp.num_slots,
        };
        let levels = |seg: &Segment| pp.levels.get(seg.levels.range()).ok_or_else(|| invalid("segment levels"));
        let (top, groups) = pp.segments.split_first().ok_or_else(|| invalid("plan has no segments"))?;
        pass.begin()?;
        pass.top_down(levels(top)?)?;
        pass.terminals(top)?;
        for group in groups {
            pass.reset_alive(group)?;
            pass.top_down(levels(group)?)?;
            pass.terminals(group)?;
            pass.bottom_up(levels(group)?)?;
        }
        pass.bottom_up(levels(top)?)
    }
}

/// One traversal's launches: the engine's buffers plus the traverser's plan.
struct Pass<'a> {
    gpu: &'a Gpu,
    plan: &'a Plan,
    tree: &'a DeviceTree,
    pp: &'a PlayerPlan,
    dp: &'a DevicePlayer,
    storage: &'a mut DeviceStorage,
    scale_slots: &'a Buffer<u32>,
    reach: &'a mut CudaSlice<f32>,
    values: &'a mut CudaSlice<f32>,
    alive: &'a mut CudaSlice<u32>,
    exact: u32,
    m: ModeArgs,
    reach_len: u32,
    values_len: u32,
    alive_len: u32,
}

impl Pass<'_> {
    /// The kernel of a `[exact, fast]` pair for this pass's precision.
    fn by_precision<'k>(&self, pair: &'k [cudarc::driver::CudaFunction; 2]) -> Result<&'k cudarc::driver::CudaFunction, GpuError> {
        pair.get(usize::from(self.exact == 0)).ok_or_else(|| invalid("kernel pair"))
    }

    /// Root reach and alive flags (slot 0 alive, every other slot dead).
    fn begin(&mut self) -> Result<(), GpuError> {
        let (reach_len, alive_len) = (self.reach_len, self.alive_len);
        begin_traversal(self.gpu, self.alive, alive_len, self.reach, reach_len, &self.dp.opp_weights)
    }

    /// Clears the alive flags of a group's reach slots: the previous group
    /// left its own there.
    fn reset_alive(&mut self, group: &Segment) -> Result<(), GpuError> {
        let span = &group.fresh_slots;
        if span.count == 0 {
            return Ok(());
        }
        let mut l = self.gpu.stream.launch_builder(&self.gpu.kernels.reset_alive);
        push_mut(&mut l, self.alive, &self.alive_len);
        l.arg(&span.start).arg(&span.count);
        let cfg = LaunchConfig {
            grid_dim: (span.count.div_ceil(THREADS).clamp(1, MAX_TERMINAL_BLOCKS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: arguments match `reset_alive`; the span is validated to lie
        // inside the slots.
        unsafe { l.launch(cfg) }?;
        Ok(())
    }

    /// Top-down: the opponent's strategy, and the dealt cards, split the
    /// opponent's reach.
    fn top_down(&mut self, levels: &[LevelRange]) -> Result<(), GpuError> {
        let (gpu, plan, tree, dp, m) = (self.gpu, self.plan, self.tree, self.dp, self.m);
        let (kernels, stream, opp_hands) = (&gpu.kernels, &gpu.stream, self.pp.opp_hands);
        for lv in levels {
            if lv.opp_count > 0 {
                match &mut *self.storage {
                    DeviceStorage::F32 { regrets, sums } => {
                        let mut l = stream.launch_builder(&kernels.opponent_split);
                        push(&mut l, &dp.opp_nodes);
                        l.arg(&lv.opp_start).arg(&lv.opp_count);
                        push(&mut l, &tree.actions);
                        push(&mut l, &tree.offsets);
                        push(&mut l, &dp.slots);
                        push(&mut l, &dp.child_slots);
                        l.arg(&*regrets).arg(&plan.storage_len);
                        push_mut(&mut l, sums, &plan.storage_len);
                        push_mut(&mut l, self.reach, &self.reach_len);
                        push_mut(&mut l, self.alive, &self.alive_len);
                        l.arg(&opp_hands).arg(&m.mode).arg(&m.strategy_discount).arg(&m.reach_weighted);
                        // SAFETY: arguments match `opponent_split`; every index the
                        // kernel reads comes from the validated plan.
                        unsafe { l.launch(per_hand_grid(opp_hands, lv.opp_count)) }?;
                    }
                    DeviceStorage::Compressed { regrets, sums, strategy_scale, .. } => {
                        let mut l = stream.launch_builder(&kernels.opponent_split_compressed);
                        push(&mut l, &dp.opp_nodes);
                        l.arg(&lv.opp_start).arg(&lv.opp_count);
                        push(&mut l, &tree.actions);
                        push(&mut l, &tree.offsets);
                        push(&mut l, self.scale_slots);
                        push(&mut l, &dp.slots);
                        push(&mut l, &dp.child_slots);
                        l.arg(&*regrets).arg(&plan.storage_len);
                        push_mut(&mut l, sums, &plan.storage_len);
                        push_mut(&mut l, strategy_scale, &plan.num_scale_slots);
                        push_mut(&mut l, self.reach, &self.reach_len);
                        push_mut(&mut l, self.alive, &self.alive_len);
                        l.arg(&opp_hands).arg(&m.mode).arg(&m.strategy_discount).arg(&m.reach_weighted);
                        // SAFETY: arguments match `opponent_split_compressed`;
                        // indices (scale slots included) are validated.
                        unsafe { l.launch(per_node_grid(lv.opp_count)) }?;
                    }
                }
            }
            if lv.chance_count > 0 {
                let mut l = stream.launch_builder(&kernels.chance_split);
                push(&mut l, &dp.chance_nodes);
                l.arg(&lv.chance_start).arg(&lv.chance_count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.first_child);
                push(&mut l, &tree.cards);
                push(&mut l, &tree.chance_factor);
                push(&mut l, &dp.slots);
                push(&mut l, &dp.child_slots);
                push(&mut l, &dp.opp_cards);
                push_mut(&mut l, self.reach, &self.reach_len);
                push_mut(&mut l, self.alive, &self.alive_len);
                l.arg(&opp_hands);
                // SAFETY: arguments match `chance_split`; indices are validated.
                unsafe { l.launch(per_hand_grid(opp_hands, lv.chance_count)) }?;
            }
        }
        Ok(())
    }

    /// The segment's fold and showdown terminals.
    fn terminals(&mut self, seg: &Segment) -> Result<(), GpuError> {
        let (gpu, tree, pp, dp) = (self.gpu, self.tree, self.pp, self.dp);
        let (kernels, stream) = (&gpu.kernels, &gpu.stream);
        let (hands, opp_hands) = (pp.hands, pp.opp_hands);
        if seg.folds.count > 0 {
            let fold_shared = opp_hands.checked_mul(4).ok_or_else(|| invalid("fold shared memory"))?;
            let mut l = stream.launch_builder(self.by_precision(&kernels.fold_values)?);
            push(&mut l, &dp.folds);
            push(&mut l, &dp.fold_payoff);
            push(&mut l, &dp.slots);
            push(&mut l, &tree.scalar);
            l.arg(&*self.alive).arg(&self.alive_len);
            l.arg(&*self.reach).arg(&self.reach_len);
            push(&mut l, &tree.rows);
            push_mut(&mut l, self.values, &self.values_len);
            push(&mut l, &dp.my_cards);
            push(&mut l, &dp.same);
            push(&mut l, &dp.opp_card_off);
            push(&mut l, &dp.opp_card_hands);
            l.arg(&seg.folds.start).arg(&seg.folds.count);
            l.arg(&hands).arg(&opp_hands);
            // SAFETY: arguments match `fold_values_{exact,fast}`; indices (the segment's
            // range included) are validated, and the dynamic shared memory
            // holds `opp_hands` floats.
            unsafe { l.launch(per_terminal_grid(seg.folds.count, THREADS, fold_shared)) }?;
        }
        if seg.showdowns.count > 0 {
            let shared = showdown_shared_bytes(pp.tables.max_opp);
            if shared > u64::from(SHOWDOWN_SHARED_LIMIT) {
                return Err(invalid("showdown lists too long for shared memory"));
            }
            let shared = shared as u32;
            let mut l = stream.launch_builder(self.by_precision(&kernels.showdown_values)?);
            push(&mut l, &dp.showdowns);
            push(&mut l, &dp.showdown_table);
            push(&mut l, &dp.win);
            push(&mut l, &dp.lose);
            push(&mut l, &dp.tie);
            push(&mut l, &dp.showdown_fold);
            push(&mut l, &dp.showdown_fold_payoff);
            push(&mut l, &dp.slots);
            push(&mut l, &tree.scalar);
            l.arg(&*self.alive).arg(&self.alive_len);
            l.arg(&*self.reach).arg(&self.reach_len);
            push(&mut l, &tree.rows);
            push_mut(&mut l, self.values, &self.values_len);
            push(&mut l, &dp.my_cards);
            push(&mut l, &dp.same);
            dp.tables.push(&mut l);
            l.arg(&seg.showdowns.start).arg(&seg.showdowns.count);
            l.arg(&hands).arg(&opp_hands).arg(&pp.tables.max_opp);
            // SAFETY: arguments match `showdown_values_{exact,fast}`; indices (the
            // segment's range included) are validated and the dynamic shared
            // memory is sized for `max_opp`.
            unsafe { l.launch(per_terminal_grid(seg.showdowns.count, THREADS, shared)) }?;
        }
        Ok(())
    }

    /// Bottom-up: values and the traverser's regret updates.
    fn bottom_up(&mut self, levels: &[LevelRange]) -> Result<(), GpuError> {
        let (gpu, plan, tree, dp, m) = (self.gpu, self.plan, self.tree, self.dp, self.m);
        let (kernels, stream, hands) = (&gpu.kernels, &gpu.stream, self.pp.hands);
        for lv in levels.iter().rev() {
            if lv.own_count > 0 {
                match &mut *self.storage {
                    DeviceStorage::F32 { regrets, sums } => {
                        let mut l = stream.launch_builder(&kernels.own_update);
                        push(&mut l, &dp.own_nodes);
                        l.arg(&lv.own_start).arg(&lv.own_count);
                        push(&mut l, &tree.actions);
                        push(&mut l, &tree.offsets);
                        push(&mut l, &tree.first_child);
                        push(&mut l, &dp.slots);
                        l.arg(&*self.alive).arg(&self.alive_len);
                        push_mut(&mut l, regrets, &plan.storage_len);
                        l.arg(&*sums).arg(&plan.storage_len);
                        push(&mut l, &dp.fused);
                        push(&mut l, &tree.rows);
                        push_mut(&mut l, self.values, &self.values_len);
                        l.arg(&hands).arg(&m.mode).arg(&m.positive_discount).arg(&m.negative_discount);
                        // SAFETY: arguments match `own_update`; indices are validated.
                        unsafe { l.launch(per_hand_grid(hands, lv.own_count)) }?;
                    }
                    DeviceStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => {
                        let mut l = stream.launch_builder(&kernels.own_update_compressed);
                        push(&mut l, &dp.own_nodes);
                        l.arg(&lv.own_start).arg(&lv.own_count);
                        push(&mut l, &tree.actions);
                        push(&mut l, &tree.offsets);
                        push(&mut l, self.scale_slots);
                        push(&mut l, &tree.first_child);
                        push(&mut l, &dp.slots);
                        l.arg(&*self.alive).arg(&self.alive_len);
                        push_mut(&mut l, regrets, &plan.storage_len);
                        push_mut(&mut l, regret_scale, &plan.num_scale_slots);
                        l.arg(&*sums).arg(&plan.storage_len);
                        l.arg(&*strategy_scale).arg(&plan.num_scale_slots);
                        push(&mut l, &dp.fused);
                        push(&mut l, &tree.rows);
                        push_mut(&mut l, self.values, &self.values_len);
                        l.arg(&hands).arg(&m.mode).arg(&m.positive_discount).arg(&m.negative_discount);
                        // SAFETY: arguments match `own_update_compressed`;
                        // indices (scale slots included) are validated.
                        unsafe { l.launch(per_node_grid(lv.own_count)) }?;
                    }
                }
            }
            // Only the summed opponent nodes: the fused ones are read
            // through their children by the own updates above.
            if lv.opp_sum_count > 0 {
                let mut l = stream.launch_builder(&kernels.opponent_sum);
                push(&mut l, &dp.opp_nodes);
                l.arg(&lv.opp_start).arg(&lv.opp_sum_count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.first_child);
                push(&mut l, &dp.slots);
                l.arg(&*self.alive).arg(&self.alive_len);
                push(&mut l, &tree.rows);
                push_mut(&mut l, self.values, &self.values_len);
                l.arg(&hands);
                // SAFETY: arguments match `opponent_sum`; indices are validated.
                unsafe { l.launch(per_hand_grid(hands, lv.opp_sum_count)) }?;
            }
            if lv.chance_count > 0 {
                let mut l = stream.launch_builder(&kernels.chance_sum);
                push(&mut l, &dp.chance_nodes);
                l.arg(&lv.chance_start).arg(&lv.chance_count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.first_child);
                push(&mut l, &tree.cards);
                push(&mut l, &tree.iso_start);
                push(&mut l, &tree.iso_count);
                push(&mut l, &tree.iso_child);
                push(&mut l, &tree.iso_pair);
                push(&mut l, &dp.perm);
                push(&mut l, &dp.my_cards);
                push(&mut l, &dp.slots);
                l.arg(&*self.alive).arg(&self.alive_len);
                push(&mut l, &tree.rows);
                push_mut(&mut l, self.values, &self.values_len);
                l.arg(&hands);
                // SAFETY: arguments match `chance_sum`; indices are validated.
                unsafe { l.launch(per_hand_grid(hands, lv.chance_count)) }?;
            }
        }
        Ok(())
    }
}
