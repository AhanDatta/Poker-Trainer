//! Device side of a three-player GPU solve (`solver::multi`): the plan and
//! the regret/strategy storage are uploaded once; each traversal is a fixed
//! sequence of level-by-level launches on one stream (see
//! `cuda/kernels/multi_traverse.cu` and `multi_leaves.cu`).
//!
//! Whole trees, either terminal model, f32 or 16-bit storage. With
//! `GpuPrecision::Exact` everything is bit-identical to the CPU; `Fast`
//! sums the leaves in double-single.

mod leaves;
#[cfg(test)]
mod tests;

use super::plan::{MultiPlan, MultiPlayerPlan, OpponentView};
use crate::cfr::{Discount, Mode};
use crate::gpu::device::{
    begin_traversal, check_memory, per_hand_grid, per_terminal_grid, push, push_mut, Buffer, DeviceStorage, HostStorage,
    ModeArgs, THREADS,
};
use crate::gpu::{invalid, Gpu, GpuError, GpuPrecision};
use crate::multi::N;
use cudarc::driver::{CudaSlice, LaunchArgs, PushKernelArg};

/// One opponent of a traverser on the device (`OpponentView`).
struct DeviceOpponent {
    hands: u32,
    weights: Buffer<f32>,
    cards: Buffer<u32>,
}

impl DeviceOpponent {
    fn upload(gpu: &Gpu, v: &OpponentView) -> Result<DeviceOpponent, GpuError> {
        Ok(DeviceOpponent { hands: v.hands, weights: gpu.upload(&v.weights)?, cards: gpu.upload(&v.cards)? })
    }
}

/// One traverser's plan on the device.
struct DevicePlayer {
    slots: [Buffer<u32>; 2],
    child_slots: [Buffer<u32>; 2],
    opp_nodes: [Buffer<u32>; 2],
    own_nodes: Buffer<u32>,
    chance_nodes: Buffer<u32>,
    strategy_update: Buffer<u32>,
    fixed: Buffer<u32>,
    fixed_payoff: Buffer<f64>,
    showdowns: Buffer<u32>,
    win: Buffer<f64>,
    lose: Buffer<f64>,
    my_cards: Buffer<u32>,
    opponents: [DeviceOpponent; 2],
}

impl DevicePlayer {
    fn upload(gpu: &Gpu, pp: &MultiPlayerPlan) -> Result<DevicePlayer, GpuError> {
        let [s0, s1] = &pp.slot;
        let [c0, c1] = &pp.child_slot;
        let [o0, o1] = &pp.opp_nodes;
        let [v0, v1] = &pp.opponents;
        Ok(DevicePlayer {
            slots: [gpu.upload(s0)?, gpu.upload(s1)?],
            child_slots: [gpu.upload(c0)?, gpu.upload(c1)?],
            opp_nodes: [gpu.upload(o0)?, gpu.upload(o1)?],
            own_nodes: gpu.upload(&pp.own_nodes)?,
            chance_nodes: gpu.upload(&pp.chance_nodes)?,
            strategy_update: gpu.upload(&pp.strategy_update)?,
            fixed: gpu.upload(&pp.fixed)?,
            fixed_payoff: gpu.upload(&pp.fixed_payoff)?,
            showdowns: gpu.upload(&pp.showdowns)?,
            win: gpu.upload(&pp.win)?,
            lose: gpu.upload(&pp.lose)?,
            my_cards: gpu.upload(&pp.my_cards)?,
            opponents: [DeviceOpponent::upload(gpu, v0)?, DeviceOpponent::upload(gpu, v1)?],
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
    /// Per node: a decision node's scale slot in 16-bit storage.
    scale_slots: Buffer<u32>,
    /// Per node: its value row (one per node: the tree isn't split yet).
    rows: Buffer<u32>,
}

impl DeviceTree {
    fn upload(gpu: &Gpu, plan: &MultiPlan) -> Result<DeviceTree, GpuError> {
        let rows: Vec<u32> = (0..plan.num_nodes).collect();
        Ok(DeviceTree {
            actions: gpu.upload(&plan.actions)?,
            first_child: gpu.upload(&plan.first_child)?,
            offsets: gpu.upload(&plan.offset)?,
            cards: gpu.upload(&plan.card)?,
            chance_factor: gpu.upload(&plan.chance_factor)?,
            scale_slots: gpu.upload(&plan.scale_slot)?,
            rows: gpu.upload(&rows)?,
        })
    }
}

/// A three-player game tree with its regrets and strategy sums on the GPU.
pub(crate) struct MultiEngine {
    gpu: Gpu,
    plan: MultiPlan,
    tree: DeviceTree,
    players: [DevicePlayer; N],
    storage: DeviceStorage,
    /// Per opponent (of whichever traverser runs): reach slots and their
    /// alive flags, sized for the largest traversal.
    reach: [CudaSlice<f32>; 2],
    alive: [CudaSlice<u32>; 2],
    /// One value row per node, sized for the largest hand count.
    values: CudaSlice<f32>,
    /// The leaf kernels' lists, groups and scratch.
    leaves: leaves::DeviceLeaves,
}

impl MultiEngine {
    /// Uploads a validated plan and the current storage.
    pub(crate) fn new(
        gpu: Gpu,
        plan: MultiPlan,
        host: HostStorage<'_>,
        precision: GpuPrecision,
    ) -> Result<MultiEngine, GpuError> {
        plan.validate()?;
        let storage_bytes = host.check(plan.storage_len, plan.num_scale_slots)?;
        let mut reach_len = [0usize; 2];
        let mut alive_len = [0usize; 2];
        let mut hands = 0usize;
        for pp in &plan.players {
            hands = hands.max(pp.hands as usize);
            let per_opponent = reach_len.iter_mut().zip(alive_len.iter_mut()).zip(&pp.num_slots).zip(&pp.opponents);
            for (((reach, alive), &slots), view) in per_opponent {
                *reach = (*reach).max(slots as usize * view.hands as usize);
                *alive = (*alive).max(slots as usize);
            }
        }
        let values_len = plan.num_nodes as usize * hands;
        let scratch = 4 * (reach_len.iter().sum::<usize>() + alive_len.iter().sum::<usize>() + values_len) as u64
            + leaves::scratch_bytes(&plan, precision)?;
        check_memory(&gpu, plan.device_bytes() + storage_bytes + scratch)?;
        let [p0, p1, p2] = &plan.players;
        let ([r0, r1], [a0, a1]) = (reach_len, alive_len);
        let engine = MultiEngine {
            tree: DeviceTree::upload(&gpu, &plan)?,
            players: [DevicePlayer::upload(&gpu, p0)?, DevicePlayer::upload(&gpu, p1)?, DevicePlayer::upload(&gpu, p2)?],
            storage: DeviceStorage::upload(&gpu, host)?,
            reach: [gpu.zeros(r0)?, gpu.zeros(r1)?],
            alive: [gpu.zeros(a0)?, gpu.zeros(a1)?],
            values: gpu.zeros(values_len)?,
            leaves: leaves::DeviceLeaves::upload(&gpu, &plan, precision)?,
            gpu,
            plan,
        };
        engine.gpu.stream.synchronize()?;
        Ok(engine)
    }

    /// Copies the regrets and strategy sums back into host storage.
    pub(crate) fn download(&self, host: HostStorage<'_>) -> Result<(), GpuError> {
        host.check(self.plan.storage_len, self.plan.num_scale_slots)?;
        self.storage.download(&self.gpu, host)?;
        self.gpu.stream.synchronize()?;
        Ok(())
    }

    /// One DCFR iteration: players 0, 1 and 2 in turn, each seeing the
    /// previous ones' updates, as on the CPU.
    pub(crate) fn cfr_iteration(&mut self, d: Discount) -> Result<(), GpuError> {
        for p in 0..N {
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

    /// Expected payoff of each player at the root (CPU:
    /// `MultiSolver::root_values`, same summation).
    pub(crate) fn root_values(&mut self, mode: Mode) -> Result<[f64; N], GpuError> {
        let mut res = [0f64; N];
        for (p, r) in res.iter_mut().enumerate() {
            let v = self.root_hand_values(p, mode)?;
            let w = self.plan.root_weights(p)?;
            *r = v.iter().zip(w).map(|(&x, &w)| x as f64 * w as f64).sum();
        }
        Ok(res)
    }

    /// Average best-response gain over the three players (CPU:
    /// `MultiSolver::convergence`, same summation).
    pub(crate) fn convergence(&mut self) -> Result<f64, GpuError> {
        let br = self.root_values(Mode::BestResponse)?;
        let ev = self.root_values(Mode::Average)?;
        Ok(br.iter().zip(&ev).map(|(b, e)| b - e).sum::<f64>() / N as f64)
    }

    /// Enqueues one traversal by player `p`: begin, levels top-down, the
    /// leaves, levels bottom-up. The root's values end up in row 0.
    fn traverse(&mut self, p: usize, mode: Mode) -> Result<(), GpuError> {
        let MultiEngine { gpu, plan, tree, players, storage, reach, alive, values, leaves, .. } = self;
        let plan: &MultiPlan = plan;
        let (Some(pp), Some(dp)) = (plan.players.get(p), players.get(p)) else {
            return Err(invalid("player out of range"));
        };
        // `MultiPlan::validate` checked these fit in u32; checked again for free.
        let mut reach_len = [0u32; 2];
        for ((len, &slots), view) in reach_len.iter_mut().zip(&pp.num_slots).zip(&pp.opponents) {
            *len = slots.checked_mul(view.hands).ok_or_else(|| invalid("reach buffer overflow"))?;
        }
        let values_len = plan.num_nodes.checked_mul(pp.hands).ok_or_else(|| invalid("value buffer overflow"))?;
        let mut pass = Pass {
            gpu,
            plan,
            tree,
            pp,
            dp,
            storage,
            reach,
            alive,
            values,
            leaves,
            p,
            m: ModeArgs::new(mode),
            reach_len,
            alive_len: pp.num_slots,
            values_len,
        };
        pass.begin()?;
        pass.top_down()?;
        pass.leaves()?;
        pass.bottom_up()
    }
}

/// One traversal's launches: the engine's buffers plus the traverser's plan.
struct Pass<'a> {
    gpu: &'a Gpu,
    plan: &'a MultiPlan,
    tree: &'a DeviceTree,
    pp: &'a MultiPlayerPlan,
    dp: &'a DevicePlayer,
    storage: &'a mut DeviceStorage,
    reach: &'a mut [CudaSlice<f32>; 2],
    alive: &'a mut [CudaSlice<u32>; 2],
    values: &'a mut CudaSlice<f32>,
    leaves: &'a mut leaves::DeviceLeaves,
    p: usize,
    m: ModeArgs,
    reach_len: [u32; 2],
    alive_len: [u32; 2],
    values_len: u32,
}

/// Element `k` of a per-opponent pair.
fn nth<T>(v: &[T; 2], k: usize) -> Result<&T, GpuError> {
    v.get(k).ok_or_else(|| invalid("opponent out of range"))
}

/// `(&mut v[k], &v[1 - k])`: the acting opponent's buffer and the other's.
fn this_and_other<T>(v: &mut [T; 2], k: usize) -> Result<(&mut T, &T), GpuError> {
    let [a, b] = v;
    match k {
        0 => Ok((a, b)),
        1 => Ok((b, a)),
        _ => Err(invalid("opponent out of range")),
    }
}

impl Pass<'_> {
    /// Both opponents' root reach (slot 0) and alive flags.
    fn begin(&mut self) -> Result<(), GpuError> {
        let per_opponent = self.reach.iter_mut().zip(self.alive.iter_mut()).zip(self.reach_len).zip(self.alive_len);
        for (k, (((reach, alive), reach_len), alive_len)) in per_opponent.enumerate() {
            let weights = &nth(&self.dp.opponents, k)?.weights;
            begin_traversal(self.gpu, alive, alive_len, reach, reach_len, weights)?;
        }
        Ok(())
    }

    /// Top-down, root first: each opponent's nodes split its reach, chance
    /// nodes split both (one launch per opponent).
    fn top_down(&mut self) -> Result<(), GpuError> {
        let Pass { gpu, plan, tree, pp, dp, storage, reach, alive, m, reach_len, alive_len, .. } = self;
        let (kernels, stream) = (&gpu.kernels, &gpu.stream);
        for lv in &pp.levels {
            for (k, span) in lv.opp.iter().enumerate() {
                if span.count == 0 {
                    continue;
                }
                let opp_hands = nth(&dp.opponents, k)?.hands;
                let (reach, _) = this_and_other(reach, k)?;
                let (alive, other_alive) = this_and_other(alive, k)?;
                let compressed = matches!(storage, DeviceStorage::Compressed { .. });
                let kernel =
                    if compressed { &kernels.multi_opponent_split_compressed } else { &kernels.multi_opponent_split };
                let mut l = stream.launch_builder(kernel);
                push(&mut l, nth(&dp.opp_nodes, k)?);
                l.arg(&span.start).arg(&span.count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.offsets);
                if compressed {
                    push(&mut l, &tree.scale_slots);
                }
                push(&mut l, &dp.strategy_update);
                push(&mut l, nth(&dp.slots, k)?);
                push(&mut l, nth(&dp.child_slots, k)?);
                push_mut(&mut l, reach, nth(reach_len, k)?);
                push_mut(&mut l, alive, nth(alive_len, k)?);
                push(&mut l, nth(&dp.slots, 1 - k)?);
                l.arg(other_alive).arg(nth(alive_len, 1 - k)?);
                let grid = match storage {
                    DeviceStorage::F32 { regrets, sums } => {
                        l.arg(&*regrets).arg(&plan.storage_len);
                        push_mut(&mut l, sums, &plan.storage_len);
                        per_hand_grid(opp_hands, span.count)
                    }
                    DeviceStorage::Compressed { regrets, sums, strategy_scale, .. } => {
                        l.arg(&*regrets).arg(&plan.storage_len);
                        push_mut(&mut l, sums, &plan.storage_len);
                        push_mut(&mut l, strategy_scale, &plan.num_scale_slots);
                        // One block per node.
                        per_terminal_grid(span.count, THREADS, 0)
                    }
                };
                l.arg(&opp_hands).arg(&m.mode).arg(&m.strategy_discount).arg(&m.reach_weighted);
                // SAFETY: arguments match `multi_opponent_split` (or its
                // 16-bit twin); every index the kernel reads, scale slots
                // included, comes from the validated plan.
                unsafe { l.launch(grid) }?;
            }
            if lv.chance.count == 0 {
                continue;
            }
            for k in 0..2 {
                let view = nth(&dp.opponents, k)?;
                let (reach, _) = this_and_other(reach, k)?;
                let (alive, other_alive) = this_and_other(alive, k)?;
                // The CPU applies the chance factor to the first opponent's reach only.
                let apply_factor = u32::from(k == 0);
                let mut l = stream.launch_builder(&kernels.multi_chance_split);
                push(&mut l, &dp.chance_nodes);
                l.arg(&lv.chance.start).arg(&lv.chance.count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.first_child);
                push(&mut l, &tree.cards);
                push(&mut l, &tree.chance_factor);
                push(&mut l, nth(&dp.slots, k)?);
                push(&mut l, nth(&dp.child_slots, k)?);
                push_mut(&mut l, reach, nth(reach_len, k)?);
                push_mut(&mut l, alive, nth(alive_len, k)?);
                push(&mut l, nth(&dp.slots, 1 - k)?);
                l.arg(other_alive).arg(nth(alive_len, 1 - k)?);
                push(&mut l, &view.cards);
                l.arg(&view.hands).arg(&apply_factor);
                // SAFETY: arguments match `multi_chance_split`; indices are validated.
                unsafe { l.launch(per_hand_grid(view.hands, lv.chance.count)) }?;
            }
        }
        Ok(())
    }

    /// Fixed-payoff leaves and showdowns.
    fn leaves(&mut self) -> Result<(), GpuError> {
        let Pass { gpu, tree, pp, dp, reach, alive, values, reach_len, alive_len, values_len, leaves, p, .. } = self;
        let args = leaves::Leaves {
            gpu,
            tree,
            dp,
            p: *p,
            hands: pp.hands,
            reach,
            reach_len,
            alive,
            alive_len,
            values,
            values_len,
        };
        leaves::launch(args, leaves)
    }

    /// Bottom-up, deepest level first: own updates, opponent sums, chance sums.
    fn bottom_up(&mut self) -> Result<(), GpuError> {
        let Pass { gpu, plan, tree, pp, dp, storage, alive, values, m, alive_len, values_len, .. } = self;
        let (kernels, stream, hands) = (&gpu.kernels, &gpu.stream, pp.hands);
        let opponents = Opponents { dp, alive, alive_len };
        for lv in pp.levels.iter().rev() {
            if lv.own.count > 0 {
                let compressed = matches!(storage, DeviceStorage::Compressed { .. });
                let kernel = if compressed { &kernels.multi_own_update_compressed } else { &kernels.multi_own_update };
                let mut l = stream.launch_builder(kernel);
                push(&mut l, &dp.own_nodes);
                l.arg(&lv.own.start).arg(&lv.own.count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.offsets);
                if compressed {
                    push(&mut l, &tree.scale_slots);
                }
                push(&mut l, &tree.first_child);
                opponents.push_liveness(&mut l)?;
                let grid = match &mut **storage {
                    DeviceStorage::F32 { regrets, sums } => {
                        push_mut(&mut l, regrets, &plan.storage_len);
                        l.arg(&*sums).arg(&plan.storage_len);
                        per_hand_grid(hands, lv.own.count)
                    }
                    DeviceStorage::Compressed { regrets, sums, regret_scale, strategy_scale } => {
                        push_mut(&mut l, regrets, &plan.storage_len);
                        push_mut(&mut l, regret_scale, &plan.num_scale_slots);
                        l.arg(&*sums).arg(&plan.storage_len);
                        l.arg(&*strategy_scale).arg(&plan.num_scale_slots);
                        per_terminal_grid(lv.own.count, THREADS, 0)
                    }
                };
                push(&mut l, &tree.rows);
                push_mut(&mut l, values, values_len);
                l.arg(&hands).arg(&m.mode).arg(&m.positive_discount).arg(&m.negative_discount);
                // SAFETY: arguments match `multi_own_update` (or its 16-bit
                // twin); indices, scale slots included, are validated.
                unsafe { l.launch(grid) }?;
            }
            for (k, span) in lv.opp.iter().enumerate() {
                if span.count == 0 {
                    continue;
                }
                let mut l = stream.launch_builder(&kernels.multi_opponent_sum);
                push(&mut l, nth(&dp.opp_nodes, k)?);
                l.arg(&span.start).arg(&span.count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.first_child);
                opponents.push_liveness(&mut l)?;
                push(&mut l, &tree.rows);
                push_mut(&mut l, values, values_len);
                l.arg(&hands);
                // SAFETY: arguments match `multi_opponent_sum`; indices are validated.
                unsafe { l.launch(per_hand_grid(hands, span.count)) }?;
            }
            if lv.chance.count > 0 {
                let mut l = stream.launch_builder(&kernels.multi_chance_sum);
                push(&mut l, &dp.chance_nodes);
                l.arg(&lv.chance.start).arg(&lv.chance.count);
                push(&mut l, &tree.actions);
                push(&mut l, &tree.first_child);
                push(&mut l, &tree.cards);
                push(&mut l, &dp.my_cards);
                opponents.push_liveness(&mut l)?;
                push(&mut l, &tree.rows);
                push_mut(&mut l, values, values_len);
                l.arg(&hands);
                // SAFETY: arguments match `multi_chance_sum`; indices are validated.
                unsafe { l.launch(per_hand_grid(hands, lv.chance.count)) }?;
            }
        }
        Ok(())
    }
}

/// Read access to both opponents' reach state, for the launches that only
/// read their liveness.
struct Opponents<'p> {
    dp: &'p DevicePlayer,
    alive: &'p [CudaSlice<u32>; 2],
    alive_len: &'p [u32; 2],
}

impl<'p> Opponents<'p> {
    /// Both opponents' slots and alive flags (the kernels' `TwoOpponents`).
    fn push_liveness(&self, l: &mut LaunchArgs<'p>) -> Result<(), GpuError> {
        for k in 0..2 {
            push(l, nth(&self.dp.slots, k)?);
            l.arg(nth(self.alive, k)?).arg(nth(self.alive_len, k)?);
        }
        Ok(())
    }
}
