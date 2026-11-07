//! The leaves on the device (`cuda/kernels/multi_leaves.cu`): the plan's
//! hand lists and leaf groups, the per-lane scratch, and the launches.

use super::{nth, DevicePlayer, DeviceTree};
use crate::gpu::device::{push, push_mut, Buffer};
use crate::gpu::multi::plan::{LeafGroups, MultiPlan, SweepLists, LEAF_LANES};
use crate::gpu::shared::to_u32;
use crate::gpu::{invalid, Gpu, GpuError, GpuPrecision};
use crate::multi::{opponents, N};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchArgs, LaunchConfig, PushKernelArg};

/// Threads per block (`lanes::kBlock`).
const BLOCK: u32 = 128;

/// Per-lane state fields of each model: `independent::kStateFields`, and
/// `exact::kStateFields` (two sets of 260).
const STATE_FIELDS: [u64; 2] = [159, 2 * 260];

/// Most lanes the scratch is sized for: about what this GPU keeps resident
/// (24 SMs, 7 blocks of 128 each at the kernels' register use). The leaf
/// kernels are latency-bound, so fewer lanes are slower (8192: 1.3x,
/// 3072: 2.3x on the chart-range flop).
const MAX_LANES: u64 = 24_576;

/// Scratch bytes the lanes may take; fewer lanes when hand lists are long.
const SCRATCH_BUDGET: u64 = 512 << 20;

/// The plan's [`SweepLists`] on the device.
struct DeviceLists {
    off: Buffer<u32>,
    strength: Buffer<u32>,
    hand: Buffer<u32>,
    cards: Buffer<u32>,
    card_off: Buffer<u32>,
    holding: Buffer<u32>,
    combo_pos: Buffer<u16>,
    hand_order: u32,
}

impl DeviceLists {
    fn upload(gpu: &Gpu, l: &SweepLists) -> Result<DeviceLists, GpuError> {
        Ok(DeviceLists {
            off: gpu.upload(&l.off)?,
            strength: gpu.upload(&l.strength)?,
            hand: gpu.upload(&l.hand)?,
            cards: gpu.upload(&l.cards)?,
            card_off: gpu.upload(&l.card_off)?,
            holding: gpu.upload(&l.holding)?,
            combo_pos: gpu.upload(&l.combo_pos)?,
            hand_order: l.orders.checked_sub(1).ok_or_else(|| invalid("no sweep orders"))?,
        })
    }

    /// In `SOLVER_LEAF_LIST_PARAMS` order.
    fn push<'a>(&'a self, l: &mut LaunchArgs<'a>) {
        push(l, &self.off);
        push(l, &self.strength);
        push(l, &self.hand);
        push(l, &self.cards);
        push(l, &self.card_off);
        push(l, &self.holding);
        push(l, &self.combo_pos);
    }
}

/// One traverser's [`LeafGroups`] on the device.
struct DeviceGroups {
    groups: Buffer<u32>,
    showdown_lanes: Buffer<u32>,
    split2: Buffer<f64>,
    split3: Buffer<f64>,
    job_slots: [Buffer<u32>; 2],
    job_off: Buffer<u32>,
    job_leaves: Buffer<u32>,
    fixed_scalar: Buffer<u32>,
    showdown_warps: u32,
    fixed_warps: u32,
}

impl DeviceGroups {
    fn upload(gpu: &Gpu, x: &LeafGroups) -> Result<DeviceGroups, GpuError> {
        let [s0, s1] = &x.job_slots;
        Ok(DeviceGroups {
            groups: gpu.upload(&x.groups)?,
            showdown_lanes: gpu.upload(&x.showdown_lanes)?,
            split2: gpu.upload(&x.split2)?,
            split3: gpu.upload(&x.split3)?,
            job_slots: [gpu.upload(s0)?, gpu.upload(s1)?],
            job_off: gpu.upload(&x.job_off)?,
            job_leaves: gpu.upload(&x.job_leaves)?,
            fixed_scalar: gpu.upload(&x.fixed_scalar)?,
            showdown_warps: to_u32(x.showdown_warps(), "showdown warps")?,
            fixed_warps: to_u32(x.fixed_warps(), "fixed warps")?,
        })
    }
}

/// Per-lane scratch (`lanes::Lane`): column `t` belongs to grid thread `t`.
struct Scratch {
    state: CudaSlice<f64>,
    state_len: u32,
    reach: CudaSlice<f32>,
    reach_len: u32,
    stash: CudaSlice<f64>,
    stash_len: u32,
    blocks: u32,
}

impl Scratch {
    /// In `SOLVER_LEAF_SCRATCH_PARAMS` order.
    fn push<'a>(&'a mut self, l: &mut LaunchArgs<'a>) {
        let Scratch { state, state_len, reach, reach_len, stash, stash_len, .. } = self;
        push_mut(l, state, state_len);
        push_mut(l, reach, reach_len);
        push_mut(l, stash, stash_len);
    }
}

/// The model's kernel index: 0 independent, 1 exact removal.
fn model(plan: &MultiPlan) -> usize {
    usize::from(plan.exact)
}

/// Lanes and scratch sizes for `plan`: `(blocks, [state, reach, stash]
/// lengths)`. A lane needs its model's state, both opponents' reach in two
/// list orders (`lanes::Lane`) and two stash rows per traverser hand.
fn scratch_shape(plan: &MultiPlan) -> Result<(u32, [u64; 3]), GpuError> {
    let mut warps = 1u64;
    let (mut reach_rows, mut hands) = (1u64, 1u64);
    for pp in &plan.players {
        let x = &pp.leaves;
        warps = warps.max(x.showdown_warps().max(x.fixed_warps()) as u64);
        reach_rows = reach_rows.max(pp.opponents.iter().map(|v| u64::from(v.hands)).sum());
        hands = hands.max(u64::from(pp.hands));
    }
    let fields = *STATE_FIELDS.get(model(plan)).ok_or_else(|| invalid("terminal model"))?;
    let per_lane = fields * 8 + 2 * reach_rows * 4 + 2 * hands * 8;
    let lanes = (warps * LEAF_LANES as u64).min(MAX_LANES).min(SCRATCH_BUDGET / per_lane);
    let blocks = (lanes / u64::from(BLOCK)).max(1);
    let lanes = blocks * u64::from(BLOCK);
    let lens = [fields * lanes, 2 * reach_rows * lanes, 2 * hands * lanes];
    for len in lens {
        u32::try_from(len).map_err(|_| invalid("leaf scratch exceeds u32 indexing"))?;
    }
    Ok((u32::try_from(blocks).map_err(|_| invalid("leaf grid"))?, lens))
}

/// Device bytes of the leaf scratch for `plan`.
pub(super) fn scratch_bytes(plan: &MultiPlan) -> Result<u64, GpuError> {
    let (_, [state, reach, stash]) = scratch_shape(plan)?;
    Ok(state * 8 + reach * 4 + stash * 8)
}

/// Everything the leaf kernels need on the device.
pub(super) struct DeviceLeaves {
    lists: DeviceLists,
    groups: [DeviceGroups; N],
    scratch: Scratch,
    model: usize,
    /// `[exact, fast]` kernel index.
    precision: usize,
}

impl DeviceLeaves {
    pub(super) fn upload(gpu: &Gpu, plan: &MultiPlan, precision: GpuPrecision) -> Result<DeviceLeaves, GpuError> {
        let (blocks, [state, reach, stash]) = scratch_shape(plan)?;
        let len = |n: u64| to_u32(n as usize, "leaf scratch");
        let [p0, p1, p2] = &plan.players;
        Ok(DeviceLeaves {
            lists: DeviceLists::upload(gpu, &plan.lists)?,
            groups: [
                DeviceGroups::upload(gpu, &p0.leaves)?,
                DeviceGroups::upload(gpu, &p1.leaves)?,
                DeviceGroups::upload(gpu, &p2.leaves)?,
            ],
            scratch: Scratch {
                state: gpu.zeros(state as usize)?,
                state_len: len(state)?,
                reach: gpu.zeros(reach as usize)?,
                reach_len: len(reach)?,
                stash: gpu.zeros(stash as usize)?,
                stash_len: len(stash)?,
                blocks,
            },
            model: model(plan),
            precision: match precision {
                GpuPrecision::Exact => 0,
                GpuPrecision::Fast => 1,
            },
        })
    }
}

/// What the leaf launches of one traversal read: the engine's reach state
/// and value rows, and the traverser's plan.
pub(super) struct Leaves<'a> {
    pub gpu: &'a Gpu,
    pub tree: &'a DeviceTree,
    pub dp: &'a DevicePlayer,
    pub p: usize,
    pub hands: u32,
    pub reach: &'a [CudaSlice<f32>; 2],
    pub reach_len: &'a [u32; 2],
    pub alive: &'a [CudaSlice<u32>; 2],
    pub alive_len: &'a [u32; 2],
    pub values: &'a mut CudaSlice<f32>,
    pub values_len: &'a u32,
}

/// Both opponents' reach state, in `SOLVER_LEAF_OPPONENT_PARAMS` order.
struct Opponents<'a> {
    dp: &'a DevicePlayer,
    reach: &'a [CudaSlice<f32>; 2],
    reach_len: &'a [u32; 2],
    alive: &'a [CudaSlice<u32>; 2],
    alive_len: &'a [u32; 2],
}

impl<'a> Opponents<'a> {
    fn push(&self, l: &mut LaunchArgs<'a>) -> Result<(), GpuError> {
        for k in 0..2 {
            l.arg(&nth(&self.dp.opponents, k)?.hands);
            push(l, nth(&self.dp.slots, k)?);
            l.arg(nth(self.alive, k)?).arg(nth(self.alive_len, k)?);
            l.arg(nth(self.reach, k)?).arg(nth(self.reach_len, k)?);
        }
        Ok(())
    }
}

/// A leaf kernel by model and precision.
fn pick(k: &[[CudaFunction; 2]; 2], model: usize, precision: usize) -> Result<&CudaFunction, GpuError> {
    k.get(model).and_then(|m| m.get(precision)).ok_or_else(|| invalid("no such leaf kernel"))
}

/// Enqueues traverser `leaves.p`'s fixed payoffs and showdowns.
pub(super) fn launch(leaves: Leaves<'_>, device: &mut DeviceLeaves) -> Result<(), GpuError> {
    let Leaves { gpu, tree, dp, p, hands, reach, reach_len, alive, alive_len, values, values_len } = leaves;
    let DeviceLeaves { lists, groups, scratch, model, precision } = device;
    let g = groups.get(p).ok_or_else(|| invalid("player out of range"))?;
    let (q, r) = opponents(p);
    let seats = [to_u32(p, "seat")?, to_u32(q, "seat")?, to_u32(r, "seat")?];
    let kernels = &gpu.kernels;
    let blocks = scratch.blocks;
    let cfg = |warps: u32| LaunchConfig {
        grid_dim: (blocks.min(warps.div_ceil(BLOCK / LEAF_LANES as u32)).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let opponents = Opponents { dp, reach, reach_len, alive, alive_len };
    if g.fixed_warps > 0 {
        let mut l = gpu.stream.launch_builder(pick(&kernels.multi_fixed, *model, *precision)?);
        push(&mut l, &g.job_slots[0]);
        push(&mut l, &g.job_slots[1]);
        push(&mut l, &g.job_off);
        push(&mut l, &g.job_leaves);
        push(&mut l, &dp.fixed);
        push(&mut l, &dp.fixed_payoff);
        push(&mut l, &g.fixed_scalar);
        push(&mut l, &tree.rows);
        push_mut(&mut l, values, values_len);
        l.arg(&hands).arg(&seats[0]).arg(&seats[1]).arg(&seats[2]).arg(&lists.hand_order);
        lists.push(&mut l);
        opponents.push(&mut l)?;
        scratch.push(&mut l);
        // SAFETY: arguments match `multi_fixed_*`; every index comes from
        // the validated plan, and the scratch has a column per thread of the
        // grid (at most `scratch.blocks` blocks).
        unsafe { l.launch(cfg(g.fixed_warps)) }?;
    }
    if g.showdown_warps > 0 {
        let mut l = gpu.stream.launch_builder(pick(&kernels.multi_showdowns, *model, *precision)?);
        push(&mut l, &g.groups);
        push(&mut l, &g.showdown_lanes);
        push(&mut l, &dp.showdowns);
        push(&mut l, &dp.win);
        push(&mut l, &dp.lose);
        push(&mut l, &g.split2);
        push(&mut l, &g.split3);
        push(&mut l, &tree.rows);
        push_mut(&mut l, values, values_len);
        l.arg(&hands).arg(&seats[0]).arg(&seats[1]).arg(&seats[2]).arg(&lists.hand_order);
        lists.push(&mut l);
        opponents.push(&mut l)?;
        scratch.push(&mut l);
        // SAFETY: arguments match `multi_showdowns_*`; as above.
        unsafe { l.launch(cfg(g.showdown_warps)) }?;
    }
    Ok(())
}
