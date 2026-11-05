//! Host-side plan of a GPU traversal: the game tree flattened into the
//! arrays the kernels index, a level schedule for each traverser, terminal
//! lists with their payoffs, and per-board showdown lookup tables.
//!
//! Built once per solve. [`Plan::validate`] checks every invariant the
//! kernels rely on (all indices in range, sizes fit the kernels' `u32`
//! indexing, a consistent schedule), so a malformed plan is an error here
//! rather than an out-of-bounds access on the device.

use super::{invalid, GpuError};
use crate::cfr::Ctx;
use crate::hands::{HandSet, StrengthTable};
use crate::tree::{Edge, Kind};
use std::ops::Range;

mod schedule;
pub(crate) use schedule::{scratch_bytes, Scratch, Segment, Span};

/// "No index" marker, shared with the kernels (`solver::kNone`).
pub(crate) const NONE: u32 = u32::MAX;

/// `Plan::actor` of a chance node (decision nodes hold the player, 0 or 1).
pub(crate) const CHANCE: u32 = 2;

/// Cards in the deck.
pub(crate) const CARDS: usize = 52;

/// Largest normalized payoff and range weight the plan accepts: with at most
/// 1326 hands, every product the terminal kernels form stays below 1e22,
/// far inside f32's (and so double-single's) range of 3.4e38.
pub(crate) const MAX_PAYOFF: f64 = 1e12;
pub(crate) const MAX_WEIGHT: f32 = 1e6;

/// Suit pairs (`iso::PAIRS`), the rows of a permutation table.
pub(crate) const SUIT_PAIRS: usize = 6;

/// Most actions per decision node (the 16-bit kernels track alive children
/// in a 32-bit mask; `kMaxActions` in traverse_compressed.cu).
pub(crate) const MAX_ACTIONS: u32 = 32;

/// Discriminants of the kernels' `enum class Mode`.
pub(crate) const MODE_CFR: u32 = 0;
pub(crate) const MODE_BEST_RESPONSE: u32 = 1;
pub(crate) const MODE_AVERAGE: u32 = 2;

/// Dynamic shared memory the showdown kernel may use without an opt-in
/// (48 KiB minus a margin for its static CUB storage).
pub(crate) const SHOWDOWN_SHARED_LIMIT: u32 = 44 * 1024;

/// Bytes of dynamic shared memory the showdown kernel needs for opponent
/// lists of up to `max_opp` hands: two f64 prefix arrays and the f32 weights.
pub(crate) fn showdown_shared_bytes(max_opp: u32) -> u64 {
    (3 * u64::from(max_opp) + 1 + CARDS as u64) * 8 + u64::from(max_opp) * 4
}

/// One tree level: ranges into the opponent-node, own-node and chance-node
/// lists. The level's first `opp_sum_count` opponent nodes get their value
/// row from `opponent_sum`; the rest are fused into their parents (see
/// [`PlayerPlan::fused`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LevelRange {
    pub opp_start: u32,
    pub opp_count: u32,
    pub opp_sum_count: u32,
    pub own_start: u32,
    pub own_count: u32,
    pub chance_start: u32,
    pub chance_count: u32,
}

/// Showdown lookup for one traverser, for every board (strength table)
/// concatenated. For table `t`:
/// * `opp_order[opp_off[t]..][..opp_len[t]]`: the opponent's hands sorted
///   by strength, weakest first;
/// * `card_off[t * 53..][..53]`: for each card, the range (relative to
///   `card_base[t]`) of `card_pos` listing the positions (in the opponent
///   order) of the opponent hands holding that card, ascending;
/// * `ranges[t * hands + h]`: `lo | hi << 16` where opponent positions
///   `[0, lo)` are weaker than the traverser's hand `h` and `[lo, hi)` tie
///   with it; `NONE` if `h` conflicts with the board;
/// * `ranks[t * hands + h]`: for each of `h`'s two cards (in `my_cards`
///   order), how many entries of that card's list are below `lo` and below
///   `hi`, packed `first_lo | first_hi << 8 | second_lo << 16 | second_hi << 24`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Tables {
    pub opp_off: Vec<u32>,
    pub opp_len: Vec<u32>,
    pub opp_order: Vec<u32>,
    pub card_base: Vec<u32>,
    pub card_off: Vec<u32>,
    pub card_pos: Vec<u32>,
    pub ranges: Vec<u32>,
    pub ranks: Vec<u32>,
    /// Longest opponent list over the tables (sizes shared memory).
    pub max_opp: u32,
}

/// Everything a traversal by one player needs.
#[derive(Clone, Debug, Default)]
pub(crate) struct PlayerPlan {
    /// The traverser's hand count.
    pub hands: u32,
    /// The opponent's hand count.
    pub opp_hands: u32,
    /// Per node: the slot holding the opponent's reach at that node.
    pub slot: Vec<u32>,
    /// Per node: the first child's slot at opponent and chance nodes
    /// (children's slots are consecutive), `NONE` elsewhere.
    pub child_slot: Vec<u32>,
    pub num_slots: u32,
    /// Slots `[0, top_slots)` belong to the top segment; groups reuse the
    /// rest (see `schedule`).
    pub top_slots: u32,
    /// Per segment: its ranges of `levels`, the terminal lists and the slots.
    pub segments: Vec<Segment>,
    /// Top-down order within each segment; bottom-up walks it backwards.
    pub levels: Vec<LevelRange>,
    pub opp_nodes: Vec<u32>,
    pub own_nodes: Vec<u32>,
    pub chance_nodes: Vec<u32>,
    /// Per node: 1 for an opponent node whose parent is one of the
    /// traverser's own nodes. Its value row is never written: the parent's
    /// update sums its children's rows instead (`ChildValues` in core.cuh),
    /// which saves writing the row and reading it back. 0 elsewhere.
    pub fused: Vec<u32>,
    pub folds: Vec<u32>,
    pub fold_payoff: Vec<f64>,
    pub showdowns: Vec<u32>,
    pub showdown_table: Vec<u32>,
    pub win: Vec<f64>,
    pub lose: Vec<f64>,
    pub tie: Vec<f64>,
    /// Per showdown: a fold terminal with the same opponent reach (both
    /// children of one of the traverser's nodes, as the CPU groups them),
    /// evaluated by the showdown kernel from the same sums and left out of
    /// `folds`; `NONE` if there is none.
    pub showdown_fold: Vec<u32>,
    pub showdown_fold_payoff: Vec<f64>,
    /// The traverser's hands' cards, packed `c1 | c2 << 8`.
    pub my_cards: Vec<u32>,
    /// Index of the opponent's identical combo, or `NONE`.
    pub same: Vec<u32>,
    /// The opponent's hands' cards, packed `c1 | c2 << 8`.
    pub opp_cards: Vec<u32>,
    /// Suit-swap permutations of the traverser's hands: row `pair` (of
    /// [`SUIT_PAIRS`], `hands` entries each) maps a hand to its image under
    /// that swap (`iso.perm`); `NONE` rows for pairs the ranges aren't
    /// symmetric in (never referenced by an isomorphic card).
    pub perm: Vec<u32>,
    /// For each card, the range of `opp_card_hands` listing the opponent
    /// hands that hold it (53 offsets).
    pub opp_card_off: Vec<u32>,
    pub opp_card_hands: Vec<u32>,
    /// The opponent's root weights (reach at the root).
    pub opp_weights: Vec<f32>,
    pub tables: Tables,
}

/// The flattened tree plus a plan per traverser.
#[derive(Clone, Debug, Default)]
pub(crate) struct Plan {
    pub num_nodes: u32,
    pub storage_len: u32,
    /// Per node: number of actions (0 for terminals).
    pub actions: Vec<u32>,
    /// Per node: first child (children are consecutive), 0 for terminals.
    pub first_child: Vec<u32>,
    /// Per node: storage offset of a decision node, 0 for terminals.
    pub offset: Vec<u32>,
    /// Per node: the index of a decision node's scale in 16-bit storage
    /// (`Node::slot`), `NONE` elsewhere. Unique per decision node.
    pub scale_slot: Vec<u32>,
    /// Number of scales in 16-bit storage (one per decision node).
    pub num_scale_slots: u32,
    /// Per node: the acting player of a decision node, [`CHANCE`] for a
    /// chance node, `NONE` for terminals.
    pub actor: Vec<u32>,
    /// Per node: the card dealt to reach it (children of chance nodes),
    /// `NONE` elsewhere.
    pub card: Vec<u32>,
    /// Per node: a chance node's reach multiplier, 0 elsewhere.
    pub chance_factor: Vec<f32>,
    /// Per node: a chance node's range of `iso_child` / `iso_pair` (its
    /// suit-isomorphic cards), empty elsewhere.
    pub iso_start: Vec<u32>,
    pub iso_count: Vec<u32>,
    /// Isomorphic card: the index (among its chance node's children) of
    /// the canonical child whose values it reuses, and the suit pair.
    pub iso_child: Vec<u32>,
    pub iso_pair: Vec<u32>,
    /// Per node: 1 for terminals the CPU evaluates one at a time
    /// (`eval_fold` / `eval_showdown`, whose rounding differs from the
    /// batched evaluators'), 0 elsewhere. Terminals inside a river subtree
    /// are batched (`river::run`) unless batching is switched off; the rest
    /// (folds before the river, showdowns right after an all-in runout) are
    /// evaluated one at a time.
    pub scalar: Vec<u32>,
    pub num_tables: u32,
    /// Per node: its segment (0 = the top of the tree, k = group k).
    pub segment_of: Vec<u32>,
    pub num_segments: u32,
    /// Per node: its row in the value buffer; rows `[0, top_rows)` belong
    /// to the top segment and the chunk roots, groups reuse the rest.
    pub row: Vec<u32>,
    pub top_rows: u32,
    pub num_rows: u32,
    pub players: [PlayerPlan; 2],
}

/// The role of a non-terminal node in one player's traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Own,
    Opponent,
    Chance,
}

/// Fields of one flattened node.
#[derive(Clone, Copy, Debug)]
struct NodeFields {
    actions: u32,
    first_child: u32,
    offset: u32,
    actor: u32,
}

pub(super) fn to_u32(x: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(x).map_err(|_| invalid(format!("{what} ({x}) exceeds the kernels' u32 indexing")))
}

pub(super) fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), GpuError> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidInput(msg()))
    }
}

/// `v[i]`, or an error naming `what`.
pub(super) fn at<T: Copy>(v: &[T], i: usize, what: &str) -> Result<T, GpuError> {
    v.get(i).copied().ok_or_else(|| invalid(format!("{what}[{i}] out of range")))
}

/// Player `p`'s half of a per-player pair.
fn side<T>(pair: &[T; 2], p: usize) -> Result<&T, GpuError> {
    pair.get(p).ok_or_else(|| invalid(format!("player {p} out of range")))
}

/// `c1 | c2 << 8`.
pub(crate) fn pack_cards(c1: u8, c2: u8) -> u32 {
    u32::from(c1) | u32::from(c2) << 8
}

pub(super) fn unpack_cards(packed: u32) -> (usize, usize) {
    ((packed & 0xFF) as usize, (packed >> 8 & 0xFF) as usize)
}

/// `off` is a 53-entry offset array starting at 0, non-decreasing and
/// ending at `total`.
pub(super) fn check_offsets(off: &[u32], total: usize, what: &str) -> Result<(), GpuError> {
    ensure(off.len() == CARDS + 1, || format!("{what}: {} offsets", off.len()))?;
    ensure(off.first() == Some(&0) && off.windows(2).all(|w| w.first() <= w.get(1)), || {
        format!("{what}: offsets out of order")
    })?;
    ensure(off.last().is_some_and(|&e| e as usize == total), || {
        format!("{what}: offsets don't end at {total}")
    })
}

impl Plan {
    /// Flattens the tree behind `ctx` as one segment.
    #[cfg(test)]
    pub(crate) fn build(ctx: &Ctx) -> Result<Plan, GpuError> {
        Self::build_with(ctx, Scratch::Unlimited)
    }

    /// Flattens the tree behind `ctx`, split into segments under `scratch`.
    pub(crate) fn build_with(ctx: &Ctx, scratch: Scratch) -> Result<Plan, GpuError> {
        let tree = ctx.tree;
        let num_nodes = to_u32(tree.nodes.len(), "tree nodes")?;
        let storage_len = u32::try_from(tree.storage_len)
            .map_err(|_| invalid(format!("storage of {} values exceeds u32 indexing", tree.storage_len)))?;
        let mut plan = Plan {
            num_nodes,
            storage_len,
            num_tables: to_u32(ctx.tables.len(), "strength tables")?,
            num_scale_slots: tree.num_player_nodes,
            ..Plan::default()
        };
        for node in &tree.nodes {
            let (actor, offset, factor) = match node.kind {
                Kind::Player(q) => (
                    u32::from(q),
                    u32::try_from(node.offset).map_err(|_| invalid("storage offset exceeds u32 indexing"))?,
                    0.0,
                ),
                Kind::Chance => (CHANCE, 0, node.chance_factor),
                Kind::Fold(_) | Kind::Showdown => (NONE, 0, 0.0),
            };
            let terminal = actor == NONE;
            plan.actor.push(actor);
            plan.actions.push(if terminal { 0 } else { u32::from(node.num_children) });
            plan.first_child.push(if terminal { 0 } else { node.first_child });
            plan.offset.push(offset);
            plan.scale_slot.push(if matches!(node.kind, Kind::Player(_)) { node.slot } else { NONE });
            plan.chance_factor.push(factor);
            plan.card.push(match node.edge {
                Edge::Card(c) => to_u32(c.index(), "card")?,
                Edge::Root | Edge::Action(_) => NONE,
            });
            plan.iso_start.push(to_u32(plan.iso_child.len(), "isomorphic cards")?);
            if actor == CHANCE {
                let info = tree
                    .chance
                    .get(node.slot as usize)
                    .ok_or_else(|| invalid("chance node without chance info"))?;
                for &(child, pair) in &info.iso {
                    plan.iso_child.push(u32::from(child));
                    plan.iso_pair.push(u32::from(pair));
                }
            }
            let count = to_u32(plan.iso_child.len(), "isomorphic cards")?
                - plan.iso_start.last().copied().unwrap_or(0);
            plan.iso_count.push(count);
        }
        let levels = plan.bfs_levels()?;
        plan.scalar = Self::scalar_terminals(ctx, &levels)?;
        plan.players = [PlayerPlan::build(ctx, &plan, 0)?, PlayerPlan::build(ctx, &plan, 1)?];
        // The schedule's device arrays have the same total length however
        // the tree is split: per player a slot, a child slot, a fused flag
        // and a list entry per node, 12 bytes per fold and 44 per showdown (an upper bound:
        // paired folds move into the showdown arrays); plus a row per node.
        let nodes = &ctx.tree.nodes;
        let folds = nodes.iter().filter(|n| matches!(n.kind, Kind::Fold(_))).count() as u64;
        let showdowns = nodes.iter().filter(|n| matches!(n.kind, Kind::Showdown)).count() as u64;
        let nn = u64::from(num_nodes);
        let schedule_bytes = 2 * (16 * nn + 12 * folds + 44 * showdowns + 64) + 4 * nn;
        let seg = plan.segment(ctx, scratch, plan.device_bytes() + schedule_bytes)?;
        plan.segment_of = seg.segment_of;
        plan.num_segments = seg.num_segments;
        plan.row = seg.row;
        plan.top_rows = seg.top_rows;
        plan.num_rows = seg.num_rows;
        for p in 0..2 {
            let mut pp = std::mem::take(plan.players.get_mut(p).ok_or_else(|| invalid("player out of range"))?);
            pp.build_schedule(ctx, &plan, p)?;
            *plan.players.get_mut(p).ok_or_else(|| invalid("player out of range"))? = pp;
        }
        plan.validate()?;
        Ok(plan)
    }

    /// Which terminals the CPU evaluates one at a time (see [`Plan::scalar`]).
    fn scalar_terminals(ctx: &Ctx, levels: &[Vec<u32>]) -> Result<Vec<u32>, GpuError> {
        let nodes = &ctx.tree.nodes;
        let starts_batch = |i: usize| {
            nodes
                .get(i)
                .is_some_and(|n| matches!(n.kind, Kind::Player(_)) && n.street == holdem::Street::River)
        };
        // Whether a node lies in (or roots) a batched river subtree.
        let mut batched = vec![false; nodes.len()];
        if let Some(root) = batched.first_mut() {
            *root = starts_batch(0);
        }
        for &id in levels.iter().flatten() {
            let i = id as usize;
            let node = nodes.get(i).ok_or_else(|| invalid("node out of range"))?;
            if matches!(node.kind, Kind::Fold(_) | Kind::Showdown) {
                continue;
            }
            let inherited = at(&batched, i, "batched")?;
            for c in node.children() {
                let start = starts_batch(c);
                *batched.get_mut(c).ok_or_else(|| invalid("child out of range"))? = inherited || start;
            }
        }
        nodes
            .iter()
            .zip(&batched)
            .map(|(n, &b)| {
                Ok(u32::from(matches!(n.kind, Kind::Fold(_) | Kind::Showdown) && !(ctx.batch_river && b)))
            })
            .collect()
    }

    /// Bytes the plan's arrays take on the device.
    pub(crate) fn device_bytes(&self) -> u64 {
        fn b<T>(v: &[T]) -> u64 {
            (v.len().max(1) * std::mem::size_of::<T>()) as u64
        }
        let tree = b(&self.actions)
            + b(&self.first_child)
            + b(&self.offset)
            + b(&self.scale_slot)
            + b(&self.card)
            + b(&self.chance_factor)
            + b(&self.iso_start)
            + b(&self.iso_count)
            + b(&self.iso_child)
            + b(&self.iso_pair)
            + b(&self.scalar)
            + b(&self.row);
        let player = |pp: &PlayerPlan| {
            let t = &pp.tables;
            b(&pp.slot)
                + b(&pp.child_slot)
                + b(&pp.opp_nodes)
                + b(&pp.own_nodes)
                + b(&pp.chance_nodes)
                + b(&pp.fused)
                + b(&pp.folds)
                + b(&pp.fold_payoff)
                + b(&pp.showdowns)
                + b(&pp.showdown_table)
                + b(&pp.win)
                + b(&pp.lose)
                + b(&pp.tie)
                + b(&pp.showdown_fold)
                + b(&pp.showdown_fold_payoff)
                + b(&pp.my_cards)
                + b(&pp.same)
                + b(&pp.opp_cards)
                + b(&pp.perm)
                + b(&pp.opp_card_off)
                + b(&pp.opp_card_hands)
                + b(&pp.opp_weights)
                + b(&t.opp_off)
                + b(&t.opp_len)
                + b(&t.opp_order)
                + b(&t.card_base)
                + b(&t.card_off)
                + b(&t.card_pos)
                + b(&t.ranges)
                + b(&t.ranks)
        };
        tree + self.players.iter().map(player).sum::<u64>()
    }

    /// Nodes by depth, root first.
    fn bfs_levels(&self) -> Result<Vec<Vec<u32>>, GpuError> {
        let mut levels: Vec<Vec<u32>> = Vec::new();
        let mut frontier = vec![0u32];
        let mut seen = 0usize;
        while !frontier.is_empty() {
            seen += frontier.len();
            ensure(seen <= self.num_nodes as usize, || "tree has a cycle".to_string())?;
            let mut next = Vec::new();
            for &id in &frontier {
                let node = self.node(id as usize)?;
                next.extend(node.first_child..node.first_child.saturating_add(node.actions));
            }
            levels.push(std::mem::replace(&mut frontier, next));
        }
        Ok(levels)
    }

    /// The flattened fields of node `i`.
    fn node(&self, i: usize) -> Result<NodeFields, GpuError> {
        Ok(NodeFields {
            actions: at(&self.actions, i, "actions")?,
            first_child: at(&self.first_child, i, "first_child")?,
            offset: at(&self.offset, i, "offset")?,
            actor: at(&self.actor, i, "actor")?,
        })
    }

    /// Children of node `i` (empty for terminals).
    fn children(&self, i: usize) -> Result<Range<usize>, GpuError> {
        let node = self.node(i)?;
        let first = node.first_child as usize;
        Ok(first..first + node.actions as usize)
    }

    /// The same plan with no opponent node fused (every opponent node's row
    /// written by `opponent_sum`), for comparing the two.
    #[cfg(test)]
    pub(crate) fn unfused(mut self) -> Plan {
        for pp in &mut self.players {
            pp.fused.iter_mut().for_each(|f| *f = 0);
            for lv in &mut pp.levels {
                lv.opp_sum_count = lv.opp_count;
            }
        }
        self
    }

    /// Checks every invariant the kernels rely on.
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        let nn = self.num_nodes as usize;
        for (name, len) in [
            ("actions", self.actions.len()),
            ("first_child", self.first_child.len()),
            ("offset", self.offset.len()),
            ("scale_slot", self.scale_slot.len()),
            ("actor", self.actor.len()),
            ("card", self.card.len()),
            ("chance_factor", self.chance_factor.len()),
            ("iso_start", self.iso_start.len()),
            ("iso_count", self.iso_count.len()),
            ("scalar", self.scalar.len()),
        ] {
            ensure(len == nn, || format!("{name} has {len} entries for {nn} nodes"))?;
        }
        ensure(self.iso_child.len() == self.iso_pair.len(), || "isomorphic card arrays".into())?;
        ensure(nn > 0 && self.actor.first().is_some_and(|&a| a < 2), || {
            "root must be a decision node".into()
        })?;
        let hands = [self.players[0].hands, self.players[1].hands];
        let mut scale_used = vec![false; self.num_scale_slots as usize];
        for i in 0..nn {
            let node = self.node(i)?;
            let scale_slot = at(&self.scale_slot, i, "scale_slot")?;
            if node.actor < 2 {
                let used = scale_used
                    .get_mut(scale_slot as usize)
                    .ok_or_else(|| invalid(format!("node {i}: scale slot {scale_slot} out of range")))?;
                ensure(!*used, || format!("node {i}: scale slot {scale_slot} shared"))?;
                *used = true;
                ensure(node.actions <= MAX_ACTIONS, || format!("node {i}: {} actions", node.actions))?;
            } else {
                ensure(scale_slot == NONE, || format!("node {i}: scale slot on a non-decision node"))?;
            }
            let card = at(&self.card, i, "card")?;
            ensure(card == NONE || (card as usize) < CARDS, || format!("node {i}: card {card}"))?;
            let (iso_start, iso_count) =
                (at(&self.iso_start, i, "iso_start")?, at(&self.iso_count, i, "iso_count")?);
            if node.actor != CHANCE {
                ensure(iso_count == 0, || format!("node {i}: isomorphic cards on a non-chance node"))?;
            }
            let scalar = at(&self.scalar, i, "scalar")?;
            ensure(scalar == 0 || (scalar == 1 && node.actor == NONE), || {
                format!("node {i}: scalar flag {scalar}")
            })?;
            if node.actor == NONE {
                ensure(node.actions == 0, || format!("terminal {i} has actions"))?;
                continue;
            }
            ensure(node.actions >= 1, || format!("node {i} has no children"))?;
            let end = u64::from(node.first_child) + u64::from(node.actions);
            ensure(node.first_child as usize > i && end <= nn as u64, || {
                format!("node {i}: children out of range")
            })?;
            if node.actor == CHANCE {
                let factor = at(&self.chance_factor, i, "chance_factor")?;
                ensure(factor.is_finite() && factor > 0.0, || format!("chance node {i}: factor {factor}"))?;
                for c in self.children(i)? {
                    let card = at(&self.card, c, "card")?;
                    ensure((card as usize) < CARDS, || format!("chance node {i}: child {c} has no card"))?;
                }
                let iso_end = u64::from(iso_start) + u64::from(iso_count);
                ensure(iso_end <= self.iso_child.len() as u64, || {
                    format!("chance node {i}: isomorphic cards")
                })?;
                for e in iso_start as usize..iso_end as usize {
                    ensure(at(&self.iso_child, e, "iso_child")? < node.actions, || {
                        format!("chance node {i}: isomorphic card {e} refers to a missing child")
                    })?;
                    ensure((at(&self.iso_pair, e, "iso_pair")? as usize) < SUIT_PAIRS, || {
                        format!("chance node {i}: isomorphic card {e} has a bad suit pair")
                    })?;
                }
                continue;
            }
            let player_hands = *hands
                .get(node.actor as usize)
                .ok_or_else(|| invalid(format!("node {i}: actor {}", node.actor)))?;
            let cells = u64::from(node.actions) * u64::from(player_hands);
            ensure(u64::from(node.offset) + cells <= u64::from(self.storage_len), || {
                format!("node {i}: storage block out of range")
            })?;
        }
        schedule::validate_segments(self)?;
        for (p, pp) in self.players.iter().enumerate() {
            self.validate_player(p, pp).map_err(|e| match e {
                GpuError::InvalidInput(s) => GpuError::InvalidInput(format!("player {p}: {s}")),
                other => other,
            })?;
        }
        Ok(())
    }

    fn validate_player(&self, p: usize, pp: &PlayerPlan) -> Result<(), GpuError> {
        let other = self.players.get(p ^ 1).ok_or_else(|| invalid("player out of range"))?;
        let (n, nq) = (pp.hands, pp.opp_hands);
        ensure(n == other.opp_hands && nq == other.hands, || "hand counts disagree".into())?;
        ensure(n >= 1 && nq >= 1, || "empty range".into())?;
        ensure(n <= NONE / 2 && nq <= NONE / 2, || "too many hands".into())?;
        schedule::validate_player_schedule(self, p, pp)?;
        // Terminal arrays line up, and a paired fold shares its showdown's
        // opponent reach.
        ensure(pp.fold_payoff.len() == pp.folds.len(), || "fold payoffs".into())?;
        let ns = pp.showdowns.len();
        ensure(
            pp.showdown_table.len() == ns
                && pp.win.len() == ns
                && pp.lose.len() == ns
                && pp.tie.len() == ns
                && pp.showdown_fold.len() == ns
                && pp.showdown_fold_payoff.len() == ns,
            || "showdown arrays".into(),
        )?;
        for (&showdown, &fold) in pp.showdowns.iter().zip(&pp.showdown_fold) {
            if fold != NONE {
                ensure(
                    at(&pp.slot, fold as usize, "slot")? == at(&pp.slot, showdown as usize, "slot")?,
                    || format!("fold {fold} paired with showdown {showdown} has another reach"),
                )?;
            }
        }
        ensure(pp.showdown_table.iter().all(|&t| t < self.num_tables), || {
            "showdown table out of range".into()
        })?;
        // Finite, and small enough that double-single products (payoff times
        // a sum of at most 1326 reach values) stay far inside f32's range.
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite() && x.abs() <= MAX_PAYOFF);
        ensure(
            finite(&pp.fold_payoff)
                && finite(&pp.win)
                && finite(&pp.lose)
                && finite(&pp.tie)
                && finite(&pp.showdown_fold_payoff),
            || "non-finite payoff".into(),
        )?;

        // Hands.
        ensure(pp.my_cards.len() == n as usize && pp.same.len() == n as usize, || "hand arrays".into())?;
        ensure(pp.opp_cards.len() == nq as usize, || "opponent hand array".into())?;
        for &packed in pp.my_cards.iter().chain(&pp.opp_cards) {
            let (a, b) = unpack_cards(packed);
            ensure(packed >> 16 == 0 && a < CARDS && b < CARDS && a != b, || {
                format!("bad hand {packed:#x}")
            })?;
        }
        // Permutations: every row an isomorphic card uses maps hands to hands.
        ensure(pp.perm.len() == SUIT_PAIRS * n as usize, || "permutation table".into())?;
        for &pair in &self.iso_pair {
            let start = pair as usize * n as usize;
            let row = pp.perm.get(start..start + n as usize).ok_or_else(|| invalid("permutation row"))?;
            ensure(row.iter().all(|&j| j < n), || format!("suit pair {pair}: permutation out of range"))?;
        }
        ensure(pp.same.iter().all(|&j| j == NONE || j < nq), || "identical-combo index out of range".into())?;
        ensure(pp.opp_weights.len() == nq as usize, || "opponent weights".into())?;
        ensure(pp.opp_weights.iter().all(|w| w.is_finite() && (0.0..=MAX_WEIGHT).contains(w)), || {
            "bad opponent weight".into()
        })?;
        check_offsets(&pp.opp_card_off, pp.opp_card_hands.len(), "opponent card lists")?;
        ensure(pp.opp_card_hands.iter().all(|&j| j < nq), || "opponent card list entry out of range".into())?;

        pp.tables.validate(self.num_tables, &pp.my_cards, pp.opp_hands)
    }
}

impl Tables {
    /// Checks the tables of `num_tables` boards for a traverser with hands
    /// `my_cards` (packed) against an opponent with `opp_hands` hands.
    pub(super) fn validate(&self, num_tables: u32, my_cards: &[u32], opp_hands: u32) -> Result<(), GpuError> {
        let t = self;
        let nt = num_tables as usize;
        let n = my_cards.len();
        ensure(t.opp_off.len() == nt && t.opp_len.len() == nt && t.card_base.len() == nt, || {
            "table arrays".into()
        })?;
        ensure(t.card_off.len() == nt * (CARDS + 1), || "table card offsets".into())?;
        ensure(t.ranges.len() == nt * n && t.ranks.len() == nt * n, || "table hand arrays".into())?;
        let shared = showdown_shared_bytes(t.max_opp);
        ensure(shared <= u64::from(SHOWDOWN_SHARED_LIMIT), || {
            format!("showdown needs {shared} bytes of shared memory")
        })?;
        for k in 0..nt {
            let off = at(&t.opp_off, k, "opp_off")? as usize;
            let len = at(&t.opp_len, k, "opp_len")? as usize;
            ensure(len <= t.max_opp as usize && len < 0xFFFF, || format!("table {k}: {len} opponent hands"))?;
            let order = t
                .opp_order
                .get(off..off + len)
                .ok_or_else(|| invalid(format!("table {k}: opponent list out of range")))?;
            ensure(order.iter().all(|&j| j < opp_hands), || {
                format!("table {k}: opponent hand out of range")
            })?;
            let card_off = t
                .card_off
                .get(k * (CARDS + 1)..(k + 1) * (CARDS + 1))
                .ok_or_else(|| invalid("table card offsets"))?;
            check_offsets(card_off, 2 * len, "table card lists")?;
            let base = at(&t.card_base, k, "card_base")? as usize;
            let pos = t
                .card_pos
                .get(base..base + 2 * len)
                .ok_or_else(|| invalid(format!("table {k}: card positions out of range")))?;
            let segment = |c: usize| -> Result<&[u32], GpuError> {
                let (b, e) =
                    (at(card_off, c, "card offset")? as usize, at(card_off, c + 1, "card offset")? as usize);
                pos.get(b..e).ok_or_else(|| invalid(format!("table {k}: card {c} list out of range")))
            };
            for c in 0..CARDS {
                let seg = segment(c)?;
                ensure(seg.len() <= 0xFF, || format!("table {k}: card {c} has {} hands", seg.len()))?;
                ensure(seg.iter().all(|&j| (j as usize) < len), || {
                    format!("table {k}: card position out of range")
                })?;
                ensure(seg.windows(2).all(|w| w.first() < w.get(1)), || {
                    format!("table {k}: card {c} list not ascending")
                })?;
            }
            for h in 0..n {
                let packed = at(&t.ranges, k * n + h, "ranges")?;
                if packed == NONE {
                    continue;
                }
                let (lo, hi) = ((packed & 0xFFFF) as usize, (packed >> 16) as usize);
                ensure(lo <= hi && hi <= len, || {
                    format!("table {k}, hand {h}: strength range out of order")
                })?;
                let r = at(&t.ranks, k * n + h, "ranks")?;
                let (c1, c2) = unpack_cards(at(my_cards, h, "my_cards")?);
                let (len1, len2) = (segment(c1)?.len() as u32, segment(c2)?.len() as u32);
                let (r1l, r1h, r2l, r2h) = (r & 0xFF, r >> 8 & 0xFF, r >> 16 & 0xFF, r >> 24);
                ensure(r1l <= r1h && r1h <= len1 && r2l <= r2h && r2h <= len2, || {
                    format!("table {k}, hand {h}: card ranks out of range")
                })?;
            }
        }
        Ok(())
    }
}

impl PlayerPlan {
    /// Player `p`'s hand data and showdown tables; the schedule comes
    /// later ([`PlayerPlan::build_schedule`]), once the tree is segmented.
    fn build(ctx: &Ctx, plan: &Plan, p: usize) -> Result<PlayerPlan, GpuError> {
        let hs = ctx.hands;
        let o = p ^ 1;
        let _ = plan;
        let mut pp = PlayerPlan {
            hands: to_u32(hs.len(p), "hands")?,
            opp_hands: to_u32(hs.len(o), "hands")?,
            ..PlayerPlan::default()
        };
        pp.my_cards = side(&hs.cards, p)?.iter().map(|&(a, b)| pack_cards(a, b)).collect();
        pp.opp_cards = side(&hs.cards, o)?.iter().map(|&(a, b)| pack_cards(a, b)).collect();
        pp.same = side(&hs.same, p)?.clone();
        let n = hs.len(p);
        pp.perm = vec![NONE; SUIT_PAIRS * n];
        for (pair, perm) in side(&ctx.iso.perm, p)?.iter().enumerate().take(SUIT_PAIRS) {
            if perm.len() == n {
                let row =
                    pp.perm.get_mut(pair * n..(pair + 1) * n).ok_or_else(|| invalid("permutation row"))?;
                for (dst, &src) in row.iter_mut().zip(perm) {
                    *dst = u32::from(src);
                }
            }
        }
        pp.opp_weights = side(&hs.weights, o)?.clone();
        let mut off = Vec::with_capacity(CARDS + 1);
        let mut list = Vec::with_capacity(2 * hs.len(o));
        for hands in side(&hs.blocked, o)?.iter().take(CARDS) {
            off.push(to_u32(list.len(), "card lists")?);
            list.extend_from_slice(hands);
        }
        off.push(to_u32(list.len(), "card lists")?);
        pp.opp_card_off = off;
        pp.opp_card_hands = list;
        pp.tables = Tables::build(ctx.tables, hs, p)?;
        Ok(pp)
    }
}

/// One hand of a strength-sorted list (weakest first).
#[derive(Clone, Copy, Debug)]
pub(super) struct Ranked {
    pub strength: u32,
    /// Index in its player's hand list.
    pub idx: u32,
    pub c1: u8,
    pub c2: u8,
}

impl Tables {
    pub(crate) fn build(tables: &[StrengthTable], hs: &HandSet, p: usize) -> Result<Tables, GpuError> {
        let o = p ^ 1;
        let my_cards = side(&hs.cards, p)?;
        let ranked =
            |e: &crate::eval::Entry| Ranked { strength: e.strength, idx: u32::from(e.idx), c1: e.c1, c2: e.c2 };
        let mut t = Tables::default();
        for table in tables {
            t.push(side(&table.lists, p)?, side(&table.lists, o)?, my_cards, ranked)?;
        }
        Ok(t)
    }

    /// Appends the table of one board: the traverser's hands `mine` and the
    /// opponent's `theirs` (both sorted by strength, read through `ranked`),
    /// for a traverser whose hands hold `my_cards`.
    pub(super) fn push<E>(
        &mut self,
        mine: &[E],
        theirs: &[E],
        my_cards: &[(u8, u8)],
        ranked: impl Fn(&E) -> Ranked,
    ) -> Result<(), GpuError> {
        let t = self;
        let n = my_cards.len();
        let len = to_u32(theirs.len(), "table opponent list")?;
        t.opp_off.push(to_u32(t.opp_order.len(), "table opponent lists")?);
        t.opp_len.push(len);
        t.max_opp = t.max_opp.max(len);
        t.opp_order.extend(theirs.iter().map(|e| ranked(e).idx));

        let mut per_card: Vec<Vec<u32>> = vec![Vec::new(); CARDS];
        for (j, e) in theirs.iter().enumerate() {
            let j = to_u32(j, "table position")?;
            let e = ranked(e);
            for c in [e.c1, e.c2] {
                per_card.get_mut(c as usize).ok_or_else(|| invalid("card out of range"))?.push(j);
            }
        }
        t.card_base.push(to_u32(t.card_pos.len(), "table card positions")?);
        let mut off = 0u32;
        for list in &per_card {
            t.card_off.push(off);
            t.card_pos.extend_from_slice(list);
            off += to_u32(list.len(), "card list")?;
        }
        t.card_off.push(off);

        let base = t.ranges.len();
        t.ranges.resize(base + n, NONE);
        t.ranks.resize(base + n, 0);
        let rank = |c: u8, k: usize| -> Result<u32, GpuError> {
            let list = per_card.get(c as usize).ok_or_else(|| invalid("card out of range"))?;
            to_u32(list.partition_point(|&j| (j as usize) < k), "card rank")
        };
        for e in mine.iter().map(&ranked) {
            let h = e.idx as usize;
            let lo = theirs.partition_point(|x| ranked(x).strength < e.strength);
            let hi = theirs.partition_point(|x| ranked(x).strength <= e.strength);
            ensure(hi < 0xFFFF, || "opponent list too long".into())?;
            let &(c1, c2) = my_cards.get(h).ok_or_else(|| invalid("table hand out of range"))?;
            let ranks = [rank(c1, lo)?, rank(c1, hi)?, rank(c2, lo)?, rank(c2, hi)?];
            ensure(ranks.iter().all(|&r| r <= 0xFF), || "card list longer than 255".into())?;
            let (Some(range), Some(packed)) = (t.ranges.get_mut(base + h), t.ranks.get_mut(base + h)) else {
                return Err(invalid("table hand out of range"));
            };
            *range = lo as u32 | (hi as u32) << 16;
            *packed = ranks[0] | ranks[1] << 8 | ranks[2] << 16 | ranks[3] << 24;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
