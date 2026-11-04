//! Host-side plan of a GPU traversal: the game tree flattened into the
//! arrays the kernels index, a level schedule for each traverser, terminal
//! lists with their payoffs, and per-board showdown lookup tables.
//!
//! Built once per solve. [`Plan::validate`] checks every invariant the
//! kernels rely on (all indices in range, sizes fit the kernels' `u32`
//! indexing, a consistent schedule), so a malformed plan is an error here
//! rather than an out-of-bounds access on the device.

use super::{invalid, GpuError};
use crate::cfr::{fold_payoff, showdown_payoffs, Ctx};
use crate::hands::{HandSet, StrengthTable};
use crate::tree::{Edge, Kind};
use std::ops::Range;

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
/// lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LevelRange {
    pub opp_start: u32,
    pub opp_count: u32,
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
    /// Top-down order; bottom-up walks it backwards.
    pub levels: Vec<LevelRange>,
    pub opp_nodes: Vec<u32>,
    pub own_nodes: Vec<u32>,
    pub chance_nodes: Vec<u32>,
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

fn to_u32(x: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(x).map_err(|_| invalid(format!("{what} ({x}) exceeds the kernels' u32 indexing")))
}

fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), GpuError> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidInput(msg()))
    }
}

/// `v[i]`, or an error naming `what`.
fn at<T: Copy>(v: &[T], i: usize, what: &str) -> Result<T, GpuError> {
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

fn unpack_cards(packed: u32) -> (usize, usize) {
    ((packed & 0xFF) as usize, (packed >> 8 & 0xFF) as usize)
}

/// `off` is a 53-entry offset array starting at 0, non-decreasing and
/// ending at `total`.
fn check_offsets(off: &[u32], total: usize, what: &str) -> Result<(), GpuError> {
    ensure(off.len() == CARDS + 1, || format!("{what}: {} offsets", off.len()))?;
    ensure(off.first() == Some(&0) && off.windows(2).all(|w| w.first() <= w.get(1)), || {
        format!("{what}: offsets out of order")
    })?;
    ensure(off.last().is_some_and(|&e| e as usize == total), || {
        format!("{what}: offsets don't end at {total}")
    })
}

impl Plan {
    /// Flattens the tree behind `ctx`.
    pub(crate) fn build(ctx: &Ctx) -> Result<Plan, GpuError> {
        let tree = ctx.tree;
        let num_nodes = to_u32(tree.nodes.len(), "tree nodes")?;
        let storage_len = u32::try_from(tree.storage_len)
            .map_err(|_| invalid(format!("storage of {} values exceeds u32 indexing", tree.storage_len)))?;
        let mut plan = Plan {
            num_nodes,
            storage_len,
            num_tables: to_u32(ctx.tables.len(), "strength tables")?,
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
        plan.players =
            [PlayerPlan::build(ctx, &plan, &levels, 0)?, PlayerPlan::build(ctx, &plan, &levels, 1)?];
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
            + b(&self.card)
            + b(&self.chance_factor)
            + b(&self.iso_start)
            + b(&self.iso_count)
            + b(&self.iso_child)
            + b(&self.iso_pair)
            + b(&self.scalar);
        let player = |pp: &PlayerPlan| {
            let t = &pp.tables;
            b(&pp.slot)
                + b(&pp.child_slot)
                + b(&pp.opp_nodes)
                + b(&pp.own_nodes)
                + b(&pp.chance_nodes)
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

    /// Checks every invariant the kernels rely on.
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        let nn = self.num_nodes as usize;
        for (name, len) in [
            ("actions", self.actions.len()),
            ("first_child", self.first_child.len()),
            ("offset", self.offset.len()),
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
        for i in 0..nn {
            let node = self.node(i)?;
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
        for (p, pp) in self.players.iter().enumerate() {
            self.validate_player(p, pp).map_err(|e| match e {
                GpuError::InvalidInput(s) => GpuError::InvalidInput(format!("player {p}: {s}")),
                other => other,
            })?;
        }
        Ok(())
    }

    fn validate_player(&self, p: usize, pp: &PlayerPlan) -> Result<(), GpuError> {
        let nn = self.num_nodes as usize;
        let other = self.players.get(p ^ 1).ok_or_else(|| invalid("player out of range"))?;
        let (n, nq) = (pp.hands, pp.opp_hands);
        ensure(n == other.opp_hands && nq == other.hands, || "hand counts disagree".into())?;
        ensure(n >= 1 && nq >= 1, || "empty range".into())?;
        ensure(n <= NONE / 2 && nq <= NONE / 2, || "too many hands".into())?;
        // Buffers the kernels index with u32.
        to_u32(pp.num_slots as usize * nq as usize, "reach buffer")?;
        to_u32(nn * n as usize, "value buffer")?;
        ensure(pp.slot.len() == nn && pp.child_slot.len() == nn, || {
            "slot arrays have the wrong length".into()
        })?;
        ensure(pp.slot.first() == Some(&0), || "root must use slot 0".into())?;

        // Slots: in range, and children consistent with their parent's role.
        for i in 0..nn {
            let slot = at(&pp.slot, i, "slot")?;
            ensure(slot < pp.num_slots, || format!("node {i}: slot out of range"))?;
            let node = self.node(i)?;
            if node.actor == NONE {
                continue;
            }
            let child_slot = at(&pp.child_slot, i, "child_slot")?;
            if node.actor < 2 && node.actor as usize == p {
                ensure(child_slot == NONE, || format!("own node {i} has child slots"))?;
                for c in self.children(i)? {
                    ensure(at(&pp.slot, c, "slot")? == slot, || {
                        format!("own node {i}: child {c} doesn't share its slot")
                    })?;
                }
            } else {
                let end = u64::from(child_slot) + u64::from(node.actions);
                ensure(child_slot != NONE && end <= u64::from(pp.num_slots), || {
                    format!("opponent or chance node {i}: child slots out of range")
                })?;
                for (a, c) in self.children(i)?.enumerate() {
                    ensure(u64::from(at(&pp.slot, c, "slot")?) == u64::from(child_slot) + a as u64, || {
                        format!("opponent or chance node {i}: child {c} has the wrong slot")
                    })?;
                }
            }
        }

        // Levels: every non-terminal node exactly once, in the list of its
        // role, and non-terminal children exactly one level deeper.
        let role = |actor: u32| -> Role {
            match actor {
                CHANCE => Role::Chance,
                a if a as usize == p => Role::Own,
                _ => Role::Opponent,
            }
        };
        let mut level_of = vec![NONE; nn];
        let (mut expect_opp, mut expect_own, mut expect_chance) = (0u32, 0u32, 0u32);
        for (d, lv) in pp.levels.iter().enumerate() {
            ensure(
                lv.opp_start == expect_opp && lv.own_start == expect_own && lv.chance_start == expect_chance,
                || format!("level {d} isn't contiguous"),
            )?;
            expect_opp = lv.opp_start.checked_add(lv.opp_count).ok_or_else(|| invalid("level overflow"))?;
            expect_own = lv.own_start.checked_add(lv.own_count).ok_or_else(|| invalid("level overflow"))?;
            expect_chance =
                lv.chance_start.checked_add(lv.chance_count).ok_or_else(|| invalid("level overflow"))?;
            let opp = pp.opp_nodes.get(lv.opp_start as usize..expect_opp as usize);
            let own = pp.own_nodes.get(lv.own_start as usize..expect_own as usize);
            let chance = pp.chance_nodes.get(lv.chance_start as usize..expect_chance as usize);
            let (Some(opp), Some(own), Some(chance)) = (opp, own, chance) else {
                return Err(invalid(format!("level {d} out of range")));
            };
            let listed = opp
                .iter()
                .map(|x| (x, Role::Opponent))
                .chain(own.iter().map(|x| (x, Role::Own)))
                .chain(chance.iter().map(|x| (x, Role::Chance)));
            for (&id, list_role) in listed {
                let i = id as usize;
                let actor = self.node(i)?.actor;
                ensure(actor != NONE && role(actor) == list_role, || format!("node {id} in the wrong list"))?;
                let entry = level_of.get_mut(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                ensure(*entry == NONE, || format!("node {id} scheduled twice"))?;
                *entry = d as u32;
            }
        }
        ensure(
            expect_opp as usize == pp.opp_nodes.len()
                && expect_own as usize == pp.own_nodes.len()
                && expect_chance as usize == pp.chance_nodes.len(),
            || "node lists have unscheduled entries".into(),
        )?;

        // Terminals: each exactly once, with its payoffs.
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
        // A paired fold shares its showdown's opponent reach.
        for (&showdown, &fold) in pp.showdowns.iter().zip(&pp.showdown_fold) {
            if fold != NONE {
                ensure(
                    at(&pp.slot, fold as usize, "slot")? == at(&pp.slot, showdown as usize, "slot")?,
                    || format!("fold {fold} paired with showdown {showdown} has another reach"),
                )?;
            }
        }
        let paired = pp.showdown_fold.iter().filter(|&&f| f != NONE);
        let mut terminal_seen = vec![false; nn];
        for &id in pp.folds.iter().chain(&pp.showdowns).chain(paired) {
            let i = id as usize;
            ensure(self.node(i)?.actor == NONE, || format!("terminal list entry {id} isn't a terminal"))?;
            let seen =
                terminal_seen.get_mut(i).ok_or_else(|| invalid(format!("terminal {id} out of range")))?;
            ensure(!*seen, || format!("terminal {id} listed twice"))?;
            *seen = true;
        }
        for i in 0..nn {
            let level = at(&level_of, i, "level")?;
            if self.node(i)?.actor == NONE {
                ensure(at(&terminal_seen, i, "terminal")?, || format!("terminal {i} not listed"))?;
                continue;
            }
            ensure(level != NONE, || format!("decision node {i} not scheduled"))?;
            for c in self.children(i)? {
                if self.node(c)?.actor != NONE {
                    ensure(at(&level_of, c, "level")? == level + 1, || {
                        format!("node {c} isn't one level below its parent {i}")
                    })?;
                }
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

        self.validate_tables(pp)
    }

    fn validate_tables(&self, pp: &PlayerPlan) -> Result<(), GpuError> {
        let t = &pp.tables;
        let nt = self.num_tables as usize;
        let n = pp.hands as usize;
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
            ensure(order.iter().all(|&j| j < pp.opp_hands), || {
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
                let (c1, c2) = unpack_cards(at(&pp.my_cards, h, "my_cards")?);
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
    fn build(ctx: &Ctx, plan: &Plan, levels: &[Vec<u32>], p: usize) -> Result<PlayerPlan, GpuError> {
        let hs = ctx.hands;
        let o = p ^ 1;
        let nn = plan.num_nodes as usize;
        let mut pp = PlayerPlan {
            hands: to_u32(hs.len(p), "hands")?,
            opp_hands: to_u32(hs.len(o), "hands")?,
            slot: vec![NONE; nn],
            child_slot: vec![NONE; nn],
            ..PlayerPlan::default()
        };
        // Slots, top-down. A level only holds children of the previous one,
        // so a node's slot is known before its children are assigned.
        let mut next_slot = 1u64;
        if let Some(root) = pp.slot.first_mut() {
            *root = 0;
        }
        for level in levels {
            let mut range = LevelRange {
                opp_start: to_u32(pp.opp_nodes.len(), "opponent nodes")?,
                own_start: to_u32(pp.own_nodes.len(), "own nodes")?,
                chance_start: to_u32(pp.chance_nodes.len(), "chance nodes")?,
                ..LevelRange::default()
            };
            for &id in level {
                let i = id as usize;
                let node = ctx.tree.nodes.get(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                let parent_slot = at(&pp.slot, i, "slot")?;
                match node.kind {
                    Kind::Player(q) if q as usize == p => {
                        pp.own_nodes.push(id);
                        for c in node.children() {
                            *pp.slot.get_mut(c).ok_or_else(|| invalid("child out of range"))? = parent_slot;
                        }
                    }
                    // Opponent and chance nodes give every child a reach of
                    // its own.
                    Kind::Player(_) | Kind::Chance => {
                        if matches!(node.kind, Kind::Chance) {
                            pp.chance_nodes.push(id);
                        } else {
                            pp.opp_nodes.push(id);
                        }
                        let first = u32::try_from(next_slot).map_err(|_| invalid("too many reach slots"))?;
                        *pp.child_slot.get_mut(i).ok_or_else(|| invalid("node out of range"))? = first;
                        for (a, c) in node.children().enumerate() {
                            *pp.slot.get_mut(c).ok_or_else(|| invalid("child out of range"))? =
                                first.saturating_add(a as u32);
                        }
                        next_slot += u64::from(node.num_children);
                    }
                    Kind::Fold(_) => {
                        pp.folds.push(id);
                        pp.fold_payoff.push(fold_payoff(ctx, node, p));
                    }
                    Kind::Showdown => {
                        let (w, l, t) = showdown_payoffs(ctx, node, p);
                        pp.showdowns.push(id);
                        pp.showdown_table
                            .push(u32::try_from(node.offset).map_err(|_| invalid("strength table index"))?);
                        pp.win.push(w);
                        pp.lose.push(l);
                        pp.tie.push(t);
                    }
                }
            }
            range.opp_count = to_u32(pp.opp_nodes.len(), "opponent nodes")? - range.opp_start;
            range.own_count = to_u32(pp.own_nodes.len(), "own nodes")? - range.own_start;
            range.chance_count = to_u32(pp.chance_nodes.len(), "chance nodes")? - range.chance_start;
            pp.levels.push(range);
        }
        pp.num_slots = u32::try_from(next_slot).map_err(|_| invalid("too many reach slots"))?;
        pp.pair_folds_with_showdowns(ctx, plan, p)?;

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

impl PlayerPlan {
    /// Moves each fold that shares its opponent reach with a showdown (the
    /// fold and the call, both children of one of the traverser's nodes)
    /// into that showdown's `showdown_fold`, like the CPU's `showdown_batch`
    /// grouping. Only batched terminals are paired: the CPU evaluates the
    /// others one at a time.
    fn pair_folds_with_showdowns(&mut self, ctx: &Ctx, plan: &Plan, p: usize) -> Result<(), GpuError> {
        let nodes = &ctx.tree.nodes;
        self.showdown_fold = vec![NONE; self.showdowns.len()];
        self.showdown_fold_payoff = vec![0.0; self.showdowns.len()];
        let mut showdown_index = vec![NONE; nodes.len()];
        for (k, &id) in self.showdowns.iter().enumerate() {
            *showdown_index.get_mut(id as usize).ok_or_else(|| invalid("showdown out of range"))? =
                to_u32(k, "showdowns")?;
        }
        let mut paired = vec![false; nodes.len()];
        for node in nodes.iter().filter(|n| matches!(n.kind, Kind::Player(q) if q as usize == p)) {
            let batched = |c: usize| at(&plan.scalar, c, "scalar").is_ok_and(|s| s == 0);
            let fold =
                node.children().find(|&c| matches!(nodes.get(c), Some(n) if matches!(n.kind, Kind::Fold(_))));
            let showdown = node
                .children()
                .find(|&c| matches!(nodes.get(c), Some(n) if matches!(n.kind, Kind::Showdown)));
            let (Some(fold), Some(showdown)) = (fold, showdown) else { continue };
            if !batched(fold) || !batched(showdown) {
                continue;
            }
            let k = at(&showdown_index, showdown, "showdown index")? as usize;
            let fold_node = nodes.get(fold).ok_or_else(|| invalid("fold out of range"))?;
            *self.showdown_fold.get_mut(k).ok_or_else(|| invalid("showdown out of range"))? =
                to_u32(fold, "fold")?;
            *self.showdown_fold_payoff.get_mut(k).ok_or_else(|| invalid("showdown out of range"))? =
                fold_payoff(ctx, fold_node, p);
            *paired.get_mut(fold).ok_or_else(|| invalid("fold out of range"))? = true;
        }
        let mut folds = Vec::with_capacity(self.folds.len());
        let mut payoffs = Vec::with_capacity(self.folds.len());
        for (&id, &pay) in self.folds.iter().zip(&self.fold_payoff) {
            if !at(&paired, id as usize, "paired")? {
                folds.push(id);
                payoffs.push(pay);
            }
        }
        self.folds = folds;
        self.fold_payoff = payoffs;
        Ok(())
    }
}

impl Tables {
    pub(crate) fn build(tables: &[StrengthTable], hs: &HandSet, p: usize) -> Result<Tables, GpuError> {
        let o = p ^ 1;
        let n = hs.len(p);
        let mut t = Tables::default();
        for table in tables {
            let (mine, theirs) = (side(&table.lists, p)?, side(&table.lists, o)?);
            let len = to_u32(theirs.len(), "table opponent list")?;
            t.opp_off.push(to_u32(t.opp_order.len(), "table opponent lists")?);
            t.opp_len.push(len);
            t.max_opp = t.max_opp.max(len);
            t.opp_order.extend(theirs.iter().map(|e| u32::from(e.idx)));

            let mut per_card: Vec<Vec<u32>> = vec![Vec::new(); CARDS];
            for (j, e) in theirs.iter().enumerate() {
                let j = to_u32(j, "table position")?;
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
            for e in mine {
                let h = e.idx as usize;
                let lo = theirs.partition_point(|x| x.strength < e.strength);
                let hi = theirs.partition_point(|x| x.strength <= e.strength);
                ensure(hi < 0xFFFF, || "opponent list too long".into())?;
                let &(c1, c2) =
                    side(&hs.cards, p)?.get(h).ok_or_else(|| invalid("table hand out of range"))?;
                let ranks = [rank(c1, lo)?, rank(c1, hi)?, rank(c2, lo)?, rank(c2, hi)?];
                ensure(ranks.iter().all(|&r| r <= 0xFF), || "card list longer than 255".into())?;
                let (Some(range), Some(packed)) = (t.ranges.get_mut(base + h), t.ranks.get_mut(base + h))
                else {
                    return Err(invalid("table hand out of range"));
                };
                *range = lo as u32 | (hi as u32) << 16;
                *packed = ranks[0] | ranks[1] << 8 | ranks[2] << 16 | ranks[3] << 24;
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests;
