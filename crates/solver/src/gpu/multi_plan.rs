//! Host-side plan of a three-player GPU traversal (`solver::multi`): the
//! tree flattened into the arrays the kernels index, a level schedule per
//! traverser with one reach slot per opponent, terminal lists with their
//! payoffs, and per-opponent showdown lookup tables (independent removal).
//!
//! What a traversal by player `p` does, and so what the plan encodes (the
//! CPU twin is `multi::cfr::traverse`, with `multi::river::subtree` for
//! batched river subtrees):
//! * it carries the reach of `p`'s two opponents `opponents(p) = (q, r)`,
//!   each in its own **slot space** (`slot[k]`, `k` = 0 for `q`, 1 for `r`):
//!   at `q`'s nodes only `q`'s reach splits (each child gets a fresh `q`
//!   slot, the `r` slot passes through), and likewise for `r`; at a chance
//!   node both split (blocked hands zeroed, the chance factor applied to
//!   `q`'s reach only, since values are bilinear in the two); at `p`'s own
//!   nodes both pass through;
//! * a node at which `p` has folded is a **leaf** worth a fixed payoff
//!   times the opponent pair mass (`pay_all`), even if it has children:
//!   nothing below it is scheduled for `p`;
//! * player `x`'s average strategy at a node is accumulated in the
//!   traversal of `strategy_updater(node, x)`, the next player after `x`
//!   still in the hand (`strategy_update` flags);
//! * a node either opponent can't reach is dead (a run-time property of
//!   the reach, tracked by the kernels as in the heads-up port).
//!
//! The plan is a single segment: tree splitting, fused opponent nodes and
//! grouping fixed-payoff leaves by reach pair (as `river::subtree` does)
//! come later. [`MultiPlan::validate`] checks every invariant the kernels
//! will rely on, so a malformed plan is an error here rather than an
//! out-of-bounds access on the device.

use super::plan::{
    at, check_offsets, ensure, pack_cards, to_u32, unpack_cards, Ranked, Span, Tables, CARDS, MAX_ACTIONS,
    MAX_PAYOFF, MAX_WEIGHT, NONE,
};
use super::{invalid, GpuError};
use crate::multi::cfr::{payoffs, strategy_updater, Ctx};
use crate::multi::tree::{Edge, Kind, Node};
use crate::multi::{opponents, N};
use std::ops::Range;

/// `MultiPlan::actor` of a chance node (decision nodes hold the player,
/// 0, 1 or 2; terminals `NONE`).
pub(crate) const CHANCE: u32 = 3;

/// `MultiPlan::folded` with every player folded.
const ALL_FOLDED: u32 = (1 << N) - 1;

/// One tree level of one traverser's schedule: ranges into the per-opponent
/// opponent-node lists, the own-node list and the chance-node list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MultiLevel {
    /// Into `opp_nodes[k]`, for opponent `k` of `opponents(p)`.
    pub opp: [Span; 2],
    pub own: Span,
    pub chance: Span,
}

/// One opponent as one traverser sees it.
#[derive(Clone, Debug, Default)]
pub(crate) struct OpponentView {
    /// The opponent's seat.
    pub player: u32,
    /// The opponent's hand count.
    pub hands: u32,
    /// The opponent's hands' cards, packed `c1 | c2 << 8`.
    pub cards: Vec<u32>,
    /// Per traverser hand: the index of the identical combo in this
    /// opponent's list, or `NONE`.
    pub same: Vec<u32>,
    /// For each card, the range of `card_hands` listing this opponent's
    /// hands that hold it (53 offsets).
    pub card_off: Vec<u32>,
    pub card_hands: Vec<u32>,
    /// The opponent's root weights (its reach at the root).
    pub weights: Vec<f32>,
    /// Showdown lookup of the traverser against this opponent, per board
    /// (layout of the heads-up [`Tables`]): enough for the independent
    /// model's `Table::beat`.
    pub tables: Tables,
}

/// Everything a traversal by one player needs.
#[derive(Clone, Debug, Default)]
pub(crate) struct MultiPlayerPlan {
    /// The traverser's hand count.
    pub hands: u32,
    /// The traverser's hands' cards, packed `c1 | c2 << 8`.
    pub my_cards: Vec<u32>,
    /// `opponents(p)`, in that order.
    pub opponents: [OpponentView; 2],
    /// Per opponent, per node: the slot holding that opponent's reach at
    /// the node; `NONE` for nodes the traversal never reaches (below a
    /// leaf).
    pub slot: [Vec<u32>; 2],
    /// Per opponent, per node: the first child's slot where that
    /// opponent's reach splits (its own nodes and chance nodes; children's
    /// slots are consecutive), `NONE` elsewhere.
    pub child_slot: [Vec<u32>; 2],
    /// Per opponent: slots used (slot 0 is the root's).
    pub num_slots: [u32; 2],
    /// Top-down order (no level is empty); bottom-up walks it backwards.
    pub levels: Vec<MultiLevel>,
    /// Per opponent: that opponent's nodes this traversal expands.
    pub opp_nodes: [Vec<u32>; 2],
    pub own_nodes: Vec<u32>,
    pub chance_nodes: Vec<u32>,
    /// Per node: 1 for an opponent node whose average strategy this
    /// traversal accumulates (`strategy_updater`), 0 elsewhere.
    pub strategy_update: Vec<u32>,
    /// Leaves with a fixed payoff (`pay_all`): uncontested wins, and every
    /// node at which the traverser has folded (terminal or not).
    pub fixed: Vec<u32>,
    pub fixed_payoff: Vec<f64>,
    /// Showdowns the traverser takes part in, their strength table and
    /// payoffs (normalized, `multi::cfr::payoffs`).
    pub showdowns: Vec<u32>,
    pub showdown_table: Vec<u32>,
    pub win: Vec<f64>,
    pub lose: Vec<f64>,
}

/// The flattened three-player tree plus a plan per traverser.
#[derive(Clone, Debug, Default)]
pub(crate) struct MultiPlan {
    pub num_nodes: u32,
    pub storage_len: u32,
    /// Per node: number of children (0 for terminals).
    pub actions: Vec<u32>,
    /// Per node: first child (children are consecutive), 0 for terminals.
    pub first_child: Vec<u32>,
    /// Per node: storage offset of a decision node, 0 elsewhere.
    pub offset: Vec<u32>,
    /// Per node: a decision node's scale index in 16-bit storage, `NONE`
    /// elsewhere. Unique per decision node.
    pub scale_slot: Vec<u32>,
    pub num_scale_slots: u32,
    /// Per node: the acting player of a decision node, [`CHANCE`] for a
    /// chance node, `NONE` for terminals.
    pub actor: Vec<u32>,
    /// Per node: the card dealt to reach it (children of chance nodes),
    /// `NONE` elsewhere.
    pub card: Vec<u32>,
    /// Per node: a chance node's reach multiplier (applied to the first
    /// opponent's reach only), 0 elsewhere.
    pub chance_factor: Vec<f32>,
    /// Per node: the players who have folded, bit `k` for player `k`.
    pub folded: Vec<u32>,
    /// Per node: the player who won uncontested (everyone else folded) at
    /// such a terminal, `NONE` elsewhere (showdowns included).
    pub winner: Vec<u32>,
    /// Per node: 1 if, evaluated as a leaf of a traversal, the CPU
    /// evaluates the node one at a time (`pay_all`, `exact::showdown`), 0
    /// if inside a batched river subtree (`river::subtree`, whose
    /// fixed-payoff leaves round differently: the pair mass is rounded to
    /// f32 before the payoff multiplies it). Batched exactly when river
    /// batching is on and the parent is a river decision node, so it
    /// doesn't depend on the traverser.
    pub scalar: Vec<u32>,
    /// Whether the CPU batches river subtrees (`Ctx::batch_river`).
    pub batch_river: bool,
    /// Exact card removal between the opponents (`Ctx::exact`); off, they
    /// are independent.
    pub exact: bool,
    pub num_tables: u32,
    pub players: [MultiPlayerPlan; N],
}

/// The role of an expanded node in one player's traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Own,
    /// An opponent's node: `k` indexes `opponents(p)`.
    Opponent(usize),
    Chance,
}

/// Which of `p`'s opponents player `x` is (`k` in `opponents(p)`).
fn opponent_index(p: usize, x: usize) -> Option<usize> {
    let (q, r) = opponents(p);
    match x {
        _ if x == q => Some(0),
        _ if x == r => Some(1),
        _ => None,
    }
}

/// `strategy_updater` on the plan's encoding: the next player after `x`
/// who hasn't folded.
fn updater(folded: u32, x: usize) -> usize {
    let a = (x + 1) % N;
    if folded >> a & 1 == 1 {
        (x + 2) % N
    } else {
        a
    }
}

fn folded_mask(node: &Node) -> u32 {
    node.folded.iter().enumerate().fold(0, |m, (k, &f)| m | u32::from(f) << k)
}

/// `v[i] = x`, or an error naming `what`.
fn put(v: &mut [u32], i: usize, x: u32, what: &str) -> Result<(), GpuError> {
    *v.get_mut(i).ok_or_else(|| invalid(format!("{what}[{i}] out of range")))? = x;
    Ok(())
}

/// Element `k` of a per-player or per-opponent array.
fn nth<'a, T>(v: &'a [T], k: usize, what: &str) -> Result<&'a T, GpuError> {
    v.get(k).ok_or_else(|| invalid(format!("{what} {k} out of range")))
}

/// Prefixes an input error with `prefix`.
fn context<T>(prefix: impl Fn() -> String, r: Result<T, GpuError>) -> Result<T, GpuError> {
    r.map_err(|e| match e {
        GpuError::InvalidInput(s) => GpuError::InvalidInput(format!("{}: {s}", prefix())),
        other => other,
    })
}

impl MultiPlan {
    /// Flattens the three-player tree behind `ctx`.
    pub(crate) fn build(ctx: &Ctx) -> Result<MultiPlan, GpuError> {
        let tree = ctx.tree;
        let nodes = &tree.nodes;
        let nn = nodes.len();
        let mut plan = MultiPlan {
            num_nodes: to_u32(nn, "tree nodes")?,
            storage_len: u32::try_from(tree.storage_len).map_err(|_| {
                invalid(format!("storage of {} values exceeds u32 indexing", tree.storage_len))
            })?,
            num_scale_slots: tree.num_player_nodes,
            batch_river: ctx.batch_river,
            exact: ctx.exact,
            num_tables: to_u32(ctx.tables.len(), "strength tables")?,
            ..MultiPlan::default()
        };
        let mut parent = vec![NONE; nn];
        for (i, node) in nodes.iter().enumerate() {
            let (actor, offset, factor, winner) = match node.kind {
                Kind::Player(x) => (
                    u32::from(x),
                    u32::try_from(node.offset).map_err(|_| invalid("storage offset exceeds u32 indexing"))?,
                    0.0,
                    NONE,
                ),
                Kind::Chance => (CHANCE, 0, node.factor, NONE),
                Kind::Won(w) => (NONE, 0, 0.0, u32::from(w)),
                Kind::Showdown => (NONE, 0, 0.0, NONE),
            };
            let terminal = actor == NONE;
            plan.actor.push(actor);
            plan.actions.push(if terminal { 0 } else { u32::from(node.num_children) });
            plan.first_child.push(if terminal { 0 } else { node.first_child });
            plan.offset.push(offset);
            plan.scale_slot.push(if matches!(node.kind, Kind::Player(_)) { node.slot } else { NONE });
            plan.chance_factor.push(factor);
            plan.folded.push(folded_mask(node));
            plan.winner.push(winner);
            plan.card.push(match node.edge {
                Edge::Card(c) => to_u32(c.index(), "card")?,
                Edge::Root | Edge::Action(_) => NONE,
            });
            if !terminal {
                for c in node.children() {
                    put(&mut parent, c, to_u32(i, "nodes")?, "parent")?;
                }
            }
        }
        // Batched leaves: children of river decision nodes, with batching on.
        // (A leaf's parent is expanded, so in a traversal that reaches it the
        // parent is either a batched subtree's root or inside one.)
        for &pa in &parent {
            let batched = ctx.batch_river
                && pa != NONE
                && nodes
                    .get(pa as usize)
                    .is_some_and(|n| matches!(n.kind, Kind::Player(_)) && n.street == holdem::Street::River);
            plan.scalar.push(u32::from(!batched));
        }
        plan.players = [
            MultiPlayerPlan::build(ctx, &plan, 0)?,
            MultiPlayerPlan::build(ctx, &plan, 1)?,
            MultiPlayerPlan::build(ctx, &plan, 2)?,
        ];
        plan.validate()?;
        Ok(plan)
    }

    /// Children of node `i` (empty for terminals).
    fn children(&self, i: usize) -> Result<Range<usize>, GpuError> {
        let first = at(&self.first_child, i, "first_child")? as usize;
        Ok(first..first + at(&self.actions, i, "actions")? as usize)
    }

    /// Per node: its parent (`NONE` for the root), checking that every node
    /// but the root has exactly one.
    fn parents(&self) -> Result<Vec<u32>, GpuError> {
        let nn = self.num_nodes as usize;
        let mut parent = vec![NONE; nn];
        for i in 0..nn {
            for c in self.children(i)? {
                let entry =
                    parent.get_mut(c).ok_or_else(|| invalid(format!("node {i}: child {c} out of range")))?;
                ensure(*entry == NONE, || format!("node {c} has two parents"))?;
                *entry = to_u32(i, "nodes")?;
            }
        }
        for (c, &pa) in parent.iter().enumerate().skip(1) {
            ensure(pa != NONE, || format!("node {c} has no parent"))?;
        }
        Ok(parent)
    }

    /// Whether player `p`'s traversal expands node `i` (a decision or
    /// chance node at which `p` hasn't folded); otherwise a reached node is
    /// a leaf.
    fn expands(&self, i: usize, p: usize) -> Result<bool, GpuError> {
        Ok(at(&self.actor, i, "actor")? != NONE && at(&self.folded, i, "folded")? >> p & 1 == 0)
    }

    /// Per node: whether player `p`'s traversal reaches it.
    fn reached(&self, p: usize) -> Result<Vec<bool>, GpuError> {
        let nn = self.num_nodes as usize;
        let mut reached = vec![false; nn];
        if let Some(root) = reached.first_mut() {
            *root = true;
        }
        // Children have larger indices than their parent (checked first).
        for i in 0..nn {
            if at(&reached, i, "reached")? && self.expands(i, p)? {
                for c in self.children(i)? {
                    *reached.get_mut(c).ok_or_else(|| invalid("child out of range"))? = true;
                }
            }
        }
        Ok(reached)
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
            ("folded", self.folded.len()),
            ("winner", self.winner.len()),
            ("scalar", self.scalar.len()),
        ] {
            ensure(len == nn, || format!("{name} has {len} entries for {nn} nodes"))?;
        }
        ensure(nn > 0 && self.actor.first().is_some_and(|&a| (a as usize) < N), || {
            "root must be a decision node".into()
        })?;
        let hands: Vec<u32> = self.players.iter().map(|pp| pp.hands).collect();
        let mut scale_used = vec![false; self.num_scale_slots as usize];
        for i in 0..nn {
            context(|| format!("node {i}"), self.validate_node(i, &hands, &mut scale_used))?;
        }
        let parent = self.parents()?;
        for i in 0..nn {
            context(|| format!("node {i}"), self.validate_edges(i, &parent))?;
        }
        for p in 0..N {
            context(|| format!("player {p}"), self.validate_player(p))?;
        }
        // Each decision node's average strategy is accumulated in exactly
        // one traversal: its strategy updater's.
        for i in 0..nn {
            let actor = at(&self.actor, i, "actor")? as usize;
            if actor >= N {
                continue;
            }
            let u = updater(at(&self.folded, i, "folded")?, actor);
            for (p, pp) in self.players.iter().enumerate() {
                let flag = at(&pp.strategy_update, i, "strategy_update")?;
                ensure(flag == u32::from(p == u), || {
                    format!("node {i}: player {actor}'s strategy marked for traverser {p}, updater is {u}")
                })?;
            }
        }
        Ok(())
    }

    /// Fields of node `i` on their own.
    fn validate_node(&self, i: usize, hands: &[u32], scale_used: &mut [bool]) -> Result<(), GpuError> {
        let nn = self.num_nodes as usize;
        let actor = at(&self.actor, i, "actor")?;
        let actions = at(&self.actions, i, "actions")?;
        let first_child = at(&self.first_child, i, "first_child")?;
        let folded = at(&self.folded, i, "folded")?;
        let winner = at(&self.winner, i, "winner")?;
        let scale_slot = at(&self.scale_slot, i, "scale_slot")?;
        let card = at(&self.card, i, "card")?;
        let scalar = at(&self.scalar, i, "scalar")?;
        ensure(folded <= ALL_FOLDED, || format!("folded flags {folded:#x}"))?;
        let active = N as u32 - folded.count_ones();
        ensure(card == NONE || (card as usize) < CARDS, || format!("card {card}"))?;
        ensure(scalar <= 1, || format!("scalar flag {scalar}"))?;
        ensure(scalar == 1 || self.batch_river, || "batched leaf with river batching off".into())?;
        let decision = (actor as usize) < N;
        if decision {
            let used = scale_used
                .get_mut(scale_slot as usize)
                .ok_or_else(|| invalid(format!("scale slot {scale_slot} out of range")))?;
            ensure(!*used, || format!("scale slot {scale_slot} shared"))?;
            *used = true;
            ensure(actions <= MAX_ACTIONS, || format!("{actions} actions"))?;
            ensure(folded >> actor & 1 == 0, || format!("player {actor} acts after folding"))?;
            let player_hands = *nth(hands, actor as usize, "player")?;
            let cells = u64::from(actions) * u64::from(player_hands);
            ensure(u64::from(at(&self.offset, i, "offset")?) + cells <= u64::from(self.storage_len), || {
                "storage block out of range".into()
            })?;
        } else {
            ensure(scale_slot == NONE, || "scale slot on a non-decision node".into())?;
        }
        if actor == NONE {
            ensure(actions == 0, || "terminal with children".into())?;
            if winner == NONE {
                ensure(active >= 2, || format!("showdown with {active} players"))?;
            } else {
                ensure((winner as usize) < N && folded == ALL_FOLDED ^ (1 << winner), || {
                    format!("player {winner} won with folded flags {folded:#x}")
                })?;
            }
            return Ok(());
        }
        ensure(decision || actor == CHANCE, || format!("actor {actor}"))?;
        ensure(winner == NONE, || "winner on a non-terminal".into())?;
        ensure(active >= 2, || format!("{active} players left at a non-terminal"))?;
        ensure(actions >= 1, || "no children".into())?;
        let end = u64::from(first_child) + u64::from(actions);
        ensure(first_child as usize > i && end <= nn as u64, || "children out of range".into())?;
        if actor == CHANCE {
            let factor = at(&self.chance_factor, i, "chance_factor")?;
            ensure(factor.is_finite() && factor > 0.0, || format!("chance factor {factor}"))?;
        }
        Ok(())
    }

    /// How node `i` relates to its parent: dealt cards, folds, rounding.
    fn validate_edges(&self, i: usize, parent: &[u32]) -> Result<(), GpuError> {
        let card = at(&self.card, i, "card")?;
        let folded = at(&self.folded, i, "folded")?;
        let scalar = at(&self.scalar, i, "scalar")?;
        let pa = at(parent, i, "parent")?;
        if pa == NONE {
            ensure(card == NONE, || "the root has a dealt card".into())?;
            return ensure(scalar == 1, || "the root is never batched".into());
        }
        let pa = pa as usize;
        let parent_actor = at(&self.actor, pa, "actor")?;
        let parent_folded = at(&self.folded, pa, "folded")?;
        // Siblings are evaluated by the same evaluator.
        let first = at(&self.first_child, pa, "first_child")? as usize;
        ensure(scalar == at(&self.scalar, first, "scalar")?, || "siblings with different rounding".into())?;
        if parent_actor == CHANCE {
            ensure((card as usize) < CARDS, || "chance child without a dealt card".into())?;
            ensure(folded == parent_folded, || "folds change at a chance node".into())?;
            ensure(scalar == 1, || "chance child batched".into())?;
            // Cards dealt by one chance node are distinct.
            for s in first..i {
                ensure(at(&self.card, s, "card")? != card, || format!("card {card} dealt twice"))?;
            }
        } else {
            ensure(card == NONE, || "dealt card below a decision node".into())?;
            // Only the actor can fold.
            let may_fold = 1u32.checked_shl(parent_actor).unwrap_or(0);
            ensure(
                folded & parent_folded == parent_folded && (folded ^ parent_folded) & !may_fold == 0,
                || {
                    format!(
                        "folded flags {folded:#x} below {parent_folded:#x} (player {parent_actor} acting)"
                    )
                },
            )?;
        }
        Ok(())
    }

    fn validate_player(&self, p: usize) -> Result<(), GpuError> {
        let pp = nth(&self.players, p, "player")?;
        let nn = self.num_nodes as usize;
        let n = pp.hands;
        ensure((1..=NONE / 2).contains(&n), || format!("{n} hands"))?;
        ensure(pp.my_cards.len() == n as usize, || "hand array".into())?;
        for &packed in &pp.my_cards {
            let (a, b) = unpack_cards(packed);
            ensure(packed >> 16 == 0 && a < CARDS && b < CARDS && a != b, || {
                format!("bad hand {packed:#x}")
            })?;
        }
        // Value rows: one per node.
        to_u32(nn * n as usize, "value buffer")?;
        let (q, r) = opponents(p);
        for (k, (view, seat)) in pp.opponents.iter().zip([q, r]).enumerate() {
            context(|| format!("opponent {k}"), self.validate_view(pp, view, seat))?;
            to_u32(*nth(&pp.num_slots, k, "opponent")? as usize * view.hands as usize, "reach buffer")?;
        }
        self.validate_schedule(p, pp)
    }

    fn validate_view(&self, pp: &MultiPlayerPlan, view: &OpponentView, seat: usize) -> Result<(), GpuError> {
        ensure(view.player as usize == seat, || format!("seat {} instead of {seat}", view.player))?;
        let other = nth(&self.players, seat, "player")?;
        let nq = view.hands;
        ensure(nq == other.hands && view.cards == other.my_cards, || {
            "hands differ from the player's own".into()
        })?;
        ensure(view.same.len() == pp.hands as usize, || "identical-combo array".into())?;
        ensure(view.same.iter().all(|&j| j == NONE || j < nq), || {
            "identical-combo index out of range".into()
        })?;
        ensure(view.weights.len() == nq as usize, || "weights".into())?;
        ensure(view.weights.iter().all(|w| w.is_finite() && (0.0..=MAX_WEIGHT).contains(w)), || {
            "bad weight".into()
        })?;
        check_offsets(&view.card_off, view.card_hands.len(), "card lists")?;
        // Each card's list holds exactly hands with that card (the chance
        // split zeroes them).
        for c in 0..CARDS {
            let (b, e) = (
                at(&view.card_off, c, "card_off")? as usize,
                at(&view.card_off, c + 1, "card_off")? as usize,
            );
            let list = view.card_hands.get(b..e).ok_or_else(|| invalid("card list out of range"))?;
            for &j in list {
                let (c1, c2) = unpack_cards(at(&view.cards, j as usize, "card list entry")?);
                ensure(c1 == c || c2 == c, || format!("card {c}'s list holds hand {j} without it"))?;
            }
        }
        view.tables.validate(self.num_tables, &pp.my_cards, nq)
    }

    /// Checks player `p`'s schedule: exactly the nodes the CPU traversal
    /// expands are scheduled, once each, level by level; reach slots follow
    /// each parent's role; every leaf is listed once, in the right list.
    fn validate_schedule(&self, p: usize, pp: &MultiPlayerPlan) -> Result<(), GpuError> {
        let nn = self.num_nodes as usize;
        for k in 0..2 {
            ensure(
                nth(&pp.slot, k, "opponent")?.len() == nn && nth(&pp.child_slot, k, "opponent")?.len() == nn,
                || "slot arrays have the wrong length".into(),
            )?;
        }
        ensure(pp.strategy_update.len() == nn, || "strategy-update flags have the wrong length".into())?;
        let reached = self.reached(p)?;
        let role = |i: usize| -> Result<Option<Role>, GpuError> {
            if !self.expands(i, p)? {
                return Ok(None);
            }
            let actor = at(&self.actor, i, "actor")?;
            Ok(Some(match actor {
                CHANCE => Role::Chance,
                a if a as usize == p => Role::Own,
                a => Role::Opponent(
                    opponent_index(p, a as usize).ok_or_else(|| invalid(format!("node {i}: actor {a}")))?,
                ),
            }))
        };

        // Slots.
        for k in 0..2 {
            let (slot, child_slot) = (nth(&pp.slot, k, "opponent")?, nth(&pp.child_slot, k, "opponent")?);
            let num_slots = *nth(&pp.num_slots, k, "opponent")?;
            ensure(slot.first() == Some(&0), || format!("root must use slot 0 of opponent {k}"))?;
            let mut fresh_used = vec![false; num_slots as usize];
            for i in 0..nn {
                let (s, cs) = (at(slot, i, "slot")?, at(child_slot, i, "child_slot")?);
                if !at(&reached, i, "reached")? {
                    ensure(s == NONE && cs == NONE, || {
                        format!("node {i} is never reached but has opponent {k}'s slots")
                    })?;
                    continue;
                }
                ensure(s < num_slots, || format!("node {i}: opponent {k}'s slot out of range"))?;
                let splits = match role(i)? {
                    None | Some(Role::Own) => false,
                    Some(Role::Chance) => true,
                    Some(Role::Opponent(o)) => o == k,
                };
                if !splits {
                    ensure(cs == NONE, || format!("node {i} has child slots of opponent {k}"))?;
                    if role(i)?.is_some() {
                        for c in self.children(i)? {
                            ensure(at(slot, c, "slot")? == s, || {
                                format!("node {i}: child {c} doesn't share opponent {k}'s slot")
                            })?;
                        }
                    }
                    continue;
                }
                let children = self.children(i)?;
                let end = u64::from(cs) + children.len() as u64;
                ensure(cs != NONE && cs >= 1 && end <= u64::from(num_slots), || {
                    format!("node {i}: opponent {k}'s child slots out of range")
                })?;
                for (a, c) in children.enumerate() {
                    let fresh = cs + a as u32;
                    ensure(at(slot, c, "slot")? == fresh, || {
                        format!("node {i}: child {c} has the wrong slot of opponent {k}")
                    })?;
                    let used =
                        fresh_used.get_mut(fresh as usize).ok_or_else(|| invalid("slot out of range"))?;
                    ensure(!*used, || format!("opponent {k}'s slot {fresh} handed out twice"))?;
                    *used = true;
                }
            }
        }

        // Strategy updates only at expanded opponent nodes.
        for (i, &f) in pp.strategy_update.iter().enumerate() {
            ensure(f <= 1, || format!("node {i}: strategy-update flag {f}"))?;
            if f == 1 {
                ensure(at(&reached, i, "reached")? && matches!(role(i)?, Some(Role::Opponent(_))), || {
                    format!("node {i}: strategy update on a node that isn't an expanded opponent node")
                })?;
            }
        }

        // Levels: every expanded node once, in the list of its role, its
        // expanded children one level deeper; no empty levels.
        let mut level_of = vec![NONE; nn];
        let mut expect = [0u64; 4];
        for (d, lv) in pp.levels.iter().enumerate() {
            let parts: [(Span, &[u32], Role); 4] = [
                (lv.opp[0], &pp.opp_nodes[0], Role::Opponent(0)),
                (lv.opp[1], &pp.opp_nodes[1], Role::Opponent(1)),
                (lv.own, &pp.own_nodes, Role::Own),
                (lv.chance, &pp.chance_nodes, Role::Chance),
            ];
            let mut size = 0u64;
            for ((span, list, list_role), want) in parts.into_iter().zip(expect.iter_mut()) {
                ensure(u64::from(span.start) == *want, || format!("level {d} isn't contiguous"))?;
                *want = u64::from(span.start) + u64::from(span.count);
                size += u64::from(span.count);
                let entries =
                    list.get(span.range()).ok_or_else(|| invalid(format!("level {d} out of range")))?;
                for &id in entries {
                    let i = id as usize;
                    ensure(at(&reached, i, "reached")? && role(i)? == Some(list_role), || {
                        format!("level {d}: node {id} in the wrong list")
                    })?;
                    let entry =
                        level_of.get_mut(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                    ensure(*entry == NONE, || format!("node {id} scheduled twice"))?;
                    *entry = d as u32;
                }
            }
            ensure(size > 0, || format!("level {d} is empty"))?;
        }
        let lengths =
            [pp.opp_nodes[0].len(), pp.opp_nodes[1].len(), pp.own_nodes.len(), pp.chance_nodes.len()];
        ensure(expect.iter().zip(lengths).all(|(&e, len)| e == len as u64), || {
            "node lists have unscheduled entries".into()
        })?;

        // Leaves: each once, fixed-payoff or showdown as its kind says.
        ensure(pp.fixed_payoff.len() == pp.fixed.len(), || "fixed payoffs".into())?;
        let ns = pp.showdowns.len();
        ensure(pp.showdown_table.len() == ns && pp.win.len() == ns && pp.lose.len() == ns, || {
            "showdown arrays".into()
        })?;
        ensure(pp.showdown_table.iter().all(|&t| t < self.num_tables), || {
            "showdown table out of range".into()
        })?;
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite() && x.abs() <= MAX_PAYOFF);
        ensure(finite(&pp.fixed_payoff) && finite(&pp.win) && finite(&pp.lose), || {
            "non-finite payoff".into()
        })?;
        let mut leaf_seen = vec![false; nn];
        let listed = pp.fixed.iter().map(|&id| (id, false)).chain(pp.showdowns.iter().map(|&id| (id, true)));
        for (id, showdown) in listed {
            let i = id as usize;
            ensure(at(&reached, i, "reached")? && !self.expands(i, p)?, || {
                format!("leaf list entry {id} isn't a leaf")
            })?;
            let is_showdown = at(&self.actor, i, "actor")? == NONE
                && at(&self.winner, i, "winner")? == NONE
                && at(&self.folded, i, "folded")? >> p & 1 == 0;
            ensure(showdown == is_showdown, || format!("leaf {id} in the wrong list"))?;
            let seen = leaf_seen.get_mut(i).ok_or_else(|| invalid(format!("leaf {id} out of range")))?;
            ensure(!*seen, || format!("leaf {id} listed twice"))?;
            *seen = true;
        }
        for i in 0..nn {
            if !at(&reached, i, "reached")? {
                continue;
            }
            let level = at(&level_of, i, "level")?;
            if !self.expands(i, p)? {
                ensure(at(&leaf_seen, i, "leaf")?, || format!("leaf {i} not listed"))?;
                continue;
            }
            ensure(level != NONE, || format!("node {i} not scheduled"))?;
            ensure(i != 0 || level == 0, || "the root isn't at level 0".into())?;
            for c in self.children(i)? {
                if self.expands(c, p)? {
                    ensure(at(&level_of, c, "level")? == level + 1, || {
                        format!("node {c} isn't one level below its parent {i}")
                    })?;
                }
            }
        }
        Ok(())
    }
}

impl MultiPlayerPlan {
    /// Player `p`'s hand data, lookup tables and schedule.
    fn build(ctx: &Ctx, plan: &MultiPlan, p: usize) -> Result<MultiPlayerPlan, GpuError> {
        let hands = ctx.hands;
        let (q, r) = opponents(p);
        let my_cards = nth(&hands.cards, p, "player")?;
        let mut pp = MultiPlayerPlan {
            hands: to_u32(my_cards.len(), "hands")?,
            my_cards: my_cards.iter().map(|&(a, b)| pack_cards(a, b)).collect(),
            opponents: [OpponentView::build(ctx, p, q)?, OpponentView::build(ctx, p, r)?],
            ..MultiPlayerPlan::default()
        };
        pp.build_schedule(ctx, plan, p)?;
        Ok(pp)
    }

    /// Levels, reach slots, strategy-update marks and leaf lists of player
    /// `p`'s traversal, breadth-first from the root.
    fn build_schedule(&mut self, ctx: &Ctx, plan: &MultiPlan, p: usize) -> Result<(), GpuError> {
        let nodes = &ctx.tree.nodes;
        let nn = plan.num_nodes as usize;
        self.slot = [vec![NONE; nn], vec![NONE; nn]];
        self.child_slot = [vec![NONE; nn], vec![NONE; nn]];
        self.strategy_update = vec![0; nn];
        for s in &mut self.slot {
            put(s, 0, 0, "slot")?;
        }
        let mut next_slot = [1u64; 2];
        let mut frontier = vec![0u32];
        while !frontier.is_empty() {
            let start = |v: &[u32]| -> Result<Span, GpuError> {
                Ok(Span { start: to_u32(v.len(), "node list")?, count: 0 })
            };
            let mut level = MultiLevel {
                opp: [start(&self.opp_nodes[0])?, start(&self.opp_nodes[1])?],
                own: start(&self.own_nodes)?,
                chance: start(&self.chance_nodes)?,
            };
            let mut next = Vec::new();
            for &id in &frontier {
                let i = id as usize;
                let node = nodes.get(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                let leaf =
                    matches!(node.kind, Kind::Won(_) | Kind::Showdown) || node.folded.get(p) == Some(&true);
                if leaf {
                    let (win, lose) = payoffs(ctx, node, p, false);
                    match node.kind {
                        Kind::Showdown if node.folded.get(p) == Some(&false) => {
                            self.showdowns.push(id);
                            self.showdown_table.push(node.slot);
                            self.win.push(win);
                            self.lose.push(lose);
                        }
                        Kind::Won(w) if w as usize == p => {
                            self.fixed.push(id);
                            self.fixed_payoff.push(win);
                        }
                        _ => {
                            self.fixed.push(id);
                            self.fixed_payoff.push(lose);
                        }
                    }
                    continue;
                }
                let slots = [at(&self.slot[0], i, "slot")?, at(&self.slot[1], i, "slot")?];
                // Opponents whose reach splits here.
                let splits: &[usize] = match node.kind {
                    Kind::Player(x) if x as usize == p => {
                        self.own_nodes.push(id);
                        &[]
                    }
                    Kind::Player(x) => {
                        let x = x as usize;
                        let k =
                            opponent_index(p, x).ok_or_else(|| invalid(format!("node {id}: actor {x}")))?;
                        self.opp_nodes.get_mut(k).ok_or_else(|| invalid("opponent"))?.push(id);
                        put(
                            &mut self.strategy_update,
                            i,
                            u32::from(strategy_updater(node, x) == p),
                            "strategy_update",
                        )?;
                        if k == 0 {
                            &[0]
                        } else {
                            &[1]
                        }
                    }
                    Kind::Chance => {
                        self.chance_nodes.push(id);
                        &[0, 1]
                    }
                    Kind::Won(_) | Kind::Showdown => &[],
                };
                let per_opponent =
                    self.slot.iter_mut().zip(self.child_slot.iter_mut()).zip(next_slot.iter_mut()).zip(slots);
                for (k, (((slot, child_slot), next), s)) in per_opponent.enumerate() {
                    if splits.contains(&k) {
                        let first = u32::try_from(*next).map_err(|_| invalid("too many reach slots"))?;
                        put(child_slot, i, first, "child_slot")?;
                        for (a, c) in node.children().enumerate() {
                            put(slot, c, first.saturating_add(a as u32), "slot")?;
                        }
                        *next += u64::from(node.num_children);
                    } else {
                        for c in node.children() {
                            put(slot, c, s, "slot")?;
                        }
                    }
                }
                for c in node.children() {
                    next.push(to_u32(c, "nodes")?);
                }
            }
            let count =
                |v: &[u32], s: Span| -> Result<u32, GpuError> { Ok(to_u32(v.len(), "node list")? - s.start) };
            level.opp[0].count = count(&self.opp_nodes[0], level.opp[0])?;
            level.opp[1].count = count(&self.opp_nodes[1], level.opp[1])?;
            level.own.count = count(&self.own_nodes, level.own)?;
            level.chance.count = count(&self.chance_nodes, level.chance)?;
            if level.opp[0].count + level.opp[1].count + level.own.count + level.chance.count > 0 {
                self.levels.push(level);
            }
            frontier = next;
        }
        self.num_slots = [
            u32::try_from(next_slot[0]).map_err(|_| invalid("too many reach slots"))?,
            u32::try_from(next_slot[1]).map_err(|_| invalid("too many reach slots"))?,
        ];
        Ok(())
    }
}

impl OpponentView {
    /// Opponent `o` as traverser `p` sees it.
    fn build(ctx: &Ctx, p: usize, o: usize) -> Result<OpponentView, GpuError> {
        let hands = ctx.hands;
        let cards = nth(&hands.cards, o, "player")?;
        let mut off = Vec::with_capacity(CARDS + 1);
        let mut list = Vec::with_capacity(2 * cards.len());
        for held in nth(&hands.blocked, o, "player")?.iter().take(CARDS) {
            off.push(to_u32(list.len(), "card lists")?);
            list.extend_from_slice(held);
        }
        off.push(to_u32(list.len(), "card lists")?);
        let my_cards = nth(&hands.cards, p, "player")?;
        let mut tables = Tables::default();
        for table in ctx.tables {
            let ranked = |e: &(u32, u32, u8, u8)| Ranked { strength: e.0, idx: e.1, c1: e.2, c2: e.3 };
            tables.push(
                nth(&table.lists, p, "player")?,
                nth(&table.lists, o, "player")?,
                my_cards,
                ranked,
            )?;
        }
        Ok(OpponentView {
            player: to_u32(o, "player")?,
            hands: to_u32(cards.len(), "hands")?,
            cards: cards.iter().map(|&(a, b)| pack_cards(a, b)).collect(),
            same: nth(nth(&hands.same, p, "player")?, o, "player")?.clone(),
            card_off: off,
            card_hands: list,
            weights: nth(&hands.weights, o, "player")?.clone(),
            tables,
        })
    }
}

#[cfg(test)]
mod tests;
