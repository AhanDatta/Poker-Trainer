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

use crate::gpu::shared::{at, ensure, pack_cards, to_u32, Span, CARDS, NONE};
use crate::gpu::tables::{Ranked, Tables};
use crate::gpu::{invalid, GpuError};
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

    /// Player `p`'s root weights (as its opponents' plans hold them).
    pub(crate) fn root_weights(&self, p: usize) -> Result<&[f32], GpuError> {
        self.players
            .iter()
            .flat_map(|pp| &pp.opponents)
            .find(|view| view.player as usize == p)
            .map(|view| view.weights.as_slice())
            .ok_or_else(|| invalid(format!("no weights for player {p}")))
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
            + b(&self.folded)
            // The engine's value-row map (one row per node).
            + b(&self.actions);
        let tables = |t: &Tables| {
            b(&t.opp_off)
                + b(&t.opp_len)
                + b(&t.opp_order)
                + b(&t.card_base)
                + b(&t.card_off)
                + b(&t.card_pos)
                + b(&t.ranges)
                + b(&t.ranks)
        };
        let view = |v: &OpponentView| {
            b(&v.cards) + b(&v.same) + b(&v.card_off) + b(&v.card_hands) + b(&v.weights) + tables(&v.tables)
        };
        let player = |pp: &MultiPlayerPlan| {
            pp.slot.iter().chain(&pp.child_slot).chain(&pp.opp_nodes).map(|v| b(v)).sum::<u64>()
                + b(&pp.own_nodes)
                + b(&pp.chance_nodes)
                + b(&pp.strategy_update)
                + b(&pp.fixed)
                + b(&pp.fixed_payoff)
                + b(&pp.showdowns)
                + b(&pp.showdown_table)
                + b(&pp.win)
                + b(&pp.lose)
                + b(&pp.my_cards)
                + pp.opponents.iter().map(view).sum::<u64>()
        };
        tree + self.players.iter().map(player).sum::<u64>()
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

mod validate;

#[cfg(test)]
mod tests;
