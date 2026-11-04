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
use crate::tree::Kind;
use std::ops::Range;

/// "No index" marker, shared with the kernels (`solver::kNone`).
pub(crate) const NONE: u32 = u32::MAX;

/// Cards in the deck.
pub(crate) const CARDS: usize = 52;

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

/// One tree level: ranges into the opponent-node and own-node lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LevelRange {
    pub opp_start: u32,
    pub opp_count: u32,
    pub own_start: u32,
    pub own_count: u32,
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
    /// Per node: the first child's slot at opponent nodes (children's slots
    /// are consecutive), `NONE` elsewhere.
    pub child_slot: Vec<u32>,
    pub num_slots: u32,
    /// Top-down order; bottom-up walks it backwards.
    pub levels: Vec<LevelRange>,
    pub opp_nodes: Vec<u32>,
    pub own_nodes: Vec<u32>,
    pub folds: Vec<u32>,
    pub fold_payoff: Vec<f64>,
    pub showdowns: Vec<u32>,
    pub showdown_table: Vec<u32>,
    pub win: Vec<f64>,
    pub lose: Vec<f64>,
    pub tie: Vec<f64>,
    /// The traverser's hands' cards, packed `c1 | c2 << 8`.
    pub my_cards: Vec<u32>,
    /// Index of the opponent's identical combo, or `NONE`.
    pub same: Vec<u32>,
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
    /// Per node: the acting player of a decision node, `NONE` for terminals.
    pub actor: Vec<u32>,
    pub num_tables: u32,
    pub players: [PlayerPlan; 2],
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
    /// Flattens the tree behind `ctx`. Trees with chance nodes (turn and
    /// flop roots) aren't supported yet.
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
            match node.kind {
                Kind::Chance => {
                    return Err(invalid(
                        "trees with chance nodes (turn and flop roots) aren't supported on the GPU yet",
                    ))
                }
                Kind::Player(q) => {
                    plan.actions.push(u32::from(node.num_children));
                    plan.first_child.push(node.first_child);
                    plan.offset.push(
                        u32::try_from(node.offset)
                            .map_err(|_| invalid("storage offset exceeds u32 indexing"))?,
                    );
                    plan.actor.push(u32::from(q));
                }
                Kind::Fold(_) | Kind::Showdown => {
                    plan.actions.push(0);
                    plan.first_child.push(0);
                    plan.offset.push(0);
                    plan.actor.push(NONE);
                }
            }
        }
        let levels = plan.bfs_levels()?;
        plan.players =
            [PlayerPlan::build(ctx, &plan, &levels, 0)?, PlayerPlan::build(ctx, &plan, &levels, 1)?];
        plan.validate()?;
        Ok(plan)
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
        ] {
            ensure(len == nn, || format!("{name} has {len} entries for {nn} nodes"))?;
        }
        ensure(nn > 0 && self.actor.first().is_some_and(|&a| a != NONE), || {
            "root must be a decision node".into()
        })?;
        let hands = [self.players[0].hands, self.players[1].hands];
        for i in 0..nn {
            let node = self.node(i)?;
            if node.actor == NONE {
                ensure(node.actions == 0, || format!("terminal {i} has actions"))?;
                continue;
            }
            let player_hands = *hands
                .get(node.actor as usize)
                .ok_or_else(|| invalid(format!("node {i}: actor {}", node.actor)))?;
            ensure(node.actions >= 1, || format!("decision node {i} has no actions"))?;
            let end = u64::from(node.first_child) + u64::from(node.actions);
            ensure(node.first_child as usize > i && end <= nn as u64, || {
                format!("node {i}: children out of range")
            })?;
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
            if node.actor as usize == p {
                ensure(child_slot == NONE, || format!("own node {i} has child slots"))?;
                for c in self.children(i)? {
                    ensure(at(&pp.slot, c, "slot")? == slot, || {
                        format!("own node {i}: child {c} doesn't share its slot")
                    })?;
                }
            } else {
                let end = u64::from(child_slot) + u64::from(node.actions);
                ensure(child_slot != NONE && end <= u64::from(pp.num_slots), || {
                    format!("opponent node {i}: child slots out of range")
                })?;
                for (a, c) in self.children(i)?.enumerate() {
                    ensure(u64::from(at(&pp.slot, c, "slot")?) == u64::from(child_slot) + a as u64, || {
                        format!("opponent node {i}: child {c} has the wrong slot")
                    })?;
                }
            }
        }

        // Levels: every decision node exactly once, in the list of its role,
        // and decision children exactly one level deeper.
        let mut level_of = vec![NONE; nn];
        let mut expect_opp = 0u32;
        let mut expect_own = 0u32;
        for (d, lv) in pp.levels.iter().enumerate() {
            ensure(lv.opp_start == expect_opp && lv.own_start == expect_own, || {
                format!("level {d} isn't contiguous")
            })?;
            expect_opp = lv.opp_start.checked_add(lv.opp_count).ok_or_else(|| invalid("level overflow"))?;
            expect_own = lv.own_start.checked_add(lv.own_count).ok_or_else(|| invalid("level overflow"))?;
            let opp = pp.opp_nodes.get(lv.opp_start as usize..expect_opp as usize);
            let own = pp.own_nodes.get(lv.own_start as usize..expect_own as usize);
            let (Some(opp), Some(own)) = (opp, own) else {
                return Err(invalid(format!("level {d} out of range")));
            };
            for (&id, is_own) in opp.iter().map(|x| (x, false)).chain(own.iter().map(|x| (x, true))) {
                let i = id as usize;
                let actor = self.node(i)?.actor;
                ensure(actor != NONE && (actor as usize == p) == is_own, || {
                    format!("node {id} in the wrong list")
                })?;
                let entry = level_of.get_mut(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                ensure(*entry == NONE, || format!("node {id} scheduled twice"))?;
                *entry = d as u32;
            }
        }
        ensure(
            expect_opp as usize == pp.opp_nodes.len() && expect_own as usize == pp.own_nodes.len(),
            || "node lists have unscheduled entries".into(),
        )?;

        // Terminals: each exactly once, with its payoffs.
        ensure(pp.fold_payoff.len() == pp.folds.len(), || "fold payoffs".into())?;
        let ns = pp.showdowns.len();
        ensure(
            pp.showdown_table.len() == ns && pp.win.len() == ns && pp.lose.len() == ns && pp.tie.len() == ns,
            || "showdown arrays".into(),
        )?;
        let mut terminal_seen = vec![false; nn];
        for &id in pp.folds.iter().chain(&pp.showdowns) {
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
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        ensure(finite(&pp.fold_payoff) && finite(&pp.win) && finite(&pp.lose) && finite(&pp.tie), || {
            "non-finite payoff".into()
        })?;

        // Hands.
        ensure(pp.my_cards.len() == n as usize && pp.same.len() == n as usize, || "hand arrays".into())?;
        for &packed in &pp.my_cards {
            let (a, b) = unpack_cards(packed);
            ensure(packed >> 16 == 0 && a < CARDS && b < CARDS && a != b, || {
                format!("bad hand {packed:#x}")
            })?;
        }
        ensure(pp.same.iter().all(|&j| j == NONE || j < nq), || "identical-combo index out of range".into())?;
        ensure(pp.opp_weights.len() == nq as usize, || "opponent weights".into())?;
        ensure(pp.opp_weights.iter().all(|w| w.is_finite() && *w >= 0.0), || "bad opponent weight".into())?;
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
                    Kind::Player(_) => {
                        pp.opp_nodes.push(id);
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
                    Kind::Chance => return Err(invalid("chance node in a river plan")),
                }
            }
            range.opp_count = to_u32(pp.opp_nodes.len(), "opponent nodes")? - range.opp_start;
            range.own_count = to_u32(pp.own_nodes.len(), "own nodes")? - range.own_start;
            pp.levels.push(range);
        }
        pp.num_slots = u32::try_from(next_slot).map_err(|_| invalid("too many reach slots"))?;

        pp.my_cards = side(&hs.cards, p)?.iter().map(|&(a, b)| pack_cards(a, b)).collect();
        pp.same = side(&hs.same, p)?.clone();
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
