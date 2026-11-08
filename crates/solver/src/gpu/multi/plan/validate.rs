//! [`MultiPlan::validate`]: every index and invariant the kernels rely on.

use super::{nth, opponent_index, updater, MultiPlan, MultiPlayerPlan, OpponentView, Role, ALL_FOLDED, CHANCE};
use crate::gpu::shared::{
    at, ensure, to_u32, unpack_cards, Span, CARDS, MAX_ACTIONS, MAX_HANDS, MAX_PAYOFF, MAX_WEIGHT,
    NONE,
};
use crate::gpu::segments::{self, Segmentation};
use crate::gpu::{invalid, GpuError};
use crate::multi::{opponents, N};

/// Prefixes an input error with `prefix`.
fn context<T>(prefix: impl Fn() -> String, r: Result<T, GpuError>) -> Result<T, GpuError> {
    r.map_err(|e| match e {
        GpuError::InvalidInput(s) => GpuError::InvalidInput(format!("{}: {s}", prefix())),
        other => other,
    })
}

impl MultiPlan {
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
        let seg = Segmentation {
            segment_of: self.segment_of.clone(),
            num_segments: self.num_segments,
            row: self.row.clone(),
            top_rows: self.top_rows,
            num_rows: self.num_rows,
        };
        context(
            || "segments".into(),
            segments::validate(&seg, nn, &parent, &|i| Ok(at(&self.actor, i, "actor")? == CHANCE)),
        )?;
        for p in 0..N {
            context(|| format!("player {p}"), self.validate_player(p))?;
        }
        context(|| "leaf lists".into(), self.lists.validate(self.num_tables, &self.players))?;
        for (p, pp) in self.players.iter().enumerate() {
            context(|| format!("player {p}'s leaf groups"), pp.leaves.validate(self, pp, p))?;
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
        let n = pp.hands;
        ensure((1..=MAX_HANDS).contains(&n), || format!("{n} hands"))?;
        ensure(pp.my_cards.len() == n as usize, || "hand array".into())?;
        for &packed in &pp.my_cards {
            let (a, b) = unpack_cards(packed);
            ensure(packed >> 16 == 0 && a < CARDS && b < CARDS && a != b, || {
                format!("bad hand {packed:#x}")
            })?;
        }
        // Value rows.
        to_u32(self.num_rows as usize * n as usize, "value buffer")?;
        let (q, r) = opponents(p);
        for (k, (view, seat)) in pp.opponents.iter().zip([q, r]).enumerate() {
            context(|| format!("opponent {k}"), self.validate_view(view, seat))?;
            to_u32(*nth(&pp.num_slots, k, "opponent")? as usize * view.hands as usize, "reach buffer")?;
        }
        self.validate_schedule(p, pp)
    }

    fn validate_view(&self, view: &OpponentView, seat: usize) -> Result<(), GpuError> {
        ensure(view.player as usize == seat, || format!("seat {} instead of {seat}", view.player))?;
        let other = nth(&self.players, seat, "player")?;
        let nq = view.hands;
        ensure(nq == other.hands && view.cards == other.my_cards, || {
            "hands differ from the player's own".into()
        })?;
        ensure(view.weights.len() == nq as usize, || "weights".into())?;
        ensure(view.weights.iter().all(|w| w.is_finite() && (0.0..=MAX_WEIGHT).contains(w)), || {
            "bad weight".into()
        })
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

        // Segments: one per plan segment, spans contiguous and covering the
        // lists, each opponent's fresh slots in its region (the top's from
        // 1, every group's from the end of the top's).
        ensure(pp.segments.len() == self.num_segments as usize, || "segment count".into())?;
        let (mut lv, mut fx, mut sd) = (0u64, 0u64, 0u64);
        for (s, seg) in pp.segments.iter().enumerate() {
            ensure(
                u64::from(seg.levels.start) == lv && u64::from(seg.fixed.start) == fx && u64::from(seg.showdowns.start) == sd,
                || format!("segment {s}: spans aren't contiguous"),
            )?;
            (lv, fx, sd) = (seg.levels.end(), seg.fixed.end(), seg.showdowns.end());
            for k in 0..2 {
                let fresh = *nth(&seg.fresh_slots, k, "opponent")?;
                let (top, num) = (*nth(&pp.top_slots, k, "opponent")?, *nth(&pp.num_slots, k, "opponent")?);
                ensure(top >= 1 && top <= num, || format!("opponent {k}'s slot regions"))?;
                if s == 0 {
                    ensure(fresh.start == 1 && fresh.end() == u64::from(top), || format!("top segment's slots of opponent {k}"))?;
                } else {
                    ensure(fresh.start == top && fresh.end() <= u64::from(num), || {
                        format!("segment {s}: opponent {k}'s slots outside the group region")
                    })?;
                }
            }
        }
        ensure(
            lv == pp.levels.len() as u64 && fx == pp.fixed.len() as u64 && sd == pp.showdowns.len() as u64,
            || "segments don't cover the lists".into(),
        )?;

        // Slots.
        for k in 0..2 {
            let (slot, child_slot) = (nth(&pp.slot, k, "opponent")?, nth(&pp.child_slot, k, "opponent")?);
            let num_slots = *nth(&pp.num_slots, k, "opponent")?;
            ensure(slot.first() == Some(&0), || format!("root must use slot 0 of opponent {k}"))?;
            let mut fresh_used: Vec<(u32, u32)> = Vec::new();
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
                let seg = at(&self.segment_of, i, "segment_of")?;
                let region = *nth(&nth(&pp.segments, seg as usize, "segment")?.fresh_slots, k, "opponent")?;
                ensure(cs != NONE && cs >= region.start && end <= region.end() && end <= u64::from(num_slots), || {
                    format!("node {i}: opponent {k}'s child slots outside its segment's region")
                })?;
                for (a, c) in children.enumerate() {
                    let fresh = cs + a as u32;
                    ensure(at(slot, c, "slot")? == fresh, || {
                        format!("node {i}: child {c} has the wrong slot of opponent {k}")
                    })?;
                    fresh_used.push((seg, fresh));
                }
            }
            fresh_used.sort_unstable();
            ensure(fresh_used.windows(2).all(|w| w.first() != w.get(1)), || {
                format!("opponent {k}'s slot handed out twice in a segment")
            })?;
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

        // Levels: every expanded node once, in its segment and the list of
        // its role, its expanded children in the same segment one level
        // deeper, chunk roots at their group's first level; no empty levels.
        let mut level_of = vec![NONE; nn];
        let mut expect = [0u64; 4];
        let mut segment_of_level = Vec::with_capacity(pp.levels.len());
        let mut depth_of_level = Vec::with_capacity(pp.levels.len());
        for (s, seg) in pp.segments.iter().enumerate() {
            for d in 0..seg.levels.count {
                segment_of_level.push(s as u32);
                depth_of_level.push(d);
            }
        }
        for (d, lv) in pp.levels.iter().enumerate() {
            let s = at(&segment_of_level, d, "level segment")?;
            let depth = at(&depth_of_level, d, "level depth")?;
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
                    ensure(at(&self.segment_of, i, "segment_of")? == s, || format!("level {d}: node {id} in the wrong segment"))?;
                    let entry =
                        level_of.get_mut(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                    ensure(*entry == NONE, || format!("node {id} scheduled twice"))?;
                    *entry = depth;
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
        let mut leaf_segment = vec![NONE; pp.fixed.len() + pp.showdowns.len()];
        for (s, seg) in pp.segments.iter().enumerate() {
            for i in seg.fixed.range() {
                *leaf_segment.get_mut(i).ok_or_else(|| invalid("fixed span"))? = s as u32;
            }
            for i in seg.showdowns.range() {
                *leaf_segment.get_mut(pp.fixed.len() + i).ok_or_else(|| invalid("showdown span"))? = s as u32;
            }
        }
        let listed = pp.fixed.iter().map(|&id| (id, false)).chain(pp.showdowns.iter().map(|&id| (id, true)));
        for (k, (id, showdown)) in listed.enumerate() {
            let i = id as usize;
            ensure(at(&self.segment_of, i, "segment_of")? == at(&leaf_segment, k, "leaf segment")?, || {
                format!("leaf {id} in another segment's list")
            })?;
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
            let seg = at(&self.segment_of, i, "segment_of")?;
            for c in self.children(i)? {
                if self.expands(c, p)? {
                    let want = if at(&self.segment_of, c, "segment_of")? == seg { level + 1 } else { 0 };
                    ensure(at(&level_of, c, "level")? == want, || {
                        format!("node {c} isn't one level below its parent {i} (or first in its group)")
                    })?;
                }
            }
        }
        Ok(())
    }
}
