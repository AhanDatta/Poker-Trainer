//! The traversal schedule. A tree too big for the device's scratch memory
//! is split into **segments** (`gpu::segments`, shared with the
//! three-player plan): the top of the tree (segment 0), and groups of
//! **chunks** (the subtrees under the root street's chance nodes: turn
//! subtrees on a flop, river subtrees on a turn), each group run in turn in
//! one shared scratch region:
//!
//! 1. top segment top-down (its reach reaches the chunk roots), terminals;
//! 2. for each group: reset its reach slots' alive flags, top-down,
//!    terminals, bottom-up (ending at the chunk roots, whose value rows are
//!    in the top region);
//! 3. top segment bottom-up.
//!
//! Chunks are disjoint subtrees, so running them group by group computes
//! exactly what one pass over the whole tree computes. A tree that fits is
//! a single segment.
//!
//! Scratch regions: value rows `[0, top_rows)` belong to the top segment and
//! the chunk roots, `[top_rows, num_rows)` to the group running; reach slots
//! `[0, top_slots)` to the top segment (and the chunk roots), `[top_slots,
//! num_slots)` to the group running. Everything here is checked by
//! [`validate_segments`] and [`validate_player_schedule`].

use super::{at, ensure, invalid, to_u32, Kind, LevelRange, Plan, PlayerPlan, Role, Span, CHANCE, NONE};
use crate::cfr::{fold_payoff, showdown_payoffs, Ctx};
use crate::gpu::segments::{self, Shape};
pub(crate) use crate::gpu::segments::{Scratch, Segmentation};
use crate::gpu::GpuError;

/// One segment of one player's traversal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Segment {
    /// Into `PlayerPlan::levels`.
    pub levels: Span,
    /// Into `folds` / `fold_payoff`.
    pub folds: Span,
    /// Into the showdown arrays.
    pub showdowns: Span,
    /// Reach slots this segment's opponent and chance nodes hand out to
    /// their children (a group's are reset before it runs).
    pub fresh_slots: Span,
}

/// Bytes of the engine's scratch buffers for the given reach slots per
/// traverser and value rows (it allocates each buffer for the larger
/// traversal): reach slots x opponent hands, rows x hands, one alive flag
/// per slot, all 4 bytes.
pub(crate) fn scratch_bytes(slots: [u64; 2], rows: u64, hands: [u64; 2]) -> u64 {
    let reach = (slots[0] * hands[1]).max(slots[1] * hands[0]);
    let values = rows * hands[0].max(hands[1]);
    let alive = slots[0].max(slots[1]);
    4 * (reach + values + alive)
}

impl Plan {
    /// Reach slots node `i` hands out to its children when player `p`
    /// traverses: one per child at opponent and chance nodes.
    fn fresh_slots(&self, i: usize, p: usize) -> Result<u64, GpuError> {
        let node = self.node(i)?;
        Ok(match node.actor {
            NONE => 0,
            CHANCE => u64::from(node.actions),
            a if a as usize == p => 0,
            _ => u64::from(node.actions),
        })
    }

    /// Splits the tree into segments under `scratch`, given the device bytes
    /// the plan itself needs (`fixed_bytes`).
    pub(super) fn segment(&self, ctx: &Ctx, scratch: Scratch, fixed_bytes: u64) -> Result<Segmentation, GpuError> {
        let nodes = &ctx.tree.nodes;
        let hands = [ctx.hands.len(0) as u64, ctx.hands.len(1) as u64];
        // Chunk roots: the children of the root street's chance nodes.
        let root_street = nodes.first().ok_or_else(|| invalid("empty tree"))?.street;
        let mut roots: Vec<usize> = Vec::new();
        for node in nodes.iter().filter(|n| matches!(n.kind, Kind::Chance) && n.street == root_street) {
            roots.extend(node.children());
        }
        let shape = Shape {
            num_nodes: nodes.len(),
            children: &|i| self.children(i),
            roots,
            fresh: &|i| Ok(vec![self.fresh_slots(i, 0)?, self.fresh_slots(i, 1)?]),
            spaces: 2,
        };
        let bytes = |rows: u64, slots: &[u64]| {
            let s = [slots.first().copied().unwrap_or(0), slots.get(1).copied().unwrap_or(0)];
            fixed_bytes + scratch_bytes(s, rows, hands)
        };
        segments::segment(&shape, scratch, &bytes)
    }

    /// Per node: its parent (`NONE` for the root), checking that every node
    /// but the root has exactly one.
    fn parents(&self) -> Result<Vec<u32>, GpuError> {
        let nn = self.num_nodes as usize;
        let mut parent = vec![NONE; nn];
        for i in 0..nn {
            for c in self.children(i)? {
                let entry = parent.get_mut(c).ok_or_else(|| invalid(format!("node {i}: child {c} out of range")))?;
                ensure(*entry == NONE, || format!("node {c} has two parents"))?;
                *entry = to_u32(i, "nodes")?;
            }
        }
        for (c, &pa) in parent.iter().enumerate().skip(1) {
            ensure(pa != NONE, || format!("node {c} has no parent"))?;
        }
        Ok(parent)
    }

    /// Per segment (index 0 empty): its chunk roots, in node order.
    fn segment_roots(&self) -> Result<Vec<Vec<u32>>, GpuError> {
        let mut roots = vec![Vec::new(); self.num_segments as usize];
        for i in 0..self.num_nodes as usize {
            let seg = at(&self.segment_of, i, "segment_of")?;
            for c in self.children(i)? {
                let child_seg = at(&self.segment_of, c, "segment_of")?;
                if child_seg != seg {
                    roots
                        .get_mut(child_seg as usize)
                        .ok_or_else(|| invalid("segment out of range"))?
                        .push(to_u32(c, "nodes")?);
                }
            }
        }
        Ok(roots)
    }
}

impl PlayerPlan {
    /// Levels, terminal lists, reach slots and segments of player `p`'s
    /// traversal, given the plan's segmentation.
    pub(super) fn build_schedule(&mut self, ctx: &Ctx, plan: &Plan, p: usize) -> Result<(), GpuError> {
        let nodes = &ctx.tree.nodes;
        let nn = plan.num_nodes as usize;
        let roots = plan.segment_roots()?;
        self.slot = vec![NONE; nn];
        self.child_slot = vec![NONE; nn];
        self.fused = vec![0; nn];
        if let Some(s) = self.slot.first_mut() {
            *s = 0;
        }
        // Per segment: folds (+ payoffs) and showdowns (+ table, payoffs).
        #[derive(Default)]
        struct Terminals {
            folds: Vec<u32>,
            fold_payoff: Vec<f64>,
            showdowns: Vec<u32>,
            table: Vec<u32>,
            win: Vec<f64>,
            lose: Vec<f64>,
            tie: Vec<f64>,
        }
        let mut terminals: Vec<Terminals> = Vec::new();
        let mut next_slot = 1u64;
        let mut top_slots = 0u64;
        let mut max_group_slots = 0u64;
        for seg in 0..plan.num_segments {
            if seg == 1 {
                top_slots = next_slot;
            }
            if seg >= 1 {
                next_slot = top_slots;
            }
            let fresh_start = next_slot;
            let level_start = self.levels.len();
            let mut t = Terminals::default();
            let mut frontier: Vec<u32> = if seg == 0 {
                vec![0]
            } else {
                roots.get(seg as usize).cloned().unwrap_or_default()
            };
            while !frontier.is_empty() {
                let mut range = LevelRange {
                    opp_start: to_u32(self.opp_nodes.len(), "opponent nodes")?,
                    own_start: to_u32(self.own_nodes.len(), "own nodes")?,
                    chance_start: to_u32(self.chance_nodes.len(), "chance nodes")?,
                    ..LevelRange::default()
                };
                let mut next = Vec::new();
                let mut fused_here = Vec::new();
                for &id in &frontier {
                    let i = id as usize;
                    let node = nodes.get(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                    let my_slot = at(&self.slot, i, "slot")?;
                    let terminal = matches!(node.kind, Kind::Fold(_) | Kind::Showdown);
                    match node.kind {
                        Kind::Player(q) if q as usize == p => {
                            self.own_nodes.push(id);
                            for c in node.children() {
                                *self.slot.get_mut(c).ok_or_else(|| invalid("child out of range"))? = my_slot;
                                if matches!(nodes.get(c).map(|n| n.kind), Some(Kind::Player(r)) if r as usize != p) {
                                    *self.fused.get_mut(c).ok_or_else(|| invalid("child out of range"))? = 1;
                                }
                            }
                        }
                        Kind::Player(_) | Kind::Chance => {
                            if matches!(node.kind, Kind::Chance) {
                                self.chance_nodes.push(id);
                            } else if at(&self.fused, i, "fused")? != 0 {
                                fused_here.push(id);
                            } else {
                                self.opp_nodes.push(id);
                            }
                            let first = u32::try_from(next_slot).map_err(|_| invalid("too many reach slots"))?;
                            *self.child_slot.get_mut(i).ok_or_else(|| invalid("node out of range"))? = first;
                            for (a, c) in node.children().enumerate() {
                                *self.slot.get_mut(c).ok_or_else(|| invalid("child out of range"))? =
                                    first.saturating_add(a as u32);
                            }
                            next_slot += u64::from(node.num_children);
                        }
                        Kind::Fold(_) => {
                            t.folds.push(id);
                            t.fold_payoff.push(fold_payoff(ctx, node, p));
                        }
                        Kind::Showdown => {
                            let (w, l, ti) = showdown_payoffs(ctx, node, p);
                            t.showdowns.push(id);
                            t.table.push(u32::try_from(node.offset).map_err(|_| invalid("strength table index"))?);
                            t.win.push(w);
                            t.lose.push(l);
                            t.tie.push(ti);
                        }
                    }
                    if !terminal {
                        for c in node.children() {
                            if at(&plan.segment_of, c, "segment_of")? == seg {
                                next.push(to_u32(c, "nodes")?);
                            }
                        }
                    }
                }
                // Summed opponent nodes first, then the fused ones.
                range.opp_sum_count = to_u32(self.opp_nodes.len(), "opponent nodes")? - range.opp_start;
                self.opp_nodes.extend_from_slice(&fused_here);
                range.opp_count = to_u32(self.opp_nodes.len(), "opponent nodes")? - range.opp_start;
                range.own_count = to_u32(self.own_nodes.len(), "own nodes")? - range.own_start;
                range.chance_count = to_u32(self.chance_nodes.len(), "chance nodes")? - range.chance_start;
                self.levels.push(range);
                frontier = next;
            }
            if seg >= 1 {
                max_group_slots = max_group_slots.max(next_slot - top_slots);
            }
            self.segments.push(Segment {
                levels: Span::of(level_start, self.levels.len(), "levels")?,
                fresh_slots: Span {
                    start: u32::try_from(fresh_start).map_err(|_| invalid("too many reach slots"))?,
                    count: u32::try_from(next_slot - fresh_start).map_err(|_| invalid("too many reach slots"))?,
                },
                ..Segment::default()
            });
            terminals.push(t);
        }
        if plan.num_segments == 1 {
            top_slots = next_slot;
        }
        self.top_slots = u32::try_from(top_slots).map_err(|_| invalid("too many reach slots"))?;
        self.num_slots =
            u32::try_from(top_slots + max_group_slots).map_err(|_| invalid("too many reach slots"))?;

        // Folds sharing a showdown's reach (both children of one of the
        // traverser's nodes, both batched) are evaluated with it, as the
        // CPU's `showdown_batch` groups them. Siblings share a segment.
        let mut showdown_at = vec![(NONE, NONE); nn];
        for (seg, t) in terminals.iter().enumerate() {
            for (k, &id) in t.showdowns.iter().enumerate() {
                *showdown_at.get_mut(id as usize).ok_or_else(|| invalid("showdown out of range"))? =
                    (to_u32(seg, "segments")?, to_u32(k, "showdowns")?);
            }
        }
        let mut pair_of: Vec<Vec<u32>> = terminals.iter().map(|t| vec![NONE; t.showdowns.len()]).collect();
        let mut paired = vec![false; nn];
        let batched = |c: usize| at(&plan.scalar, c, "scalar").is_ok_and(|s| s == 0);
        for node in nodes.iter().filter(|n| matches!(n.kind, Kind::Player(q) if q as usize == p)) {
            let kind = |c: usize| nodes.get(c).map(|n| n.kind);
            let fold = node.children().find(|&c| matches!(kind(c), Some(Kind::Fold(_))));
            let showdown = node.children().find(|&c| matches!(kind(c), Some(Kind::Showdown)));
            let (Some(fold), Some(showdown)) = (fold, showdown) else { continue };
            if !batched(fold) || !batched(showdown) {
                continue;
            }
            let (seg, k) = at(&showdown_at, showdown, "showdown index")?;
            let entry = pair_of
                .get_mut(seg as usize)
                .and_then(|v| v.get_mut(k as usize))
                .ok_or_else(|| invalid("showdown out of range"))?;
            *entry = to_u32(fold, "fold")?;
            *paired.get_mut(fold).ok_or_else(|| invalid("fold out of range"))? = true;
        }

        // Concatenate the segments' terminal lists.
        for ((t, pairs), segment) in terminals.into_iter().zip(pair_of).zip(self.segments.iter_mut()) {
            let fold_start = self.folds.len();
            for (&id, &pay) in t.folds.iter().zip(&t.fold_payoff) {
                if !at(&paired, id as usize, "paired")? {
                    self.folds.push(id);
                    self.fold_payoff.push(pay);
                }
            }
            segment.folds = Span::of(fold_start, self.folds.len(), "folds")?;
            let showdown_start = self.showdowns.len();
            for &fold in &pairs {
                let pay = match fold {
                    NONE => 0.0,
                    f => {
                        let node = nodes.get(f as usize).ok_or_else(|| invalid("fold out of range"))?;
                        fold_payoff(ctx, node, p)
                    }
                };
                self.showdown_fold.push(fold);
                self.showdown_fold_payoff.push(pay);
            }
            self.showdowns.extend_from_slice(&t.showdowns);
            self.showdown_table.extend_from_slice(&t.table);
            self.win.extend_from_slice(&t.win);
            self.lose.extend_from_slice(&t.lose);
            self.tie.extend_from_slice(&t.tie);
            segment.showdowns = Span::of(showdown_start, self.showdowns.len(), "showdowns")?;
        }
        Ok(())
    }
}

/// Checks the plan-level segmentation: segments form the top of the tree
/// plus groups hanging off the root street's chance nodes, and value rows
/// are in the right region and unique where they must be.
pub(super) fn validate_segments(plan: &Plan) -> Result<(), GpuError> {
    let seg = Segmentation {
        segment_of: plan.segment_of.clone(),
        num_segments: plan.num_segments,
        row: plan.row.clone(),
        top_rows: plan.top_rows,
        num_rows: plan.num_rows,
    };
    let parent = plan.parents()?;
    segments::validate(&seg, plan.num_nodes as usize, &parent, &|i| Ok(plan.node(i)?.actor == CHANCE))
}

/// Checks player `p`'s schedule: every node scheduled once in its segment,
/// levels consistent, reach slots consistent with each parent's role and
/// within the segment's region, terminal lists complete.
pub(super) fn validate_player_schedule(plan: &Plan, p: usize, pp: &PlayerPlan) -> Result<(), GpuError> {
    let nn = plan.num_nodes as usize;
    let (n, nq) = (pp.hands, pp.opp_hands);
    to_u32(pp.num_slots as usize * nq as usize, "reach buffer")?;
    to_u32(plan.num_rows as usize * n as usize, "value buffer")?;
    ensure(pp.slot.len() == nn && pp.child_slot.len() == nn, || "slot arrays have the wrong length".into())?;
    ensure(pp.slot.first() == Some(&0), || "root must use slot 0".into())?;
    ensure(pp.top_slots >= 1 && pp.top_slots <= pp.num_slots, || "slot regions".into())?;
    ensure(pp.segments.len() == plan.num_segments as usize, || "segment count".into())?;

    // Segment spans: contiguous, covering every list.
    let (mut lv, mut fo, mut sd) = (0u64, 0u64, 0u64);
    for (s, seg) in pp.segments.iter().enumerate() {
        ensure(u64::from(seg.levels.start) == lv && u64::from(seg.folds.start) == fo && u64::from(seg.showdowns.start) == sd, || {
            format!("segment {s}: spans aren't contiguous")
        })?;
        (lv, fo, sd) = (seg.levels.end(), seg.folds.end(), seg.showdowns.end());
        let fresh = seg.fresh_slots;
        if s == 0 {
            ensure(fresh.start == 1 && fresh.end() == u64::from(pp.top_slots), || "top segment's slots".into())?;
        } else {
            ensure(fresh.start == pp.top_slots && fresh.end() <= u64::from(pp.num_slots), || {
                format!("segment {s}: slots outside the group region")
            })?;
        }
    }
    ensure(lv == pp.levels.len() as u64 && fo == pp.folds.len() as u64 && sd == pp.showdowns.len() as u64, || {
        "segments don't cover the lists".into()
    })?;

    // Slots: in range; children consistent with their parent's role; fresh
    // child slots inside the parent's segment region, never handed out
    // twice within a segment.
    let mut fresh_used: Vec<(u32, u32)> = Vec::new();
    for i in 0..nn {
        let slot = at(&pp.slot, i, "slot")?;
        ensure(slot < pp.num_slots, || format!("node {i}: slot out of range"))?;
        let node = plan.node(i)?;
        if node.actor == NONE {
            continue;
        }
        let seg = at(&plan.segment_of, i, "segment_of")?;
        let region = pp.segments.get(seg as usize).ok_or_else(|| invalid("segment out of range"))?.fresh_slots;
        let child_slot = at(&pp.child_slot, i, "child_slot")?;
        if node.actor < 2 && node.actor as usize == p {
            ensure(child_slot == NONE, || format!("own node {i} has child slots"))?;
            for c in plan.children(i)? {
                ensure(at(&pp.slot, c, "slot")? == slot, || format!("own node {i}: child {c} doesn't share its slot"))?;
            }
        } else {
            let end = u64::from(child_slot) + u64::from(node.actions);
            ensure(child_slot != NONE && child_slot >= region.start && end <= region.end(), || {
                format!("opponent or chance node {i}: child slots outside its segment's region")
            })?;
            for (a, c) in plan.children(i)?.enumerate() {
                ensure(u64::from(at(&pp.slot, c, "slot")?) == u64::from(child_slot) + a as u64, || {
                    format!("opponent or chance node {i}: child {c} has the wrong slot")
                })?;
                fresh_used.push((seg, child_slot + a as u32));
            }
        }
    }
    fresh_used.sort_unstable();
    ensure(fresh_used.windows(2).all(|w| w.first() != w.get(1)), || "a reach slot handed out twice in a segment".into())?;

    // Fused opponent nodes: only below the traverser's own nodes (whose
    // update reads them), never the root, and never with fused children
    // (the sum reaches one level down).
    ensure(pp.fused.len() == nn, || "fused flags have the wrong length".into())?;
    let parent = plan.parents()?;
    for (i, &f) in pp.fused.iter().enumerate() {
        ensure(f <= 1, || format!("node {i}: fused flag {f}"))?;
        if f == 0 {
            continue;
        }
        let actor = plan.node(i)?.actor;
        ensure(actor < 2 && actor as usize != p, || format!("fused node {i} isn't an opponent node"))?;
        let pa = at(&parent, i, "parent")?;
        ensure(pa != NONE, || "the root can't be fused".into())?;
        let parent_actor = plan.node(pa as usize)?.actor;
        ensure(parent_actor < 2 && parent_actor as usize == p, || {
            format!("fused node {i}: its parent isn't the traverser's node")
        })?;
        for c in plan.children(i)? {
            ensure(at(&pp.fused, c, "fused")? == 0, || format!("fused node {i} has a fused child {c}"))?;
        }
    }

    // Levels: every non-terminal node once, in its segment and the list of
    // its role; children in the same segment one level deeper; chunk roots
    // at level 0 of their group.  Each level's opponent nodes: the summed
    // ones, then the fused ones.
    let role = |actor: u32| -> Role {
        match actor {
            CHANCE => Role::Chance,
            a if a as usize == p => Role::Own,
            _ => Role::Opponent,
        }
    };
    let mut level_of = vec![NONE; nn];
    let (mut expect_opp, mut expect_own, mut expect_chance) = (0u32, 0u32, 0u32);
    for (s, seg) in pp.segments.iter().enumerate() {
        let levels = pp.levels.get(seg.levels.range()).ok_or_else(|| invalid("levels out of range"))?;
        for (d, lv) in levels.iter().enumerate() {
            ensure(lv.opp_start == expect_opp && lv.own_start == expect_own && lv.chance_start == expect_chance, || {
                format!("segment {s}, level {d} isn't contiguous")
            })?;
            expect_opp = lv.opp_start.checked_add(lv.opp_count).ok_or_else(|| invalid("level overflow"))?;
            expect_own = lv.own_start.checked_add(lv.own_count).ok_or_else(|| invalid("level overflow"))?;
            expect_chance = lv.chance_start.checked_add(lv.chance_count).ok_or_else(|| invalid("level overflow"))?;
            ensure(lv.opp_sum_count <= lv.opp_count, || format!("segment {s}, level {d}: opponent sum count"))?;
            let opp = pp.opp_nodes.get(lv.opp_start as usize..expect_opp as usize);
            let own = pp.own_nodes.get(lv.own_start as usize..expect_own as usize);
            let chance = pp.chance_nodes.get(lv.chance_start as usize..expect_chance as usize);
            let (Some(opp), Some(own), Some(chance)) = (opp, own, chance) else {
                return Err(invalid(format!("segment {s}, level {d} out of range")));
            };
            for (k, &id) in opp.iter().enumerate() {
                let want = u32::from(k as u64 >= u64::from(lv.opp_sum_count));
                ensure(at(&pp.fused, id as usize, "fused")? == want, || {
                    format!("segment {s}, level {d}: opponent node {id} in the wrong part of the level")
                })?;
            }
            let listed = opp
                .iter()
                .map(|x| (x, Role::Opponent))
                .chain(own.iter().map(|x| (x, Role::Own)))
                .chain(chance.iter().map(|x| (x, Role::Chance)));
            for (&id, list_role) in listed {
                let i = id as usize;
                let actor = plan.node(i)?.actor;
                ensure(actor != NONE && role(actor) == list_role, || format!("node {id} in the wrong list"))?;
                ensure(at(&plan.segment_of, i, "segment_of")? as usize == s, || format!("node {id} in the wrong segment"))?;
                let entry = level_of.get_mut(i).ok_or_else(|| invalid(format!("node {id} out of range")))?;
                ensure(*entry == NONE, || format!("node {id} scheduled twice"))?;
                *entry = d as u32;
            }
        }
    }
    ensure(
        expect_opp as usize == pp.opp_nodes.len()
            && expect_own as usize == pp.own_nodes.len()
            && expect_chance as usize == pp.chance_nodes.len(),
        || "node lists have unscheduled entries".into(),
    )?;

    // Terminals: each once, in its segment's lists.
    let mut terminal_seen = vec![false; nn];
    for (s, seg) in pp.segments.iter().enumerate() {
        let folds = pp.folds.get(seg.folds.range()).ok_or_else(|| invalid("folds out of range"))?;
        let showdowns = pp.showdowns.get(seg.showdowns.range()).ok_or_else(|| invalid("showdowns out of range"))?;
        let paired = pp.showdown_fold.get(seg.showdowns.range()).ok_or_else(|| invalid("pairs out of range"))?;
        for &id in folds.iter().chain(showdowns).chain(paired.iter().filter(|&&f| f != NONE)) {
            let i = id as usize;
            ensure(plan.node(i)?.actor == NONE, || format!("terminal list entry {id} isn't a terminal"))?;
            ensure(at(&plan.segment_of, i, "segment_of")? as usize == s, || format!("terminal {id} in the wrong segment"))?;
            let seen = terminal_seen.get_mut(i).ok_or_else(|| invalid(format!("terminal {id} out of range")))?;
            ensure(!*seen, || format!("terminal {id} listed twice"))?;
            *seen = true;
        }
    }
    for i in 0..nn {
        let level = at(&level_of, i, "level")?;
        if plan.node(i)?.actor == NONE {
            ensure(at(&terminal_seen, i, "terminal")?, || format!("terminal {i} not listed"))?;
            continue;
        }
        ensure(level != NONE, || format!("decision node {i} not scheduled"))?;
        let seg = at(&plan.segment_of, i, "segment_of")?;
        for c in plan.children(i)? {
            if plan.node(c)?.actor == NONE {
                continue;
            }
            let child_level = at(&level_of, c, "level")?;
            if at(&plan.segment_of, c, "segment_of")? == seg {
                ensure(child_level == level + 1, || format!("node {c} isn't one level below its parent {i}"))?;
            } else {
                ensure(child_level == 0, || format!("chunk root {c} isn't at its group's first level"))?;
            }
        }
    }
    Ok(())
}
