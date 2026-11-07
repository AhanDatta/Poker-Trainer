//! Plan data of the leaf kernels (`cuda/kernels/multi_leaves.cu`), and its
//! validation.
//!
//! The CPU evaluates a leaf with sweeps over the opponents' hands: in hand
//! order for the compatible mass (`Hands::compat`, and the pair sums of
//! `exact::fold_values`), in a board's strength order for showdowns
//! (`Table::beat`, `exact::showdown`), each traverser hand reading the
//! sums at its turn. The order of a sweep depends only on the board and
//! the folded players, so the kernels run one leaf per GPU lane, a warp's
//! 32 lanes in lockstep through the same sweep with their own reach, in the
//! CPU's operation order (so the f64 path is bit-identical, rounding noise
//! included). What they need:
//! * [`SweepLists`]: every seat's hand list in each order (one per final
//!   board, then hand order), each with per-card position lists (the exact
//!   model's `by_card`, in the order hands enter) and a combo-to-position
//!   map (its `x`, and the identical combo of the independent model), so
//!   that "is this combo in the set" is a position test;
//! * per traverser ([`LeafGroups`]): showdowns grouped into warps that
//!   share a board and folded players, and fixed-payoff leaves grouped by
//!   their pair of reach slots (leaves with the same reach have the same
//!   value per unit of payoff; `river::subtree` groups them the same way).

use super::{at, ensure, nth, pack_cards, to_u32, MultiPlan, MultiPlayerPlan, CARDS};
use crate::gpu::shared::{unpack_cards, MAX_HANDS, MAX_PAYOFF};
use crate::gpu::{invalid, GpuError};
use crate::multi::cfr::Ctx;
use crate::multi::N;
use holdem::combo::NUM_COMBOS;

/// Leaves per warp (`kLanes` in lanes.cuh).
pub(crate) const LEAF_LANES: usize = 32;

/// "Not in this list" in the 16-bit position arrays (`kNoPosition`).
pub(crate) const NO_POSITION: u16 = u16::MAX;

/// Entries of a list's card-offset array: one per card, plus the end.
const CARD_OFFSETS: usize = CARDS + 1;

/// `holdem::combo::combo_index` on card indices (the kernels' `combo_index`).
pub(crate) fn combo_of(a: usize, b: usize) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    hi * hi.saturating_sub(1) / 2 + lo
}

/// Every seat's hand list in every sweep order. List `k = order * N +
/// seat`; orders `0..num_tables` are the final boards' strength orders
/// (`Table::lists`, hands on the board left out), order `num_tables` is
/// hand order (every hand).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SweepLists {
    pub orders: u32,
    /// Per list: its entries are `off[k]..off[k + 1]`.
    pub off: Vec<u32>,
    /// Per entry: the hand's strength on the board (0 in hand order), its
    /// index in the seat's hand list, and its cards (packed, in the order
    /// the CPU reads them).
    pub strength: Vec<u32>,
    pub hand: Vec<u32>,
    pub cards: Vec<u32>,
    /// Per list, per card: `card_off[k * 53 + c]..card_off[k * 53 + c + 1]`
    /// is the range of `holding` listing the entries with card `c`, by
    /// ascending position within the list: `position | other card << 16`.
    pub card_off: Vec<u32>,
    pub holding: Vec<u32>,
    /// Per list, per combo: its position in the list, or [`NO_POSITION`].
    pub combo_pos: Vec<u16>,
}

impl SweepLists {
    pub(crate) fn build(ctx: &Ctx) -> Result<SweepLists, GpuError> {
        let mut lists = SweepLists { orders: to_u32(ctx.tables.len() + 1, "sweep orders")?, ..SweepLists::default() };
        lists.off.push(0);
        for table in ctx.tables {
            for seat in 0..N {
                let entries = nth(&table.lists, seat, "player")?.iter().map(|&(s, h, a, b)| (s, h, a, b));
                lists.push(entries)?;
            }
        }
        for seat in 0..N {
            let cards = nth(&ctx.hands.cards, seat, "player")?;
            lists.push(cards.iter().enumerate().map(|(i, &(a, b))| (0, i as u32, a, b)))?;
        }
        Ok(lists)
    }

    fn push(&mut self, entries: impl Iterator<Item = (u32, u32, u8, u8)>) -> Result<(), GpuError> {
        let mut by_card = vec![Vec::new(); CARDS];
        let mut combo_pos = vec![NO_POSITION; NUM_COMBOS];
        for (pos, (strength, hand, a, b)) in entries.enumerate() {
            let pos16 = u16::try_from(pos).ok().filter(|&x| x != NO_POSITION).ok_or_else(|| invalid("list too long"))?;
            self.strength.push(strength);
            self.hand.push(hand);
            self.cards.push(pack_cards(a, b));
            for (c, other) in [(a, b), (b, a)] {
                let entry = u32::from(pos16) | u32::from(other) << 16;
                by_card.get_mut(c as usize).ok_or_else(|| invalid(format!("card {c}")))?.push(entry);
            }
            *combo_pos.get_mut(combo_of(a.into(), b.into())).ok_or_else(|| invalid("combo"))? = pos16;
        }
        for held in &by_card {
            self.card_off.push(to_u32(self.holding.len(), "card lists")?);
            self.holding.extend_from_slice(held);
        }
        self.card_off.push(to_u32(self.holding.len(), "card lists")?);
        self.combo_pos.extend_from_slice(&combo_pos);
        self.off.push(to_u32(self.hand.len(), "list entries")?);
        Ok(())
    }

    /// Number of lists.
    pub(crate) fn lists(&self) -> usize {
        self.orders as usize * N
    }

    /// Checks every index the kernels read: list `k`'s entries hold hands
    /// of seat `k % N` with those hands' cards (`players[seat].my_cards`),
    /// strength-sorted in board orders and the identity in hand order; the
    /// card and combo maps point at exactly the right entries.
    pub(crate) fn validate(&self, num_tables: u32, players: &[MultiPlayerPlan; N]) -> Result<(), GpuError> {
        ensure(u64::from(self.orders) == u64::from(num_tables) + 1, || "one sweep order per board plus hand order".into())?;
        let lists = self.lists();
        let entries = self.hand.len();
        ensure(self.off.len() == lists + 1 && self.off.first() == Some(&0), || "list offsets".into())?;
        ensure(self.off.windows(2).all(|w| w.first() <= w.get(1)), || "list offsets out of order".into())?;
        ensure(self.off.last().is_some_and(|&e| e as usize == entries), || "list offsets don't end at the entries".into())?;
        ensure(self.strength.len() == entries && self.cards.len() == entries, || "entry arrays".into())?;
        ensure(self.card_off.len() == lists * CARD_OFFSETS, || "card offsets".into())?;
        ensure(self.combo_pos.len() == lists * NUM_COMBOS, || "combo positions".into())?;
        let mut card_end = 0usize;
        for k in 0..lists {
            let (order, seat) = (k / N, k % N);
            let player = nth(players, seat, "player")?;
            let (start, end) = (at(&self.off, k, "off")? as usize, at(&self.off, k + 1, "off")? as usize);
            let len = end - start;
            let hand_order = order as u32 == num_tables;
            let what = || format!("list {k} (order {order}, seat {seat})");
            ensure(len <= MAX_HANDS as usize && len <= player.hands as usize, || format!("{}: {len} entries", what()))?;
            if hand_order {
                ensure(len == player.hands as usize, || format!("{}: not every hand", what()))?;
            }
            let combo_pos = self
                .combo_pos
                .get(k * NUM_COMBOS..(k + 1) * NUM_COMBOS)
                .ok_or_else(|| invalid("combo positions out of range"))?;
            let mut last_strength = 0u32;
            for pos in 0..len {
                let e = start + pos;
                let hand = at(&self.hand, e, "hand")?;
                let cards = at(&self.cards, e, "cards")?;
                ensure(hand < player.hands && at(&player.my_cards, hand as usize, "my_cards")? == cards, || {
                    format!("{}: entry {pos} doesn't match the seat's hand {hand}", what())
                })?;
                ensure(!hand_order || hand as usize == pos, || format!("{}: hand order broken at {pos}", what()))?;
                let strength = at(&self.strength, e, "strength")?;
                ensure(strength >= last_strength && (!hand_order || strength == 0), || {
                    format!("{}: strengths out of order at {pos}", what())
                })?;
                last_strength = strength;
                let (a, b) = unpack_cards(cards);
                ensure(at(combo_pos, combo_of(a, b), "combo_pos")? as usize == pos, || {
                    format!("{}: combo of entry {pos} doesn't map back to it", what())
                })?;
            }
            // Every mapped combo is an entry with those cards (each entry's
            // combo maps back to it, so the map is a bijection onto the list).
            let mapped = combo_pos.iter().filter(|&&x| x != NO_POSITION).count();
            ensure(mapped == len, || format!("{}: {mapped} combos map into {len} entries", what()))?;
            // Per-card entries: contiguous with the previous list, ascending,
            // in range, holding the card and naming the other one; 2 per
            // entry.
            let offs = self
                .card_off
                .get(k * CARD_OFFSETS..(k + 1) * CARD_OFFSETS)
                .ok_or_else(|| invalid("card offsets out of range"))?;
            ensure(offs.first().is_some_and(|&o| o as usize == card_end), || format!("{}: card lists not contiguous", what()))?;
            ensure(offs.windows(2).all(|w| w.first() <= w.get(1)), || format!("{}: card offsets out of order", what()))?;
            for c in 0..CARDS {
                let (b, e) = (at(offs, c, "card_off")? as usize, at(offs, c + 1, "card_off")? as usize);
                let held = self.holding.get(b..e).ok_or_else(|| invalid(format!("{}: card list out of range", what())))?;
                let position = |e: &u32| (e & 0xFFFF) as usize;
                ensure(held.windows(2).all(|w| w.first().map(position) < w.get(1).map(position)), || {
                    format!("{}: card {c}'s positions not ascending", what())
                })?;
                for &entry in held {
                    let (pos, other) = (position(&entry), (entry >> 16) as usize);
                    ensure(pos < len, || format!("{}: card {c}'s list points past the end", what()))?;
                    let (x, y) = unpack_cards(at(&self.cards, start + pos, "cards")?);
                    ensure((x, y) == (c, other) || (x, y) == (other, c), || {
                        format!("{}: card {c}'s entry {pos} isn't the combo with card {other}", what())
                    })?;
                }
            }
            card_end = at(offs, CARDS, "card_off")? as usize;
            ensure(card_end - at(offs, 0, "card_off")? as usize == 2 * len, || {
                format!("{}: card lists don't hold every entry twice", what())
            })?;
        }
        ensure(card_end == self.holding.len(), || "card lists past the last list".into())
    }

    /// Device bytes.
    pub(crate) fn device_bytes(&self) -> u64 {
        fn b<T>(v: &[T]) -> u64 {
            (v.len().max(1) * std::mem::size_of::<T>()) as u64
        }
        b(&self.off) + b(&self.strength) + b(&self.hand) + b(&self.cards) + b(&self.card_off) + b(&self.holding) + b(&self.combo_pos)
    }
}

/// One traverser's leaf groups.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LeafGroups {
    /// Showdown warps, 4 values each: the sweep order (the board's table),
    /// the folded players (bit per seat), and the range of
    /// `showdown_lanes` they evaluate (at most [`LEAF_LANES`]).
    pub groups: Vec<u32>,
    /// Indices into the traverser's `showdowns`, warp by warp.
    pub showdown_lanes: Vec<u32>,
    /// Per showdown: the payoff when tied with one opponent and with both,
    /// `lose + (win - lose) / k` (as `exact::showdown_lanes` computes it;
    /// unused with independent opponents).
    pub split2: Vec<f64>,
    pub split3: Vec<f64>,
    /// Fixed-payoff jobs, one per distinct pair of reach slots (warp `w`
    /// runs jobs `32 w..32 w + 32`): the slots, and the range of
    /// `job_leaves` (indices into the traverser's `fixed`) sharing them.
    pub job_slots: [Vec<u32>; 2],
    pub job_off: Vec<u32>,
    pub job_leaves: Vec<u32>,
    /// Per fixed leaf: 1 if the CPU rounds `pay * pairs` once, 0 if it
    /// rounds the pair mass to f32 first (batched river leaves; the plan's
    /// `scalar`).
    pub fixed_scalar: Vec<u32>,
}

/// Values per showdown group.
pub(crate) const GROUP_FIELDS: usize = 4;

impl LeafGroups {
    pub(crate) fn build(plan: &MultiPlan, pp: &MultiPlayerPlan) -> Result<LeafGroups, GpuError> {
        let mut x = LeafGroups::default();
        // Showdowns: by board, then folded players, then list order.
        let mut keyed = Vec::with_capacity(pp.showdowns.len());
        for (i, (&node, &table)) in pp.showdowns.iter().zip(&pp.showdown_table).enumerate() {
            keyed.push((table, at(&plan.folded, node as usize, "folded")?, to_u32(i, "showdowns")?));
        }
        keyed.sort_unstable();
        for chunk in keyed.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
            for warp in chunk.chunks(LEAF_LANES) {
                let first = warp.first().ok_or_else(|| invalid("empty showdown group"))?;
                x.groups.extend([first.0, first.1, to_u32(x.showdown_lanes.len(), "showdown lanes")?, to_u32(warp.len(), "lanes")?]);
                x.showdown_lanes.extend(warp.iter().map(|e| e.2));
            }
        }
        for (&win, &lose) in pp.win.iter().zip(&pp.lose) {
            x.split2.push(split(win, lose, 2.0));
            x.split3.push(split(win, lose, 3.0));
        }
        // Fixed leaves: by their pair of reach slots.
        let mut keyed = Vec::with_capacity(pp.fixed.len());
        for (i, &node) in pp.fixed.iter().enumerate() {
            let slots = [at(&pp.slot[0], node as usize, "slot")?, at(&pp.slot[1], node as usize, "slot")?];
            keyed.push((slots, to_u32(i, "fixed leaves")?));
            x.fixed_scalar.push(at(&plan.scalar, node as usize, "scalar")?);
        }
        keyed.sort_unstable();
        x.job_off.push(0);
        for job in keyed.chunk_by(|a, b| a.0 == b.0) {
            let first = job.first().ok_or_else(|| invalid("empty fixed job"))?;
            for (slots, &s) in x.job_slots.iter_mut().zip(&first.0) {
                slots.push(s);
            }
            x.job_leaves.extend(job.iter().map(|e| e.1));
            x.job_off.push(to_u32(x.job_leaves.len(), "fixed jobs")?);
        }
        Ok(x)
    }

    /// Number of fixed-payoff jobs.
    pub(crate) fn jobs(&self) -> usize {
        self.job_off.len().saturating_sub(1)
    }

    /// Number of showdown warps.
    pub(crate) fn showdown_warps(&self) -> usize {
        self.groups.len() / GROUP_FIELDS
    }

    /// Number of fixed-payoff warps.
    pub(crate) fn fixed_warps(&self) -> usize {
        self.jobs().div_ceil(LEAF_LANES)
    }

    /// Checks traverser `p`'s groups: every showdown in exactly one warp of
    /// its board and folded players, every fixed leaf in exactly one job of
    /// its reach slots, the split payoffs and rounding flags as built.
    pub(crate) fn validate(&self, plan: &MultiPlan, pp: &MultiPlayerPlan, p: usize) -> Result<(), GpuError> {
        let ns = pp.showdowns.len();
        let (groups, rest) = self.groups.as_chunks::<GROUP_FIELDS>();
        ensure(rest.is_empty(), || "showdown groups".into())?;
        ensure(self.showdown_lanes.len() == ns, || "showdown lanes".into())?;
        let mut seen = vec![false; ns];
        let mut next = 0u64;
        for (g, &[order, folded, start, count]) in groups.iter().enumerate() {
            ensure(order < plan.num_tables, || format!("showdown group {g}: board {order} out of range"))?;
            ensure(folded >> p & 1 == 0 && folded < 1 << N, || format!("showdown group {g}: folded {folded:#b}"))?;
            ensure(u64::from(start) == next && (1..=LEAF_LANES as u32).contains(&count), || {
                format!("showdown group {g}: lanes {start}+{count}")
            })?;
            next += u64::from(count);
            for l in start..start + count {
                let i = at(&self.showdown_lanes, l as usize, "showdown_lanes")? as usize;
                let entry = seen.get_mut(i).ok_or_else(|| invalid(format!("showdown group {g}: showdown {i}")))?;
                ensure(!*entry, || format!("showdown {i} in two groups"))?;
                *entry = true;
                let node = at(&pp.showdowns, i, "showdowns")? as usize;
                ensure(at(&pp.showdown_table, i, "showdown_table")? == order && at(&plan.folded, node, "folded")? == folded, || {
                    format!("showdown group {g}: showdown {i} has another board or folded players")
                })?;
            }
        }
        ensure(next == ns as u64, || "showdowns left out of the groups".into())?;
        ensure(self.split2.len() == ns && self.split3.len() == ns, || "split payoffs".into())?;
        for i in 0..ns {
            let (win, lose) = (at(&pp.win, i, "win")?, at(&pp.lose, i, "lose")?);
            let (s2, s3) = (at(&self.split2, i, "split2")?, at(&self.split3, i, "split3")?);
            ensure(s2.to_bits() == split(win, lose, 2.0).to_bits() && s3.to_bits() == split(win, lose, 3.0).to_bits(), || {
                format!("showdown {i}: split payoffs")
            })?;
            ensure(s2.abs() <= MAX_PAYOFF && s3.abs() <= MAX_PAYOFF, || format!("showdown {i}: split payoff too large"))?;
        }

        let nf = pp.fixed.len();
        let jobs = self.jobs();
        ensure(self.job_off.first() == Some(&0) && self.job_off.last().is_some_and(|&e| e as usize == nf), || {
            "fixed job offsets".into()
        })?;
        ensure(self.job_leaves.len() == nf && self.fixed_scalar.len() == nf, || "fixed job arrays".into())?;
        let mut seen = vec![false; nf];
        for k in 0..2 {
            let slots = nth(&self.job_slots, k, "opponent")?;
            ensure(slots.len() == jobs, || "fixed job slots".into())?;
            let num = *nth(&pp.num_slots, k, "opponent")?;
            ensure(slots.iter().all(|&s| s < num), || format!("fixed job slot of opponent {k} out of range"))?;
        }
        for j in 0..jobs {
            let (b, e) = (at(&self.job_off, j, "job_off")? as usize, at(&self.job_off, j + 1, "job_off")? as usize);
            ensure(b < e, || format!("fixed job {j} is empty"))?;
            let leaves = self.job_leaves.get(b..e).ok_or_else(|| invalid(format!("fixed job {j} out of range")))?;
            for &i in leaves {
                let i = i as usize;
                let entry = seen.get_mut(i).ok_or_else(|| invalid(format!("fixed job {j}: leaf {i}")))?;
                ensure(!*entry, || format!("fixed leaf {i} in two jobs"))?;
                *entry = true;
                let node = at(&pp.fixed, i, "fixed")? as usize;
                for k in 0..2 {
                    ensure(at(nth(&pp.slot, k, "opponent")?, node, "slot")? == at(nth(&self.job_slots, k, "opponent")?, j, "job_slots")?, || {
                        format!("fixed job {j}: leaf {i} has other reach")
                    })?;
                }
            }
        }
        for (i, &node) in pp.fixed.iter().enumerate() {
            ensure(at(&self.fixed_scalar, i, "fixed_scalar")? == at(&plan.scalar, node as usize, "scalar")?, || {
                format!("fixed leaf {i}: rounding flag")
            })?;
        }
        Ok(())
    }

    /// Device bytes.
    pub(crate) fn device_bytes(&self) -> u64 {
        fn b<T>(v: &[T]) -> u64 {
            (v.len().max(1) * std::mem::size_of::<T>()) as u64
        }
        b(&self.groups)
            + b(&self.showdown_lanes)
            + b(&self.split2)
            + b(&self.split3)
            + self.job_slots.iter().map(|v| b(v)).sum::<u64>()
            + b(&self.job_off)
            + b(&self.job_leaves)
            + b(&self.fixed_scalar)
    }
}

/// `lose + (win - lose) / k`: the pot split `k` ways (`exact::showdown_lanes`).
fn split(win: f64, lose: f64, k: f64) -> f64 {
    lose + (win - lose) / k
}
