//! Per-board showdown lookup tables: where each of the traverser's hands
//! falls in a board's strength-sorted list of one opponent's hands. Built
//! by the heads-up plan for its one opponent and by the three-player plan
//! for each opponent; read by the showdown kernels.

use super::shared::{at, check_offsets, ensure, to_u32, unpack_cards, CARDS, NONE, SHOWDOWN_SHARED_LIMIT, showdown_shared_bytes};
use super::{invalid, GpuError};
use crate::hands::{HandSet, StrengthTable};

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

/// One hand of a strength-sorted list (weakest first).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ranked {
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
    pub(crate) fn push<E>(
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

impl Tables {
    /// Checks the tables of `num_tables` boards for a traverser with hands
    /// `my_cards` (packed) against an opponent with `opp_hands` hands.
    pub(crate) fn validate(&self, num_tables: u32, my_cards: &[u32], opp_hands: u32) -> Result<(), GpuError> {
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

/// Player `p`'s half of a per-player pair.
fn side<T>(pair: &[T; 2], p: usize) -> Result<&T, GpuError> {
    pair.get(p).ok_or_else(|| invalid(format!("player {p} out of range")))
}
