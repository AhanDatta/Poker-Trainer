//! Terminal-node evaluation in O(n) per node.
//!
//! Card removal is handled by inclusion–exclusion: for a hand `{a, b}` the
//! compatible opponent reach is `total - minus[a] - minus[b] + reach(same)`,
//! where `minus[c]` is the reach of opponent hands containing card `c` and
//! `same` is the identical combo (subtracted twice, so added back once).
//! Showdowns sweep both players' hands sorted by strength.

use crate::hands::{HandSet, StrengthTable, NONE};

/// One hand in a strength-sorted list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Entry {
    pub strength: u32,
    pub idx: u16,
    pub c1: u8,
    pub c2: u8,
}

#[inline(always)]
fn accumulate(c1: u8, c2: u8, w: f32, sum: &mut f64, minus: &mut [f64; 52]) {
    if w != 0.0 {
        let w = w as f64;
        *sum += w;
        // SAFETY: card indices are < 52.
        unsafe {
            *minus.get_unchecked_mut(c1 as usize) += w;
            *minus.get_unchecked_mut(c2 as usize) += w;
        }
    }
}

#[inline(always)]
fn minus2(minus: &[f64; 52], c1: u8, c2: u8) -> f64 {
    // SAFETY: card indices are < 52.
    unsafe { minus.get_unchecked(c1 as usize) + minus.get_unchecked(c2 as usize) }
}

/// `out[i] = payoff * (compatible opponent reach of hand i)`.
pub(crate) fn eval_fold(hs: &HandSet, p: usize, opp_reach: &[f32], payoff: f64, out: &mut [f32]) {
    let o = p ^ 1;
    let opp_cards = &hs.cards[o];
    debug_assert_eq!(opp_cards.len(), opp_reach.len());
    let mut total = 0f64;
    let mut minus = [0f64; 52];
    for (&(a, b), &w) in opp_cards.iter().zip(opp_reach) {
        accumulate(a, b, w, &mut total, &mut minus);
    }
    if total == 0.0 {
        out.fill(0.0);
        return;
    }
    let my_cards = &hs.cards[p];
    let same = &hs.same[p];
    for ((o, &(a, b)), &sm) in out.iter_mut().zip(my_cards).zip(same) {
        let s = if sm == NONE {
            0.0
        } else {
            // SAFETY: `same` indices point into the opponent's hand list.
            unsafe { *opp_reach.get_unchecked(sm as usize) as f64 }
        };
        *o = (payoff * (total - minus2(&minus, a, b) + s)) as f32;
    }
}

/// Showdown values: `win`/`lose`/`tie` are the (normalized) payoffs of
/// player `p` for each outcome.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_showdown(
    hs: &HandSet,
    table: &StrengthTable,
    p: usize,
    opp_reach: &[f32],
    win: f64,
    lose: f64,
    tie: f64,
    out: &mut [f32],
) {
    out.fill(0.0);
    let o = p ^ 1;
    let mine: &[Entry] = &table.lists[p];
    let theirs: &[Entry] = &table.lists[o];
    let reach = |e: &Entry| -> f32 {
        // SAFETY: entry indices are valid hand indices of the opponent.
        unsafe { *opp_reach.get_unchecked(e.idx as usize) }
    };
    // SAFETY (all unchecked accesses below): `k < mine.len()`,
    // `j < theirs.len()` are checked by the loop conditions and entry
    // indices are valid hand indices of the respective player.

    // Pass 1 (ascending): opponent hands strictly weaker than ours.
    let mut sum = 0f64;
    let mut minus = [0f64; 52];
    let mut j = 0;
    let mut k = 0;
    while k < mine.len() {
        let s = unsafe { mine.get_unchecked(k).strength };
        while j < theirs.len() {
            let e = unsafe { theirs.get_unchecked(j) };
            if e.strength >= s {
                break;
            }
            accumulate(e.c1, e.c2, reach(e), &mut sum, &mut minus);
            j += 1;
        }
        while k < mine.len() {
            let e = unsafe { mine.get_unchecked(k) };
            if e.strength != s {
                break;
            }
            unsafe {
                *out.get_unchecked_mut(e.idx as usize) = (win * (sum - minus2(&minus, e.c1, e.c2))) as f32;
            }
            k += 1;
        }
    }

    // Pass 2 (descending): opponent hands strictly stronger than ours.
    sum = 0.0;
    minus = [0f64; 52];
    let mut j = theirs.len();
    let mut k = mine.len();
    while k > 0 {
        let s = unsafe { mine.get_unchecked(k - 1).strength };
        while j > 0 {
            let e = unsafe { theirs.get_unchecked(j - 1) };
            if e.strength <= s {
                break;
            }
            accumulate(e.c1, e.c2, reach(e), &mut sum, &mut minus);
            j -= 1;
        }
        while k > 0 {
            let e = unsafe { mine.get_unchecked(k - 1) };
            if e.strength != s {
                break;
            }
            unsafe {
                *out.get_unchecked_mut(e.idx as usize) += (lose * (sum - minus2(&minus, e.c1, e.c2))) as f32;
            }
            k -= 1;
        }
    }

    if tie == 0.0 {
        return;
    }
    // Pass 3: opponent hands of exactly equal strength.
    let same = &hs.same[p];
    let mut j = 0;
    let mut k = 0;
    while k < mine.len() {
        let s = mine[k].strength;
        while j < theirs.len() && theirs[j].strength < s {
            j += 1;
        }
        let mut eq_sum = 0f64;
        let mut eq_minus = [0f64; 52];
        let mut jj = j;
        while jj < theirs.len() && theirs[jj].strength == s {
            let e = &theirs[jj];
            accumulate(e.c1, e.c2, reach(e), &mut eq_sum, &mut eq_minus);
            jj += 1;
        }
        while k < mine.len() && mine[k].strength == s {
            let e = &mine[k];
            let i = e.idx as usize;
            let same_w = match same[i] {
                NONE => 0.0,
                q => opp_reach[q as usize] as f64,
            };
            let eq = eq_sum - minus2(&eq_minus, e.c1, e.c2) + same_w;
            out[i] += (tie * eq) as f32;
            k += 1;
        }
    }
}
