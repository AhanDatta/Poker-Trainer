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

    if tie != 0.0 {
        add_ties(hs, table, p, opp_reach, tie, out);
    }
}

/// Number of terminals evaluated together by the batched evaluators.
pub(crate) const LANES: usize = 4;
type V = [f64; LANES];

#[inline(always)]
fn vadd(a: &mut V, b: &V) {
    for i in 0..LANES {
        a[i] += b[i];
    }
}

#[inline(always)]
fn nonzero(w: &V) -> bool {
    w.iter().any(|&x| x != 0.0)
}

thread_local! {
    static BATCH: std::cell::RefCell<(Vec<V>, Vec<V>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Fold terminals evaluated together: `outs[t][h] = payoffs[t] * compat(h)`
/// under `reaches[t]` (at most [`LANES`] terminals).
#[inline(never)]
pub(crate) fn fold_batch(
    hs: &HandSet,
    p: usize,
    reaches: &[&[f32]],
    payoffs: &[f64],
    outs: &mut [&mut [f32]],
) {
    let k = reaches.len();
    debug_assert!(k <= LANES && payoffs.len() == k && outs.len() == k);
    let o = p ^ 1;
    let mut total: V = [0.0; LANES];
    let mut minus = [[0f64; LANES]; 52];
    for (j, &(a, b)) in hs.cards[o].iter().enumerate() {
        let mut w: V = [0.0; LANES];
        for t in 0..k {
            w[t] = reaches[t][j] as f64;
        }
        if !nonzero(&w) {
            continue;
        }
        vadd(&mut total, &w);
        vadd(&mut minus[a as usize], &w);
        vadd(&mut minus[b as usize], &w);
    }
    let same = &hs.same[p];
    for (h, &(a, b)) in hs.cards[p].iter().enumerate() {
        let (ma, mb) = (&minus[a as usize], &minus[b as usize]);
        let s = same[h];
        for t in 0..k {
            let sw = if s == NONE { 0.0 } else { reaches[t][s as usize] as f64 };
            outs[t][h] = (payoffs[t] * (total[t] - ma[t] - mb[t] + sw)) as f32;
        }
    }
}

/// Showdown terminals on the same board evaluated together (at most
/// [`LANES`]). `outs` must be zero-initialized. `folds[t]` optionally gives
/// a fold terminal with the same opponent reach as showdown `t` (its payoff
/// and output), which is read off the first pass.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn showdown_batch(
    hs: &HandSet,
    table: &StrengthTable,
    p: usize,
    reaches: &[&[f32]],
    win: &[f64],
    lose: &[f64],
    tie: &[f64],
    outs: &mut [&mut [f32]],
    folds: &mut [Option<(f64, &mut [f32])>],
) {
    let k = reaches.len();
    debug_assert!(k <= LANES && win.len() == k && lose.len() == k && tie.len() == k && outs.len() == k);
    debug_assert!(folds.len() == k);
    let o = p ^ 1;
    let mine: &[Entry] = &table.lists[p];
    let theirs: &[Entry] = &table.lists[o];
    let mut wv: V = [0.0; LANES];
    let mut lv: V = [0.0; LANES];
    wv[..k].copy_from_slice(win);
    lv[..k].copy_from_slice(lose);
    BATCH.with(|cell| {
        let (inter, res) = &mut *cell.borrow_mut();
        // Opponent reach gathered in strength order, so both sweeps read
        // memory sequentially.
        inter.clear();
        inter.extend(theirs.iter().map(|e| {
            let mut w: V = [0.0; LANES];
            for t in 0..k {
                w[t] = reaches[t][e.idx as usize] as f64;
            }
            w
        }));
        res.clear();
        res.resize(mine.len(), [0.0; LANES]);

        // Pass 1 (ascending): opponent hands strictly weaker than ours.
        let mut sum: V = [0.0; LANES];
        let mut minus = [[0f64; LANES]; 52];
        let (mut i, mut j) = (0, 0);
        while i < mine.len() {
            let s = mine[i].strength;
            while j < theirs.len() && theirs[j].strength < s {
                let w = &inter[j];
                if nonzero(w) {
                    let e = &theirs[j];
                    vadd(&mut sum, w);
                    vadd(&mut minus[e.c1 as usize], w);
                    vadd(&mut minus[e.c2 as usize], w);
                }
                j += 1;
            }
            while i < mine.len() && mine[i].strength == s {
                let e = &mine[i];
                let (ma, mb) = (&minus[e.c1 as usize], &minus[e.c2 as usize]);
                let r = &mut res[i];
                for t in 0..LANES {
                    r[t] = wv[t] * (sum[t] - ma[t] - mb[t]);
                }
                i += 1;
            }
        }

        // A fold terminal with the same opponent reach as a showdown is
        // worth `payoff * (all compatible reach)`: finish this pass over the
        // stronger opponent hands and read it off, instead of a separate
        // fold sweep.
        if folds.iter().any(|f| f.is_some()) {
            while j < theirs.len() {
                let w = &inter[j];
                if nonzero(w) {
                    let e = &theirs[j];
                    vadd(&mut sum, w);
                    vadd(&mut minus[e.c1 as usize], w);
                    vadd(&mut minus[e.c2 as usize], w);
                }
                j += 1;
            }
            let same = &hs.same[p];
            for (t, f) in folds.iter_mut().enumerate() {
                let Some((pay, out)) = f else { continue };
                let reach = reaches[t];
                for ((o, &(a, b)), &sm) in out.iter_mut().zip(&hs.cards[p]).zip(same) {
                    let sw = if sm == NONE { 0.0 } else { reach[sm as usize] as f64 };
                    *o = (*pay * (sum[t] - minus[a as usize][t] - minus[b as usize][t] + sw)) as f32;
                }
            }
        }

        // Pass 2 (descending): opponent hands strictly stronger than ours.
        let mut sum: V = [0.0; LANES];
        let mut minus = [[0f64; LANES]; 52];
        let (mut i, mut j) = (mine.len(), theirs.len());
        while i > 0 {
            let s = mine[i - 1].strength;
            while j > 0 && theirs[j - 1].strength > s {
                let w = &inter[j - 1];
                if nonzero(w) {
                    let e = &theirs[j - 1];
                    vadd(&mut sum, w);
                    vadd(&mut minus[e.c1 as usize], w);
                    vadd(&mut minus[e.c2 as usize], w);
                }
                j -= 1;
            }
            while i > 0 && mine[i - 1].strength == s {
                let e = &mine[i - 1];
                let (ma, mb) = (&minus[e.c1 as usize], &minus[e.c2 as usize]);
                let r = &mut res[i - 1];
                for t in 0..LANES {
                    r[t] += lv[t] * (sum[t] - ma[t] - mb[t]);
                }
                i -= 1;
            }
        }

        for (e, r) in mine.iter().zip(res.iter()) {
            for t in 0..k {
                outs[t][e.idx as usize] = r[t] as f32;
            }
        }
    });

    // Ties only pay something when commitments since the root differ
    // (mid-street roots) or with rake: rare, so reuse the scalar code.
    for t in 0..k {
        if tie[t] != 0.0 {
            add_ties(hs, table, p, reaches[t], tie[t], outs[t]);
        }
    }
}

/// Adds `tie * (reach of equal-strength opponent hands)` to `out`.
fn add_ties(hs: &HandSet, table: &StrengthTable, p: usize, opp_reach: &[f32], tie: f64, out: &mut [f32]) {
    let o = p ^ 1;
    let mine = &table.lists[p];
    let theirs = &table.lists[o];
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
            accumulate(e.c1, e.c2, opp_reach[e.idx as usize], &mut eq_sum, &mut eq_minus);
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
