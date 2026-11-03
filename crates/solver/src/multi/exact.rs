//! Exact card removal between the two opponents.
//!
//! For a traverser hand `h = {a, b}` a terminal needs sums over pairs of
//! opponent hands `(i, j)` with `h`, `i` and `j` pairwise disjoint. By
//! inclusion–exclusion over the cards two hands can share,
//!
//! `pairs(h) = tQ'·tR' − Σ_c mQ'[c]·mR'[c] + Σ_k xQ'(k)·xR'(k)`
//!
//! where `t` is a set's total reach, `m[c]` the reach of its hands holding
//! card `c`, `x(k)` the reach of combo `k`, and `'` restricts to hands
//! disjoint from `h` (`mQ'[c] = mQ[c] − xQ(c,a) − xQ(c,b)`). Expanding the
//! middle sum leaves per-set statistics that are updated as hands enter a
//! set, plus `Σ_c xQ(c,a)·xR(c,b)`, read from per-card lists of the hands
//! present, so a query costs about as much as the ranges are dense.
//! (Independent opponents need O(1) per hand: `pairs ≈ tQ'·tR'`.)
//!
//! Everything is generic over a number of lanes `L`: terminals on the same
//! board (e.g. all showdowns of a river subtree) sweep the same hands in the
//! same order and differ only in reach, so `L` of them share one sweep.

use super::hands::{Hands, Table};
use super::{opponents, N};
use std::any::Any;
use std::cell::RefCell;

type V<const L: usize> = [f64; L];

#[inline(always)]
fn axpy<const L: usize>(acc: &mut V<L>, a: &V<L>, b: &V<L>) {
    for l in 0..L {
        acc[l] += a[l] * b[l];
    }
}

/// Per-card sums of one player's hands in a set (per lane).
struct Sums<const L: usize> {
    t: V<L>,
    m: [V<L>; 52],
    /// `x[u][v] = x[v][u]` = reach of combo `{u, v}`.
    x: Vec<[V<L>; 52]>,
    /// Hands in the set by card: `by_card[u]` lists `(v, reach)` for each
    /// combo `{u, v}`.
    by_card: Vec<Vec<(u8, V<L>)>>,
    added: Vec<(u8, u8)>,
}

impl<const L: usize> Sums<L> {
    fn new() -> Sums<L> {
        Sums {
            t: [0.0; L],
            m: [[0.0; L]; 52],
            x: vec![[[0.0; L]; 52]; 52],
            by_card: vec![Vec::new(); 52],
            added: Vec::new(),
        }
    }

    fn add(&mut self, u: u8, v: u8, w: &V<L>) {
        let (ui, vi) = (u as usize, v as usize);
        for l in 0..L {
            self.t[l] += w[l];
            self.m[ui][l] += w[l];
            self.m[vi][l] += w[l];
            self.x[ui][vi][l] += w[l];
            self.x[vi][ui][l] += w[l];
        }
        self.by_card[ui].push((v, *w));
        self.by_card[vi].push((u, *w));
        self.added.push((u, v));
    }

    fn clear(&mut self) {
        for &(u, v) in &self.added {
            self.x[u as usize][v as usize] = [0.0; L];
            self.x[v as usize][u as usize] = [0.0; L];
            self.by_card[u as usize].clear();
            self.by_card[v as usize].clear();
        }
        self.added.clear();
        self.m = [[0.0; L]; 52];
        self.t = [0.0; L];
    }
}

/// Two sets (one per opponent) and their pair statistics, kept up to date
/// as hands are added:
/// * `s = Σ_c mQ[c]·mR[c]`;
/// * `xqr[a] = Σ_c mQ[c]·xR(c,a)` and `xrq[a] = Σ_c mR[c]·xQ(c,a)`;
/// * `same = Σ_k xQ(k)·xR(k)`, `same_card[a]` its part over combos with `a`.
struct Pairs<const L: usize> {
    q: Sums<L>,
    r: Sums<L>,
    s: V<L>,
    xqr: [V<L>; 52],
    xrq: [V<L>; 52],
    same: V<L>,
    same_card: [V<L>; 52],
}

impl<const L: usize> Pairs<L> {
    fn new() -> Pairs<L> {
        Pairs {
            q: Sums::new(),
            r: Sums::new(),
            s: [0.0; L],
            xqr: [[0.0; L]; 52],
            xrq: [[0.0; L]; 52],
            same: [0.0; L],
            same_card: [[0.0; L]; 52],
        }
    }

    fn clear(&mut self) {
        self.q.clear();
        self.r.clear();
        self.s = [0.0; L];
        self.xqr = [[0.0; L]; 52];
        self.xrq = [[0.0; L]; 52];
        self.same = [0.0; L];
        self.same_card = [[0.0; L]; 52];
    }

    fn add_q(&mut self, u: u8, v: u8, w: &V<L>) {
        Self::add(
            &mut self.q,
            &self.r,
            &mut self.xqr,
            &mut self.xrq,
            u,
            v,
            w,
            &mut self.s,
            &mut self.same,
            &mut self.same_card,
        );
    }

    fn add_r(&mut self, u: u8, v: u8, w: &V<L>) {
        Self::add(
            &mut self.r,
            &self.q,
            &mut self.xrq,
            &mut self.xqr,
            u,
            v,
            w,
            &mut self.s,
            &mut self.same,
            &mut self.same_card,
        );
    }

    /// Adds combo `{u, v}` with reach `w` to set `me`; `mine_x[a]` is
    /// `Σ_c m_me[c]·x_other(c,a)` and `other_x[a]` is `Σ_c m_other[c]·x_me(c,a)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn add(
        me: &mut Sums<L>,
        other: &Sums<L>,
        mine_x: &mut [V<L>; 52],
        other_x: &mut [V<L>; 52],
        u: u8,
        v: u8,
        w: &V<L>,
        s: &mut V<L>,
        same: &mut V<L>,
        same_card: &mut [V<L>; 52],
    ) {
        let (ui, vi) = (u as usize, v as usize);
        let xo = other.x[ui][vi];
        for l in 0..L {
            s[l] += w[l] * (other.m[ui][l] + other.m[vi][l]);
            // x_me(u, v) grows by w.
            other_x[vi][l] += other.m[ui][l] * w[l];
            other_x[ui][l] += other.m[vi][l] * w[l];
            let sm = w[l] * xo[l];
            same[l] += sm;
            same_card[ui][l] += sm;
            same_card[vi][l] += sm;
        }
        // m_me[u] and m_me[v] grow by w.
        for (a, x) in &other.by_card[ui] {
            axpy(&mut mine_x[*a as usize], w, x);
        }
        for (a, x) in &other.by_card[vi] {
            axpy(&mut mine_x[*a as usize], w, x);
        }
        me.add(u, v, w);
    }

    /// Reach of disjoint `(q hand, r hand)` pairs that are also disjoint
    /// from `{a, b}`, per lane.
    #[inline]
    fn pairs(&self, a: u8, b: u8) -> V<L> {
        let (a, b) = (a as usize, b as usize);
        let (q, r) = (&self.q, &self.r);
        // Y(a,b) + Y(b,a) with Y(a,b) = Σ_c xQ(c,a)·xR(c,b).
        let (ra, rb) = (&r.x[a], &r.x[b]);
        let mut y = [0.0; L];
        for (c, x) in &q.by_card[a] {
            axpy(&mut y, x, &rb[*c as usize]);
        }
        for (c, x) in &q.by_card[b] {
            axpy(&mut y, x, &ra[*c as usize]);
        }
        let (qab, rab) = (q.x[a][b], r.x[a][b]);
        let mut out = [0.0; L];
        for l in 0..L {
            let tq = q.t[l] - q.m[a][l] - q.m[b][l] + qab[l];
            let tr = r.t[l] - r.m[a][l] - r.m[b][l] + rab[l];
            // Σ_c (mQ[c] − xQ(c,a) − xQ(c,b))·(mR[c] − xR(c,a) − xR(c,b)),
            // without cards a and b (they can't be in compatible hands).
            let shared = self.s[l] - self.xqr[a][l] - self.xqr[b][l] - self.xrq[a][l] - self.xrq[b][l]
                + self.same_card[a][l]
                + self.same_card[b][l]
                + y[l]
                - (q.m[a][l] - qab[l]) * (r.m[a][l] - rab[l])
                - (q.m[b][l] - qab[l]) * (r.m[b][l] - rab[l]);
            let same = self.same[l] - self.same_card[a][l] - self.same_card[b][l] + qab[l] * rab[l];
            out[l] = (tq * tr - shared + same).max(0.0);
        }
        out
    }
}

thread_local! {
    /// Reused `(Pairs<L>, Pairs<L>)` per lane count.
    static SCRATCH: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

fn with_pairs<const L: usize, R>(f: impl FnOnce(&mut Pairs<L>, &mut Pairs<L>) -> R) -> R {
    let mut slot: Box<(Pairs<L>, Pairs<L>)> = SCRATCH
        .with(|c| {
            let mut v = c.borrow_mut();
            let i = v.iter().position(|b| b.is::<(Pairs<L>, Pairs<L>)>())?;
            Some(v.swap_remove(i).downcast().expect("checked"))
        })
        .unwrap_or_else(|| Box::new((Pairs::new(), Pairs::new())));
    let (a, b) = &mut *slot;
    a.clear();
    b.clear();
    let r = f(a, b);
    a.clear();
    b.clear();
    SCRATCH.with(|c| c.borrow_mut().push(slot));
    r
}

/// Reach of hand `i` in each lane.
#[inline]
fn lane_weights<const L: usize>(reaches: &[&[f32]; L], i: usize) -> Option<V<L>> {
    let mut w = [0.0; L];
    let mut any = false;
    for l in 0..L {
        w[l] = reaches[l][i] as f64;
        any |= w[l] != 0.0;
    }
    any.then_some(w)
}

fn add_all<const L: usize>(
    set: &mut Pairs<L>,
    hands: &Hands,
    q: usize,
    rq: &[&[f32]; L],
    r: usize,
    rr: &[&[f32]; L],
) {
    for (i, &(u, v)) in hands.cards[q].iter().enumerate() {
        if let Some(w) = lane_weights(rq, i) {
            set.add_q(u, v, &w);
        }
    }
    for (i, &(u, v)) in hands.cards[r].iter().enumerate() {
        if let Some(w) = lane_weights(rr, i) {
            set.add_r(u, v, &w);
        }
    }
}

/// For each hand of `p`: reach of compatible opponent pairs (no strength
/// condition). `rq`/`rr` are the reaches of `opponents(p)`.
pub(super) fn total_pairs(hands: &Hands, p: usize, rq: &[f32], rr: &[f32]) -> Vec<f64> {
    let (q, r) = opponents(p);
    with_pairs::<1, _>(|all, _| {
        add_all(all, hands, q, &[rq], r, &[rr]);
        hands.cards[p].iter().map(|&(a, b)| all.pairs(a, b)[0]).collect()
    })
}

/// `outs[l][h] = pays[l] * pairs_l(h)` for terminals whose payoff doesn't
/// depend on the cards (someone won uncontested, or `p` folded).
pub(super) fn fold_values_lanes<const L: usize>(
    hands: &Hands,
    p: usize,
    rq: &[&[f32]; L],
    rr: &[&[f32]; L],
    pays: &V<L>,
    outs: &mut [&mut [f32]; L],
) {
    let (q, r) = opponents(p);
    with_pairs::<L, _>(|all, _| {
        add_all(all, hands, q, rq, r, rr);
        for (h, &(a, b)) in hands.cards[p].iter().enumerate() {
            let v = all.pairs(a, b);
            for l in 0..L {
                outs[l][h] = (pays[l] * v[l]) as f32;
            }
        }
    });
}

/// Single-lane [`fold_values_lanes`].
pub(super) fn fold_values(hands: &Hands, p: usize, rq: &[f32], rr: &[f32], pay: f64, out: &mut [f32]) {
    fold_values_lanes::<1>(hands, p, &[rq], &[rr], &[pay], &mut [out]);
}

/// Showdown values of `p` with exact card removal, `L` terminals at once
/// (same board, same folded opponents): `win[l]` when `p` alone has the
/// best hand, `lose[l]` otherwise, and the pot split evenly between tied
/// best hands (payoffs already normalized). A folded opponent's cards still
/// block. `outs` must cover `p`'s hands; hands not on this board's lists
/// get 0.
#[allow(clippy::too_many_arguments)]
pub(super) fn showdown_lanes<const L: usize>(
    table: &Table,
    folded: [bool; N],
    p: usize,
    rq: &[&[f32]; L],
    rr: &[&[f32]; L],
    win: &V<L>,
    lose: &V<L>,
    outs: &mut [&mut [f32]; L],
) {
    let (q, r) = opponents(p);
    let (lp, lq, lr) = (&table.lists[p], &table.lists[q], &table.lists[r]);
    let (q_in, r_in) = (!folded[q], !folded[r]);
    for o in outs.iter_mut() {
        o.fill(0.0);
    }
    // Per hand of p (list order): pairs where p is strictly best, best
    // together with others (`le`), and tied with both.
    let mut res: Vec<(V<L>, V<L>, V<L>)> = vec![([0.0; L], [0.0; L], [0.0; L]); lp.len()];
    with_pairs::<L, _>(|lt, eq| {
        let add_q = |set: &mut Pairs<L>, e: &(u32, u32, u8, u8)| {
            if let Some(w) = lane_weights(rq, e.1 as usize) {
                set.add_q(e.2, e.3, &w);
            }
        };
        let add_r = |set: &mut Pairs<L>, e: &(u32, u32, u8, u8)| {
            if let Some(w) = lane_weights(rr, e.1 as usize) {
                set.add_r(e.2, e.3, &w);
            }
        };
        // A folded opponent's hand only blocks cards: all of it is in the
        // set from the start.
        if !q_in {
            lq.iter().for_each(|e| add_q(lt, e));
        }
        if !r_in {
            lr.iter().for_each(|e| add_r(lt, e));
        }
        let (mut iq, mut ir, mut i) = (0, 0, 0);
        while i < lp.len() {
            let s = lp[i].0;
            if q_in {
                while iq < lq.len() && lq[iq].0 < s {
                    add_q(lt, &lq[iq]);
                    iq += 1;
                }
            }
            if r_in {
                while ir < lr.len() && lr[ir].0 < s {
                    add_r(lt, &lr[ir]);
                    ir += 1;
                }
            }
            let start = i;
            while i < lp.len() && lp[i].0 == s {
                let a = lt.pairs(lp[i].2, lp[i].3);
                res[i] = (a, a, [0.0; L]);
                i += 1;
            }
            // Opponent hands of equal strength.
            let eq_q = if q_in { lq[iq..].iter().take_while(|e| e.0 == s).count() } else { 0 };
            let eq_r = if r_in { lr[ir..].iter().take_while(|e| e.0 == s).count() } else { 0 };
            if eq_q + eq_r == 0 {
                continue;
            }
            for e in &lq[iq..iq + eq_q] {
                add_q(lt, e);
            }
            for e in &lr[ir..ir + eq_r] {
                add_r(lt, e);
            }
            let tie_both = eq_q > 0 && eq_r > 0;
            if tie_both {
                for e in &lq[iq..iq + eq_q] {
                    add_q(eq, e);
                }
                for e in &lr[ir..ir + eq_r] {
                    add_r(eq, e);
                }
            }
            for k in start..i {
                let (a, b) = (lp[k].2, lp[k].3);
                res[k].1 = lt.pairs(a, b);
                if tie_both {
                    res[k].2 = eq.pairs(a, b);
                }
            }
            if tie_both {
                eq.clear();
            }
            iq += eq_q;
            ir += eq_r;
        }
        if q_in {
            lq[iq..].iter().for_each(|e| add_q(lt, e));
        }
        if r_in {
            lr[ir..].iter().for_each(|e| add_r(lt, e));
        }
        // With one opponent folded, "tied with both" can't happen and a tie
        // with the live opponent is a two-way split.
        let both = q_in && r_in;
        for (e, (alone, le, all_tied)) in lp.iter().zip(res.iter()) {
            let total = lt.pairs(e.2, e.3);
            for l in 0..L {
                let (two, three) = if both {
                    (le[l] - alone[l] - all_tied[l], all_tied[l])
                } else {
                    (le[l] - alone[l], 0.0)
                };
                let split = |k: f64| lose[l] + (win[l] - lose[l]) / k;
                let v =
                    win[l] * alone[l] + split(2.0) * two + split(3.0) * three + lose[l] * (total[l] - le[l]);
                outs[l][e.1 as usize] = v as f32;
            }
        }
    });
}

/// Single-lane [`showdown_lanes`].
#[allow(clippy::too_many_arguments)]
pub(super) fn showdown(
    table: &Table,
    folded: [bool; N],
    p: usize,
    rq: &[f32],
    rr: &[f32],
    win: f64,
    lose: f64,
    out: &mut [f32],
) {
    showdown_lanes::<1>(table, folded, p, &[rq], &[rr], &[win], &[lose], &mut [out]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use holdem::card::parse_cards;
    use holdem::eval::EvalMask;
    use holdem::{Card, CardSet, Range};
    use rand::{Rng, SeedableRng};

    /// Brute force over all hand triples.
    #[allow(clippy::too_many_arguments)]
    fn brute(
        hands: &Hands,
        board: &[Card],
        p: usize,
        reach: &[Vec<f32>; N],
        folded: [bool; N],
        win: f64,
        lose: f64,
    ) -> Vec<f64> {
        let (q, r) = opponents(p);
        let mask = EvalMask::from_cards(board);
        let strength = |(a, b): (u8, u8)| mask.add(Card::from_index(a)).add(Card::from_index(b)).evaluate();
        let dead = CardSet::from_cards(board);
        let live =
            |(a, b): (u8, u8)| !dead.contains(Card::from_index(a)) && !dead.contains(Card::from_index(b));
        let disjoint = |x: (u8, u8), y: (u8, u8)| x.0 != y.0 && x.0 != y.1 && x.1 != y.0 && x.1 != y.1;
        hands.cards[p]
            .iter()
            .map(|&h| {
                if !live(h) {
                    return 0.0;
                }
                let mut v = 0.0;
                for (i, &x) in hands.cards[q].iter().enumerate() {
                    if !live(x) || !disjoint(h, x) {
                        continue;
                    }
                    for (j, &y) in hands.cards[r].iter().enumerate() {
                        if !live(y) || !disjoint(h, y) || !disjoint(x, y) {
                            continue;
                        }
                        let w = reach[q][i] as f64 * reach[r][j] as f64;
                        let sh = strength(h);
                        let mut best = sh;
                        if !folded[q] {
                            best = best.max(strength(x));
                        }
                        if !folded[r] {
                            best = best.max(strength(y));
                        }
                        let pay = if folded[p] || sh < best {
                            lose
                        } else {
                            let k = 1 + [(q, x), (r, y)]
                                .iter()
                                .filter(|&&(o, c)| !folded[o] && strength(c) == best)
                                .count();
                            lose + (win - lose) / k as f64
                        };
                        v += w * pay;
                    }
                }
                v
            })
            .collect()
    }

    #[test]
    fn exact_terminal_matches_brute_force() {
        for board in ["Kd9s5h2c2d", "AhAsAdKcQh", "7c8c9cTc2h", "Js9s4d2c3h"] {
            check_board(board);
        }
    }

    fn check_board(board: &str) {
        let board = parse_cards(board).unwrap();
        // Overlapping ranges with many shared cards and ties (the board
        // pairs, and K9 / 99 / A2 / 22-type hands collide).
        let ranges = [
            "AA, KK, K9, 99, A2, 55, QJs, T9s, 76s, A5s, 53s",
            "AK, KQ, K9s, 99, 22, A9, J9s, 98s, 65s, 43s",
            "AQ, KJ, 99, 55, A5, K5, T9, 86s, 64s, 32s",
        ]
        .map(|r| Range::parse(r).unwrap().into_weights());
        let hands = Hands::new(&ranges, CardSet::from_cards(&board), 0.0, &Default::default());
        let table = Table::new(&hands, &board);
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for case in 0..12 {
            let reach: [Vec<f32>; N] =
                std::array::from_fn(|x| (0..hands.len(x)).map(|_| rng.random_range(0.0..1.0f32)).collect());
            let folded = match case % 4 {
                0 | 1 => [false; N],
                2 => [false, false, true],
                _ => [false, true, false],
            };
            let p = case % N;
            let invest = [100, 250, 400];
            let win = 900.0 + (0..N).filter(|&j| j != p).map(|j| invest[j] as f64).sum::<f64>() - 300.0;
            let lose = -(invest[p] as f64 + 300.0);
            let (q, r) = opponents(p);
            let mut out = vec![0f32; hands.len(p)];
            if folded[p] {
                fold_values(&hands, p, &reach[q], &reach[r], lose, &mut out);
                // Hands blocked by the turn/river get no value in a showdown.
                let dead = CardSet::from_cards(&board);
                for (o, &(a, b)) in out.iter_mut().zip(&hands.cards[p]) {
                    if dead.contains(Card::from_index(a)) || dead.contains(Card::from_index(b)) {
                        *o = 0.0;
                    }
                }
            } else {
                showdown(&table, folded, p, &reach[q], &reach[r], win, lose, &mut out);
            }
            let want = brute(&hands, &board, p, &reach, folded, win, lose);
            for (h, (&a, &b)) in out.iter().zip(&want).enumerate() {
                assert!(
                    (a as f64 - b).abs() <= 1e-3 * (1.0 + b.abs()),
                    "board {board:?}, case {case}, player {p}, hand {h}: {a} vs {b}"
                );
            }
        }
    }
}
