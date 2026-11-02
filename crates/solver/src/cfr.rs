//! Discounted CFR over the vectorized (range-vs-range) game tree.
//!
//! [`traverse`] receives the opponent's reach probabilities (one per
//! opponent hand) and writes the traversing player's counterfactual values
//! (one per own hand) into `out`. The same traversal implements a DCFR
//! iteration, a best response, and an evaluation of the average strategies.
//! Terminal values are normalized by the total weight of compatible hand
//! pairs, so summing `weight * value` over a player's hands gives an
//! expected payoff in chips.
//!
//! Payoffs: `u_i = (chips i ends with) - (chips i had at the root) - pot/2`,
//! which is zero-sum without rake.

use crate::eval::{eval_fold, eval_showdown};
use crate::hands::{HandSet, StrengthTable};
use crate::iso::Isomorphism;
use crate::river;
use crate::scratch::Buf;
use crate::storage::Storage;
use crate::tree::{Kind, Node, Tree};
use holdem::Street;
use rayon::prelude::*;

pub(crate) struct Ctx<'a> {
    pub tree: &'a Tree,
    pub store: &'a Storage,
    pub hands: &'a HandSet,
    pub tables: &'a [StrengthTable],
    pub iso: &'a Isomorphism,
    pub half_pot: f64,
    pub norm: f64,
    /// Use the breadth-first batched evaluator for river subtrees.
    pub batch_river: bool,
}

/// DCFR discount factors for one iteration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Discount {
    pub pos: f32,
    pub neg: f32,
    pub strat: f32,
}

/// DCFR hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcfrParams {
    pub alpha: f64,
    /// Negative regrets are multiplied by `t^beta / (t^beta + 1)`; `None`
    /// clips them to zero every iteration (DCFR+).
    pub beta: Option<f64>,
    pub gamma: f64,
    /// Restart the strategy average at iterations 4, 16, 64, 256, ... so
    /// the early, poor iterations drop out of it (as b-inary/postflop-solver
    /// does).
    pub restart_average: bool,
}

impl Default for DcfrParams {
    fn default() -> Self {
        DcfrParams { alpha: 1.5, beta: Some(0.0), gamma: 2.0, restart_average: false }
    }
}

impl Discount {
    /// Factors applied at (1-based) iteration `t` to the values accumulated
    /// over the previous iterations.
    pub fn new(t: u32, params: &DcfrParams) -> Discount {
        let prev = (t.max(1) - 1) as f64;
        let pa = prev.powf(params.alpha);
        let tf = t.max(1) as f64;
        let neg = match params.beta {
            Some(b) => {
                let pb = prev.powf(b);
                (pb / (pb + 1.0)) as f32
            }
            None => 0.0,
        };
        // Iterations since the average was (re)started, 0-based.
        let mut since = tf - 1.0;
        if params.restart_average {
            let t0 = t.max(1) - 1;
            let p4 = if t0 == 0 { 0 } else { 1u32 << ((31 - t0.leading_zeros()) & !1) };
            since = (t0 - p4) as f64;
        }
        let strat = (since / (since + 1.0)).powf(params.gamma) as f32;
        Discount { pos: (pa / (pa + 1.0)) as f32, neg, strat }
    }
}

/// Normalizes action-major values per hand (uniform when the column sums to 0).
pub(crate) fn normalize_in_place(s: &mut [f32], num_actions: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut denom = Buf::from_slice(&s[..n]);
    for row in s.chunks_exact(n).skip(1) {
        for (d, &v) in denom.iter_mut().zip(row) {
            *d += v;
        }
    }
    // One division per hand instead of one per (action, hand).
    for d in denom.iter_mut() {
        *d = if *d > 0.0 { 1.0 / *d } else { 0.0 };
    }
    let uniform = 1.0 / num_actions as f32;
    for row in s.chunks_exact_mut(n) {
        for (v, &inv) in row.iter_mut().zip(denom.iter()) {
            *v = if inv > 0.0 { *v * inv } else { uniform };
        }
    }
}

/// Regret matching in place: strategy proportional to positive regrets.
pub(crate) fn regret_match(s: &mut [f32], num_actions: usize, n: usize) {
    for x in s.iter_mut() {
        *x = x.max(0.0);
    }
    normalize_in_place(s, num_actions, n);
    debug_assert!(is_distribution(s, num_actions, n), "regret matching produced an invalid strategy");
}

/// Debug sentinel: every hand's action probabilities are in [0, 1] and sum to 1.
pub(crate) fn is_distribution(s: &[f32], num_actions: usize, n: usize) -> bool {
    (0..n).all(|h| {
        let mut sum = 0f32;
        for a in 0..num_actions {
            let x = s[a * n + h];
            if !(-1e-6..=1.0 + 1e-5).contains(&x) {
                return false;
            }
            sum += x;
        }
        (sum - 1.0).abs() < 1e-3
    })
}

/// Normalized payoff of a fold terminal for player `p`.
#[inline]
pub(crate) fn fold_payoff(ctx: &Ctx, node: &Node, p: usize) -> f64 {
    let Kind::Fold(f) = node.kind else { unreachable!() };
    let f = f as usize;
    let v = if f == p {
        -(ctx.half_pot + node.invest[p] as f64)
    } else {
        ctx.half_pot + node.invest[f] as f64 - node.rake as f64
    };
    v * ctx.norm
}

/// Normalized `(win, lose, tie)` payoffs of a showdown for player `p`.
#[inline]
pub(crate) fn showdown_payoffs(ctx: &Ctx, node: &Node, p: usize) -> (f64, f64, f64) {
    let o = p ^ 1;
    let rake = node.rake as f64;
    let win = (ctx.half_pot + node.invest[o] as f64 - rake) * ctx.norm;
    let lose = -(ctx.half_pot + node.invest[p] as f64) * ctx.norm;
    let tie = 0.5 * ((node.invest[o] - node.invest[p]) as f64 - rake) * ctx.norm;
    (win, lose, tie)
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode {
    /// One DCFR update for the traverser (and the opponent's strategy sums).
    Cfr(Discount),
    /// The traverser best-responds to the opponent's average strategy.
    BestResponse,
    /// Both players play their average strategies.
    Average,
}

/// Values of player `p` at node `id` given the opponent's reach.
pub(crate) fn traverse(ctx: &Ctx, id: usize, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    traverse_inner(ctx, id, p, opp_reach, mode, out);
    debug_assert!(out.iter().all(|v| v.is_finite()), "non-finite values at node {id}");
}

fn traverse_inner(ctx: &Ctx, id: usize, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    debug_assert!(opp_reach.iter().all(|&r| r >= 0.0 && r.is_finite()), "invalid reach at node {id}");
    let node = &ctx.tree.nodes[id];
    match node.kind {
        Kind::Fold(_) => eval_fold(ctx.hands, p, opp_reach, fold_payoff(ctx, node, p), out),
        Kind::Showdown => {
            let (w, l, t) = showdown_payoffs(ctx, node, p);
            eval_showdown(ctx.hands, &ctx.tables[node.offset as usize], p, opp_reach, w, l, t, out);
        }
        Kind::Chance => chance(ctx, node, p, opp_reach, mode, out),
        Kind::Player(_) if ctx.batch_river && node.street == Street::River => {
            river::subtree(ctx, id, p, opp_reach, mode, out)
        }
        Kind::Player(q) if q as usize == p => own_node(ctx, node, p, opp_reach, mode, out),
        Kind::Player(_) => opponent_node(ctx, node, p, opp_reach, mode, out),
    }
}

fn own_node(ctx: &Ctx, node: &Node, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    let na = node.num_actions();
    let n = out.len();
    let mut cfv = Buf::uninit(na * n);
    for (row, c) in cfv.chunks_exact_mut(n.max(1)).zip(node.children()) {
        traverse(ctx, c, p, opp_reach, mode, row);
    }
    if n == 0 {
        return;
    }
    own_node_combine(ctx, node, n, &cfv, mode, out);
}

/// Values of a traverser node from its children's values (`cfv`, action
/// major), with the regret update in CFR mode. Regrets are decoded and
/// encoded once.
#[inline]
pub(crate) fn own_node_combine(ctx: &Ctx, node: &Node, n: usize, cfv: &[f32], mode: Mode, out: &mut [f32]) {
    let na = node.num_actions();
    let b = node.block(n);
    let combine = |s: &[f32], out: &mut [f32]| {
        out.fill(0.0);
        for (srow, crow) in s.chunks_exact(n).zip(cfv.chunks_exact(n)) {
            for ((o, &x), &v) in out.iter_mut().zip(srow).zip(crow) {
                *o += x * v;
            }
        }
    };
    match mode {
        Mode::BestResponse => {
            out.copy_from_slice(&cfv[..n]);
            for row in cfv.chunks_exact(n).skip(1) {
                for (o, &v) in out.iter_mut().zip(row) {
                    *o = o.max(v);
                }
            }
        }
        Mode::Average => {
            let mut s = Buf::uninit(na * n);
            unsafe { ctx.store.load_strat(b, &mut s) };
            normalize_in_place(&mut s, na, n);
            combine(&s, out);
        }
        Mode::Cfr(d) => unsafe {
            ctx.store.update_regrets(b, |r| {
                let mut s = Buf::from_slice(r);
                regret_match(&mut s, na, n);
                combine(&s, out);
                let out: &[f32] = out;
                for (rrow, crow) in r.chunks_exact_mut(n).zip(cfv.chunks_exact(n)) {
                    for ((x, &c), &o) in rrow.iter_mut().zip(crow).zip(out) {
                        let disc = if *x > 0.0 { d.pos } else { d.neg };
                        *x = *x * disc + c - o;
                    }
                }
            });
        },
    }
}

fn opponent_node(ctx: &Ctx, node: &Node, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    let na = node.num_actions();
    let nq = opp_reach.len();
    let n = out.len();
    let mut s = Buf::uninit(na * nq);
    if nq > 0 {
        let b = node.block(nq);
        unsafe {
            match mode {
                Mode::Cfr(d) => {
                    ctx.store.load_regrets(b, &mut s);
                    regret_match(&mut s, na, nq);
                    ctx.store.update_strat(b, |ss| {
                        for (ss_row, s_row) in ss.chunks_exact_mut(nq).zip(s.chunks_exact(nq)) {
                            for ((x, &y), &w) in ss_row.iter_mut().zip(s_row).zip(opp_reach) {
                                *x = *x * d.strat + y * w;
                            }
                        }
                    });
                }
                _ => {
                    ctx.store.load_strat(b, &mut s);
                    normalize_in_place(&mut s, na, nq);
                }
            }
        }
        for row in s.chunks_exact_mut(nq) {
            for (x, &r) in row.iter_mut().zip(opp_reach) {
                *x *= r;
            }
        }
    }
    out.fill(0.0);
    let mut tmp = Buf::uninit(n);
    for (a, c) in node.children().enumerate() {
        let r_a = &s[a * nq..(a + 1) * nq];
        if r_a.iter().all(|&w| w == 0.0) {
            continue;
        }
        traverse(ctx, c, p, r_a, mode, &mut tmp);
        for (o, &v) in out.iter_mut().zip(tmp.iter()) {
            *o += v;
        }
    }
}

fn chance(ctx: &Ctx, node: &Node, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    let n = out.len();
    if n == 0 || opp_reach.iter().all(|&w| w == 0.0) {
        out.fill(0.0);
        return;
    }
    let o = p ^ 1;
    let factor = node.chance_factor;
    let k = node.num_actions();
    // Pooled: a fresh ~200 KB allocation per chance node made glibc return
    // the memory to the kernel and fault it back in (page zeroing and TLB
    // shootdowns took ~15% of flop solves). Every row is fully written.
    let mut rows = Buf::uninit(k * n);
    rows.par_chunks_mut(n).zip(node.children().into_par_iter()).for_each(|(row, c)| {
        let card = ctx.tree.card(c).index();
        let mut r = Buf::from_slice(opp_reach);
        for x in r.iter_mut() {
            *x *= factor;
        }
        for &h in &ctx.hands.blocked[o][card] {
            r[h as usize] = 0.0;
        }
        traverse(ctx, c, p, &r, mode, row);
        for &h in &ctx.hands.blocked[p][card] {
            row[h as usize] = 0.0;
        }
    });
    let mut acc = vec![0f64; n];
    for row in rows.chunks_exact(n) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v as f64;
        }
    }
    // Isomorphic cards: the canonical child's values with suits swapped.
    for &(ci, pair) in &ctx.tree.chance[node.slot as usize].iso {
        let row = &rows[ci as usize * n..(ci as usize + 1) * n];
        let perm = &ctx.iso.perm[p][pair as usize];
        for (a, &j) in acc.iter_mut().zip(perm) {
            *a += row[j as usize] as f64;
        }
    }
    for (o, a) in out.iter_mut().zip(acc) {
        *o = a as f32;
    }
}

/// Expected payoff (chips) of each player under `mode` at the root.
pub(crate) fn root_values(ctx: &Ctx, mode: Mode) -> [f64; 2] {
    let mut res = [0f64; 2];
    for p in 0..2 {
        let mut v = vec![0f32; ctx.hands.len(p)];
        traverse(ctx, 0, p, &ctx.hands.weights[p ^ 1], mode, &mut v);
        res[p] = v.iter().zip(&ctx.hands.weights[p]).map(|(&x, &w)| x as f64 * w as f64).sum();
    }
    res
}

/// Exploitability (chips): average gain of the two best responses over the
/// average strategy profile.
pub(crate) fn exploitability(ctx: &Ctx, raked: bool) -> f64 {
    let br = root_values(ctx, Mode::BestResponse);
    if raked {
        let ev = root_values(ctx, Mode::Average);
        ((br[0] - ev[0]) + (br[1] - ev[1])) / 2.0
    } else {
        (br[0] + br[1]) / 2.0
    }
}
