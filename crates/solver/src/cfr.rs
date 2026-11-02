//! Discounted CFR over the vectorized (range-vs-range) game tree.
//!
//! Each recursive call receives the opponent's reach probabilities (one per
//! opponent hand) and writes the traversing player's counterfactual values
//! (one per own hand) into `out`. Terminal values are normalized by the total
//! weight of compatible hand pairs, so summing `weight * value` over a
//! player's hands gives an expected payoff in chips.
//!
//! Payoffs are zero-sum: `u_i = (chips i ends with) - (chips i had at the
//! root) - root_pot / 2`.

use crate::eval::{eval_fold, eval_showdown};
use crate::hands::{HandSet, StrengthTable};
use crate::tree::{Kind, Node};
use rayon::prelude::*;

pub(crate) struct Ctx<'a> {
    pub hands: &'a HandSet,
    pub tables: &'a [StrengthTable],
    pub half_pot: f64,
    pub norm: f64,
}

/// DCFR discount factors for one iteration (α = 1.5, β = 0, γ = 2).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Discount {
    pub pos: f32,
    pub neg: f32,
    pub strat: f32,
}

impl Discount {
    /// Factors applied at (1-based) iteration `t` to the values accumulated
    /// over the previous iterations.
    pub fn new(t: u32) -> Discount {
        let prev = (t.max(1) - 1) as f64;
        let pa = prev.powf(1.5);
        let tf = t.max(1) as f64;
        Discount { pos: (pa / (pa + 1.0)) as f32, neg: 0.5, strat: ((tf - 1.0) / tf).powi(2) as f32 }
    }
}

/// Normalizes action-major values per hand (uniform when the column sums to 0).
pub(crate) fn normalize_in_place(s: &mut [f32], num_actions: usize, n: usize) {
    let mut denom = s[..n].to_vec();
    for a in 1..num_actions {
        for (d, &v) in denom.iter_mut().zip(&s[a * n..(a + 1) * n]) {
            *d += v;
        }
    }
    let uniform = 1.0 / num_actions as f32;
    for row in s.chunks_exact_mut(n) {
        for (v, &d) in row.iter_mut().zip(&denom) {
            *v = if d > 0.0 { *v / d } else { uniform };
        }
    }
}

/// Regret matching: strategy proportional to positive regrets.
#[inline]
fn regret_matching(regrets: &[f32], num_actions: usize, n: usize) -> Vec<f32> {
    let mut s: Vec<f32> = regrets.iter().map(|&r| r.max(0.0)).collect();
    normalize_in_place(&mut s, num_actions, n);
    s
}

pub(crate) fn average_strategy(node: &Node, n: usize) -> Vec<f32> {
    let mut s = node.strat_sum.clone();
    normalize_in_place(&mut s, node.num_actions(), n);
    s
}

fn terminal_values(ctx: &Ctx, node: &Node, p: usize, opp_reach: &[f32], out: &mut [f32]) {
    let o = p ^ 1;
    match node.kind {
        Kind::Fold(f) => {
            let f = f as usize;
            let payoff = if f == p {
                -(ctx.half_pot + node.invest[p] as f64)
            } else {
                ctx.half_pot + node.invest[f] as f64
            };
            eval_fold(ctx.hands, p, opp_reach, payoff * ctx.norm, out);
        }
        Kind::Showdown => {
            let win = (ctx.half_pot + node.invest[o] as f64) * ctx.norm;
            let lose = -(ctx.half_pot + node.invest[p] as f64) * ctx.norm;
            let tie = 0.5 * (node.invest[o] - node.invest[p]) as f64 * ctx.norm;
            eval_showdown(ctx.hands, &ctx.tables[node.table as usize], p, opp_reach, win, lose, tie, out);
        }
        _ => unreachable!(),
    }
}

/// Opponent reach after a chance card (hands containing it are impossible).
fn chance_child_reach(ctx: &Ctx, p: usize, opp_reach: &[f32], factor: f32, card: usize) -> Vec<f32> {
    let mut r: Vec<f32> = opp_reach.iter().map(|&w| w * factor).collect();
    for &k in &ctx.hands.blocked[p ^ 1][card] {
        r[k as usize] = 0.0;
    }
    r
}

/// Sums per-card value rows into `out` (in f64).
fn sum_rows(rows: &[f32], n: usize, out: &mut [f32]) {
    let mut acc = vec![0f64; n];
    for row in rows.chunks_exact(n) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v as f64;
        }
    }
    for (o, a) in out.iter_mut().zip(acc) {
        *o = a as f32;
    }
}

/// Zeroes the values of `p`'s hands that contain `card`.
fn zero_blocked(ctx: &Ctx, p: usize, card: usize, row: &mut [f32]) {
    for &k in &ctx.hands.blocked[p][card] {
        row[k as usize] = 0.0;
    }
}

/// One DCFR traversal for player `p`. Updates `p`'s regrets and the
/// opponent's average-strategy accumulators; writes `p`'s values to `out`.
pub(crate) fn cfr(ctx: &Ctx, node: &mut Node, p: usize, opp_reach: &[f32], d: Discount, out: &mut [f32]) {
    let n = out.len();
    match node.kind {
        Kind::Fold(_) | Kind::Showdown => terminal_values(ctx, node, p, opp_reach, out),
        Kind::Chance => {
            if opp_reach.iter().all(|&w| w == 0.0) {
                out.fill(0.0);
                return;
            }
            let factor = node.chance_factor;
            let mut rows = vec![0f32; node.cards.len() * n];
            node.children.par_iter_mut().zip(rows.par_chunks_mut(n)).zip(node.cards.par_iter()).for_each(
                |((child, row), &card)| {
                    let r = chance_child_reach(ctx, p, opp_reach, factor, card.index());
                    cfr(ctx, child, p, &r, d, row);
                    zero_blocked(ctx, p, card.index(), row);
                },
            );
            sum_rows(&rows, n, out);
        }
        Kind::Player(q) if q as usize == p => {
            let na = node.num_actions();
            let strategy = regret_matching(&node.regrets, na, n);
            let mut cfv = vec![0f32; na * n];
            for (child, row) in node.children.iter_mut().zip(cfv.chunks_exact_mut(n)) {
                cfr(ctx, child, p, opp_reach, d, row);
            }
            out.fill(0.0);
            for (s, c) in strategy.chunks_exact(n).zip(cfv.chunks_exact(n)) {
                for ((o, &s), &c) in out.iter_mut().zip(s).zip(c) {
                    *o += s * c;
                }
            }
            let (pos, neg) = (d.pos, d.neg);
            for (r, c) in node.regrets.chunks_exact_mut(n).zip(cfv.chunks_exact(n)) {
                for ((r, &c), &o) in r.iter_mut().zip(c).zip(out.iter()) {
                    let disc = if *r > 0.0 { pos } else { neg };
                    *r = *r * disc + c - o;
                }
            }
        }
        Kind::Player(q) => {
            let q = q as usize;
            let na = node.num_actions();
            let nq = opp_reach.len();
            let mut reach_a = regret_matching(&node.regrets, na, nq);
            let g = d.strat;
            for (ss, s) in node.strat_sum.chunks_exact_mut(nq).zip(reach_a.chunks_exact_mut(nq)) {
                for ((ss, s), &r) in ss.iter_mut().zip(s.iter_mut()).zip(opp_reach) {
                    *s *= r;
                    *ss = *ss * g + *s;
                }
            }
            debug_assert_eq!(nq, ctx.hands.len(q));
            out.fill(0.0);
            let mut tmp = vec![0f32; n];
            for (child, r_a) in node.children.iter_mut().zip(reach_a.chunks_exact(nq)) {
                if r_a.iter().all(|&w| w == 0.0) {
                    continue;
                }
                cfr(ctx, child, p, r_a, d, &mut tmp);
                for (o, &v) in out.iter_mut().zip(&tmp) {
                    *o += v;
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// The traverser plays a best response.
    BestResponse,
    /// The traverser plays its average strategy.
    Average,
}

/// Values of player `p` when the opponent plays its average strategy and `p`
/// plays according to `mode`.
pub(crate) fn evaluate(ctx: &Ctx, node: &Node, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    let n = out.len();
    match node.kind {
        Kind::Fold(_) | Kind::Showdown => terminal_values(ctx, node, p, opp_reach, out),
        Kind::Chance => {
            if opp_reach.iter().all(|&w| w == 0.0) {
                out.fill(0.0);
                return;
            }
            let factor = node.chance_factor;
            let mut rows = vec![0f32; node.cards.len() * n];
            node.children.par_iter().zip(rows.par_chunks_mut(n)).zip(node.cards.par_iter()).for_each(
                |((child, row), &card)| {
                    let r = chance_child_reach(ctx, p, opp_reach, factor, card.index());
                    evaluate(ctx, child, p, &r, mode, row);
                    zero_blocked(ctx, p, card.index(), row);
                },
            );
            sum_rows(&rows, n, out);
        }
        Kind::Player(q) if q as usize == p => {
            let na = node.num_actions();
            let mut cfv = vec![0f32; na * n];
            for (child, row) in node.children.iter().zip(cfv.chunks_exact_mut(n)) {
                evaluate(ctx, child, p, opp_reach, mode, row);
            }
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
                    let s = average_strategy(node, n);
                    out.fill(0.0);
                    for (s, c) in s.chunks_exact(n).zip(cfv.chunks_exact(n)) {
                        for ((o, &s), &c) in out.iter_mut().zip(s).zip(c) {
                            *o += s * c;
                        }
                    }
                }
            }
        }
        Kind::Player(_) => {
            let nq = opp_reach.len();
            let mut reach_a = average_strategy(node, nq);
            for s in reach_a.chunks_exact_mut(nq) {
                for (s, &r) in s.iter_mut().zip(opp_reach) {
                    *s *= r;
                }
            }
            out.fill(0.0);
            let mut tmp = vec![0f32; n];
            for (child, r_a) in node.children.iter().zip(reach_a.chunks_exact(nq)) {
                if r_a.iter().all(|&w| w == 0.0) {
                    continue;
                }
                evaluate(ctx, child, p, r_a, mode, &mut tmp);
                for (o, &v) in out.iter_mut().zip(&tmp) {
                    *o += v;
                }
            }
        }
    }
}

/// Exploitability (in chips): average gain of the two best responses.
pub(crate) fn exploitability(ctx: &Ctx, root: &Node) -> f64 {
    let mut total = 0f64;
    for p in 0..2 {
        let mut v = vec![0f32; ctx.hands.len(p)];
        evaluate(ctx, root, p, &ctx.hands.weights[p ^ 1], Mode::BestResponse, &mut v);
        total += v.iter().zip(&ctx.hands.weights[p]).map(|(&x, &w)| x as f64 * w as f64).sum::<f64>();
    }
    total / 2.0
}
