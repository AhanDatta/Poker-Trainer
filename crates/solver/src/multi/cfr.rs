//! Traversals of the three-player tree: one DCFR update for a player, a
//! best response, or the values of the average strategies.
//!
//! A traversal for player `p` carries the reach vectors of `p`'s two
//! opponents (`rq`, `rr`, see [`opponents`]) and writes `p`'s
//! counterfactual values (one per hand of `p`), normalized so that summing
//! `weight * value` over `p`'s hands gives an expected payoff in chips.
//! Payoffs are measured relative to an even split of the root pot.

use super::exact;
use super::hands::{Hands, Table};
use super::opponents;
use super::tree::{Kind, Node, Tree};
use super::N;
use crate::cfr::{normalize_in_place, own_node_update, Mode};
use crate::scratch::Buf;
use crate::storage::Storage;
use rayon::prelude::*;

pub(crate) struct Ctx<'a> {
    pub tree: &'a Tree,
    pub store: &'a Storage,
    pub hands: &'a Hands,
    pub tables: &'a [Table],
    pub third_pot: f64,
    pub root_pot: f64,
    /// Per player: 1 / (initial weight of compatible hand triples).
    pub norm: [f64; N],
    /// Exact card removal between the opponents.
    pub exact: bool,
    /// Evaluate river subtrees breadth-first with batched terminals.
    pub batch_river: bool,
}

/// `(win, lose)` payoffs of `p` at a node, already normalized; with `unit`,
/// every outcome is worth 1 (the probability mass reaching the node).
pub(crate) fn payoffs(ctx: &Ctx, node: &Node, p: usize, unit: bool) -> (f64, f64) {
    let norm = ctx.norm[p];
    if unit {
        return (norm, norm);
    }
    let others: f64 = (0..N).filter(|&j| j != p).map(|j| node.invest[j] as f64).sum();
    let win = ctx.root_pot + others - node.rake as f64 - ctx.third_pot;
    let lose = -(node.invest[p] as f64 + ctx.third_pot);
    (win * norm, lose * norm)
}

/// `out[h] = pay * (reach of opponent hand pairs compatible with h)`.
pub(crate) fn pay_all(ctx: &Ctx, p: usize, rq: &[f32], rr: &[f32], pay: f64, out: &mut [f32]) {
    if ctx.exact {
        exact::fold_values(ctx.hands, p, rq, rr, pay, out);
    } else {
        let (q, r) = opponents(p);
        let cq = ctx.hands.compat(p, q, rq);
        let cr = ctx.hands.compat(p, r, rr);
        for ((o, a), b) in out.iter_mut().zip(cq).zip(cr) {
            *o = (pay * a * b) as f32;
        }
    }
}

fn terminal(ctx: &Ctx, node: &Node, p: usize, rq: &[f32], rr: &[f32], unit: bool, out: &mut [f32]) {
    let (win, lose) = payoffs(ctx, node, p, unit);
    match node.kind {
        Kind::Won(w) => pay_all(ctx, p, rq, rr, if w as usize == p { win } else { lose }, out),
        Kind::Showdown if node.folded[p] => pay_all(ctx, p, rq, rr, lose, out),
        Kind::Showdown => {
            let table = &ctx.tables[node.slot as usize];
            if ctx.exact {
                exact::showdown(table, node.folded, p, rq, rr, win, lose, out);
            } else {
                independent_showdown(ctx.hands, table, node.folded, p, rq, rr, win, lose, out);
            }
        }
        _ => unreachable!("not a terminal"),
    }
}

/// Showdown with the opponents' hands treated as independent:
/// `P(win) = P(beat q) * P(beat r)`, ties against an opponent counting half.
#[allow(clippy::too_many_arguments)]
pub(crate) fn independent_showdown(
    hands: &Hands,
    table: &Table,
    folded: [bool; N],
    p: usize,
    rq: &[f32],
    rr: &[f32],
    win: f64,
    lose: f64,
    out: &mut [f32],
) {
    let (q, r) = opponents(p);
    let cq = hands.compat(p, q, rq);
    let cr = hands.compat(p, r, rr);
    let bq = if folded[q] { cq.clone() } else { table.beat(hands, p, q, rq) };
    let br = if folded[r] { cr.clone() } else { table.beat(hands, p, r, rr) };
    for h in 0..out.len() {
        let total = cq[h] * cr[h];
        let pwin = bq[h] * br[h];
        out[h] = (win * pwin + lose * (total - pwin)) as f32;
    }
}

/// Values of player `p` at node `id`; `unit` gives probability masses.
#[allow(clippy::too_many_arguments)]
pub(super) fn traverse(
    ctx: &Ctx,
    id: usize,
    p: usize,
    rq: &[f32],
    rr: &[f32],
    mode: Mode,
    unit: bool,
    out: &mut [f32],
) {
    let node = &ctx.tree.nodes[id];
    match node.kind {
        Kind::Won(_) | Kind::Showdown => terminal(ctx, node, p, rq, rr, unit, out),
        // Once p has folded its payoff is fixed (it loses what it put in),
        // whatever the others do, so the subtree's value is that payoff times
        // the opponent mass here (with exact runouts the mass is the same
        // below, and p has no decisions left there).
        _ if node.folded[p] => {
            let (_, lose) = payoffs(ctx, node, p, unit);
            pay_all(ctx, p, rq, rr, lose, out)
        }
        // A node an opponent can't reach is worth 0 and isn't updated.
        _ if rq.iter().all(|&w| w == 0.0) || rr.iter().all(|&w| w == 0.0) => out.fill(0.0),
        Kind::Chance => chance(ctx, node, p, rq, rr, mode, unit, out),
        Kind::Player(_) if ctx.batch_river && node.street == holdem::Street::River => {
            super::river::subtree(ctx, id, p, rq, rr, mode, unit, out)
        }
        Kind::Player(x) if x as usize == p => own_node(ctx, node, p, rq, rr, mode, unit, out),
        Kind::Player(x) => opponent_node(ctx, node, x as usize, p, rq, rr, mode, unit, out),
    }
    debug_assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite values at node {id} ({:?}, player {p}, {mode:?})",
        node.kind
    );
}

#[allow(clippy::too_many_arguments)]
fn own_node(
    ctx: &Ctx,
    node: &Node,
    p: usize,
    rq: &[f32],
    rr: &[f32],
    mode: Mode,
    unit: bool,
    out: &mut [f32],
) {
    let na = node.num_actions();
    let n = out.len();
    if n == 0 {
        return;
    }
    let mut cfv = Buf::uninit(na * n);
    for (row, c) in cfv.chunks_exact_mut(n).zip(node.children()) {
        traverse(ctx, c, p, rq, rr, mode, unit, row);
    }
    own_node_combine(ctx, node, n, &cfv, mode, out);
}

/// Value of a traverser node from its children's values (`cfv`, action
/// major), with the regret update in CFR mode.
pub(super) fn own_node_combine(ctx: &Ctx, node: &Node, n: usize, cfv: &[f32], mode: Mode, out: &mut [f32]) {
    own_node_update(ctx.store, node.block(n), node.num_actions(), n, cfv, mode, out);
}

/// The traversal in which player `x`'s average strategy at `node` is
/// accumulated (once per iteration, with `x`'s reach at hand): the next
/// player after `x` who hasn't folded there. Traversals skip what follows
/// their player's fold, but never the nodes of a player still in the hand.
#[inline]
pub(crate) fn strategy_updater(node: &Node, x: usize) -> usize {
    let a = (x + 1) % N;
    if node.folded[a] {
        (x + 2) % N
    } else {
        a
    }
}

/// Accumulates `x`'s current strategy `s` (action-major, `na` x `nx`) into
/// the node's average, weighted by `x`'s reach unless the DCFR scheme says
/// otherwise.
pub(super) fn accumulate_strategy(
    ctx: &Ctx,
    node: &Node,
    nx: usize,
    s: &[f32],
    reach: &[f32],
    d: &crate::cfr::Discount,
) {
    let b = node.block(nx);
    unsafe {
        ctx.store.update_strat(b, |ss| {
            for (ss_row, s_row) in ss.chunks_exact_mut(nx).zip(s.chunks_exact(nx)) {
                if d.reach_weighted {
                    for ((x, &y), &w) in ss_row.iter_mut().zip(s_row).zip(reach) {
                        *x = *x * d.strat + y * w;
                    }
                } else {
                    for (x, &y) in ss_row.iter_mut().zip(s_row) {
                        *x = *x * d.strat + y;
                    }
                }
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn opponent_node(
    ctx: &Ctx,
    node: &Node,
    x: usize,
    p: usize,
    rq: &[f32],
    rr: &[f32],
    mode: Mode,
    unit: bool,
    out: &mut [f32],
) {
    let na = node.num_actions();
    let nx = ctx.hands.len(x);
    let n = out.len();
    let b = node.block(nx);
    let mut s = Buf::uninit(na * nx);
    unsafe {
        match mode {
            Mode::Cfr(_) => ctx.store.load_current_strategy(b, na, nx, &mut s),
            _ => {
                ctx.store.load_strat(b, &mut s);
                normalize_in_place(&mut s, na, nx);
            }
        }
    }
    let (q, _) = opponents(p);
    let rx = if x == q { rq } else { rr };
    if let Mode::Cfr(d) = mode {
        if strategy_updater(node, x) == p {
            accumulate_strategy(ctx, node, nx, &s, rx, &d);
        }
    }
    out.fill(0.0);
    let mut tmp = Buf::uninit(n);
    let mut child = Buf::uninit(nx);
    for (row, c) in s.chunks_exact(nx.max(1)).zip(node.children()) {
        let mut any = false;
        for ((d, &w), &a) in child.iter_mut().zip(rx).zip(row) {
            *d = w * a;
            any |= *d != 0.0;
        }
        if !any {
            continue;
        }
        if x == q {
            traverse(ctx, c, p, &child, rr, mode, unit, &mut tmp);
        } else {
            traverse(ctx, c, p, rq, &child, mode, unit, &mut tmp);
        }
        for (o, &v) in out.iter_mut().zip(tmp.iter()) {
            *o += v;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn chance(ctx: &Ctx, node: &Node, p: usize, rq: &[f32], rr: &[f32], mode: Mode, unit: bool, out: &mut [f32]) {
    let n = out.len();
    if n == 0 {
        return;
    }
    let (q, r) = opponents(p);
    let factor = node.factor;
    let k = node.num_actions();
    let mut rows = Buf::uninit(k * n);
    rows.par_chunks_mut(n).zip(node.children().into_par_iter()).for_each(|(row, c)| {
        let card = ctx.tree.card(c).index();
        // The chance factor is applied once (the values are bilinear in
        // the two opponents' reaches).
        let mut a = Buf::from_slice(rq);
        for x in a.iter_mut() {
            *x *= factor;
        }
        for &h in &ctx.hands.blocked[q][card] {
            a[h as usize] = 0.0;
        }
        let mut b = Buf::from_slice(rr);
        for &h in &ctx.hands.blocked[r][card] {
            b[h as usize] = 0.0;
        }
        traverse(ctx, c, p, &a, &b, mode, unit, row);
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
    for (o, a) in out.iter_mut().zip(acc) {
        *o = a as f32;
    }
}

/// Per player: `1 / Σ_h weight(h) * pairs(h)` over the initial weights.
pub(super) fn norms(hands: &Hands, exact: bool) -> [f64; N] {
    std::array::from_fn(|p| {
        let (q, r) = opponents(p);
        let pairs: Vec<f64> = if exact {
            exact::total_pairs(hands, p, &hands.weights[q], &hands.weights[r])
        } else {
            let cq = hands.compat(p, q, &hands.weights[q]);
            let cr = hands.compat(p, r, &hands.weights[r]);
            cq.iter().zip(&cr).map(|(a, b)| a * b).collect()
        };
        let total: f64 = (0..hands.len(p)).map(|h| hands.weights[p][h] as f64 * pairs[h]).sum();
        if total > 0.0 {
            1.0 / total
        } else {
            0.0
        }
    })
}
