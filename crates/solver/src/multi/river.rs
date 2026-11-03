//! Breadth-first evaluation of three-player river subtrees.
//!
//! A river subtree has no chance nodes and a single board, so (as in the
//! heads-up solver) it is processed in phases instead of depth-first:
//! 1. top-down: both opponents' reach at every node;
//! 2. all leaves at once: with exact card removal, showdowns that have the
//!    same folded players share one strength sweep, [`LANES`] at a time;
//!    leaves whose payoff doesn't depend on the cards (an uncontested win,
//!    or the traverser folded) are grouped by reach so each distinct pair of
//!    opponent reaches is summed once;
//! 3. bottom-up: values, and the traverser's regret and strategy updates.

use super::cfr::{
    accumulate_strategy, independent_showdown, own_node_combine, pay_all, payoffs, strategy_updater, Ctx,
};
use super::exact;
use super::opponents;
use super::tree::Kind;
use crate::cfr::{normalize_in_place, Mode};
use std::collections::HashMap;

/// Leaves with a fixed payoff, by their pair of opponent reach slots:
/// `(item, payoff)`.
type FixedLeaves = HashMap<(u32, u32), Vec<(usize, f64)>>;

/// Terminals evaluated per exact sweep.
pub(super) const LANES: usize = 4;

#[derive(Clone, Copy)]
struct Item {
    id: u32,
    first_child: u32,
    /// Reach slots of the two opponents.
    sq: u32,
    sr: u32,
    dead: bool,
}

fn slot_of(buf: &[f32], len: usize, slot: u32) -> &[f32] {
    &buf[slot as usize * len..(slot as usize + 1) * len]
}

/// Values of `p` at river decision node `root` (same contract as
/// [`super::cfr::traverse`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn subtree(
    ctx: &Ctx,
    root: usize,
    p: usize,
    rq: &[f32],
    rr: &[f32],
    mode: Mode,
    unit: bool,
    out: &mut [f32],
) {
    let nodes = &ctx.tree.nodes;
    let (q, r) = opponents(p);
    let (nq, nr, n) = (ctx.hands.len(q), ctx.hands.len(r), out.len());
    if n == 0 {
        return;
    }
    // Phase 0: items in breadth-first order. Nodes after the traverser
    // folded are leaves.
    let expands = |id: usize| matches!(nodes[id].kind, Kind::Player(_)) && !nodes[id].folded[p];
    let mut items = vec![Item { id: root as u32, first_child: 0, sq: 0, sr: 0, dead: false }];
    let mut k = 0;
    while k < items.len() {
        let id = items[k].id as usize;
        if expands(id) {
            items[k].first_child = items.len() as u32;
            for c in nodes[id].children() {
                items.push(Item { id: c as u32, first_child: 0, sq: 0, sr: 0, dead: false });
            }
        }
        k += 1;
    }

    // Phase 1: top-down opponent reach (one slot per distinct vector).
    let mut reach_q: Vec<f32> = rq.to_vec();
    let mut reach_r: Vec<f32> = rr.to_vec();
    let mut strat = Vec::new();
    for k in 0..items.len() {
        let it = items[k];
        let id = it.id as usize;
        if !expands(id) {
            continue;
        }
        let node = &nodes[id];
        let fc = it.first_child as usize;
        let na = node.num_actions();
        if it.dead {
            for c in fc..fc + na {
                items[c].dead = true;
            }
            continue;
        }
        let Kind::Player(x) = node.kind else { unreachable!() };
        let x = x as usize;
        if x == p {
            for c in fc..fc + na {
                items[c].sq = it.sq;
                items[c].sr = it.sr;
            }
            continue;
        }
        let nx = if x == q { nq } else { nr };
        strat.resize(na * nx, 0.0);
        let b = node.block(nx);
        unsafe {
            match mode {
                Mode::Cfr(_) => ctx.store.load_current_strategy(b, na, nx, &mut strat),
                _ => {
                    ctx.store.load_strat(b, &mut strat);
                    normalize_in_place(&mut strat, na, nx);
                }
            }
        }
        if let Mode::Cfr(d) = mode {
            if strategy_updater(node, x) == p {
                let rx = if x == q { slot_of(&reach_q, nq, it.sq) } else { slot_of(&reach_r, nr, it.sr) };
                accumulate_strategy(ctx, node, nx, &strat, rx, &d);
            }
        }
        for a in 0..na {
            let c = fc + a;
            let row = &strat[a * nx..(a + 1) * nx];
            let (buf, parent) = if x == q { (&mut reach_q, it.sq) } else { (&mut reach_r, it.sr) };
            let slot = buf.len() / nx;
            let start = parent as usize * nx;
            let mut any = false;
            for j in 0..nx {
                let v = buf[start + j] * row[j];
                any |= v != 0.0;
                buf.push(v);
            }
            items[c].sq = if x == q { slot as u32 } else { it.sq };
            items[c].sr = if x == q { it.sr } else { slot as u32 };
            items[c].dead = !any;
        }
    }
    let slot_q = |s: u32| &reach_q[s as usize * nq..(s as usize + 1) * nq];
    let slot_r = |s: u32| &reach_r[s as usize * nr..(s as usize + 1) * nr];

    // Phase 2: leaves.
    let mut vals = vec![0f32; items.len() * n];
    {
        let mut rows: Vec<Option<&mut [f32]>> = vals.chunks_exact_mut(n).map(Some).collect();
        // Showdowns by folded players (exact removal), and leaves with a
        // fixed payoff by their pair of reach slots.
        let mut showdowns: HashMap<[bool; 3], Vec<usize>> = HashMap::new();
        let mut fixed: FixedLeaves = HashMap::new();
        for (k, it) in items.iter().enumerate() {
            let id = it.id as usize;
            if it.dead || expands(id) {
                continue;
            }
            let node = &nodes[id];
            let (win, lose) = payoffs(ctx, node, p, unit);
            match node.kind {
                Kind::Showdown if !node.folded[p] => {
                    if ctx.exact {
                        showdowns.entry(node.folded).or_default().push(k);
                    } else {
                        let table = &ctx.tables[node.slot as usize];
                        let row = rows[k].as_deref_mut().unwrap();
                        independent_showdown(
                            ctx.hands,
                            table,
                            node.folded,
                            p,
                            slot_q(it.sq),
                            slot_r(it.sr),
                            win,
                            lose,
                            row,
                        );
                    }
                }
                Kind::Won(w) if w as usize == p => fixed.entry((it.sq, it.sr)).or_default().push((k, win)),
                _ => fixed.entry((it.sq, it.sr)).or_default().push((k, lose)),
            }
        }
        for (folded, ks) in &showdowns {
            for chunk in ks.chunks(LANES) {
                let table = &ctx.tables[nodes[items[chunk[0]].id as usize].slot as usize];
                let mut win = [0.0; LANES];
                let mut lose = [0.0; LANES];
                let empty_q = vec![0f32; nq];
                let empty_r = vec![0f32; nr];
                let mut qs: [&[f32]; LANES] = [&empty_q; LANES];
                let mut rs: [&[f32]; LANES] = [&empty_r; LANES];
                let mut spare: Vec<Vec<f32>> = (chunk.len()..LANES).map(|_| vec![0f32; n]).collect();
                let mut taken: Vec<&mut [f32]> = chunk.iter().map(|&k| rows[k].take().unwrap()).collect();
                for (l, &k) in chunk.iter().enumerate() {
                    let it = items[k];
                    let node = &nodes[it.id as usize];
                    debug_assert_eq!(
                        node.slot, nodes[items[chunk[0]].id as usize].slot,
                        "one board per subtree"
                    );
                    (win[l], lose[l]) = payoffs(ctx, node, p, unit);
                    qs[l] = slot_q(it.sq);
                    rs[l] = slot_r(it.sr);
                }
                {
                    let mut it_outs =
                        taken.iter_mut().map(|s| &mut **s).chain(spare.iter_mut().map(|v| v.as_mut_slice()));
                    let mut outs: [&mut [f32]; LANES] = std::array::from_fn(|_| it_outs.next().unwrap());
                    exact::showdown_lanes::<LANES>(table, *folded, p, &qs, &rs, &win, &lose, &mut outs);
                }
                for (&k, row) in chunk.iter().zip(taken) {
                    rows[k] = Some(row);
                }
            }
        }
        // Fixed payoffs: one pair sum per distinct reach pair.
        let groups: Vec<_> = fixed.iter().collect();
        let mut totals = vec![vec![0f32; n]; LANES];
        for chunk in groups.chunks(LANES) {
            let empty_q = vec![0f32; nq];
            let empty_r = vec![0f32; nr];
            let mut qs: [&[f32]; LANES] = [&empty_q; LANES];
            let mut rs: [&[f32]; LANES] = [&empty_r; LANES];
            for (l, ((sq, sr), _)) in chunk.iter().enumerate() {
                qs[l] = slot_q(*sq);
                rs[l] = slot_r(*sr);
            }
            if ctx.exact {
                let mut it_outs = totals.iter_mut();
                let mut outs: [&mut [f32]; LANES] =
                    std::array::from_fn(|_| it_outs.next().unwrap().as_mut_slice());
                exact::fold_values_lanes::<LANES>(ctx.hands, p, &qs, &rs, &[1.0; LANES], &mut outs);
            } else {
                for l in 0..chunk.len() {
                    pay_all(ctx, p, qs[l], rs[l], 1.0, &mut totals[l]);
                }
            }
            for (l, (_, leaves)) in chunk.iter().enumerate() {
                for &(k, pay) in leaves.iter() {
                    let row = rows[k].as_deref_mut().unwrap();
                    for (o, &t) in row.iter_mut().zip(&totals[l]) {
                        *o = (pay * t as f64) as f32;
                    }
                }
            }
        }
    }

    // Phase 3: bottom-up values (and the traverser's updates).
    for k in (0..items.len()).rev() {
        let it = items[k];
        let id = it.id as usize;
        if it.dead || !expands(id) {
            continue;
        }
        let node = &nodes[id];
        let na = node.num_actions();
        let fc = it.first_child as usize;
        let (head, tail) = vals.split_at_mut(fc * n);
        let me = &mut head[k * n..(k + 1) * n];
        let child = &tail[..na * n];
        match node.kind {
            Kind::Player(x) if x as usize == p => own_node_combine(ctx, node, n, child, mode, me),
            _ => {
                me.fill(0.0);
                for row in child.chunks_exact(n) {
                    for (m, &v) in me.iter_mut().zip(row) {
                        *m += v;
                    }
                }
            }
        }
    }
    out.copy_from_slice(&vals[..n]);
}
