//! Breadth-first evaluation of river subtrees.
//!
//! A river subtree (a river decision node and everything below it) has no
//! chance nodes, so it can be processed in three phases instead of
//! depth-first recursion:
//! 1. top-down: the opponent's reach at every node (and, in CFR mode, the
//!    opponent's strategy-sum update);
//! 2. all terminals at once: showdowns (which share one board) and folds
//!    are evaluated in batches of [`LANES`] with one sweep per batch;
//! 3. bottom-up: values and regret updates of the traverser's nodes.
//!
//! This is the hot path of turn and flop solves; it must give exactly the
//! same results as the recursive traversal (see the tests).

use crate::cfr::{fold_payoff, normalize_in_place, own_node_combine, showdown_payoffs, Ctx, Mode};
use crate::eval::{fold_batch, showdown_batch, LANES};
use crate::tree::Kind;
use std::cell::RefCell;

#[derive(Clone, Copy)]
struct Item {
    id: u32,
    /// Index of the first child item (children are contiguous).
    first_child: u32,
    /// Opponent-reach slot.
    reach: u32,
    /// Subtree has zero opponent reach: values are zero, no updates.
    dead: bool,
}

#[derive(Default)]
struct Scratch {
    items: Vec<Item>,
    reach: Vec<f32>,
    vals: Vec<f32>,
    strat: Vec<f32>,
    /// Terminal groups: (reach slot, showdown item, fold item), NONE if absent.
    groups: Vec<(u32, u32, u32)>,
    /// Reach slot -> group index.
    group_of: Vec<u32>,
}

const NONE: u32 = u32::MAX;

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Resizes `v` to `len` without clearing existing elements (only growth is
/// zero-filled), for buffers that are completely overwritten.
fn set_len_unzeroed(v: &mut Vec<f32>, len: usize) {
    if v.len() >= len {
        v.truncate(len);
    } else {
        v.resize(len, 0.0);
    }
}

pub(crate) fn subtree(ctx: &Ctx, root: usize, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32]) {
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        run(ctx, root, p, opp_reach, mode, out, &mut s);
    });
}

fn run(ctx: &Ctx, root: usize, p: usize, opp_reach: &[f32], mode: Mode, out: &mut [f32], s: &mut Scratch) {
    let nodes = &ctx.tree.nodes;
    let n = out.len();
    let nq = opp_reach.len();
    let Scratch { items, reach, vals, strat, groups, group_of } = s;

    // Phase 0: items in breadth-first order (parents before children).
    items.clear();
    let root_dead = opp_reach.iter().all(|&w| w == 0.0);
    items.push(Item { id: root as u32, first_child: 0, reach: 0, dead: root_dead });
    let mut k = 0;
    let mut reach_slots = 1usize;
    while k < items.len() {
        let node = &nodes[items[k].id as usize];
        if let Kind::Player(q) = node.kind {
            items[k].first_child = items.len() as u32;
            for c in node.children() {
                items.push(Item { id: c as u32, first_child: 0, reach: 0, dead: false });
            }
            if q as usize != p {
                reach_slots += node.num_actions();
            }
        }
        k += 1;
    }

    // Phase 1: top-down opponent reach. Every slot is written before it is
    // read (dead items' slots are never read), so no zeroing.
    set_len_unzeroed(reach, reach_slots * nq);
    reach[..nq].copy_from_slice(opp_reach);
    let mut next_slot = 1usize;
    for k in 0..items.len() {
        let it = items[k];
        let node = &nodes[it.id as usize];
        let Kind::Player(q) = node.kind else { continue };
        let na = node.num_actions();
        let fc = it.first_child as usize;
        if it.dead {
            for c in fc..fc + na {
                items[c].dead = true;
            }
            continue;
        }
        if q as usize == p {
            for c in fc..fc + na {
                items[c].reach = it.reach;
            }
            continue;
        }
        // Opponent node: its strategy splits the reach.
        let b = node.block(nq);
        set_len_unzeroed(strat, na * nq); // overwritten by the load below
        unsafe {
            match mode {
                Mode::Cfr(d) => {
                    ctx.store.load_current_strategy(b, na, nq, strat);
                    let r = &reach[it.reach as usize * nq..(it.reach as usize + 1) * nq];
                    let g = d.strat;
                    let weighted = d.reach_weighted;
                    ctx.store.update_strat(b, |ss| {
                        for (ss_row, s_row) in ss.chunks_exact_mut(nq).zip(strat.chunks_exact(nq)) {
                            if weighted {
                                for ((x, &y), &w) in ss_row.iter_mut().zip(s_row).zip(r) {
                                    *x = *x * g + y * w;
                                }
                            } else {
                                for (x, &y) in ss_row.iter_mut().zip(s_row) {
                                    *x = *x * g + y;
                                }
                            }
                        }
                    });
                }
                _ => {
                    ctx.store.load_strat(b, strat);
                    normalize_in_place(strat, na, nq);
                }
            }
        }
        let parent = it.reach as usize * nq;
        for a in 0..na {
            let slot = next_slot;
            next_slot += 1;
            let (head, tail) = reach.split_at_mut(slot * nq);
            let src = &head[parent..parent + nq];
            let dst = &mut tail[..nq];
            let row = &strat[a * nq..(a + 1) * nq];
            let mut any = false;
            for ((d, &r), &x) in dst.iter_mut().zip(src).zip(row) {
                *d = r * x;
                any |= *d != 0.0;
            }
            items[fc + a].reach = slot as u32;
            items[fc + a].dead = !any;
        }
    }

    // Phase 2: terminals. Terminals that share an opponent reach (a fold
    // and the call next to it, both the traverser's actions) share one
    // strength sweep, which yields every showdown and fold value.
    vals.clear();
    vals.resize(items.len() * n, 0.0);
    if n == 0 {
        return;
    }
    let reach_of = |slot: u32| -> &[f32] { &reach[slot as usize * nq..(slot as usize + 1) * nq] };
    group_of.clear();
    group_of.resize(reach_slots, NONE);
    groups.clear();
    let mut board_table = None;
    for (k, it) in items.iter().enumerate() {
        if it.dead {
            continue;
        }
        let node = &nodes[it.id as usize];
        let is_showdown = match node.kind {
            Kind::Showdown => true,
            Kind::Fold(_) => false,
            _ => continue,
        };
        if is_showdown {
            debug_assert!(board_table.is_none_or(|t| t == node.offset), "one board per subtree");
            board_table = Some(node.offset);
        }
        // A reach is shared by at most one showdown and one fold (both
        // children of one traverser node); anything else gets its own group.
        let g = group_of[it.reach as usize];
        let fits = g != NONE && {
            let grp = groups[g as usize];
            if is_showdown {
                grp.1 == NONE
            } else {
                grp.2 == NONE
            }
        };
        let g = if fits {
            g as usize
        } else {
            group_of[it.reach as usize] = groups.len() as u32;
            groups.push((it.reach, NONE, NONE));
            groups.len() - 1
        };
        if is_showdown {
            groups[g].1 = k as u32;
        } else {
            groups[g].2 = k as u32;
        }
    }
    // Groups with a showdown first.
    groups.sort_unstable_by_key(|g| g.1 == NONE);
    let split = groups.iter().position(|g| g.1 == NONE).unwrap_or(groups.len());
    // Item value slots are disjoint `n`-sized chunks of `vals`.
    let mut slots: Vec<Option<&mut [f32]>> = vals.chunks_exact_mut(n).map(Some).collect();
    if let Some(t) = board_table {
        let table = &ctx.tables[t as usize];
        for chunk in groups[..split].chunks(LANES) {
            let reaches: Vec<&[f32]> = chunk.iter().map(|g| reach_of(g.0)).collect();
            let (mut win, mut lose, mut tie) = (Vec::new(), Vec::new(), Vec::new());
            let mut outs = Vec::with_capacity(chunk.len());
            let mut folds = Vec::with_capacity(chunk.len());
            for &(_, sd, fd) in chunk {
                let (w, l, t) = showdown_payoffs(ctx, &nodes[items[sd as usize].id as usize], p);
                win.push(w);
                lose.push(l);
                tie.push(t);
                outs.push(slots[sd as usize].take().unwrap());
                folds.push((fd != NONE).then(|| {
                    (
                        fold_payoff(ctx, &nodes[items[fd as usize].id as usize], p),
                        slots[fd as usize].take().unwrap(),
                    )
                }));
            }
            showdown_batch(ctx.hands, table, p, &reaches, &win, &lose, &tie, &mut outs, &mut folds);
        }
    }
    for chunk in groups[split..].chunks(LANES) {
        let reaches: Vec<&[f32]> = chunk.iter().map(|g| reach_of(g.0)).collect();
        let payoffs: Vec<f64> =
            chunk.iter().map(|g| fold_payoff(ctx, &nodes[items[g.2 as usize].id as usize], p)).collect();
        let mut outs: Vec<&mut [f32]> = chunk.iter().map(|g| slots[g.2 as usize].take().unwrap()).collect();
        fold_batch(ctx.hands, p, &reaches, &payoffs, &mut outs);
    }
    drop(slots);

    // Phase 3: bottom-up values (and the traverser's regret updates).
    for k in (0..items.len()).rev() {
        let it = items[k];
        let node = &nodes[it.id as usize];
        let Kind::Player(q) = node.kind else { continue };
        if it.dead {
            continue;
        }
        let na = node.num_actions();
        let fc = it.first_child as usize;
        let (head, tail) = vals.split_at_mut(fc * n);
        let me = &mut head[k * n..(k + 1) * n];
        let child = &tail[..na * n];
        if q as usize != p {
            me.fill(0.0);
            for row in child.chunks_exact(n) {
                for (m, &v) in me.iter_mut().zip(row) {
                    *m += v;
                }
            }
            continue;
        }
        own_node_combine(ctx, node, n, child, mode, me);
    }
    out.copy_from_slice(&vals[..n]);
}
