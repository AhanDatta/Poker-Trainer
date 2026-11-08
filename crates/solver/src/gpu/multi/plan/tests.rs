//! 3-way plan construction and validation (no device needed).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::gpu::shared::{unpack_cards, MAX_ACTIONS};
use crate::gpu::multi::testutil::{arb_multi_config, spots};
use crate::multi::cfr::{independent_showdown, pay_all};
use crate::multi::{MultiConfig, MultiSolver};
use holdem::Street;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

mod leaves;
mod mutations;
mod split;

/// `cfg` with each terminal model and river batching on and off (batching
/// only takes effect with exact removal).
fn variants(cfg: &MultiConfig) -> Vec<(bool, bool, MultiSolver)> {
    let mut out = Vec::new();
    for exact in [false, true] {
        for batch in [true, false] {
            let mut c = cfg.clone();
            c.exact_removal = exact;
            let mut m = MultiSolver::new(c).unwrap();
            m.set_batch_river(batch);
            out.push((exact, batch, m));
        }
    }
    out
}

fn plan_for(m: &MultiSolver) -> MultiPlan {
    MultiPlan::build(&m.ctx()).unwrap()
}

/// What the CPU traversal of player `p` visits (`cfr::traverse`, switching
/// to `river::subtree` at river decision nodes when batching): expanded
/// nodes with their depth, and leaves with whether they're batched.
struct Walk {
    expanded: Vec<(usize, u32)>,
    leaves: Vec<(usize, bool)>,
}

fn cpu_walk(ctx: &Ctx, p: usize) -> Walk {
    fn go(ctx: &Ctx, id: usize, p: usize, depth: u32, in_batch: bool, w: &mut Walk) {
        let node = &ctx.tree.nodes[id];
        match node.kind {
            Kind::Won(_) | Kind::Showdown => w.leaves.push((id, in_batch)),
            _ if node.folded[p] => w.leaves.push((id, in_batch)),
            Kind::Chance => {
                w.expanded.push((id, depth));
                for c in node.children() {
                    go(ctx, c, p, depth + 1, false, w);
                }
            }
            Kind::Player(_) => {
                let batch = in_batch || (ctx.batch_river && node.street == Street::River);
                w.expanded.push((id, depth));
                for c in node.children() {
                    go(ctx, c, p, depth + 1, batch, w);
                }
            }
        }
    }
    let mut w = Walk { expanded: Vec::new(), leaves: Vec::new() };
    go(ctx, 0, p, 0, false, &mut w);
    w
}

fn sorted(v: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut v: Vec<usize> = v.into_iter().collect();
    v.sort_unstable();
    v
}

fn ids(v: &[u32]) -> Vec<usize> {
    sorted(v.iter().map(|&i| i as usize))
}

/// The schedule of every traverser is the CPU traversal: the same nodes
/// expanded, in the right lists, at their depth; the same leaves, each in
/// the right list with the CPU's payoff; the CPU evaluator's rounding.
fn check_schedule(ctx: &Ctx, plan: &MultiPlan) {
    let nodes = &ctx.tree.nodes;
    for (p, pp) in plan.players.iter().enumerate() {
        let w = cpu_walk(ctx, p);
        let (q, r) = opponents(p);
        let of = |pred: &dyn Fn(Kind) -> bool| {
            sorted(w.expanded.iter().map(|e| e.0).filter(|&i| pred(nodes[i].kind)))
        };
        assert_eq!(ids(&pp.own_nodes), of(&|k| k == Kind::Player(p as u8)), "player {p}: own nodes");
        assert_eq!(ids(&pp.opp_nodes[0]), of(&|k| k == Kind::Player(q as u8)), "player {p}: first opponent");
        assert_eq!(ids(&pp.opp_nodes[1]), of(&|k| k == Kind::Player(r as u8)), "player {p}: second opponent");
        assert_eq!(ids(&pp.chance_nodes), of(&|k| k == Kind::Chance), "player {p}: chance nodes");
        let mut level_of = vec![u32::MAX; nodes.len()];
        for (d, lv) in pp.levels.iter().enumerate() {
            for (span, list) in [
                (lv.opp[0], &pp.opp_nodes[0]),
                (lv.opp[1], &pp.opp_nodes[1]),
                (lv.own, &pp.own_nodes),
                (lv.chance, &pp.chance_nodes),
            ] {
                for &i in &list[span.range()] {
                    level_of[i as usize] = d as u32;
                }
            }
        }
        for &(i, depth) in &w.expanded {
            assert_eq!(level_of[i], depth, "player {p}: node {i}'s level");
        }
        let showdowns: Vec<usize> = sorted(
            w.leaves.iter().map(|l| l.0).filter(|&i| nodes[i].kind == Kind::Showdown && !nodes[i].folded[p]),
        );
        assert_eq!(ids(&pp.showdowns), showdowns, "player {p}: showdowns");
        assert_eq!(pp.fixed.len() + pp.showdowns.len(), w.leaves.len(), "player {p}: leaves");
        for &(i, batched) in &w.leaves {
            assert_eq!(plan.scalar[i] == 0, batched, "player {p}: leaf {i}'s evaluator");
            let node = &nodes[i];
            let (win, lose) = payoffs(ctx, node, p, false);
            if let Some(k) = pp.showdowns.iter().position(|&s| s as usize == i) {
                assert_eq!(
                    (pp.win[k], pp.lose[k], pp.showdown_table[k]),
                    (win, lose, node.slot),
                    "showdown {i}"
                );
            } else {
                let k = pp.fixed.iter().position(|&f| f as usize == i).unwrap();
                let want = if node.kind == Kind::Won(p as u8) { win } else { lose };
                assert_eq!(pp.fixed_payoff[k], want, "player {p}: fixed leaf {i}");
            }
        }
    }
}

/// Reach slots: two nodes share opponent `k`'s slot exactly when the CPU
/// passes them the same reach vector of that opponent, i.e. when no node of
/// that opponent's and no chance node lies between them.
fn check_slots(plan: &MultiPlan) {
    let nn = plan.num_nodes as usize;
    let parents = plan.parents().unwrap();
    for (p, pp) in plan.players.iter().enumerate() {
        let reached = plan.reached(p).unwrap();
        for k in 0..2 {
            let seat = pp.opponents[k].player;
            // Per node: the topmost node sharing its reach of opponent k.
            let mut anchor = vec![usize::MAX; nn];
            let mut owner: HashMap<u32, usize> = HashMap::new();
            for i in 0..nn {
                if !reached[i] {
                    assert_eq!(pp.slot[k][i], NONE, "player {p}: unreached node {i}");
                    continue;
                }
                let pa = parents[i];
                anchor[i] =
                    if pa == NONE || plan.actor[pa as usize] == seat || plan.actor[pa as usize] == CHANCE {
                        i
                    } else {
                        anchor[pa as usize]
                    };
                let slot = pp.slot[k][i];
                let first = *owner.entry(slot).or_insert(anchor[i]);
                assert_eq!(first, anchor[i], "player {p}, opponent {k}: slot {slot} holds two reach vectors");
            }
        }
    }
}

/// Each decision node's average strategy is accumulated in the traversal
/// `strategy_updater` names, and only there.
fn check_strategy_updates(ctx: &Ctx, plan: &MultiPlan) {
    for (i, node) in ctx.tree.nodes.iter().enumerate() {
        let flags: Vec<u32> = plan.players.iter().map(|pp| pp.strategy_update[i]).collect();
        match node.kind {
            Kind::Player(x) => {
                let u = strategy_updater(node, x as usize);
                let want: Vec<u32> = (0..N).map(|p| u32::from(p == u)).collect();
                assert_eq!(flags, want, "node {i}");
            }
            _ => assert_eq!(flags, vec![0; N], "node {i}"),
        }
    }
}

fn check_all(m: &MultiSolver, plan: &MultiPlan) {
    let ctx = m.ctx();
    assert_eq!(plan.num_nodes as usize, ctx.tree.nodes.len());
    check_schedule(&ctx, plan);
    check_slots(plan);
    check_strategy_updates(&ctx, plan);
}

#[test]
fn multi_test_spots_build_and_match_the_cpu() {
    for (name, cfg) in spots() {
        for (exact, batch, m) in variants(&cfg) {
            let plan = plan_for(&m);
            plan.validate().unwrap();
            assert_eq!((plan.exact, plan.batch_river), (exact, exact && batch), "{name}");
            check_all(&m, &plan);
            // Every decision node is someone's, in someone's schedule.
            let decisions = plan.actor.iter().filter(|&&a| (a as usize) < N).count();
            let marked: u32 = plan.players.iter().map(|pp| pp.strategy_update.iter().sum::<u32>()).sum();
            assert_eq!(marked as usize, decisions, "{name}: strategy updates");
        }
    }
}

/// The spots have the shapes they were picked for.
#[test]
fn spots_cover_the_tree_shapes() {
    let spots: HashMap<&str, MultiConfig> = spots().into_iter().collect();
    let build = |name: &str| {
        let m = MultiSolver::new(spots[name].clone()).unwrap();
        let plan = plan_for(&m);
        (m, plan)
    };
    // A player folded at the root: its whole traversal is one fixed leaf.
    let (_, plan) = build("first player folded");
    let p0 = &plan.players[0];
    assert!(p0.levels.is_empty() && p0.showdowns.is_empty() && p0.num_slots == [1, 1]);
    assert_eq!((p0.fixed.clone(), p0.slot[0][0], p0.slot[1][0]), (vec![0], 0, 0));
    assert!(plan.players[1].levels.len() > 2 && plan.players[2].levels.len() > 2);
    // All-in runouts: chance nodes whose children are showdowns.
    let (_, plan) = build("all-in turn");
    let runout = (0..plan.num_nodes as usize)
        .filter(|&i| plan.actor[i] == CHANCE)
        .any(|i| plan.children(i).unwrap().any(|c| plan.actor[c] == NONE && plan.winner[c] == NONE));
    assert!(runout, "no all-in runout");
    // Two levels of chance nodes on the flop.
    let (_, plan) = build("sampled flop");
    let chance_children_of_chance =
        (0..plan.num_nodes as usize).filter(|&i| plan.actor[i] == CHANCE).any(|i| {
            let mut stack: Vec<usize> = plan.children(i).unwrap().collect();
            while let Some(c) = stack.pop() {
                if plan.actor[c] == CHANCE {
                    return true;
                }
                stack.extend(plan.children(c).unwrap());
            }
            false
        });
    assert!(chance_children_of_chance);
    // Every traverser cuts off subtrees after its folds somewhere, and a cut
    // node can be a decision node (not just an uncontested win).
    let (_, plan) = build("sampled turn");
    for (p, pp) in plan.players.iter().enumerate() {
        assert!(
            pp.fixed.iter().any(|&f| plan.actor[f as usize] != NONE),
            "player {p}: no cut-off decision node"
        );
        assert!(
            pp.fixed.iter().any(|&f| plan.winner[f as usize] == p as u32),
            "player {p}: never wins uncontested"
        );
        assert!(pp.strategy_update.contains(&1), "player {p}: updates no strategy");
    }
}

/// Terminals follow the CPU evaluator: one at a time with independent
/// opponents (the CPU never batches those) or batching off; with exact
/// removal, batched below river decision nodes and one at a time elsewhere.
#[test]
fn scalar_flags_follow_the_cpu_evaluators() {
    let cfg = spots().into_iter().find(|s| s.0 == "all-in turn").unwrap().1;
    for (exact, batch, m) in variants(&cfg) {
        let plan = plan_for(&m);
        let batched = plan.scalar.iter().filter(|&&s| s == 0).count();
        if exact && batch {
            assert!(batched > 0 && batched < plan.scalar.len());
            // Runout showdowns (children of chance nodes) are one at a time.
            for i in 0..plan.num_nodes as usize {
                if plan.actor[i] == CHANCE {
                    assert!(plan.children(i).unwrap().all(|c| plan.scalar[c] == 1));
                }
            }
        } else {
            assert_eq!(batched, 0, "exact {exact}, batch {batch}");
        }
    }
}

/// Guards against vacuous property tests: most generated spots must be
/// solvable (the rest have an empty range after board removal).
#[test]
fn generated_spots_are_mostly_solvable() {
    use proptest::strategy::ValueTree;
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let strategy = arb_multi_config();
    let total = 64;
    let ok = (0..total)
        .filter(|_| MultiSolver::new(strategy.new_tree(&mut runner).unwrap().current().0).is_ok())
        .count();
    assert!(ok * 4 >= total * 3, "only {ok} of {total} generated spots are solvable");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random spots give valid plans whose schedules, slots, strategy
    /// marks, payoffs and rounding flags match the CPU traversal.
    #[test]
    fn random_plans_validate_and_match_the_cpu((cfg, batch) in arb_multi_config()) {
        let Ok(mut m) = MultiSolver::new(cfg) else { return Ok(()) };
        m.set_batch_river(batch);
        let plan = MultiPlan::build(&m.ctx()).unwrap();
        prop_assert!(plan.validate().is_ok());
        check_all(&m, &plan);
    }
}
