//! 3-way plan construction and validation (no device needed).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::gpu::testutil::random_range;
use crate::multi::cfr::{independent_showdown, pay_all};
use crate::multi::tests::config;
use crate::multi::{multiway_dcfr, MultiConfig, MultiSolver, MultiState};
use crate::{ChanceMode, StorageMode};
use holdem::{Card, Chips, PostflopMenu, Street, StreetSizes};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// The ranges of `three_way_turn_spot` and `batched_river_matches_recursion`.
const RANGES: [&str; 3] = [
    "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
    "AA, KK, AJ, QJ, JT, 99, 77, KQs",
    "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
];

/// The ranges of `narrow_overlapping_ranges_stay_finite`.
const NARROW: [&str; 3] = ["AA, KK, QQ, AK, AQs", "AK, AQ, AJs, KK, QQ, JJ", "AQ, AJ, KQ, TT, 99, 88, A5s"];

fn sampled(mut c: MultiConfig, turn: usize, river: usize, seed: u64) -> MultiConfig {
    c.chance = ChanceMode::Sampled { turn, river, seed };
    c
}

/// The spots of the `multi` tests, plus shapes they lack: an all-in runout,
/// a player folded at the root, a sampled flop.
fn spots() -> Vec<(&'static str, MultiConfig)> {
    let hu = ["AA, KK, QJs, T9s, 76s, A5s", "AK, QQ, JJ, KQs, 98s, 65s", "any"];
    let mut allin = config("Jh8d4c2s", RANGES, [false; N]);
    allin.root = MultiState::street_start(Street::Turn, 900, [400, 400, 3000], [false; N], 100);
    vec![
        ("river, third player folded", config("Kd9s5h2c2d", hu, [false, false, true])),
        ("river", config("Jh8d4c2s9s", RANGES, [false; N])),
        ("sampled turn", sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 12, 1)),
        ("exact turn", config("Jh8d4c2s", RANGES, [false; N])),
        ("narrow turn", config("Ah7c2d5s", NARROW, [false; N])),
        ("all-in turn", sampled(allin, 0, 5, 3)),
        ("first player folded", sampled(config("Jh8d4c2s", RANGES, [true, false, false]), 0, 6, 2)),
        ("sampled flop", sampled(config("Ks7d2c", NARROW, [false; N]), 2, 2, 5)),
    ]
}

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

/// Random reach with about a quarter of the hands at zero.
fn random_reach(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| if rng.random::<f64>() < 0.25 { 0.0 } else { rng.random_range(0.0f32..1.0) }).collect()
}

/// `Hands::compat` from an opponent view's card lists.
fn plan_compat(pp: &MultiPlayerPlan, view: &OpponentView, reach: &[f32]) -> Vec<f64> {
    let total: f64 = reach.iter().map(|&x| f64::from(x)).sum();
    let minus: Vec<f64> = (0..CARDS)
        .map(|c| {
            view.card_hands[view.card_off[c] as usize..view.card_off[c + 1] as usize]
                .iter()
                .map(|&j| f64::from(reach[j as usize]))
                .sum()
        })
        .collect();
    (0..pp.hands as usize)
        .map(|h| {
            let (c1, c2) = unpack_cards(pp.my_cards[h]);
            let same = match view.same[h] {
                NONE => 0.0,
                j => f64::from(reach[j as usize]),
            };
            (total - minus[c1] - minus[c2] + same).max(0.0)
        })
        .collect()
}

/// `Table::beat` from an opponent view's lookup table `t` (the heads-up
/// showdown formula's weaker and tied sums).
fn plan_beat(pp: &MultiPlayerPlan, view: &OpponentView, t: usize, reach: &[f32]) -> Vec<f64> {
    let tb = &view.tables;
    let n = pp.hands as usize;
    let (off, len) = (tb.opp_off[t] as usize, tb.opp_len[t] as usize);
    let w: Vec<f64> = tb.opp_order[off..off + len].iter().map(|&j| f64::from(reach[j as usize])).collect();
    let mut prefix = vec![0f64; len + 1];
    for j in 0..len {
        prefix[j + 1] = prefix[j] + w[j];
    }
    let card_off = &tb.card_off[t * (CARDS + 1)..(t + 1) * (CARDS + 1)];
    let base = tb.card_base[t] as usize;
    let card_prefix = |c: usize| -> Vec<f64> {
        let mut v = vec![0.0];
        for &j in &tb.card_pos[base + card_off[c] as usize..base + card_off[c + 1] as usize] {
            v.push(v.last().unwrap() + w[j as usize]);
        }
        v
    };
    (0..n)
        .map(|h| {
            let packed = tb.ranges[t * n + h];
            if packed == NONE {
                return 0.0;
            }
            let (lo, hi) = ((packed & 0xFFFF) as usize, (packed >> 16) as usize);
            let r = tb.ranks[t * n + h];
            let (r1l, r1h, r2l, r2h) = (
                (r & 0xFF) as usize,
                (r >> 8 & 0xFF) as usize,
                (r >> 16 & 0xFF) as usize,
                (r >> 24) as usize,
            );
            let (c1, c2) = unpack_cards(pp.my_cards[h]);
            let (f, s) = (card_prefix(c1), card_prefix(c2));
            let weaker = prefix[lo] - f[r1l] - s[r2l];
            let same = match view.same[h] {
                NONE => 0.0,
                j => f64::from(reach[j as usize]),
            };
            let equal = (prefix[hi] - prefix[lo]) - ((f[r1h] - f[r1l]) + (s[r2h] - s[r2l])) + same;
            (weaker + 0.5 * equal).max(0.0)
        })
        .collect()
}

fn assert_close_f64(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let scale = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-30);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 1e-12 * scale, "{what}: hand {i}: {x} vs {y}");
    }
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 2e-6 * scale, "{what}: hand {i}: {x} vs {y} (scale {scale})");
    }
}

/// The plan's lookup data reproduces the CPU's independent-opponent
/// terminals (per-opponent compatible and beaten mass, showdown values and
/// fixed payoffs) for random reach, on every board of the plan.
fn check_independent_terminals(m: &MultiSolver, plan: &MultiPlan, seed: u64) {
    let ctx = m.ctx();
    assert!(!ctx.exact);
    let mut rng = StdRng::seed_from_u64(seed);
    for (p, pp) in plan.players.iter().enumerate() {
        let (q, r) = opponents(p);
        let rq = random_reach(&mut rng, pp.opponents[0].hands as usize);
        let rr = random_reach(&mut rng, pp.opponents[1].hands as usize);
        let cq = plan_compat(pp, &pp.opponents[0], &rq);
        let cr = plan_compat(pp, &pp.opponents[1], &rr);
        assert_close_f64(&cq, &ctx.hands.compat(p, q, &rq), "compat q");
        assert_close_f64(&cr, &ctx.hands.compat(p, r, &rr), "compat r");
        for t in 0..plan.num_tables as usize {
            let table = &ctx.tables[t];
            assert_close_f64(
                &plan_beat(pp, &pp.opponents[0], t, &rq),
                &table.beat(ctx.hands, p, q, &rq),
                "beat q",
            );
            assert_close_f64(
                &plan_beat(pp, &pp.opponents[1], t, &rr),
                &table.beat(ctx.hands, p, r, &rr),
                "beat r",
            );
        }
        // A few of the plan's showdowns and fixed leaves, valued from the
        // plan's sums with the CPU formula.
        for k in (0..pp.showdowns.len()).step_by(pp.showdowns.len().div_ceil(8).max(1)) {
            let node = &ctx.tree.nodes[pp.showdowns[k] as usize];
            let t = pp.showdown_table[k] as usize;
            let bq = if node.folded[q] { cq.clone() } else { plan_beat(pp, &pp.opponents[0], t, &rq) };
            let br = if node.folded[r] { cr.clone() } else { plan_beat(pp, &pp.opponents[1], t, &rr) };
            let (win, lose) = (pp.win[k], pp.lose[k]);
            let want: Vec<f32> = (0..pp.hands as usize)
                .map(|h| (win * bq[h] * br[h] + lose * (cq[h] * cr[h] - bq[h] * br[h])) as f32)
                .collect();
            let mut cpu = vec![0f32; pp.hands as usize];
            independent_showdown(ctx.hands, &ctx.tables[t], node.folded, p, &rq, &rr, win, lose, &mut cpu);
            assert_close(&want, &cpu, "showdown");
        }
        for k in (0..pp.fixed.len()).step_by(pp.fixed.len().div_ceil(8).max(1)) {
            let pay = pp.fixed_payoff[k];
            let want: Vec<f32> = (0..pp.hands as usize).map(|h| (pay * cq[h] * cr[h]) as f32).collect();
            let mut cpu = vec![0f32; pp.hands as usize];
            pay_all(&ctx, p, &rq, &rr, pay, &mut cpu);
            assert_close(&want, &cpu, "fixed payoff");
        }
    }
}

#[test]
fn lookup_tables_reproduce_cpu_independent_terminals() {
    for (name, mut cfg) in spots() {
        cfg.exact_removal = false;
        let m = MultiSolver::new(cfg).unwrap();
        let plan = plan_for(&m);
        // (A player folded at the root takes part in no showdown.)
        let in_showdowns = plan.players.iter().filter(|pp| !pp.showdowns.is_empty()).count();
        assert!(in_showdowns >= 2, "{name}: {in_showdowns} players in showdowns");
        for seed in 0..3 {
            check_independent_terminals(&m, &plan, seed);
        }
    }
}

/// Every mutation must be rejected.
fn assert_all_rejected(plan: &MultiPlan, mutations: Vec<(&str, Mutation)>) {
    for (what, mutate) in mutations {
        let mut bad = plan.clone();
        mutate(&mut bad);
        let r = bad.validate();
        eprintln!("MUT {what}: {r:?}");
        assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{what}: accepted");
    }
}

type Mutation = Box<dyn Fn(&mut MultiPlan)>;

fn find(plan: &MultiPlan, pred: impl Fn(usize) -> bool) -> usize {
    (0..plan.num_nodes as usize).find(|&i| pred(i)).unwrap()
}

#[test]
fn validate_rejects_corrupted_trees() {
    let cfg = spots().into_iter().find(|s| s.0 == "all-in turn").unwrap().1;
    let mut cfg = cfg;
    cfg.exact_removal = true;
    let m = MultiSolver::new(cfg).unwrap();
    let plan = plan_for(&m);
    plan.validate().unwrap();
    let parents = plan.parents().unwrap();
    let decision = find(&plan, |i| i > 0 && (plan.actor[i] as usize) < N);
    let other_decision = find(&plan, |i| i > decision && (plan.actor[i] as usize) < N);
    let chance = find(&plan, |i| plan.actor[i] == CHANCE);
    let dealt = plan.first_child[chance] as usize;
    let won = find(&plan, |i| plan.winner[i] != NONE);
    let showdown = find(&plan, |i| plan.actor[i] == NONE && plan.winner[i] == NONE);
    let below_decision = plan.first_child[0] as usize;
    // A batched leaf that isn't its parent's first child.
    let batched = find(&plan, |i| plan.scalar[i] == 0 && i != plan.first_child[parents[i] as usize] as usize);
    // A player in the hand at the chance node who doesn't act right after it.
    let free = (0..N).find(|&b| plan.folded[chance] >> b & 1 == 0 && b as u32 != plan.actor[dealt]).unwrap();
    let mutations: Vec<(&str, Mutation)> = vec![
        ("storage block out of range", Box::new(|p| p.offset[0] = p.storage_len)),
        ("children out of range", Box::new(|p| p.first_child[0] = p.num_nodes)),
        ("children before their parent", Box::new(move |p| p.first_child[decision] = 0)),
        ("two parents", Box::new(move |p| p.first_child[other_decision] = p.first_child[decision])),
        ("root actor changed", Box::new(|p| p.actor[0] = (p.actor[0] + 1) % 3)),
        ("root made a chance node", Box::new(|p| p.actor[0] = CHANCE)),
        ("actor out of range", Box::new(move |p| p.actor[decision] = 7)),
        ("actor folded", Box::new(move |p| p.folded[decision] |= 1 << p.actor[decision])),
        ("folded flags out of range", Box::new(|p| p.folded[0] = 8)),
        ("fold forgotten below a fold", Box::new(move |p| p.folded[won] = 0)),
        (
            "fold by a player not acting",
            Box::new(move |p| p.folded[below_decision] |= 1 << ((p.actor[0] + 2) % 3)),
        ),
        ("fold at a chance node", Box::new(move |p| p.folded[dealt] |= 1 << free)),
        ("wrong winner", Box::new(move |p| p.winner[won] = (p.winner[won] + 1) % 3)),
        ("winner out of range", Box::new(move |p| p.winner[won] = 3)),
        ("winner on a showdown", Box::new(move |p| p.winner[showdown] = 0)),
        ("winner on a decision node", Box::new(move |p| p.winner[decision] = 0)),
        ("showdown with one player", Box::new(move |p| p.folded[showdown] = 0b011)),
        ("terminal with children", Box::new(move |p| p.actions[showdown] = 1)),
        ("decision without children", Box::new(move |p| p.actions[decision] = 0)),
        ("too many actions", Box::new(|p| p.actions[0] = MAX_ACTIONS + 1)),
        ("scale slot shared", Box::new(move |p| p.scale_slot[other_decision] = p.scale_slot[decision])),
        ("scale slot out of range", Box::new(move |p| p.scale_slot[decision] = p.num_scale_slots)),
        ("scale slot on a chance node", Box::new(move |p| p.scale_slot[chance] = 0)),
        ("dealt card missing", Box::new(move |p| p.card[dealt] = NONE)),
        ("dealt card out of the deck", Box::new(move |p| p.card[dealt] = 52)),
        ("card dealt twice", Box::new(move |p| p.card[dealt + 1] = p.card[dealt])),
        ("card below a decision node", Box::new(move |p| p.card[below_decision] = 5)),
        ("card at the root", Box::new(|p| p.card[0] = 5)),
        ("zero chance factor", Box::new(move |p| p.chance_factor[chance] = 0.0)),
        ("non-finite chance factor", Box::new(move |p| p.chance_factor[chance] = f32::INFINITY)),
        ("scalar flag 2", Box::new(move |p| p.scalar[decision] = 2)),
        ("root batched", Box::new(|p| p.scalar[0] = 0)),
        (
            "chance child batched",
            Box::new(move |p| {
                for c in p.first_child[chance]..p.first_child[chance] + p.actions[chance] {
                    p.scalar[c as usize] = 0;
                }
            }),
        ),
        ("siblings with different rounding", Box::new(move |p| p.scalar[batched] = 1)),
        ("batched leaf with batching off", Box::new(|p| p.batch_river = false)),
        (
            "node array missing an entry",
            Box::new(|p| {
                p.winner.pop();
            }),
        ),
        ("no nodes", Box::new(|p| p.num_nodes = 0)),
    ];
    assert_all_rejected(&plan, mutations);
}

#[test]
fn validate_rejects_corrupted_schedules() {
    let cfg = spots().into_iter().find(|s| s.0 == "sampled turn").unwrap().1;
    let m = MultiSolver::new(cfg).unwrap();
    let plan = plan_for(&m);
    plan.validate().unwrap();
    // Player 1 traverses; its opponents are 0 and 2.
    let pp = &plan.players[1];
    let opp0 = pp.opp_nodes[0][1] as usize;
    let opp0_other = pp.opp_nodes[0][2] as usize;
    let opp1 = pp.opp_nodes[1][0] as usize;
    let chance = pp.chance_nodes[0] as usize;
    let own = pp.own_nodes[0] as usize;
    let child_of_opp0 = plan.first_child[opp0] as usize;
    // A decision node at which player 1 folded, and a node below it.
    let cut = *pp.fixed.iter().find(|&&f| plan.actor[f as usize] != NONE).unwrap() as usize;
    let below_cut = plan.first_child[cut] as usize;
    let updated = find(&plan, |i| pp.strategy_update[i] == 1);
    // Player 2's nodes are updated by player 0 unless player 0 folded.
    let not_updated = find(&plan, |i| pp.opp_nodes[1].contains(&(i as u32)) && pp.strategy_update[i] == 0);
    let actor_of_not_updated = plan.actor[not_updated] as usize;
    let mutations: Vec<(&str, Mutation)> = vec![
        (
            "slot out of range",
            Box::new(move |p| p.players[1].slot[0][child_of_opp0] = p.players[1].num_slots[0]),
        ),
        ("root not in slot 0", Box::new(|p| p.players[1].slot[1][0] = 1)),
        ("opponent child slot shifted", Box::new(move |p| p.players[1].child_slot[0][opp0] += 1)),
        ("second opponent's child slot shifted", Box::new(move |p| p.players[1].child_slot[1][opp1] += 1)),
        ("chance child slot shifted", Box::new(move |p| p.players[1].child_slot[1][chance] += 1)),
        (
            "chance node splits one opponent only",
            Box::new(move |p| p.players[1].child_slot[0][chance] = NONE),
        ),
        (
            "child slots at the root's slot",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.child_slot[0][opp0] = 0;
                for (a, c) in (p.first_child[opp0]..p.first_child[opp0] + p.actions[opp0]).enumerate() {
                    pp.slot[0][c as usize] = a as u32;
                }
            }),
        ),
        (
            "other opponent's slot not passed through",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.slot[1][child_of_opp0] = pp.num_slots[1] - 1;
            }),
        ),
        ("child slots at an own node", Box::new(move |p| p.players[1].child_slot[0][own] = 1)),
        (
            "slot handed out twice",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                let first = pp.child_slot[0][opp0].min(pp.child_slot[0][opp0_other]);
                for node in [opp0, opp0_other] {
                    pp.child_slot[0][node] = first;
                    for (a, c) in (p.first_child[node]..p.first_child[node] + p.actions[node]).enumerate() {
                        pp.slot[0][c as usize] = first + a as u32;
                    }
                }
            }),
        ),
        ("unreached node given a slot", Box::new(move |p| p.players[1].slot[0][below_cut] = 0)),
        ("cut-off node given child slots", Box::new(move |p| p.players[1].child_slot[0][cut] = 1)),
        ("reach buffer overflows u32 indexing", Box::new(|p| p.players[1].num_slots[0] = u32::MAX)),
        (
            "node scheduled twice",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.own_nodes.push(pp.own_nodes[0]);
                pp.levels.last_mut().unwrap().own.count += 1;
            }),
        ),
        (
            "cut-off node scheduled",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.own_nodes.push(cut as u32);
                pp.levels.last_mut().unwrap().own.count += 1;
            }),
        ),
        (
            "node in the wrong list",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                let a = pp.opp_nodes[0].iter().position(|&x| x as usize == opp0).unwrap();
                pp.opp_nodes[0][a] = own as u32;
            }),
        ),
        ("level not contiguous", Box::new(|p| p.players[1].levels[1].own.start += 1)),
        ("level past its list", Box::new(|p| p.players[1].levels.last_mut().unwrap().chance.count += 1)),
        (
            "empty level",
            Box::new(|p| {
                let pp = &mut p.players[1];
                let end = |v: &[u32]| Span { start: v.len() as u32, count: 0 };
                let lv = MultiLevel {
                    opp: [end(&pp.opp_nodes[0]), end(&pp.opp_nodes[1])],
                    own: end(&pp.own_nodes),
                    chance: end(&pp.chance_nodes),
                };
                pp.levels.push(lv);
            }),
        ),
        ("levels swapped", Box::new(|p| p.players[1].levels.swap(1, 2))),
        ("strategy update dropped", Box::new(move |p| p.players[1].strategy_update[updated] = 0)),
        (
            "strategy update in the wrong traversal",
            Box::new(move |p| p.players[1].strategy_update[not_updated] = 1),
        ),
        (
            "strategy update in the actor's own traversal",
            Box::new(move |p| p.players[actor_of_not_updated].strategy_update[not_updated] = 1),
        ),
        ("strategy update on a chance node", Box::new(move |p| p.players[1].strategy_update[chance] = 1)),
        ("strategy-update flag 2", Box::new(move |p| p.players[1].strategy_update[updated] = 2)),
        (
            "leaf missing",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.fixed.pop();
                pp.fixed_payoff.pop();
            }),
        ),
        (
            "leaf listed twice",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.fixed.push(pp.fixed[0]);
                pp.fixed_payoff.push(0.0);
            }),
        ),
        (
            "showdown listed as a fixed leaf",
            Box::new(|p| {
                let pp = &mut p.players[1];
                let s = pp.showdowns.remove(0);
                pp.showdown_table.remove(0);
                pp.win.remove(0);
                pp.lose.remove(0);
                pp.fixed.push(s);
                pp.fixed_payoff.push(0.0);
            }),
        ),
        (
            "fixed leaf listed as a showdown",
            Box::new(|p| {
                let pp = &mut p.players[1];
                let f = pp.fixed.remove(0);
                pp.fixed_payoff.remove(0);
                pp.showdowns.push(f);
                pp.showdown_table.push(0);
                pp.win.push(0.0);
                pp.lose.push(0.0);
            }),
        ),
        (
            "expanded node listed as a leaf",
            Box::new(move |p| {
                let pp = &mut p.players[1];
                pp.fixed[0] = own as u32;
            }),
        ),
        (
            "fixed payoffs missing",
            Box::new(|p| {
                p.players[1].fixed_payoff.pop();
            }),
        ),
        ("non-finite payoff", Box::new(|p| p.players[1].win[0] = f64::NAN)),
        ("payoff too large", Box::new(|p| p.players[2].fixed_payoff[0] = 1e13)),
        ("showdown table out of range", Box::new(|p| p.players[0].showdown_table[0] = p.num_tables)),
        (
            "slot arrays short",
            Box::new(|p| {
                p.players[0].child_slot[1].pop();
            }),
        ),
    ];
    assert_all_rejected(&plan, mutations);
}

#[test]
fn validate_rejects_corrupted_hand_data() {
    let cfg = spots().into_iter().find(|s| s.0 == "river").unwrap().1;
    let m = MultiSolver::new(cfg).unwrap();
    let plan = plan_for(&m);
    plan.validate().unwrap();
    let view = &plan.players[0].opponents[1];
    let first_ranked = view.tables.ranges.iter().position(|&r| r != NONE).unwrap();
    // A card whose list is non-empty, and a hand of that opponent without it.
    let c = (0..CARDS).find(|&c| view.card_off[c] < view.card_off[c + 1]).unwrap();
    let entry = view.card_off[c] as usize;
    let without = (0..view.hands as usize)
        .find(|&j| {
            let (a, b) = unpack_cards(view.cards[j]);
            a != c && b != c
        })
        .unwrap() as u32;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("bad hand", Box::new(|p| p.players[0].my_cards[0] = pack_cards(52, 1))),
        ("hand with a repeated card", Box::new(|p| p.players[0].my_cards[1] = pack_cards(7, 7))),
        ("hand counts disagree", Box::new(|p| p.players[1].hands += 1)),
        (
            "empty range",
            Box::new(|p| {
                p.players[1].hands = 0;
                p.players[1].my_cards.clear();
            }),
        ),
        ("node count disagrees with the arrays", Box::new(|p| p.num_nodes += 1)),
        ("wrong seat", Box::new(|p| p.players[0].opponents[0].player = 2)),
        ("seats swapped", Box::new(|p| p.players[2].opponents.swap(0, 1))),
        ("opponent's cards differ", Box::new(|p| p.players[0].opponents[1].cards.swap(0, 1))),
        (
            "identical combo out of range",
            Box::new(|p| {
                let v = &mut p.players[1].opponents[0];
                v.same[0] = v.hands;
            }),
        ),
        (
            "identical combos missing",
            Box::new(|p| {
                p.players[1].opponents[1].same.pop();
            }),
        ),
        ("negative weight", Box::new(|p| p.players[1].opponents[0].weights[0] = -1.0)),
        ("NaN weight", Box::new(|p| p.players[2].opponents[1].weights[0] = f32::NAN)),
        ("weight too large", Box::new(|p| p.players[0].opponents[0].weights[0] = 1e7)),
        (
            "weights missing",
            Box::new(|p| {
                p.players[0].opponents[0].weights.pop();
            }),
        ),
        (
            "card list entry out of range",
            Box::new(|p| {
                let v = &mut p.players[2].opponents[0];
                v.card_hands[0] = v.hands;
            }),
        ),
        (
            "card list entry without the card",
            Box::new(move |p| p.players[0].opponents[1].card_hands[entry] = without),
        ),
        ("card offsets out of order", Box::new(|p| p.players[0].opponents[0].card_off[1] = u32::MAX)),
        (
            "card offsets short",
            Box::new(|p| {
                p.players[0].opponents[0].card_off.pop();
            }),
        ),
        (
            "opponent order out of range",
            Box::new(|p| {
                let v = &mut p.players[0].opponents[1];
                v.tables.opp_order[0] = v.hands;
            }),
        ),
        ("card position out of range", Box::new(|p| p.players[1].opponents[1].tables.card_pos[0] = 60_000)),
        (
            "card rank past its list",
            Box::new(move |p| p.players[0].opponents[1].tables.ranks[first_ranked] = u32::MAX),
        ),
        (
            "strength range past the list",
            Box::new(move |p| p.players[0].opponents[1].tables.ranges[first_ranked] = 0xFFFE_0000),
        ),
        ("too much shared memory", Box::new(|p| p.players[2].opponents[1].tables.max_opp = 5000)),
        ("tables for another board count", Box::new(|p| p.num_tables += 1)),
    ];
    assert_all_rejected(&plan, mutations);
}

/// A random three-player spot: street (flop and turn with sampled
/// runouts), board, ranges, pot, stacks (some short, for all-in runouts),
/// a player folded at the root or not, bet menu, rake, terminal model, and
/// river batching. Some have an empty range after board removal; callers
/// skip those.
fn arb_multi_config() -> impl Strategy<Value = (MultiConfig, bool)> {
    let board = prop::sample::subsequence((0u8..52).collect::<Vec<u8>>(), 5).prop_shuffle();
    let street = prop_oneof![3 => Just(5usize), 3 => Just(4usize), 1 => Just(3usize)];
    let ranges = [(any::<u64>(), 0.02f64..0.2), (any::<u64>(), 0.02f64..0.2), (any::<u64>(), 0.02f64..0.2)];
    let stack = || prop_oneof![1 => 50 as Chips..400, 3 => 400 as Chips..6000];
    let stakes = (100 as Chips..2000, [stack(), stack(), stack()]);
    let folded = prop_oneof![3 => Just(None), 1 => (0usize..3).prop_map(Some)];
    let shape =
        (0usize..3, any::<bool>(), (1usize..3, 1usize..4, any::<u64>()), any::<bool>(), any::<bool>());
    (board, street, ranges, stakes, folded, shape).prop_map(
        |(board, len, ranges, (pot, stacks), folded, (m, raked, (turn, river, seed), exact, batch))| {
            let board: Vec<Card> = board.into_iter().take(len).map(Card::from_index).collect();
            let street = Street::from_board_len(len);
            // Thinner ranges on the flop keep the CPU side quick.
            let thin = if len == 3 { 0.3 } else { 1.0 };
            let ranges = ranges.map(|(s, d)| random_range(s, d * thin));
            let sizes = match m {
                0 => StreetSizes::new(&["66%"], &["3x"]),
                1 => StreetSizes::new(&["33%", "100%"], &[]),
                _ => StreetSizes::new(&["75%"], &["allin"]),
            };
            let menu = PostflopMenu {
                flop: sizes.clone(),
                turn: sizes.clone(),
                river: sizes,
                max_raises: 1 + (m % 2) as u32,
                ..PostflopMenu::default()
            };
            let mut fold = [false; N];
            if let Some(f) = folded {
                fold[f] = true;
            }
            let cfg = MultiConfig {
                board,
                ranges,
                root: MultiState::street_start(street, pot, stacks, fold, 100),
                menu,
                future_menu: None,
                big_blind: 100,
                chance: if len == 5 { ChanceMode::Exact } else { ChanceMode::Sampled { turn, river, seed } },
                prune: 0.0,
                keep: Default::default(),
                max_memory_bytes: 1 << 28,
                rake_rate: if raked { 0.05 } else { 0.0 },
                rake_cap: if raked { 300.0 } else { 0.0 },
                exact_removal: exact,
                storage: StorageMode::F32,
                dcfr: multiway_dcfr(),
            };
            (cfg, batch)
        },
    )
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
    /// marks, payoffs and rounding flags match the CPU traversal; with
    /// independent opponents the lookup tables reproduce its terminals.
    #[test]
    fn random_plans_validate_and_match_the_cpu((cfg, batch) in arb_multi_config(), seed in any::<u64>()) {
        let Ok(mut m) = MultiSolver::new(cfg) else { return Ok(()) };
        m.set_batch_river(batch);
        let plan = MultiPlan::build(&m.ctx()).unwrap();
        prop_assert!(plan.validate().is_ok());
        check_all(&m, &plan);
        if !plan.exact {
            check_independent_terminals(&m, &plan, seed);
        }
    }
}
