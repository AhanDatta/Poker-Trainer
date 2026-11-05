//! Plan construction and validation (no device needed).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::eval::{eval_fold, eval_showdown};
use crate::gpu::testutil::{
    allin_turn_config, arb_river_config, arb_turn_config, iso_turn_config, mid_street_config, river_config,
    sampled_flop_config, turn_config,
};
use crate::{SolveConfig, Solver};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn plan_for(cfg: SolveConfig) -> (Solver, Plan) {
    let s = Solver::new(cfg).unwrap();
    let plan = Plan::build(&s.ctx()).unwrap();
    (s, plan)
}

/// Showdown values from the plan's lookup tables, computed in f64 with the
/// kernel's formula (`showdown_values` in terminal.cu).
fn table_showdown(pp: &PlayerPlan, t: usize, reach: &[f32], win: f64, lose: f64, tie: f64) -> Vec<f32> {
    let tb = &pp.tables;
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
        let seg = &tb.card_pos[base + card_off[c] as usize..base + card_off[c + 1] as usize];
        let mut v = vec![0.0];
        for &j in seg {
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
            let stronger = (prefix[len] - prefix[hi]) - (f[f.len() - 1] - f[r1h]) - (s[s.len() - 1] - s[r2h]);
            let mut v = (win * weaker + lose * stronger) as f32;
            if tie != 0.0 {
                let same = match pp.same[h] {
                    NONE => 0.0,
                    j => f64::from(reach[j as usize]),
                };
                let eq = (prefix[hi] - prefix[lo]) - ((f[r1h] - f[r1l]) + (s[r2h] - s[r2l])) + same;
                v += (tie * eq) as f32;
            }
            v
        })
        .collect()
}

/// Fold values from the plan's per-card opponent lists (`fold_values`).
fn plan_fold(pp: &PlayerPlan, reach: &[f32], payoff: f64) -> Vec<f32> {
    let total: f64 = reach.iter().map(|&x| f64::from(x)).sum();
    let minus: Vec<f64> = (0..CARDS)
        .map(|c| {
            pp.opp_card_hands[pp.opp_card_off[c] as usize..pp.opp_card_off[c + 1] as usize]
                .iter()
                .map(|&j| f64::from(reach[j as usize]))
                .sum()
        })
        .collect();
    (0..pp.hands as usize)
        .map(|h| {
            let (c1, c2) = unpack_cards(pp.my_cards[h]);
            let same = match pp.same[h] {
                NONE => 0.0,
                j => f64::from(reach[j as usize]),
            };
            (payoff * (total - minus[c1] - minus[c2] + same)) as f32
        })
        .collect()
}

/// Random opponent reach with about a quarter of the hands at zero.
fn random_reach(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| if rng.random::<f64>() < 0.25 { 0.0 } else { rng.random_range(0.0f32..1.0) }).collect()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 2e-6 * scale, "{what}: hand {i}: {x} vs {y} (scale {scale})");
    }
}

#[test]
fn river_plans_validate() {
    for cfg in [river_config(), mid_street_config(false), mid_street_config(true)] {
        let (s, plan) = plan_for(cfg);
        assert_eq!(plan.num_nodes as usize, s.tree.nodes.len());
        for pp in &plan.players {
            assert_eq!(pp.opp_nodes.len() + pp.own_nodes.len(), s.tree.num_player_nodes as usize);
            assert!(!pp.showdowns.is_empty() && !pp.folds.is_empty());
        }
    }
}

/// Turn and flop trees: every chance node scheduled, its children dealt a
/// card, isomorphic cards present exactly when the spot has a symmetry.
#[test]
fn chance_plans_validate() {
    for (name, cfg, iso) in [
        ("turn", turn_config(), false),
        ("isomorphic turn", iso_turn_config(), true),
        ("all-in turn", allin_turn_config(), false),
        // Sampled chance nodes don't use isomorphism, but all-in runouts are
        // enumerated exactly and clubs/hearts are interchangeable on Ks7s2d.
        ("sampled flop", sampled_flop_config(), true),
    ] {
        let (s, plan) = plan_for(cfg);
        let chance_nodes =
            s.tree.nodes.iter().filter(|n| matches!(n.kind, crate::tree::Kind::Chance)).count();
        assert!(chance_nodes > 0, "{name}: no chance nodes");
        for pp in &plan.players {
            assert_eq!(pp.chance_nodes.len(), chance_nodes, "{name}");
        }
        assert_eq!(!plan.iso_child.is_empty(), iso, "{name}: isomorphic cards");
        assert!(plan.num_tables > 1, "{name}: one strength table per final board");
    }
}

/// The all-in turn has chance nodes whose children are showdowns.
#[test]
fn chance_children_can_be_terminals() {
    let (_, plan) = plan_for(allin_turn_config());
    let terminal_children = (0..plan.num_nodes as usize)
        .filter(|&i| plan.actor[i] == CHANCE)
        .flat_map(|i| plan.children(i).unwrap())
        .filter(|&c| plan.actor[c] == NONE)
        .count();
    assert!(terminal_children > 0);
}

#[test]
fn validate_rejects_corrupted_plans() {
    let (_, plan) = plan_for(river_config());
    plan.validate().unwrap();
    let first_ranked = plan.players[0].tables.ranges.iter().position(|&r| r != NONE).unwrap();
    let opp_node = plan.players[0].opp_nodes[0] as usize;
    type Mutation = Box<dyn Fn(&mut Plan)>;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("slot out of range", Box::new(|p| p.players[0].slot[3] = p.players[0].num_slots)),
        ("storage block out of range", Box::new(|p| p.offset[0] = p.storage_len)),
        ("children out of range", Box::new(|p| p.first_child[0] = p.num_nodes)),
        ("root actor flipped", Box::new(|p| p.actor[0] ^= 1)),
        (
            "node scheduled twice",
            Box::new(|p| {
                let pp = &mut p.players[1];
                pp.opp_nodes.push(pp.opp_nodes[0]);
                pp.levels.last_mut().unwrap().opp_count += 1;
            }),
        ),
        (
            "terminal missing",
            Box::new(|p| {
                p.players[0].folds.pop();
                p.players[0].fold_payoff.pop();
            }),
        ),
        ("child slot shifted", Box::new(move |p| p.players[0].child_slot[opp_node] += 1)),
        ("card rank past its list", Box::new(move |p| p.players[0].tables.ranks[first_ranked] = u32::MAX)),
        (
            "strength range past the list",
            Box::new(move |p| p.players[0].tables.ranges[first_ranked] = 0xFFFE_0000),
        ),
        (
            "opponent order out of range",
            Box::new(|p| p.players[0].tables.opp_order[0] = p.players[0].opp_hands),
        ),
        ("card position out of range", Box::new(|p| p.players[1].tables.card_pos[0] = 60_000)),
        ("card out of the deck", Box::new(|p| p.players[0].my_cards[0] = pack_cards(52, 1))),
        ("identical combo out of range", Box::new(|p| p.players[0].same[0] = p.players[0].opp_hands)),
        (
            "card list entry out of range",
            Box::new(|p| p.players[1].opp_card_hands[0] = p.players[1].opp_hands),
        ),
        ("too much shared memory", Box::new(|p| p.players[0].tables.max_opp = 5000)),
        ("non-finite payoff", Box::new(|p| p.players[0].win[0] = f64::NAN)),
        ("payoff too large for double-single", Box::new(|p| p.players[1].lose[0] = 1e13)),
        ("weight too large for double-single", Box::new(|p| p.players[0].opp_weights[0] = 1e7)),
        ("negative weight", Box::new(|p| p.players[1].opp_weights[0] = -1.0)),
        ("showdown table out of range", Box::new(|p| p.players[0].showdown_table[0] = p.num_tables)),
    ];
    for (what, mutate) in mutations {
        let mut bad = plan.clone();
        mutate(&mut bad);
        assert!(matches!(bad.validate(), Err(GpuError::InvalidInput(_))), "{what}: accepted");
    }

    // Chance-node fields, on an isomorphic turn tree.
    let (_, plan) = plan_for(iso_turn_config());
    plan.validate().unwrap();
    let chance = (0..plan.num_nodes as usize).find(|&i| plan.actor[i] == CHANCE).unwrap();
    let dealt = plan.first_child[chance] as usize;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("dealt card missing", Box::new(move |p| p.card[dealt] = NONE)),
        ("dealt card out of the deck", Box::new(move |p| p.card[dealt] = 52)),
        ("zero chance factor", Box::new(move |p| p.chance_factor[chance] = 0.0)),
        ("non-finite chance factor", Box::new(move |p| p.chance_factor[chance] = f32::INFINITY)),
        ("isomorphic card's child out of range", Box::new(|p| p.iso_child[0] = 60)),
        ("bad suit pair", Box::new(|p| p.iso_pair[0] = 6)),
        (
            "isomorphic range past the list",
            Box::new(move |p| p.iso_count[chance] = p.iso_child.len() as u32 + 1),
        ),
        ("isomorphic cards on a decision node", Box::new(|p| p.iso_count[0] = 1)),
        (
            "permutation out of range",
            Box::new(|p| {
                let row = p.iso_pair[0] as usize * p.players[1].hands as usize;
                p.players[1].perm[row] = p.players[1].hands;
            }),
        ),
        ("chance child slot shifted", Box::new(move |p| p.players[0].child_slot[chance] += 1)),
        ("chance node in the opponent list", Box::new(move |p| p.actor[chance] = 1)),
        ("opponent card out of the deck", Box::new(|p| p.players[0].opp_cards[0] = pack_cards(1, 60))),
    ];
    for (what, mutate) in mutations {
        let mut bad = plan.clone();
        mutate(&mut bad);
        assert!(matches!(bad.validate(), Err(GpuError::InvalidInput(_))), "{what}: accepted");
    }
}

/// Guards against vacuous property tests: most generated spots must be
/// solvable (the rest have an empty range after board removal).
#[test]
fn generated_spots_are_mostly_solvable() {
    use proptest::strategy::ValueTree;
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let strategy = arb_river_config();
    let total = 64;
    let ok =
        (0..total).filter(|_| Solver::new(strategy.new_tree(&mut runner).unwrap().current()).is_ok()).count();
    assert!(ok * 4 >= total * 3, "only {ok} of {total} generated spots are solvable");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// The precomputed lookup reproduces the CPU's showdown and fold
    /// evaluation for random spots and random reach.
    #[test]
    fn lookup_tables_reproduce_cpu_terminals(cfg in arb_river_config(), seed in any::<u64>()) {
        let Ok(s) = Solver::new(cfg) else { return Ok(()) };
        let plan = Plan::build(&s.ctx()).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        for p in 0..2 {
            let pp = &plan.players[p];
            let reach = random_reach(&mut rng, pp.opp_hands as usize);
            for (win, lose, tie) in [(1.0, -1.0, 0.0), (0.7, -1.3, 0.25)] {
                let mut cpu = vec![0f32; pp.hands as usize];
                eval_showdown(&s.hands, &s.tables[0], p, &reach, win, lose, tie, &mut cpu);
                assert_close(&table_showdown(pp, 0, &reach, win, lose, tie), &cpu, "showdown");
            }
            let mut cpu = vec![0f32; pp.hands as usize];
            eval_fold(&s.hands, p, &reach, 0.37, &mut cpu);
            assert_close(&plan_fold(pp, &reach, 0.37), &cpu, "fold");
        }
    }

    #[test]
    fn random_turn_plans_validate(cfg in arb_turn_config()) {
        let Ok(s) = Solver::new(cfg) else { return Ok(()) };
        let plan = Plan::build(&s.ctx()).unwrap();
        prop_assert!(plan.validate().is_ok());
    }

    #[test]
    fn random_river_plans_validate(cfg in arb_river_config()) {
        let Ok(s) = Solver::new(cfg) else { return Ok(()) };
        let plan = Plan::build(&s.ctx()).unwrap();
        prop_assert!(plan.validate().is_ok());
    }
}

/// Terminals follow the CPU evaluator that handles them: batched inside
/// river subtrees, one at a time before the river and right after an all-in
/// runout, and one at a time everywhere when batching is off.
#[test]
fn scalar_flags_follow_the_cpu_evaluators() {
    use crate::tree::Kind;
    let (_, plan) = plan_for(river_config());
    assert!(plan.scalar.iter().all(|&x| x == 0), "river root: everything is batched");

    let (s, plan) = plan_for(allin_turn_config());
    let mut seen = [0usize; 3];
    for (id, node) in s.tree.nodes.iter().enumerate() {
        let scalar = plan.scalar[id] == 1;
        match node.kind {
            Kind::Fold(_) if node.street == holdem::Street::Turn => {
                assert!(scalar, "turn fold {id}");
                seen[0] += 1;
            }
            Kind::Showdown => {
                // After an all-in call the chance node's child is the
                // showdown itself; otherwise a river decision node comes first.
                let parent_is_chance = (0..s.tree.nodes.len()).any(|q| {
                    matches!(s.tree.nodes[q].kind, Kind::Chance) && s.tree.nodes[q].children().contains(&id)
                });
                assert_eq!(scalar, parent_is_chance, "showdown {id}");
                seen[if parent_is_chance { 1 } else { 2 }] += 1;
            }
            Kind::Fold(_) => assert!(!scalar, "river fold {id}"),
            _ => assert!(!scalar, "non-terminal {id}"),
        }
    }
    assert!(seen.iter().all(|&k| k > 0), "spot lacks a terminal kind: {seen:?}");

    let mut s = Solver::new(turn_config()).unwrap();
    s.set_batch_river(false);
    let plan = Plan::build(&s.ctx()).unwrap();
    for (id, node) in s.tree.nodes.iter().enumerate() {
        let terminal = matches!(node.kind, Kind::Fold(_) | Kind::Showdown);
        assert_eq!(plan.scalar[id] == 1, terminal, "batching off, node {id}");
    }
}

/// Folds are paired with the showdown that shares their reach exactly where
/// the CPU groups them (both children of one of the traverser's nodes, both
/// batched), and leave the fold list.
#[test]
fn folds_pair_with_their_showdowns() {
    use crate::tree::Kind;
    for cfg in [river_config(), turn_config(), allin_turn_config()] {
        let (s, plan) = plan_for(cfg);
        for (p, pp) in plan.players.iter().enumerate() {
            let mut expected = 0;
            for node in &s.tree.nodes {
                let Kind::Player(q) = node.kind else { continue };
                if q as usize != p {
                    continue;
                }
                let kind = |c: usize| s.tree.nodes[c].kind;
                let fold = node.children().find(|&c| matches!(kind(c), Kind::Fold(_)));
                let showdown = node.children().find(|&c| matches!(kind(c), Kind::Showdown));
                if let (Some(f), Some(sd)) = (fold, showdown) {
                    if plan.scalar[f] == 0 && plan.scalar[sd] == 0 {
                        expected += 1;
                        let k = pp.showdowns.iter().position(|&x| x as usize == sd).unwrap();
                        assert_eq!(pp.showdown_fold[k] as usize, f);
                        assert!(!pp.folds.contains(&(f as u32)), "paired fold {f} still listed");
                    }
                }
            }
            let paired = pp.showdown_fold.iter().filter(|&&f| f != NONE).count();
            assert_eq!(paired, expected, "player {p}");
            assert!(paired > 0, "player {p}: spot has no pairs");
        }
    }
}

#[test]
fn validate_rejects_bad_pairings() {
    let (_, plan) = plan_for(river_config());
    let k = plan.players[0].showdown_fold.iter().position(|&f| f != NONE).unwrap();
    let paired = plan.players[0].showdown_fold[k];
    let unpaired = plan.players[0].folds[0];
    let mut listed_twice = plan.clone();
    listed_twice.players[0].folds.push(paired);
    listed_twice.players[0].fold_payoff.push(0.0);
    assert!(matches!(listed_twice.validate(), Err(GpuError::InvalidInput(_))), "fold listed twice accepted");
    // Swap a paired and an unpaired fold: each terminal is still listed once,
    // but the newly paired fold has another opponent reach.
    let mut other_reach = plan.clone();
    let pp = &mut other_reach.players[0];
    assert_ne!(pp.slot[unpaired as usize], pp.slot[paired as usize]);
    pp.showdown_fold[k] = unpaired;
    pp.folds[0] = paired;
    match other_reach.validate() {
        Err(GpuError::InvalidInput(msg)) => assert!(msg.contains("another reach"), "{msg}"),
        other => panic!("mispaired fold accepted: {other:?}"),
    }
    let mut bad_payoff = plan;
    bad_payoff.players[0].showdown_fold_payoff[k] = f64::INFINITY;
    assert!(matches!(bad_payoff.validate(), Err(GpuError::InvalidInput(_))), "infinite payoff accepted");
}

/// Spots whose root street has chance nodes, so their trees can be split.
fn splittable_spots() -> Vec<(&'static str, SolveConfig)> {
    vec![
        ("turn", turn_config()),
        ("isomorphic turn", iso_turn_config()),
        ("all-in turn", allin_turn_config()),
        ("sampled flop", sampled_flop_config()),
    ]
}

fn split_plan(cfg: SolveConfig, scratch: Scratch) -> (Solver, Plan) {
    let s = Solver::new(cfg).unwrap();
    let plan = Plan::build_with(&s.ctx(), scratch).unwrap();
    (s, plan)
}

/// Bytes of the scratch buffers `Engine::new` allocates for `plan`.
fn plan_scratch(plan: &Plan) -> u64 {
    let [p0, p1] = &plan.players;
    scratch_bytes(
        [u64::from(p0.num_slots), u64::from(p1.num_slots)],
        u64::from(plan.num_rows),
        [u64::from(p0.hands), u64::from(p1.hands)],
    )
}

/// Chunks of a tree: children of the root street's chance nodes.
fn chunk_count(s: &Solver) -> usize {
    let street = s.tree.nodes[0].street;
    s.tree.nodes.iter().filter(|n| matches!(n.kind, Kind::Chance) && n.street == street).map(|n| n.num_children as usize).sum()
}

fn sorted(v: &[u32]) -> Vec<u32> {
    let mut v = v.to_vec();
    v.sort_unstable();
    v
}

/// Forced splitting gives one group per chunk (or groups of several),
/// reuses value rows and reach slots, and schedules exactly the nodes and
/// terminals of the whole plan.
#[test]
fn forced_groups_split_trees_with_chunks() {
    for (name, cfg) in splittable_spots() {
        let (s, whole) = split_plan(cfg.clone(), Scratch::Unlimited);
        assert_eq!((whole.num_segments, whole.num_rows), (1, whole.num_nodes), "{name}: whole plan");
        let chunks = chunk_count(&s);
        assert!(chunks > 1, "{name}: no chunks");
        for max_nodes in [1, whole.num_nodes / 4] {
            let (_, plan) = split_plan(cfg.clone(), Scratch::ForceGroups { max_nodes });
            plan.validate().unwrap();
            let groups = plan.num_segments as usize - 1;
            if max_nodes == 1 {
                assert_eq!(groups, chunks, "{name}: one group per chunk");
            } else {
                assert!(groups > 1 && groups < chunks, "{name}: {groups} groups of {chunks} chunks");
            }
            assert!(plan.num_rows < plan.num_nodes, "{name}: value rows aren't reused");
            assert!(plan_scratch(&plan) < plan_scratch(&whole), "{name}: no scratch saved");
            for (p, (pp, wp)) in plan.players.iter().zip(&whole.players).enumerate() {
                assert!(pp.num_slots < wp.num_slots, "{name}, player {p}: reach slots aren't reused");
                assert_eq!(pp.segments.len(), groups + 1);
                for (a, b, what) in [
                    (&pp.own_nodes, &wp.own_nodes, "own nodes"),
                    (&pp.opp_nodes, &wp.opp_nodes, "opponent nodes"),
                    (&pp.chance_nodes, &wp.chance_nodes, "chance nodes"),
                    (&pp.folds, &wp.folds, "folds"),
                    (&pp.showdowns, &wp.showdowns, "showdowns"),
                    (&pp.showdown_fold, &wp.showdown_fold, "paired folds"),
                ] {
                    assert_eq!(sorted(a), sorted(b), "{name}, player {p}: {what}");
                }
            }
        }
    }
}

/// Without chance nodes on the root street there is nothing to split.
#[test]
fn trees_without_chunks_stay_whole() {
    for cfg in [river_config(), mid_street_config(true)] {
        let (s, plan) = split_plan(cfg, Scratch::ForceGroups { max_nodes: 1 });
        assert_eq!(chunk_count(&s), 0);
        assert_eq!((plan.num_segments, plan.num_rows), (1, plan.num_nodes));
    }
}

/// A budget splits the tree only when the whole would not fit, into groups
/// that do; a budget too small for one chunk is an out-of-memory error.
/// (On these small trees the plan arrays and the top segment are much of the
/// total, so only budgets a little below the whole tree's need can fit.)
#[test]
fn budget_splits_only_as_needed() {
    for (name, cfg) in splittable_spots() {
        let (s, whole) = split_plan(cfg.clone(), Scratch::Unlimited);
        let need = whole.device_bytes() + plan_scratch(&whole);
        let (_, roomy) = split_plan(cfg.clone(), Scratch::Budget(2 * need));
        assert_eq!(roomy.num_segments, 1, "{name}: split with room to spare");
        let mut fitted = 0;
        for budget in [need - 1, need * 19 / 20, need * 9 / 10, need * 3 / 4, need / 2] {
            match Plan::build_with(&s.ctx(), Scratch::Budget(budget)) {
                Ok(plan) => {
                    assert!(plan.num_segments > 1, "{name}: not split under {budget} of {need} bytes");
                    let used = plan.device_bytes() + plan_scratch(&plan);
                    assert!(used <= budget, "{name}: {used} bytes over the budget of {budget}");
                    fitted += 1;
                }
                Err(GpuError::OutOfMemory { needed, available }) => {
                    assert!(needed > budget && available == budget, "{name}: {needed} vs {available}");
                }
                Err(e) => panic!("{name}: {e}"),
            }
        }
        assert!(fitted >= 2, "{name}: only {fitted} budgets fitted");
        let tiny = Scratch::Budget(whole.device_bytes());
        assert!(matches!(Plan::build_with(&s.ctx(), tiny), Err(GpuError::OutOfMemory { .. })), "{name}");
    }
}

#[test]
fn validate_rejects_corrupted_split_plans() {
    let (_, plan) = split_plan(turn_config(), Scratch::ForceGroups { max_nodes: 1 });
    plan.validate().unwrap();
    assert!(plan.num_segments > 3);
    let nn = plan.num_nodes as usize;
    let parent = |c: usize| (0..nn).find(|&i| plan.children(i).unwrap().contains(&c)).unwrap();
    let segment_of = &plan.segment_of;
    let group = |g: u32| (0..nn).filter(move |&i| segment_of[i] == g);
    let root1 = group(1).find(|&i| plan.segment_of[parent(i)] == 0).unwrap();
    let mut inner = group(1).filter(|&i| i != root1);
    let (inner1, inner1b) = (inner.next().unwrap(), inner.next().unwrap());
    let opp = plan.players[0].opp_nodes.iter().map(|&i| i as usize).find(|&i| plan.segment_of[i] == 1).unwrap();
    type Mutation = Box<dyn Fn(&mut Plan)>;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("group row in the top region", Box::new(move |p| p.row[inner1] = p.top_rows - 1)),
        ("group rows shared", Box::new(move |p| p.row[inner1b] = p.row[inner1])),
        ("top rows shared", Box::new(move |p| p.row[root1] = p.row[0])),
        ("row out of range", Box::new(move |p| p.row[inner1] = p.num_rows)),
        ("node moved to another group", Box::new(move |p| p.segment_of[inner1] = 2)),
        ("chunk root moved to the top", Box::new(move |p| p.segment_of[root1] = 0)),
        ("segment out of range", Box::new(move |p| p.segment_of[inner1] = p.num_segments)),
        ("root in a group", Box::new(|p| p.segment_of[0] = 1)),
        (
            "group child slots in the top region",
            Box::new(move |p| {
                let pp = &mut p.players[0];
                let first = pp.top_slots - 1;
                pp.child_slot[opp] = first;
                let children = p.first_child[opp] as usize..(p.first_child[opp] + p.actions[opp]) as usize;
                for (a, c) in children.enumerate() {
                    pp.slot[c] = first + a as u32;
                }
            }),
        ),
        ("group slot region moved", Box::new(|p| p.players[1].segments[1].fresh_slots.start += 1)),
        ("group slots past the end", Box::new(|p| p.players[1].segments[2].fresh_slots.count = p.players[1].num_slots)),
        (
            "top slot region cut short",
            Box::new(|p| {
                let pp = &mut p.players[0];
                pp.top_slots -= 1;
                pp.segments[0].fresh_slots.count -= 1;
                for s in &mut pp.segments[1..] {
                    s.fresh_slots.start -= 1;
                }
            }),
        ),
        (
            "segment levels overlap",
            Box::new(|p| {
                let s = &mut p.players[0].segments[1].levels;
                s.start -= 1;
                s.count += 1;
            }),
        ),
        (
            "level in the wrong segment",
            Box::new(|p| {
                let s = &mut p.players[1].segments;
                s[0].levels.count += 1;
                s[1].levels.start += 1;
                s[1].levels.count -= 1;
            }),
        ),
        (
            "showdown in the wrong segment",
            Box::new(|p| {
                let s = &mut p.players[0].segments;
                s[1].showdowns.count += 1;
                s[2].showdowns.start += 1;
                s[2].showdowns.count -= 1;
            }),
        ),
        ("schedule missing a segment", Box::new(|p| {
            p.players[1].segments.pop();
        })),
        ("value buffer too small", Box::new(|p| p.num_rows = p.top_rows)),
    ];
    for (what, mutate) in mutations {
        let mut bad = plan.clone();
        mutate(&mut bad);
        assert!(matches!(bad.validate(), Err(GpuError::InvalidInput(_))), "{what}: accepted");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Any group size gives a valid plan, split exactly when there are chunks.
    #[test]
    fn random_split_plans_validate(cfg in arb_turn_config(), max_nodes in 1u32..3000) {
        let Ok(s) = Solver::new(cfg) else { return Ok(()) };
        let plan = Plan::build_with(&s.ctx(), Scratch::ForceGroups { max_nodes }).unwrap();
        prop_assert!(plan.validate().is_ok());
        prop_assert_eq!(plan.num_segments > 1, chunk_count(&s) > 0);
    }
}
