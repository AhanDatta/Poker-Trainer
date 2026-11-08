//! Split plans: the same traversal as the whole plan, segment by segment,
//! and every corruption of the segment data rejected.

use super::*;
use crate::gpu::segments::Scratch;

fn split_plan(m: &MultiSolver, max_nodes: u32) -> MultiPlan {
    MultiPlan::build_with(&m.ctx(), Scratch::ForceGroups { max_nodes }, 0).unwrap()
}

fn sorted_ids(v: &[u32]) -> Vec<u32> {
    let mut v = v.to_vec();
    v.sort_unstable();
    v
}

/// A split plan schedules the same nodes, in the same lists, with the same
/// leaves, strategy marks and payoffs as the whole plan; only levels, slots
/// and rows are laid out per segment.
fn check_same_traversal(whole: &MultiPlan, split: &MultiPlan, what: &str) {
    assert!(split.num_segments > 1, "{what}: not split");
    for (p, (w, s)) in whole.players.iter().zip(&split.players).enumerate() {
        for k in 0..2 {
            assert_eq!(sorted_ids(&w.opp_nodes[k]), sorted_ids(&s.opp_nodes[k]), "{what}: player {p}, opponent {k}");
        }
        assert_eq!(sorted_ids(&w.own_nodes), sorted_ids(&s.own_nodes), "{what}: player {p}: own nodes");
        assert_eq!(sorted_ids(&w.chance_nodes), sorted_ids(&s.chance_nodes), "{what}: player {p}: chance nodes");
        assert_eq!(w.strategy_update, s.strategy_update, "{what}: player {p}: strategy marks");
        let leaves = |pp: &MultiPlayerPlan| {
            let mut v: Vec<(u32, u64)> = pp.fixed.iter().zip(&pp.fixed_payoff).map(|(&n, x)| (n, x.to_bits())).collect();
            v.extend(pp.showdowns.iter().zip(&pp.win).map(|(&n, x)| (n, x.to_bits())));
            v.sort_unstable();
            v
        };
        assert_eq!(leaves(w), leaves(s), "{what}: player {p}: leaves");
        // Each node reached in both or neither.
        for i in 0..whole.num_nodes as usize {
            assert_eq!(w.slot[0][i] == NONE, s.slot[0][i] == NONE, "{what}: player {p}: node {i}'s reach");
        }
        assert!(s.num_slots[0] <= w.num_slots[0] && s.num_slots[1] <= w.num_slots[1], "{what}: more slots than whole");
    }
    assert!(split.num_rows <= whole.num_rows, "{what}: more rows than whole");
}

#[test]
fn split_plans_match_whole_plans() {
    for (name, cfg) in spots() {
        for exact in [false, true] {
            let mut cfg = cfg.clone();
            cfg.exact_removal = exact;
            let m = MultiSolver::new(cfg).unwrap();
            let whole = plan_for(&m);
            let roots_split = whole.num_nodes / 4;
            for max_nodes in [1, roots_split] {
                let Ok(split) = MultiPlan::build_with(&m.ctx(), Scratch::ForceGroups { max_nodes }, 0) else {
                    panic!("{name}: split plan failed")
                };
                if split.num_segments == 1 {
                    // No chance nodes on the root street (a river spot).
                    assert!(name.starts_with("river"), "{name}: not split");
                    continue;
                }
                check_same_traversal(&whole, &split, &format!("{name}, exact {exact}, {max_nodes}"));
            }
        }
    }
}

/// Every corruption of the segmentation and the per-segment schedule is
/// rejected.
#[test]
fn validate_rejects_corrupted_segments() {
    let mut cfg = spots().into_iter().find(|s| s.0 == "sampled turn").unwrap().1;
    cfg.exact_removal = true;
    let m = MultiSolver::new(cfg).unwrap();
    let good = split_plan(&m, 1);
    good.validate().unwrap();
    assert!(good.num_segments > 2);
    // A node of the second group, and the first group's first chunk root.
    let in_group = (0..good.num_nodes as usize).find(|&i| good.segment_of[i] == 2).unwrap();
    type Mutation = (&'static str, Box<dyn Fn(&mut MultiPlan)>);
    let mutations: Vec<Mutation> = vec![
        ("segment count", Box::new(|pl| pl.num_segments += 1)),
        ("node in another group", Box::new(move |pl| pl.segment_of[in_group] = 1)),
        ("root in a group", Box::new(|pl| pl.segment_of[0] = 1)),
        ("row out of range", Box::new(|pl| pl.row[1] = pl.num_rows)),
        ("top rows shared", Box::new(|pl| pl.row[1] = pl.row[0])),
        ("group node in the top rows", Box::new(move |pl| pl.row[in_group] = 0)),
        ("player's segment count", Box::new(|pl| {
            pl.players[0].segments.pop();
        })),
        ("level span", Box::new(|pl| pl.players[1].segments[1].levels.count += 1)),
        ("fixed span", Box::new(|pl| pl.players[2].segments[0].fixed.count += 1)),
        ("showdown span", Box::new(|pl| {
            let s = &mut pl.players[0].segments;
            s[0].showdowns.count += 1;
            s[1].showdowns.start += 1;
        })),
        ("top slots", Box::new(|pl| pl.players[0].top_slots[1] += 1)),
        ("group slots outside the region", Box::new(|pl| pl.players[1].segments[1].fresh_slots[0].start = 0)),
        ("group slots past the end", Box::new(|pl| {
            let pp = &mut pl.players[2];
            pp.num_slots[1] = pp.top_slots[1];
        })),
        ("group spans", Box::new(|pl| pl.players[0].segments[1].groups.start += 1)),
        ("job spans", Box::new(|pl| {
            let s = &mut pl.players[1].segments;
            s[0].jobs.count += 1;
            s[1].jobs.start += 1;
        })),
    ];
    for (what, mutate) in mutations {
        let mut plan = good.clone();
        mutate(&mut plan);
        assert!(plan.validate().is_err(), "{what} was accepted");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Random spots split into one chunk per group: valid plans with the
    /// whole plan's traversal.
    #[test]
    fn random_split_plans_match_whole_plans((cfg, batch) in arb_multi_config(), max_nodes in 1u32..200) {
        let Ok(mut m) = MultiSolver::new(cfg) else { return Ok(()) };
        m.set_batch_river(batch);
        let whole = plan_for(&m);
        let split = MultiPlan::build_with(&m.ctx(), Scratch::ForceGroups { max_nodes }, 0).unwrap();
        prop_assert!(split.validate().is_ok());
        if split.num_segments > 1 {
            check_same_traversal(&whole, &split, "random spot");
        }
    }
}
