//! Plan construction and validation (no device needed).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use super::*;
use crate::eval::{eval_fold, eval_showdown};
use crate::gpu::testutil::{arb_river_config, mid_street_config, river_config, turn_config};
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

#[test]
fn trees_with_chance_nodes_are_declined() {
    let s = Solver::new(turn_config()).unwrap();
    assert!(matches!(Plan::build(&s.ctx()), Err(GpuError::InvalidInput(_))));
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
        ("negative weight", Box::new(|p| p.players[1].opp_weights[0] = -1.0)),
        ("showdown table out of range", Box::new(|p| p.players[0].showdown_table[0] = p.num_tables)),
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
    fn random_river_plans_validate(cfg in arb_river_config()) {
        let Ok(s) = Solver::new(cfg) else { return Ok(()) };
        let plan = Plan::build(&s.ctx()).unwrap();
        prop_assert!(plan.validate().is_ok());
    }
}
