//! Property-based tests: the fast evaluators against brute force, and
//! solver-wide invariants on random spots.

use crate::eval::{eval_fold, eval_showdown, fold_batch, showdown_batch, LANES};
use crate::hands::{HandSet, StrengthTable};
use crate::*;
use holdem::combo::{combo_cards, HandClass, NUM_CLASSES, NUM_COMBOS};
use holdem::eval::EvalMask;
use holdem::{Card, CardSet, PostflopMenu, Range, Street, StreetSizes};
use proptest::prelude::*;

fn cards(n: usize) -> impl Strategy<Value = Vec<Card>> {
    prop::sample::subsequence((0u8..52).collect::<Vec<_>>(), n)
        .prop_shuffle()
        .prop_map(|v| v.into_iter().map(Card::from_index).collect())
}

/// A random range made of a few hand classes with random weights.
fn range() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::btree_map(0usize..NUM_CLASSES, 1u8..=4, 3..14).prop_map(|m| {
        let mut r = Range::empty();
        for (g, w) in m {
            r.set_class(HandClass::from_grid_index(g), w as f32 / 4.0);
        }
        r.into_weights()
    })
}

fn hands_for(board: &[Card], r0: &[f32], r1: &[f32]) -> Option<HandSet> {
    let hs = HandSet::new(&[r0.to_vec(), r1.to_vec()], CardSet::from_cards(board), 0.0, &[vec![], vec![]]);
    (!hs.is_empty(0) && !hs.is_empty(1) && hs.num_combinations > 0.0).then_some(hs)
}

fn conflict(a: (u8, u8), b: (u8, u8)) -> bool {
    a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1
}

fn strength(board: &[Card], h: (u8, u8)) -> u32 {
    EvalMask::from_cards(board).add(Card::from_index(h.0)).add(Card::from_index(h.1)).evaluate()
}

/// O(n²) showdown values.
fn brute_showdown(hs: &HandSet, board: &[Card], p: usize, r: &[f32], w: f64, l: f64, t: f64) -> Vec<f64> {
    let o = p ^ 1;
    let bset = CardSet::from_cards(board);
    let blocked = |h: (u8, u8)| bset.contains(Card::from_index(h.0)) || bset.contains(Card::from_index(h.1));
    hs.cards[p]
        .iter()
        .map(|&h| {
            if blocked(h) {
                return 0.0;
            }
            let sh = strength(board, h);
            let mut v = 0.0;
            for (j, &oh) in hs.cards[o].iter().enumerate() {
                if blocked(oh) || conflict(h, oh) {
                    continue;
                }
                let so = strength(board, oh);
                v += r[j] as f64
                    * if sh > so {
                        w
                    } else if sh < so {
                        l
                    } else {
                        t
                    };
            }
            v
        })
        .collect()
}

fn reach_vec(n: usize, seed: u64) -> Vec<f32> {
    // Deterministic pseudo-random reach in [0, 1] with some exact zeros.
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((x >> 33) % 1000) as f32 / 1000.0;
            if v < 0.2 {
                0.0
            } else {
                v
            }
        })
        .collect()
}

fn close(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= 1e-4 * (1.0 + scale)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Inclusion–exclusion card removal equals direct summation.
    #[test]
    fn compatible_reach_matches_brute_force(board in cards(5), r0 in range(), r1 in range(), seed in any::<u64>()) {
        let Some(hs) = hands_for(&board, &r0, &r1) else { return Ok(()) };
        let r = reach_vec(hs.len(1), seed);
        let fast = hs.compatible_reach(0, &r);
        for (i, &h) in hs.cards[0].iter().enumerate() {
            let brute: f64 = hs.cards[1].iter().enumerate()
                .filter(|(_, &o)| !conflict(h, o))
                .map(|(j, _)| r[j] as f64)
                .sum();
            prop_assert!(close(fast[i], brute, brute), "{} vs {}", fast[i], brute);
        }
    }

    /// Strength-sorted sweeps equal the O(n²) showdown, ties included.
    #[test]
    fn showdown_matches_brute_force(
        board in cards(5), r0 in range(), r1 in range(), seed in any::<u64>(),
        p in 0usize..2, tie in prop_oneof![Just(0.0f64), -50.0f64..50.0],
    ) {
        let Some(hs) = hands_for(&board, &r0, &r1) else { return Ok(()) };
        let table = StrengthTable::new(&hs, &board);
        let r = reach_vec(hs.len(p ^ 1), seed);
        let (w, l) = (150.0, -100.0);
        let mut out = vec![0f32; hs.len(p)];
        eval_showdown(&hs, &table, p, &r, w, l, tie, &mut out);
        let brute = brute_showdown(&hs, &board, p, &r, w, l, tie);
        // Values are sums of terms up to `mag`; a result near zero can still
        // carry f32 rounding of those terms.
        let mag = (w.abs() + l.abs() + tie.abs()) * r.iter().map(|&x| x as f64).sum::<f64>();
        for (a, b) in out.iter().zip(&brute) {
            prop_assert!(close(*a as f64, *b, b.abs() * 10.0 + mag * 0.01), "{} vs {}", a, b);
        }
        // The batched evaluator agrees with the scalar one.
        let k = 1 + (seed % LANES as u64) as usize;
        let reaches: Vec<Vec<f32>> = (0..k).map(|t| reach_vec(hs.len(p ^ 1), seed ^ t as u64)).collect();
        let refs: Vec<&[f32]> = reaches.iter().map(|v| v.as_slice()).collect();
        let wins: Vec<f64> = (0..k).map(|t| 100.0 + t as f64).collect();
        let loses: Vec<f64> = (0..k).map(|t| -80.0 - t as f64).collect();
        let ties: Vec<f64> = vec![tie; k];
        let mut bufs: Vec<Vec<f32>> = vec![vec![0.0; hs.len(p)]; k];
        let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
        // Every other lane also has a fold terminal with the same reach.
        let mut fold_bufs: Vec<Vec<f32>> = vec![vec![0.0; hs.len(p)]; k];
        let mut folds: Vec<Option<(f64, &mut [f32])>> = fold_bufs
            .iter_mut()
            .enumerate()
            .map(|(t, v)| (t % 2 == 0).then(|| (-2.5 - t as f64, v.as_mut_slice())))
            .collect();
        showdown_batch(&hs, &table, p, &refs, &wins, &loses, &ties, &mut outs, &mut folds);
        for t in 0..k {
            let mut single = vec![0f32; hs.len(p)];
            eval_showdown(&hs, &table, p, &reaches[t], wins[t], loses[t], ties[t], &mut single);
            let mag = (wins[t].abs() + loses[t].abs() + ties[t].abs())
                * reaches[t].iter().map(|&x| x as f64).sum::<f64>();
            for (a, b) in bufs[t].iter().zip(&single) {
                prop_assert!(close(*a as f64, *b as f64, b.abs() as f64 * 10.0 + mag * 0.01));
            }
            if t % 2 == 0 {
                let mut fold = vec![0f32; hs.len(p)];
                eval_fold(&hs, p, &reaches[t], -2.5 - t as f64, &mut fold);
                for (a, b) in fold_bufs[t].iter().zip(&fold) {
                    prop_assert!(close(*a as f64, *b as f64, b.abs() as f64 * 10.0), "fold {} vs {}", a, b);
                }
            }
        }
    }

    /// Fold values: scalar, batched and brute force agree.
    #[test]
    fn fold_matches_brute_force(board in cards(3), r0 in range(), r1 in range(), seed in any::<u64>(), p in 0usize..2) {
        let Some(hs) = hands_for(&board, &r0, &r1) else { return Ok(()) };
        let r = reach_vec(hs.len(p ^ 1), seed);
        let mut out = vec![0f32; hs.len(p)];
        eval_fold(&hs, p, &r, -3.5, &mut out);
        for (i, &h) in hs.cards[p].iter().enumerate() {
            let brute: f64 = hs.cards[p ^ 1].iter().enumerate()
                .filter(|(_, &o)| !conflict(h, o))
                .map(|(j, _)| r[j] as f64 * -3.5)
                .sum();
            prop_assert!(close(out[i] as f64, brute, brute.abs()));
        }
        let mut b = vec![0f32; hs.len(p)];
        fold_batch(&hs, p, &[&r], &[-3.5], &mut [&mut b]);
        prop_assert!(b.iter().zip(&out).all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs())));
    }
}

fn river_config(
    board: Vec<Card>,
    r0: Vec<f32>,
    r1: Vec<f32>,
    pot: i64,
    stack: i64,
    two_sizes: bool,
    raise: bool,
) -> SolveConfig {
    let bet: &[&str] = if two_sizes { &["50%", "100%"] } else { &["75%"] };
    let rs: &[&str] = if raise { &["3x"] } else { &[] };
    let s = StreetSizes::new(bet, rs);
    let menu = PostflopMenu { flop: s.clone(), turn: s.clone(), river: s, ..PostflopMenu::default() };
    let street = Street::from_board_len(board.len());
    SolveConfig::new(board, [r0, r1], BettingState::street_start(street, pot, [stack, stack], 100), menu)
}

/// All decision nodes reachable by actions only (no chance steps).
fn decision_paths(s: &Solver) -> Vec<Vec<Step>> {
    let mut out = Vec::new();
    let mut stack = vec![Vec::new()];
    while let Some(path) = stack.pop() {
        if !s.is_decision(&path) {
            continue;
        }
        for a in s.actions_at(&path).unwrap() {
            let mut p = path.clone();
            p.push(Step::Action(a));
            stack.push(p);
        }
        out.push(path);
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Solver-wide invariants on random river spots.
    #[test]
    fn river_solutions_are_well_formed(
        board in cards(5), r0 in range(), r1 in range(),
        pot in 200i64..3000, stack in 100i64..6000, two in any::<bool>(), raise in any::<bool>(),
    ) {
        let cfg = river_config(board, r0, r1, pot, stack, two, raise);
        let Ok(mut s) = Solver::new(cfg) else { return Ok(()) };
        let opts = SolveOptions { max_iterations: 30, target_exploitability_pct: 0.0, check_every: 30 };
        let pr = s.solve(&opts, |_| true);
        // Exploitability is non-negative and the game is zero-sum.
        prop_assert!(pr.exploitability >= -1e-3 * pot as f64, "{:?}", pr);
        let v = s.root_values();
        prop_assert!((v[0] + v[1]).abs() <= 1e-3 * pot as f64, "{:?}", v);
        let root_ranges = s.ranges_after(&[]).unwrap();
        for path in decision_paths(&s) {
            let info = s.query(&path).unwrap();
            let n = info.num_hands();
            for h in 0..n {
                let mix: f32 = info.hand_strategy(h).iter().sum();
                prop_assert!((mix - 1.0).abs() < 1e-3);
                prop_assert!(info.hand_strategy(h).iter().all(|&x| (-1e-6..=1.0 + 1e-6).contains(&x)));
                for (a, &ev) in info.actions.iter().zip(info.hand_ev(h).iter()) {
                    prop_assert!(ev.is_finite());
                    if *a == Action::Fold {
                        prop_assert!(ev.abs() < 1e-3, "fold EV {}", ev);
                    }
                }
            }
            // Narrowed ranges never exceed the root ranges.
            let r = s.ranges_after(&path).unwrap();
            for x in 0..2 {
                prop_assert!(r[x].iter().zip(&root_ranges[x]).all(|(a, b)| *a >= 0.0 && *a <= b + 1e-6));
            }
        }
    }

    /// From identical solver states, a batched traversal and a plain
    /// recursive one give the same values and leave the same regrets and
    /// strategy sums. (Whole multi-iteration runs can't be compared: a
    /// regret that is 1e-7 in one summation order and exactly 0 in the
    /// other flips regret matching, and CFR amplifies the difference.)
    #[test]
    fn batched_equals_recursive(board in cards(5), r0 in range(), r1 in range(), pot in 200i64..3000, stack in 100i64..6000) {
        let cfg = river_config(board, r0, r1, pot, stack, true, true);
        let (Ok(mut a), Ok(mut b)) = (Solver::new(cfg.clone()), Solver::new(cfg)) else { return Ok(()) };
        b.set_batch_river(false);
        a.solve(&SolveOptions { max_iterations: 2, target_exploitability_pct: 0.0, check_every: 2 }, |_| true);
        let d = crate::cfr::Discount::new(3, &a.config.dcfr);
        let close = |x: &[f32], y: &[f32]| {
            let scale = x.iter().chain(y).fold(0f32, |m, v| m.max(v.abs()));
            x.iter().zip(y).all(|(u, v)| (u - v).abs() <= 1e-4 * (1.0 + scale))
        };
        for p in 0..2 {
            b.store.copy_from(&a.store);
            for mode in [Mode::Average, Mode::BestResponse, Mode::Cfr(d)] {
                let mut va = vec![0f32; a.hands.len(p)];
                let mut vb = vec![0f32; b.hands.len(p)];
                crate::cfr::traverse(&a.ctx(), 0, p, &a.hands.weights[p ^ 1], mode, &mut va);
                crate::cfr::traverse(&b.ctx(), 0, p, &b.hands.weights[p ^ 1], mode, &mut vb);
                prop_assert!(close(&va, &vb), "{:?} values differ for player {}", mode, p);
            }
            for id in 0..a.tree.nodes.len() {
                let node = a.tree.nodes[id];
                let crate::tree::Kind::Player(q) = node.kind else { continue };
                let blk = node.block(a.hands.len(q as usize));
                let (mut ra, mut rb) = (vec![0f32; blk.len], vec![0f32; blk.len]);
                let (mut sa, mut sb) = (vec![0f32; blk.len], vec![0f32; blk.len]);
                unsafe {
                    a.store.load_regrets(blk, &mut ra);
                    b.store.load_regrets(blk, &mut rb);
                    a.store.load_strat(blk, &mut sa);
                    b.store.load_strat(blk, &mut sb);
                }
                prop_assert!(close(&ra, &rb), "regrets differ at node {}", id);
                prop_assert!(close(&sa, &sb), "strategy sums differ at node {}", id);
            }
        }
    }

    /// Suit isomorphism is lossless on random monotone/two-tone turns.
    #[test]
    fn isomorphism_is_lossless(
        ranks in prop::sample::subsequence((0u8..13).collect::<Vec<_>>(), 4),
        two_tone in any::<bool>(), r0 in range(), r1 in range(),
    ) {
        // Spades for the first three cards; the fourth is a spade (monotone)
        // or a heart (two-tone).
        let mut board: Vec<Card> = ranks[..3].iter().map(|&r| Card::new(r, 3)).collect();
        board.push(Card::new(ranks[3], if two_tone { 2 } else { 3 }));
        let mut cfg = river_config(board, r0, r1, 1000, 3000, false, true);
        cfg.storage = StorageMode::F32;
        let mut cfg_full = cfg.clone();
        cfg_full.isomorphism = false;
        let (Ok(mut a), Ok(mut b)) = (Solver::new(cfg), Solver::new(cfg_full)) else { return Ok(()) };
        let opts = SolveOptions { max_iterations: 4, target_exploitability_pct: 0.0, check_every: 4 };
        let pa = a.solve(&opts, |_| true);
        let pb = b.solve(&opts, |_| true);
        prop_assert!(a.isomorphic_pairs() >= 1);
        prop_assert!((pa.exploitability - pb.exploitability).abs() <= 1e-3 * (1.0 + pb.exploitability.abs()), "{:?} {:?}", pa, pb);
        let (qa, qb) = (a.query(&[]).unwrap(), b.query(&[]).unwrap());
        prop_assert!(qa.strategy.iter().zip(&qb.strategy).all(|(x, y)| (x - y).abs() < 1e-4));
        prop_assert!(qa.ev.iter().zip(&qb.ev).all(|(x, y)| (x - y).abs() < 1e-2 * (1.0 + y.abs())));
    }
}

#[test]
fn combos_cover_all() {
    assert_eq!((0..NUM_COMBOS).map(combo_cards).count(), 1326);
}
