//! Every leaf of a traversal against the CPU's evaluator on the same reach
//! (bit for bit with `GpuPrecision::Exact`, within one f32 rounding with
//! `Fast`), both against a brute force over hand triples, and fast solves.

use super::*;
use crate::multi::cfr::{independent_showdown, pay_all};
use crate::multi::exact;
use holdem::eval::EvalMask;
use holdem::{Card, CardSet};

/// One leaf of a finished traversal: its GPU row, the CPU's value of it on
/// the GPU's reach, and two magnitudes per hand: `scale = size * c_q * c_r`
/// (`size` = `|pay|`, or `2 max(|win|, |lose|)` at showdowns), the size of
/// the result, which bounds what one f32 rounding can move; and `floor =
/// size * T_q * T_r` with `T` an opponent's total reach, the size of the
/// terms the leaf's f64 sums cancel, which bounds their rounding noise (a
/// hand that blocks every remaining combo of an opponent is worth 0, which
/// the CPU's f64 sums leave as noise of ~1e-16 of the floor).
struct Leaf {
    node: u32,
    gpu: Vec<f32>,
    cpu: Vec<f32>,
    scale: Vec<f64>,
    floor: f64,
    /// Both opponents' reach at the leaf.
    reach: [Vec<f32>; 2],
    /// `(win, lose)` at a showdown, `(pay, pay)` at a fixed payoff.
    pay: (f64, f64),
    showdown: bool,
}

/// Runs traversal `(p, mode)` and returns every live leaf.
fn leaves_after(m: &MultiSolver, engine: &mut MultiEngine, p: usize, mode: Mode) -> Vec<Leaf> {
    engine.traverse(p, mode).unwrap();
    let stream = &engine.gpu.stream;
    let values = stream.clone_dtoh(&engine.values).unwrap();
    let reach = [stream.clone_dtoh(&engine.reach[0]).unwrap(), stream.clone_dtoh(&engine.reach[1]).unwrap()];
    let alive = [stream.clone_dtoh(&engine.alive[0]).unwrap(), stream.clone_dtoh(&engine.alive[1]).unwrap()];
    let ctx = m.ctx();
    let pp = &engine.plan.players[p];
    let (q, r) = opponents(p);
    let n = pp.hands as usize;
    let lens = [pp.opponents[0].hands as usize, pp.opponents[1].hands as usize];
    let slot_reach = |k: usize, node: u32| -> Vec<f32> {
        let s = pp.slot[k][node as usize] as usize;
        reach[k][s * lens[k]..(s + 1) * lens[k]].to_vec()
    };
    let live = |node: u32| (0..2).all(|k| alive[k][pp.slot[k][node as usize] as usize] != 0);
    let mut out = Vec::new();
    let mut push = |node: u32, cpu: Vec<f32>, reach: [Vec<f32>; 2], pay: (f64, f64), showdown: bool| {
        let cq = ctx.hands.compat(p, q, &reach[0]);
        let cr = ctx.hands.compat(p, r, &reach[1]);
        let size = pay.0.abs().max(pay.1.abs()) * if showdown { 2.0 } else { 1.0 };
        let scale = (0..n).map(|h| size * cq[h] * cr[h]).collect();
        let total = |w: &[f32]| w.iter().map(|&x| x as f64).sum::<f64>();
        let floor = size * total(&reach[0]) * total(&reach[1]);
        let gpu = values[node as usize * n..(node as usize + 1) * n].to_vec();
        out.push(Leaf { node, gpu, cpu, scale, floor, reach, pay, showdown });
    };
    for (k, &node) in pp.fixed.iter().enumerate() {
        if !live(node) {
            continue;
        }
        let rs = [slot_reach(0, node), slot_reach(1, node)];
        let pay = pp.fixed_payoff[k];
        let mut cpu = vec![0f32; n];
        if engine.plan.scalar[node as usize] == 1 {
            pay_all(&ctx, p, &rs[0], &rs[1], pay, &mut cpu);
        } else {
            // `river::subtree`: the pair mass rounded to f32 first.
            pay_all(&ctx, p, &rs[0], &rs[1], 1.0, &mut cpu);
            cpu.iter_mut().for_each(|v| *v = (pay * *v as f64) as f32);
        }
        push(node, cpu, rs, (pay, pay), false);
    }
    for (k, &node) in pp.showdowns.iter().enumerate() {
        if !live(node) {
            continue;
        }
        let rs = [slot_reach(0, node), slot_reach(1, node)];
        let folded = ctx.tree.nodes[node as usize].folded;
        let table = &ctx.tables[pp.showdown_table[k] as usize];
        let (win, lose) = (pp.win[k], pp.lose[k]);
        let mut cpu = vec![0f32; n];
        if ctx.exact {
            exact::showdown(table, folded, p, &rs[0], &rs[1], win, lose, &mut cpu);
        } else {
            independent_showdown(ctx.hands, table, folded, p, &rs[0], &rs[1], win, lose, &mut cpu);
        }
        push(node, cpu, rs, (win, lose), true);
    }
    out
}

/// A random-storage spot with an engine at `precision`.
fn random_engine(cfg: &MultiConfig, model: Model, seed: u64, precision: GpuPrecision) -> Option<(MultiSolver, MultiEngine)> {
    let gpu = device()?;
    let mut m = solver(cfg, model);
    randomize_storage(&mut m, seed);
    let engine = engine_with(&mut m, gpu, precision);
    Some((m, engine))
}

const MODES: [Mode; 2] = [Mode::Average, Mode::BestResponse];

/// Every leaf value of every traversal, bit for bit, on every spot and
/// model (random storage; the root-level tests can miss a leaf whose
/// difference a sum rounds away).
#[test]
fn every_leaf_matches_cpu_bit_for_bit() {
    for (k, (name, cfg, model)) in variants().into_iter().enumerate() {
        let Some((m, mut engine)) = random_engine(&cfg, model, 40 + k as u64, GpuPrecision::Exact) else { return };
        let mut checked = 0usize;
        for p in 0..N {
            for mode in MODES {
                for leaf in leaves_after(&m, &mut engine, p, mode) {
                    let (diff, total) = differing(&bits(&leaf.gpu), &bits(&leaf.cpu));
                    assert!(diff == 0, "{name}, player {p}, {mode:?}, leaf {}: {diff} of {total} differ", leaf.node);
                    checked += total;
                }
            }
        }
        assert!(checked > 0, "{name}: no live leaves");
    }
}

/// The f32 epsilon.
const EPS: f64 = f32::EPSILON as f64;

/// The f64 noise floor of a leaf's sums, relative to `Leaf::floor`: far
/// above f64's own rounding (n * 2^-53 for n <= 2652 terms is 3e-13), far
/// below an f32 rounding step.
const NOISE: f64 = 1e-12;

/// The tolerance of one leaf value: `ulps` f32 roundings of its scale plus
/// the f64 noise floor.
fn tolerance(leaf: &Leaf, h: usize, ulps: f64) -> f64 {
    ulps * EPS * leaf.scale[h] + NOISE * leaf.floor
}

/// The fast precision (double-single sums) on the same reach: each value
/// within one f32 rounding of the CPU's (`2 * eps * scale`, plus the f64
/// noise floor: double-single's own error is ~2^-44 of the floor), and
/// nearly all identical. Plain f32 sums would miss both bounds.
#[test]
fn fast_leaves_are_within_one_rounding_of_the_cpu() {
    for (k, (name, cfg, model)) in variants().into_iter().enumerate() {
        let Some((m, mut engine)) = random_engine(&cfg, model, 80 + k as u64, GpuPrecision::Fast) else { return };
        let (mut total, mut identical) = (0usize, 0usize);
        for p in 0..N {
            for mode in MODES {
                for leaf in leaves_after(&m, &mut engine, p, mode) {
                    for (h, (&g, &c)) in leaf.gpu.iter().zip(&leaf.cpu).enumerate() {
                        total += 1;
                        if g.to_bits() == c.to_bits() {
                            identical += 1;
                            continue;
                        }
                        let d = (g as f64 - c as f64).abs();
                        assert!(
                            d <= tolerance(&leaf, h, 2.0),
                            "{name}, player {p}, {mode:?}, leaf {}, hand {h}: {g:e} vs {c:e} (scale {:e})",
                            leaf.node,
                            leaf.scale[h]
                        );
                    }
                }
            }
        }
        assert!(identical * 1000 >= total * 999, "{name}: only {identical} of {total} identical");
    }
}

/// `pairs`-weighted payoff of traverser hand `h` at a leaf, summed over
/// every pair of opponent hands disjoint from `h` and from each other.
fn brute_leaf(m: &MultiSolver, p: usize, leaf: &Leaf, board: &[Card], folded: [bool; N], h: usize) -> f64 {
    let ctx = m.ctx();
    let (q, r) = opponents(p);
    let cards = &ctx.hands.cards;
    let mine = cards[p][h];
    let dead = CardSet::from_cards(board);
    let on_board = |(a, b): (u8, u8)| dead.contains(Card::from_index(a)) || dead.contains(Card::from_index(b));
    let disjoint = |x: (u8, u8), y: (u8, u8)| x.0 != y.0 && x.0 != y.1 && x.1 != y.0 && x.1 != y.1;
    let mask = EvalMask::from_cards(board);
    let strength = |(a, b): (u8, u8)| mask.add(Card::from_index(a)).add(Card::from_index(b)).evaluate();
    if leaf.showdown && on_board(mine) && ctx.exact {
        return 0.0;
    }
    let mut v = 0.0;
    for (i, &x) in cards[q].iter().enumerate() {
        let wq = leaf.reach[0][i] as f64;
        if wq == 0.0 || !disjoint(mine, x) {
            continue;
        }
        for (j, &y) in cards[r].iter().enumerate() {
            let wr = leaf.reach[1][j] as f64;
            if wr == 0.0 || !disjoint(mine, y) || (ctx.exact && !disjoint(x, y)) {
                continue;
            }
            let pay = if !leaf.showdown {
                leaf.pay.0
            } else {
                let (win, lose) = leaf.pay;
                if ctx.exact {
                    let sh = strength(mine);
                    let live = [(q, x), (r, y)].into_iter().filter(|&(o, _)| !folded[o]);
                    let best = live.clone().map(|(_, c)| strength(c)).fold(sh, std::cmp::max);
                    if sh < best {
                        lose
                    } else {
                        let k = 1 + live.filter(|&(_, c)| strength(c) == best).count();
                        lose + (win - lose) / k as f64
                    }
                } else {
                    // Independent opponents: P(win) = P(beat q) P(beat r),
                    // a tie with one counting half (no strength on the board
                    // for a traverser hand on its cards).
                    let beats = |o: usize, c: (u8, u8)| -> f64 {
                        if folded[o] {
                            1.0
                        } else if on_board(mine) {
                            0.0
                        } else {
                            match strength(mine).cmp(&strength(c)) {
                                std::cmp::Ordering::Greater => 1.0,
                                std::cmp::Ordering::Equal => 0.5,
                                std::cmp::Ordering::Less => 0.0,
                            }
                        }
                    };
                    let pwin = beats(q, x) * beats(r, y);
                    win * pwin + lose * (1.0 - pwin)
                }
            };
            v += wq * wr * pay;
        }
    }
    v
}

/// Both models and precisions against the brute force on small spots:
/// within two f32 roundings of the scale plus the f64 noise floor (the
/// brute force's own f64 sums, and the leaf's).
#[test]
fn leaves_match_a_brute_force_over_hand_triples() {
    let small = ["river, third player folded", "narrow turn", "river"];
    for (name, cfg) in spots().into_iter().filter(|s| small.contains(&s.0)) {
        for model in [MODELS[0], MODELS[2]] {
            for precision in [GpuPrecision::Exact, GpuPrecision::Fast] {
                let Some((m, mut engine)) = random_engine(&cfg, model, 7, precision) else { return };
                let mut checked = 0;
                for p in 0..N {
                    for leaf in leaves_after(&m, &mut engine, p, Mode::Average) {
                        let ctx = m.ctx();
                        let node = &ctx.tree.nodes[leaf.node as usize];
                        // A showdown's final board (fixed payoffs don't look at it).
                        let board: Vec<Card> =
                            if leaf.showdown { ctx.tree.boards[node.slot as usize].clone() } else { Vec::new() };
                        for h in 0..leaf.gpu.len() {
                            let want = brute_leaf(&m, p, &leaf, &board, node.folded, h);
                            let d = (leaf.gpu[h] as f64 - want).abs();
                            assert!(
                                d <= tolerance(&leaf, h, 2.0),
                                "{name} ({model:?}, {precision:?}), player {p}, leaf {}, hand {h}: {} vs {want:e}",
                                leaf.node,
                                leaf.gpu[h]
                            );
                            checked += 1;
                        }
                    }
                }
                assert!(checked > 100, "{name}: only {checked} values checked");
            }
        }
    }
}

/// Fast-precision solves after 5 iterations: exploitability within 3% and
/// the range-level root strategy within 0.01 of the CPU's. Per-hand values
/// can't be held to the leaf bound: regret matching jumps where a regret
/// crosses 0, so a value that is 0 in exact arithmetic (noise on the CPU, 0
/// or other noise on the GPU) flips an indifferent hand between uniform and
/// pure play, and CFR carries that forward. Measured worst after 5
/// iterations: 1.2% (sampled flop) and 0.003 (first player folded), with
/// independent opponents no difference at all; the CPU's own batched and
/// unbatched river leaves (one rounding apart) drift 0.005% and 1.5e-4.
#[test]
fn fast_short_solves_match_cpu() {
    for (name, cfg, model) in variants() {
        let Some(_) = device() else { return };
        let mut cpu = solver(&cfg, model);
        let mut fast = solver(&cfg, model);
        fast.use_gpu(GpuPrecision::Fast).unwrap();
        let pc = cpu.solve(&opts(5, 5), |_| true);
        let pf = fast.solve(&opts(5, 5), |_| true);
        assert!(fast.uses_gpu() && fast.gpu_error().is_none(), "{name}: {:?}", fast.gpu_error());
        let rel = (pc.exploitability - pf.exploitability).abs() / pc.exploitability.abs().max(1e-9);
        assert!(rel <= 0.03, "{name}: exploitability {} vs {}", pc.exploitability, pf.exploitability);
        let (a, b) = (cpu.query(&[]).unwrap().range_strategy(), fast.query(&[]).unwrap().range_strategy());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 0.01, "{name}: range strategy {a:?} vs {b:?}");
        }
    }
}

/// A fast-precision solve to 1% of the pot: as many iterations as the CPU
/// (within 20%) and the same range-level root strategy (CFR amplifies
/// rounding differences in single hands near indifference, so those can't
/// be compared, as heads-up).
#[test]
fn fast_converged_solve_matches_cpu() {
    let cfg = sampled(config("Jh8d4c2s", RANGES, [false; N]), 0, 12, 1);
    for model in [MODELS[0], MODELS[1]] {
        let Some(_) = device() else { return };
        let mut cpu = solver(&cfg, model);
        let mut fast = solver(&cfg, model);
        fast.use_gpu(GpuPrecision::Fast).unwrap();
        let opts = SolveOptions { max_iterations: 300, target_exploitability_pct: 1.0, check_every: 10 };
        let pc = cpu.solve(&opts, |_| true);
        let pf = fast.solve(&opts, |_| true);
        assert!(fast.gpu_error().is_none(), "{:?}", fast.gpu_error());
        assert!(pc.exploitability_pct <= 1.0 && pf.exploitability_pct <= 1.0, "{model:?}: {pc:?} {pf:?}");
        let (ic, i_fast) = (f64::from(pc.iteration), f64::from(pf.iteration));
        assert!((ic - i_fast).abs() <= 0.2 * ic, "{model:?}: iterations to target: cpu {ic}, gpu {i_fast}");
        let (a, b) = (cpu.query(&[]).unwrap().range_strategy(), fast.query(&[]).unwrap().range_strategy());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 0.03, "{model:?}: {a:?} vs {b:?}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Random spots at the fast precision: every leaf within one rounding of
    /// the CPU, and 3 iterations close to it.
    #[test]
    fn random_spots_match_cpu_fast((cfg, batch) in arb_multi_config(), seed in any::<u64>()) {
        if MultiSolver::new(cfg.clone()).is_err() {
            return Ok(());
        }
        let model = Model { exact: cfg.exact_removal, batch };
        let Some((m, mut engine)) = random_engine(&cfg, model, seed, GpuPrecision::Fast) else { return Ok(()) };
        for p in 0..N {
            for leaf in leaves_after(&m, &mut engine, p, Mode::Average) {
                for (h, (&g, &c)) in leaf.gpu.iter().zip(&leaf.cpu).enumerate() {
                    prop_assert!((g as f64 - c as f64).abs() <= tolerance(&leaf, h, 2.0), "leaf {}: {} vs {}", leaf.node, g, c);
                }
            }
        }
    }
}

