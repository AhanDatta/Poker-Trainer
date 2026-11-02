//! Compares fast (sampled-runout) flop solves against exact flop solves.
//!
//! For several decision nodes it reports how far the fast solution is from
//! the exact one, in the terms the trainer uses for grading: following the
//! fast solve's best action, how much EV does a hand lose according to the
//! exact solve?
//!
//! Usage: cargo run --release -p solver --example compare -- [board...]
//! Env: TARGET (exploitability % pot, default 0.5), SAMPLES ("8x6,12x8").

use holdem::card::parse_cards;
use holdem::{Action, PostflopMenu, Range, Street, StreetSizes};
use solver::*;
use std::time::Instant;

const SB_OPEN: &str = "22+, A2+, K2+, Q2+, J2+, T2s+, T5o+, 92s+, 96o+, 82s+, 85o+, 72s+, 75o+, 62s+, 64o+, 52s+, 54o, 42s+, 32s";
const BB_CALL: &str = "22-99, A2s-A9s, A2o-AJo, K2s-KJs, K5o-KJo, Q2s-QJs, Q8o-QJo, J5s-JTs, J8o-JTo, T6s-T9s, T8o-T9o, 96s-98s, 98o, 85s+, 87o, 74s+, 76o, 63s+, 65o, 53s+, 43s";

fn config(board: &str, chance: ChanceMode) -> SolveConfig {
    let board = parse_cards(board).unwrap();
    let mut cfg = SolveConfig::new(
        board,
        [Range::parse(BB_CALL).unwrap().into_weights(), Range::parse(SB_OPEN).unwrap().into_weights()],
        BettingState::street_start(Street::Flop, 500, [9750, 9750], 100),
        PostflopMenu::default(),
    );
    cfg.future_menu = Some(PostflopMenu {
        turn: StreetSizes::new(&["66%"], &["allin"]),
        river: StreetSizes::new(&["75%"], &["allin"]),
        max_raises: 2,
        ..PostflopMenu::default()
    });
    cfg.chance = chance;
    cfg.max_memory_bytes = 12 << 30;
    cfg
}

fn solve(board: &str, chance: ChanceMode, target: f64) -> (Solver, f64, f64) {
    let t = Instant::now();
    let mut s = Solver::new(config(board, chance)).unwrap();
    let opts = SolveOptions { max_iterations: 1000, target_exploitability_pct: target, check_every: 10 };
    let p = s.solve(&opts, |_| true);
    (s, p.exploitability_pct, t.elapsed().as_secs_f64())
}

struct Diff {
    range_freq: f64,
    strat_l1: f64,
    ev_abs_pct: f64,
    loss_pct: f64,
    disagree: f64,
    worst_loss_pct: f64,
}

/// Compares two solutions at one node (per-hand values weighted by the
/// exact solve's reach at that node).
fn compare(exact: &NodeInfo, fast: &NodeInfo) -> Diff {
    assert_eq!(exact.actions, fast.actions);
    let n = exact.num_hands();
    let na = exact.actions.len();
    let pot = exact.pot as f64;
    let rf_e = exact.range_strategy();
    let rf_f = fast.range_strategy();
    let range_freq = rf_e.iter().zip(&rf_f).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let (mut wsum, mut l1, mut evd, mut loss, mut dis, mut worst) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    for h in 0..n {
        let w = exact.reach[h] as f64;
        if w <= 0.0 {
            continue;
        }
        let hf = fast.hand_index(exact.combos[h] as usize).unwrap();
        let se = exact.hand_strategy(h);
        let sf = fast.hand_strategy(hf);
        let ee = exact.hand_ev(h);
        let ef = fast.hand_ev(hf);
        let best_e = (0..na).max_by(|&a, &b| ee[a].partial_cmp(&ee[b]).unwrap()).unwrap();
        let best_f = (0..na).max_by(|&a, &b| ef[a].partial_cmp(&ef[b]).unwrap()).unwrap();
        let l = (ee[best_e] - ee[best_f]) as f64 / pot * 100.0;
        wsum += w;
        l1 += w * (0..na).map(|a| (se[a] - sf[a]).abs() as f64).sum::<f64>() / 2.0;
        evd += w * (0..na).map(|a| (ee[a] - ef[a]).abs() as f64).sum::<f64>() / na as f64 / pot * 100.0;
        loss += w * l;
        dis += w * if best_e != best_f && l > 0.5 { 1.0 } else { 0.0 };
        worst = worst.max(l);
    }
    Diff {
        range_freq: range_freq * 100.0,
        strat_l1: l1 / wsum * 100.0,
        ev_abs_pct: evd / wsum,
        loss_pct: loss / wsum,
        disagree: dis / wsum * 100.0,
        worst_loss_pct: worst,
    }
}

fn main() {
    let boards: Vec<String> = std::env::args().skip(1).collect();
    let boards = if boards.is_empty() { vec!["Ks7s2d".to_string()] } else { boards };
    let target: f64 = std::env::var("TARGET").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let samples: Vec<(usize, usize)> = std::env::var("SAMPLES")
        .unwrap_or_else(|_| "8x6,12x8".into())
        .split(',')
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    for board in &boards {
        let (exact, ex, te) = solve(board, ChanceMode::Exact, target);
        println!("\n=== {board}: exact solve {te:.0}s, exploitability {ex:.2}% pot");
        let nodes: Vec<(&str, Vec<Step>)> = vec![
            ("OOP first to act", vec![]),
            ("IP vs check", vec![Step::Action(Action::Check)]),
            ("IP vs 33% bet", vec![Step::Action(exact.actions_at(&[]).unwrap()[1])]),
        ];
        for &(t, r) in &samples {
            let (fast, fx, tf) = solve(board, ChanceMode::Sampled { turn: t, river: r, seed: 1 }, target);
            println!("  fast {t}x{r}: {tf:.0}s, exploitability {fx:.2}% pot (of the sampled game)");
            for (name, path) in &nodes {
                let d = compare(&exact.query(path).unwrap(), &fast.query(path).unwrap());
                println!(
                    "    {name:<18} range freq Δ {:4.1}pp | strategy Δ {:4.1}% | EV Δ {:4.2}% pot | \
                     following fast loses {:4.2}% pot avg (worst hand {:4.1}%), hands with a different best action {:4.1}%",
                    d.range_freq, d.strat_l1, d.ev_abs_pct, d.loss_pct, d.worst_loss_pct, d.disagree
                );
            }
        }
    }
}
