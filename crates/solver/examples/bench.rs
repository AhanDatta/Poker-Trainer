//! Benchmarks realistic heads-up single-raised-pot solves.
//!
//! Usage: cargo run --release -p solver --example bench -- [flop-exact|flop-fast|turn|river|all]

use holdem::card::parse_cards;
use holdem::{PostflopMenu, Range, Street, StreetSizes};
use solver::*;
use std::time::Instant;

const SB_OPEN: &str = "22+, A2+, K2+, Q2+, J2s+, J5o+, T2s+, T6o+, 92s+, 96o+, 82s+, 85o+, 72s+, 75o+, 62s+, 64o+, 52s+, 54o, 42s+, 32s";
const BB_CALL: &str = "22-99, A2s-A9s, A2o-AJo, K2s-KJs, K5o-KJo, Q2s-QJs, Q8o-QJo, J5s-JTs, J8o-JTo, T6s-T9s, T8o-T9o, 96s-98s, 98o, 85s+, 87o, 74s+, 76o, 63s+, 65o, 53s+, 43s";

fn simple_menu() -> PostflopMenu {
    PostflopMenu {
        turn: StreetSizes::new(&["66%"], &["3x"]),
        river: StreetSizes::new(&["75%"], &["3x"]),
        max_raises: 2,
        ..PostflopMenu::default()
    }
}

fn run(name: &str, board: &str, chance: ChanceMode, future: Option<PostflopMenu>) {
    let board = parse_cards(board).unwrap();
    let street = Street::from_board_len(board.len());
    let cfg = SolveConfig {
        board,
        ranges: [
            Range::parse(BB_CALL).unwrap().into_weights(),
            Range::parse(SB_OPEN).unwrap().into_weights(),
        ],
        root: BettingState::street_start(street, 500, [9750, 9750], 100),
        menu: PostflopMenu::default(),
        future_menu: future,
        big_blind: 100,
        chance,
        prune: 0.0,
        keep: [vec![], vec![]],
        max_memory_bytes: 12 << 30,
    };
    let est = Solver::estimate(&cfg).unwrap();
    println!(
        "== {name}: {} player nodes, {} chance, {} terminal, ~{:.2} GB",
        est.player_nodes,
        est.chance_nodes,
        est.terminal_nodes,
        est.memory_bytes() as f64 / 1e9
    );
    let t0 = Instant::now();
    let mut s = Solver::new(cfg).unwrap();
    println!("   hands: {} vs {}, build {:.2?}", s.hands().len(0), s.hands().len(1), t0.elapsed());
    let t1 = Instant::now();
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let opts =
        SolveOptions { max_iterations: iters, target_exploitability_pct: 0.3, check_every: 10.min(iters) };
    let p = s.solve(&opts, |p| {
        if p.iteration % 50 == 0 || p.exploitability_pct < 1.0 {
            println!(
                "   it {:4}: expl {:.3}% pot  ({:.2?})",
                p.iteration,
                p.exploitability_pct,
                t1.elapsed()
            );
        }
        true
    });
    let el = t1.elapsed();
    println!(
        "   done: {} iterations, {:.3}% pot, {:.2?} ({:.1} ms/iter)",
        p.iteration,
        p.exploitability_pct,
        el,
        el.as_secs_f64() * 1000.0 / p.iteration as f64
    );
    let root = s.query(&[]).unwrap();
    let rs = root.range_strategy();
    println!(
        "   root (OOP) strategy: {:?}",
        root.actions.iter().zip(rs).map(|(a, f)| format!("{a:?} {:.1}%", f * 100.0)).collect::<Vec<_>>()
    );
}

fn main() {
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if what == "river" || what == "all" {
        run("river", "Ks7d2c9h4s", ChanceMode::Exact, None);
    }
    if what == "turn" || what == "all" {
        run("turn", "Ks7d2c9h", ChanceMode::Exact, None);
    }
    if what == "flop-fast" || what == "all" {
        run("flop sampled 10x8", "Ks7d2c", ChanceMode::Sampled { turn: 10, river: 8, seed: 1 }, None);
    }
    if what == "flop-fast-simple" || what == "all" {
        run(
            "flop sampled 8x6, simplified turn/river",
            "Ks7d2c",
            ChanceMode::Sampled { turn: 8, river: 6, seed: 1 },
            Some(simple_menu()),
        );
    }
    if what == "flop-exact" {
        run("flop exact (simplified turn/river)", "Ks7d2c", ChanceMode::Exact, Some(simple_menu()));
    }
}
