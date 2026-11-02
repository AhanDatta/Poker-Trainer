//! Prints game-tree sizes for several menu variants (no solving).
use holdem::card::parse_cards;
use holdem::{PostflopMenu, Range, Street, StreetSizes};
use solver::*;

const SB_OPEN: &str = "22+, A2+, K2+, Q2+, J2+, T2s+, T5o+, 92s+, 96o+, 82s+, 85o+, 72s+, 75o+, 62s+, 64o+, 52s+, 54o, 42s+, 32s";
const BB_CALL: &str = "22-99, A2s-A9s, A2o-AJo, K2s-KJs, K5o-KJo, Q2s-QJs, Q8o-QJo, J5s-JTs, J8o-JTo, T6s-T9s, T8o-T9o, 96s-98s, 98o, 85s+, 87o, 74s+, 76o, 63s+, 65o, 53s+, 43s";

fn main() {
    let board = parse_cards("Ks7d2c").unwrap();
    let base = PostflopMenu::default();
    let future = |turn: &str, river: &str, raise: &[&str], max_raises: u32| PostflopMenu {
        turn: StreetSizes::new(&[turn], raise),
        river: StreetSizes::new(&[river], raise),
        max_raises,
        ..PostflopMenu::default()
    };
    let variants: Vec<(&str, PostflopMenu, Option<PostflopMenu>)> = vec![
        ("full menu everywhere", base.clone(), None),
        ("future 1 size, raise 3x, cap 2", base.clone(), Some(future("66%", "75%", &["3x"], 2))),
        ("future 1 size, raise 3x, cap 3", base.clone(), Some(future("66%", "75%", &["3x"], 3))),
        ("future 1 size, raise=allin, cap 2", base.clone(), Some(future("66%", "75%", &["allin"], 2))),
        ("future 1 size, no raises (cap 1)", base.clone(), Some(future("66%", "75%", &["3x"], 1))),
        (
            "flop cap 2, future 1 size cap 2",
            PostflopMenu { max_raises: 2, ..base.clone() },
            Some(future("66%", "75%", &["3x"], 2)),
        ),
    ];
    for (name, menu, fut) in variants {
        let mut cfg = SolveConfig::new(
            board.clone(),
            [Range::parse(BB_CALL).unwrap().into_weights(), Range::parse(SB_OPEN).unwrap().into_weights()],
            BettingState::street_start(Street::Flop, 500, [9750, 9750], 100),
            menu,
        );
        cfg.future_menu = fut;
        cfg.max_memory_bytes = u64::MAX;
        let e = Solver::estimate(&cfg).unwrap();
        println!(
            "{name:<38} player {:>9}  terminal {:>9}  f32 {:>6.2} GB  compressed {:>6.2} GB",
            e.player_nodes,
            e.terminal_nodes,
            e.memory_bytes_f32() as f64 / 1e9,
            TreeStats { compressed: true, ..e }.memory_bytes() as f64 / 1e9
        );
    }
}
