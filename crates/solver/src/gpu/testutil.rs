//! Spots shared by the GPU tests: fixed river configurations and a proptest
//! strategy for random river spots.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use crate::gpu::{Gpu, GpuError};
use crate::{BettingState, SolveConfig, StorageMode};

/// The device, or `None` (the test skips) on a machine without one. Any
/// other failure to initialize fails the test.
pub(crate) fn device() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e @ (GpuError::DriverMissing | GpuError::NoDevice | GpuError::DriverTooOld { .. })) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
        Err(e) => panic!("GPU initialization failed: {e}"),
    }
}
use holdem::card::parse_cards;
use holdem::{Card, Chips, PostflopMenu, Range, Street, StreetSizes};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub(crate) fn menu(bet: &[&str], raise: &[&str]) -> PostflopMenu {
    let s = StreetSizes::new(bet, raise);
    PostflopMenu {
        flop: s.clone(),
        turn: s.clone(),
        river: s,
        max_raises: 3,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merge_threshold: 0.0,
    }
}

pub(crate) fn config(
    board: &str,
    oop: &str,
    ip: &str,
    pot: Chips,
    stack: Chips,
    menu: PostflopMenu,
) -> SolveConfig {
    let board = parse_cards(board).unwrap();
    let street = Street::from_board_len(board.len());
    let mut c = SolveConfig::new(
        board,
        [Range::parse(oop).unwrap().into_weights(), Range::parse(ip).unwrap().into_weights()],
        BettingState::street_start(street, pot, [stack, stack], 100),
        menu,
    );
    c.storage = StorageMode::F32;
    c
}

/// The ranges of `batched_river_matches_recursion` on a river board, with
/// the default (live-game) bet menu.
pub(crate) fn river_config() -> SolveConfig {
    config(
        "Qh9d4c2sKs",
        "AQ, KQ, QJs, JTs, 99, 44, T9s, 87s, A5s, KJs",
        "AA, KK, AQ, QT, 99, JJ, TT, 98s, KJs, 65s, AJs",
        600,
        4000,
        PostflopMenu::default(),
    )
}

/// The same turn spot `batched_river_matches_recursion` uses (has chance
/// nodes, so the GPU must decline it for now).
pub(crate) fn turn_config() -> SolveConfig {
    config(
        "Qh9d4c2s",
        "AQ, KQ, QJs, JTs, 99, 44, T9s, 87s, A5s, KJs",
        "AA, KK, AQ, QT, 99, JJ, TT, 98s, KJs, 65s, AJs",
        600,
        4000,
        PostflopMenu::default(),
    )
}

/// OOP facing a half-pot bet: ties pay something (unequal commitments),
/// and optionally rake is taken.
pub(crate) fn mid_street_config(rake: bool) -> SolveConfig {
    let mut c = config(
        "Kc8d5h3s2c",
        "AA, KK, KQ, KJ, 88, 55, A5s, 76s, QJs, JTs, 44",
        "AK, KQ, KJ, KT, 99, 33, A4s, T9s, 87s, 66",
        1000,
        5000,
        menu(&["50%", "100%"], &["3x"]),
    );
    c.root = facing_bet(1000, 5000, 500);
    if rake {
        c.rake_rate = 0.05;
        c.rake_cap = 300.0;
    }
    c
}

/// River state where OOP faces a bet of `bet` into `pot` (which excludes it).
pub(crate) fn facing_bet(pot: Chips, stack: Chips, bet: Chips) -> BettingState {
    BettingState {
        street: Street::River,
        pot: pot + bet,
        commit: [0, bet],
        stack: [stack, stack - bet],
        current_bet: bet,
        last_raise_inc: bet,
        num_raises: 1,
        to_act: 0,
        needs: [true, false],
    }
}

/// Random weights: each combo is in the range with probability `density`,
/// with a weight in [0.05, 1].
fn random_range(seed: u64, density: f64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..1326)
        .map(|_| if rng.random::<f64>() < density { rng.random_range(0.05f32..=1.0) } else { 0.0 })
        .collect()
}

/// A random river spot: board, ranges (sparse to dense), pot and stack
/// (including no stack behind), bet menu, optionally facing a bet, and
/// optionally raked. Some generated spots have an empty range after board
/// removal; callers skip those (`Solver::new` rejects them).
pub(crate) fn arb_river_config() -> impl Strategy<Value = SolveConfig> {
    let board = prop::sample::subsequence((0u8..52).collect::<Vec<u8>>(), 5).prop_shuffle();
    let ranges = ((any::<u64>(), 0.02f64..0.5), (any::<u64>(), 0.02f64..0.5));
    let stakes = (100 as Chips..3000, prop_oneof![1 => Just(0 as Chips), 4 => 100 as Chips..8000]);
    let menus = 0usize..3;
    let shape = (any::<bool>(), any::<bool>());
    (board, ranges, stakes, menus, shape).prop_map(
        |(board, ((s0, d0), (s1, d1)), (pot, stack), m, (facing, raked))| {
            let menu = match m {
                0 => menu(&["50%", "100%"], &["100%"]),
                1 => menu(&["33%"], &[]),
                _ => menu(&["75%", "150%"], &["2.5x"]),
            };
            let board: Vec<Card> = board.into_iter().map(Card::from_index).collect();
            let mut c = SolveConfig::new(
                board,
                [random_range(s0, d0), random_range(s1, d1)],
                BettingState::street_start(Street::River, pot, [stack, stack], 100),
                menu,
            );
            c.storage = StorageMode::F32;
            if facing && stack > pot / 2 {
                c.root = facing_bet(pot, stack, pot / 2);
            }
            if raked {
                c.rake_rate = 0.05;
                c.rake_cap = 300.0;
            }
            c
        },
    )
}
