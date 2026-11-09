//! Three-player spots shared by the GPU tests: the `multi` test spots plus
//! shapes they lack, and a proptest strategy for random spots.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use crate::gpu::testutil::random_range;
use crate::multi::tests::config;
use crate::multi::{multiway_dcfr, MultiConfig, MultiState, N};
use crate::{ChanceMode, StorageMode};
use holdem::{Card, Chips, PostflopMenu, Street, StreetSizes};
use proptest::prelude::*;

/// The ranges of `three_way_turn_spot` and `batched_river_matches_recursion`.
pub(crate) const RANGES: [&str; 3] = [
    "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
    "AA, KK, AJ, QJ, JT, 99, 77, KQs",
    "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
];

/// The ranges of `narrow_overlapping_ranges_stay_finite`.
pub(crate) const NARROW: [&str; 3] = ["AA, KK, QQ, AK, AQs", "AK, AQ, AJs, KK, QQ, JJ", "AQ, AJ, KQ, TT, 99, 88, A5s"];

pub(crate) fn sampled(mut c: MultiConfig, turn: usize, river: usize, seed: u64) -> MultiConfig {
    c.chance = ChanceMode::Sampled { turn, river, seed };
    c
}

/// The spots of the `multi` tests, plus shapes they lack: an all-in runout,
/// a player folded at the root, a sampled flop.
pub(crate) fn spots() -> Vec<(&'static str, MultiConfig)> {
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

/// A random three-player spot: street (flop and turn with sampled
/// runouts), board, ranges, pot, stacks (some short, for all-in runouts),
/// a player folded at the root or not, bet menu, rake, terminal model, and
/// river batching. Some have an empty range after board removal; callers
/// skip those.
pub(crate) fn arb_multi_config() -> impl Strategy<Value = (MultiConfig, bool)> {
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
                share_runouts: false,
                storage: StorageMode::F32,
                dcfr: multiway_dcfr(),
                backend: crate::Backend::Cpu,
            };
            (cfg, batch)
        },
    )
}

