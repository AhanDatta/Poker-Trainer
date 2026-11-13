//! Measuring what hands realize in one spot on one flop: a postflop solve
//! with the real bet sizes, and a check-down solve that pays every hand
//! exactly its equity share of the pot.

use super::spots::Spot;
use crate::spec::{GameSpec, BB};
use holdem::combo::COMBO_CLASS;
use holdem::{combo_cards, Card, CardSet, Chips, PostflopMenu, Street, StreetSizes, NUM_CLASSES, NUM_COMBOS};
use solver::{Backend, BettingState, ChanceMode, SolveConfig, SolveOptions};
use std::time::Instant;

/// How runouts are dealt in the solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chance {
    /// Every turn and river card.
    Exact,
    /// A stratified sample of turns and rivers (the trainer's fast mode is
    /// 49 turns and 10 rivers), seeded from the board.
    Sampled { turn: usize, river: usize },
}

/// The postflop game the spots are solved in.
#[derive(Clone, Debug, PartialEq)]
pub struct PostflopSetup {
    /// Sizes on the flop (and later streets without `future_menu`).
    pub menu: PostflopMenu,
    /// Sizes on the turn and river.
    pub future_menu: Option<PostflopMenu>,
    pub chance: Chance,
    pub backend: Backend,
    /// Stop at this exploitability, in % of the pot.
    pub target_exploitability_pct: f64,
    pub max_iterations: u32,
    pub rake_rate: f64,
    /// Most rake per pot, in chips.
    pub rake_cap: Chips,
    /// Every hand class gets at least this fraction of the range's largest
    /// weight, so that every class is measured.
    pub floor: f32,
    pub max_memory_bytes: u64,
}

impl PostflopSetup {
    /// The trainer's review settings for flop solves (the default table
    /// sizes on the flop; one size and all-in raises on the turn and river),
    /// with `spec`'s rake, on the CPU.
    pub fn review(spec: &GameSpec) -> PostflopSetup {
        let future = PostflopMenu {
            turn: StreetSizes::new(&["66%"], &["allin"]),
            river: StreetSizes::new(&["75%"], &["allin"]),
            max_raises: 2,
            ..PostflopMenu::default()
        };
        PostflopSetup {
            menu: PostflopMenu::default(),
            future_menu: Some(future),
            chance: Chance::Exact,
            backend: Backend::Cpu,
            target_exploitability_pct: 0.5,
            max_iterations: 400,
            rake_rate: spec.rake_rate,
            rake_cap: spec.rake_cap(),
            floor: 0.01,
            max_memory_bytes: 5_000_000_000,
        }
    }

    /// The same game with no bets at all: the hands check down, and each is
    /// worth its equity share of the pot after rake. One iteration solves it,
    /// on the CPU.
    pub fn check_down(&self) -> PostflopSetup {
        PostflopSetup {
            menu: check_only_menu(),
            future_menu: None,
            backend: Backend::Cpu,
            target_exploitability_pct: 0.0,
            max_iterations: 1,
            ..self.clone()
        }
    }
}

/// A menu without bets: every decision is a check.
pub fn check_only_menu() -> PostflopMenu {
    let none = StreetSizes { bet: Vec::new(), raise: Vec::new() };
    PostflopMenu {
        flop: none.clone(),
        turn: none.clone(),
        river: none,
        add_allin_threshold: 0.0,
        ..PostflopMenu::default()
    }
}

/// The solver's ranges for `spot` on `board`: each player's chart range
/// with every class raised to `floor` times the range's largest weight, and
/// the combos that share a card with the board removed.
pub fn solve_ranges(spot: &Spot, board: &[Card], floor: f32) -> [Vec<f32>; 2] {
    let dead = CardSet::from_cards(board);
    spot.classes.map(|classes| {
        let max = classes.iter().copied().fold(0f32, f32::max);
        let low = floor.max(0.0) * max;
        (0..NUM_COMBOS)
            .map(|c| {
                let (a, b) = combo_cards(c);
                if dead.contains(a) || dead.contains(b) {
                    0.0
                } else {
                    classes[COMBO_CLASS[c] as usize].max(low)
                }
            })
            .collect()
    })
}

/// The solver input for `spot` on `board` in `setup`.
pub fn config(spot: &Spot, board: &[Card], setup: &PostflopSetup) -> SolveConfig {
    let root = BettingState::street_start(Street::Flop, spot.pot, [spot.behind; 2], BB);
    let ranges = solve_ranges(spot, board, setup.floor);
    let mut cfg = SolveConfig::new(board.to_vec(), ranges, root, setup.menu.clone());
    cfg.future_menu = setup.future_menu.clone();
    cfg.big_blind = BB;
    cfg.chance = match setup.chance {
        Chance::Exact => ChanceMode::Exact,
        Chance::Sampled { turn, river } => ChanceMode::Sampled { turn, river, seed: board_seed(board) },
    };
    cfg.prune = 0.0;
    cfg.max_memory_bytes = setup.max_memory_bytes;
    cfg.rake_rate = setup.rake_rate;
    cfg.rake_cap = setup.rake_cap as f64;
    cfg.backend = setup.backend;
    cfg
}

/// A seed for sampled runouts that depends only on the board (as the
/// trainer's).
fn board_seed(board: &[Card]) -> u64 {
    board.iter().fold(17u64, |h, c| h.wrapping_mul(31).wrapping_add(c.index() as u64))
}

/// One player's result on one flop, per hand class.
#[derive(Clone, Debug, PartialEq)]
pub struct SeatSums {
    /// Weight of each of the class's combos in the calibration: the spot's
    /// reach times the combo's share of the player's solve range.
    pub weight: Vec<f64>,
    /// Sum over the class's combos of their value with the real bet sizes
    /// (chips won from the pot minus chips put in on the flop and after).
    pub value: Vec<f64>,
    /// The same with checks only: the equity share of the pot after rake.
    pub check_down: Vec<f64>,
    /// Combos of the class that don't share a card with the board.
    pub combos: Vec<u32>,
}

/// How a solve went.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveInfo {
    pub iterations: u32,
    pub exploitability_pct: f64,
    /// `"Cpu"`, `"GpuFast"` or `"GpuExact"`.
    pub solved_on: String,
    /// Why the GPU wasn't used, if it was asked for and not used.
    pub gpu_error: Option<String>,
    pub seconds: f64,
}

/// Per player and class: the sum of the combos' values, and the number of
/// combos; and how the solve went.
type SolveValues = ([Vec<f64>; 2], [Vec<u32>; 2], SolveInfo);

/// Each class's values in one solve of `spot` on `board`.
fn solve_values(spot: &Spot, board: &[Card], setup: &PostflopSetup) -> Result<SolveValues, String> {
    let start = Instant::now();
    let cfg = config(spot, board, setup);
    let mut solver = solver::Solver::new(cfg).map_err(|e| format!("{e:?}"))?;
    let opts = SolveOptions {
        max_iterations: setup.max_iterations.max(1),
        target_exploitability_pct: setup.target_exploitability_pct,
        check_every: 10,
    };
    let progress = solver.solve(&opts, |_| true);
    let mut sums = [vec![0f64; NUM_CLASSES], vec![0f64; NUM_CLASSES]];
    let mut combos = [vec![0u32; NUM_CLASSES], vec![0u32; NUM_CLASSES]];
    for p in 0..2 {
        let values = solver.hand_values(p);
        let hands = solver.hands();
        for (i, &combo) in hands.combos[p].iter().enumerate() {
            let class = COMBO_CLASS[combo as usize] as usize;
            sums[p][class] += values[i] as f64;
            combos[p][class] += 1;
        }
    }
    let info = SolveInfo {
        iterations: progress.iteration,
        exploitability_pct: progress.exploitability_pct,
        solved_on: format!("{:?}", solver.solved_on()),
        gpu_error: solver.gpu_error().map(str::to_string),
        seconds: start.elapsed().as_secs_f64(),
    };
    Ok((sums, combos, info))
}

/// Solves `spot` on `board` with `setup`'s bet sizes and checked down, and
/// returns both players' sums, in seat order, and how the real solve went.
pub fn measure(
    spot: &Spot,
    board: &[Card],
    setup: &PostflopSetup,
) -> Result<([SeatSums; 2], SolveInfo), String> {
    if board.len() != 3 || CardSet::from_cards(board).len() != 3 {
        return Err(format!("a flop is three different cards, not {}", holdem::card::cards_to_string(board)));
    }
    let (value, combos, info) = solve_values(spot, board, setup)?;
    let (check_down, combos_again, _) = solve_values(spot, board, &setup.check_down())?;
    debug_assert_eq!(combos, combos_again);
    let ranges = solve_ranges(spot, board, setup.floor);
    let sums = [0, 1].map(|p| {
        // Every live combo of a class has the same weight.
        let mut class_weight = [0f64; NUM_CLASSES];
        for (c, &w) in ranges[p].iter().enumerate().filter(|(_, &w)| w > 0.0) {
            class_weight[COMBO_CLASS[c] as usize] = w as f64;
        }
        let total: f64 = ranges[p].iter().map(|&w| w as f64).sum();
        SeatSums {
            weight: class_weight.iter().map(|w| spot.reach * w / total).collect(),
            value: value[p].clone(),
            check_down: check_down[p].clone(),
            combos: combos[p].clone(),
        }
    });
    Ok((sums, info))
}
