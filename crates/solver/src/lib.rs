//! Heads-up postflop solver based on Discounted CFR.
//!
//! The solver works on the full range-vs-range game: every decision node
//! stores, for the acting player, one regret and one strategy accumulator
//! per `(action, hand)`. A single traversal handles all hands at once by
//! propagating the opponent's reach vector down the tree and returning the
//! traverser's counterfactual values (see [`cfr`]).
//!
//! Performance features:
//! * flat node arena; optional 16-bit compressed storage (half the memory
//!   and memory traffic);
//! * suit isomorphism: mirror-image turn/river cards are solved once;
//! * river subtrees are evaluated breadth-first with all showdowns (and
//!   folds) of a subtree swept together in SIMD-friendly batches;
//! * O(n) showdown evaluation via strength-sorted sweeps with
//!   inclusion–exclusion card removal;
//! * the bet tree comes from [`holdem::menu`], the same code the table uses,
//!   so every action played at the table exists in the tree.
//!
//! The design was informed by reading b-inary/postflop-solver.

// Index loops are clearer than iterator chains in the numeric code here.
#![allow(clippy::needless_range_loop)]

mod cfr;
mod eval;
#[cfg(feature = "gpu")]
pub mod gpu;
mod hands;
mod iso;
pub mod multi;
#[cfg(test)]
mod multi_ref;
mod river;
mod sampling;
mod scratch;
mod storage;
mod tree;

pub use cfr::DcfrParams;
use cfr::{Ctx, Discount, Mode};
pub use hands::HandSet;
use hands::StrengthTable;
use holdem::{Action, Card, CardSet, Chips, PostflopMenu, Street};
pub use iso::{board_symmetric_pairs, iso_orbit, swap_card};
use iso::{Isomorphism, PAIRS};
use rayon::prelude::*;
use std::fmt;
use storage::Storage;
pub use storage::StorageMode;
pub use tree::{BettingState, ChanceMode, Transition, TreeStats};
use tree::{Builder, Kind, Tree, TreeParams};

/// Whether this build can solve on the GPU (the `gpu` feature).
pub const GPU_SUPPORT: bool = cfg!(feature = "gpu");

/// Where a solve ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolvedOn {
    /// The CPU (also after a GPU that couldn't be used or failed mid-solve;
    /// [`Solver::gpu_error`] says why).
    #[default]
    Cpu,
    /// The GPU with double-single terminal sums (`GpuPrecision::Fast`).
    GpuFast,
    /// The GPU with f64 terminal sums, bit-identical to the CPU
    /// (`GpuPrecision::Exact`).
    GpuExact,
}

/// Where solves run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// The CPU.
    #[default]
    Cpu,
    /// The GPU, when the solver is built with the `gpu` feature and an NVIDIA
    /// GPU with enough free memory is present. Otherwise, or if the GPU fails
    /// during a solve, the solve runs on the CPU and [`Solver::gpu_error`]
    /// says why. The GPU is held only while `solve` runs.
    Gpu,
    /// Like `Gpu` for turn and flop roots; river roots stay on the CPU (their
    /// small trees solve about as fast there).
    Auto,
}

/// Everything needed to build a solvable game.
#[derive(Clone, Debug)]
pub struct SolveConfig {
    /// Board at the root (3, 4 or 5 cards).
    pub board: Vec<Card>,
    /// Range weights (1326 entries) for player 0 (OOP) and player 1 (IP).
    pub ranges: [Vec<f32>; 2],
    /// Betting state at the root.
    pub root: BettingState,
    /// Sizes used on the root street (and later streets unless
    /// `future_menu` is set).
    pub menu: PostflopMenu,
    /// Optional (usually simplified) sizes for streets after the root street.
    pub future_menu: Option<PostflopMenu>,
    pub big_blind: Chips,
    pub chance: ChanceMode,
    /// Combos below `prune * max weight` are dropped (0 keeps everything).
    pub prune: f32,
    /// Combos that must never be pruned (e.g. the hero's actual hand).
    pub keep: [Vec<usize>; 2],
    /// Refuse to build trees needing more memory than this.
    pub max_memory_bytes: u64,
    /// Rake as a fraction of the pot (taken from every postflop pot).
    pub rake_rate: f64,
    /// Maximum rake per pot, in chips.
    pub rake_cap: f64,
    pub storage: StorageMode,
    /// Solve mirror-image (suit-isomorphic) runouts only once.
    pub isomorphism: bool,
    pub dcfr: DcfrParams,
    /// Where solves run (the CPU by default).
    pub backend: Backend,
}

impl SolveConfig {
    /// A config with default options (exact runouts, no rake, automatic
    /// storage compression, isomorphism on, 8 GB limit).
    pub fn new(
        board: Vec<Card>,
        ranges: [Vec<f32>; 2],
        root: BettingState,
        menu: PostflopMenu,
    ) -> SolveConfig {
        SolveConfig {
            board,
            ranges,
            root,
            menu,
            future_menu: None,
            big_blind: 100,
            chance: ChanceMode::Exact,
            prune: 0.0,
            keep: [Vec::new(), Vec::new()],
            max_memory_bytes: 8 << 30,
            rake_rate: 0.0,
            rake_cap: 0.0,
            storage: StorageMode::Auto,
            isomorphism: true,
            dcfr: DcfrParams::default(),
            backend: Backend::Cpu,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub max_iterations: u32,
    /// Stop once exploitability is at most this percentage of the root pot.
    pub target_exploitability_pct: f64,
    /// Compute exploitability at least every this many iterations. Near
    /// the target, checks are scheduled for when the convergence trend
    /// predicts it will be reached (but at most 4 times as often), so the
    /// solve stops closer to the target.
    pub check_every: u32,
}

impl Default for SolveOptions {
    fn default() -> Self {
        SolveOptions { max_iterations: 1000, target_exploitability_pct: 0.5, check_every: 10 }
    }
}

/// Iterations until the next exploitability check: fits `e ∝ t^-k` through
/// the last two checks and aims at the iteration where the target is
/// predicted, between `max_gap / 4` and `max_gap` iterations ahead.
pub(crate) fn next_check_gap(prev: Option<(u32, f64)>, (t, e): (u32, f64), target: f64, max_gap: u32) -> u32 {
    let min_gap = (max_gap / 4).max(1);
    let Some((t0, e0)) = prev else { return max_gap };
    if !(e < e0 && e > target && t > t0 && target > 0.0) {
        return max_gap;
    }
    let k = (e0 / e).ln() / (t as f64 / t0 as f64).ln();
    let predicted = t as f64 * (e / target).powf(1.0 / k);
    let gap = (predicted - t as f64).ceil();
    if gap.is_finite() {
        (gap.max(0.0) as u32).clamp(min_gap, max_gap)
    } else {
        max_gap
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub iteration: u32,
    pub max_iterations: u32,
    /// Exploitability in chips.
    pub exploitability: f64,
    /// Exploitability as a percentage of the root pot.
    pub exploitability_pct: f64,
    /// Whether the solve stopped (target reached, iteration cap, or cancelled).
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    EmptyRange(usize),
    InvalidBoard(String),
    TooLarge { needed: u64, limit: u64 },
    BadPath(String),
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::EmptyRange(p) => write!(f, "player {p} has an empty range"),
            SolverError::InvalidBoard(s) => write!(f, "invalid board: {s}"),
            SolverError::TooLarge { needed, limit } => write!(
                f,
                "game tree needs ~{:.2} GB but the limit is {:.2} GB",
                *needed as f64 / 1e9,
                *limit as f64 / 1e9
            ),
            SolverError::BadPath(s) => write!(f, "bad path: {s}"),
        }
    }
}

impl std::error::Error for SolverError {}

/// One step along a path from the root: an action or a dealt card.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    Action(Action),
    Card(Card),
}

/// Strategy and values at a decision node.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Acting player (0 = OOP, 1 = IP).
    pub player: usize,
    pub actions: Vec<Action>,
    /// Combo index of each of the acting player's hands.
    pub combos: Vec<u16>,
    /// Average strategy, action-major: `strategy[a * n + i]`.
    pub strategy: Vec<f32>,
    /// EV of each action in chips, measured from this decision on
    /// (folding is worth 0), action-major.
    pub ev: Vec<f32>,
    /// Acting player's reach weights at this node (per hand).
    pub reach: Vec<f32>,
    /// Fraction of each player's root range that reaches this node.
    pub reach_fraction: [f64; 2],
    /// Pot at this node.
    pub pot: Chips,
    /// Chips each player invested since the root.
    pub invest: [Chips; 2],
}

impl NodeInfo {
    pub fn num_hands(&self) -> usize {
        self.combos.len()
    }

    /// Hand index of a combo, if it's in the acting player's range.
    pub fn hand_index(&self, combo: usize) -> Option<usize> {
        self.combos.iter().position(|&c| c as usize == combo)
    }

    pub fn hand_strategy(&self, i: usize) -> Vec<f32> {
        let n = self.num_hands();
        (0..self.actions.len()).map(|a| self.strategy[a * n + i]).collect()
    }

    pub fn hand_ev(&self, i: usize) -> Vec<f32> {
        let n = self.num_hands();
        (0..self.actions.len()).map(|a| self.ev[a * n + i]).collect()
    }

    /// Range-wide action frequencies (weighted by reach).
    pub fn range_strategy(&self) -> Vec<f64> {
        let n = self.num_hands();
        let total: f64 = self.reach.iter().map(|&w| w as f64).sum();
        (0..self.actions.len())
            .map(|a| {
                if total <= 0.0 {
                    return 0.0;
                }
                (0..n).map(|i| self.reach[i] as f64 * self.strategy[a * n + i] as f64).sum::<f64>() / total
            })
            .collect()
    }
}

/// Position reached by following a path, in the solver's canonical
/// (isomorphism-reduced) hand space.
struct Walk {
    node: usize,
    /// Reach of both players, indexed by canonical hand.
    reach: [Vec<f32>; 2],
    /// Real hand index -> canonical hand index.
    to_canon: [Vec<u32>; 2],
}

/// A built (and possibly solved) game.
pub struct Solver {
    config: SolveConfig,
    hands: HandSet,
    iso: Isomorphism,
    tree: Tree,
    store: Storage,
    tables: Vec<StrengthTable>,
    stats: TreeStats,
    iterations: u32,
    exploitability: f64,
    batch_river: bool,
    /// The GPU engine after a successful `use_gpu`.
    #[cfg(feature = "gpu")]
    gpu: Option<Box<gpu::Engine>>,
    /// Why the last solve didn't use the GPU, or abandoned it.
    gpu_error: Option<String>,
    /// Where the last solve ran.
    solved_on: SolvedOn,
}

impl Solver {
    fn validate(config: &SolveConfig) -> Result<HandSet, SolverError> {
        let n = config.board.len();
        if !(3..=5).contains(&n) {
            return Err(SolverError::InvalidBoard(format!("{n} cards")));
        }
        let set = CardSet::from_cards(&config.board);
        if set.len() != n {
            return Err(SolverError::InvalidBoard("duplicate cards".into()));
        }
        if Street::from_board_len(n) != config.root.street {
            return Err(SolverError::InvalidBoard("board doesn't match the root street".into()));
        }
        let hands = HandSet::new(&config.ranges, set, config.prune, &config.keep);
        for p in 0..2 {
            if hands.is_empty(p) {
                return Err(SolverError::EmptyRange(p));
            }
        }
        if hands.num_combinations <= 0.0 {
            return Err(SolverError::EmptyRange(0));
        }
        Ok(hands)
    }

    fn params<'a>(config: &'a SolveConfig, hands: &HandSet, iso: &'a Isomorphism) -> TreeParams<'a> {
        TreeParams {
            root_street: config.root.street,
            root_board: &config.board,
            menu: &config.menu,
            future_menu: config.future_menu.as_ref().unwrap_or(&config.menu),
            big_blind: config.big_blind,
            chance: config.chance,
            num_hands: [hands.len(0), hands.len(1)],
            rake_rate: config.rake_rate,
            rake_cap: config.rake_cap,
            iso,
        }
    }

    fn compressed(config: &SolveConfig, stats: &TreeStats) -> bool {
        match config.storage {
            StorageMode::F32 => false,
            StorageMode::Compressed => true,
            StorageMode::Auto => stats.memory_bytes_f32() > 1 << 30,
        }
    }

    /// Counts the tree size (and the memory it would need) without building it.
    pub fn estimate(config: &SolveConfig) -> Result<TreeStats, SolverError> {
        let hands = Self::validate(config)?;
        let iso = Isomorphism::new(&hands, config.isomorphism);
        let params = Self::params(config, &hands, &iso);
        let mut stats = tree::count_tree(&params, &config.root);
        stats.compressed = Self::compressed(config, &stats);
        Ok(stats)
    }

    /// Builds the game tree and the showdown tables.
    pub fn new(config: SolveConfig) -> Result<Solver, SolverError> {
        let hands = Self::validate(&config)?;
        let iso = Isomorphism::new(&hands, config.isomorphism);
        let (tree, stats) = {
            let params = Self::params(&config, &hands, &iso);
            let mut stats = tree::count_tree(&params, &config.root);
            stats.compressed = Self::compressed(&config, &stats);
            let needed = stats.memory_bytes();
            if needed > config.max_memory_bytes {
                return Err(SolverError::TooLarge { needed, limit: config.max_memory_bytes });
            }
            (Builder::new(&params).build(&config.root), stats)
        };
        let store = Storage::new(tree.storage_len as usize, tree.num_player_nodes as usize, stats.compressed);
        let tables: Vec<StrengthTable> =
            tree.boards.par_iter().map(|b| StrengthTable::new(&hands, b)).collect();
        Ok(Solver {
            config,
            hands,
            iso,
            tree,
            store,
            tables,
            stats,
            iterations: 0,
            exploitability: f64::INFINITY,
            batch_river: true,
            #[cfg(feature = "gpu")]
            gpu: None,
            gpu_error: None,
            solved_on: SolvedOn::Cpu,
        })
    }

    pub fn config(&self) -> &SolveConfig {
        &self.config
    }

    pub fn stats(&self) -> TreeStats {
        self.stats
    }

    /// Bytes actually allocated for regrets and strategies.
    pub fn storage_bytes(&self) -> u64 {
        self.store.bytes()
    }

    pub fn is_compressed(&self) -> bool {
        self.store.is_compressed()
    }

    /// Number of suit pairs treated as isomorphic.
    pub fn isomorphic_pairs(&self) -> usize {
        self.iso.range_symmetric.iter().filter(|&&x| x).count()
    }

    pub fn hands(&self) -> &HandSet {
        &self.hands
    }

    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    pub fn root_pot(&self) -> Chips {
        self.config.root.pot
    }

    /// Exploitability (chips) at the last check.
    pub fn last_exploitability(&self) -> f64 {
        self.exploitability
    }

    /// Switches the breadth-first river evaluator off (for testing: both
    /// paths must give the same results).
    #[doc(hidden)]
    pub fn set_batch_river(&mut self, on: bool) {
        self.batch_river = on;
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx {
            tree: &self.tree,
            store: &self.store,
            hands: &self.hands,
            tables: &self.tables,
            iso: &self.iso,
            half_pot: self.config.root.pot as f64 / 2.0,
            norm: 1.0 / self.hands.num_combinations,
            batch_river: self.batch_river,
        }
    }

    fn raked(&self) -> bool {
        self.config.rake_rate > 0.0 && self.config.rake_cap > 0.0
    }

    /// Exploitability of the current average strategy, in chips.
    pub fn exploitability(&self) -> f64 {
        cfr::exploitability(&self.ctx(), self.raked())
    }

    /// Expected payoff (chips, relative to the root with the pot split
    /// evenly) of each player under the average strategies.
    pub fn root_values(&self) -> [f64; 2] {
        cfr::root_values(&self.ctx(), Mode::Average)
    }

    /// Value of each of player `p`'s hands (in `hands().combos[p]` order)
    /// under the average strategies, in chips: what the hand gets back from
    /// the pot minus what it puts in after the root, against the opponent's
    /// root range with card removal. A hand that folds at once is worth 0;
    /// one that no opponent hand is compatible with is worth 0.
    pub fn hand_values(&self, p: usize) -> Vec<f32> {
        assert!(p < 2, "heads-up: player 0 or 1");
        let ctx = self.ctx();
        let opp = &self.hands.weights[p ^ 1];
        let compat = self.hands.compatible_reach(p, opp);
        let mut v = vec![0f32; self.hands.len(p)];
        cfr::traverse(&ctx, 0, p, opp, Mode::Average, &mut v);
        v.iter()
            .zip(&compat)
            .map(|(&x, &c)| {
                let denom = c * ctx.norm;
                if denom > 0.0 {
                    (x as f64 / denom + ctx.half_pot) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Runs DCFR iterations until the target exploitability or iteration cap
    /// is reached. `on_progress` is called after every exploitability check;
    /// returning `false` cancels the solve.
    pub fn solve(&mut self, opts: &SolveOptions, on_progress: impl FnMut(&Progress) -> bool) -> Progress {
        self.solve_cancellable(opts, &std::sync::atomic::AtomicBool::new(false), on_progress)
    }

    /// Like [`Self::solve`], but stops within one iteration once `cancel` is
    /// set (`on_progress` only runs at exploitability checks, which can be
    /// many seconds apart on big trees).
    pub fn solve_cancellable(
        &mut self,
        opts: &SolveOptions,
        cancel: &std::sync::atomic::AtomicBool,
        mut on_progress: impl FnMut(&Progress) -> bool,
    ) -> Progress {
        let borrowed = self.borrow_gpu();
        let progress = self.solve_on_current_backend(opts, cancel, &mut on_progress);
        if borrowed {
            self.release_gpu();
        }
        progress
    }

    /// Whether `config.backend` asks for the GPU for this tree.
    fn wants_gpu(&self) -> bool {
        match self.config.backend {
            Backend::Cpu => false,
            Backend::Gpu => true,
            Backend::Auto => self.config.root.street != Street::River,
        }
    }

    /// Takes the GPU for one solve if `config.backend` asks for it and no
    /// engine is in place yet; true if it did. A GPU that can't be used is
    /// recorded in [`Self::gpu_error`] and the solve stays on the CPU.
    fn borrow_gpu(&mut self) -> bool {
        if !self.wants_gpu() {
            return false;
        }
        #[cfg(feature = "gpu")]
        {
            if self.gpu.is_some() {
                return false;
            }
            match self.use_gpu(gpu::GpuPrecision::default()) {
                Ok(()) => true,
                Err(e) => {
                    self.gpu_error = Some(e.to_string());
                    false
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            self.gpu_error = Some("this build has no GPU support (the solver's `gpu` feature)".into());
            false
        }
    }

    /// Frees the GPU (its memory and context). Regrets and strategy sums
    /// are on the host after every solve, so nothing is lost; a later
    /// `use_gpu` or GPU-backed solve uploads them again.
    pub fn release_gpu(&mut self) {
        #[cfg(feature = "gpu")]
        {
            self.gpu = None;
        }
    }

    fn solve_on_current_backend(
        &mut self,
        opts: &SolveOptions,
        cancel: &std::sync::atomic::AtomicBool,
        mut on_progress: impl FnMut(&Progress) -> bool,
    ) -> Progress {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            let result = self
                .run_solve(
                    opts,
                    cancel,
                    &mut on_progress,
                    &mut Self::gpu_iteration,
                    &mut Self::gpu_exploitability,
                )
                .and_then(|p| self.download_from_gpu().map(|()| p));
            match result {
                Ok(p) => {
                    if let Some(engine) = &self.gpu {
                        self.solved_on = match engine.precision() {
                            gpu::GpuPrecision::Fast => SolvedOn::GpuFast,
                            gpu::GpuPrecision::Exact => SolvedOn::GpuExact,
                        };
                    }
                    return p;
                }
                Err(e) => self.fall_back_to_cpu(e),
            }
        }
        self.solved_on = SolvedOn::Cpu;
        match self.run_solve(
            opts,
            cancel,
            &mut on_progress,
            &mut Self::cpu_iteration,
            &mut Self::cpu_exploitability,
        ) {
            Ok(p) => p,
            Err(never) => match never {},
        }
    }

    /// One DCFR iteration on the CPU: player 0's update, then player 1's.
    fn cpu_iteration(&mut self) -> Result<(), std::convert::Infallible> {
        let d = Discount::new(self.iterations, &self.config.dcfr);
        let ctx = self.ctx();
        for p in 0..2 {
            let mut out = vec![0f32; self.hands.len(p)];
            cfr::traverse(&ctx, 0, p, &self.hands.weights[p ^ 1], Mode::Cfr(d), &mut out);
        }
        Ok(())
    }

    fn cpu_exploitability(&mut self) -> Result<f64, std::convert::Infallible> {
        Ok(cfr::exploitability(&self.ctx(), self.raked()))
    }

    /// The solve loop shared by the backends: `iterate` runs iteration
    /// `self.iterations` (already incremented), `measure` returns the
    /// exploitability of the current average strategy.
    fn run_solve<E>(
        &mut self,
        opts: &SolveOptions,
        cancel: &std::sync::atomic::AtomicBool,
        on_progress: &mut dyn FnMut(&Progress) -> bool,
        iterate: &mut dyn FnMut(&mut Solver) -> Result<(), E>,
        measure: &mut dyn FnMut(&mut Solver) -> Result<f64, E>,
    ) -> Result<Progress, E> {
        let pot = self.config.root.pot as f64;
        let check_every = opts.check_every.max(1);
        let mut progress = Progress {
            iteration: self.iterations,
            max_iterations: opts.max_iterations,
            exploitability: self.exploitability,
            exploitability_pct: 100.0 * self.exploitability / pot,
            finished: false,
        };
        let target = opts.target_exploitability_pct * pot / 100.0;
        let mut next_check = self.iterations + check_every;
        let mut last_check: Option<(u32, f64)> = None;
        while self.iterations < opts.max_iterations {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                progress.finished = true;
                return Ok(progress);
            }
            self.iterations += 1;
            iterate(self)?;
            let last = self.iterations >= opts.max_iterations;
            if self.iterations >= next_check || last {
                let e = measure(self)?;
                self.exploitability = e;
                progress.iteration = self.iterations;
                progress.exploitability = e;
                progress.exploitability_pct = 100.0 * e / pot;
                let done = progress.exploitability_pct <= opts.target_exploitability_pct || last;
                progress.finished = done;
                let keep_going = on_progress(&progress);
                if done || !keep_going {
                    progress.finished = true;
                    return Ok(progress);
                }
                let t = self.iterations;
                next_check = t + next_check_gap(last_check, (t, e), target, check_every);
                last_check = Some((t, e));
            }
        }
        progress.finished = true;
        Ok(progress)
    }

    /// Moves the solve to the GPU until [`Self::release_gpu`] (experimental;
    /// `config.backend` does this for the duration of each solve instead).
    /// Trees whose scratch buffers don't fit in free GPU memory run in groups
    /// of subtrees; one that doesn't fit even so is declined. The current
    /// regrets and strategy sums are uploaded, so a solve can continue on the
    /// GPU; results are copied back when `solve` returns, after which `query`
    /// and the other read methods work as usual. If the GPU fails during a
    /// solve, the solver discards the GPU state, restarts the solve from
    /// scratch on the CPU, and [`Self::gpu_error`] says what happened.
    /// `precision` picks how terminal sums are accumulated
    /// ([`gpu::GpuPrecision`]).
    #[cfg(feature = "gpu")]
    pub fn use_gpu(&mut self, precision: gpu::GpuPrecision) -> Result<(), gpu::GpuError> {
        let device = gpu::Gpu::new()?;
        let budget = gpu::Engine::scratch_budget(&device, &self.store.gpu_view())?;
        self.use_gpu_with(device, precision, gpu::Scratch::Budget(budget))
    }

    /// [`Solver::use_gpu`] with the tree split as `scratch` says (trees too
    /// big for the device's memory run in groups of subtrees).
    #[cfg(feature = "gpu")]
    pub(crate) fn use_gpu_with(
        &mut self,
        device: gpu::Gpu,
        precision: gpu::GpuPrecision,
        scratch: gpu::Scratch,
    ) -> Result<(), gpu::GpuError> {
        let plan = gpu::Plan::build_with(&self.ctx(), scratch)?;
        self.gpu = Some(Box::new(gpu::Engine::new(device, plan, self.store.gpu_view(), precision)?));
        self.gpu_error = None;
        Ok(())
    }

    /// Whether solves currently run on the GPU (after [`Self::use_gpu`];
    /// GPU-backed solves hold it only while they run).
    pub fn uses_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Where the last solve ran (the CPU before any solve).
    pub fn solved_on(&self) -> SolvedOn {
        self.solved_on
    }

    /// Why the GPU wasn't used (`config.backend`) or was abandoned during a
    /// solve, if so.
    pub fn gpu_error(&self) -> Option<&str> {
        self.gpu_error.as_deref()
    }

    #[cfg(feature = "gpu")]
    fn gpu_engine(&mut self) -> Result<&mut gpu::Engine, gpu::GpuError> {
        self.gpu.as_deref_mut().ok_or_else(|| gpu::GpuError::InvalidInput("no GPU engine".into()))
    }

    #[cfg(feature = "gpu")]
    fn gpu_iteration(&mut self) -> Result<(), gpu::GpuError> {
        let d = Discount::new(self.iterations, &self.config.dcfr);
        self.gpu_engine()?.cfr_iteration(d)
    }

    #[cfg(feature = "gpu")]
    fn gpu_exploitability(&mut self) -> Result<f64, gpu::GpuError> {
        let raked = self.raked();
        self.gpu_engine()?.exploitability(raked)
    }

    /// Copies the GPU's regrets and strategy sums into host storage.
    #[cfg(feature = "gpu")]
    fn download_from_gpu(&mut self) -> Result<(), gpu::GpuError> {
        let Some(engine) = &self.gpu else { return Ok(()) };
        engine.download(self.store.gpu_view())
    }

    /// After a GPU failure: drops the GPU state and resets the solve, so it
    /// restarts on the CPU from a clean slate.
    #[cfg(feature = "gpu")]
    fn fall_back_to_cpu(&mut self, e: gpu::GpuError) {
        self.gpu = None;
        self.gpu_error = Some(e.to_string());
        self.store = Storage::new(
            self.tree.storage_len as usize,
            self.tree.num_player_nodes as usize,
            self.stats.compressed,
        );
        self.iterations = 0;
        self.exploitability = f64::INFINITY;
    }

    fn average_strategy(&self, id: usize, n: usize) -> Vec<f32> {
        let node = &self.tree.nodes[id];
        let mut s = vec![0f32; node.num_actions() * n];
        if n > 0 {
            unsafe { self.store.load_strat(node.block(n), &mut s) };
            cfr::normalize_in_place(&mut s, node.num_actions(), n);
        }
        s
    }

    /// Follows `path` from the root.
    fn walk(&self, path: &[Step]) -> Result<Walk, SolverError> {
        let mut id = 0usize;
        let mut reach = self.hands.weights.clone();
        let mut to_canon: [Vec<u32>; 2] =
            [(0..self.hands.len(0) as u32).collect(), (0..self.hands.len(1) as u32).collect()];
        // Real suit -> canonical suit.
        let mut suit_map = [0u8, 1, 2, 3];
        for (depth, step) in path.iter().enumerate() {
            let node = self.tree.nodes[id];
            match (node.kind, step) {
                (Kind::Player(q), Step::Action(a)) => {
                    let actions = self.tree.actions(id);
                    let ai = actions.iter().position(|x| x == a).ok_or_else(|| {
                        SolverError::BadPath(format!(
                            "action {a:?} not available at depth {depth} (have {actions:?})"
                        ))
                    })?;
                    let q = q as usize;
                    let n = self.hands.len(q);
                    let s = self.average_strategy(id, n);
                    for i in 0..n {
                        reach[q][i] *= s[ai * n + i];
                    }
                    id = node.first_child as usize + ai;
                }
                (Kind::Chance, Step::Card(c)) => {
                    let cc = Card::new(c.rank(), suit_map[c.suit() as usize]);
                    let children = node.children();
                    let canonical = match children.clone().find(|&ch| self.tree.card(ch) == cc) {
                        Some(ch) => {
                            id = ch;
                            cc
                        }
                        None => {
                            let info = &self.tree.chance[node.slot as usize];
                            let j = info.iso_cards.iter().position(|&x| x == cc).ok_or_else(|| {
                                SolverError::BadPath(format!("card {c} not in the tree at depth {depth}"))
                            })?;
                            let (ci, pair) = info.iso[j];
                            let (a, b) = PAIRS[pair as usize];
                            for s in suit_map.iter_mut() {
                                *s = if *s == a {
                                    b
                                } else if *s == b {
                                    a
                                } else {
                                    *s
                                };
                            }
                            for x in 0..2 {
                                let perm = &self.iso.perm[x][pair as usize];
                                let mut r2 = vec![0f32; reach[x].len()];
                                for (h, &w) in reach[x].iter().enumerate() {
                                    r2[perm[h] as usize] = w;
                                }
                                reach[x] = r2;
                                for t in to_canon[x].iter_mut() {
                                    *t = perm[*t as usize] as u32;
                                }
                            }
                            id = children.start + ci as usize;
                            self.tree.card(id)
                        }
                    };
                    for x in 0..2 {
                        for &k in &self.hands.blocked[x][canonical.index()] {
                            reach[x][k as usize] = 0.0;
                        }
                    }
                }
                _ => {
                    return Err(SolverError::BadPath(format!(
                        "step {step:?} doesn't match node kind {:?} at depth {depth}",
                        node.kind
                    )))
                }
            }
        }
        Ok(Walk { node: id, reach, to_canon })
    }

    /// Whether `path` leads to a decision node.
    pub fn is_decision(&self, path: &[Step]) -> bool {
        matches!(self.walk(path), Ok(w) if matches!(self.tree.nodes[w.node].kind, Kind::Player(_)))
    }

    /// Actions available after `path` (empty for non-decision nodes).
    pub fn actions_at(&self, path: &[Step]) -> Result<Vec<Action>, SolverError> {
        let w = self.walk(path)?;
        Ok(match self.tree.nodes[w.node].kind {
            Kind::Player(_) => self.tree.actions(w.node),
            _ => Vec::new(),
        })
    }

    /// Strategy and per-action EVs at the decision node reached by `path`.
    pub fn query(&self, path: &[Step]) -> Result<NodeInfo, SolverError> {
        let w = self.walk(path)?;
        let node = self.tree.nodes[w.node];
        let Kind::Player(q) = node.kind else {
            return Err(SolverError::BadPath("not a decision node".into()));
        };
        let q = q as usize;
        let o = q ^ 1;
        let n = self.hands.len(q);
        let na = node.num_actions();
        let ctx = self.ctx();
        let compat = self.hands.compatible_reach(q, &w.reach[o]);
        let child_vals: Vec<Vec<f32>> = node
            .children()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&c| {
                let mut v = vec![0f32; n];
                cfr::traverse(&ctx, c, q, &w.reach[o], Mode::Average, &mut v);
                v
            })
            .collect();
        let offset = ctx.half_pot + node.invest[q] as f64;
        let strat_c = self.average_strategy(w.node, n);
        let mut ev = vec![0f32; na * n];
        let mut strategy = vec![0f32; na * n];
        let mut reach_real = vec![0f32; n];
        for h in 0..n {
            let hc = w.to_canon[q][h] as usize;
            reach_real[h] = w.reach[q][hc];
            let denom = compat[hc] * ctx.norm;
            for a in 0..na {
                strategy[a * n + h] = strat_c[a * n + hc];
                ev[a * n + h] =
                    if denom > 0.0 { (child_vals[a][hc] as f64 / denom + offset) as f32 } else { 0.0 };
            }
        }
        let frac = |p: usize| {
            let root: f64 = self.hands.weights[p].iter().map(|&x| x as f64).sum();
            let now: f64 = w.reach[p].iter().map(|&x| x as f64).sum();
            if root > 0.0 {
                now / root
            } else {
                0.0
            }
        };
        Ok(NodeInfo {
            player: q,
            actions: self.tree.actions(w.node),
            combos: self.hands.combos[q].clone(),
            strategy,
            ev,
            reach: reach_real,
            reach_fraction: [frac(0), frac(1)],
            pot: self.config.root.pot + node.invest[0] as Chips + node.invest[1] as Chips,
            invest: [node.invest[0] as Chips, node.invest[1] as Chips],
        })
    }

    /// Both players' range weights (1326 entries each) after following
    /// `path`: root weights times the probability of every action taken.
    pub fn ranges_after(&self, path: &[Step]) -> Result<[Vec<f32>; 2], SolverError> {
        let w = self.walk(path)?;
        let mut out = [vec![0f32; 1326], vec![0f32; 1326]];
        for p in 0..2 {
            for (h, &c) in self.hands.combos[p].iter().enumerate() {
                out[p][c as usize] = w.reach[p][w.to_canon[p][h] as usize];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;
