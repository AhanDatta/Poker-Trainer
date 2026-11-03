//! Three-player postflop solver for multiway pots.
//!
//! DCFR on the three-player game, with these approximations:
//! * callers build trees with sampled runouts and simplified later streets
//!   (a 3-way flop with every runout needs far more memory than a desktop
//!   has; turns and rivers can be exact);
//! * optionally, the two opponents of a hand are treated as independent
//!   (`exact_removal: false`): card removal between the traverser and each
//!   opponent is exact, but the opponents' hands aren't checked against
//!   each other, so showdowns factor into per-opponent sweeps. With
//!   `exact_removal` terminal values sum over disjoint hand pairs exactly
//!   (see `exact`).
//!
//! CFR has no equilibrium guarantee with three players; in practice (as in
//! Pluribus) it produces sensible strategies. Convergence is reported as
//! the average best-response gain (NashConv / 3).
//!
//! A pot that loses a player is still a three-player game here (the folded
//! player's money is dead, its cards still block), so the tree handles 2 or
//! 3 active players.
//!
//! Layout (as in the heads-up solver): a flat node arena, regrets and
//! strategy sums in shared storage (16-bit compressed for big trees), and
//! pooled scratch buffers. Each player's average strategy accumulates its
//! current strategy (unweighted) at its own nodes, and a traversal skips the
//! subtrees after the traverser folded: its payoff there is fixed.

mod cfr;
mod exact;
mod hands;
mod river;
mod tree;

use crate::cfr::{normalize_in_place, Discount, Mode};
use crate::storage::{Storage, StorageMode};
use crate::{ChanceMode, DcfrParams, Progress, SolveOptions, SolverError, Step};
use cfr::{norms, traverse, Ctx};
use hands::{Hands, Table};
use holdem::combo::NUM_COMBOS;
use holdem::{Action, Card, CardSet, Chips, PostflopMenu, Situation, Street};
use rayon::prelude::*;
use tree::{Edge, Kind, Tree};

/// Number of players.
pub const N: usize = 3;
const NONE: u32 = u32::MAX;

/// Betting state of a three-player street (player 0 acts first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiState {
    pub street: Street,
    pub pot: Chips,
    pub commit: [Chips; N],
    pub stack: [Chips; N],
    pub folded: [bool; N],
    pub current_bet: Chips,
    pub last_raise_inc: Chips,
    pub num_raises: u32,
    pub to_act: usize,
    pub needs: [bool; N],
}

/// What an action leads to in a three-player street.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// The street continues with this state.
    Continue(MultiState),
    /// Everyone else folded: the given player wins.
    Won(usize),
    /// The street's betting is over.
    StreetOver(MultiState),
}

impl MultiState {
    /// State at the start of a street: first active player with chips acts.
    pub fn street_start(
        street: Street,
        pot: Chips,
        stack: [Chips; N],
        folded: [bool; N],
        big_blind: Chips,
    ) -> MultiState {
        let needs = [0, 1, 2].map(|i| !folded[i] && stack[i] > 0);
        MultiState {
            street,
            pot,
            commit: [0; N],
            stack,
            folded,
            current_bet: 0,
            last_raise_inc: big_blind,
            num_raises: 0,
            to_act: (0..N).find(|&i| needs[i]).unwrap_or(0),
            needs,
        }
    }

    pub fn active(&self) -> usize {
        self.folded.iter().filter(|&&f| !f).count()
    }

    pub fn situation(&self, big_blind: Chips) -> Situation {
        let p = self.to_act;
        let opp_max = (0..N)
            .filter(|&q| q != p && !self.folded[q])
            .map(|q| self.commit[q] + self.stack[q])
            .max()
            .unwrap_or(0);
        Situation {
            street: self.street,
            pot: self.pot,
            to_call: (self.current_bet - self.commit[p]).max(0).min(self.stack[p]),
            current_bet: self.current_bet,
            max_to: (self.commit[p] + self.stack[p]).min(opp_max),
            min_raise_to: self.current_bet + self.last_raise_inc.max(big_blind),
            num_raises: self.num_raises,
            limpers: 0,
            big_blind,
        }
    }

    /// The state after `action` when the betting round continues.
    pub fn after(&self, action: Action) -> Option<MultiState> {
        match self.apply(action).0 {
            Next::Continue(s) => Some(s),
            _ => None,
        }
    }

    /// Where `action` leads.
    pub fn transition(&self, action: Action) -> Next {
        self.apply(action).0
    }

    fn apply(&self, action: Action) -> (Next, Chips) {
        let p = self.to_act;
        let mut s = *self;
        let mut put = 0;
        match action {
            Action::Fold => s.folded[p] = true,
            Action::Check => {}
            Action::Call => put = (s.current_bet - s.commit[p]).max(0).min(s.stack[p]),
            Action::Raise(to) => {
                put = to - s.commit[p];
                let inc = to - s.current_bet;
                if inc >= s.last_raise_inc {
                    s.last_raise_inc = inc;
                }
                s.current_bet = to;
                s.num_raises += 1;
                for q in 0..N {
                    if q != p && !s.folded[q] && s.stack[q] > 0 {
                        s.needs[q] = true;
                    }
                }
            }
        }
        s.commit[p] += put;
        s.stack[p] -= put;
        s.pot += put;
        s.needs[p] = false;
        if s.active() == 1 {
            let w = (0..N).find(|&q| !s.folded[q]).unwrap();
            return (Next::Won(w), put);
        }
        match (1..N).map(|k| (p + k) % N).find(|&q| s.needs[q] && !s.folded[q]) {
            Some(q) => {
                s.to_act = q;
                (Next::Continue(s), put)
            }
            None => (Next::StreetOver(s), put),
        }
    }
}

/// Inputs of a three-player solve (players in postflop order).
#[derive(Clone, Debug)]
pub struct MultiConfig {
    pub board: Vec<Card>,
    pub ranges: [Vec<f32>; N],
    pub root: MultiState,
    pub menu: PostflopMenu,
    pub future_menu: Option<PostflopMenu>,
    pub big_blind: Chips,
    pub chance: ChanceMode,
    pub prune: f32,
    pub keep: [Vec<usize>; N],
    pub max_memory_bytes: u64,
    pub rake_rate: f64,
    pub rake_cap: f64,
    /// Exact card removal between the two opponents; off, their hands are
    /// treated as independent.
    pub exact_removal: bool,
    /// 16-bit compressed regrets and strategies (`Auto`: above 1 GB).
    pub storage: StorageMode,
    pub dcfr: DcfrParams,
}

/// DCFR scheme for three players: reach-weighted averages without restarts
/// (restarts left the average stuck or bouncing in 3-player tests; with the
/// heads-up scheme's unweighted, restarted averages a narrow-range turn
/// ended at 0.3% of the pot instead of 0.08% after 300 iterations).
pub fn multiway_dcfr() -> DcfrParams {
    DcfrParams { alpha: 1.5, beta: Some(0.0), gamma: 3.0, restart_average: false, unweighted_average: false }
}

/// The two opponents of `p` (in seat order).
pub(super) fn opponents(p: usize) -> (usize, usize) {
    match p {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Strategy and values at a decision node of a three-player solve.
pub type MultiNodeInfo = crate::NodeInfo;

pub struct MultiSolver {
    config: MultiConfig,
    hands: Hands,
    tables: Vec<Table>,
    tree: Tree,
    store: Storage,
    norm: [f64; N],
    iterations: u32,
    convergence: f64,
    batch_river: bool,
}

impl MultiSolver {
    pub fn new(config: MultiConfig) -> Result<MultiSolver, SolverError> {
        let n = config.board.len();
        if !(3..=5).contains(&n) || Street::from_board_len(n) != config.root.street {
            return Err(SolverError::InvalidBoard(format!("{n} cards for {:?}", config.root.street)));
        }
        let hands =
            Hands::new(&config.ranges, CardSet::from_cards(&config.board), config.prune, &config.keep);
        for p in 0..N {
            if hands.len(p) == 0 {
                return Err(SolverError::EmptyRange(p));
            }
        }
        // Stop building as soon as even compressed storage can't fit.
        let min_bytes = Storage::bytes_per_value(config.storage != StorageMode::F32);
        let tree = Tree::build(&config, &hands, config.max_memory_bytes / min_bytes)
            .map_err(|_| too_large(&config, u64::MAX))?;
        let compressed = match config.storage {
            StorageMode::F32 => false,
            StorageMode::Compressed => true,
            StorageMode::Auto => tree.storage_len * Storage::bytes_per_value(false) > 1 << 30,
        };
        let needed = tree.storage_len * Storage::bytes_per_value(compressed)
            + (tree.nodes.len() * std::mem::size_of::<tree::Node>()) as u64;
        if needed > config.max_memory_bytes {
            return Err(too_large(&config, needed));
        }
        let tables = tree.boards.par_iter().map(|bd| Table::new(&hands, bd)).collect();
        let store = Storage::new(tree.storage_len as usize, tree.num_player_nodes as usize, compressed);
        let norm = norms(&hands, config.exact_removal);
        Ok(MultiSolver {
            config,
            hands,
            tables,
            tree,
            store,
            norm,
            iterations: 0,
            convergence: f64::INFINITY,
            batch_river: true,
        })
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx {
            tree: &self.tree,
            store: &self.store,
            hands: &self.hands,
            tables: &self.tables,
            third_pot: self.config.root.pot as f64 / N as f64,
            root_pot: self.config.root.pot as f64,
            norm: self.norm,
            exact: self.config.exact_removal,
            // Batching pays off with the exact (expensive) showdowns; with
            // independent opponents the recursive traversal is faster.
            batch_river: self.batch_river && self.config.exact_removal,
        }
    }

    /// Switches the breadth-first river evaluator off (for testing: both
    /// paths must give the same results).
    #[doc(hidden)]
    pub fn set_batch_river(&mut self, on: bool) {
        self.batch_river = on;
    }

    fn root_values(&self, mode: Mode) -> [f64; N] {
        let ctx = self.ctx();
        std::array::from_fn(|p| {
            let (q, r) = opponents(p);
            let mut out = vec![0f32; self.hands.len(p)];
            let w = &self.hands.weights;
            traverse(&ctx, 0, p, &w[q], &w[r], mode, false, &mut out);
            out.iter().zip(&w[p]).map(|(&x, &w)| x as f64 * w as f64).sum()
        })
    }

    /// Average best-response gain over the three players (chips).
    pub fn convergence(&self) -> f64 {
        let br = self.root_values(Mode::BestResponse);
        let ev = self.root_values(Mode::Average);
        (0..N).map(|p| br[p] - ev[p]).sum::<f64>() / N as f64
    }

    /// Expected payoff of each player (chips, relative to an even split of
    /// the root pot) under the average strategies.
    pub fn values(&self) -> [f64; N] {
        self.root_values(Mode::Average)
    }

    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Bytes of regret and strategy storage.
    pub fn storage_bytes(&self) -> u64 {
        self.store.bytes()
    }

    pub fn is_compressed(&self) -> bool {
        self.store.is_compressed()
    }

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
        let pot = self.config.root.pot as f64;
        let check = opts.check_every.max(1);
        let target = opts.target_exploitability_pct * pot / 100.0;
        let mut next_check = self.iterations + check;
        let mut last_check: Option<(u32, f64)> = None;
        let mut progress = Progress {
            iteration: self.iterations,
            max_iterations: opts.max_iterations,
            exploitability: self.convergence,
            exploitability_pct: 100.0 * self.convergence / pot,
            finished: false,
        };
        while self.iterations < opts.max_iterations {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                progress.finished = true;
                return progress;
            }
            self.iterations += 1;
            let mut d = Discount::new(self.iterations, &self.config.dcfr);
            // Reach-weighted sums can freeze in 16-bit storage (increments
            // below half a quantization step round away): add unweighted
            // strategies there, as the heads-up solver does.
            d.reach_weighted &= !self.store.is_compressed();
            let ctx = self.ctx();
            for p in 0..N {
                let (q, r) = opponents(p);
                let mut out = vec![0f32; self.hands.len(p)];
                let w = &self.hands.weights;
                traverse(&ctx, 0, p, &w[q], &w[r], Mode::Cfr(d), false, &mut out);
            }
            let last = self.iterations >= opts.max_iterations;
            if self.iterations >= next_check || last {
                self.convergence = self.convergence();
                progress.iteration = self.iterations;
                progress.exploitability = self.convergence;
                progress.exploitability_pct = 100.0 * self.convergence / pot;
                let done = progress.exploitability_pct <= opts.target_exploitability_pct || last;
                progress.finished = done;
                if !on_progress(&progress) || done {
                    progress.finished = true;
                    return progress;
                }
                let (t, e) = (self.iterations, self.convergence);
                next_check = t + crate::next_check_gap(last_check, (t, e), target, check);
                last_check = Some((t, e));
            }
        }
        progress.finished = true;
        progress
    }

    /// Average strategy of a player node (action-major).
    fn average_strategy(&self, id: usize) -> Vec<f32> {
        let node = &self.tree.nodes[id];
        let Kind::Player(x) = node.kind else { return Vec::new() };
        let n = self.hands.len(x as usize);
        let mut s = vec![0f32; node.num_actions() * n];
        unsafe { self.store.load_strat(node.block(n), &mut s) };
        normalize_in_place(&mut s, node.num_actions(), n);
        s
    }

    /// Follows `path` (actions within the root street, or dealt cards).
    fn walk(&self, path: &[Step]) -> Result<(usize, [Vec<f32>; N]), SolverError> {
        let mut id = 0;
        let mut reach = self.hands.weights.clone();
        for (depth, step) in path.iter().enumerate() {
            let node = &self.tree.nodes[id];
            let want = match step {
                Step::Action(a) => Edge::Action(*a),
                Step::Card(c) => Edge::Card(*c),
            };
            let child = node.children().find(|&c| self.tree.nodes[c].edge == want);
            match (node.kind, step, child) {
                (Kind::Player(x), Step::Action(_), Some(c)) => {
                    let x = x as usize;
                    let n = self.hands.len(x);
                    let ai = c - node.first_child as usize;
                    let s = self.average_strategy(id);
                    for h in 0..n {
                        reach[x][h] *= s[ai * n + h];
                    }
                    id = c;
                }
                (Kind::Chance, Step::Card(card), Some(c)) => {
                    for (x, r) in reach.iter_mut().enumerate() {
                        for &h in &self.hands.blocked[x][card.index()] {
                            r[h as usize] = 0.0;
                        }
                    }
                    id = c;
                }
                (Kind::Player(_), Step::Action(a), None) => {
                    return Err(SolverError::BadPath(format!(
                        "action {a:?} not in the tree at depth {depth} ({:?})",
                        self.tree.actions(id)
                    )))
                }
                (Kind::Chance, Step::Card(c), None) => {
                    return Err(SolverError::BadPath(format!("card {c} not in the tree at depth {depth}")))
                }
                _ => {
                    return Err(SolverError::BadPath(format!(
                        "step {step:?} doesn't fit the tree at depth {depth}"
                    )))
                }
            }
        }
        Ok((id, reach))
    }

    /// Strategy and per-action EVs (chips from the decision on) at a node.
    /// `reach_fraction` holds the acting player's and the least-reached
    /// opponent's share of their root range; `invest[0]` is the acting
    /// player's investment since the root.
    pub fn query(&self, path: &[Step]) -> Result<MultiNodeInfo, SolverError> {
        let ctx = self.ctx();
        let (id, reach) = self.walk(path)?;
        let node = &self.tree.nodes[id];
        let Kind::Player(x) = node.kind else {
            return Err(SolverError::BadPath("not a decision node".into()));
        };
        let x = x as usize;
        let n = self.hands.len(x);
        let na = node.num_actions();
        let (q, r) = opponents(x);
        let rows: Vec<(Vec<f32>, Vec<f32>)> = node
            .children()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&c| {
                let mut v = vec![0f32; n];
                let mut mass = vec![0f32; n];
                traverse(&ctx, c, x, &reach[q], &reach[r], Mode::Average, false, &mut v);
                traverse(&ctx, c, x, &reach[q], &reach[r], Mode::Average, true, &mut mass);
                (v, mass)
            })
            .collect();
        let mut ev = vec![0f32; na * n];
        for (a, (v, mass)) in rows.iter().enumerate() {
            for h in 0..n {
                ev[a * n + h] = if mass[h] > 0.0 {
                    (v[h] as f64 / mass[h] as f64 + ctx.third_pot + node.invest[x] as f64) as f32
                } else {
                    0.0
                };
            }
        }
        let frac = |p: usize| {
            let root: f64 = self.hands.weights[p].iter().map(|&w| w as f64).sum();
            let now: f64 = reach[p].iter().map(|&w| w as f64).sum();
            if root > 0.0 {
                now / root
            } else {
                0.0
            }
        };
        Ok(crate::NodeInfo {
            player: x,
            actions: self.tree.actions(id),
            combos: self.hands.combos[x].clone(),
            strategy: self.average_strategy(id),
            ev,
            reach: reach[x].clone(),
            reach_fraction: [frac(x), frac(q).min(frac(r))],
            pot: self.config.root.pot + node.invest.iter().map(|&i| i as Chips).sum::<Chips>(),
            invest: [node.invest[x] as Chips, node.invest[q] as Chips],
        })
    }

    /// Ranges (1326 weights) of the three players after `path`.
    pub fn ranges_after(&self, path: &[Step]) -> Result<[Vec<f32>; N], SolverError> {
        let (_, reach) = self.walk(path)?;
        let mut out = [vec![0f32; NUM_COMBOS], vec![0f32; NUM_COMBOS], vec![0f32; NUM_COMBOS]];
        for p in 0..N {
            for (i, &c) in self.hands.combos[p].iter().enumerate() {
                out[p][c as usize] = reach[p][i];
            }
        }
        Ok(out)
    }

    /// Hand index lookup (combo -> index) for player `p`.
    pub fn has_combo(&self, p: usize, combo: usize) -> bool {
        self.hands.lookup[p][combo] != NONE
    }

    /// Switches the terminal model (exact card removal between opponents or
    /// independent opponents) for later solving and evaluation; used to
    /// measure how a strategy found with one model fares under the other.
    pub fn set_exact_removal(&mut self, on: bool) {
        self.config.exact_removal = on;
        self.norm = norms(&self.hands, on);
    }

    /// Average strategy at the root, per action, over the acting player's
    /// whole range.
    pub fn root_range_strategy(&self) -> Vec<f64> {
        self.query(&[]).map(|i| i.range_strategy()).unwrap_or_default()
    }

    /// Number of `(action, hand)` cells in the tree.
    pub fn cells(&self) -> u64 {
        self.tree.storage_len
    }

    /// Number of tree nodes.
    pub fn num_nodes(&self) -> usize {
        self.tree.nodes.len()
    }

    /// Showdown terminals reached by betting vs. by an all-in runout
    /// (diagnostics).
    #[doc(hidden)]
    pub fn showdown_counts(&self) -> (usize, usize) {
        let mut by_bet = 0;
        let mut runout = 0;
        for node in &self.tree.nodes {
            for c in node.children() {
                if self.tree.nodes[c].kind == Kind::Showdown {
                    if node.kind == Kind::Chance {
                        runout += 1;
                    } else {
                        by_bet += 1;
                    }
                }
            }
        }
        (by_bet, runout)
    }
}

fn too_large(config: &MultiConfig, needed: u64) -> SolverError {
    SolverError::TooLarge { needed, limit: config.max_memory_bytes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BettingState, SolveConfig, Solver};
    use holdem::card::parse_cards;
    use holdem::{Range, StreetSizes};

    fn menu() -> PostflopMenu {
        let s = StreetSizes::new(&["66%"], &["3x"]);
        PostflopMenu { flop: s.clone(), turn: s.clone(), river: s, max_raises: 2, ..PostflopMenu::default() }
    }

    fn config(board: &str, ranges: [&str; N], folded: [bool; N]) -> MultiConfig {
        let board = parse_cards(board).unwrap();
        let street = Street::from_board_len(board.len());
        MultiConfig {
            board,
            ranges: ranges.map(|r| Range::parse(r).unwrap().into_weights()),
            root: MultiState::street_start(street, 900, [3000, 3000, 3000], folded, 100),
            menu: menu(),
            future_menu: None,
            big_blind: 100,
            chance: ChanceMode::Exact,
            prune: 0.0,
            keep: Default::default(),
            max_memory_bytes: 1 << 30,
            rake_rate: 0.0,
            rake_cap: 0.0,
            exact_removal: false,
            storage: StorageMode::F32,
            dcfr: multiway_dcfr(),
        }
    }

    /// With the third player already folded holding any two cards, the
    /// three-player solver is exactly the heads-up game.
    #[test]
    fn matches_heads_up_when_third_player_folded() {
        let r0 = "AA, KK, QJs, T9s, 76s, A5s";
        let r1 = "AK, QQ, JJ, KQs, 98s, 65s";
        let mut m = MultiSolver::new(config("Kd9s5h2c2d", [r0, r1, "any"], [false, false, true])).unwrap();
        let hu_cfg = SolveConfig::new(
            parse_cards("Kd9s5h2c2d").unwrap(),
            [Range::parse(r0).unwrap().into_weights(), Range::parse(r1).unwrap().into_weights()],
            BettingState::street_start(Street::River, 900, [3000, 3000], 100),
            menu(),
        );
        let mut h = Solver::new(hu_cfg).unwrap();
        // The update order differs (three traversals per iteration), so
        // compare converged solutions.
        let opts = SolveOptions { max_iterations: 600, target_exploitability_pct: 0.0, check_every: 600 };
        m.solve(&opts, |_| true);
        h.solve(&opts, |_| true);
        for path in [vec![], vec![Step::Action(Action::Check)]] {
            let (qm, qh) = (m.query(&path).unwrap(), h.query(&path).unwrap());
            assert_eq!(qm.actions, qh.actions);
            assert_eq!(qm.combos, qh.combos);
            for (a, b) in qm.range_strategy().iter().zip(qh.range_strategy()) {
                assert!((a - b).abs() < 0.01, "range frequency {a} vs {b}");
            }
            let ds = qm.strategy.iter().zip(&qh.strategy).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(ds < 0.05, "strategy differs by {ds}");
            // EVs of actions that are actually played (others are poorly
            // determined by any CFR solve).
            let de = (0..qm.ev.len())
                .filter(|&k| qm.strategy[k] > 0.05 && qh.strategy[k] > 0.05)
                .map(|k| (qm.ev[k] - qh.ev[k]).abs())
                .fold(0.0, f32::max);
            assert!(de < 1.5, "EV differs by {de} chips (pot 900)");
        }
    }

    /// A real three-way turn spot converges and gives well-formed answers.
    #[test]
    fn three_way_turn_spot() {
        let mut cfg = config(
            "Jh8d4c2s",
            [
                "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
                "AA, KK, AJ, QJ, JT, 99, 77, KQs",
                "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
            ],
            [false; N],
        );
        cfg.chance = ChanceMode::Sampled { turn: 0, river: 12, seed: 1 };
        let mut m = MultiSolver::new(cfg).unwrap();
        let opts = SolveOptions { max_iterations: 150, target_exploitability_pct: 1.0, check_every: 25 };
        let mut history = Vec::new();
        let p = m.solve(&opts, |p| {
            history.push(p.exploitability_pct);
            true
        });
        assert!(history.first().unwrap() > history.last().unwrap(), "{history:?}");
        assert!(p.exploitability_pct < 5.0, "{history:?}");
        // Treating the opponents as independent makes the game only
        // approximately zero-sum.
        let v = m.values();
        assert!(v.iter().sum::<f64>().abs() < 0.03 * 900.0, "zero-sum: {v:?}");
        for path in [
            vec![],
            vec![Step::Action(Action::Check)],
            vec![Step::Action(Action::Check), Step::Action(Action::Check)],
        ] {
            let q = m.query(&path).unwrap();
            for h in 0..q.num_hands() {
                let mix: f32 = q.hand_strategy(h).iter().sum();
                assert!((mix - 1.0).abs() < 1e-3);
            }
        }
        // Facing a bet: folding is worth exactly zero.
        let bet = m.query(&[]).unwrap().actions[1];
        let q = m.query(&[Step::Action(bet)]).unwrap();
        assert_eq!(q.actions[0], Action::Fold);
        for h in 0..q.num_hands() {
            assert!(q.hand_ev(h)[0].abs() < 1e-3);
        }
        let r = m.ranges_after(&[Step::Action(bet)]).unwrap();
        assert!(r.iter().all(|v| v.iter().all(|&x| (0.0..=1.0 + 1e-6).contains(&x))));
    }

    /// The rebuilt solver plays the same game as the previous implementation
    /// (kept in `multi_ref`): same tree, and the same values, best-response
    /// gains and EVs for the same (uniform) strategies.
    #[test]
    fn same_game_as_previous_solver() {
        let ranges = [
            "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
            "AA, KK, AJ, QJ, JT, 99, 77, KQs",
            "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
        ];
        for (exact, chance) in [
            (true, ChanceMode::Exact),
            (false, ChanceMode::Exact),
            (true, ChanceMode::Sampled { turn: 0, river: 9, seed: 4 }),
        ] {
            let mut cfg = config("Jh8d4c2s", ranges, [false; N]);
            cfg.exact_removal = exact;
            cfg.chance = chance;
            let new = MultiSolver::new(cfg.clone()).unwrap();
            let old_cfg = crate::multi_ref::MultiConfig {
                board: cfg.board.clone(),
                ranges: cfg.ranges.clone(),
                root: crate::multi_ref::MultiState::street_start(
                    Street::Turn,
                    900,
                    [3000; N],
                    [false; N],
                    100,
                ),
                menu: cfg.menu.clone(),
                future_menu: None,
                big_blind: 100,
                chance,
                prune: 0.0,
                keep: Default::default(),
                max_memory_bytes: 1 << 30,
                rake_rate: 0.0,
                rake_cap: 0.0,
                exact_removal: exact,
            };
            let old = crate::multi_ref::MultiSolver::new(old_cfg).unwrap();
            assert_eq!(new.cells(), old.cells());
            // All-in runouts are now sampled in sampled trees (the old solver
            // dealt every card there), so only the tree shape compares.
            if chance != ChanceMode::Exact {
                continue;
            }
            // With exact removal and every runout the folded-player shortcut is
            // exact; otherwise the folded player's mass below isn't quite
            // conserved in the old full traversal.
            let tol = if exact && chance == ChanceMode::Exact { 1e-3 } else { 0.03 } * 900.0;
            let (vn, vo) = (new.values(), old.values());
            for p in 0..N {
                assert!((vn[p] - vo[p]).abs() < tol, "{exact} {chance:?}: values {vn:?} vs {vo:?}");
            }
            let (cn, co) = (new.convergence(), old.convergence());
            assert!((cn - co).abs() < tol, "{exact} {chance:?}: convergence {cn} vs {co}");
            let (qn, qo) = (new.query(&[]).unwrap(), old.query(&[]).unwrap());
            assert_eq!(qn.actions, qo.actions);
            assert_eq!(qn.strategy, qo.strategy);
            let de = qn.ev.iter().zip(&qo.ev).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(de < tol as f32, "{exact} {chance:?}: EVs differ by {de}");
        }
    }

    /// The batched river evaluator gives the same results as the recursive
    /// traversal (up to rounding, which a few CFR iterations amplify a bit).
    #[test]
    fn batched_river_matches_recursion() {
        let ranges = [
            "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
            "AA, KK, AJ, QJ, JT, 99, 77, KQs",
            "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
        ];
        for (board, exact) in
            [("Jh8d4c2s9s", true), ("Jh8d4c2s9s", false), ("Jh8d4c2s", true), ("Jh8d4c2s", false)]
        {
            let mut cfg = config(board, ranges, [false; N]);
            cfg.exact_removal = exact;
            if board.len() == 8 {
                cfg.chance = ChanceMode::Sampled { turn: 0, river: 6, seed: 2 };
            }
            let mut a = MultiSolver::new(cfg.clone()).unwrap();
            let mut b = MultiSolver::new(cfg).unwrap();
            b.set_batch_river(false);
            // One iteration: the paths differ only by rounding. Later
            // iterations amplify that (regret signs near zero flip), so the
            // check after four is looser.
            for (iters, tol_v, tol_s) in [(1, 1e-3, 1e-5), (4, 0.5, 1e-2)] {
                let opts = SolveOptions {
                    max_iterations: iters,
                    target_exploitability_pct: 0.0,
                    check_every: iters,
                };
                a.solve(&opts, |_| true);
                b.solve(&opts, |_| true);
                let (va, vb) = (a.values(), b.values());
                for p in 0..N {
                    assert!((va[p] - vb[p]).abs() < tol_v, "{board} {exact} it {iters}: {va:?} vs {vb:?}");
                }
                assert!((a.convergence() - b.convergence()).abs() < tol_v, "{board} {exact} it {iters}");
                for path in [vec![], vec![Step::Action(Action::Check)]] {
                    let (qa, qb) = (a.query(&path).unwrap(), b.query(&path).unwrap());
                    let ds =
                        qa.strategy.iter().zip(&qb.strategy).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    assert!(
                        ds < tol_s as f32,
                        "{board} {exact} it {iters} {path:?}: strategies differ by {ds}"
                    );
                }
            }
        }
    }

    /// Narrow, overlapping ranges (many hand triples impossible) stay finite
    /// over a long solve.
    #[test]
    fn narrow_overlapping_ranges_stay_finite() {
        for exact in [true, false] {
            let mut cfg = config(
                "Ah7c2d5s",
                ["AA, KK, QQ, AK, AQs", "AK, AQ, AJs, KK, QQ, JJ", "AQ, AJ, KQ, TT, 99, 88, A5s"],
                [false; N],
            );
            cfg.exact_removal = exact;
            let mut m = MultiSolver::new(cfg).unwrap();
            let opts = SolveOptions { max_iterations: 150, target_exploitability_pct: 0.0, check_every: 50 };
            let p = m.solve(&opts, |_| true);
            assert!(p.exploitability.is_finite(), "exact {exact}: {p:?}");
            assert!(m.values().iter().all(|v| v.is_finite()));
        }
    }

    /// With exact card removal between opponents (and no rake) the game is
    /// exactly zero-sum, for any strategies.
    #[test]
    fn exact_removal_is_zero_sum() {
        let mut cfg = config(
            "Ah7c2d5s",
            ["AA, KK, QQ, AK, AQs", "AK, AQ, AJs, KK, QQ, JJ", "AQ, AJ, KQ, TT, 99, 88, A5s"],
            [false; N],
        );
        cfg.exact_removal = true;
        let mut m = MultiSolver::new(cfg).unwrap();
        for iters in [1, 30] {
            let opts =
                SolveOptions { max_iterations: iters, target_exploitability_pct: 0.0, check_every: iters };
            m.solve(&opts, |_| true);
            let v = m.values();
            assert!(v.iter().sum::<f64>().abs() < 1e-3 * 900.0, "after {iters} iterations: {v:?}");
        }
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;
    use holdem::card::parse_cards;
    use holdem::{Range, StreetSizes};
    use std::time::Instant;

    /// Prints tree sizes and solve times of typical 3-way flop setups.
    #[test]
    #[ignore]
    fn three_way_flop_sizes() {
        let ranges = [
            "22-99, A2s-A9s, ATo-AJo, K9s-KJs, KTo+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s",
            "22+, A2s+, ATo+, K9s+, KJo+, Q9s+, QJo, J9s+, T9s, 98s, 87s, 76s",
            "22-JJ, A9s-AQs, AJo-AQo, KTs+, KQo, QTs+, JTs, T9s, 98s",
        ];
        let board = parse_cards("Ks7d2c").unwrap();
        for (label, future_raise, cap, chance) in [
            ("future 1 size, no raises, 6x4", "3x", 1, ChanceMode::Sampled { turn: 6, river: 4, seed: 1 }),
            (
                "future 1 size, allin raise, 8x6",
                "allin",
                2,
                ChanceMode::Sampled { turn: 8, river: 6, seed: 1 },
            ),
            ("future 1 size, no raises, 12x6", "3x", 1, ChanceMode::Sampled { turn: 12, river: 6, seed: 1 }),
        ] {
            let future = PostflopMenu {
                turn: StreetSizes::new(&["66%"], &[future_raise]),
                river: StreetSizes::new(&["75%"], &[future_raise]),
                max_raises: cap,
                ..PostflopMenu::default()
            };
            let cfg = MultiConfig {
                board: board.clone(),
                ranges: ranges.map(|r| Range::parse(r).unwrap().into_weights()),
                root: MultiState::street_start(Street::Flop, 800, [9750, 9750, 9750], [false; N], 100),
                menu: PostflopMenu { max_raises: 2, ..PostflopMenu::default() },
                future_menu: Some(future),
                big_blind: 100,
                chance,
                prune: 0.0,
                keep: Default::default(),
                max_memory_bytes: 8 << 30,
                rake_rate: 0.0,
                rake_cap: 0.0,
                exact_removal: false,
                storage: StorageMode::Auto,
                dcfr: multiway_dcfr(),
            };
            let t = Instant::now();
            let mut m = MultiSolver::new(cfg).unwrap();
            let cells = m.cells();
            let opts = SolveOptions { max_iterations: 300, target_exploitability_pct: 1.0, check_every: 20 };
            let p = m.solve(&opts, |_| true);
            println!(
                "{label:<36} cells {cells:>10} (~{:.2} GB) iters {} conv {:.2}% pot, {:.1}s",
                cells as f64 * 8.0 / 1e9,
                p.iteration,
                p.exploitability_pct,
                t.elapsed().as_secs_f64()
            );
        }
    }
}
