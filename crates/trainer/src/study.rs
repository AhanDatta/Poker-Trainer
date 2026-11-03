//! Solver study mode: browse the preflop charts and the solver's strategy
//! in any spot, GTO Wizard style.
//!
//! * [`PreflopNav`] walks the preflop charts. At every decision it gives the
//!   chart's mix per hand class and the acting seat's range so far.
//! * [`Spot`] is a postflop starting point: players, ranges, pot, stacks and
//!   board, taken from a preflop line or typed by hand.
//! * [`Study`] browses a spot's postflop game tree. Each street is solved
//!   when it's reached, with the full bet menu, from the ranges the previous
//!   street's solution narrows to (the way hand reviews work). Entering a
//!   line the solver almost never takes re-solves the spot from there with
//!   that range widened. Solves and node queries run on background threads.

use crate::config::{FlopMode, SolverSettings, TrainerConfig};
use crate::fmt::{action_label, ActionContext, BB};
use crate::preflop::{availability, chart_policy, line_range, spot_key, ChartSet, PreAction, PreDecision};
use crate::review::{
    class_grid, hu_config, multi_config, postflop_order_n, run_hu, run_multi, ClassCell, SolveSink, CANCELLED,
};
use holdem::combo::{combo_mask, HandClass, COMBO_CLASS, NUM_CLASSES, NUM_COMBOS};
use holdem::game::Phase;
use holdem::{Action, BetMenu, Card, CardSet, Chips, HandState, Range, Street, TableConfig};
use solver::multi::{MultiSolver, MultiState, Next};
use solver::{BettingState, NodeInfo, Progress, Solver, SolverError, Step, Transition};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Solve-queue priority of study solves: ahead of every hand review.
const PRIORITY: u64 = 0;
/// Finished solves kept for lines you navigated away from. Turn and river
/// solves are small; flop solves can take gigabytes, so flop re-solves are
/// dropped as soon as you leave them (the spot's own flop solve stays).
const KEEP_OFF_LINE: usize = 8;
const KEEP_OFF_LINE_FLOPS: usize = 0;

/// Inputs shared by the preflop and postflop browsers.
#[derive(Clone)]
pub struct StudySettings {
    /// Table size of the preflop charts.
    pub players: usize,
    pub stack_bb: f64,
    pub menu: BetMenu,
    pub charts: Arc<ChartSet>,
    pub solver: SolverSettings,
    pub rake_rate: f64,
    pub rake_cap: Chips,
}

impl StudySettings {
    pub fn from_config(cfg: &TrainerConfig, charts: Arc<ChartSet>) -> StudySettings {
        StudySettings {
            players: cfg.players,
            stack_bb: cfg.stack_bb,
            menu: cfg.menu.clone(),
            charts,
            solver: cfg.solver.clone(),
            rake_rate: cfg.rake_pct / 100.0,
            rake_cap: (cfg.rake_cap_bb * BB as f64).round() as Chips,
        }
    }
}

/// Number of combos of a class that don't use a board card.
pub fn live_combos(class: usize, board: &[Card]) -> usize {
    let dead = CardSet::from_cards(board).0;
    HandClass::from_grid_index(class).combos().into_iter().filter(|&c| combo_mask(c) & dead == 0).count()
}

/// Share of each class's live combos that is in a range (0..=1, grid order).
pub fn class_fill(weights: &[f32], board: &[Card]) -> Vec<f32> {
    let dead = CardSet::from_cards(board).0;
    let mut sum = vec![0f32; NUM_CLASSES];
    let mut live = vec![0f32; NUM_CLASSES];
    for (c, &w) in weights.iter().enumerate() {
        if combo_mask(c) & dead != 0 {
            continue;
        }
        let g = COMBO_CLASS[c] as usize;
        sum[g] += w;
        live[g] += 1.0;
    }
    (0..NUM_CLASSES).map(|g| if live[g] > 0.0 { (sum[g] / live[g]).clamp(0.0, 1.0) } else { 0.0 }).collect()
}

/// Number of combos in a range (sum of weights), ignoring board-blocked ones.
pub fn range_combos(weights: &[f32], board: &[Card]) -> f32 {
    let dead = CardSet::from_cards(board).0;
    weights.iter().enumerate().filter(|&(c, _)| combo_mask(c) & dead == 0).map(|(_, &w)| w).sum()
}

fn fraction(now: &[f32], root: &[f32]) -> f64 {
    let r: f64 = root.iter().map(|&w| w as f64).sum();
    let n: f64 = now.iter().map(|&w| w as f64).sum();
    if r > 0.0 {
        n / r
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Preflop

/// A preflop decision in study mode.
#[derive(Clone, Debug)]
pub struct PreflopNode {
    pub seat: usize,
    pub position: String,
    pub actions: Vec<Action>,
    pub labels: Vec<String>,
    /// Chart mix per class over `actions` (grid order); `weight` is the
    /// share of the class still in the seat's range (0..=1). `None` where
    /// the class is no longer in the range.
    pub grid: Vec<Option<ClassCell>>,
    /// Frequency of each action over the seat's whole range.
    pub range_strategy: Vec<f64>,
}

/// Where the preflop line stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreflopOutcome {
    Deciding,
    /// Everyone else folded.
    Won(usize),
    /// All-in before the flop: nothing left to decide.
    AllIn,
    /// These seats see the flop (postflop order).
    Flop(Vec<usize>),
}

/// One action of the preflop line.
#[derive(Clone, Debug)]
pub struct PreflopStep {
    pub seat: usize,
    pub position: String,
    pub action: Action,
    pub label: String,
}

/// Walks the preflop charts.
#[derive(Clone)]
pub struct PreflopNav {
    settings: Arc<StudySettings>,
    state: HandState,
    decisions: Vec<PreDecision>,
    line: Vec<PreflopStep>,
}

impl PreflopNav {
    pub fn new(settings: Arc<StudySettings>) -> PreflopNav {
        let state = Self::fresh(&settings);
        PreflopNav { settings, state, decisions: Vec::new(), line: Vec::new() }
    }

    fn fresh(s: &StudySettings) -> HandState {
        let n = s.players;
        let table = TableConfig::new(n, BB / 2, BB).with_rake(s.rake_rate, s.rake_cap);
        let stack = (s.stack_bb * BB as f64).round() as Chips;
        // The cards are never looked at.
        let hole =
            (0..n).map(|i| [Card::from_index(2 * i as u8), Card::from_index(2 * i as u8 + 1)]).collect();
        HandState::new(table, n - 1, vec![stack; n], hole)
    }

    pub fn settings(&self) -> &Arc<StudySettings> {
        &self.settings
    }

    pub fn state(&self) -> &HandState {
        &self.state
    }

    pub fn line(&self) -> &[PreflopStep] {
        &self.line
    }

    fn context(&self, p: usize) -> ActionContext {
        let sit = self.state.situation(p);
        ActionContext {
            to_call: sit.to_call,
            pot: sit.pot,
            all_in_to: self.state.street_commit[p] + self.state.stacks[p],
            facing_bet: self.state.current_bet > 0,
            preflop: true,
        }
    }

    /// The range a seat's preflop line implies under the charts.
    pub fn range_of(&self, seat: usize) -> Range {
        line_range(&self.decisions, seat, |d| chart_policy(&self.settings.charts, d)).0
    }

    pub fn node(&self) -> Option<PreflopNode> {
        if self.state.phase != Phase::Betting || self.state.street != Street::Preflop {
            return None;
        }
        let p = self.state.to_act?;
        let avail = availability(&self.state, &self.settings.menu)?;
        let position = self.state.position_name(p).to_string();
        let d = PreDecision {
            seat: p,
            key: spot_key(&self.decisions, &position),
            position: position.clone(),
            avail: avail.clone(),
            action: PreAction::Fold,
        };
        let freqs = chart_policy(&self.settings.charts, &d);
        let abs: Vec<PreAction> = PreAction::ALL.into_iter().filter(|&a| avail.has(a)).collect();
        let actions: Vec<Action> = abs.iter().filter_map(|&a| avail.concrete(a)).collect();
        let ctx = self.context(p);
        let labels = actions.iter().map(|&a| action_label(a, &ctx)).collect();
        let class_w = self.range_of(p).class_weights();
        let mut grid = Vec::with_capacity(NUM_CLASSES);
        let mut total = vec![0f64; abs.len()];
        let mut mass = 0f64;
        for (g, &w) in class_w.iter().enumerate() {
            if w <= 0.0 {
                grid.push(None);
                continue;
            }
            let raw: Vec<f32> = abs.iter().map(|a| freqs[g][a.index()]).collect();
            let sum: f32 = raw.iter().sum();
            let strategy: Vec<f32> = if sum > 0.0 {
                raw.iter().map(|x| x / sum).collect()
            } else {
                let mut s = vec![0.0; abs.len()];
                s[0] = 1.0;
                s
            };
            let combos = HandClass::from_grid_index(g).num_combos() as f64 * w as f64;
            for (t, &x) in total.iter_mut().zip(&strategy) {
                *t += combos * x as f64;
            }
            mass += combos;
            grid.push(Some(ClassCell { weight: w, strategy }));
        }
        let range_strategy = total.iter().map(|t| if mass > 0.0 { t / mass } else { 0.0 }).collect();
        Some(PreflopNode { seat: p, position, actions, labels, grid, range_strategy })
    }

    fn record(&mut self, a: Action) -> Result<(), String> {
        let p = self.state.to_act.ok_or("nobody is to act")?;
        let avail = availability(&self.state, &self.settings.menu).ok_or("nobody is to act")?;
        let abs = avail.classify(a).ok_or("not a preflop menu action")?;
        let position = self.state.position_name(p).to_string();
        let key = spot_key(&self.decisions, &position);
        let label = action_label(a, &self.context(p));
        self.state.apply(a).map_err(|e| e.to_string())?;
        self.decisions.push(PreDecision { seat: p, position: position.clone(), key, avail, action: abs });
        self.line.push(PreflopStep { seat: p, position, action: a, label });
        Ok(())
    }

    /// Takes action `i` of the current node.
    pub fn act(&mut self, i: usize) -> Result<(), String> {
        let node = self.node().ok_or("the preflop betting is over")?;
        let a = *node.actions.get(i).ok_or("no such action")?;
        self.record(a)
    }

    /// Undoes the last action; `false` at the start.
    pub fn back(&mut self) -> bool {
        if self.line.is_empty() {
            return false;
        }
        let mut actions: Vec<Action> = self.line.iter().map(|s| s.action).collect();
        actions.pop();
        self.reset();
        for a in actions {
            self.record(a).expect("replaying a legal line");
        }
        true
    }

    pub fn reset(&mut self) {
        self.state = Self::fresh(&self.settings);
        self.decisions.clear();
        self.line.clear();
    }

    pub fn outcome(&self) -> PreflopOutcome {
        let st = &self.state;
        if st.phase == Phase::Betting && st.street == Street::Preflop {
            return PreflopOutcome::Deciding;
        }
        let in_hand = st.players_in_hand();
        if in_hand.len() == 1 {
            return PreflopOutcome::Won(in_hand[0]);
        }
        if st.is_runout() {
            return PreflopOutcome::AllIn;
        }
        PreflopOutcome::Flop(postflop_order_n(st, &in_hand))
    }

    /// The postflop spot this line leads to (board still empty), for two or
    /// three players.
    pub fn spot(&self) -> Option<Spot> {
        let PreflopOutcome::Flop(seats) = self.outcome() else { return None };
        if !(2..=3).contains(&seats.len()) {
            return None;
        }
        Some(Spot {
            names: seats.iter().map(|&s| self.state.position_name(s).to_string()).collect(),
            ranges: seats.iter().map(|&s| self.range_of(s).into_weights()).collect(),
            pot: self.state.pot(),
            stacks: seats.iter().map(|&s| self.state.stacks[s]).collect(),
            board: Vec::new(),
            preflop: self.line_text(),
        })
    }

    /// The line as text, folds before the first entrant left out.
    pub fn line_text(&self) -> String {
        let mut entered = false;
        let mut parts = Vec::new();
        for s in &self.line {
            if s.action == Action::Fold && !entered {
                continue;
            }
            entered = true;
            parts.push(format!("{} {}", s.position, s.label));
        }
        parts.join(" · ")
    }
}

// ---------------------------------------------------------------------------
// Spots

/// A postflop starting point. Players are listed in acting order (out of
/// position first).
#[derive(Clone, Debug)]
pub struct Spot {
    pub names: Vec<String>,
    /// 1326 weights per player.
    pub ranges: Vec<Vec<f32>>,
    pub pot: Chips,
    /// Chips behind per player.
    pub stacks: Vec<Chips>,
    /// 3 to 5 cards.
    pub board: Vec<Card>,
    /// The preflop line (empty for typed spots).
    pub preflop: String,
}

impl Spot {
    pub fn default_names(n: usize) -> Vec<String> {
        match n {
            2 => vec!["OOP".into(), "IP".into()],
            _ => vec!["OOP".into(), "Middle".into(), "IP".into()],
        }
    }

    /// A spot from typed ranges (range notation, e.g. `22+,A2s+,KTo+`),
    /// pot and effective stack in big blinds, and board (e.g. `Ks7d2c`).
    pub fn from_text(
        names: Vec<String>,
        ranges: &[String],
        pot_bb: f64,
        stack_bb: f64,
        board: &str,
    ) -> Result<Spot, String> {
        let mut parsed = Vec::new();
        for (name, text) in names.iter().zip(ranges) {
            let r = Range::parse(text).map_err(|e| format!("{name}'s range: {e}"))?;
            parsed.push(r.into_weights());
        }
        let board = holdem::card::parse_cards(board).map_err(|e| format!("board: {e}"))?;
        let spot = Spot {
            stacks: vec![(stack_bb * BB as f64).round() as Chips; names.len()],
            names,
            ranges: parsed,
            pot: (pot_bb * BB as f64).round() as Chips,
            board,
            preflop: String::new(),
        };
        spot.validate()?;
        Ok(spot)
    }

    pub fn street(&self) -> Street {
        Street::from_board_len(self.board.len())
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.names.len();
        if !(2..=3).contains(&n) {
            return Err("study mode supports 2 or 3 players".into());
        }
        if self.ranges.len() != n || self.stacks.len() != n {
            return Err("one range and stack per player".into());
        }
        if !(3..=5).contains(&self.board.len()) {
            return Err(format!("the board needs 3 to 5 cards (got {})", self.board.len()));
        }
        if CardSet::from_cards(&self.board).len() != self.board.len() {
            return Err("the board has the same card twice".into());
        }
        if self.pot <= 0 {
            return Err("the pot must be positive".into());
        }
        for (name, r) in self.names.iter().zip(&self.ranges) {
            if r.len() != NUM_COMBOS || range_combos(r, &self.board) <= 0.0 {
                return Err(format!("{name}'s range is empty (after removing hands with board cards)"));
            }
        }
        if self.stacks.iter().filter(|&&s| s > 0).count() < 2 {
            return Err("at least two players need chips behind".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Postflop

/// Betting state of a heads-up or three-way street.
#[derive(Clone, Copy, Debug)]
enum AnyState {
    Hu(BettingState),
    Multi(MultiState),
}

enum Flow {
    Continue(AnyState),
    StreetOver(AnyState),
    /// Everyone else folded (index into the segment's players).
    Won(usize),
}

impl AnyState {
    fn street(&self) -> Street {
        match self {
            AnyState::Hu(s) => s.street,
            AnyState::Multi(s) => s.street,
        }
    }
    fn actor(&self) -> usize {
        match self {
            AnyState::Hu(s) => s.to_act,
            AnyState::Multi(s) => s.to_act,
        }
    }
    fn pot(&self) -> Chips {
        match self {
            AnyState::Hu(s) => s.pot,
            AnyState::Multi(s) => s.pot,
        }
    }
    fn len(&self) -> usize {
        match self {
            AnyState::Hu(_) => 2,
            AnyState::Multi(_) => 3,
        }
    }
    fn stack(&self, i: usize) -> Chips {
        match self {
            AnyState::Hu(s) => s.stack[i],
            AnyState::Multi(s) => s.stack[i],
        }
    }
    fn folded(&self, i: usize) -> bool {
        match self {
            AnyState::Hu(_) => false,
            AnyState::Multi(s) => s.folded[i],
        }
    }
    fn context(&self) -> ActionContext {
        let (sit, commit, stack, bet) = match self {
            AnyState::Hu(s) => (s.situation(BB), s.commit[s.to_act], s.stack[s.to_act], s.current_bet),
            AnyState::Multi(s) => (s.situation(BB), s.commit[s.to_act], s.stack[s.to_act], s.current_bet),
        };
        ActionContext {
            to_call: sit.to_call,
            pot: self.pot(),
            all_in_to: commit + stack,
            facing_bet: bet > 0,
            preflop: false,
        }
    }
    fn flow(&self, a: Action) -> Flow {
        match self {
            AnyState::Hu(s) => match s.transition(a) {
                Transition::Continue(n) => Flow::Continue(AnyState::Hu(n)),
                Transition::StreetOver(n) => Flow::StreetOver(AnyState::Hu(n)),
                Transition::Fold(p) => Flow::Won(p ^ 1),
            },
            AnyState::Multi(s) => match s.transition(a) {
                Next::Continue(n) => Flow::Continue(AnyState::Multi(n)),
                Next::StreetOver(n) => Flow::StreetOver(AnyState::Multi(n)),
                Next::Won(w) => Flow::Won(w),
            },
        }
    }
}

enum AnySolver {
    Hu(Solver),
    Multi(MultiSolver),
}

impl AnySolver {
    fn query(&self, path: &[Step]) -> Result<NodeInfo, SolverError> {
        match self {
            AnySolver::Hu(s) => s.query(path),
            AnySolver::Multi(s) => s.query(path),
        }
    }
    fn ranges_after(&self, path: &[Step]) -> Result<Vec<Vec<f32>>, SolverError> {
        Ok(match self {
            AnySolver::Hu(s) => s.ranges_after(path)?.to_vec(),
            AnySolver::Multi(s) => s.ranges_after(path)?.to_vec(),
        })
    }
}

/// State of one street solve.
#[derive(Clone, Debug, Default)]
pub struct SolveStatus {
    pub label: String,
    pub stage: String,
    pub since: Option<Instant>,
    pub progress: Option<Progress>,
    pub done: bool,
    pub cancelled: bool,
    pub error: Option<String>,
    /// Exploitability (heads-up) or convergence (3-way), % of the pot.
    pub exploitability_pct: f64,
    pub notes: Vec<String>,
}

#[derive(Default)]
struct Job {
    status: Mutex<SolveStatus>,
    solver: RwLock<Option<AnySolver>>,
    cancel: AtomicBool,
    /// Node results by path from the segment root (`None` while computing).
    nodes: Mutex<HashMap<Vec<Step>, Option<Arc<NodeResult>>>>,
}

impl Job {
    fn status(&self) -> SolveStatus {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    fn with_status(&self, f: impl FnOnce(&mut SolveStatus)) {
        f(&mut self.status.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

impl SolveSink for Job {
    fn stage(&self, s: String) {
        self.with_status(|st| {
            st.stage = s;
            st.since = Some(Instant::now());
            st.progress = None;
        });
    }
    fn note(&self, s: String) {
        self.with_status(|st| st.notes.push(s));
    }
    fn progress(&self, p: Progress) {
        self.with_status(|st| st.progress = Some(p));
    }
    fn cancel_flag(&self) -> &AtomicBool {
        &self.cancel
    }
}

/// A solved part of the line: a street from its start, or from a node
/// where it was re-solved.
struct Segment {
    /// Spot players in this segment, in acting order.
    players: Vec<usize>,
    board: Vec<Card>,
    root: AnyState,
    job: Arc<Job>,
}

/// What a node query produced.
enum NodeResult {
    View(Arc<NodeView>),
    /// The line is rare for the solver: re-solve from here with these
    /// (widened) ranges. `who` are the spot players whose range was widened.
    Rare {
        ranges: Vec<Vec<f32>>,
        who: Vec<usize>,
    },
    Error(String),
}

/// The solver's view of a node.
#[derive(Clone, Debug)]
pub struct NodeView {
    /// Acting spot player (`None` at the end of a street).
    pub actor: Option<usize>,
    pub actions: Vec<Action>,
    pub labels: Vec<String>,
    /// The acting player's per-combo strategy, EVs and reach.
    pub info: Option<NodeInfo>,
    /// The acting player's strategy per class (`weight` = summed reach).
    pub grid: Vec<Option<ClassCell>>,
    pub range_strategy: Vec<f64>,
    /// Every spot player's range here (1326 weights; zero once folded).
    pub ranges: Vec<Vec<f32>>,
    pub pot: Chips,
    pub to_call: Chips,
    /// Chips behind per spot player.
    pub stacks: Vec<Chips>,
    pub folded: Vec<bool>,
    pub exploitability_pct: f64,
    /// Three players in this street: approximate solve.
    pub multiway: bool,
}

/// One item of the line from the spot's root.
#[derive(Clone, Debug)]
pub enum LineItem {
    /// A street starts with these new board cards.
    Street(Street, Vec<Card>),
    Action {
        player: usize,
        label: String,
    },
}

/// Where the browser is.
#[derive(Clone, Debug)]
pub enum StudyPos {
    /// The current street is being solved.
    Solving(SolveStatus),
    /// The strategy at this node is being computed.
    Loading,
    Decision(Arc<NodeView>),
    /// The street is over: choose the next card. `view` holds the ranges
    /// going into the next street.
    NeedCard {
        next: Street,
        view: Option<Arc<NodeView>>,
    },
    /// The hand is over.
    Over {
        text: String,
        view: Option<Arc<NodeView>>,
    },
    Failed(String),
}

enum LocEnd {
    Decision,
    StreetOver(AnyState),
    Won(usize),
    Broken(String),
}

struct Loc {
    seg: Arc<Segment>,
    /// Keys of the segments along the line.
    chain: Vec<Vec<Step>>,
    local: Vec<Step>,
    state: AnyState,
    end: LocEnd,
    line: Vec<LineItem>,
    board: Vec<Card>,
}

/// Browses a spot's postflop game tree.
pub struct Study {
    settings: Arc<StudySettings>,
    spot: Spot,
    steps: Vec<Step>,
    /// Solves by the line (from the spot root) that leads to their root.
    segments: HashMap<Vec<Step>, Arc<Segment>>,
    /// Off-line segments kept for going back, oldest first.
    kept: Vec<Vec<Step>>,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Drop for Study {
    fn drop(&mut self) {
        for seg in self.segments.values() {
            seg.job.cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl Study {
    /// Starts solving the spot's first street.
    pub fn new(settings: Arc<StudySettings>, spot: Spot) -> Result<Study, String> {
        spot.validate()?;
        let pool = if settings.solver.threads > 0 {
            rayon::ThreadPoolBuilder::new().num_threads(settings.solver.threads).build().ok().map(Arc::new)
        } else {
            None
        };
        let n = spot.names.len();
        let street = spot.street();
        let root = if n == 2 {
            AnyState::Hu(BettingState::street_start(street, spot.pot, [spot.stacks[0], spot.stacks[1]], BB))
        } else {
            AnyState::Multi(MultiState::street_start(
                street,
                spot.pot,
                [spot.stacks[0], spot.stacks[1], spot.stacks[2]],
                [false; 3],
                BB,
            ))
        };
        let mut study =
            Study { settings, spot, steps: Vec::new(), segments: HashMap::new(), kept: Vec::new(), pool };
        let seg =
            Segment { players: (0..n).collect(), board: study.spot.board.clone(), root, job: Arc::default() };
        let ranges = study.spot.ranges.clone();
        study.add_segment(Vec::new(), seg, ranges, Vec::new());
        Ok(study)
    }

    pub fn spot(&self) -> &Spot {
        &self.spot
    }

    pub fn settings(&self) -> &Arc<StudySettings> {
        &self.settings
    }

    pub fn at_root(&self) -> bool {
        self.steps.is_empty()
    }

    /// Board at the current position.
    pub fn board(&self) -> Vec<Card> {
        self.locate().board
    }

    /// The line from the spot's root to the current position.
    pub fn line(&self) -> Vec<LineItem> {
        self.locate().line
    }

    /// Notes of the solves along the line.
    pub fn notes(&self) -> Vec<String> {
        let loc = self.locate();
        let mut out = Vec::new();
        for key in &loc.chain {
            if let Some(seg) = self.segments.get(key) {
                for n in seg.job.status().notes {
                    if !out.contains(&n) {
                        out.push(n);
                    }
                }
            }
        }
        out
    }

    /// Status of the solve the current position belongs to.
    pub fn solve_status(&self) -> SolveStatus {
        self.locate().seg.job.status()
    }

    fn add_segment(&mut self, key: Vec<Step>, seg: Segment, ranges: Vec<Vec<f32>>, notes: Vec<String>) {
        let street = seg.root.street();
        let multiway = seg.players.len() == 3;
        let mut label = street.to_string();
        if !key.is_empty() && !matches!(key.last(), Some(Step::Card(_))) {
            label.push_str(" re-solve");
        }
        let mut all_notes = notes;
        if multiway {
            all_notes.push(format!(
                "{street}: {}",
                crate::review::multiway_note(self.settings.solver.multiway_exact_removal)
            ));
        } else if street == Street::Flop && self.settings.solver.flop_mode == FlopMode::Fast {
            all_notes.push("Flop solved with sampled turn and river cards (fast mode): approximate.".into());
        }
        seg.job.with_status(|st| {
            st.label = label.clone();
            st.stage = format!("{label}: starting");
            st.since = Some(Instant::now());
            st.notes = all_notes;
        });
        let seg = Arc::new(seg);
        self.segments.insert(key, Arc::clone(&seg));
        self.spawn_solve(seg, ranges, label);
    }

    fn spawn_solve(&self, seg: Arc<Segment>, ranges: Vec<Vec<f32>>, label: String) {
        let settings = Arc::clone(&self.settings);
        let pool = self.pool.clone();
        std::thread::Builder::new()
            .name("study-solve".into())
            .spawn(move || {
                let job = &seg.job;
                let run = || -> Result<(AnySolver, f64), String> {
                    let s = &settings.solver;
                    let menu = &settings.menu.postflop;
                    match seg.root {
                        AnyState::Hu(root) => {
                            let r: [Vec<f32>; 2] = [ranges[0].clone(), ranges[1].clone()];
                            let cfg = hu_config(
                                s,
                                menu,
                                &seg.board,
                                r,
                                root,
                                settings.rake_rate,
                                settings.rake_cap,
                            );
                            run_hu(cfg, s, PRIORITY, pool.as_deref(), &label, &**job)
                                .map(|(x, e)| (AnySolver::Hu(x), e))
                        }
                        AnyState::Multi(root) => {
                            let r: [Vec<f32>; 3] = [ranges[0].clone(), ranges[1].clone(), ranges[2].clone()];
                            let cfg = multi_config(
                                s,
                                menu,
                                &seg.board,
                                r,
                                root,
                                Default::default(),
                                settings.rake_rate,
                                settings.rake_cap,
                            );
                            run_multi(cfg, s, PRIORITY, pool.as_deref(), &label, &**job)
                                .map(|(x, e)| (AnySolver::Multi(x), e))
                        }
                    }
                };
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|e| {
                        let msg = e
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| e.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic".into());
                        Err(format!("internal error: {msg}"))
                    });
                match result {
                    Ok((solver, e)) => {
                        *job.solver.write().unwrap_or_else(|e| e.into_inner()) = Some(solver);
                        job.with_status(|st| {
                            st.done = true;
                            st.exploitability_pct = e;
                            st.stage = format!("{label}: solved");
                        });
                    }
                    Err(e) if e == CANCELLED => job.with_status(|st| st.cancelled = true),
                    Err(e) => job.with_status(|st| {
                        st.done = true;
                        st.error = Some(format!("{label}: {e}"));
                    }),
                }
            })
            .expect("spawn study solve");
    }

    /// Follows the line through the solved segments.
    fn locate(&self) -> Loc {
        let mut key: Vec<Step> = Vec::new();
        let mut seg = Arc::clone(&self.segments[&key]);
        let mut chain = vec![key.clone()];
        let mut local = Vec::new();
        let mut state = seg.root;
        let mut board = self.spot.board.clone();
        let mut line = vec![LineItem::Street(state.street(), board.clone())];
        let mut closed: Option<AnyState> = None;
        let mut end = LocEnd::Decision;
        for (i, &step) in self.steps.iter().enumerate() {
            match step {
                Step::Action(a) => {
                    if closed.is_some() {
                        end = LocEnd::Broken("an action after the end of the street".into());
                        break;
                    }
                    let player = seg.players[state.actor()];
                    line.push(LineItem::Action { player, label: action_label(a, &state.context()) });
                    local.push(step);
                    match state.flow(a) {
                        Flow::Continue(s) => {
                            state = s;
                            if let Some(r) = self.segments.get(&self.steps[..=i]) {
                                key = self.steps[..=i].to_vec();
                                chain.push(key.clone());
                                seg = Arc::clone(r);
                                local.clear();
                                state = seg.root;
                            }
                        }
                        Flow::StreetOver(s) => closed = Some(s),
                        Flow::Won(w) => {
                            end = LocEnd::Won(seg.players[w]);
                            break;
                        }
                    }
                }
                Step::Card(c) => {
                    if closed.take().is_none() {
                        end = LocEnd::Broken("a card before the end of the street".into());
                        break;
                    }
                    board.push(c);
                    line.push(LineItem::Street(Street::from_board_len(board.len()), vec![c]));
                    match self.segments.get(&self.steps[..=i]) {
                        Some(n) => {
                            key = self.steps[..=i].to_vec();
                            chain.push(key.clone());
                            seg = Arc::clone(n);
                            local.clear();
                            state = seg.root;
                        }
                        None => {
                            end = LocEnd::Broken("no solve for this street".into());
                            break;
                        }
                    }
                }
            }
        }
        if matches!(end, LocEnd::Decision) {
            if let Some(s) = closed {
                end = LocEnd::StreetOver(s);
            }
        }
        Loc { seg, chain, local, state, end, line, board }
    }

    /// Where the browser is; starts whatever computation that needs.
    pub fn poll(&mut self) -> StudyPos {
        let loc = self.locate();
        if let LocEnd::Broken(e) = &loc.end {
            return StudyPos::Failed(format!("internal error: {e}"));
        }
        if let LocEnd::Won(p) = loc.end {
            return StudyPos::Over {
                text: format!("{} wins the pot (everyone else folded).", self.spot.names[p]),
                view: None,
            };
        }
        let status = loc.seg.job.status();
        if let Some(e) = status.error {
            return StudyPos::Failed(e);
        }
        if !status.done {
            return StudyPos::Solving(status);
        }
        let decision = matches!(loc.end, LocEnd::Decision);
        let Some(result) = self.node(&loc, decision, status.exploitability_pct) else {
            return StudyPos::Loading;
        };
        match (&*result, &loc.end) {
            (NodeResult::Error(e), _) => StudyPos::Failed(e.clone()),
            (NodeResult::Rare { ranges, who }, _) => {
                let (ranges, who) = (ranges.clone(), who.clone());
                self.start_resolve(&loc, ranges, &who);
                StudyPos::Solving(self.locate().seg.job.status())
            }
            (NodeResult::View(v), LocEnd::Decision) => StudyPos::Decision(Arc::clone(v)),
            (NodeResult::View(v), LocEnd::StreetOver(s)) => {
                let view = Some(Arc::clone(v));
                let can_bet = (0..s.len()).filter(|&i| !s.folded(i) && s.stack(i) > 0).count();
                match s.street().next() {
                    None => StudyPos::Over { text: "Showdown.".into(), view },
                    Some(_) if can_bet < 2 => StudyPos::Over {
                        text: "All-in: no more decisions; the board is dealt out.".into(),
                        view,
                    },
                    Some(next) => StudyPos::NeedCard { next, view },
                }
            }
            (NodeResult::View(_), _) => StudyPos::Failed("internal error: unexpected node".into()),
        }
    }

    /// The cached result for the current node, or `None` while it's computed.
    fn node(&self, loc: &Loc, decision: bool, exploitability_pct: f64) -> Option<Arc<NodeResult>> {
        let job = Arc::clone(&loc.seg.job);
        {
            let mut nodes = job.nodes.lock().unwrap_or_else(|e| e.into_inner());
            match nodes.get(&loc.local) {
                Some(Some(r)) => return Some(Arc::clone(r)),
                Some(None) => return None,
                None => {
                    nodes.insert(loc.local.clone(), None);
                }
            }
        }
        let seg = Arc::clone(&loc.seg);
        let local = loc.local.clone();
        let state = loc.state;
        let n_spot = self.spot.names.len();
        let threshold = self.settings.solver.off_equilibrium_threshold;
        std::thread::Builder::new()
            .name("study-query".into())
            .spawn(move || {
                let compute = || -> Result<NodeResult, String> {
                    let guard = seg.job.solver.read().unwrap_or_else(|e| e.into_inner());
                    let solver = guard.as_ref().ok_or("not solved")?;
                    let now = solver.ranges_after(&local).map_err(|e| e.to_string())?;
                    let k = seg.players.len();
                    if decision && !local.is_empty() {
                        let root = solver.ranges_after(&[]).map_err(|e| e.to_string())?;
                        let rare: Vec<usize> = (0..k)
                            .filter(|&i| !state.folded(i) && fraction(&now[i], &root[i]) < threshold)
                            .collect();
                        if !rare.is_empty() {
                            let mut ranges = now.clone();
                            for &i in &rare {
                                for (w, &r) in ranges[i].iter_mut().zip(&root[i]) {
                                    *w = w.max(0.05 * r);
                                }
                            }
                            let who = rare.iter().map(|&i| seg.players[i]).collect();
                            return Ok(NodeResult::Rare { ranges, who });
                        }
                    }
                    let mut ranges = vec![vec![0f32; NUM_COMBOS]; n_spot];
                    let mut stacks = vec![0; n_spot];
                    let mut folded = vec![true; n_spot];
                    for (i, &p) in seg.players.iter().enumerate() {
                        stacks[p] = state.stack(i);
                        folded[p] = state.folded(i);
                        if !state.folded(i) {
                            ranges[p] = now[i].clone();
                        }
                    }
                    let ctx = state.context();
                    let mut view = NodeView {
                        actor: None,
                        actions: Vec::new(),
                        labels: Vec::new(),
                        info: None,
                        grid: Vec::new(),
                        range_strategy: Vec::new(),
                        ranges,
                        pot: state.pot(),
                        to_call: 0,
                        stacks,
                        folded,
                        exploitability_pct,
                        multiway: k == 3,
                    };
                    if decision {
                        let info = solver.query(&local).map_err(|e| e.to_string())?;
                        view.actor = Some(seg.players[info.player]);
                        view.labels = info.actions.iter().map(|&a| action_label(a, &ctx)).collect();
                        view.actions = info.actions.clone();
                        view.grid = class_grid(&info);
                        view.range_strategy = info.range_strategy();
                        view.to_call = ctx.to_call;
                        view.info = Some(info);
                    }
                    Ok(NodeResult::View(Arc::new(view)))
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(compute))
                    .unwrap_or_else(|_| Err("internal error while reading the solution".into()))
                    .unwrap_or_else(NodeResult::Error);
                seg.job.nodes.lock().unwrap_or_else(|e| e.into_inner()).insert(local, Some(Arc::new(result)));
            })
            .expect("spawn study query");
        None
    }

    fn start_resolve(&mut self, loc: &Loc, ranges: Vec<Vec<f32>>, who: &[usize]) {
        let key = self.steps.clone();
        if self.segments.contains_key(&key) {
            return;
        }
        let names: Vec<&str> = who.iter().map(|&p| self.spot.names[p].as_str()).collect();
        let note = format!(
            "{}: {} almost never reach{} this point (under {:.1}% of the range), so the spot was re-solved from here with that range widened.",
            loc.state.street(),
            names.join(" and "),
            if names.len() == 1 { "es" } else { "" },
            100.0 * self.settings.solver.off_equilibrium_threshold
        );
        let seg = Segment {
            players: loc.seg.players.clone(),
            board: loc.board.clone(),
            root: loc.state,
            job: Arc::default(),
        };
        self.add_segment(key, seg, ranges, vec![note]);
    }

    /// Takes `action` at the current decision.
    pub fn act(&mut self, action: Action) -> Result<(), String> {
        let loc = self.locate();
        if !matches!(loc.end, LocEnd::Decision) {
            return Err("nobody is to act here".into());
        }
        if !matches!(loc.state.flow(action), Flow::Continue(_) | Flow::StreetOver(_) | Flow::Won(_)) {
            return Err("illegal action".into());
        }
        self.steps.push(Step::Action(action));
        self.collect();
        Ok(())
    }

    /// Cards that can come next (not on the board).
    pub fn live_cards(&self) -> Vec<Card> {
        let dead = CardSet::from_cards(&self.board());
        Card::all().filter(|c| !dead.contains(*c)).collect()
    }

    /// Deals the next street's card at the end of a street and starts
    /// solving that street.
    pub fn deal(&mut self, card: Card) -> Result<(), String> {
        let loc = self.locate();
        let LocEnd::StreetOver(end) = loc.end else { return Err("the street isn't over".into()) };
        if loc.board.contains(&card) {
            return Err(format!("{card} is already on the board"));
        }
        let Some(next) = end.street().next() else { return Err("the river is the last street".into()) };
        let active: Vec<usize> = (0..end.len()).filter(|&i| !end.folded(i)).collect();
        if active.iter().filter(|&&i| end.stack(i) > 0).count() < 2 {
            return Err("all-in: no more decisions".into());
        }
        let (mut ranges, root) = {
            let guard = loc.seg.job.solver.read().unwrap_or_else(|e| e.into_inner());
            let solver = guard.as_ref().ok_or("this street isn't solved yet")?;
            let now = solver.ranges_after(&loc.local).map_err(|e| e.to_string())?;
            let root = solver.ranges_after(&[]).map_err(|e| e.to_string())?;
            (now, root)
        };
        let threshold = self.settings.solver.off_equilibrium_threshold;
        let mut widened = Vec::new();
        for &i in &active {
            if fraction(&ranges[i], &root[i]) < threshold {
                for (w, &r) in ranges[i].iter_mut().zip(&root[i]) {
                    *w = w.max(0.05 * r);
                }
                widened.push(self.spot.names[loc.seg.players[i]].clone());
            }
        }
        let mut key = self.steps.clone();
        key.push(Step::Card(card));
        if !self.segments.contains_key(&key) {
            let mut board = loc.board.clone();
            board.push(card);
            let players: Vec<usize> = active.iter().map(|&i| loc.seg.players[i]).collect();
            let seg_ranges: Vec<Vec<f32>> = active.iter().map(|&i| ranges[i].clone()).collect();
            let pot = end.pot();
            let root = if active.len() == 2 {
                AnyState::Hu(BettingState::street_start(
                    next,
                    pot,
                    [end.stack(active[0]), end.stack(active[1])],
                    BB,
                ))
            } else {
                let st = [end.stack(active[0]), end.stack(active[1]), end.stack(active[2])];
                AnyState::Multi(MultiState::street_start(next, pot, st, [false; 3], BB))
            };
            let mut notes = Vec::new();
            if !widened.is_empty() {
                notes.push(format!(
                    "{next}: {} rarely get{} here, so {} range was widened to keep the solve meaningful.",
                    widened.join(" and "),
                    if widened.len() == 1 { "s" } else { "" },
                    if widened.len() == 1 { "that" } else { "those" }
                ));
            }
            let seg = Segment { players, board, root, job: Arc::default() };
            self.steps.push(Step::Card(card));
            self.add_segment(key, seg, seg_ranges, notes);
        } else {
            self.steps.push(Step::Card(card));
        }
        self.collect();
        Ok(())
    }

    /// Goes back one step (an action or a dealt card); `false` at the root.
    pub fn back(&mut self) -> bool {
        let popped = self.steps.pop().is_some();
        self.collect();
        popped
    }

    /// Back to the spot's root.
    pub fn home(&mut self) {
        self.steps.clear();
        self.collect();
    }

    /// Drops solves that are off the current line: unfinished ones are
    /// cancelled; finished ones are kept (within limits) for going back.
    fn collect(&mut self) {
        let chain = self.locate().chain;
        let mut drop_keys = Vec::new();
        for (key, seg) in &self.segments {
            if chain.contains(key) {
                continue;
            }
            let st = seg.job.status();
            if !st.done || st.error.is_some() {
                seg.job.cancel.store(true, Ordering::Relaxed);
                drop_keys.push(key.clone());
            }
        }
        for k in &drop_keys {
            self.segments.remove(k);
        }
        self.kept.retain(|k| self.segments.contains_key(k) && !chain.contains(k));
        for key in self.segments.keys() {
            if !chain.contains(key) && !self.kept.contains(key) {
                self.kept.push(key.clone());
            }
        }
        // Chain segments become the most recent when they leave the line.
        let is_flop = |s: &Study, k: &Vec<Step>| s.segments[k].root.street() == Street::Flop;
        while self.kept.iter().filter(|k| is_flop(self, k)).count() > KEEP_OFF_LINE_FLOPS {
            let i = self.kept.iter().position(|k| is_flop(self, k)).unwrap();
            let k = self.kept.remove(i);
            self.segments.remove(&k);
        }
        while self.kept.len() > KEEP_OFF_LINE {
            let k = self.kept.remove(0);
            self.segments.remove(&k);
        }
    }

    /// Number of solves held (on and off the line).
    pub fn num_solves(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use holdem::card::parse_cards;
    use std::time::Duration;

    fn settings(players: usize) -> Arc<StudySettings> {
        let mut cfg = TrainerConfig { players, ..TrainerConfig::default() };
        cfg.solver.flop_mode = FlopMode::Fast;
        cfg.solver.flop_turn_samples = 3;
        cfg.solver.flop_river_samples = 2;
        cfg.solver.max_iterations = 60;
        cfg.solver.target_exploitability_pct = 2.0;
        cfg.solver.multiway_turn_samples = 2;
        cfg.solver.multiway_river_samples = 2;
        cfg.solver.multiway_target_pct = 5.0;
        Arc::new(StudySettings::from_config(&cfg, Arc::new(ChartSet::builtin(players))))
    }

    /// Polls until the position stops changing state (solves and queries done).
    fn settle(study: &mut Study) -> StudyPos {
        for _ in 0..6000 {
            match study.poll() {
                StudyPos::Solving(_) | StudyPos::Loading => std::thread::sleep(Duration::from_millis(10)),
                p => return p,
            }
        }
        panic!("study didn't settle");
    }

    #[test]
    fn preflop_walks_the_charts() {
        let mut nav = PreflopNav::new(settings(6));
        let node = nav.node().unwrap();
        assert_eq!(node.position, "UTG");
        assert_eq!(node.actions[0], Action::Fold);
        let f: f64 = node.range_strategy.iter().sum();
        assert!((f - 1.0).abs() < 1e-6);
        // UTG opens a minority of hands; AA always continues.
        assert!(node.range_strategy[0] > 0.6, "{:?}", node.range_strategy);
        let aa = node.grid[HandClass::new(12, 12, false).grid_index()].as_ref().unwrap();
        assert!(aa.strategy[0] < 0.01);
        // Fold to the button, who opens; the blinds: SB folds, BB calls.
        for _ in 0..3 {
            nav.act(0).unwrap();
        }
        let btn = nav.node().unwrap();
        assert_eq!(btn.position, "BTN");
        let raise = btn.actions.iter().position(|a| matches!(a, Action::Raise(_))).unwrap();
        nav.act(raise).unwrap();
        nav.act(0).unwrap(); // SB folds
        let bb = nav.node().unwrap();
        assert_eq!(bb.position, "BB");
        let call = bb.actions.iter().position(|&a| a == Action::Call).unwrap();
        nav.act(call).unwrap();
        let PreflopOutcome::Flop(seats) = nav.outcome() else { panic!("{:?}", nav.outcome()) };
        assert_eq!(seats.len(), 2);
        let spot = nav.spot().unwrap();
        assert_eq!(spot.names, vec!["BB", "BTN"]);
        // The button's opening range is much wider than UTG's but not everything.
        let btn_combos = range_combos(&spot.ranges[1], &[]);
        assert!((250.0..900.0).contains(&btn_combos), "{btn_combos}");
        assert!(spot.preflop.starts_with("BTN"), "{}", spot.preflop);
        // Back undoes one action at a time.
        assert!(nav.back());
        assert_eq!(nav.outcome(), PreflopOutcome::Deciding);
        assert_eq!(nav.node().unwrap().position, "BB");
    }

    #[test]
    fn typed_spots_are_validated() {
        let names = Spot::default_names(2);
        let ok = Spot::from_text(names.clone(), &["QQ+,AK".into(), "TT-77,AQs".into()], 6.0, 97.0, "Ks7d2c");
        let spot = ok.unwrap();
        assert_eq!(spot.pot, 600);
        assert_eq!(spot.street(), Street::Flop);
        assert!(
            Spot::from_text(names.clone(), &["QQ+".into(), "nonsense".into()], 6.0, 97.0, "Ks7d2c").is_err()
        );
        assert!(Spot::from_text(names.clone(), &["QQ+".into(), "KK".into()], 6.0, 97.0, "KsKd").is_err());
        // KK is dead on a board with three kings... and two of them in hand.
        let e = Spot::from_text(names, &["QQ+".into(), "KsKh".into()], 6.0, 97.0, "Ks7d2c").unwrap_err();
        assert!(e.contains("empty"), "{e}");
    }

    #[test]
    fn browse_a_river_spot() {
        let names = Spot::default_names(2);
        let spot =
            Spot::from_text(names, &["QQ+,AK,76s".into(), "TT-77,AQs,KQs".into()], 10.0, 50.0, "Ks7d2c9h3s")
                .unwrap();
        let mut study = Study::new(settings(2), spot).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(0));
        assert_eq!(v.actions[0], Action::Check);
        assert!((v.range_strategy.iter().sum::<f64>() - 1.0).abs() < 1e-3);
        let info = v.info.as_ref().unwrap();
        assert_eq!(info.strategy.len(), v.actions.len() * info.num_hands());
        // Ranges: OOP's full range, IP's too.
        assert!(
            (range_combos(&v.ranges[0], &study.board())
                - range_combos(&study.spot().ranges[0], &study.board()))
            .abs()
                < 1.0
        );
        // Check, then IP bets: OOP faces a bet.
        study.act(Action::Check).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(1));
        let bet = *v.actions.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
        study.act(bet).unwrap();
        let pos = settle(&mut study);
        let v = match pos {
            StudyPos::Decision(v) => v,
            other => panic!("{other:?}"),
        };
        assert_eq!(v.actor, Some(0));
        assert!(v.to_call > 0);
        study.act(Action::Call).unwrap();
        assert!(matches!(settle(&mut study), StudyPos::Over { .. }));
        assert_eq!(study.line().len(), 4);
        assert!(study.back());
        study.act(Action::Fold).unwrap();
        match settle(&mut study) {
            StudyPos::Over { text, .. } => assert!(text.starts_with("IP wins"), "{text}"),
            other => panic!("{other:?}"),
        }
        study.home();
        assert!(study.at_root());
    }

    #[test]
    fn turn_cards_start_a_new_solve() {
        let names = Spot::default_names(2);
        let spot = Spot::from_text(
            names,
            &["QQ+,AK,T9s".into(), "JJ-88,AQs,KQs,QJs".into()],
            10.0,
            40.0,
            "Ks7d2c9h",
        )
        .unwrap();
        let mut study = Study::new(settings(2), spot).unwrap();
        assert!(matches!(settle(&mut study), StudyPos::Decision(_)));
        study.act(Action::Check).unwrap();
        settle(&mut study);
        study.act(Action::Check).unwrap();
        let StudyPos::NeedCard { next, view } = settle(&mut study) else { panic!() };
        assert_eq!(next, Street::River);
        assert!(view.unwrap().actor.is_none());
        assert!(study.deal(parse_cards("9h").unwrap()[0]).is_err(), "card on the board");
        study.deal(parse_cards("3s").unwrap()[0]).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(0));
        assert_eq!(study.board().len(), 5);
        assert_eq!(study.num_solves(), 2);
        // Back to the card choice, a different river: both solves are kept.
        study.back();
        study.deal(parse_cards("4s").unwrap()[0]).unwrap();
        assert!(matches!(settle(&mut study), StudyPos::Decision(_)));
        assert_eq!(study.num_solves(), 3);
        study.back();
        study.deal(parse_cards("3s").unwrap()[0]).unwrap();
        assert!(matches!(study.poll(), StudyPos::Decision(_) | StudyPos::Loading), "cached");
    }

    #[test]
    fn rare_lines_are_re_solved() {
        // OOP is polarized (aces or air); IP holds a pure bluff catcher, so
        // IP never raises a bet.
        let names = Spot::default_names(2);
        let spot =
            Spot::from_text(names, &["AA,QJo".into(), "TT".into()], 10.0, 100.0, "AsKd8c6h2s").unwrap();
        let mut study = Study::new(settings(2), spot).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        let bet = (0..v.actions.len())
            .find(|&a| matches!(v.actions[a], Action::Raise(_)) && v.range_strategy[a] > 0.05)
            .unwrap_or_else(|| panic!("OOP never bets: {:?} {:?}", v.labels, v.range_strategy));
        study.act(v.actions[bet]).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(1));
        let raise = (0..v.actions.len())
            .find(|&a| matches!(v.actions[a], Action::Raise(_)) && v.range_strategy[a] < 0.005)
            .unwrap_or_else(|| panic!("no rare raise: {:?} {:?}", v.labels, v.range_strategy));
        study.act(v.actions[raise]).unwrap();
        let pos = settle(&mut study);
        let StudyPos::Decision(v) = pos else { panic!("{pos:?}") };
        assert_eq!(v.actor, Some(0));
        assert!(study.notes().iter().any(|n| n.contains("re-solved")), "{:?}", study.notes());
        assert_eq!(study.num_solves(), 2);
        // Going back keeps the finished re-solve for later.
        study.back();
        assert!(matches!(settle(&mut study), StudyPos::Decision(_)));
        assert_eq!(study.num_solves(), 2);
    }

    #[test]
    fn three_way_spots() {
        let names = Spot::default_names(3);
        let spot = Spot::from_text(
            names,
            &["QQ+,AK".into(), "TT-77,AQs".into(), "66-22,KQs".into()],
            9.0,
            50.0,
            "Ks7d2c9h",
        )
        .unwrap();
        let mut study = Study::new(settings(6), spot).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert!(v.multiway);
        assert_eq!(v.actor, Some(0));
        assert_eq!(v.ranges.len(), 3);
        study.act(Action::Check).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(1));
        let bet = *v.actions.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
        study.act(bet).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(2));
        study.act(Action::Fold).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert_eq!(v.actor, Some(0));
        assert!(v.folded[2]);
        assert!(v.ranges[2].iter().all(|&w| w == 0.0));
        study.act(Action::Call).unwrap();
        let StudyPos::NeedCard { .. } = settle(&mut study) else { panic!() };
        // Two players left: the river is solved heads-up.
        study.deal(parse_cards("3s").unwrap()[0]).unwrap();
        let StudyPos::Decision(v) = settle(&mut study) else { panic!() };
        assert!(!v.multiway);
        assert_eq!(v.actor, Some(0));
    }
}
