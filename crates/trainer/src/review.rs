//! Background solver review of the hero's postflop decisions.
//!
//! A [`Review`] runs on its own thread while the hand is being played:
//! 1. when the flop is dealt it builds both players' ranges from the
//!    preflop line (charts for the GTO variant; the villain's AI style for the
//!    "actual range" variant) and starts solving the flop;
//! 2. when a street's betting is over it grades every hero decision on that
//!    street, narrows both ranges along the actual line, and solves the next
//!    street as soon as it is dealt (re-solving);
//! 3. if a player takes a line the solver almost never takes, the node is
//!    re-solved from that point with floored ranges so the advice stays
//!    meaningful.

use crate::ai::profile_policy;
use crate::config::{FlopMode, SolverBackend, SolverSettings, VillainRange};
use crate::solver_tag::{SolveSummary, SolverTag};
use crate::fmt::{action_label, ActionContext, BB};
use crate::preflop::{chart_policy, describe, line_range, ChartSet, PreDecision};
use crate::profiles::Profile;
use holdem::combo::{combo_index, COMBO_CLASS, NUM_CLASSES};
use holdem::game::Phase;
use holdem::menu::postflop_categorized;
use holdem::{Action, BetMenu, Card, Category, HandState, Street};
use serde::{Deserialize, Serialize};
use solver::multi::{MultiConfig, MultiSolver, MultiState};
use solver::{
    BettingState, ChanceMode, NodeInfo, Progress, SolveConfig, SolveOptions, Solver, SolverError, Step,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Runs solves one at a time (each uses every core and up to gigabytes of
/// memory), oldest hand first. A plain mutex isn't fair: a later hand's
/// flop solve could slip in between a hand's flop and turn solves, so every
/// review finished late.
struct SolveQueue {
    state: Mutex<QueueState>,
    cv: Condvar,
}

struct QueueState {
    busy: bool,
    next_ticket: u64,
    /// Waiting solves as `(hand id, ticket)`: the smallest goes next.
    waiting: std::collections::BTreeSet<(u64, u64)>,
}

static SOLVE_QUEUE: SolveQueue = SolveQueue {
    state: Mutex::new(QueueState { busy: false, next_ticket: 0, waiting: std::collections::BTreeSet::new() }),
    cv: Condvar::new(),
};

/// Exclusive right to run a solve; released on drop.
struct SolveTurn;

impl Drop for SolveTurn {
    fn drop(&mut self) {
        SOLVE_QUEUE.state.lock().unwrap_or_else(|e| e.into_inner()).busy = false;
        SOLVE_QUEUE.cv.notify_all();
    }
}

/// Waits for this hand's turn to solve; `None` if cancelled while waiting.
/// `on_wait` is told how many solves are ahead whenever that changes.
fn wait_solve_turn(hand: u64, cancel: &AtomicBool, mut on_wait: impl FnMut(usize)) -> Option<SolveTurn> {
    let q = &SOLVE_QUEUE;
    let mut st = q.state.lock().unwrap_or_else(|e| e.into_inner());
    let key = (hand, st.next_ticket);
    st.next_ticket += 1;
    st.waiting.insert(key);
    let mut reported = usize::MAX;
    loop {
        if cancel.load(Ordering::Relaxed) {
            st.waiting.remove(&key);
            q.cv.notify_all();
            return None;
        }
        if !st.busy && st.waiting.first() == Some(&key) {
            st.waiting.remove(&key);
            st.busy = true;
            return Some(SolveTurn);
        }
        let ahead = st.waiting.range(..key).count() + usize::from(st.busy);
        if ahead != reported {
            reported = ahead;
            on_wait(ahead);
        }
        st = q.cv.wait_timeout(st, Duration::from_millis(200)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

pub(crate) const CANCELLED: &str = "cancelled";

/// Which villain range a solve used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Variant {
    Gto,
    Actual,
}

impl Variant {
    pub fn label(self) -> &'static str {
        match self {
            Variant::Gto => "GTO ranges",
            Variant::Actual => "villain's actual range",
        }
    }
}

/// Chess-style classification of a decision by EV loss (% of the pot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Verdict {
    Best,
    Good,
    Inaccuracy,
    Mistake,
    Blunder,
}

impl Verdict {
    pub fn from_loss_pct(pct: f32) -> Verdict {
        if pct <= 1.0 {
            Verdict::Best
        } else if pct <= 3.0 {
            Verdict::Good
        } else if pct <= 6.0 {
            Verdict::Inaccuracy
        } else if pct <= 12.0 {
            Verdict::Mistake
        } else {
            Verdict::Blunder
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Verdict::Best => "Best",
            Verdict::Good => "Good",
            Verdict::Inaccuracy => "Inaccuracy",
            Verdict::Mistake => "Mistake",
            Verdict::Blunder => "Blunder",
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Verdict::Best => "✓",
            Verdict::Good => "✓",
            Verdict::Inaccuracy => "?!",
            Verdict::Mistake => "?",
            Verdict::Blunder => "??",
        }
    }

    pub const ALL: [Verdict; 5] =
        [Verdict::Best, Verdict::Good, Verdict::Inaccuracy, Verdict::Mistake, Verdict::Blunder];
}

/// Strategy of one hand class at a node (for the 13x13 grid).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassCell {
    /// Total reach weight of the class's combos at this node.
    pub weight: f32,
    /// Weighted average strategy over the class's combos (saved to 4
    /// decimals: it is only displayed).
    #[serde(with = "quantized")]
    pub strategy: Vec<f32>,
}

/// Serializes frequencies in 1/10000ths, which keeps saved grids small.
mod quantized {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &[f32], s: S) -> Result<S::Ok, S::Error> {
        let q: Vec<u16> = v.iter().map(|&x| (x.clamp(0.0, 1.0) * 10000.0).round() as u16).collect();
        q.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
        Ok(Vec::<u16>::deserialize(d)?.into_iter().map(|x| x as f32 / 10000.0).collect())
    }
}

/// The solver's verdict on one hero decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionReview {
    pub variant: Variant,
    pub street: Street,
    pub board: Vec<Card>,
    /// Earlier actions on this street, e.g. "BB Check · SB Bet 2bb (33%)".
    pub history: String,
    pub pot: holdem::Chips,
    pub to_call: holdem::Chips,
    pub hero_combo: usize,
    pub actions: Vec<Action>,
    pub labels: Vec<String>,
    pub chosen: usize,
    pub best: usize,
    /// Solver mix for the hero's hand.
    pub hero_strategy: Vec<f32>,
    /// EV of each action for the hero's hand (chips, measured from the decision).
    pub hero_ev: Vec<f32>,
    pub ev_loss: f32,
    pub ev_loss_pct: f32,
    pub verdict: Verdict,
    /// Range-wide action frequencies.
    pub range_strategy: Vec<f64>,
    /// Per-class strategy (grid order); `None` where the class isn't in range.
    pub grid: Vec<Option<ClassCell>>,
    pub exploitability_pct: f64,
    pub notes: Vec<String>,
    /// Where the decision was made (for the session's frequency score).
    #[serde(default)]
    pub spot: SpotInfo,
    /// The solver that produced this review.
    #[serde(default)]
    pub solver: SolverTag,
}

/// The kind of spot a decision was made in, and the frequency category of
/// each of its actions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpotInfo {
    /// Players in the pot (2 or 3).
    pub players: u8,
    /// Last to act on the street.
    pub in_position: bool,
    /// Bets and raises already made on this street (0 = none, 1 = facing a
    /// bet, 2+ = facing a raise).
    pub facing: u8,
    /// Index of the decision in the hand's action history.
    pub history_index: usize,
    /// Category of each action (see [`holdem::Category`]).
    pub categories: Vec<String>,
}

impl SpotInfo {
    fn new(
        players: usize,
        in_position: bool,
        sit: &holdem::Situation,
        menu: &holdem::PostflopMenu,
        actions: &[Action],
        history_index: usize,
    ) -> SpotInfo {
        let cats = postflop_categorized(sit, menu);
        let categories = actions
            .iter()
            .map(|a| match cats.iter().find(|(x, _)| x == a) {
                Some((_, c)) => c.to_string(),
                // Not from the menu (can't happen for the street's root menu).
                None => match a {
                    Action::Raise(to) if *to >= sit.max_to => Category::AllIn.to_string(),
                    other => format!("{other:?}"),
                },
            })
            .collect();
        SpotInfo {
            players: players as u8,
            in_position,
            facing: sit.num_raises.min(u8::MAX as u32) as u8,
            history_index,
            categories,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReviewStatus {
    pub stage: String,
    /// When the current stage started (to show elapsed time).
    pub stage_since: Option<Instant>,
    pub progress: Option<Progress>,
    pub decisions: Vec<DecisionReview>,
    pub notes: Vec<String>,
    pub finished: bool,
    /// Stopped before completion (the hand was abandoned or the player
    /// moved on to the next hand).
    pub cancelled: bool,
    pub error: Option<String>,
}

/// Static inputs of a review.
#[derive(Clone)]
pub struct ReviewSetup {
    /// Hand number: earlier hands' solves run first.
    pub hand_id: u64,
    pub hero: usize,
    /// Style of every seat (`None` for the hero).
    pub profiles: Vec<Option<Profile>>,
    pub charts: Arc<ChartSet>,
    pub menu: BetMenu,
    pub settings: SolverSettings,
    /// Rake fraction and cap (chips), as at the table.
    pub rake_rate: f64,
    pub rake_cap: holdem::Chips,
}

/// The hand as seen by the review thread.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub state: HandState,
    pub preflop: Vec<PreDecision>,
}

struct Shared {
    snap: Mutex<Snapshot>,
    cv: Condvar,
    status: Mutex<ReviewStatus>,
    cancel: AtomicBool,
}

impl Shared {
    fn wait_for(&self, pred: impl Fn(&Snapshot) -> bool) -> Result<Snapshot, String> {
        let mut snap = self.snap.lock().unwrap();
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            if pred(&snap) {
                return Ok(snap.clone());
            }
            snap = self.cv.wait_timeout(snap, Duration::from_millis(200)).unwrap().0;
        }
    }

    fn with_status(&self, f: impl FnOnce(&mut ReviewStatus)) {
        f(&mut self.status.lock().unwrap());
    }

    fn set_stage(&self, s: String) {
        self.with_status(|st| {
            st.stage = s;
            st.stage_since = Some(Instant::now());
            st.progress = None;
        });
    }

    fn note(&self, s: String) {
        self.with_status(|st| st.notes.push(s));
    }
}

/// Handle to a running review.
pub struct Review {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl Review {
    pub fn start(setup: ReviewSetup, initial: Snapshot) -> Review {
        let shared = Arc::new(Shared {
            snap: Mutex::new(initial),
            cv: Condvar::new(),
            status: Mutex::new(ReviewStatus { stage: "Waiting for the flop".into(), ..Default::default() }),
            cancel: AtomicBool::new(false),
        });
        let worker = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("review".into())
            .spawn(move || {
                // A panic must not leave the review "solving" forever (and
                // its message hidden behind the full-screen UI): report it.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Worker::new(&worker, &setup).run()
                }))
                .unwrap_or_else(|e| {
                    let msg = e
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| e.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".into());
                    Err(format!("internal error: {msg}"))
                });
                worker.with_status(|st| {
                    st.finished = true;
                    st.progress = None;
                    match result {
                        Ok(()) => st.stage = "Review complete".into(),
                        Err(e) if e == CANCELLED => {
                            st.stage = "Review cancelled".into();
                            st.cancelled = true;
                        }
                        Err(e) => {
                            st.stage = "Review failed".into();
                            st.error = Some(e);
                        }
                    }
                });
            })
            .expect("spawn review thread");
        Review { shared, handle: Some(handle) }
    }

    /// A review read back from a save: finished, with no thread.
    pub fn finished(snap: Snapshot, status: ReviewStatus) -> Review {
        let status = ReviewStatus { finished: true, progress: None, stage_since: None, ..status };
        let shared = Arc::new(Shared {
            snap: Mutex::new(snap),
            cv: Condvar::new(),
            status: Mutex::new(status),
            cancel: AtomicBool::new(false),
        });
        Review { shared, handle: None }
    }

    /// Runs `f` on the current status without copying it.
    pub fn with_status<R>(&self, f: impl FnOnce(&ReviewStatus) -> R) -> R {
        f(&self.shared.status.lock().unwrap())
    }

    /// Pushes the latest state of the hand.
    pub fn update(&self, snap: Snapshot) {
        *self.shared.snap.lock().unwrap() = snap;
        self.shared.cv.notify_all();
    }

    pub fn status(&self) -> ReviewStatus {
        self.shared.status.lock().unwrap().clone()
    }

    pub fn is_finished(&self) -> bool {
        self.shared.status.lock().unwrap().finished
    }

    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
        self.shared.cv.notify_all();
    }

    /// Blocks until the review thread exits.
    pub fn wait(mut self) -> ReviewStatus {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.status()
    }
}

impl Drop for Review {
    fn drop(&mut self) {
        // An abandoned hand never finishes; make sure the thread exits.
        if !self.is_finished() {
            let st = self.shared.snap.lock().unwrap();
            let done = st.state.phase == Phase::Finished;
            drop(st);
            if !done {
                self.cancel();
            }
        }
    }
}

/// `[out of position, in position]` for two seats after the flop.
pub fn postflop_order(state: &HandState, a: usize, b: usize) -> [usize; 2] {
    let n = state.num_players();
    let offset = |s: usize| (s + n - state.button - 1) % n;
    if offset(a) < offset(b) {
        [a, b]
    } else {
        [b, a]
    }
}

/// Pot and per-seat stacks at the start of `street`.
pub fn street_start(state: &HandState, street: Street) -> (holdem::Chips, Vec<holdem::Chips>) {
    let n = state.num_players();
    let mut committed = state.total_commit.clone();
    for r in &state.history {
        if r.street >= street {
            committed[r.player] -= r.put_in;
        }
    }
    let pot = committed.iter().sum();
    let stacks = (0..n).map(|p| state.starting_stacks[p] - committed[p]).collect();
    (pot, stacks)
}

struct Worker<'a> {
    shared: &'a Shared,
    setup: &'a ReviewSetup,
    /// Optional dedicated thread pool for the solver.
    pool: Option<rayon::ThreadPool>,
}

impl<'a> Worker<'a> {
    fn new(shared: &'a Shared, setup: &'a ReviewSetup) -> Worker<'a> {
        let pool = if setup.settings.threads > 0 {
            rayon::ThreadPoolBuilder::new().num_threads(setup.settings.threads).build().ok()
        } else {
            None
        };
        Worker { shared, setup, pool }
    }

    fn variants(&self) -> Vec<Variant> {
        match self.setup.settings.villain_range {
            VillainRange::Gto => vec![Variant::Gto],
            VillainRange::Actual => vec![Variant::Actual],
            VillainRange::Both => vec![Variant::Gto, Variant::Actual],
        }
    }

    fn run(&self) -> Result<(), String> {
        let setup = self.setup;
        let snap = self.shared.wait_for(|s| s.state.board.len() >= 3 || s.state.phase == Phase::Finished)?;
        if snap.state.board.len() < 3 {
            self.shared.set_stage("No flop: nothing to review".into());
            return Ok(());
        }
        let hero = setup.hero;
        let st0 = &snap.state;
        let hero_combo = combo_index(st0.hole[hero][0], st0.hole[hero][1]);
        let variants = self.variants();
        // Seats that saw the flop.
        let flop_seats: Vec<usize> = in_hand_at(st0, Street::Flop);

        // Preflop ranges of everyone who saw the flop, per variant.
        let mut ranges: Vec<HashMap<usize, Vec<f32>>> = Vec::new();
        for &v in &variants {
            let mut map = HashMap::new();
            for &seat in &flop_seats {
                let villain_profile = setup.profiles.get(seat).cloned().flatten();
                let (r, uncovered) = match (&villain_profile, v) {
                    (Some(prof), Variant::Actual) => {
                        line_range(&snap.preflop, seat, |d| profile_policy(&setup.charts, prof, d))
                    }
                    _ => line_range(&snap.preflop, seat, |d| chart_policy(&setup.charts, d)),
                };
                for idx in uncovered {
                    let d = &snap.preflop[idx];
                    let who =
                        if seat == hero { "You".to_string() } else { st0.position_name(seat).to_string() };
                    let msg = format!(
                        "{who} chose to {} preflop ({}), which the charts never do there; that range was approximated by the hands the charts continue with.",
                        describe(d.action, &d.avail),
                        st0.position_name(seat)
                    );
                    let dup = self.shared.status.lock().unwrap().notes.contains(&msg);
                    if !dup {
                        self.shared.note(msg);
                    }
                }
                map.insert(seat, r.into_weights());
            }
            ranges.push(map);
        }

        for street in [Street::Flop, Street::Turn, Street::River] {
            let need = street.board_len();
            let snap =
                self.shared.wait_for(|s| s.state.board.len() >= need || s.state.phase == Phase::Finished)?;
            if snap.state.board.len() < need {
                break;
            }
            let active = in_hand_at(&snap.state, street);
            if !active.contains(&hero) {
                break;
            }
            let (_, by_seat) = street_start(&snap.state, street);
            if active.iter().filter(|&&s| by_seat[s] > 0).count() < 2 {
                break; // all-in: no more decisions
            }
            match active.len() {
                2 => {
                    let seats = postflop_order(&snap.state, active[0], active[1]);
                    self.heads_up_street(street, seats, hero_combo, &variants, &mut ranges)?;
                }
                3 => {
                    let seats = postflop_order_n(&snap.state, &active);
                    self.three_way_street(
                        street,
                        [seats[0], seats[1], seats[2]],
                        hero_combo,
                        &variants,
                        &mut ranges,
                    )?;
                }
                k => {
                    self.shared.note(format!(
                        "{street}: {k} players saw this street; pots with 4+ players aren't analyzed (yet)."
                    ));
                    let _ = self
                        .shared
                        .wait_for(|s| s.state.street > street || s.state.phase == Phase::Finished)?;
                }
            }
            let snap = self.shared.snap.lock().unwrap().clone();
            if snap.state.phase == Phase::Finished && snap.state.board.len() == need {
                break;
            }
        }
        Ok(())
    }

    /// Solves and grades one street of a heads-up pot (`seats` = [OOP, IP]).
    fn heads_up_street(
        &self,
        street: Street,
        seats: [usize; 2],
        hero_combo: usize,
        variants: &[Variant],
        ranges: &mut [HashMap<usize, Vec<f32>>],
    ) -> Result<(), String> {
        let setup = self.setup;
        let hero = setup.hero;
        let hero_p = if seats[0] == hero { 0 } else { 1 };
        let need = street.board_len();
        let snap = self.shared.snap.lock().unwrap().clone();
        let board: Vec<Card> = snap.state.board[..need].to_vec();
        let (pot, by_seat) = street_start(&snap.state, street);
        let stacks = [by_seat[seats[0]], by_seat[seats[1]]];
        let root = BettingState::street_start(street, pot, stacks, BB);

        let mut solved: Vec<Option<(Solver, SolveSummary)>> = Vec::new();
        for (vi, &v) in variants.iter().enumerate() {
            let label = format!("{street} ({})", v.label());
            let pair = [ranges[vi][&seats[0]].clone(), ranges[vi][&seats[1]].clone()];
            let cfg = self.make_config(&board, &pair, root, hero_combo, hero_p);
            match self.solve(cfg, &label) {
                Ok(x) => solved.push(Some(x)),
                Err(e) if e == CANCELLED => return Err(e),
                Err(e) => {
                    self.shared.note(format!("{label}: {e}"));
                    solved.push(None);
                }
            }
        }

        self.shared.set_stage(format!("{street} solved; waiting for the street to finish"));
        let snap = self.shared.wait_for(|s| s.state.street > street || s.state.phase == Phase::Finished)?;
        let state = &snap.state;
        let actions: Vec<(usize, Action)> =
            state.history.iter().filter(|r| r.street == street).map(|r| (r.player, r.action)).collect();
        let history_index: Vec<usize> =
            state.history.iter().enumerate().filter(|(_, r)| r.street == street).map(|(i, _)| i).collect();
        // Betting state before each action.
        let mut states = vec![root];
        for &(_, a) in &actions {
            match states.last().unwrap().after(a) {
                Some(s) => states.push(s),
                None => break,
            }
        }
        let pos = |seat: usize| state.position_name(seat);

        for (vi, &v) in variants.iter().enumerate() {
            let Some((base, base_expl)) = &solved[vi] else { continue };
            let mut owned: Option<(Solver, SolveSummary)> = None;
            let mut offset = 0;
            for k in 0..actions.len() {
                let (seat, a) = actions[k];
                if seat != hero || k >= states.len() {
                    continue;
                }
                let (cur, cur_expl) = match &owned {
                    Some((s, e)) => (s, e.clone()),
                    None => (base, base_expl.clone()),
                };
                let path: Vec<Step> = actions[offset..k].iter().map(|&(_, a)| Step::Action(a)).collect();
                let mut info = cur.query(&path).map_err(|e| e.to_string())?;
                let mut notes = Vec::new();
                let mut expl = cur_expl;
                let off = info.reach_fraction.iter().any(|&f| f < setup.settings.off_equilibrium_threshold);
                if off && k > offset {
                    let who = if info.reach_fraction[1 - hero_p] < setup.settings.off_equilibrium_threshold {
                        "Villain"
                    } else {
                        "You"
                    };
                    let root_ranges = cur.ranges_after(&[]).map_err(|e| e.to_string())?;
                    let mut r = cur.ranges_after(&path).map_err(|e| e.to_string())?;
                    floor_ranges(
                        &mut r,
                        &root_ranges,
                        &info.reach_fraction,
                        setup.settings.off_equilibrium_threshold,
                    );
                    let label = format!("{street} re-solve ({})", v.label());
                    let cfg = self.make_config(&board, &r, states[k], hero_combo, hero_p);
                    match self.solve(cfg, &label) {
                        Ok((s, e)) => {
                            owned = Some((s, e.clone()));
                            offset = k;
                            info = owned.as_ref().unwrap().0.query(&[]).map_err(|e| e.to_string())?;
                            expl = e;
                            notes.push(format!(
                                "{who} took a line the solver almost never takes here, so this spot was re-solved from this point."
                            ));
                        }
                        Err(e) if e == CANCELLED => return Err(e),
                        Err(e) => notes.push(format!("Re-solve failed: {e}")),
                    }
                }
                let history = (0..k)
                    .map(|j| {
                        format!("{} {}", pos(actions[j].0), action_label(actions[j].1, &context(&states[j])))
                    })
                    .collect::<Vec<_>>()
                    .join(" · ");
                if street == Street::Flop && setup.settings.flop_mode == FlopMode::Fast {
                    notes.push("Flop solved with sampled river cards (fast mode): approximate.".into());
                }
                let spot = SpotInfo::new(
                    2,
                    hero_p == 1,
                    &states[k].situation(BB),
                    &setup.menu.postflop,
                    &info.actions,
                    history_index[k],
                );
                let review = grade(
                    &info,
                    &context(&states[k]),
                    states[k].pot,
                    hero_combo,
                    a,
                    v,
                    street,
                    &board,
                    history,
                    expl,
                    notes,
                    spot,
                )?;
                self.shared.with_status(|st| st.decisions.push(review));
            }

            if street != Street::River && state.board.len() > need {
                let cur = match &owned {
                    Some((s, _)) => s,
                    None => base,
                };
                let path: Vec<Step> = actions[offset..].iter().map(|&(_, a)| Step::Action(a)).collect();
                if let Ok(mut next) = cur.ranges_after(&path) {
                    let root_ranges = cur.ranges_after(&[]).map_err(|e| e.to_string())?;
                    let frac = [fraction(&next[0], &root_ranges[0]), fraction(&next[1], &root_ranges[1])];
                    if frac.iter().any(|&f| f < setup.settings.off_equilibrium_threshold) {
                        floor_ranges(
                            &mut next,
                            &root_ranges,
                            &frac,
                            setup.settings.off_equilibrium_threshold,
                        );
                    }
                    let [r0, r1] = next;
                    ranges[vi].insert(seats[0], r0);
                    ranges[vi].insert(seats[1], r1);
                }
            }
        }
        Ok(())
    }

    /// Solves (approximately) and grades one street of a three-way pot.
    fn three_way_street(
        &self,
        street: Street,
        seats: [usize; 3],
        hero_combo: usize,
        variants: &[Variant],
        ranges: &mut [HashMap<usize, Vec<f32>>],
    ) -> Result<(), String> {
        let setup = self.setup;
        let s = &setup.settings;
        let hero = setup.hero;
        let hero_p = seats.iter().position(|&x| x == hero).unwrap();
        let need = street.board_len();
        let snap = self.shared.snap.lock().unwrap().clone();
        let board: Vec<Card> = snap.state.board[..need].to_vec();
        let (pot, by_seat) = street_start(&snap.state, street);
        let root = MultiState::street_start(street, pot, seats.map(|x| by_seat[x]), [false; 3], BB);

        let mut solved: Vec<Option<(MultiSolver, SolveSummary)>> = Vec::new();
        for (vi, &v) in variants.iter().enumerate() {
            let label = format!("{street} 3-way ({})", v.label());
            let mut rs: [Vec<f32>; 3] = seats.map(|x| ranges[vi][&x].clone());
            let orbit = solver::iso_orbit(&board, hero_combo);
            let max_w = rs[hero_p].iter().copied().fold(0f32, f32::max);
            let floor = if max_w > 0.0 { s.hero_hand_floor * max_w } else { 1.0 };
            for &c in &orbit {
                rs[hero_p][c] = rs[hero_p][c].max(floor);
            }
            let mut keep: [Vec<usize>; 3] = Default::default();
            keep[hero_p] = orbit;
            let cfg = multi_config(
                s,
                &setup.menu.postflop,
                &board,
                rs,
                root,
                keep,
                setup.rake_rate,
                setup.rake_cap,
            );
            match self.solve_multi(cfg, &label) {
                Ok(x) => solved.push(Some(x)),
                Err(e) if e == CANCELLED => return Err(e),
                Err(e) => {
                    self.shared.note(format!("{label}: {e}"));
                    solved.push(None);
                }
            }
        }

        self.shared.set_stage(format!("{street} solved; waiting for the street to finish"));
        let snap = self.shared.wait_for(|s| s.state.street > street || s.state.phase == Phase::Finished)?;
        let state = &snap.state;
        let actions: Vec<(usize, Action)> =
            state.history.iter().filter(|r| r.street == street).map(|r| (r.player, r.action)).collect();
        let history_index: Vec<usize> =
            state.history.iter().enumerate().filter(|(_, r)| r.street == street).map(|(i, _)| i).collect();
        let mut states = vec![root];
        for &(seat, a) in &actions {
            debug_assert_eq!(seats[states.last().unwrap().to_act], seat, "3-way state out of sync");
            match states.last().unwrap().after(a) {
                Some(st) => states.push(st),
                None => break,
            }
        }
        let pos = |seat: usize| state.position_name(seat);
        for (vi, &v) in variants.iter().enumerate() {
            let Some((solver, conv)) = &solved[vi] else { continue };
            for k in 0..actions.len() {
                let (seat, a) = actions[k];
                if seat != hero || k >= states.len() {
                    continue;
                }
                let path: Vec<Step> = actions[..k].iter().map(|&(_, a)| Step::Action(a)).collect();
                let info = match solver.query(&path) {
                    Ok(i) => i,
                    Err(e) => {
                        self.shared.note(format!("{street}: this line isn't in the 3-way tree ({e})."));
                        break;
                    }
                };
                let history = (0..k)
                    .map(|j| {
                        format!(
                            "{} {}",
                            pos(actions[j].0),
                            action_label(actions[j].1, &context_multi(&states[j]))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" · ");
                let notes = vec![multiway_note(s.multiway_exact_removal)];
                let spot = SpotInfo::new(
                    3,
                    hero_p == 2,
                    &states[k].situation(BB),
                    &setup.menu.postflop,
                    &info.actions,
                    history_index[k],
                );
                let review = grade(
                    &info,
                    &context_multi(&states[k]),
                    states[k].pot,
                    hero_combo,
                    a,
                    v,
                    street,
                    &board,
                    history,
                    conv.clone(),
                    notes,
                    spot,
                )?;
                self.shared.with_status(|st| st.decisions.push(review));
            }
            if street != Street::River && state.board.len() > need {
                let path: Vec<Step> = actions.iter().map(|&(_, a)| Step::Action(a)).collect();
                if let Ok(next) = solver.ranges_after(&path) {
                    for (i, r) in next.into_iter().enumerate() {
                        if r.iter().any(|&w| w > 0.0) {
                            ranges[vi].insert(seats[i], r);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn make_config(
        &self,
        board: &[Card],
        ranges: &[Vec<f32>; 2],
        root: BettingState,
        hero_combo: usize,
        hero_p: usize,
    ) -> SolveConfig {
        let s = &self.setup.settings;
        let mut ranges = ranges.clone();
        let max_w = ranges[hero_p].iter().copied().fold(0f32, f32::max);
        let floor = if max_w > 0.0 { s.hero_hand_floor * max_w } else { 1.0 };
        // Floor the hero's hand and its suit-isomorphic images alike, so the
        // ranges stay symmetric and the solver can merge mirrored runouts.
        let orbit = solver::iso_orbit(board, hero_combo);
        for &c in &orbit {
            let w = &mut ranges[hero_p][c];
            *w = w.max(floor);
        }
        let mut cfg = hu_config(
            s,
            &self.setup.menu.postflop,
            board,
            ranges,
            root,
            self.setup.rake_rate,
            self.setup.rake_cap,
        );
        cfg.keep[hero_p] = orbit;
        cfg
    }

    fn solve(&self, cfg: SolveConfig, label: &str) -> Result<(Solver, SolveSummary), String> {
        if let Some(dir) = std::env::var_os("POKER_TRAINER_DUMP") {
            dump_config(std::path::Path::new(&dir), self.setup.hand_id, label, &cfg);
        }
        run_hu(cfg, &self.setup.settings, self.setup.hand_id, self.pool.as_ref(), label, self.shared)
    }

    fn solve_multi(&self, cfg: MultiConfig, label: &str) -> Result<(MultiSolver, SolveSummary), String> {
        run_multi(cfg, &self.setup.settings, self.setup.hand_id, self.pool.as_ref(), label, self.shared)
    }
}

/// Receives a solve's stage changes, notes and progress.
pub(crate) trait SolveSink {
    fn stage(&self, s: String);
    fn note(&self, s: String);
    fn progress(&self, p: Progress);
    fn cancel_flag(&self) -> &AtomicBool;
}

impl SolveSink for Shared {
    fn stage(&self, s: String) {
        self.set_stage(s);
    }
    fn note(&self, s: String) {
        Shared::note(self, s);
    }
    fn progress(&self, p: Progress) {
        self.with_status(|st| st.progress = Some(p));
    }
    fn cancel_flag(&self) -> &AtomicBool {
        &self.cancel
    }
}

/// What a 3-way solve approximates (shown with its results).
pub(crate) fn multiway_note(exact_removal: bool) -> String {
    format!(
        "3-way pot: approximate solve (no equilibrium guarantee for 3 players; {}sampled runouts; simplified later streets).",
        if exact_removal { "" } else { "opponents' hands treated as independent; " }
    )
}

/// Heads-up solver inputs as reviews and study mode use them: the root
/// street gets the full menu; a flop solve uses `flop_future_menu` for the
/// turn and river, and samples them in fast mode.
pub(crate) fn hu_config(
    s: &SolverSettings,
    menu: &holdem::PostflopMenu,
    board: &[Card],
    ranges: [Vec<f32>; 2],
    root: BettingState,
    rake_rate: f64,
    rake_cap: holdem::Chips,
) -> SolveConfig {
    let flop = root.street == Street::Flop;
    let seed = board.iter().fold(17u64, |h, c| h.wrapping_mul(31).wrapping_add(c.index() as u64));
    let mut cfg = SolveConfig::new(board.to_vec(), ranges, root, menu.clone());
    cfg.chance = if flop && s.flop_mode == FlopMode::Fast {
        ChanceMode::Sampled { turn: s.flop_turn_samples, river: s.flop_river_samples, seed }
    } else {
        ChanceMode::Exact
    };
    cfg.future_menu = if flop { Some(s.flop_future_menu.clone()) } else { None };
    cfg.big_blind = BB;
    cfg.prune = s.prune;
    cfg.max_memory_bytes = (s.max_memory_gb * 1e9) as u64;
    cfg.rake_rate = rake_rate;
    cfg.rake_cap = rake_cap as f64;
    cfg.backend = s.backend.into();
    cfg
}

/// Three-player solver inputs (sampled runouts, simplified later streets).
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_config(
    s: &SolverSettings,
    menu: &holdem::PostflopMenu,
    board: &[Card],
    ranges: [Vec<f32>; 3],
    root: MultiState,
    keep: [Vec<usize>; 3],
    rake_rate: f64,
    rake_cap: holdem::Chips,
) -> MultiConfig {
    let seed = board.iter().fold(23u64, |h, c| h.wrapping_mul(31).wrapping_add(c.index() as u64));
    MultiConfig {
        board: board.to_vec(),
        ranges,
        root,
        menu: menu.clone(),
        future_menu: Some(s.multiway_future_menu.clone()),
        big_blind: BB,
        chance: match root.street {
            Street::Flop => {
                ChanceMode::Sampled { turn: s.multiway_turn_samples, river: s.multiway_river_samples, seed }
            }
            Street::Turn => ChanceMode::Sampled { turn: 0, river: s.multiway_river_samples * 2, seed },
            _ => ChanceMode::Exact,
        },
        prune: s.prune.max(0.002),
        keep,
        max_memory_bytes: (s.max_memory_gb * 1e9) as u64,
        rake_rate,
        rake_cap: rake_cap as f64,
        exact_removal: s.multiway_exact_removal,
        storage: solver::StorageMode::Auto,
        dcfr: solver::multi::multiway_dcfr(),
        backend: s.backend.into(),
    }
}

/// Waits for the solve queue (`priority`: smaller goes first), builds the
/// heads-up tree (falling back to fast mode if the exact tree is too big)
/// and solves it. Returns the solver and its exploitability (% pot).
pub(crate) fn run_hu(
    mut cfg: SolveConfig,
    s: &SolverSettings,
    priority: u64,
    pool: Option<&rayon::ThreadPool>,
    label: &str,
    sink: &(dyn SolveSink + Sync),
) -> Result<(Solver, SolveSummary), String> {
    let cancel = sink.cancel_flag();
    sink.stage(format!("{label}: waiting for the solver"));
    let Some(_turn) = wait_solve_turn(priority, cancel, |ahead| {
        sink.stage(format!(
            "{label}: waiting for the solver ({ahead} solve{} ahead)",
            if ahead == 1 { "" } else { "s" }
        ))
    }) else {
        return Err(CANCELLED.into());
    };
    sink.stage(format!("{label}: building the game tree"));
    let mut solver = match Solver::new(cfg.clone()) {
        Ok(x) => x,
        Err(SolverError::TooLarge { needed, .. }) if cfg.chance == ChanceMode::Exact => {
            sink.note(format!(
                "{label}: the exact tree needs ~{:.1} GB (limit {:.1} GB), so fast sampled mode was used.",
                needed as f64 / 1e9,
                s.max_memory_gb
            ));
            cfg.chance =
                ChanceMode::Sampled { turn: s.flop_turn_samples, river: s.flop_river_samples, seed: 7 };
            Solver::new(cfg).map_err(|e| e.to_string())?
        }
        Err(e) => return Err(e.to_string()),
    };
    sink.stage(format!("{label}: solving ({} vs {} combos)", solver.hands().len(0), solver.hands().len(1)));
    // Exploitability checks cost about one iteration; check big trees less often.
    let big = solver.stats().total_nodes() > 200_000;
    let opts = SolveOptions {
        max_iterations: s.max_iterations,
        target_exploitability_pct: s.target_exploitability_pct,
        check_every: if big { 20 } else { 10 },
    };
    let mut run = || {
        solver.solve_cancellable(&opts, cancel, |p| {
            sink.progress(*p);
            !cancel.load(Ordering::Relaxed)
        })
    };
    let progress = match pool {
        Some(pool) => pool.install(run),
        None => run(),
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(CANCELLED.into());
    }
    if let (SolverBackend::Gpu, Some(e)) = (s.backend, solver.gpu_error()) {
        sink.note(format!("{label}: solved on the CPU (GPU not used: {e})."));
    }
    let gpu_fallback = match s.backend {
        SolverBackend::Cpu => None,
        SolverBackend::Auto | SolverBackend::Gpu => solver.gpu_error().map(str::to_string),
    };
    let summary = SolveSummary {
        exploitability_pct: progress.exploitability_pct,
        solver: SolverTag { by: solver.solved_on().into(), gpu_fallback },
    };
    Ok((solver, summary))
}

/// Like [`run_hu`] for a three-player spot (approximate).
pub(crate) fn run_multi(
    cfg: MultiConfig,
    s: &SolverSettings,
    priority: u64,
    pool: Option<&rayon::ThreadPool>,
    label: &str,
    sink: &(dyn SolveSink + Sync),
) -> Result<(MultiSolver, SolveSummary), String> {
    let cancel = sink.cancel_flag();
    sink.stage(format!("{label}: waiting for the solver"));
    let Some(_turn) = wait_solve_turn(priority, cancel, |ahead| {
        sink.stage(format!(
            "{label}: waiting for the solver ({ahead} solve{} ahead)",
            if ahead == 1 { "" } else { "s" }
        ))
    }) else {
        return Err(CANCELLED.into());
    };
    sink.stage(format!("{label}: building the game tree"));
    let mut solver = MultiSolver::new(cfg).map_err(|e| e.to_string())?;
    sink.stage(format!("{label}: solving (approximate)"));
    let opts = SolveOptions {
        max_iterations: s.max_iterations,
        target_exploitability_pct: s.multiway_target_pct,
        check_every: 20,
    };
    let mut run = || {
        solver.solve_cancellable(&opts, cancel, |p| {
            sink.progress(*p);
            !cancel.load(Ordering::Relaxed)
        })
    };
    let progress = match pool {
        Some(pool) => pool.install(run),
        None => run(),
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(CANCELLED.into());
    }
    if let (SolverBackend::Gpu, Some(e)) = (s.backend, solver.gpu_error()) {
        sink.note(format!("{label}: solved on the CPU (GPU not used: {e})."));
    }
    let gpu_fallback = match s.backend {
        SolverBackend::Cpu => None,
        SolverBackend::Auto | SolverBackend::Gpu => solver.gpu_error().map(str::to_string),
    };
    let summary = SolveSummary {
        exploitability_pct: progress.exploitability_pct,
        solver: SolverTag { by: solver.solved_on().into(), gpu_fallback },
    };
    Ok((solver, summary))
}

/// Seats still in the hand at the start of `street`.
pub fn in_hand_at(state: &HandState, street: Street) -> Vec<usize> {
    let mut folded = vec![false; state.num_players()];
    for r in &state.history {
        if r.street < street && r.action == Action::Fold {
            folded[r.player] = true;
        }
    }
    (0..state.num_players()).filter(|&p| !folded[p]).collect()
}

/// Seats sorted by postflop acting order (first to act first).
pub fn postflop_order_n(state: &HandState, seats: &[usize]) -> Vec<usize> {
    let n = state.num_players();
    let mut v = seats.to_vec();
    v.sort_by_key(|&s| (s + n - state.button - 1) % n);
    v
}

fn context_multi(st: &MultiState) -> ActionContext {
    let sit = st.situation(BB);
    ActionContext {
        to_call: sit.to_call,
        pot: st.pot,
        all_in_to: st.commit[st.to_act] + st.stack[st.to_act],
        facing_bet: st.current_bet > 0,
        preflop: false,
    }
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

/// For players whose range almost never reaches the current node, keep a
/// fraction of their earlier range so the solver has something to work with.
/// Debugging aid: with `POKER_TRAINER_DUMP=<dir>` set, every heads-up solve
/// writes its inputs (board, ranges, betting state, menu) to a text file, so
/// a spot can be inspected or re-solved with another solver.
fn dump_config(dir: &std::path::Path, hand: u64, label: &str, cfg: &SolveConfig) {
    let range = |w: &[f32]| -> String {
        (0..w.len())
            .filter(|&c| w[c] > 0.0)
            .map(|c| {
                let (a, b) = holdem::combo_cards(c);
                let (hi, lo) = if a.rank() >= b.rank() { (a, b) } else { (b, a) };
                format!("{hi}{lo}:{}", w[c])
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let r = &cfg.root;
    let board: String = cfg.board.iter().map(|c| c.to_string()).collect();
    let text = format!(
        "label={label}\nboard={board}\nstreet={:?}\npot={}\ncommit={},{}\nstack={},{}\nto_act={}\n\
         current_bet={}\nnum_raises={}\nrake={},{}\nmenu={:?}\nfuture_menu={:?}\noop={}\nip={}\n",
        r.street,
        r.pot,
        r.commit[0],
        r.commit[1],
        r.stack[0],
        r.stack[1],
        r.to_act,
        r.current_bet,
        r.num_raises,
        cfg.rake_rate,
        cfg.rake_cap,
        cfg.menu,
        cfg.future_menu,
        range(&cfg.ranges[0]),
        range(&cfg.ranges[1]),
    );
    let name: String = format!("hand{hand:04}-{label}")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(format!("{name}.txt")), text);
}

fn floor_ranges(ranges: &mut [Vec<f32>; 2], root: &[Vec<f32>; 2], frac: &[f64; 2], threshold: f64) {
    for x in 0..2 {
        if frac[x] < threshold {
            for (w, &r) in ranges[x].iter_mut().zip(&root[x]) {
                *w = w.max(0.05 * r);
            }
        }
    }
}

fn context(st: &BettingState) -> ActionContext {
    let sit = st.situation(BB);
    ActionContext {
        to_call: sit.to_call,
        pot: st.pot,
        all_in_to: st.commit[st.to_act] + st.stack[st.to_act],
        facing_bet: st.current_bet > 0,
        preflop: false,
    }
}

/// Per-class strategy of the acting player at a node (grid order; `None`
/// where the class isn't in range). A cell's weight is the summed reach of
/// its combos.
pub fn class_grid(info: &NodeInfo) -> Vec<Option<ClassCell>> {
    let n = info.num_hands();
    let na = info.actions.len();
    let mut sums: Vec<(f32, Vec<f32>)> = vec![(0.0, vec![0.0; na]); NUM_CLASSES];
    for h in 0..n {
        let w = info.reach[h];
        if w <= 0.0 {
            continue;
        }
        let g = COMBO_CLASS[info.combos[h] as usize] as usize;
        sums[g].0 += w;
        for a in 0..na {
            sums[g].1[a] += w * info.strategy[a * n + h];
        }
    }
    sums.into_iter()
        .map(|(w, s)| {
            if w > 0.0 {
                Some(ClassCell { weight: w, strategy: s.into_iter().map(|x| x / w).collect() })
            } else {
                None
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn grade(
    info: &NodeInfo,
    ctx: &ActionContext,
    pot: holdem::Chips,
    hero_combo: usize,
    chosen: Action,
    variant: Variant,
    street: Street,
    board: &[Card],
    history: String,
    solve: SolveSummary,
    notes: Vec<String>,
    spot: SpotInfo,
) -> Result<DecisionReview, String> {
    let i = info.hand_index(hero_combo).ok_or_else(|| "your hand isn't in the solved range".to_string())?;
    let strat = info.hand_strategy(i);
    let ev = info.hand_ev(i);
    let chosen_idx = info
        .actions
        .iter()
        .position(|&a| a == chosen)
        .ok_or_else(|| format!("{chosen:?} isn't in the solver's tree"))?;
    let best = (0..ev.len())
        .max_by(|&x, &y| ev[x].partial_cmp(&ev[y]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    let loss = (ev[best] - ev[chosen_idx]).max(0.0);
    debug_assert!((strat.iter().sum::<f32>() - 1.0).abs() < 1e-3, "solver mix doesn't sum to 1");
    debug_assert!(ev.iter().all(|v| v.is_finite()), "non-finite EVs");
    let pot_f = pot.max(1) as f32;
    let loss_pct = 100.0 * loss / pot_f;
    let labels = info.actions.iter().map(|&a| action_label(a, ctx)).collect();

    let grid = class_grid(info);

    Ok(DecisionReview {
        variant,
        street,
        board: board.to_vec(),
        history,
        pot,
        to_call: ctx.to_call,
        hero_combo,
        actions: info.actions.clone(),
        labels,
        chosen: chosen_idx,
        best,
        hero_strategy: strat,
        hero_ev: ev,
        ev_loss: loss,
        ev_loss_pct: loss_pct,
        verdict: Verdict::from_loss_pct(loss_pct),
        range_strategy: info.range_strategy(),
        grid,
        exploitability_pct: solve.exploitability_pct,
        notes,
        spot,
        solver: solve.solver,
    })
}
