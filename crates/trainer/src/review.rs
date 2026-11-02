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
use crate::config::{FlopMode, SolverSettings, VillainRange};
use crate::fmt::{action_label, ActionContext, BB};
use crate::preflop::{chart_policy, describe, line_range, ChartSet, PreDecision};
use crate::profiles::Profile;
use holdem::combo::{combo_index, COMBO_CLASS, NUM_CLASSES};
use holdem::game::Phase;
use holdem::{Action, BetMenu, Card, HandState, Street};
use solver::{
    BettingState, ChanceMode, NodeInfo, Progress, SolveConfig, SolveOptions, Solver, SolverError, Step,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// Serializes solves across hands so reviews don't fight over the CPU.
static SOLVE_LOCK: Mutex<()> = Mutex::new(());

const CANCELLED: &str = "cancelled";

/// Which villain range a solve used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Debug)]
pub struct ClassCell {
    /// Total reach weight of the class's combos at this node.
    pub weight: f32,
    /// Weighted average strategy over the class's combos.
    pub strategy: Vec<f32>,
}

/// The solver's verdict on one hero decision.
#[derive(Clone, Debug)]
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
}

#[derive(Clone, Debug, Default)]
pub struct ReviewStatus {
    pub stage: String,
    pub progress: Option<Progress>,
    pub decisions: Vec<DecisionReview>,
    pub notes: Vec<String>,
    pub finished: bool,
    pub error: Option<String>,
}

/// Static inputs of a review.
#[derive(Clone)]
pub struct ReviewSetup {
    pub hero: usize,
    pub villain: usize,
    pub villain_profile: Profile,
    pub charts: Arc<ChartSet>,
    pub menu: BetMenu,
    pub settings: SolverSettings,
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
                let result = Worker::new(&worker, &setup).run();
                worker.with_status(|st| {
                    st.finished = true;
                    st.progress = None;
                    match result {
                        Ok(()) => st.stage = "Review complete".into(),
                        Err(e) if e == CANCELLED => st.stage = "Review cancelled".into(),
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
        if snap.state.players_in_hand().len() != 2 {
            self.shared.note("Multiway postflop pots can't be solved yet.".into());
            return Ok(());
        }
        let st0 = &snap.state;
        let (hero, villain) = (setup.hero, setup.villain);
        // Heads-up: the big blind is out of position (solver player 0).
        let seats = [st0.bb_seat(), st0.sb_seat()];
        let hero_p = if seats[0] == hero { 0 } else { 1 };
        let hero_combo = combo_index(st0.hole[hero][0], st0.hole[hero][1]);
        let variants = self.variants();

        let mut ranges: Vec<[Vec<f32>; 2]> = Vec::new();
        for &v in &variants {
            let mut pr: [Vec<f32>; 2] = Default::default();
            for x in 0..2 {
                let seat = seats[x];
                let (r, uncovered) = if seat == villain && v == Variant::Actual {
                    line_range(&snap.preflop, seat, |d| {
                        profile_policy(&setup.charts, &setup.villain_profile, d)
                    })
                } else {
                    line_range(&snap.preflop, seat, |d| chart_policy(&setup.charts, d))
                };
                for idx in uncovered {
                    let d = &snap.preflop[idx];
                    let who = if seat == hero { "You" } else { "Villain" };
                    let msg = format!(
                        "{who} chose to {} preflop ({}), which the charts never do there; that range was approximated by the hands the charts continue with.",
                        describe(d.action, &d.avail),
                        snap.state.position_name(seat)
                    );
                    let dup = self.shared.status.lock().unwrap().notes.contains(&msg);
                    if !dup {
                        self.shared.note(msg);
                    }
                }
                pr[x] = r.into_weights();
            }
            ranges.push(pr);
        }

        for street in [Street::Flop, Street::Turn, Street::River] {
            let need = street.board_len();
            let snap =
                self.shared.wait_for(|s| s.state.board.len() >= need || s.state.phase == Phase::Finished)?;
            if snap.state.board.len() < need {
                break;
            }
            let board: Vec<Card> = snap.state.board[..need].to_vec();
            let (pot, by_seat) = street_start(&snap.state, street);
            let stacks = [by_seat[seats[0]], by_seat[seats[1]]];
            if stacks[0] == 0 || stacks[1] == 0 {
                break;
            }
            let root = BettingState::street_start(street, pot, stacks, BB);

            let mut solved: Vec<Option<(Solver, f64)>> = Vec::new();
            for (vi, &v) in variants.iter().enumerate() {
                let label = format!("{street} ({})", v.label());
                let cfg = self.make_config(street, &board, &ranges[vi], root, hero_combo, hero_p);
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
            let snap =
                self.shared.wait_for(|s| s.state.street > street || s.state.phase == Phase::Finished)?;
            let state = &snap.state;
            let actions: Vec<(usize, Action)> =
                state.history.iter().filter(|r| r.street == street).map(|r| (r.player, r.action)).collect();
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
                let mut owned: Option<(Solver, f64)> = None;
                let mut offset = 0;
                for k in 0..actions.len() {
                    let (seat, a) = actions[k];
                    if seat != hero || k >= states.len() {
                        continue;
                    }
                    let (cur, cur_expl) = match &owned {
                        Some((s, e)) => (s, *e),
                        None => (base, *base_expl),
                    };
                    let path: Vec<Step> = actions[offset..k].iter().map(|&(_, a)| Step::Action(a)).collect();
                    let mut info = cur.query(&path).map_err(|e| e.to_string())?;
                    let mut notes = Vec::new();
                    let mut expl = cur_expl;
                    let off =
                        info.reach_fraction.iter().any(|&f| f < setup.settings.off_equilibrium_threshold);
                    if off && k > offset {
                        let who =
                            if info.reach_fraction[1 - hero_p] < setup.settings.off_equilibrium_threshold {
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
                        let cfg = self.make_config(street, &board, &r, states[k], hero_combo, hero_p);
                        match self.solve(cfg, &label) {
                            Ok((s, e)) => {
                                owned = Some((s, e));
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
                            let ctx = context(&states[j]);
                            format!("{} {}", pos(actions[j].0), action_label(actions[j].1, &ctx))
                        })
                        .collect::<Vec<_>>()
                        .join(" · ");
                    if street == Street::Flop && setup.settings.flop_mode == FlopMode::Fast {
                        notes.push(
                            "Flop solved with sampled turn/river cards (fast mode): approximate.".into(),
                        );
                    }
                    let review =
                        grade(&info, &states[k], hero_combo, a, v, street, &board, history, expl, notes)?;
                    self.shared.with_status(|st| st.decisions.push(review));
                }

                if street != Street::River && state.board.len() > need {
                    let (cur, _) = match &owned {
                        Some((s, e)) => (s, *e),
                        None => (base, *base_expl),
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
                        ranges[vi] = next;
                    }
                }
            }
            if state.phase == Phase::Finished && state.board.len() == need {
                break;
            }
        }
        Ok(())
    }

    fn make_config(
        &self,
        street: Street,
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
        let w = &mut ranges[hero_p][hero_combo];
        *w = w.max(floor);
        let flop = street == Street::Flop;
        let seed = board.iter().fold(17u64, |h, c| h.wrapping_mul(31).wrapping_add(c.index() as u64));
        let chance = if flop && s.flop_mode == FlopMode::Fast {
            ChanceMode::Sampled { turn: s.flop_turn_samples, river: s.flop_river_samples, seed }
        } else {
            ChanceMode::Exact
        };
        let mut keep = [Vec::new(), Vec::new()];
        keep[hero_p].push(hero_combo);
        SolveConfig {
            board: board.to_vec(),
            ranges,
            root,
            menu: self.setup.menu.postflop.clone(),
            future_menu: if flop { Some(s.flop_future_menu.clone()) } else { None },
            big_blind: BB,
            chance,
            prune: s.prune,
            keep,
            max_memory_bytes: (s.max_memory_gb * 1e9) as u64,
        }
    }

    fn solve(&self, mut cfg: SolveConfig, label: &str) -> Result<(Solver, f64), String> {
        let shared = self.shared;
        shared.set_stage(format!("{label}: waiting for the solver"));
        let _guard = SOLVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if shared.cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        shared.set_stage(format!("{label}: building the game tree"));
        let s = &self.setup.settings;
        let mut solver = match Solver::new(cfg.clone()) {
            Ok(x) => x,
            Err(SolverError::TooLarge { needed, .. }) if cfg.chance == ChanceMode::Exact => {
                shared.note(format!(
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
        shared.set_stage(format!(
            "{label}: solving ({} vs {} combos)",
            solver.hands().len(0),
            solver.hands().len(1)
        ));
        let opts = SolveOptions {
            max_iterations: s.max_iterations,
            target_exploitability_pct: s.target_exploitability_pct,
            check_every: 10,
        };
        let mut run = || {
            solver.solve(&opts, |p| {
                shared.with_status(|st| st.progress = Some(*p));
                !shared.cancel.load(Ordering::Relaxed)
            })
        };
        let progress = match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        };
        if shared.cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        Ok((solver, progress.exploitability_pct))
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

#[allow(clippy::too_many_arguments)]
fn grade(
    info: &NodeInfo,
    state: &BettingState,
    hero_combo: usize,
    chosen: Action,
    variant: Variant,
    street: Street,
    board: &[Card],
    history: String,
    exploitability_pct: f64,
    notes: Vec<String>,
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
    let pot = state.pot.max(1) as f32;
    let loss_pct = 100.0 * loss / pot;
    let ctx = context(state);
    let labels = info.actions.iter().map(|&a| action_label(a, &ctx)).collect();

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
    let grid = sums
        .into_iter()
        .map(|(w, s)| {
            if w > 0.0 {
                Some(ClassCell { weight: w, strategy: s.into_iter().map(|x| x / w).collect() })
            } else {
                None
            }
        })
        .collect();

    Ok(DecisionReview {
        variant,
        street,
        board: board.to_vec(),
        history,
        pot: state.pot,
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
        exploitability_pct,
        notes,
    })
}
