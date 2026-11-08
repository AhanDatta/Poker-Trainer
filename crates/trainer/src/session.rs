//! A training session: deals hands, picks opponents, keeps history and stats.

use crate::config::{GameType, HeroPosition, TrainerConfig};
use crate::fmt::BB;
use crate::hand::{HandSetup, LiveHand};
use crate::preflop::ChartSet;
use crate::profiles::Profile;
use crate::review::{ReviewStatus, Variant, Verdict};
use crate::save::{self, LoadedSession, SessionMeta, SessionStore, SessionSummary};
use crate::scoring::{self, FrequencyReport, ScoreOptions, ScoredDecision};
use holdem::{Chips, TableConfig};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Session {
    pub config: TrainerConfig,
    charts: Arc<ChartSet>,
    rng: StdRng,
    next_id: u64,
    /// The opponent in each seat (`None` for the hero's seat). Players stay
    /// for the session; now and then one leaves and a new one sits down.
    lineup: Vec<Option<Profile>>,
    button: usize,
    /// Finished hands (most recent last).
    pub history: Vec<LiveHand>,
    /// `history[..settled]` are final (their reviews are over) and counted
    /// in `settled_stats` and `scored`, so stats don't rescan every hand.
    settled: usize,
    settled_stats: Stats,
    /// Decisions of the settled hands as the frequency score sees them.
    scored: Vec<ScoredDecision>,
    /// Pot weighting `scored` was built with.
    scored_weighted: bool,
    report: Option<FrequencyReport>,
    /// Saving: where sessions go, this session's directory (made when the
    /// first hand is saved) and how many hands of `history` are saved.
    save_root: Option<PathBuf>,
    store: Option<SessionStore>,
    saved: usize,
    created: u64,
    save_error: Option<String>,
}

/// Seed of the frequency score's simulations (fixed: the score of a
/// session doesn't change between looks).
const SCORE_SEED: u64 = 0x5c0e_5eed;

/// Aggregate statistics over finished hands.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub hands: usize,
    pub net_chips: Chips,
    pub decisions: usize,
    pub verdicts: BTreeMap<Verdict, usize>,
    /// Sum of EV losses in % of pot.
    pub total_loss_pct: f64,
    pub reviews_pending: usize,
}

impl Stats {
    /// Adds a finished hand (and its review so far).
    fn add_hand(&mut self, h: &LiveHand) {
        self.hands += 1;
        self.net_chips += h.hero_net().unwrap_or(0);
        if let Some(r) = h.review() {
            r.with_status(|st| self.add_review(st));
        }
    }

    fn add_review(&mut self, st: &ReviewStatus) {
        if !st.finished {
            self.reviews_pending += 1;
        }
        let has_gto = st.decisions.iter().any(|x| x.variant == Variant::Gto);
        for d in &st.decisions {
            // Count each decision once (the GTO variant when both exist).
            if d.variant == Variant::Actual && has_gto {
                continue;
            }
            self.decisions += 1;
            *self.verdicts.entry(d.verdict).or_default() += 1;
            self.total_loss_pct += d.ev_loss_pct as f64;
        }
    }

    fn merge(&mut self, o: &Stats) {
        self.hands += o.hands;
        self.net_chips += o.net_chips;
        self.decisions += o.decisions;
        for (v, n) in &o.verdicts {
            *self.verdicts.entry(*v).or_default() += n;
        }
        self.total_loss_pct += o.total_loss_pct;
        self.reviews_pending += o.reviews_pending;
    }

    pub fn avg_loss_pct(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.total_loss_pct / self.decisions as f64
        }
    }

    /// Share of decisions graded Best or Good.
    pub fn accuracy(&self) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        let good = self.verdicts.get(&Verdict::Best).copied().unwrap_or(0)
            + self.verdicts.get(&Verdict::Good).copied().unwrap_or(0);
        good as f64 / self.decisions as f64
    }
}

/// Charts for the configured table size (file or built-in).
/// Built-in charts for the table size, overlaid with the configured chart
/// file (whose spots take precedence) if there is one.
fn load_charts(config: &TrainerConfig) -> Result<ChartSet, String> {
    let builtin = ChartSet::builtin(config.players);
    Ok(match config.charts_for(config.players) {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            builtin.overlay(ChartSet::parse(&text, config.players).map_err(|e| format!("{path}: {e}"))?)
        }
        None => builtin,
    })
}

impl Session {
    pub fn new(config: TrainerConfig) -> Result<Session, String> {
        config.validate()?;
        let charts = load_charts(&config)?;
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        if let Some(t) = Some(config.solver.threads).filter(|&t| t > 0) {
            // Best effort: the global pool can only be configured once.
            let _ = rayon::ThreadPoolBuilder::new().num_threads(t).build_global();
        }
        let mut s = Session {
            config,
            charts: Arc::new(charts),
            rng,
            next_id: 1,
            lineup: Vec::new(),
            button: 0,
            history: Vec::new(),
            settled: 0,
            settled_stats: Stats::default(),
            scored: Vec::new(),
            scored_weighted: true,
            report: None,
            save_root: None,
            store: None,
            saved: 0,
            created: save::now_secs(),
            save_error: None,
        };
        s.scored_weighted = s.config.scoring.pot_weighted;
        s.button = s.rng.random_range(0..s.config.players);
        s.reset_lineup();
        Ok(s)
    }

    /// A style drawn for a new player: the chosen style, or a draw weighted
    /// by the game type (custom tables fill extra seats as a typical game).
    fn draw_style(&mut self) -> Profile {
        if !self.config.opponent.eq_ignore_ascii_case("random") {
            if let Some(p) = self.config.profile_by_name(&self.config.opponent) {
                return p;
            }
        }
        let game = match self.config.game {
            GameType::Custom => GameType::Typical,
            g => g,
        };
        let weights = game.weights();
        let total: f64 = weights.iter().map(|w| w.1).sum();
        let mut x = self.rng.random::<f64>() * total;
        for &(name, w) in weights {
            if x < w {
                return self.config.profile_by_name(name).expect("preset style");
            }
            x -= w;
        }
        self.config.profile_by_name(weights[0].0).expect("preset style")
    }

    /// Seats a whole new lineup.
    fn reset_lineup(&mut self) {
        let n = self.config.players;
        let hero = 0;
        let custom =
            self.config.opponent.eq_ignore_ascii_case("random") && self.config.game == GameType::Custom;
        let mut players: Vec<Profile> = Vec::new();
        if custom {
            for (name, &count) in &self.config.custom_table {
                if let Some(p) = self.config.profile_by_name(name) {
                    players.extend(std::iter::repeat_n(p, count));
                }
            }
            players.shuffle(&mut self.rng);
            players.truncate(n - 1);
        }
        while players.len() < n - 1 {
            let p = self.draw_style();
            players.push(p);
        }
        let mut it = players.into_iter();
        self.lineup = (0..n).map(|s| if s == hero { None } else { it.next() }).collect();
    }

    /// The opponent in each seat (`None` for the hero).
    pub fn lineup(&self) -> &[Option<Profile>] {
        &self.lineup
    }

    /// Changes the kind of game and seats a new lineup.
    pub fn set_game(&mut self, game: GameType) {
        self.config.opponent = "random".into();
        self.config.game = game;
        self.reset_lineup();
    }

    /// Sets how many players of `style` sit at a custom table.
    pub fn set_custom_count(&mut self, style: &str, count: usize) {
        if count == 0 {
            self.config.custom_table.remove(style);
        } else {
            self.config.custom_table.insert(style.to_string(), count);
        }
        if self.config.game == GameType::Custom {
            self.reset_lineup();
        }
    }

    /// Changes the table size (2..=9 players) and reloads the matching charts.
    /// Chosen hero positions the new table doesn't have are dropped (see
    /// [`HeroPosition::for_table`]).
    pub fn set_players(&mut self, players: usize) -> Result<(), String> {
        let mut cfg = self.config.clone();
        cfg.players = players;
        cfg.hero_position = cfg.hero_position.for_table(players);
        cfg.validate()?;
        self.charts = Arc::new(load_charts(&cfg)?);
        self.config = cfg;
        self.button = self.rng.random_range(0..players);
        self.reset_lineup();
        Ok(())
    }

    /// Changes the opponent style ("random" = by game type, or a style name
    /// for every opponent) and seats a new lineup.
    pub fn set_opponent(&mut self, name: &str) -> Result<(), String> {
        if name.eq_ignore_ascii_case("random") {
            self.config.opponent = "random".into();
        } else {
            let p = self.config.profile_by_name(name).ok_or_else(|| format!("unknown style {name:?}"))?;
            self.config.opponent = p.name.clone();
        }
        self.reset_lineup();
        Ok(())
    }

    pub fn charts(&self) -> &Arc<ChartSet> {
        &self.charts
    }

    /// Deals a new hand. Every hand starts with fresh `stack_bb` stacks.
    /// The hero sits in seat 0; the button position decides the hero's
    /// position (rotating, random, or only the chosen positions).
    pub fn new_hand(&mut self) -> LiveHand {
        // Moving on abandons the solves of earlier hands: they would
        // otherwise keep the CPU (and the solve queue) busy while this hand
        // is played.
        self.cancel_pending_reviews();
        let n = self.config.players;
        let hero = 0;
        let rotate = (self.button + 1) % n;
        let button = match &self.config.hero_position {
            HeroPosition::Random => self.rng.random_range(0..n),
            HeroPosition::Alternate => rotate,
            // The button moves on (as in rotation) until the hero is in a
            // chosen position; with one position it stays put.
            seats @ HeroPosition::Seats(_) => (1..=n)
                .map(|k| (self.button + k) % n)
                .find(|&b| seats.allows(n, (hero + n - b) % n))
                .unwrap_or(rotate),
        };
        self.button = button;
        // Table changes: now and then a player leaves and a new one (drawn
        // from the game type) sits down.
        let mut new_players = Vec::new();
        if self.lineup.len() != n || self.config.seat_change_pct >= 100.0 {
            self.reset_lineup();
        } else if self.config.seat_change_pct > 0.0 {
            for seat in (0..n).filter(|&s| s != hero) {
                if self.rng.random::<f64>() * 100.0 < self.config.seat_change_pct {
                    let custom = self.config.opponent.eq_ignore_ascii_case("random")
                        && self.config.game == GameType::Custom;
                    // A custom table keeps its exact mix.
                    if !custom {
                        self.lineup[seat] = Some(self.draw_style());
                    }
                    new_players.push(seat);
                }
            }
        }
        let profiles: Vec<Option<Profile>> = self.lineup.clone();
        let stack = (self.config.stack_bb * BB as f64).round() as Chips;
        let setup = HandSetup {
            id: self.next_id,
            table: TableConfig::new(n, BB / 2, BB).with_rake(
                self.config.rake_pct / 100.0,
                (self.config.rake_cap_bb * BB as f64).round() as Chips,
            ),
            button,
            stacks: vec![stack; n],
            hero,
            profiles,
            menu: self.config.menu.clone(),
            charts: Arc::clone(&self.charts),
            review: Some(self.config.solver.clone()),
        };
        self.next_id += 1;
        let seed: u64 = self.rng.random();
        let mut hand = LiveHand::new(setup, StdRng::seed_from_u64(seed));
        hand.set_roll_seed(crate::hand::mix64(seed ^ 0x0d1c_e5ee_d000_0001));
        for seat in new_players {
            hand.note(format!("Player {} left; a new player sat down.", seat + 1));
        }
        hand
    }

    /// Stores a finished (or abandoned) hand in the history.
    /// Cancels every review of a finished hand that is still running.
    pub fn cancel_pending_reviews(&self) {
        for h in &self.history {
            if let Some(r) = h.review() {
                if !r.is_finished() {
                    r.cancel();
                }
            }
        }
    }

    pub fn finish_hand(&mut self, hand: LiveHand) {
        if hand.is_finished() {
            self.history.push(hand);
        }
    }

    /// Totals over every finished hand.
    pub fn stats(&self) -> Stats {
        let mut s = self.settled_stats.clone();
        for h in &self.history[self.settled..] {
            s.add_hand(h);
        }
        s
    }

    /// The frequency score of the settled hands' heads-up decisions.
    pub fn report(&self) -> Option<&FrequencyReport> {
        self.report.as_ref()
    }

    /// Counts hands whose reviews are over (in order), saves them, and
    /// updates the frequency score. Cheap when nothing changed; call it
    /// regularly (e.g. every UI tick).
    pub fn refresh(&mut self) {
        self.settle(false);
    }

    /// Settles hands from `self.settled` on; `force` also settles hands
    /// whose review is still running, as they are now.
    fn settle(&mut self, force: bool) {
        let weighted = self.config.scoring.pot_weighted;
        if weighted != self.scored_weighted {
            self.scored_weighted = weighted;
            self.scored.clear();
            for h in &self.history[..self.settled] {
                if let Some(r) = h.review() {
                    r.with_status(|st| self.scored.extend(scoring::hand_decisions(&st.decisions, weighted)));
                }
            }
            self.update_report();
        }
        let start = self.settled;
        while self.settled < self.history.len() {
            let h = &self.history[self.settled];
            if !force && h.review().is_some_and(|r| !r.is_finished()) {
                break;
            }
            let mut st = Stats::default();
            st.add_hand(h);
            st.reviews_pending = 0;
            self.settled_stats.merge(&st);
            if let Some(r) = h.review() {
                r.with_status(|st| self.scored.extend(scoring::hand_decisions(&st.decisions, weighted)));
            }
            self.settled += 1;
        }
        if self.settled > start {
            self.update_report();
        }
        if self.saved < self.settled {
            self.save_settled();
        }
    }

    fn update_report(&mut self) {
        let opts = ScoreOptions { simulations: self.config.scoring.simulations, seed: SCORE_SEED };
        self.report = FrequencyReport::compute(&self.scored, opts);
    }

    /// Saves finished hands to the session's directory (from now on).
    pub fn start_saving(&mut self, root: PathBuf) {
        self.save_root = Some(root);
        self.settle(false);
    }

    /// Directory name of the saved session (once a hand was saved).
    pub fn save_id(&self) -> Option<&str> {
        self.store.as_ref().map(|s| s.id.as_str())
    }

    pub fn is_saving(&self) -> bool {
        self.save_root.is_some()
    }

    /// The last saving error (once).
    pub fn take_save_error(&mut self) -> Option<String> {
        self.save_error.take()
    }

    fn save_settled(&mut self) {
        let Some(root) = self.save_root.clone() else { return };
        if self.store.is_none() {
            match SessionStore::create(&root, self.created) {
                Ok(st) => self.store = Some(st),
                Err(e) => {
                    self.save_error = Some(format!("Couldn't save the session in {}: {e}", root.display()));
                    self.save_root = None;
                    return;
                }
            }
        }
        let hands: Vec<_> = self.history[self.saved..self.settled].iter().map(|h| h.to_saved()).collect();
        let store = self.store.as_ref().unwrap();
        let res = store.append_hands(&hands).and_then(|()| store.write_meta(&self.meta()));
        match res {
            Ok(()) => self.saved = self.settled,
            Err(e) => {
                self.save_error = Some(format!("Couldn't save to {}: {e}", store.dir().display()));
                self.save_root = None;
            }
        }
    }

    /// The session's saved header.
    pub fn meta(&self) -> SessionMeta {
        let st = &self.settled_stats;
        SessionMeta {
            version: save::FORMAT_VERSION,
            id: self.save_id().unwrap_or_default().to_string(),
            created: self.created,
            updated: save::now_secs(),
            config: self.config.clone(),
            lineup: self.lineup.clone(),
            button: self.button,
            next_id: self.next_id,
            summary: SessionSummary {
                hands: st.hands,
                net_chips: st.net_chips,
                decisions: st.decisions,
                score: self.report.as_ref().map(|r| r.calibration.score()),
            },
        }
    }

    /// Before quitting or switching sessions: stops running reviews (waiting
    /// up to `wait` for them to wind down), then saves every finished hand
    /// and the current settings.
    pub fn flush(&mut self, wait: Duration) {
        self.cancel_pending_reviews();
        let t = Instant::now();
        while t.elapsed() < wait && self.history.iter().any(|h| h.review().is_some_and(|r| !r.is_finished()))
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        self.settle(true);
        if let (Some(store), Some(_)) = (&self.store, &self.save_root) {
            if let Err(e) = store.write_meta(&self.meta()) {
                self.save_error = Some(format!("Couldn't save to {}: {e}", store.dir().display()));
            }
        }
    }

    /// Continues a saved session. Settings come from the save, except where
    /// sessions are kept and the seed, which come from `base`.
    pub fn resume(loaded: LoadedSession, root: PathBuf, base: &TrainerConfig) -> Result<Session, String> {
        let meta = loaded.meta;
        let mut config = meta.config.clone();
        config.save_sessions = base.save_sessions;
        config.sessions_dir = base.sessions_dir.clone();
        // A seeded session continues with new deals, not the first ones again.
        config.seed = base.seed.map(|s| crate::hand::mix64(s ^ meta.next_id));
        let mut s = Session::new(config)?;
        let n = s.config.players;
        if meta.lineup.len() == n && meta.lineup.iter().filter(|p| p.is_none()).count() == 1 {
            s.lineup = meta.lineup.clone();
        }
        s.button = meta.button % n;
        s.next_id = meta.next_id.max(1);
        s.created = meta.created;
        let (menu, charts) = (s.config.menu.clone(), Arc::clone(&s.charts));
        s.history = loaded
            .hands
            .into_iter()
            .map(|h| LiveHand::from_saved(h, menu.clone(), Arc::clone(&charts)))
            .collect();
        s.saved = s.history.len();
        s.store = Some(SessionStore::open(&root, &meta.id).map_err(|e| e.to_string())?);
        s.save_root = Some(root);
        s.settle(false);
        Ok(s)
    }
}
