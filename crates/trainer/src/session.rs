//! A training session: deals hands, picks opponents, keeps history and stats.

use crate::config::{GameType, HeroPosition, TrainerConfig};
use crate::fmt::BB;
use crate::hand::{HandSetup, LiveHand};
use crate::preflop::ChartSet;
use crate::profiles::Profile;
use crate::review::{ReviewStatus, Verdict};
use holdem::{Chips, TableConfig};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Arc;

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
}

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
        };
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
    pub fn set_players(&mut self, players: usize) -> Result<(), String> {
        let mut cfg = self.config.clone();
        cfg.players = players;
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
    /// position (rotating, random, or pinned to a blind).
    pub fn new_hand(&mut self) -> LiveHand {
        // Moving on abandons the solves of earlier hands: they would
        // otherwise keep the CPU (and the solve queue) busy while this hand
        // is played.
        self.cancel_pending_reviews();
        let n = self.config.players;
        let hero = 0;
        let button = match self.config.hero_position {
            // Heads-up the button is the small blind.
            HeroPosition::Sb => {
                if n == 2 {
                    hero
                } else {
                    (hero + n - 1) % n
                }
            }
            HeroPosition::Bb => {
                if n == 2 {
                    (hero + 1) % n
                } else {
                    (hero + n - 2) % n
                }
            }
            HeroPosition::Random => self.rng.random_range(0..n),
            HeroPosition::Alternate => (self.button + 1) % n,
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

    pub fn stats(&self) -> Stats {
        let mut s = Stats::default();
        for h in &self.history {
            s.hands += 1;
            s.net_chips += h.hero_net().unwrap_or(0);
            if let Some(r) = h.review() {
                let st: ReviewStatus = r.status();
                if !st.finished {
                    s.reviews_pending += 1;
                }
                for d in &st.decisions {
                    // Count each decision once (the GTO variant when both exist).
                    if d.variant == crate::review::Variant::Actual
                        && st.decisions.iter().any(|x| x.variant == crate::review::Variant::Gto)
                    {
                        continue;
                    }
                    s.decisions += 1;
                    *s.verdicts.entry(d.verdict).or_default() += 1;
                    s.total_loss_pct += d.ev_loss_pct as f64;
                }
            }
        }
        s
    }
}
