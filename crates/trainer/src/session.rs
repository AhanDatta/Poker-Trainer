//! A training session: deals hands, picks opponents, keeps history and stats.

use crate::config::{HeroPosition, TrainerConfig};
use crate::fmt::BB;
use crate::hand::{HandSetup, LiveHand};
use crate::preflop::ChartSet;
use crate::profiles::Profile;
use crate::review::{ReviewStatus, Verdict};
use holdem::{Chips, TableConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct Session {
    pub config: TrainerConfig,
    charts: Arc<ChartSet>,
    rng: StdRng,
    next_id: u64,
    /// Fixed style for every opponent (when one was chosen).
    session_profile: Option<Profile>,
    /// Per-seat styles kept for the whole session (random, not per hand).
    seat_profiles: Vec<Profile>,
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
fn load_charts(config: &TrainerConfig) -> Result<ChartSet, String> {
    Ok(match config.charts_for(config.players) {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            ChartSet::parse(&text, config.players).map_err(|e| format!("{path}: {e}"))?
        }
        None => ChartSet::builtin(config.players),
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
            session_profile: None,
            seat_profiles: Vec::new(),
            button: 0,
            history: Vec::new(),
        };
        if !s.config.opponent.eq_ignore_ascii_case("random") {
            s.session_profile = s.config.profile_by_name(&s.config.opponent);
        }
        s.button = s.rng.random_range(0..s.config.players);
        s.reset_seat_profiles();
        Ok(s)
    }

    fn random_profile(&mut self) -> Profile {
        let all = self.config.profiles();
        all[self.rng.random_range(0..all.len())].clone()
    }

    /// New random styles for every seat (used when styles are random per
    /// session rather than per hand).
    fn reset_seat_profiles(&mut self) {
        let n = self.config.players;
        self.seat_profiles = (0..n).map(|_| self.random_profile()).collect();
    }

    /// Changes the table size (2..=9 players) and reloads the matching charts.
    pub fn set_players(&mut self, players: usize) -> Result<(), String> {
        let mut cfg = self.config.clone();
        cfg.players = players;
        cfg.validate()?;
        self.charts = Arc::new(load_charts(&cfg)?);
        self.config = cfg;
        self.button = self.rng.random_range(0..players);
        self.reset_seat_profiles();
        Ok(())
    }

    /// Changes the opponent style ("random" or a style name).
    pub fn set_opponent(&mut self, name: &str) -> Result<(), String> {
        if name.eq_ignore_ascii_case("random") {
            self.config.opponent = "random".into();
            self.session_profile = None;
            self.reset_seat_profiles();
            return Ok(());
        }
        let p = self.config.profile_by_name(name).ok_or_else(|| format!("unknown style {name:?}"))?;
        self.config.opponent = p.name.clone();
        self.session_profile = Some(p);
        Ok(())
    }

    pub fn charts(&self) -> &Arc<ChartSet> {
        &self.charts
    }

    /// Deals a new hand. Every hand starts with fresh `stack_bb` stacks.
    /// The hero sits in seat 0; the button position decides the hero's
    /// position (rotating, random, or pinned to a blind).
    pub fn new_hand(&mut self) -> LiveHand {
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
        let per_hand_random = self.session_profile.is_none() && self.config.random_each_hand;
        let profiles: Vec<Option<Profile>> = (0..n)
            .map(|s| {
                if s == hero {
                    None
                } else if let Some(p) = &self.session_profile {
                    Some(p.clone())
                } else if per_hand_random {
                    Some(self.random_profile())
                } else {
                    Some(self.seat_profiles[s].clone())
                }
            })
            .collect();
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
        LiveHand::new(setup, StdRng::seed_from_u64(seed))
    }

    /// Stores a finished (or abandoned) hand in the history.
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
