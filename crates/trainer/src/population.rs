//! All-AI table simulations and HUD-style statistics per style.
//!
//! Used to check that the AI styles play like the player types they are
//! named after (VPIP, PFR, 3-bet, c-bet, aggression, showdown frequency,
//! win rate), and to tune them.

use crate::hand::{HandSetup, LiveHand, StepResult};
use crate::preflop::ChartSet;
use crate::profiles::Profile;
use holdem::{Action, BetMenu, Chips, Street, TableConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Arc;

const BB: Chips = 100;

/// Counters for one style (numerator, opportunities).
#[derive(Clone, Debug, Default)]
pub struct StyleStats {
    pub hands: u64,
    pub vpip: u64,
    pub pfr: u64,
    pub three_bet: (u64, u64),
    pub fold_to_three_bet: (u64, u64),
    pub cbet: (u64, u64),
    pub fold_to_cbet: (u64, u64),
    /// Postflop bets + raises, and calls.
    pub aggressive: u64,
    pub calls: u64,
    pub saw_flop: u64,
    /// Flops seen with 3+ players.
    pub multiway_flops: u64,
    pub showdowns: u64,
    pub won_at_showdown: u64,
    /// Net chips won.
    pub net: i64,
}

fn pct(n: u64, d: u64) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

impl StyleStats {
    pub fn vpip_pct(&self) -> f64 {
        pct(self.vpip, self.hands)
    }
    pub fn pfr_pct(&self) -> f64 {
        pct(self.pfr, self.hands)
    }
    pub fn three_bet_pct(&self) -> f64 {
        pct(self.three_bet.0, self.three_bet.1)
    }
    pub fn fold_to_three_bet_pct(&self) -> f64 {
        pct(self.fold_to_three_bet.0, self.fold_to_three_bet.1)
    }
    pub fn cbet_pct(&self) -> f64 {
        pct(self.cbet.0, self.cbet.1)
    }
    pub fn fold_to_cbet_pct(&self) -> f64 {
        pct(self.fold_to_cbet.0, self.fold_to_cbet.1)
    }
    /// Postflop aggression factor: (bets + raises) / calls.
    pub fn af(&self) -> f64 {
        if self.calls == 0 {
            self.aggressive as f64
        } else {
            self.aggressive as f64 / self.calls as f64
        }
    }
    /// Went to showdown, as a share of flops seen.
    pub fn wtsd_pct(&self) -> f64 {
        pct(self.showdowns, self.saw_flop)
    }
    /// Won money at showdown, as a share of showdowns.
    pub fn wsd_pct(&self) -> f64 {
        pct(self.won_at_showdown, self.showdowns)
    }
    pub fn bb_per_100(&self) -> f64 {
        if self.hands == 0 {
            0.0
        } else {
            self.net as f64 / BB as f64 / self.hands as f64 * 100.0
        }
    }

    pub fn row(&self, name: &str) -> String {
        format!(
            "{name:<16} {:>7} {:>5.1} {:>5.1} {:>5.1} {:>6.1} {:>5.1} {:>6.1} {:>5.2} {:>5.1} {:>5.1} {:>8.1}",
            self.hands,
            self.vpip_pct(),
            self.pfr_pct(),
            self.three_bet_pct(),
            self.fold_to_three_bet_pct(),
            self.cbet_pct(),
            self.fold_to_cbet_pct(),
            self.af(),
            self.wtsd_pct(),
            self.wsd_pct(),
            self.bb_per_100()
        )
    }

    pub fn header() -> String {
        format!(
            "{:<16} {:>7} {:>5} {:>5} {:>5} {:>6} {:>5} {:>6} {:>5} {:>5} {:>5} {:>8}",
            "style", "hands", "VPIP", "PFR", "3bet", "F3bet", "Cbet", "FCbet", "AF", "WTSD", "W$SD", "bb/100"
        )
    }
}

/// Plays `hands` hands with the given styles seated around the table (the
/// seats rotate relative to the button every hand) and returns the stats
/// per style name.
pub fn simulate(
    seats: &[Profile],
    charts: &ChartSet,
    hands: usize,
    seed: u64,
) -> BTreeMap<String, StyleStats> {
    simulate_lineups(|_, _| seats.to_vec(), charts, hands, seed)
}

/// Like [`simulate`], with the lineup for hand `i` given by `lineup(i, rng)`.
pub fn simulate_lineups(
    mut lineup: impl FnMut(usize, &mut StdRng) -> Vec<Profile>,
    charts: &ChartSet,
    hands: usize,
    seed: u64,
) -> BTreeMap<String, StyleStats> {
    let charts = Arc::new(charts.clone());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out: BTreeMap<String, StyleStats> = BTreeMap::new();
    for i in 0..hands {
        let seats = lineup(i, &mut rng);
        let seats = &seats[..];
        let n = seats.len();
        let setup = HandSetup {
            id: i as u64 + 1,
            table: TableConfig::new(n, BB / 2, BB),
            button: i % n,
            stacks: vec![100 * BB; n],
            hero: 0,
            profiles: seats.iter().cloned().map(Some).collect(),
            menu: BetMenu::default(),
            charts: Arc::clone(&charts),
            review: None,
        };
        let mut hand = LiveHand::new(setup, StdRng::seed_from_u64(rng.random()));
        loop {
            match hand.step() {
                StepResult::Finished => break,
                StepResult::HeroToAct => {
                    hand.autoplay_hero().expect("hero AI acts");
                }
                _ => {}
            }
        }
        record(&hand, seats, &mut out);
    }
    out
}

fn record(hand: &LiveHand, seats: &[Profile], out: &mut BTreeMap<String, StyleStats>) {
    let st = &hand.state;
    let n = seats.len();
    let settle = st.settle();
    let hist = &st.history;
    // Preflop.
    let pre: Vec<_> = hist.iter().filter(|r| r.street == Street::Preflop).collect();
    let mut raises = 0;
    let mut opener: Option<usize> = None;
    let mut last_raiser: Option<usize> = None;
    let mut vpip = vec![false; n];
    let mut pfr = vec![false; n];
    let mut three_opp = vec![false; n];
    let mut three = vec![false; n];
    let mut f3_opp = vec![false; n];
    let mut f3 = vec![false; n];
    for r in &pre {
        let p = r.player;
        if raises == 1 && Some(p) != opener {
            three_opp[p] = true;
            if matches!(r.action, Action::Raise(_)) {
                three[p] = true;
            }
        }
        if raises == 2 && Some(p) == opener && !f3_opp[p] {
            f3_opp[p] = true;
            f3[p] = r.action == Action::Fold;
        }
        match r.action {
            Action::Call => vpip[p] = true,
            Action::Raise(_) => {
                vpip[p] = true;
                pfr[p] = true;
                raises += 1;
                if opener.is_none() {
                    opener = Some(p);
                }
                last_raiser = Some(p);
            }
            _ => {}
        }
    }
    // Flop.
    let saw_flop: Vec<bool> = (0..n)
        .map(|p| st.board.len() >= 3 && !pre.iter().any(|r| r.player == p && r.action == Action::Fold))
        .collect();
    let flop: Vec<_> = hist.iter().filter(|r| r.street == Street::Flop).collect();
    let mut cbet_opp = vec![false; n];
    let mut cbet = vec![false; n];
    let mut fcb_opp = vec![false; n];
    let mut fcb = vec![false; n];
    let mut bet_made = false;
    let mut cbet_made = false;
    let mut faced = vec![false; n];
    for r in &flop {
        let p = r.player;
        if cbet_made && !faced[p] && Some(p) != last_raiser {
            faced[p] = true;
            fcb_opp[p] = true;
            fcb[p] = r.action == Action::Fold;
        }
        if !bet_made && Some(p) == last_raiser && !cbet_opp[p] {
            cbet_opp[p] = true;
            if matches!(r.action, Action::Raise(_)) {
                cbet[p] = true;
                cbet_made = true;
            }
        }
        if matches!(r.action, Action::Raise(_)) {
            bet_made = true;
        }
    }
    for p in 0..n {
        let s = out.entry(seats[p].name.clone()).or_default();
        s.hands += 1;
        s.vpip += vpip[p] as u64;
        s.pfr += pfr[p] as u64;
        s.three_bet.0 += three[p] as u64;
        s.three_bet.1 += three_opp[p] as u64;
        s.fold_to_three_bet.0 += f3[p] as u64;
        s.fold_to_three_bet.1 += f3_opp[p] as u64;
        s.cbet.0 += cbet[p] as u64;
        s.cbet.1 += cbet_opp[p] as u64;
        s.fold_to_cbet.0 += fcb[p] as u64;
        s.fold_to_cbet.1 += fcb_opp[p] as u64;
        for r in hist.iter().filter(|r| r.player == p && r.street != Street::Preflop) {
            match r.action {
                Action::Raise(_) => s.aggressive += 1,
                Action::Call => s.calls += 1,
                _ => {}
            }
        }
        if saw_flop[p] {
            s.saw_flop += 1;
            if saw_flop.iter().filter(|&&x| x).count() >= 3 {
                s.multiway_flops += 1;
            }
            if settle.showdown && !st.folded[p] {
                s.showdowns += 1;
                if settle.net[p] > 0 {
                    s.won_at_showdown += 1;
                }
            }
        }
        s.net += settle.net[p];
    }
}
