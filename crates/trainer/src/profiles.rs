//! Opponent styles.
//!
//! A [`Profile`] describes how an AI deviates from the baseline:
//! * preflop: `looseness` widens/narrows continuing ranges along the
//!   preflop strength order, `aggression` converts calls into raises (or
//!   raises into calls);
//! * postflop: thresholds and frequencies used by the heuristic decision
//!   rules in [`crate::ai`].

use crate::preflop::{Availability, Freqs, PreAction};
use crate::strength;
use holdem::combo::{HandClass, NUM_CLASSES};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    pub name: String,
    pub description: String,
    /// -1 (very tight) .. 1 (very loose) preflop.
    pub looseness: f32,
    /// -1 (passive: flats instead of raising) .. 1 (raises instead of calling).
    pub aggression: f32,
    /// Probability of betting/raising a hand that clears the value threshold.
    pub bet_freq: f32,
    /// Propensity to bluff with weak hands (scaled by draws/fold equity).
    pub bluff_freq: f32,
    /// 0 (folds to pressure) .. 1 (calling station).
    pub call_down: f32,
    /// Equity (vs. its read of your range) needed to bet for value.
    pub value_threshold: f32,
    /// Equity needed to raise for value.
    pub raise_threshold: f32,
    /// -1 (small sizes) .. 1 (big sizes / overbets).
    pub size_bias: f32,
    /// Probability of checking/calling a strong hand to trap.
    pub slowplay: f32,
}

impl Default for Profile {
    fn default() -> Self {
        Profile::reg()
    }
}

impl Profile {
    pub fn reg() -> Profile {
        Profile {
            name: "Reg".into(),
            description: "Solid, balanced regular close to the charts".into(),
            looseness: 0.0,
            aggression: 0.0,
            bet_freq: 0.75,
            bluff_freq: 0.3,
            call_down: 0.45,
            value_threshold: 0.62,
            raise_threshold: 0.82,
            size_bias: 0.0,
            slowplay: 0.15,
        }
    }

    pub fn nit() -> Profile {
        Profile {
            name: "Nit".into(),
            description: "Tight and passive: plays few hands, rarely bluffs, folds to pressure".into(),
            looseness: -0.55,
            aggression: -0.4,
            bet_freq: 0.6,
            bluff_freq: 0.07,
            call_down: 0.2,
            value_threshold: 0.7,
            raise_threshold: 0.9,
            size_bias: -0.3,
            slowplay: 0.25,
        }
    }

    pub fn tag() -> Profile {
        Profile {
            name: "TAG".into(),
            description: "Tight-aggressive: selective preflop, bets and raises its good hands".into(),
            looseness: -0.25,
            aggression: 0.4,
            bet_freq: 0.85,
            bluff_freq: 0.3,
            call_down: 0.35,
            value_threshold: 0.6,
            raise_threshold: 0.8,
            size_bias: 0.2,
            slowplay: 0.1,
        }
    }

    pub fn lag() -> Profile {
        Profile {
            name: "LAG".into(),
            description: "Loose-aggressive: wide ranges, lots of pressure and bluffs".into(),
            looseness: 0.45,
            aggression: 0.6,
            bet_freq: 0.9,
            bluff_freq: 0.5,
            call_down: 0.5,
            value_threshold: 0.58,
            raise_threshold: 0.75,
            size_bias: 0.4,
            slowplay: 0.1,
        }
    }

    pub fn station() -> Profile {
        Profile {
            name: "Calling Station".into(),
            description: "Loose-passive: limps and calls a lot, rarely raises or folds".into(),
            looseness: 0.7,
            aggression: -0.7,
            bet_freq: 0.4,
            bluff_freq: 0.05,
            call_down: 0.9,
            value_threshold: 0.7,
            raise_threshold: 0.92,
            size_bias: -0.5,
            slowplay: 0.3,
        }
    }

    pub fn maniac() -> Profile {
        Profile {
            name: "Maniac".into(),
            description: "Hyper-aggressive: raises and overbets constantly, bluffs a ton".into(),
            looseness: 0.85,
            aggression: 0.95,
            bet_freq: 0.95,
            bluff_freq: 0.75,
            call_down: 0.65,
            value_threshold: 0.52,
            raise_threshold: 0.68,
            size_bias: 0.9,
            slowplay: 0.05,
        }
    }

    /// All built-in presets.
    pub fn presets() -> Vec<Profile> {
        vec![
            Profile::reg(),
            Profile::nit(),
            Profile::tag(),
            Profile::lag(),
            Profile::station(),
            Profile::maniac(),
        ]
    }

    pub fn by_name(name: &str) -> Option<Profile> {
        let key = name.to_ascii_lowercase().replace([' ', '-', '_'], "");
        Profile::presets()
            .into_iter()
            .find(|p| p.name.to_ascii_lowercase().replace([' ', '-', '_'], "") == key)
            .or_else(|| match key.as_str() {
                "station" | "fish" => Some(Profile::station()),
                "balanced" | "regular" => Some(Profile::reg()),
                _ => None,
            })
    }

    /// Adjusts baseline (chart) frequencies for all 169 classes at one
    /// decision according to this profile's preflop tendencies.
    pub fn adjust_preflop(&self, base: &[Freqs], avail: &Availability) -> Vec<Freqs> {
        let mut f: Vec<Freqs> = base.to_vec();
        let combos: Vec<f32> =
            (0..NUM_CLASSES).map(|g| HandClass::from_grid_index(g).num_combos() as f32).collect();
        let order = strength::classes_by_strength(); // strongest first
        let fold_i = PreAction::Fold.index();
        let pas_i = PreAction::Passive.index();
        let raise_i = PreAction::Raise.index();
        let allin_i = PreAction::AllIn.index();
        let can_fold = avail.has(PreAction::Fold);
        let can_pass = avail.has(PreAction::Passive);
        let can_raise = avail.has(PreAction::Raise);

        // 1) Looseness: change how much of the range continues.
        if can_fold && self.looseness != 0.0 {
            let total: f32 = combos.iter().sum();
            let cont: f32 = (0..NUM_CLASSES).map(|g| (1.0 - f[g][fold_i]) * combos[g]).sum::<f32>() / total;
            let l = self.looseness.clamp(-1.0, 1.0);
            let target = if l > 0.0 { cont + l * (1.0 - cont) * 0.7 } else { cont * (1.0 + l * 0.7) };
            let mut delta = (target - cont) * total; // in combos
            if delta > 0.0 {
                // Strongest folding hands start continuing (mostly by calling).
                let into = if can_pass {
                    pas_i
                } else if can_raise {
                    raise_i
                } else {
                    allin_i
                };
                for &g in &order {
                    if delta <= 0.0 {
                        break;
                    }
                    let room = f[g][fold_i];
                    if room <= 0.0 {
                        continue;
                    }
                    let add = room.min(delta / combos[g]);
                    f[g][fold_i] -= add;
                    f[g][into] += add;
                    delta -= add * combos[g];
                }
            } else {
                // Weakest continuing hands start folding.
                let mut remove = -delta;
                for &g in order.iter().rev() {
                    if remove <= 0.0 {
                        break;
                    }
                    let c = 1.0 - f[g][fold_i];
                    if c <= 0.0 {
                        continue;
                    }
                    let take = c.min(remove / combos[g]);
                    let scale = (c - take) / c;
                    for a in [pas_i, raise_i, allin_i] {
                        f[g][a] *= scale;
                    }
                    f[g][fold_i] = 1.0 - (c - take);
                    remove -= take * combos[g];
                }
            }
        }

        // 2) Aggression: convert between calling and raising.
        let g_ = self.aggression.clamp(-1.0, 1.0);
        if g_ > 0.0 {
            for row in f.iter_mut() {
                if can_raise {
                    let mv = row[pas_i] * g_ * 0.6;
                    row[pas_i] -= mv;
                    row[raise_i] += mv;
                } else if avail.has(PreAction::AllIn) && can_fold {
                    let mv = row[pas_i] * g_ * 0.15;
                    row[pas_i] -= mv;
                    row[allin_i] += mv;
                }
            }
        } else if g_ < 0.0 && can_pass {
            for row in f.iter_mut() {
                let mv = row[raise_i] * (-g_) * 0.6;
                row[raise_i] -= mv;
                row[pas_i] += mv;
            }
        }
        f.iter().map(|row| avail.remap(*row)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::ChartSet;
    use holdem::Action;

    fn open_avail() -> Availability {
        Availability {
            actions: [
                Some(Action::Fold),
                Some(Action::Call),
                Some(Action::Raise(250)),
                Some(Action::Raise(10000)),
            ],
        }
    }

    fn vpip(freqs: &[Freqs]) -> f32 {
        let mut v = 0.0;
        for (g, f) in freqs.iter().enumerate() {
            v += (1.0 - f[0]) * HandClass::from_grid_index(g).num_combos() as f32;
        }
        v / 1326.0
    }

    #[test]
    fn styles_change_ranges() {
        let charts = ChartSet::hu_100bb();
        let avail = open_avail();
        let base: Vec<Freqs> = (0..NUM_CLASSES).map(|g| charts.freqs("-", g, &avail).0).collect();
        let nit = Profile::nit().adjust_preflop(&base, &avail);
        let station = Profile::station().adjust_preflop(&base, &avail);
        let reg = Profile::reg().adjust_preflop(&base, &avail);
        assert!((vpip(&reg) - vpip(&base)).abs() < 1e-4);
        assert!(vpip(&nit) < vpip(&base) - 0.2, "nit {}", vpip(&nit));
        assert!(vpip(&station) > vpip(&base), "station {}", vpip(&station));
        // Stations limp a lot.
        let limp: f32 = station.iter().map(|f| f[1]).sum();
        assert!(limp > 20.0);
        for f in nit.iter().chain(&station) {
            let s: f32 = f.iter().sum();
            assert!((s - 1.0).abs() < 1e-4);
            assert!(f.iter().all(|&x| x >= -1e-6));
        }
    }
}
