//! Trainer configuration (TOML).

use crate::profiles::Profile;
use holdem::{BetMenu, PostflopMenu, StreetSizes};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HeroPosition {
    Random,
    Alternate,
    Sb,
    Bb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlopMode {
    /// Stratified sample of turn/river cards: seconds, approximate.
    Fast,
    /// Every turn and river card: exact, but large and slow.
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VillainRange {
    /// Villain's range comes from the GTO charts for its preflop line.
    Gto,
    /// Villain's range is what its AI style actually plays preflop.
    Actual,
    /// Solve both and show them side by side.
    Both,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverSettings {
    pub enabled: bool,
    pub flop_mode: FlopMode,
    pub flop_turn_samples: usize,
    pub flop_river_samples: usize,
    pub villain_range: VillainRange,
    /// Stop each solve at this exploitability (% of the pot).
    pub target_exploitability_pct: f64,
    pub max_iterations: u32,
    /// Exact flop solves larger than this fall back to fast mode.
    pub max_memory_gb: f64,
    /// Drop combos below this fraction of the range's largest weight.
    pub prune: f32,
    /// Re-solve from a decision node when a player's range reaching it is
    /// below this fraction of its street-start range (off-equilibrium line).
    pub off_equilibrium_threshold: f64,
    /// Minimum weight (relative to the range's max) given to your actual
    /// hand so the solver always has a strategy for it.
    pub hero_hand_floor: f32,
    /// Bet sizes used for the turn and river *inside flop solves*
    /// (simplified to keep flop trees small).
    pub flop_future_menu: PostflopMenu,
    /// Worker threads for the solver (0 = all cores).
    pub threads: usize,
}

impl Default for SolverSettings {
    fn default() -> Self {
        let future = PostflopMenu {
            turn: StreetSizes::new(&["66%"], &["3x"]),
            river: StreetSizes::new(&["75%"], &["3x"]),
            max_raises: 2,
            ..PostflopMenu::default()
        };
        SolverSettings {
            enabled: true,
            flop_mode: FlopMode::Fast,
            flop_turn_samples: 8,
            flop_river_samples: 6,
            villain_range: VillainRange::Gto,
            target_exploitability_pct: 0.5,
            max_iterations: 400,
            max_memory_gb: 6.0,
            prune: 0.001,
            off_equilibrium_threshold: 0.015,
            hero_hand_floor: 0.02,
            flop_future_menu: future,
            threads: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainerConfig {
    /// Number of players at the table (only 2 is supported for now).
    pub players: usize,
    /// Starting stack in big blinds (every hand starts fresh).
    pub stack_bb: f64,
    /// Opponent style: a preset name (Reg, Nit, TAG, LAG, Calling Station,
    /// Maniac), the name of a custom profile, or "random".
    pub opponent: String,
    /// With `opponent = "random"`: pick a new style every hand (otherwise
    /// once per session).
    pub random_each_hand: bool,
    /// Reveal the opponent's style during the hand (it's always shown in the
    /// review).
    pub show_opponent_style: bool,
    pub hero_position: HeroPosition,
    /// Preflop chart file (built-in heads-up 100bb charts when unset).
    pub charts_file: Option<String>,
    /// Milliseconds between AI actions in the UI.
    pub ai_delay_ms: u64,
    /// Random seed (unset = random).
    pub seed: Option<u64>,
    pub menu: BetMenu,
    pub solver: SolverSettings,
    /// Extra opponent styles.
    pub custom_profiles: Vec<Profile>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            players: 2,
            stack_bb: 100.0,
            opponent: "random".into(),
            random_each_hand: true,
            show_opponent_style: false,
            hero_position: HeroPosition::Alternate,
            charts_file: None,
            ai_delay_ms: 450,
            seed: None,
            menu: BetMenu::default(),
            solver: SolverSettings::default(),
            custom_profiles: Vec::new(),
        }
    }
}

impl TrainerConfig {
    pub fn from_toml(text: &str) -> Result<TrainerConfig, String> {
        let cfg: TrainerConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &str) -> Result<TrainerConfig, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        TrainerConfig::from_toml(&text).map_err(|e| format!("{path}: {e}"))
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("config serializes")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.players != 2 {
            return Err(format!("players = {}: only heads-up (2) is implemented so far", self.players));
        }
        if !(5.0..=1000.0).contains(&self.stack_bb) {
            return Err("stack_bb must be between 5 and 1000".into());
        }
        if !self.opponent.eq_ignore_ascii_case("random") && self.profile_by_name(&self.opponent).is_none() {
            return Err(format!("unknown opponent style {:?}", self.opponent));
        }
        Ok(())
    }

    /// All available styles (presets + custom).
    pub fn profiles(&self) -> Vec<Profile> {
        let mut v = Profile::presets();
        v.extend(self.custom_profiles.iter().cloned());
        v
    }

    pub fn profile_by_name(&self, name: &str) -> Option<Profile> {
        self.custom_profiles
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
            .cloned()
            .or_else(|| Profile::by_name(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_roundtrip() {
        let cfg = TrainerConfig::default();
        let text = cfg.to_toml();
        let back = TrainerConfig::from_toml(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn partial_toml() {
        let cfg = TrainerConfig::from_toml(
            "opponent = \"maniac\"\n[solver]\nflop_mode = \"exact\"\nvillain_range = \"both\"\n[menu.postflop.flop]\nbet = [\"50%\"]\nraise = [\"2.5x\"]\n",
        )
        .unwrap();
        assert_eq!(cfg.opponent, "maniac");
        assert_eq!(cfg.solver.flop_mode, FlopMode::Exact);
        assert_eq!(cfg.solver.villain_range, VillainRange::Both);
        assert_eq!(cfg.menu.postflop.flop.bet.len(), 1);
        assert_eq!(cfg.stack_bb, 100.0);
        assert!(TrainerConfig::from_toml("players = 6").is_err());
        assert!(TrainerConfig::from_toml("opponent = \"shark\"").is_err());
    }
}
