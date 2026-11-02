//! Trainer configuration (TOML).

use crate::profiles::Profile;
use holdem::{BetMenu, PostflopMenu, StreetSizes};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// Every turn card but a stratified sample of river cards: ~4x faster
    /// than exact, slightly approximate.
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
    /// Three-way pots: sampled turn and river cards for flop solves.
    pub multiway_turn_samples: usize,
    pub multiway_river_samples: usize,
    /// Three-way pots: stop at this average best-response gain (% pot).
    pub multiway_target_pct: f64,
    /// Three-way pots: sizes on later streets (kept small: no raises).
    pub multiway_future_menu: PostflopMenu,
}

impl Default for SolverSettings {
    fn default() -> Self {
        // Later streets inside flop solves: one bet size, raises only all-in.
        // Keeps exact heads-up flop trees around 2-3 GB.
        let future = PostflopMenu {
            turn: StreetSizes::new(&["66%"], &["allin"]),
            river: StreetSizes::new(&["75%"], &["allin"]),
            max_raises: 2,
            ..PostflopMenu::default()
        };
        SolverSettings {
            enabled: true,
            flop_mode: FlopMode::Exact,
            // Fast mode: every turn card, a stratified sample of rivers.
            // (Sampling turns too was measured to be much less accurate.)
            flop_turn_samples: 49,
            flop_river_samples: 10,
            villain_range: VillainRange::Gto,
            target_exploitability_pct: 0.5,
            max_iterations: 400,
            max_memory_gb: 5.0,
            prune: 0.001,
            off_equilibrium_threshold: 0.015,
            hero_hand_floor: 0.02,
            flop_future_menu: future,
            threads: 0,
            multiway_turn_samples: 8,
            multiway_river_samples: 4,
            multiway_target_pct: 1.5,
            multiway_future_menu: PostflopMenu {
                turn: StreetSizes::new(&["66%"], &["3x"]),
                river: StreetSizes::new(&["75%"], &["3x"]),
                max_raises: 1,
                ..PostflopMenu::default()
            },
        }
    }
}

/// Kind of game: how likely each opponent style is to sit at the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GameType {
    /// Mostly winning regulars.
    Tough,
    /// A mix of regulars and recreational players.
    Typical,
    /// Mostly recreational players: fish, stations, weak-tight players.
    Soft,
    /// Maniacs and loose-aggressive players.
    Wild,
    /// Exactly the styles in `custom_table`.
    Custom,
}

impl GameType {
    pub const ALL: [GameType; 5] =
        [GameType::Tough, GameType::Typical, GameType::Soft, GameType::Wild, GameType::Custom];

    pub fn label(self) -> &'static str {
        match self {
            GameType::Tough => "Tough game",
            GameType::Typical => "Typical game",
            GameType::Soft => "Soft game",
            GameType::Wild => "Wild game",
            GameType::Custom => "Custom table",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            GameType::Tough => "mostly regs and TAGs, a few LAGs and nits",
            GameType::Typical => "regs and TAGs plus some fish, rocks and the odd maniac",
            GameType::Soft => "mostly fish, calling stations and weak-tight players",
            GameType::Wild => "maniacs and LAGs, with stations and fish",
            GameType::Custom => "your own count of each style",
        }
    }

    /// Relative chance of each style (by preset name) for the preset game
    /// types; empty for `Custom`.
    pub fn weights(self) -> &'static [(&'static str, f64)] {
        match self {
            GameType::Tough => &[("Reg", 4.0), ("TAG", 3.0), ("LAG", 2.0), ("Nit", 1.0)],
            GameType::Typical => &[
                ("Reg", 3.0),
                ("TAG", 2.0),
                ("Nit", 1.5),
                ("LAG", 1.0),
                ("Rock", 1.0),
                ("Fish", 2.0),
                ("Calling Station", 0.5),
                ("Maniac", 0.3),
            ],
            GameType::Soft => &[
                ("Reg", 1.5),
                ("TAG", 0.5),
                ("Nit", 1.0),
                ("Rock", 2.0),
                ("Fish", 3.0),
                ("Calling Station", 2.0),
                ("LAG", 0.5),
                ("Maniac", 0.5),
            ],
            GameType::Wild => {
                &[("Maniac", 3.0), ("LAG", 2.5), ("Calling Station", 1.5), ("Fish", 1.5), ("Reg", 1.0)]
            }
            GameType::Custom => &[],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainerConfig {
    /// Number of players at the table (2 = heads-up, 6 = 6-max, 9 = full ring).
    pub players: usize,
    /// Starting stack in big blinds (every hand starts fresh).
    pub stack_bb: f64,
    /// Opponent style: "random" (drawn according to `game`), or a style
    /// name (preset or custom profile) for every opponent.
    pub opponent: String,
    /// With `opponent = "random"`: the kind of game, i.e. how likely each
    /// style is to sit at the table.
    pub game: GameType,
    /// With `game = "custom"`: how many opponents of each style sit at the
    /// table, e.g. `{ Reg = 2, Fish = 2, Maniac = 1 }`. Seats beyond the
    /// counts are filled as in a typical game.
    pub custom_table: BTreeMap<String, usize>,
    /// Chance (percent, per opponent and hand) that a player leaves and a
    /// new one sits down. 0 keeps the lineup for the session; 100 draws a
    /// new lineup every hand.
    pub seat_change_pct: f64,
    /// Reveal the opponent's style during the hand (it's always shown in the
    /// review).
    pub show_opponent_style: bool,
    pub hero_position: HeroPosition,
    /// Rake in percent of the pot (no flop, no drop). 0 disables rake.
    pub rake_pct: f64,
    /// Rake cap in big blinds.
    pub rake_cap_bb: f64,
    /// Optional preflop chart files per table size (see `import-charts`).
    /// Their spots replace the built-in charts' (solver-based heads-up and
    /// 6-max charts); spots no chart covers use position-aware rules.
    pub charts_hu: Option<String>,
    pub charts_6max: Option<String>,
    pub charts_9max: Option<String>,
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
            game: GameType::Typical,
            custom_table: [("Reg", 2), ("TAG", 1), ("Fish", 1), ("Maniac", 1)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            seat_change_pct: 2.0,
            show_opponent_style: false,
            hero_position: HeroPosition::Alternate,
            rake_pct: 0.0,
            rake_cap_bb: 3.0,
            charts_hu: None,
            charts_6max: None,
            charts_9max: None,
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
        if !(2..=9).contains(&self.players) {
            return Err(format!("players = {}: tables of 2 to 9 players are supported", self.players));
        }
        if !(5.0..=1000.0).contains(&self.stack_bb) {
            return Err("stack_bb must be between 5 and 1000".into());
        }
        if !(0.0..=20.0).contains(&self.rake_pct) || self.rake_cap_bb < 0.0 {
            return Err("rake_pct must be within 0-20 and rake_cap_bb non-negative".into());
        }
        if !self.opponent.eq_ignore_ascii_case("random") && self.profile_by_name(&self.opponent).is_none() {
            return Err(format!("unknown opponent style {:?}", self.opponent));
        }
        for name in self.custom_table.keys() {
            if self.profile_by_name(name).is_none() {
                return Err(format!("custom_table: unknown style {name:?}"));
            }
        }
        if !(0.0..=100.0).contains(&self.seat_change_pct) {
            return Err("seat_change_pct must be within 0-100".into());
        }
        Ok(())
    }

    /// Chart file configured for a table size.
    pub fn charts_for(&self, players: usize) -> Option<&String> {
        match players {
            0..=2 => self.charts_hu.as_ref(),
            3..=6 => self.charts_6max.as_ref(),
            _ => self.charts_9max.as_ref(),
        }
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
        assert!(TrainerConfig::from_toml("players = 6").is_ok());
        assert!(TrainerConfig::from_toml("players = 12").is_err());
        assert!(TrainerConfig::from_toml("opponent = \"shark\"").is_err());
    }
}
