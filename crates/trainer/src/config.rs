//! Trainer configuration (TOML).

use crate::profiles::Profile;
use holdem::{BetMenu, PostflopMenu, StreetSizes};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Where the hero sits. The hero always has seat 0; the button is placed
/// around it.
///
/// In TOML: `"alternate"`, `"random"`, a position name such as `"btn"`, or
/// a list of them such as `["co", "btn"]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeroPosition {
    /// The button moves every hand (heads-up: alternate SB and BB).
    Alternate,
    /// A random position every hand.
    Random,
    /// Only these positions, by the table's names (upper case, no
    /// duplicates, never empty), in rotation order: the button moves on
    /// until the hero is in one of them. One name keeps the hero there.
    Seats(Vec<String>),
}

impl HeroPosition {
    /// Only the given positions (any case). Errors for an empty list or a
    /// name no table size has; duplicates are dropped.
    pub fn seats<S: AsRef<str>>(names: impl IntoIterator<Item = S>) -> Result<HeroPosition, String> {
        let mut seats: Vec<String> = Vec::new();
        for name in names {
            let name = name.as_ref().trim().to_ascii_uppercase();
            if !(2..=9).any(|n| holdem::game::position_offset(n, &name).is_some()) {
                return Err(format!(
                    "hero_position: unknown position {name:?} (use \"alternate\", \"random\", or positions \
                     such as \"utg\", \"hj\", \"co\", \"btn\", \"sb\", \"bb\")"
                ));
            }
            if !seats.contains(&name) {
                seats.push(name);
            }
        }
        if seats.is_empty() {
            return Err("hero_position: the list of positions is empty".into());
        }
        Ok(HeroPosition::Seats(seats))
    }

    /// Whether the hero may play the position `offset` seats after the
    /// button at a table of `players`.
    pub fn allows(&self, players: usize, offset: usize) -> bool {
        match self {
            HeroPosition::Alternate | HeroPosition::Random => true,
            HeroPosition::Seats(seats) => {
                seats.iter().any(|s| holdem::game::position_offset(players, s) == Some(offset))
            }
        }
    }

    /// Checks that a table of `players` has every chosen position.
    pub fn validate(&self, players: usize) -> Result<(), String> {
        let HeroPosition::Seats(seats) = self else { return Ok(()) };
        match seats.iter().find(|s| holdem::game::position_offset(players, s).is_none()) {
            Some(s) => Err(format!(
                "hero_position: a {players}-player table has no {s} (its positions: {})",
                holdem::game::positions(players).join(", ")
            )),
            None => Ok(()),
        }
    }

    /// The chosen positions with `name` ticked or unticked, in the preflop
    /// order of a table of `players` (rotation and random count as none
    /// chosen). Errors when no position would be left, or the table has no
    /// `name`.
    pub fn toggled(&self, players: usize, name: &str) -> Result<HeroPosition, String> {
        let Some(offset) = holdem::game::position_offset(players, name) else {
            return Err(format!("A {players}-player table has no {name}."));
        };
        let on = |o: usize| {
            let chosen = matches!(self, HeroPosition::Seats(_)) && self.allows(players, o);
            chosen != (o == offset)
        };
        let names = holdem::game::positions(players)
            .into_iter()
            .filter(|p| holdem::game::position_offset(players, p).is_some_and(on));
        HeroPosition::seats(names).map_err(|_| "At least one position stays ticked.".to_string())
    }

    /// The setting for a table of `players`: chosen positions the table
    /// doesn't have are dropped, and rotation is used if none is left.
    pub fn for_table(&self, players: usize) -> HeroPosition {
        match self {
            HeroPosition::Seats(seats) => {
                let kept: Vec<&String> =
                    seats.iter().filter(|s| holdem::game::position_offset(players, s).is_some()).collect();
                HeroPosition::seats(kept).unwrap_or(HeroPosition::Alternate)
            }
            other => other.clone(),
        }
    }
}

impl Serialize for HeroPosition {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            HeroPosition::Alternate => s.serialize_str("alternate"),
            HeroPosition::Random => s.serialize_str("random"),
            HeroPosition::Seats(seats) if seats.len() == 1 => s.serialize_str(&seats[0].to_ascii_lowercase()),
            HeroPosition::Seats(seats) => {
                seats.iter().map(|p| p.to_ascii_lowercase()).collect::<Vec<_>>().serialize(s)
            }
        }
    }
}

impl<'de> Deserialize<'de> for HeroPosition {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<HeroPosition, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            One(String),
            Many(Vec<String>),
        }
        let pos = match Repr::deserialize(d)? {
            Repr::One(s) if s.eq_ignore_ascii_case("alternate") => Ok(HeroPosition::Alternate),
            Repr::One(s) if s.eq_ignore_ascii_case("random") => Ok(HeroPosition::Random),
            Repr::One(s) => HeroPosition::seats([s]),
            Repr::Many(v) => HeroPosition::seats(v),
        };
        pos.map_err(serde::de::Error::custom)
    }
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

/// Where heads-up solves run (three-way solves always use the CPU).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolverBackend {
    /// The CPU.
    #[default]
    Cpu,
    /// The GPU, for turn and flop spots, when the build has GPU support and
    /// an NVIDIA GPU is free; the CPU otherwise.
    Auto,
    /// The GPU for every heads-up solve; the CPU (with a note saying why)
    /// when it can't be used.
    Gpu,
}

impl From<SolverBackend> for solver::Backend {
    fn from(b: SolverBackend) -> solver::Backend {
        match b {
            SolverBackend::Cpu => solver::Backend::Cpu,
            SolverBackend::Auto => solver::Backend::Auto,
            SolverBackend::Gpu => solver::Backend::Gpu,
        }
    }
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
    /// Where heads-up solves run.
    pub backend: SolverBackend,
    /// Three-way pots: sampled turn and river cards for flop solves.
    pub multiway_turn_samples: usize,
    pub multiway_river_samples: usize,
    /// Three-way pots: stop at this average best-response gain (% pot).
    pub multiway_target_pct: f64,
    /// Exact card removal between the two opponents in 3-way solves (the
    /// default; about 2.5x slower). Off, their hands are treated as
    /// independent, which is close with wide ranges but can be off by
    /// several % of the pot when narrow ranges overlap (e.g. squeeze pots).
    pub multiway_exact_removal: bool,
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
            backend: SolverBackend::Cpu,
            multiway_turn_samples: 8,
            multiway_river_samples: 4,
            multiway_target_pct: 1.5,
            multiway_exact_removal: true,
            multiway_future_menu: PostflopMenu {
                turn: StreetSizes::new(&["66%"], &["3x"]),
                river: StreetSizes::new(&["75%"], &["3x"]),
                max_raises: 1,
                ..PostflopMenu::default()
            },
        }
    }
}

/// The session's frequency score (see [`crate::scoring`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoringSettings {
    /// Weight each decision by the pot (in big blinds), so frequency leaks
    /// in big pots count more.
    pub pot_weighted: bool,
    /// Monte Carlo samples of a perfect mixer used to tell leaks from luck.
    pub simulations: usize,
}

impl Default for ScoringSettings {
    fn default() -> Self {
        ScoringSettings { pot_weighted: true, simulations: 1000 }
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
    /// Where you sit: rotate, random, or only the chosen positions.
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
    pub scoring: ScoringSettings,
    /// Show a random number (0-99) at each of your decisions, to help you
    /// play mixed strategies (e.g. bet when it's below 30).
    pub show_rng: bool,
    /// Save finished hands so the session can be continued later.
    pub save_sessions: bool,
    /// Where sessions are saved (default: the user data directory, e.g.
    /// ~/.local/share/poker-trainer/sessions).
    pub sessions_dir: Option<String>,
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
            scoring: ScoringSettings::default(),
            show_rng: true,
            save_sessions: true,
            sessions_dir: None,
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
        self.hero_position.validate(self.players)?;
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

    /// Directory of saved sessions: `sessions_dir`, or the platform's data
    /// directory (`None` when there's no home directory to put it in).
    pub fn sessions_root(&self) -> Option<std::path::PathBuf> {
        if let Some(d) = &self.sessions_dir {
            return Some(d.into());
        }
        let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        let base = if cfg!(windows) {
            env("APPDATA")
        } else if cfg!(target_os = "macos") {
            env("HOME").map(|h| h.join("Library/Application Support"))
        } else {
            env("XDG_DATA_HOME").or_else(|| env("HOME").map(|h| h.join(".local/share")))
        };
        base.map(|b| b.join("poker-trainer").join("sessions"))
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
    #[test]
    fn solver_backend_parses_and_defaults_to_the_cpu() {
        use super::{SolverBackend, TrainerConfig};
        assert_eq!(TrainerConfig::default().solver.backend, SolverBackend::Cpu);
        for (text, want) in [("cpu", SolverBackend::Cpu), ("auto", SolverBackend::Auto), ("gpu", SolverBackend::Gpu)] {
            let cfg: TrainerConfig = toml::from_str(&format!("[solver]\nbackend = \"{text}\"\n")).unwrap();
            assert_eq!(cfg.solver.backend, want);
            let back = toml::to_string(&cfg).unwrap();
            assert!(back.contains(&format!("backend = \"{text}\"")), "{back}");
            assert_eq!(solver::Backend::from(want), match want {
                SolverBackend::Cpu => solver::Backend::Cpu,
                SolverBackend::Auto => solver::Backend::Auto,
                SolverBackend::Gpu => solver::Backend::Gpu,
            });
        }
        assert!(toml::from_str::<TrainerConfig>("[solver]\nbackend = \"tpu\"\n").is_err());
    }

    use super::*;

    #[test]
    fn toml_roundtrip() {
        let cfg = TrainerConfig::default();
        let text = cfg.to_toml();
        let back = TrainerConfig::from_toml(&text).unwrap();
        assert_eq!(cfg, back);
    }

    fn pos(text: &str) -> Result<HeroPosition, String> {
        TrainerConfig::from_toml(&format!("players = 9\nhero_position = {text}\n")).map(|c| c.hero_position)
    }

    #[test]
    fn hero_position_parses() {
        assert_eq!(pos("\"alternate\""), Ok(HeroPosition::Alternate));
        assert_eq!(pos("\"Random\""), Ok(HeroPosition::Random));
        assert_eq!(pos("\"sb\""), Ok(HeroPosition::Seats(vec!["SB".into()])));
        assert_eq!(pos("\"utg+2\""), Ok(HeroPosition::Seats(vec!["UTG+2".into()])));
        let co_btn = HeroPosition::Seats(vec!["CO".into(), "BTN".into()]);
        assert_eq!(pos("[\"co\", \"BTN\", \"Co\"]"), Ok(co_btn));
        assert!(pos("\"shark\"").unwrap_err().contains("unknown position"));
        assert!(pos("[]").unwrap_err().contains("empty"));
        assert!(pos("[\"btn\", \"rotate\"]").is_err());
        assert!(pos("3").is_err());
        // A position the table size doesn't have.
        let err = TrainerConfig::from_toml("players = 6\nhero_position = \"lj\"\n").unwrap_err();
        assert!(err.contains("6-player table has no LJ"), "{err}");
        assert!(err.contains("UTG, HJ, CO, BTN, SB, BB"), "{err}");
        assert!(TrainerConfig::from_toml("players = 2\nhero_position = \"btn\"\n").is_ok());
        assert!(TrainerConfig::from_toml("players = 2\nhero_position = \"co\"\n").is_err());
    }

    #[test]
    fn hero_position_roundtrips() {
        for p in [
            HeroPosition::Alternate,
            HeroPosition::Random,
            HeroPosition::seats(["btn"]).unwrap(),
            HeroPosition::seats(["UTG", "UTG+1", "BB"]).unwrap(),
        ] {
            let cfg = TrainerConfig { players: 9, hero_position: p.clone(), ..TrainerConfig::default() };
            let text = cfg.to_toml();
            assert_eq!(TrainerConfig::from_toml(&text).unwrap().hero_position, p, "{text}");
            let json = serde_json::to_string(&cfg).unwrap();
            assert_eq!(serde_json::from_str::<TrainerConfig>(&json).unwrap().hero_position, p, "{json}");
        }
        let one = HeroPosition::seats(["BTN"]).unwrap();
        let one = TrainerConfig { hero_position: one, ..TrainerConfig::default() };
        assert!(one.to_toml().contains("hero_position = \"btn\""));
        let two = TrainerConfig { hero_position: HeroPosition::seats(["co", "btn"]).unwrap(), ..one };
        let text = two.to_toml();
        assert!(text.contains("hero_position = [\n    \"co\",\n    \"btn\",\n]"), "{text}");
        assert!(TrainerConfig::default().to_toml().contains("hero_position = \"alternate\""));
    }

    #[test]
    fn hero_position_for_other_tables() {
        let p = HeroPosition::seats(["UTG+2", "BTN", "SB"]).unwrap();
        assert_eq!(p.for_table(9), p);
        assert_eq!(p.for_table(6), HeroPosition::seats(["BTN", "SB"]).unwrap());
        assert_eq!(HeroPosition::seats(["LJ", "UTG+1"]).unwrap().for_table(6), HeroPosition::Alternate);
        assert_eq!(HeroPosition::Random.for_table(2), HeroPosition::Random);
        for n in 2..=9 {
            for p in [&p, &HeroPosition::Alternate, &HeroPosition::Random] {
                assert_eq!(p.for_table(n).validate(n), Ok(()));
            }
        }
    }

    #[test]
    fn hero_position_toggles() {
        let btn = HeroPosition::seats(["BTN"]).unwrap();
        // Ticking keeps the table's preflop order.
        let p = btn.toggled(6, "utg").unwrap();
        assert_eq!(p, HeroPosition::seats(["UTG", "BTN"]).unwrap());
        assert_eq!(p.toggled(6, "BTN").unwrap(), HeroPosition::seats(["UTG"]).unwrap());
        // The last one can't be unticked.
        assert!(btn.toggled(6, "BTN").is_err());
        assert!(btn.toggled(6, "UTG+2").is_err());
        // From rotation nothing is ticked yet.
        assert_eq!(HeroPosition::Alternate.toggled(9, "CO").unwrap(), HeroPosition::seats(["CO"]).unwrap());
        // Heads-up BTN is the SB.
        assert_eq!(btn.toggled(2, "BB").unwrap(), HeroPosition::seats(["SB", "BB"]).unwrap());
    }

    proptest::proptest! {
        /// Ticking a position flips exactly that one; the result is valid
        /// for the table.
        #[test]
        fn toggling_flips_one_position(n in 2usize..=9, mask in 1u32..512, pick in 0usize..9) {
            let all = holdem::game::positions(n);
            let mask = mask & ((1 << n) - 1);
            proptest::prop_assume!(mask != 0);
            let chosen: Vec<&str> = (0..n).filter(|i| mask >> i & 1 == 1).map(|i| all[i]).collect();
            let p = HeroPosition::seats(&chosen).unwrap();
            proptest::prop_assert_eq!(p.validate(n), Ok(()));
            let name = all[pick % n];
            match p.toggled(n, name) {
                Ok(q) => {
                    proptest::prop_assert_eq!(q.validate(n), Ok(()));
                    for o in 0..n {
                        let flipped = holdem::game::position_offset(n, name) == Some(o);
                        proptest::prop_assert_eq!(q.allows(n, o), p.allows(n, o) != flipped);
                    }
                }
                Err(_) => proptest::prop_assert_eq!(chosen, vec![name]),
            }
        }
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
