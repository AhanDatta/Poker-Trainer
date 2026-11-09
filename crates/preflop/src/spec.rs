//! The game the preflop model solves: table size, stacks, rake, raise sizes
//! and the rules that keep the betting tree small.

use holdem::{BetMenu, Chips};
use std::fmt;

/// Chips per big blind in the model's game (the small blind is half).
pub const BB: Chips = 100;

/// Largest table the model handles.
pub const MAX_PLAYERS: usize = 9;

/// Who may open-limp (call the big blind before any raise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limp {
    /// Raise or fold first in, from every position.
    Never,
    /// Only the small blind, when everyone before it folded.
    SmallBlind,
    /// Every position, including over-limps.
    Any,
}

/// Rules that prune the betting tree. Actions they remove get frequency 0
/// in the charts; the trainer's fallback rules cover lines the tree lacks.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeRules {
    pub limp: Limp,
    /// All-in is offered once this many raises were made, and whenever the
    /// menu has no smaller raise left. 0 offers it everywhere.
    pub allin_after_raises: u32,
    /// Most cold callers of one raise: once this many players called it, a
    /// player making their first decision can only fold or re-raise. The
    /// big blind, and players who already acted, can always call.
    /// `usize::MAX` for no limit.
    pub max_cold_callers: usize,
    /// Building stops with an error beyond this many nodes.
    pub max_nodes: usize,
}

impl Default for TreeRules {
    fn default() -> Self {
        TreeRules {
            limp: Limp::Never,
            allin_after_raises: 2,
            max_cold_callers: usize::MAX,
            max_nodes: 5_000_000,
        }
    }
}

/// A preflop game with equal stacks.
#[derive(Clone, Debug, PartialEq)]
pub struct GameSpec {
    pub players: usize,
    /// Every player's stack, in big blinds.
    pub stack_bb: f64,
    /// Rake as a fraction of pots that see a flop.
    pub rake_rate: f64,
    /// Most rake per hand, in big blinds.
    pub rake_cap_bb: f64,
    /// Raise sizes (the preflop part) and the all-in threshold.
    pub menu: BetMenu,
    pub rules: TreeRules,
}

/// Why a [`GameSpec`] can't be solved.
#[derive(Clone, Debug, PartialEq)]
pub enum SpecError {
    Players(usize),
    Stack(f64),
    Rake {
        rate: f64,
        cap_bb: f64,
    },
    Menu(String),
    /// The tree outgrew `TreeRules::max_nodes`.
    TooLarge {
        limit: usize,
    },
    /// A betting round ended with unequal commitments (can't happen with
    /// equal stacks; reported instead of producing wrong payoffs).
    UnequalCommitments(String),
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecError::Players(n) => write!(f, "{n} players: 2 to {MAX_PLAYERS} are supported"),
            SpecError::Stack(s) => write!(f, "stack of {s} big blinds: it must be between 2 and 1000"),
            SpecError::Rake { rate, cap_bb } => {
                write!(
                    f,
                    "rake {rate} capped at {cap_bb} bb: the rate must be in [0, 1) and the cap at least 0"
                )
            }
            SpecError::Menu(why) => write!(f, "preflop raise sizes: {why}"),
            SpecError::TooLarge { limit } => {
                write!(f, "the betting tree has more than {limit} nodes; tighten the tree rules")
            }
            SpecError::UnequalCommitments(line) => write!(f, "unequal commitments after {line}"),
        }
    }
}

impl std::error::Error for SpecError {}

impl GameSpec {
    /// A `players`-handed game with 100bb stacks, the default raise sizes
    /// and tree rules, and no rake.
    pub fn new(players: usize) -> GameSpec {
        GameSpec {
            players,
            stack_bb: 100.0,
            rake_rate: 0.0,
            rake_cap_bb: 0.0,
            menu: BetMenu::default(),
            rules: TreeRules::default(),
        }
    }

    pub fn with_rake(mut self, rate: f64, cap_bb: f64) -> GameSpec {
        self.rake_rate = rate;
        self.rake_cap_bb = cap_bb;
        self
    }

    pub fn with_stack(mut self, stack_bb: f64) -> GameSpec {
        self.stack_bb = stack_bb;
        self
    }

    /// Every player's stack in chips.
    pub fn stack(&self) -> Chips {
        (self.stack_bb * BB as f64).round() as Chips
    }

    /// The rake cap in chips.
    pub fn rake_cap(&self) -> Chips {
        (self.rake_cap_bb * BB as f64).round() as Chips
    }

    /// Rake taken from a pot of `pot` chips that sees a flop.
    pub fn rake(&self, pot: Chips) -> Chips {
        ((pot as f64 * self.rake_rate).floor() as Chips).min(self.rake_cap()).max(0)
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        if !(2..=MAX_PLAYERS).contains(&self.players) {
            return Err(SpecError::Players(self.players));
        }
        if !(2.0..=1000.0).contains(&self.stack_bb) {
            return Err(SpecError::Stack(self.stack_bb));
        }
        let rake_ok =
            (0.0..1.0).contains(&self.rake_rate) && self.rake_cap_bb >= 0.0 && self.rake_cap_bb.is_finite();
        if !rake_ok {
            return Err(SpecError::Rake { rate: self.rake_rate, cap_bb: self.rake_cap_bb });
        }
        let pre = &self.menu.preflop;
        let sizes = [pre.open_bb, pre.iso_bb].into_iter().chain(pre.reraise_mult.iter().copied());
        for size in sizes {
            if !size.is_finite() || size <= 1.0 {
                return Err(SpecError::Menu(format!("{size} is not a raise (sizes must be above 1)")));
            }
        }
        if !pre.iso_per_limper_bb.is_finite() || pre.iso_per_limper_bb < 0.0 {
            return Err(SpecError::Menu(format!("{} extra per limper", pre.iso_per_limper_bb)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rake_is_capped_and_floored() {
        let spec = GameSpec::new(6).with_rake(0.05, 3.0);
        assert_eq!(spec.rake(550), 27);
        assert_eq!(spec.rake(20_000), 300);
        assert_eq!(GameSpec::new(6).rake(20_000), 0);
    }

    #[test]
    fn validation_rejects_bad_specs() {
        assert_eq!(GameSpec::new(6).validate(), Ok(()));
        assert_eq!(GameSpec::new(1).validate(), Err(SpecError::Players(1)));
        assert_eq!(GameSpec::new(10).validate(), Err(SpecError::Players(10)));
        assert!(GameSpec::new(6).with_stack(1.0).validate().is_err());
        assert!(GameSpec::new(6).with_stack(f64::NAN).validate().is_err());
        assert!(GameSpec::new(6).with_rake(1.0, 3.0).validate().is_err());
        assert!(GameSpec::new(6).with_rake(0.05, -1.0).validate().is_err());
        let mut spec = GameSpec::new(6);
        spec.menu.preflop.reraise_mult = vec![3.5, 0.9];
        assert!(matches!(spec.validate(), Err(SpecError::Menu(_))));
    }
}
