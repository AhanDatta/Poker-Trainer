//! NLHE trainer: plays hands against AI opponents and reviews the hero's
//! decisions with the solver.
//!
//! * [`session::Session`] deals hands and tracks history/stats.
//! * [`hand::LiveHand`] runs one hand (AI decisions, dealing, logging).
//! * [`review::Review`] solves the hand's postflop streets in the background
//!   and grades every hero decision.
//! * [`ai`] / [`profiles`] implement the opponent styles.
//! * [`preflop`] holds the preflop charts and abstract preflop actions.
//! * [`study`] is the solver study mode (browse charts and solutions).

// Index loops are clearer than iterator chains in the numeric code here.
#![allow(clippy::needless_range_loop)]

pub mod ai;
pub mod charts_import;
pub mod config;
pub mod export;
pub mod fmt;
pub mod hand;
pub mod population;
pub mod preflop;
pub mod profiles;
pub mod review;
pub mod save;
pub mod scoring;
pub mod session;
pub mod solver_tag;
pub mod strength;
pub mod study;

pub use config::{
    FlopMode, GameType, HeroPosition, ScoringSettings, SolverBackend, SolverSettings, TrainerConfig, VillainRange,
};
pub use hand::{LiveHand, LogEntry, StepResult};
pub use profiles::Profile;
pub use review::{DecisionReview, Review, ReviewStatus, SpotInfo, Variant, Verdict};
pub use solver_tag::{SolveSummary, SolvedBy, SolverTag};
pub use scoring::{FrequencyReport, ScoreOptions, ScoredDecision};
pub use session::{Session, Stats};
