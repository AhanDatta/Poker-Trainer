//! NLHE trainer: plays hands against AI opponents and reviews the hero's
//! decisions with the solver.
//!
//! * [`session::Session`] deals hands and tracks history/stats.
//! * [`hand::LiveHand`] runs one hand (AI decisions, dealing, logging).
//! * [`review::Review`] solves the hand's postflop streets in the background
//!   and grades every hero decision.
//! * [`ai`] / [`profiles`] implement the opponent styles.
//! * [`preflop`] holds the preflop charts and abstract preflop actions.

// Index loops are clearer than iterator chains in the numeric code here.
#![allow(clippy::needless_range_loop)]

pub mod ai;
pub mod config;
pub mod fmt;
pub mod hand;
pub mod preflop;
pub mod profiles;
pub mod review;
pub mod session;
pub mod strength;

pub use config::{FlopMode, HeroPosition, SolverSettings, TrainerConfig, VillainRange};
pub use hand::{LiveHand, LogEntry, StepResult};
pub use profiles::Profile;
pub use review::{DecisionReview, Review, ReviewStatus, Variant, Verdict};
pub use session::{Session, Stats};
