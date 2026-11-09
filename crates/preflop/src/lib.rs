//! A preflop model: it solves only the preflop betting, and values a pot
//! that sees a flop as each player's equity times an equity-realization
//! factor, instead of solving the postflop game.
//!
//! * [`spec`] says which game is solved (table size, stacks, rake, raise
//!   sizes, tree rules).
//! * [`tree`] is the preflop betting tree, built with the trainer's game
//!   engine and keyed by the trainer's spot keys.
//! * [`realize`] holds the realization factors and what they depend on.
//! * [`deal`] and [`rng`] sample hole cards and boards.
//! * [`solver`] finds the strategies (Monte Carlo CFR).
//! * [`charts`] turns a solve into the trainer's charts and compares charts.

// Index loops are clearer than iterator chains in the numeric code here.
#![allow(clippy::needless_range_loop)]

pub mod charts;
pub mod deal;
pub mod realize;
pub mod rng;
pub mod solver;
pub mod spec;
pub mod tree;

pub use realize::{Context, Realization};
pub use solver::{Options, Solver};
pub use spec::{GameSpec, Limp, SpecError, TreeRules, BB, MAX_PLAYERS};
pub use tree::{Decision, Leaf, LeafKind, Node, Tree, NO_CHILD};
