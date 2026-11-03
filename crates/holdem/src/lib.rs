//! Core No-Limit Hold'em primitives shared by the solver, the AI opponents and
//! the trainer front-end.
//!
//! * [`card`]  – cards, card sets and the deck.
//! * [`eval`]  – a fast 5–7 card hand evaluator.
//! * [`combo`] – indexing of the 1326 hole-card combinations and the 169 hand classes.
//! * [`range`] – weighted ranges and range-notation parsing (`"22+, A2s+, KTo+"`).
//! * [`menu`]  – the bet-size menu. The *same* code produces the legal action
//!   list for the live game and for the solver's game tree, which guarantees
//!   that every action taken at the table exists in the solved tree.
//! * [`game`]  – an N-player NLHE hand state machine (blinds, betting rounds,
//!   side pots, showdown).

// Index loops are clearer than iterator chains in the numeric code here.
#![allow(clippy::needless_range_loop)]

pub mod card;
pub mod combo;
pub mod eval;
pub mod game;
pub mod menu;
pub mod range;

pub use card::{Card, CardSet, Deck, ParseError};
pub use combo::{combo_cards, combo_index, HandClass, NUM_CLASSES, NUM_COMBOS};
pub use eval::{evaluate_cards, HandCategory, HandValue};
pub use game::{Action, ActionRecord, Chips, GameError, HandState, Street, TableConfig};
pub use menu::{BetMenu, BetSize, Category, PostflopMenu, PreflopMenu, Situation, StreetSizes};
pub use range::Range;
