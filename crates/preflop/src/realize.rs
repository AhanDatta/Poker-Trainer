//! Equity realization: what a pot that sees a flop is worth to each player.
//!
//! The model values such a pot for a player as
//! `factor × equity × pot − chips put in`, where the factor depends on the
//! player's hand class and on the [`Context`]: how the pot was built, the
//! player's part in it and its position after the flop. A factor above 1
//! means the hand wins more than its share of the pot (position, initiative,
//! playability), below 1 less.

use crate::tree::Leaf;
use holdem::NUM_CLASSES;

/// How much money went in before the flop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotKind {
    Limped,
    Raised,
    ThreeBet,
    /// Four-bet or more.
    FourBet,
}

/// Whether the player made the last raise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Aggressor,
    Caller,
}

/// Where the player acts after the flop, among the players in the pot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Position {
    /// Last to act.
    InPosition,
    /// First to act.
    OutOfPosition,
    /// Neither (three or more players).
    Middle,
}

/// What a realization factor depends on, besides the hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Context {
    pub pot: PotKind,
    pub role: Role,
    pub position: Position,
    /// Three or more players see the flop.
    pub multiway: bool,
}

impl Context {
    /// Number of contexts ([`Context::index`] is below this).
    pub const COUNT: usize = 4 * 2 * 3 * 2;

    pub fn index(self) -> usize {
        ((self.pot as usize * 2 + self.role as usize) * 3 + self.position as usize) * 2
            + self.multiway as usize
    }

    pub fn from_index(index: usize) -> Context {
        assert!(index < Context::COUNT);
        let (rest, multiway) = (index / 2, index % 2 == 1);
        let (rest, position) = (rest / 3, rest % 3);
        let (pot, role) = (rest / 2, rest % 2);
        Context {
            pot: [PotKind::Limped, PotKind::Raised, PotKind::ThreeBet, PotKind::FourBet][pot],
            role: [Role::Aggressor, Role::Caller][role],
            position: [Position::InPosition, Position::OutOfPosition, Position::Middle][position],
            multiway,
        }
    }

    pub fn all() -> impl Iterator<Item = Context> {
        (0..Context::COUNT).map(Context::from_index)
    }

    /// The context of `seat` (which must be in the hand) at `leaf` of a
    /// `players`-handed game. Seat 0 is the button; after the flop the
    /// seats act in order from seat 1, the button last.
    pub fn of(leaf: &Leaf, seat: usize, players: usize) -> Context {
        debug_assert!(leaf.is_live(seat));
        let order = |s: usize| (s + players - 1) % players;
        let live = (0..players).filter(|&s| leaf.is_live(s));
        let (first, last) = live.fold((usize::MAX, 0), |(lo, hi), s| (lo.min(order(s)), hi.max(order(s))));
        Context {
            pot: match leaf.raises {
                0 => PotKind::Limped,
                1 => PotKind::Raised,
                2 => PotKind::ThreeBet,
                _ => PotKind::FourBet,
            },
            role: if leaf.aggressor == Some(seat as u8) { Role::Aggressor } else { Role::Caller },
            position: if order(seat) == last {
                Position::InPosition
            } else if order(seat) == first {
                Position::OutOfPosition
            } else {
                Position::Middle
            },
            multiway: leaf.num_live() > 2,
        }
    }
}

/// Realization factors per context and hand class.
#[derive(Clone, Debug, PartialEq)]
pub struct Realization {
    factors: Vec<[f32; NUM_CLASSES]>,
    /// Whether the factors already account for the rake (as factors measured
    /// from raked postflop solves do). If not, the rake on the flop pot is
    /// taken off before the factor applies.
    pub includes_rake: bool,
}

/// The measured factors shipped with the crate (see [`Realization::builtin`]).
const BUILTIN_TEXT: &str = include_str!("../data/realization_6max_100bb.txt");

impl Realization {
    /// Where [`Realization::builtin`] comes from, for chart headers.
    pub const BUILTIN_NAME: &'static str =
        "built-in, measured from heads-up postflop solves of 6-max 100bb (crates/preflop/data)";

    /// The factors measured by `examples/calibrate.rs` from heads-up postflop
    /// solves of the built-in 6-max 100bb charts. Multiway contexts copy the
    /// heads-up ones, the middle position averages in and out of position,
    /// and limped pots copy the raised pot's caller.
    pub fn builtin() -> Realization {
        Realization::parse(BUILTIN_TEXT).expect("the built-in factors file parses (tested)")
    }

    /// Every hand realizes exactly its equity.
    pub fn neutral() -> Realization {
        Realization { factors: vec![[1.0; NUM_CLASSES]; Context::COUNT], includes_rake: false }
    }

    /// One factor per position for every hand: a rough stand-in until
    /// factors are measured.
    pub fn positional(in_position: f32, out_of_position: f32, middle: f32) -> Realization {
        let mut r = Realization::neutral();
        for ctx in Context::all() {
            let f = match ctx.position {
                Position::InPosition => in_position,
                Position::OutOfPosition => out_of_position,
                Position::Middle => middle,
            };
            r.set(ctx, [f; NUM_CLASSES]);
        }
        r
    }

    #[inline]
    pub fn factor(&self, context: usize, class: usize) -> f32 {
        self.factors[context][class]
    }

    pub fn factors(&self, ctx: Context) -> &[f32; NUM_CLASSES] {
        &self.factors[ctx.index()]
    }

    /// Sets the factors of a context. Factors must be finite and not
    /// negative.
    pub fn set(&mut self, ctx: Context, factors: [f32; NUM_CLASSES]) {
        assert!(
            factors.iter().all(|f| f.is_finite() && *f >= 0.0),
            "realization factors must be finite and >= 0"
        );
        self.factors[ctx.index()] = factors;
    }
}

mod text;

#[cfg(test)]
mod tests;
