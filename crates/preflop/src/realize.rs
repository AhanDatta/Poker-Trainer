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

impl Realization {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{GameSpec, MAX_PLAYERS};
    use crate::tree::{LeafKind, Tree};

    #[test]
    fn context_index_round_trips() {
        let mut seen = [false; Context::COUNT];
        for ctx in Context::all() {
            assert_eq!(Context::from_index(ctx.index()), ctx);
            assert!(!std::mem::replace(&mut seen[ctx.index()], true));
        }
        assert!(seen.iter().all(|&s| s));
    }

    fn leaf(live: &[usize], raises: u8, aggressor: Option<u8>) -> Leaf {
        Leaf {
            kind: LeafKind::Flop,
            pot: 0,
            rake: 0,
            commit: [0; MAX_PLAYERS],
            live: live.iter().fold(0, |m, &s| m | 1 << s),
            raises,
            aggressor,
        }
    }

    #[test]
    fn contexts_of_typical_pots() {
        // 6-max: BTN (seat 0) opens, BB (seat 2) calls.
        let l = leaf(&[0, 2], 1, Some(0));
        let btn = Context::of(&l, 0, 6);
        let bb = Context::of(&l, 2, 6);
        assert_eq!(
            btn,
            Context {
                pot: PotKind::Raised,
                role: Role::Aggressor,
                position: Position::InPosition,
                multiway: false
            }
        );
        assert_eq!(
            bb,
            Context {
                pot: PotKind::Raised,
                role: Role::Caller,
                position: Position::OutOfPosition,
                multiway: false
            }
        );
        // SB (seat 1) 3-bets a CO (seat 5) open, BB and CO call.
        let l = leaf(&[1, 2, 5], 2, Some(1));
        assert_eq!(Context::of(&l, 1, 6).position, Position::OutOfPosition);
        assert_eq!(Context::of(&l, 2, 6).position, Position::Middle);
        assert_eq!(Context::of(&l, 5, 6).position, Position::InPosition);
        assert!(Context::of(&l, 5, 6).multiway);
        assert_eq!(Context::of(&l, 1, 6).role, Role::Aggressor);
        assert_eq!(Context::of(&l, 5, 6).pot, PotKind::ThreeBet);
        // Heads-up the small blind has the button and acts last.
        let l = leaf(&[0, 1], 0, None);
        assert_eq!(Context::of(&l, 0, 2).position, Position::InPosition);
        assert_eq!(Context::of(&l, 1, 2).position, Position::OutOfPosition);
        assert_eq!(Context::of(&l, 1, 2).pot, PotKind::Limped);
    }

    /// In every flop leaf of a real tree exactly one player is in position
    /// and one out of position, and at most one is the aggressor.
    #[test]
    fn every_flop_leaf_has_one_player_in_and_one_out_of_position() {
        let mut spec = GameSpec::new(6);
        spec.rules.max_cold_callers = 1;
        let tree = Tree::build(&spec).unwrap();
        let mut flops = 0;
        for l in tree.leaves().filter(|l| l.kind == LeafKind::Flop) {
            flops += 1;
            let ctxs: Vec<Context> = (0..6).filter(|&s| l.is_live(s)).map(|s| Context::of(l, s, 6)).collect();
            let count = |p: Position| ctxs.iter().filter(|c| c.position == p).count();
            assert_eq!((count(Position::InPosition), count(Position::OutOfPosition)), (1, 1));
            assert_eq!(count(Position::Middle), ctxs.len() - 2);
            assert!(ctxs.iter().filter(|c| c.role == Role::Aggressor).count() <= 1);
            assert!(ctxs.iter().all(|c| c.multiway == (ctxs.len() > 2)));
        }
        assert!(flops > 100);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn negative_factors_are_rejected() {
        Realization::neutral().set(Context::from_index(0), [-0.1; NUM_CLASSES]);
    }
}
