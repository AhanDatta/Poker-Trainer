//! Hole-card combinations (1326) and hand classes (169).

use crate::card::{parse_rank, Card, ParseError, RANK_CHARS};
use std::fmt;
use std::str::FromStr;

/// Number of distinct two-card hole-card combinations.
pub const NUM_COMBOS: usize = 1326;
/// Number of strategically distinct preflop hand classes (pairs, suited, offsuit).
pub const NUM_CLASSES: usize = 169;

/// Index of the combination `{a, b}` in `0..1326`.
#[inline]
pub const fn combo_index(a: Card, b: Card) -> usize {
    let (lo, hi) = if a.index() < b.index() { (a.index(), b.index()) } else { (b.index(), a.index()) };
    hi * (hi - 1) / 2 + lo
}

const fn build_combos() -> [(Card, Card); NUM_COMBOS] {
    let mut out = [(Card::from_index(0), Card::from_index(0)); NUM_COMBOS];
    let mut hi = 1;
    while hi < 52 {
        let mut lo = 0;
        while lo < hi {
            out[hi * (hi - 1) / 2 + lo] = (Card::from_index(hi as u8), Card::from_index(lo as u8));
            lo += 1;
        }
        hi += 1;
    }
    out
}

/// All combinations; each entry is `(higher-index card, lower-index card)`,
/// i.e. the first card has the higher (or equal) rank.
pub static COMBOS: [(Card, Card); NUM_COMBOS] = build_combos();

/// The two cards of a combination, higher-ranked card first.
#[inline]
pub fn combo_cards(index: usize) -> (Card, Card) {
    COMBOS[index]
}

/// Bit mask (see [`crate::CardSet`]) of a combination's two cards.
#[inline]
pub fn combo_mask(index: usize) -> u64 {
    let (a, b) = COMBOS[index];
    a.bit() | b.bit()
}

/// Formats a combination such as `AsKd`.
pub fn combo_to_string(index: usize) -> String {
    let (a, b) = COMBOS[index];
    format!("{a}{b}")
}

/// A preflop hand class such as `AKs`, `AKo` or `QQ`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct HandClass {
    /// Higher rank (0 = 2 … 12 = A).
    pub hi: u8,
    /// Lower rank (equal to `hi` for pairs).
    pub lo: u8,
    /// Suited flag (always false for pairs).
    pub suited: bool,
}

impl HandClass {
    pub fn new(r1: u8, r2: u8, suited: bool) -> HandClass {
        let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
        HandClass { hi, lo, suited: suited && hi != lo }
    }

    pub fn is_pair(self) -> bool {
        self.hi == self.lo
    }

    /// Class of a concrete combination.
    pub fn of_cards(a: Card, b: Card) -> HandClass {
        HandClass::new(a.rank(), b.rank(), a.suit() == b.suit())
    }

    pub fn of_combo(index: usize) -> HandClass {
        let (a, b) = COMBOS[index];
        HandClass::of_cards(a, b)
    }

    /// Position in the usual 13x13 grid (row-major, `A` first). Suited hands
    /// are above the diagonal, offsuit hands below.
    pub fn grid_index(self) -> usize {
        let (r_hi, r_lo) = (12 - self.hi as usize, 12 - self.lo as usize);
        if self.suited {
            r_hi * 13 + r_lo
        } else {
            r_lo * 13 + r_hi
        }
    }

    pub fn from_grid_index(index: usize) -> HandClass {
        let (row, col) = (index / 13, index % 13);
        let (r1, r2) = (12 - row as u8, 12 - col as u8);
        if row == col {
            HandClass::new(r1, r2, false)
        } else if row < col {
            HandClass::new(r1, r2, true)
        } else {
            HandClass::new(r1, r2, false)
        }
    }

    /// All 169 classes in grid order.
    pub fn all() -> impl Iterator<Item = HandClass> {
        (0..NUM_CLASSES).map(HandClass::from_grid_index)
    }

    /// Number of combinations in this class (6, 4 or 12).
    pub fn num_combos(self) -> usize {
        if self.is_pair() {
            6
        } else if self.suited {
            4
        } else {
            12
        }
    }

    /// Combination indices belonging to this class.
    pub fn combos(self) -> Vec<usize> {
        let mut out = Vec::with_capacity(12);
        for s1 in 0..4u8 {
            for s2 in 0..4u8 {
                let a = Card::new(self.hi, s1);
                let b = Card::new(self.lo, s2);
                if a == b {
                    continue;
                }
                let ok = if self.is_pair() {
                    s1 < s2
                } else if self.suited {
                    s1 == s2
                } else {
                    s1 != s2
                };
                if ok {
                    out.push(combo_index(a, b));
                }
            }
        }
        out
    }
}

impl fmt::Display for HandClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (h, l) = (RANK_CHARS[self.hi as usize], RANK_CHARS[self.lo as usize]);
        if self.is_pair() {
            write!(f, "{h}{l}")
        } else if self.suited {
            write!(f, "{h}{l}s")
        } else {
            write!(f, "{h}{l}o")
        }
    }
}

impl FromStr for HandClass {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chars: Vec<char> = s.trim().chars().collect();
        let err = || ParseError(format!("invalid hand class: {s:?}"));
        if chars.len() < 2 || chars.len() > 3 {
            return Err(err());
        }
        let r1 = parse_rank(chars[0]).ok_or_else(err)?;
        let r2 = parse_rank(chars[1]).ok_or_else(err)?;
        let suited = match chars.get(2).map(|c| c.to_ascii_lowercase()) {
            None if r1 == r2 => false,
            Some('s') if r1 != r2 => true,
            Some('o') if r1 != r2 => false,
            _ => return Err(err()),
        };
        Ok(HandClass::new(r1, r2, suited))
    }
}

const fn build_combo_class() -> [u8; NUM_COMBOS] {
    let mut out = [0u8; NUM_COMBOS];
    let mut i = 0;
    while i < NUM_COMBOS {
        let (a, b) = COMBOS[i];
        let (hi, lo) = (a.rank() as usize, b.rank() as usize); // a has the higher index => rank >= b's
        let suited = a.suit() == b.suit();
        let (r_hi, r_lo) = (12 - hi, 12 - lo);
        out[i] = if hi == lo {
            (r_hi * 13 + r_hi) as u8
        } else if suited {
            (r_hi * 13 + r_lo) as u8
        } else {
            (r_lo * 13 + r_hi) as u8
        };
        i += 1;
    }
    out
}

/// Grid index (see [`HandClass::grid_index`]) of every combination.
pub static COMBO_CLASS: [u8; NUM_COMBOS] = build_combo_class();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_indexing_is_bijective() {
        let mut seen = vec![false; NUM_COMBOS];
        for a in Card::all() {
            for b in Card::all() {
                if a == b {
                    continue;
                }
                let i = combo_index(a, b);
                assert_eq!(i, combo_index(b, a));
                let (x, y) = combo_cards(i);
                assert!((x == a && y == b) || (x == b && y == a));
                assert!(x.rank() >= y.rank());
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn classes() {
        let total: usize = HandClass::all().map(|c| c.num_combos()).sum();
        assert_eq!(total, NUM_COMBOS);
        for c in HandClass::all() {
            assert_eq!(HandClass::from_grid_index(c.grid_index()), c);
            let combos = c.combos();
            assert_eq!(combos.len(), c.num_combos());
            for i in combos {
                assert_eq!(HandClass::of_combo(i), c);
                assert_eq!(COMBO_CLASS[i] as usize, c.grid_index());
            }
            assert_eq!(c.to_string().parse::<HandClass>().unwrap(), c);
        }
        assert_eq!("AKs".parse::<HandClass>().unwrap().grid_index(), 1);
        assert_eq!("AKo".parse::<HandClass>().unwrap().grid_index(), 13);
        assert_eq!("22".parse::<HandClass>().unwrap().grid_index(), 168);
    }
}
