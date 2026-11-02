//! Suit isomorphism.
//!
//! Two suits are interchangeable at a chance node when the board holds the
//! same ranks in both and both players' ranges are invariant under swapping
//! them. Then the subtree after a card of one suit is the mirror image of the
//! subtree after the same rank in the other suit, so only one of them (the
//! *canonical* card, lowest suit of its class) is built and solved; the
//! other's values are the canonical values with hand indices permuted.

use crate::hands::HandSet;
use holdem::combo::combo_index;
use holdem::Card;

/// The six unordered suit pairs.
pub(crate) const PAIRS: [(u8, u8); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

pub(crate) fn pair_index(a: u8, b: u8) -> usize {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    PAIRS.iter().position(|&x| x == (a, b)).expect("distinct suits")
}

/// `card` with suits `a` and `b` exchanged.
pub fn swap_card(card: Card, a: u8, b: u8) -> Card {
    let s = card.suit();
    let s2 = if s == a {
        b
    } else if s == b {
        a
    } else {
        s
    };
    Card::new(card.rank(), s2)
}

/// Rank mask of each suit on a board.
fn suit_ranks(board: &[Card]) -> [u16; 4] {
    let mut m = [0u16; 4];
    for c in board {
        m[c.suit() as usize] |= 1 << c.rank();
    }
    m
}

/// Suit pairs that the board alone doesn't distinguish.
pub fn board_symmetric_pairs(board: &[Card]) -> Vec<(u8, u8)> {
    let m = suit_ranks(board);
    PAIRS.iter().copied().filter(|&(a, b)| m[a as usize] == m[b as usize]).collect()
}

/// All combos obtained from `combo` by board-symmetric suit swaps
/// (including `combo` itself).
pub fn iso_orbit(board: &[Card], combo: usize) -> Vec<usize> {
    let pairs = board_symmetric_pairs(board);
    let mut orbit = vec![combo];
    let mut i = 0;
    while i < orbit.len() {
        let (x, y) = holdem::combo_cards(orbit[i]);
        for &(a, b) in &pairs {
            let c = combo_index(swap_card(x, a, b), swap_card(y, a, b));
            if !orbit.contains(&c) {
                orbit.push(c);
            }
        }
        i += 1;
    }
    orbit
}

/// One isomorphic chance outcome: `card` is solved via `canonical`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IsoRef {
    pub card: Card,
    pub canonical: Card,
    pub pair: u8,
}

pub(crate) struct Isomorphism {
    pub enabled: bool,
    /// Whether both ranges are symmetric under each suit swap.
    pub range_symmetric: [bool; 6],
    /// Hand permutation for each player and suit pair (empty if invalid).
    pub perm: [[Vec<u16>; 6]; 2],
}

impl Isomorphism {
    pub fn new(hands: &HandSet, enabled: bool) -> Isomorphism {
        let mut range_symmetric = [false; 6];
        let mut perm: [[Vec<u16>; 6]; 2] = Default::default();
        if enabled {
            'pairs: for (pi, &(a, b)) in PAIRS.iter().enumerate() {
                let mut pp: [Vec<u16>; 2] = Default::default();
                for p in 0..2 {
                    let w = &hands.weights[p];
                    let max_w = w.iter().copied().fold(0f32, f32::max).max(1e-30);
                    let mut v = Vec::with_capacity(hands.len(p));
                    for (h, &(c1, c2)) in hands.cards[p].iter().enumerate() {
                        let s1 = swap_card(Card::from_index(c1), a, b);
                        let s2 = swap_card(Card::from_index(c2), a, b);
                        let Some(j) = hands.index_of(p, combo_index(s1, s2)) else { continue 'pairs };
                        if (w[h] - w[j]).abs() > 1e-4 * max_w {
                            continue 'pairs;
                        }
                        v.push(j as u16);
                    }
                    pp[p] = v;
                }
                range_symmetric[pi] = true;
                let [p0, p1] = pp;
                perm[0][pi] = p0;
                perm[1][pi] = p1;
            }
        }
        Isomorphism { enabled, range_symmetric, perm }
    }

    /// Splits the cards that can be dealt on `board` into canonical cards
    /// and isomorphic references to them.
    pub fn chance_cards(&self, board: &[Card], available: &[Card]) -> (Vec<Card>, Vec<IsoRef>) {
        if !self.enabled || !self.range_symmetric.iter().any(|&s| s) {
            return (available.to_vec(), Vec::new());
        }
        let m = suit_ranks(board);
        let valid = |a: u8, b: u8| {
            let pi = pair_index(a, b);
            self.range_symmetric[pi] && m[a as usize] == m[b as usize]
        };
        let mut canonical = Vec::new();
        let mut refs = Vec::new();
        for &c in available {
            let s = c.suit();
            match (0..s).find(|&s0| valid(s0, s)) {
                Some(s0) => refs.push(IsoRef {
                    card: c,
                    canonical: Card::new(c.rank(), s0),
                    pair: pair_index(s0, s) as u8,
                }),
                None => canonical.push(c),
            }
        }
        (canonical, refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use holdem::card::parse_cards;

    #[test]
    fn symmetric_pairs() {
        let b = parse_cards("Ks7s2d").unwrap();
        // Clubs (0) and hearts (2) are both empty.
        assert_eq!(board_symmetric_pairs(&b), vec![(0, 2)]);
        let b = parse_cards("Ks7d2c").unwrap();
        assert!(board_symmetric_pairs(&b).is_empty());
        let b = parse_cards("KsQs2s").unwrap();
        assert_eq!(board_symmetric_pairs(&b).len(), 3);
    }

    #[test]
    fn orbit() {
        let b = parse_cards("KsQs2s").unwrap();
        let c = parse_cards("AhKh").unwrap();
        let orbit = iso_orbit(&b, combo_index(c[0], c[1]));
        assert_eq!(orbit.len(), 3); // AcKc, AdKd, AhKh
    }
}
