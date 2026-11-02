//! Per-player hand lists and showdown strength tables.

use crate::eval::Entry;
use holdem::combo::{combo_cards, NUM_COMBOS};
use holdem::eval::EvalMask;
use holdem::{Card, CardSet};

pub(crate) const NONE: u32 = u32::MAX;

/// The private hands each player can hold in the solved game.
///
/// Hands are combos with positive weight that don't conflict with the root
/// board; they're indexed `0..len` per player in increasing combo order.
#[derive(Clone, Debug)]
pub struct HandSet {
    /// Combo index (`0..1326`) of every hand.
    pub combos: [Vec<u16>; 2],
    /// Card indices of every hand.
    pub cards: [Vec<(u8, u8)>; 2],
    /// Initial (root) weights.
    pub weights: [Vec<f32>; 2],
    /// Index of the identical combo in the opponent's list (or `NONE`).
    pub(crate) same: [Vec<u32>; 2],
    /// For every card, the hands that contain it.
    pub(crate) blocked: [Vec<Vec<u32>>; 2],
    /// Combo index -> hand index (or `NONE`).
    pub(crate) lookup: [Vec<u32>; 2],
    /// Total weight of compatible (non-overlapping) hand pairs.
    pub num_combinations: f64,
}

impl HandSet {
    /// Builds the hand lists. Combos whose weight is below `prune` times the
    /// player's maximum weight are dropped unless listed in `keep`.
    pub fn new(ranges: &[Vec<f32>; 2], board: CardSet, prune: f32, keep: &[Vec<usize>; 2]) -> HandSet {
        let mut combos: [Vec<u16>; 2] = Default::default();
        let mut cards: [Vec<(u8, u8)>; 2] = Default::default();
        let mut weights: [Vec<f32>; 2] = Default::default();
        for p in 0..2 {
            let max_w = ranges[p].iter().copied().fold(0f32, f32::max);
            let cutoff = max_w * prune;
            for c in 0..NUM_COMBOS {
                let (a, b) = combo_cards(c);
                if board.contains(a) || board.contains(b) {
                    continue;
                }
                let w = ranges[p][c];
                let forced = keep[p].contains(&c);
                if (w > 0.0 && w >= cutoff) || (forced && w > 0.0) {
                    combos[p].push(c as u16);
                    cards[p].push((a.index() as u8, b.index() as u8));
                    weights[p].push(w);
                }
            }
        }
        let mut lookup = [vec![NONE; NUM_COMBOS], vec![NONE; NUM_COMBOS]];
        for p in 0..2 {
            for (i, &c) in combos[p].iter().enumerate() {
                lookup[p][c as usize] = i as u32;
            }
        }
        let same = [
            combos[0].iter().map(|&c| lookup[1][c as usize]).collect(),
            combos[1].iter().map(|&c| lookup[0][c as usize]).collect(),
        ];
        let mut blocked: [Vec<Vec<u32>>; 2] = [vec![Vec::new(); 52], vec![Vec::new(); 52]];
        for p in 0..2 {
            for (i, &(a, b)) in cards[p].iter().enumerate() {
                blocked[p][a as usize].push(i as u32);
                blocked[p][b as usize].push(i as u32);
            }
        }
        let mut hs = HandSet { combos, cards, weights, same, blocked, lookup, num_combinations: 0.0 };
        hs.num_combinations = hs.compatible_weight();
        hs
    }

    pub fn len(&self, p: usize) -> usize {
        self.combos[p].len()
    }

    pub fn is_empty(&self, p: usize) -> bool {
        self.combos[p].is_empty()
    }

    /// Hand index of a combo for player `p`.
    pub fn index_of(&self, p: usize, combo: usize) -> Option<usize> {
        match self.lookup[p][combo] {
            NONE => None,
            i => Some(i as usize),
        }
    }

    /// `sum_i sum_o w0(i) w1(o) [i, o compatible]`.
    fn compatible_weight(&self) -> f64 {
        let compat = self.compatible_reach(0, &self.weights[1]);
        compat.iter().zip(&self.weights[0]).map(|(&c, &w)| c * w as f64).sum()
    }

    /// For each hand of `p`, the total opponent reach over non-conflicting
    /// opponent hands.
    pub(crate) fn compatible_reach(&self, p: usize, opp_reach: &[f32]) -> Vec<f64> {
        let o = p ^ 1;
        let mut total = 0f64;
        let mut minus = [0f64; 52];
        for (k, &w) in opp_reach.iter().enumerate() {
            if w != 0.0 {
                let (a, b) = self.cards[o][k];
                let w = w as f64;
                total += w;
                minus[a as usize] += w;
                minus[b as usize] += w;
            }
        }
        (0..self.len(p))
            .map(|i| {
                let (a, b) = self.cards[p][i];
                let s = match self.same[p][i] {
                    NONE => 0.0,
                    j => opp_reach[j as usize] as f64,
                };
                (total - minus[a as usize] - minus[b as usize] + s).max(0.0)
            })
            .collect()
    }
}

/// Hands of both players sorted by showdown strength on one final board.
/// Only hands that don't conflict with the board are listed.
#[derive(Clone, Debug, Default)]
pub(crate) struct StrengthTable {
    /// Entries sorted by strength ascending.
    pub lists: [Vec<Entry>; 2],
}

impl StrengthTable {
    pub fn new(hands: &HandSet, board: &[Card]) -> StrengthTable {
        let mask = EvalMask::from_cards(board);
        let board_set = CardSet::from_cards(board);
        let mut lists: [Vec<Entry>; 2] = Default::default();
        for p in 0..2 {
            let mut v: Vec<Entry> = hands.cards[p]
                .iter()
                .enumerate()
                .filter_map(|(i, &(a, b))| {
                    let (ca, cb) = (Card::from_index(a), Card::from_index(b));
                    if board_set.contains(ca) || board_set.contains(cb) {
                        None
                    } else {
                        Some(Entry { strength: mask.add(ca).add(cb).evaluate(), idx: i as u16, c1: a, c2: b })
                    }
                })
                .collect();
            v.sort_unstable();
            lists[p] = v;
        }
        StrengthTable { lists }
    }
}
