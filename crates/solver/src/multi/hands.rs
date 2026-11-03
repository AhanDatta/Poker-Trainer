//! Hand lists and per-board strength tables of the three players.

use super::{N, NONE};
use holdem::combo::{combo_cards, NUM_COMBOS};
use holdem::eval::EvalMask;
use holdem::{Card, CardSet};

/// Hand lists of the three players.
pub(super) struct Hands {
    pub combos: [Vec<u16>; N],
    pub cards: [Vec<(u8, u8)>; N],
    pub weights: [Vec<f32>; N],
    pub lookup: [Vec<u32>; N],
    pub blocked: [Vec<Vec<u32>>; N],
    /// `same[p][q][i]`: index in `q`'s list of `p`'s hand `i` (or NONE).
    pub same: [[Vec<u32>; N]; N],
}

impl Hands {
    pub fn new(ranges: &[Vec<f32>; N], board: CardSet, prune: f32, keep: &[Vec<usize>; N]) -> Hands {
        let mut combos: [Vec<u16>; N] = Default::default();
        let mut cards: [Vec<(u8, u8)>; N] = Default::default();
        let mut weights: [Vec<f32>; N] = Default::default();
        for p in 0..N {
            let max_w = ranges[p].iter().copied().fold(0f32, f32::max);
            for c in 0..NUM_COMBOS {
                let (a, b) = combo_cards(c);
                if board.contains(a) || board.contains(b) {
                    continue;
                }
                let w = ranges[p][c];
                if w > 0.0 && (w >= prune * max_w || keep[p].contains(&c)) {
                    combos[p].push(c as u16);
                    cards[p].push((a.index() as u8, b.index() as u8));
                    weights[p].push(w);
                }
            }
        }
        let mut lookup = [vec![NONE; NUM_COMBOS], vec![NONE; NUM_COMBOS], vec![NONE; NUM_COMBOS]];
        let mut blocked: [Vec<Vec<u32>>; N] = Default::default();
        for p in 0..N {
            blocked[p] = vec![Vec::new(); 52];
            for (i, &c) in combos[p].iter().enumerate() {
                lookup[p][c as usize] = i as u32;
                let (a, b) = cards[p][i];
                blocked[p][a as usize].push(i as u32);
                blocked[p][b as usize].push(i as u32);
            }
        }
        let mut same: [[Vec<u32>; N]; N] = Default::default();
        for p in 0..N {
            for q in 0..N {
                same[p][q] = combos[p].iter().map(|&c| lookup[q][c as usize]).collect();
            }
        }
        Hands { combos, cards, weights, lookup, blocked, same }
    }

    pub fn len(&self, p: usize) -> usize {
        self.combos[p].len()
    }

    /// For each hand of `p`: `q`'s total reach over compatible hands.
    pub fn compat(&self, p: usize, q: usize, r: &[f32]) -> Vec<f64> {
        let mut total = 0f64;
        let mut minus = [0f64; 52];
        for (&(a, b), &w) in self.cards[q].iter().zip(r) {
            if w != 0.0 {
                let w = w as f64;
                total += w;
                minus[a as usize] += w;
                minus[b as usize] += w;
            }
        }
        (0..self.len(p))
            .map(|i| {
                let (a, b) = self.cards[p][i];
                let s = match self.same[p][q][i] {
                    NONE => 0.0,
                    j => r[j as usize] as f64,
                };
                (total - minus[a as usize] - minus[b as usize] + s).max(0.0)
            })
            .collect()
    }
}

/// Hands of the three players sorted by strength on one final board.
pub(super) struct Table {
    pub lists: [Vec<(u32, u32, u8, u8)>; N],
}

impl Table {
    pub fn new(hands: &Hands, board: &[Card]) -> Table {
        let mask = EvalMask::from_cards(board);
        let set = CardSet::from_cards(board);
        let mut lists: [Vec<(u32, u32, u8, u8)>; N] = Default::default();
        for p in 0..N {
            let mut v: Vec<(u32, u32, u8, u8)> = hands.cards[p]
                .iter()
                .enumerate()
                .filter_map(|(i, &(a, b))| {
                    let (ca, cb) = (Card::from_index(a), Card::from_index(b));
                    (!set.contains(ca) && !set.contains(cb))
                        .then(|| (mask.add(ca).add(cb).evaluate(), i as u32, a, b))
                })
                .collect();
            v.sort_unstable();
            lists[p] = v;
        }
        Table { lists }
    }

    /// For each hand of `p`: `q`'s compatible reach over weaker hands plus
    /// half of the equal ones.
    pub fn beat(&self, hands: &Hands, p: usize, q: usize, r: &[f32]) -> Vec<f64> {
        let mut out = vec![0f64; hands.len(p)];
        let (mine, theirs) = (&self.lists[p], &self.lists[q]);
        let mut sum = 0f64;
        let mut minus = [0f64; 52];
        let (mut i, mut j) = (0, 0);
        while i < mine.len() {
            let s = mine[i].0;
            while j < theirs.len() && theirs[j].0 < s {
                let (_, idx, a, b) = theirs[j];
                let w = r[idx as usize] as f64;
                if w != 0.0 {
                    sum += w;
                    minus[a as usize] += w;
                    minus[b as usize] += w;
                }
                j += 1;
            }
            let mut eq = 0f64;
            let mut eq_minus = [0f64; 52];
            let mut jj = j;
            while jj < theirs.len() && theirs[jj].0 == s {
                let (_, idx, a, b) = theirs[jj];
                let w = r[idx as usize] as f64;
                eq += w;
                eq_minus[a as usize] += w;
                eq_minus[b as usize] += w;
                jj += 1;
            }
            while i < mine.len() && mine[i].0 == s {
                let (_, h, a, b) = mine[i];
                let same = match hands.same[p][q][h as usize] {
                    NONE => 0.0,
                    k => r[k as usize] as f64,
                };
                let weaker = sum - minus[a as usize] - minus[b as usize];
                let equal = eq - eq_minus[a as usize] - eq_minus[b as usize] + same;
                out[h as usize] = (weaker + 0.5 * equal).max(0.0);
                i += 1;
            }
        }
        out
    }
}
