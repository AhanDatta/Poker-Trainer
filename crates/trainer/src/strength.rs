//! Preflop hand strength (all-in equity against a random hand) for the 169
//! hand classes, used to order hands when AI styles widen or tighten ranges.

use holdem::card::Card;
use holdem::combo::{HandClass, NUM_CLASSES};
use holdem::eval::EvalMask;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::OnceLock;

static EQUITY: OnceLock<[f32; NUM_CLASSES]> = OnceLock::new();
static RANK: OnceLock<[f32; NUM_CLASSES]> = OnceLock::new();

fn compute_equities() -> [f32; NUM_CLASSES] {
    const SAMPLES: usize = 6000;
    let mut out = [0f32; NUM_CLASSES];
    for (g, slot) in out.iter_mut().enumerate() {
        let class = HandClass::from_grid_index(g);
        let combo = class.combos()[0];
        let (a, b) = holdem::combo_cards(combo);
        let mut rng = StdRng::seed_from_u64(1000 + g as u64);
        let mut won = 0f64;
        for _ in 0..SAMPLES {
            let mut used = a.bit() | b.bit();
            let mut draw = |rng: &mut StdRng| loop {
                let c = Card::from_index(rng.random_range(0..52u8));
                if used & c.bit() == 0 {
                    used |= c.bit();
                    return c;
                }
            };
            let (o1, o2) = (draw(&mut rng), draw(&mut rng));
            let mut board = EvalMask::default();
            for _ in 0..5 {
                board = board.add(draw(&mut rng));
            }
            let mine = board.add(a).add(b).evaluate();
            let theirs = board.add(o1).add(o2).evaluate();
            won += if mine > theirs {
                1.0
            } else if mine == theirs {
                0.5
            } else {
                0.0
            };
        }
        *slot = (won / SAMPLES as f64) as f32;
    }
    out
}

/// All-in equity of each class (grid order) against a random hand.
pub fn class_equity() -> &'static [f32; NUM_CLASSES] {
    EQUITY.get_or_init(compute_equities)
}

/// Percentile rank of each class by equity, weighted by combo counts:
/// 1.0 for the best class, close to 0 for the worst.
pub fn class_rank() -> &'static [f32; NUM_CLASSES] {
    RANK.get_or_init(|| {
        let eq = class_equity();
        let mut order: Vec<usize> = (0..NUM_CLASSES).collect();
        order.sort_by(|&x, &y| eq[x].partial_cmp(&eq[y]).unwrap());
        let mut rank = [0f32; NUM_CLASSES];
        let mut cum = 0f32;
        for &g in &order {
            let n = HandClass::from_grid_index(g).num_combos() as f32;
            cum += n;
            rank[g] = cum / 1326.0;
        }
        rank
    })
}

/// Classes ordered from strongest to weakest.
pub fn classes_by_strength() -> Vec<usize> {
    let eq = class_equity();
    let mut order: Vec<usize> = (0..NUM_CLASSES).collect();
    order.sort_by(|&x, &y| eq[y].partial_cmp(&eq[x]).unwrap());
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq(s: &str) -> f32 {
        class_equity()[s.parse::<HandClass>().unwrap().grid_index()]
    }

    #[test]
    fn known_equities() {
        assert!((eq("AA") - 0.85).abs() < 0.02);
        assert!((eq("72o") - 0.35).abs() < 0.02);
        assert!(eq("AKs") > eq("AKo"));
        assert!(eq("KK") > eq("AKs"));
        let order = classes_by_strength();
        assert_eq!(HandClass::from_grid_index(order[0]).to_string(), "AA");
        assert!(class_rank()["AA".parse::<HandClass>().unwrap().grid_index()] > 0.99);
    }
}
