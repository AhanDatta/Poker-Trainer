//! Preflop hand strength (all-in equity against a random hand) for the 169
//! hand classes, used to order hands when AI styles widen or tighten ranges.

use holdem::card::Card;
use holdem::combo::{HandClass, NUM_CLASSES};
use holdem::eval::EvalMask;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::OnceLock;

static EQUITY: OnceLock<[f32; NUM_CLASSES]> = OnceLock::new();
static SCORE: OnceLock<[f32; NUM_CLASSES]> = OnceLock::new();
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

/// Equity of each class against a hand from the strongest 15% of hands.
fn equity_vs_strong() -> [f32; NUM_CLASSES] {
    const SAMPLES: usize = 3000;
    let eq = class_equity();
    let mut order: Vec<usize> = (0..NUM_CLASSES).collect();
    order.sort_by(|&x, &y| eq[y].partial_cmp(&eq[x]).unwrap());
    let mut strong: Vec<(Card, Card)> = Vec::new();
    for &g in &order {
        if strong.len() as f32 >= 0.15 * 1326.0 {
            break;
        }
        strong.extend(HandClass::from_grid_index(g).combos().into_iter().map(holdem::combo_cards));
    }
    let mut out = [0f32; NUM_CLASSES];
    for (g, slot) in out.iter_mut().enumerate() {
        let (a, b) = holdem::combo_cards(HandClass::from_grid_index(g).combos()[0]);
        let mut rng = StdRng::seed_from_u64(5000 + g as u64);
        let mut won = 0f64;
        let mut n = 0usize;
        while n < SAMPLES {
            let (o1, o2) = strong[rng.random_range(0..strong.len())];
            let mut used = a.bit() | b.bit();
            if used & (o1.bit() | o2.bit()) != 0 {
                continue;
            }
            used |= o1.bit() | o2.bit();
            let mut board = EvalMask::default();
            let mut k = 0;
            while k < 5 {
                let c = Card::from_index(rng.random_range(0..52u8));
                if used & c.bit() == 0 {
                    used |= c.bit();
                    board = board.add(c);
                    k += 1;
                }
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
            n += 1;
        }
        *slot = (won / SAMPLES as f64) as f32;
    }
    out
}

/// Preflop playability score: a blend of equity against a random hand and
/// against strong hands, plus small bonuses for suited and connected hands
/// (which realize their equity better than raw equity suggests).
pub fn class_score() -> &'static [f32; NUM_CLASSES] {
    SCORE.get_or_init(|| {
        let eq = class_equity();
        let strong = equity_vs_strong();
        let mut out = [0f32; NUM_CLASSES];
        for g in 0..NUM_CLASSES {
            let c = HandClass::from_grid_index(g);
            let gap = c.hi - c.lo;
            let bonus = if c.suited { 0.02 } else { 0.0 }
                + if !c.is_pair() && gap <= 2 { 0.012 / gap as f32 } else { 0.0 };
            out[g] = 0.4 * eq[g] + 0.6 * strong[g] + bonus;
        }
        out
    })
}

/// Percentile rank of each class by [`class_score`], weighted by combo
/// counts: 1.0 for the best class, close to 0 for the worst.
pub fn class_rank() -> &'static [f32; NUM_CLASSES] {
    RANK.get_or_init(|| {
        let eq = class_score();
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

/// Classes ordered from strongest to weakest (by [`class_score`]).
pub fn classes_by_strength() -> Vec<usize> {
    let eq = class_score();
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
        // Playability: suited connectors outrank weak offsuit kings.
        let r = |s: &str| class_rank()[s.parse::<HandClass>().unwrap().grid_index()];
        assert!(r("76s") > r("K2o"), "{} {}", r("76s"), r("K2o"));
        assert!(r("AKs") > r("JJ") || r("AKs") > 0.97);
    }
}
