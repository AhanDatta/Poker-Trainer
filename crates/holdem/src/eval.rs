//! Fast hand evaluator for 5, 6 or 7 cards.
//!
//! Cards are accumulated into an [`EvalMask`]: a `u64` holding four 16-bit
//! lanes, one per suit, with bit `r` of lane `s` set when the card of rank `r`
//! and suit `s` is present. From the four per-suit rank sets we derive the
//! "ranks present at least 1/2/3/4 times" sets with a handful of bitwise ops,
//! which is enough to classify any hand without lookup tables.
//!
//! The resulting [`HandValue`] is a `u32` where a larger value is a stronger
//! hand: `category << 26 | tie-break payload`.

use crate::card::Card;
use std::fmt;

/// Comparable strength of a hand (higher is better).
pub type HandValue = u32;

const CATEGORY_SHIFT: u32 = 26;

/// Hand categories in increasing order of strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandCategory {
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    Trips = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    Quads = 7,
    StraightFlush = 8,
}

impl HandCategory {
    pub fn of(value: HandValue) -> HandCategory {
        match value >> CATEGORY_SHIFT {
            0 => HandCategory::HighCard,
            1 => HandCategory::Pair,
            2 => HandCategory::TwoPair,
            3 => HandCategory::Trips,
            4 => HandCategory::Straight,
            5 => HandCategory::Flush,
            6 => HandCategory::FullHouse,
            7 => HandCategory::Quads,
            _ => HandCategory::StraightFlush,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            HandCategory::HighCard => "high card",
            HandCategory::Pair => "pair",
            HandCategory::TwoPair => "two pair",
            HandCategory::Trips => "three of a kind",
            HandCategory::Straight => "straight",
            HandCategory::Flush => "flush",
            HandCategory::FullHouse => "full house",
            HandCategory::Quads => "four of a kind",
            HandCategory::StraightFlush => "straight flush",
        }
    }
}

impl fmt::Display for HandCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Cards in the evaluator's per-suit lane layout.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct EvalMask(pub u64);

impl EvalMask {
    #[inline]
    pub fn card_bit(card: Card) -> u64 {
        1u64 << (card.suit() as u32 * 16 + card.rank() as u32)
    }

    #[inline]
    pub fn from_cards(cards: &[Card]) -> EvalMask {
        EvalMask(cards.iter().fold(0, |m, &c| m | Self::card_bit(c)))
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, card: Card) -> EvalMask {
        EvalMask(self.0 | Self::card_bit(card))
    }

    #[inline]
    pub fn evaluate(self) -> HandValue {
        evaluate_mask(self.0)
    }
}

/// Evaluates a slice of 5 to 7 cards.
#[inline]
pub fn evaluate_cards(cards: &[Card]) -> HandValue {
    EvalMask::from_cards(cards).evaluate()
}

#[inline]
fn high_bit(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// Keeps the `n` most significant set bits of `x`.
#[inline]
fn top_bits(mut x: u32, n: u32) -> u32 {
    let mut out = 0;
    for _ in 0..n {
        if x == 0 {
            break;
        }
        let b = 1 << high_bit(x);
        out |= b;
        x ^= b;
    }
    out
}

/// Returns the rank of the highest card of a straight contained in `ranks`.
#[inline]
fn straight_high(ranks: u32) -> Option<u32> {
    const WHEEL: u32 = 0b1_0000_0000_1111; // A 5 4 3 2
    let run = ranks & (ranks << 1) & (ranks << 2) & (ranks << 3) & (ranks << 4);
    if run != 0 {
        Some(high_bit(run))
    } else if ranks & WHEEL == WHEEL {
        Some(3) // five-high
    } else {
        None
    }
}

#[inline]
fn value(category: HandCategory, payload: u32) -> HandValue {
    ((category as u32) << CATEGORY_SHIFT) | payload
}

/// Evaluates cards given in [`EvalMask`] layout.
#[inline]
pub fn evaluate_mask(mask: u64) -> HandValue {
    let s0 = (mask & 0x1FFF) as u32;
    let s1 = ((mask >> 16) & 0x1FFF) as u32;
    let s2 = ((mask >> 32) & 0x1FFF) as u32;
    let s3 = ((mask >> 48) & 0x1FFF) as u32;

    // With at most 7 cards a flush can't coexist with quads or a full house,
    // so a flush (or straight flush) can be returned immediately.
    for suited in [s0, s1, s2, s3] {
        if suited.count_ones() >= 5 {
            return match straight_high(suited) {
                Some(h) => value(HandCategory::StraightFlush, h),
                None => value(HandCategory::Flush, top_bits(suited, 5)),
            };
        }
    }

    let any = s0 | s1 | s2 | s3;
    let two = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);
    let three = (s0 & s1 & s2) | (s0 & s1 & s3) | (s0 & s2 & s3) | (s1 & s2 & s3);
    let four = s0 & s1 & s2 & s3;

    if four != 0 {
        let q = high_bit(four);
        let kicker = top_bits(any & !(1 << q), 1);
        return value(HandCategory::Quads, (q << 13) | kicker);
    }
    if three != 0 {
        let t = high_bit(three);
        let rest = two & !(1 << t);
        if rest != 0 {
            let p = high_bit(rest);
            return value(HandCategory::FullHouse, (t << 13) | (1 << p));
        }
    }
    if let Some(h) = straight_high(any) {
        return value(HandCategory::Straight, h);
    }
    if three != 0 {
        let t = high_bit(three);
        let kickers = top_bits(any & !(1 << t), 2);
        return value(HandCategory::Trips, (t << 13) | kickers);
    }
    if two.count_ones() >= 2 {
        let pairs = top_bits(two, 2);
        let kicker = top_bits(any & !pairs, 1);
        return value(HandCategory::TwoPair, (pairs << 13) | kicker);
    }
    if two != 0 {
        let p = high_bit(two);
        let kickers = top_bits(any & !(1 << p), 3);
        return value(HandCategory::Pair, (p << 13) | kickers);
    }
    value(HandCategory::HighCard, top_bits(any, 5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn ev(s: &str) -> HandValue {
        evaluate_cards(&parse_cards(s).unwrap())
    }

    #[test]
    fn categories() {
        assert_eq!(HandCategory::of(ev("AsKsQsJsTs2d3c")), HandCategory::StraightFlush);
        assert_eq!(HandCategory::of(ev("As2s3s4s5s9d9c")), HandCategory::StraightFlush);
        assert_eq!(HandCategory::of(ev("AsAhAdAc5s9d9c")), HandCategory::Quads);
        assert_eq!(HandCategory::of(ev("AsAhAd9c5s9d2c")), HandCategory::FullHouse);
        assert_eq!(HandCategory::of(ev("AsAhAd9c9s9d2c")), HandCategory::FullHouse);
        assert_eq!(HandCategory::of(ev("As9s5s3s2sKdKc")), HandCategory::Flush);
        assert_eq!(HandCategory::of(ev("As2d3h4c5s9d9c")), HandCategory::Straight);
        assert_eq!(HandCategory::of(ev("7s7d7h4c5sKdJc")), HandCategory::Trips);
        assert_eq!(HandCategory::of(ev("7s7d4h4c5sKdJc")), HandCategory::TwoPair);
        assert_eq!(HandCategory::of(ev("7s7d4h3c5sKdJc")), HandCategory::Pair);
        assert_eq!(HandCategory::of(ev("7s8d4h3c2sKdJc")), HandCategory::HighCard);
    }

    #[test]
    fn ordering() {
        // Straights: wheel < six-high < broadway.
        assert!(ev("As2d3h4c5s") < ev("6s2d3h4c5s"));
        assert!(ev("6s2d3h4c5s") < ev("AsKdQhJcTs"));
        // Kickers.
        assert!(ev("AsAd9h4c2s") < ev("AsAdTh4c2s"));
        assert!(ev("AsAd9h4c2s") > ev("KsKdQh4c2s"));
        // Two pair: higher top pair wins, then lower pair, then kicker.
        assert!(ev("AsAd2h2c3s") > ev("KsKdQhQc3s"));
        assert!(ev("AsAd3h3c2s") > ev("AsAd2h2cKs"));
        assert!(ev("AsAd3h3c5s") > ev("AsAd3h3c4s"));
        // Full house: trips first.
        assert!(ev("3s3d3hAcAs") > ev("2s2d2hAcAs"));
        assert!(ev("3s3d3hAcAs") > ev("3s3d3hKcKs"));
        // Board plays: equal.
        assert_eq!(ev("AsKdQhJcTs2d3c"), ev("AsKdQhJcTs4d5c"));
        // Best five of seven are used: second trips counts as the pair.
        assert_eq!(ev("AsAhAd9c9s9d2c"), ev("AsAhAd9c9s2d3c"));
    }

    /// Enumerates all 2,598,960 five-card hands and checks the category counts.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn five_card_category_counts() {
        let mut counts = [0u32; 9];
        let cards: Vec<Card> = Card::all().collect();
        for a in 0..52 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        let m4 = EvalMask::from_cards(&[cards[a], cards[b], cards[c], cards[d]]);
                        for e in d + 1..52 {
                            let v = m4.add(cards[e]).evaluate();
                            counts[HandCategory::of(v) as usize] += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(counts, [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]);
    }

    /// Cross-checks 7-card evaluation against brute force "best 5 of 7".
    #[test]
    fn seven_card_matches_best_five() {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(7);
        let mut deck: Vec<Card> = Card::all().collect();
        for _ in 0..20_000 {
            deck.shuffle(&mut rng);
            let seven = &deck[..7];
            let mut best = 0;
            for skip1 in 0..7 {
                for skip2 in skip1 + 1..7 {
                    let five: Vec<Card> =
                        (0..7).filter(|&i| i != skip1 && i != skip2).map(|i| seven[i]).collect();
                    best = best.max(evaluate_cards(&five));
                }
            }
            assert_eq!(evaluate_cards(seven), best, "{seven:?}");
        }
    }
}
