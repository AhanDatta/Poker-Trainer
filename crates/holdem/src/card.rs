//! Cards, card sets and the deck.

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

pub const NUM_CARDS: usize = 52;
pub const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

/// Error produced when parsing cards, ranges or bet sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// A playing card. Encoded as `rank * 4 + suit` where rank `0 = 2 … 12 = A`
/// and suit `0 = c, 1 = d, 2 = h, 3 = s`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Card(u8);

impl TryFrom<String> for Card {
    type Error = ParseError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<Card> for String {
    fn from(c: Card) -> String {
        c.to_string()
    }
}

impl Card {
    #[inline]
    pub const fn new(rank: u8, suit: u8) -> Card {
        Card(rank * 4 + suit)
    }

    /// Builds a card from its index in `0..52`.
    #[inline]
    pub const fn from_index(index: u8) -> Card {
        Card(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 2
    }

    #[inline]
    pub const fn suit(self) -> u8 {
        self.0 & 3
    }

    /// Bit of this card in a [`CardSet`].
    #[inline]
    pub const fn bit(self) -> u64 {
        1u64 << self.0
    }

    pub fn rank_char(self) -> char {
        RANK_CHARS[self.rank() as usize]
    }

    pub fn suit_char(self) -> char {
        SUIT_CHARS[self.suit() as usize]
    }

    /// Iterator over all 52 cards.
    pub fn all() -> impl Iterator<Item = Card> {
        (0..NUM_CARDS as u8).map(Card)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank_char(), self.suit_char())
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

pub fn parse_rank(c: char) -> Option<u8> {
    let c = c.to_ascii_uppercase();
    RANK_CHARS.iter().position(|&r| r == c).map(|r| r as u8)
}

pub fn parse_suit(c: char) -> Option<u8> {
    let c = c.to_ascii_lowercase();
    SUIT_CHARS.iter().position(|&s| s == c).map(|s| s as u8)
}

impl FromStr for Card {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut chars = s.trim().chars();
        let (Some(r), Some(su), None) = (chars.next(), chars.next(), chars.next()) else {
            return Err(ParseError(format!("invalid card: {s:?}")));
        };
        match (parse_rank(r), parse_suit(su)) {
            (Some(r), Some(su)) => Ok(Card::new(r, su)),
            _ => Err(ParseError(format!("invalid card: {s:?}"))),
        }
    }
}

/// Parses a list of cards such as `"AhKd7c"` or `"Ah Kd 7c"`.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, ParseError> {
    let compact: String = s.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    if !compact.len().is_multiple_of(2) {
        return Err(ParseError(format!("invalid card list: {s:?}")));
    }
    let chars: Vec<char> = compact.chars().collect();
    chars.chunks(2).map(|pair| pair.iter().collect::<String>().parse()).collect()
}

/// Formats a list of cards without separators, e.g. `AhKd7c`.
pub fn cards_to_string(cards: &[Card]) -> String {
    cards.iter().map(|c| c.to_string()).collect()
}

/// A set of cards stored as a 52-bit mask (bit `i` = card with index `i`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CardSet(pub u64);

impl CardSet {
    pub const EMPTY: CardSet = CardSet(0);

    pub fn from_cards(cards: &[Card]) -> CardSet {
        CardSet(cards.iter().fold(0, |m, c| m | c.bit()))
    }

    #[inline]
    pub fn contains(self, card: Card) -> bool {
        self.0 & card.bit() != 0
    }

    #[inline]
    pub fn insert(&mut self, card: Card) {
        self.0 |= card.bit();
    }

    #[inline]
    pub fn remove(&mut self, card: Card) {
        self.0 &= !card.bit();
    }

    #[inline]
    pub fn with(self, card: Card) -> CardSet {
        CardSet(self.0 | card.bit())
    }

    #[inline]
    pub fn intersects(self, other: CardSet) -> bool {
        self.0 & other.0 != 0
    }

    #[inline]
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Card> {
        let mut m = self.0;
        std::iter::from_fn(move || {
            if m == 0 {
                None
            } else {
                let i = m.trailing_zeros() as u8;
                m &= m - 1;
                Some(Card(i))
            }
        })
    }
}

impl fmt::Debug for CardSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cards: Vec<Card> = self.iter().collect();
        write!(f, "{{{}}}", cards_to_string(&cards))
    }
}

/// A shuffled deck from which cards are dealt.
#[derive(Clone, Debug)]
pub struct Deck {
    cards: Vec<Card>,
}

impl Deck {
    /// A full deck shuffled with `rng`.
    pub fn shuffled<R: Rng + ?Sized>(rng: &mut R) -> Deck {
        let mut cards: Vec<Card> = Card::all().collect();
        cards.shuffle(rng);
        Deck { cards }
    }

    /// A shuffled deck that excludes `dead` cards.
    pub fn shuffled_without<R: Rng + ?Sized>(rng: &mut R, dead: CardSet) -> Deck {
        let mut cards: Vec<Card> = Card::all().filter(|c| !dead.contains(*c)).collect();
        cards.shuffle(rng);
        Deck { cards }
    }

    pub fn deal(&mut self) -> Card {
        self.cards.pop().expect("deck is empty")
    }

    pub fn deal_n(&mut self, n: usize) -> Vec<Card> {
        (0..n).map(|_| self.deal()).collect()
    }

    pub fn remaining(&self) -> usize {
        self.cards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_roundtrip() {
        for c in Card::all() {
            let s = c.to_string();
            assert_eq!(s.parse::<Card>().unwrap(), c);
        }
        assert_eq!("As".parse::<Card>().unwrap(), Card::new(12, 3));
        assert_eq!("2c".parse::<Card>().unwrap().index(), 0);
        assert!("1s".parse::<Card>().is_err());
        assert!("Ax".parse::<Card>().is_err());
    }

    #[test]
    fn parse_card_lists() {
        let cards = parse_cards("Ah Kd 7c").unwrap();
        assert_eq!(cards_to_string(&cards), "AhKd7c");
        assert_eq!(parse_cards("AhKd7c").unwrap(), cards);
        assert!(parse_cards("AhK").is_err());
    }

    #[test]
    fn card_set_ops() {
        let cards = parse_cards("AhKd7c").unwrap();
        let set = CardSet::from_cards(&cards);
        assert_eq!(set.len(), 3);
        for c in &cards {
            assert!(set.contains(*c));
        }
        assert_eq!(set.iter().count(), 3);
    }
}
