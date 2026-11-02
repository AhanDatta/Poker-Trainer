//! Weighted hole-card ranges.
//!
//! A [`Range`] stores one weight per hole-card combination (1326 entries).
//! Weights are usually in `0..=1` and represent the fraction of the time the
//! combination is in the range.
//!
//! ## Notation
//! Comma-separated tokens, each optionally followed by `:weight`:
//! * `AA`, `AKs`, `AKo`, `AK` (suited + offsuit)
//! * `TT+` (TT through AA), `A2s+` (A2s through AKs), `KTo+`, `A5+`
//! * `99-55`, `A9s-A6s`, `KTo-K7o`
//! * concrete combinations such as `AsKs`
//! * `any` / `random` for all 1326 combinations

use crate::card::{parse_cards, parse_rank, CardSet, ParseError};
use crate::combo::{combo_index, combo_mask, HandClass, COMBO_CLASS, NUM_CLASSES, NUM_COMBOS};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, PartialEq)]
pub struct Range {
    weights: Vec<f32>,
}

impl Default for Range {
    fn default() -> Self {
        Range::empty()
    }
}

impl Range {
    pub fn empty() -> Range {
        Range { weights: vec![0.0; NUM_COMBOS] }
    }

    pub fn full() -> Range {
        Range { weights: vec![1.0; NUM_COMBOS] }
    }

    pub fn from_weights(weights: Vec<f32>) -> Range {
        assert_eq!(weights.len(), NUM_COMBOS);
        Range { weights }
    }

    pub fn parse(s: &str) -> Result<Range, ParseError> {
        s.parse()
    }

    #[inline]
    pub fn weight(&self, combo: usize) -> f32 {
        self.weights[combo]
    }

    #[inline]
    pub fn set(&mut self, combo: usize, w: f32) {
        self.weights[combo] = w;
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    pub fn into_weights(self) -> Vec<f32> {
        self.weights
    }

    pub fn set_class(&mut self, class: HandClass, w: f32) {
        for c in class.combos() {
            self.weights[c] = w;
        }
    }

    /// Sum of weights (i.e. the number of combinations in the range).
    pub fn num_combos(&self) -> f32 {
        self.weights.iter().sum()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.iter().all(|&w| w <= 0.0)
    }

    /// Zeroes every combination that shares a card with `dead`.
    pub fn remove_dead(&mut self, dead: CardSet) {
        for (i, w) in self.weights.iter_mut().enumerate() {
            if combo_mask(i) & dead.0 != 0 {
                *w = 0.0;
            }
        }
    }

    pub fn without_dead(&self, dead: CardSet) -> Range {
        let mut r = self.clone();
        r.remove_dead(dead);
        r
    }

    /// Element-wise product with another weight vector.
    pub fn multiply(&mut self, factors: &[f32]) {
        for (w, f) in self.weights.iter_mut().zip(factors) {
            *w *= f;
        }
    }

    pub fn scale(&mut self, factor: f32) {
        for w in &mut self.weights {
            *w *= factor;
        }
    }

    /// Average weight of each of the 169 classes, in grid order.
    pub fn class_weights(&self) -> [f32; NUM_CLASSES] {
        let mut sum = [0f32; NUM_CLASSES];
        let mut cnt = [0f32; NUM_CLASSES];
        for (i, &w) in self.weights.iter().enumerate() {
            let c = COMBO_CLASS[i] as usize;
            sum[c] += w;
            cnt[c] += 1.0;
        }
        let mut out = [0f32; NUM_CLASSES];
        for i in 0..NUM_CLASSES {
            out[i] = sum[i] / cnt[i];
        }
        out
    }

    /// A per-class listing such as `AA, AKs:0.5, ...` (debug / display aid).
    pub fn to_class_string(&self) -> String {
        let cw = self.class_weights();
        let mut parts = Vec::new();
        for (i, &w) in cw.iter().enumerate() {
            if w > 0.0 {
                let class = HandClass::from_grid_index(i);
                if (w - 1.0).abs() < 1e-4 {
                    parts.push(class.to_string());
                } else {
                    parts.push(format!("{class}:{w:.3}"));
                }
            }
        }
        parts.join(", ")
    }
}

impl fmt::Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Range({:.1} combos: {})", self.num_combos(), self.to_class_string())
    }
}

/// A class pattern: a specific class, or "both suited and offsuit" when
/// `suited` is `None` (and the ranks differ).
#[derive(Clone, Copy)]
struct Pattern {
    hi: u8,
    lo: u8,
    suited: Option<bool>,
}

impl Pattern {
    fn parse(s: &str) -> Result<Pattern, ParseError> {
        let err = || ParseError(format!("invalid hand pattern: {s:?}"));
        let chars: Vec<char> = s.chars().collect();
        if chars.len() < 2 || chars.len() > 3 {
            return Err(err());
        }
        let r1 = parse_rank(chars[0]).ok_or_else(err)?;
        let r2 = parse_rank(chars[1]).ok_or_else(err)?;
        let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
        let suited = match chars.get(2).map(|c| c.to_ascii_lowercase()) {
            None => None,
            Some('s') if hi != lo => Some(true),
            Some('o') if hi != lo => Some(false),
            _ => return Err(err()),
        };
        Ok(Pattern { hi, lo, suited })
    }

    fn classes(self) -> Vec<HandClass> {
        if self.hi == self.lo {
            return vec![HandClass::new(self.hi, self.lo, false)];
        }
        match self.suited {
            Some(s) => vec![HandClass::new(self.hi, self.lo, s)],
            None => vec![HandClass::new(self.hi, self.lo, true), HandClass::new(self.hi, self.lo, false)],
        }
    }

    fn with_lo(self, lo: u8) -> Pattern {
        Pattern { lo, ..self }
    }

    fn pair(rank: u8) -> Pattern {
        Pattern { hi: rank, lo: rank, suited: None }
    }
}

fn expand_token(tok: &str) -> Result<Vec<usize>, ParseError> {
    let lower = tok.to_ascii_lowercase();
    if lower == "any" || lower == "random" || lower == "all" {
        return Ok((0..NUM_COMBOS).collect());
    }
    // Concrete combination, e.g. "AsKs".
    if tok.len() == 4 {
        if let Ok(cards) = parse_cards(tok) {
            if cards[0] == cards[1] {
                return Err(ParseError(format!("duplicate card in {tok:?}")));
            }
            return Ok(vec![combo_index(cards[0], cards[1])]);
        }
    }
    let mut patterns = Vec::new();
    if let Some((a, b)) = tok.split_once('-') {
        let (pa, pb) = (Pattern::parse(a.trim())?, Pattern::parse(b.trim())?);
        if pa.hi == pa.lo && pb.hi == pb.lo {
            let (lo, hi) = (pa.hi.min(pb.hi), pa.hi.max(pb.hi));
            patterns.extend((lo..=hi).map(Pattern::pair));
        } else if pa.hi == pb.hi && pa.suited == pb.suited && pa.hi != pa.lo && pb.hi != pb.lo {
            let (lo, hi) = (pa.lo.min(pb.lo), pa.lo.max(pb.lo));
            patterns.extend((lo..=hi).map(|l| pa.with_lo(l)));
        } else {
            return Err(ParseError(format!("invalid span: {tok:?}")));
        }
    } else if let Some(base) = tok.strip_suffix('+') {
        let p = Pattern::parse(base.trim())?;
        if p.hi == p.lo {
            patterns.extend((p.hi..=12).map(Pattern::pair));
        } else {
            patterns.extend((p.lo..p.hi).map(|l| p.with_lo(l)));
        }
    } else {
        patterns.push(Pattern::parse(tok)?);
    }
    Ok(patterns.into_iter().flat_map(|p| p.classes()).flat_map(|c| c.combos()).collect())
}

impl FromStr for Range {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut range = Range::empty();
        for raw in s.split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let (tok, weight) = match raw.split_once(':') {
                Some((t, w)) => {
                    let w: f32 =
                        w.trim().parse().map_err(|_| ParseError(format!("invalid weight in {raw:?}")))?;
                    (t.trim(), w)
                }
                None => (raw, 1.0),
            };
            for c in expand_token(tok)? {
                range.weights[c] = weight;
            }
        }
        Ok(range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn count(s: &str) -> f32 {
        Range::parse(s).unwrap().num_combos()
    }

    #[test]
    fn notation() {
        assert_eq!(count("AA"), 6.0);
        assert_eq!(count("AKs"), 4.0);
        assert_eq!(count("AKo"), 12.0);
        assert_eq!(count("AK"), 16.0);
        assert_eq!(count("22+"), 78.0);
        assert_eq!(count("TT+"), 30.0);
        assert_eq!(count("A2s+"), 48.0);
        assert_eq!(count("KTo+"), 36.0);
        assert_eq!(count("99-55"), 30.0);
        assert_eq!(count("A9s-A6s"), 16.0);
        assert_eq!(count("AsKs"), 1.0);
        assert_eq!(count("any"), 1326.0);
        assert_eq!(count("AA, KK:0.5"), 9.0);
        assert_eq!(count("A5+"), 16.0 * 9.0);
        assert!(Range::parse("AKx").is_err());
        assert!(Range::parse("AA-KQs").is_err());
    }

    #[test]
    fn dead_cards() {
        let mut r = Range::parse("AA").unwrap();
        r.remove_dead(CardSet::from_cards(&parse_cards("As").unwrap()));
        assert_eq!(r.num_combos(), 3.0);
    }

    #[test]
    fn class_weights() {
        let r = Range::parse("AKs:0.5, QQ").unwrap();
        let cw = r.class_weights();
        assert_eq!(cw["AKs".parse::<HandClass>().unwrap().grid_index()], 0.5);
        assert_eq!(cw["QQ".parse::<HandClass>().unwrap().grid_index()], 1.0);
        assert_eq!(cw["AKo".parse::<HandClass>().unwrap().grid_index()], 0.0);
    }
}
