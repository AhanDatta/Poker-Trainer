//! Stratified sampling of chance cards for the fast (approximate) mode.

use holdem::Card;
use rand::Rng;

/// Picks `k` of the `available` cards so that every suit is represented in
/// proportion to how many of its cards remain (largest-remainder rounding)
/// and, within a suit, ranks are spread evenly with a random offset.
pub(crate) fn sample_cards<R: Rng>(available: &[Card], k: usize, rng: &mut R) -> Vec<Card> {
    if k >= available.len() {
        return available.to_vec();
    }
    let mut by_suit: [Vec<Card>; 4] = Default::default();
    for &c in available {
        by_suit[c.suit() as usize].push(c);
    }
    for v in &mut by_suit {
        v.sort_unstable_by_key(|c| c.rank());
    }
    let total = available.len() as f64;
    let mut quotas = [0usize; 4];
    let mut fracs: Vec<(f64, f64, usize)> = Vec::with_capacity(4);
    for s in 0..4 {
        let exact = k as f64 * by_suit[s].len() as f64 / total;
        quotas[s] = exact.floor() as usize;
        fracs.push((exact - exact.floor(), rng.random::<f64>(), s));
    }
    let mut assigned: usize = quotas.iter().sum();
    fracs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for &(_, _, s) in &fracs {
        if assigned >= k {
            break;
        }
        if quotas[s] < by_suit[s].len() {
            quotas[s] += 1;
            assigned += 1;
        }
    }
    let mut out = Vec::with_capacity(k);
    for s in 0..4 {
        let (m, q) = (by_suit[s].len(), quotas[s]);
        if q == 0 {
            continue;
        }
        let step = m as f64 / q as f64;
        let offset = rng.random::<f64>() * step;
        for j in 0..q {
            let idx = ((offset + j as f64 * step).floor() as usize).min(m - 1);
            out.push(by_suit[s][idx]);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn stratified_by_suit() {
        let mut rng = StdRng::seed_from_u64(1);
        // Flop with two spades: 11 spades remain out of 49.
        let board = holdem::card::parse_cards("AsKs7d").unwrap();
        let available: Vec<Card> = Card::all().filter(|c| !board.contains(c)).collect();
        let picks = sample_cards(&available, 12, &mut rng);
        assert_eq!(picks.len(), 12);
        let spades = picks.iter().filter(|c| c.suit() == 3).count();
        assert!((2..=3).contains(&spades), "{spades}");
        for p in &picks {
            assert!(available.contains(p));
        }
    }
}
