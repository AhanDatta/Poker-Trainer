//! Session frequency score: how closely the hero's action frequencies match
//! the solver's mixes over many decisions.
//!
//! No score computed decision by decision can reward mixing (its expected
//! value is linear in the hero's frequencies, so some pure strategy always
//! does at least as well), so these statistics are over the whole session:
//!
//! * **Calibration** (per spot type: street × facing nothing / a bet / a
//!   raise × position). For every action category (fold, check, call, each
//!   menu size, all-in) the expected share is the sum of the solver's
//!   probabilities for the hero's actual hands; the observed share is how
//!   often the hero chose it. Their total-variation distance is compared with
//!   what a player who samples exactly from the solver's mixes would show on
//!   the same decisions (Monte Carlo), which gives the noise level and a
//!   p-value.
//! * **Aggression bias and mixing**: each choice is placed on the solver's
//!   cumulative distribution with actions ordered passive → aggressive, at
//!   the midpoint of the chosen action's interval (`u = F(a−) + p(a)/2`).
//!   For a player who samples from the mixes `E[u] = 1/2` exactly, so the
//!   mean of `u − 1/2` measures aggression, and the spread of `u` around
//!   1/2 relative to its exact expectation measures mixing (below 1: the
//!   solver's main option too often; above 1: the rare options too often).
//!
//! Decisions can be weighted by the pot (in big blinds), so leaks in big
//! pots count more.

use crate::review::{DecisionReview, Variant};
use holdem::Street;
use std::collections::BTreeMap;

/// What the hero was facing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Facing {
    Nothing,
    Bet,
    Raise,
}

impl Facing {
    pub fn from_raises(n: u8) -> Facing {
        match n {
            0 => Facing::Nothing,
            1 => Facing::Bet,
            _ => Facing::Raise,
        }
    }
}

/// A spot type: one row of the calibration table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpotKey {
    pub street: Street,
    pub facing: Facing,
    pub in_position: bool,
}

impl SpotKey {
    /// "Flop, OOP, first to bet" style description.
    pub fn label(&self) -> String {
        let pos = if self.in_position { "IP" } else { "OOP" };
        let facing = match self.facing {
            Facing::Nothing => "no bet",
            Facing::Bet => "vs bet",
            Facing::Raise => "vs raise",
        };
        format!("{} {pos} {facing}", self.street)
    }

    /// How a category reads in this spot ("Bet 33%", "Raise 3x").
    pub fn category_label(&self, category: &str) -> String {
        match category {
            "Fold" | "Check" | "Call" | "All-in" => category.to_string(),
            size if self.facing == Facing::Nothing => format!("Bet {size}"),
            size => format!("Raise {size}"),
        }
    }
}

/// One decision as the score sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredDecision {
    pub spot: SpotKey,
    pub weight: f64,
    /// Solver mix for the hero's hand, actions ordered passive → aggressive.
    pub probs: Vec<f64>,
    /// Category of each action.
    pub categories: Vec<String>,
    pub chosen: usize,
}

impl ScoredDecision {
    /// The decision as scored, or `None` when it doesn't count: multiway
    /// pots (only heads-up pots are scored) and reviews without categories.
    pub fn from_review(d: &DecisionReview, pot_weighted: bool) -> Option<ScoredDecision> {
        let n = d.actions.len();
        if d.spot.players != 2 || d.spot.categories.len() != n || d.hero_strategy.len() != n || d.chosen >= n
        {
            return None;
        }
        let mut probs: Vec<f64> = d.hero_strategy.iter().map(|&p| (p as f64).max(0.0)).collect();
        let total: f64 = probs.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        probs.iter_mut().for_each(|p| *p /= total);
        let weight = if pot_weighted { (d.pot as f64 / crate::fmt::BB as f64).max(0.01) } else { 1.0 };
        Some(ScoredDecision {
            spot: SpotKey {
                street: d.street,
                facing: Facing::from_raises(d.spot.facing),
                in_position: d.spot.in_position,
            },
            weight,
            probs,
            categories: d.spot.categories.clone(),
            chosen: d.chosen,
        })
    }
}

/// The decisions of one hand's review that the score counts: the GTO
/// variant, or the villain's-actual-range variant when that's all there is.
pub fn hand_decisions(decisions: &[DecisionReview], pot_weighted: bool) -> Vec<ScoredDecision> {
    let has_gto = decisions.iter().any(|d| d.variant == Variant::Gto);
    decisions
        .iter()
        .filter(|d| !has_gto || d.variant == Variant::Gto)
        .filter_map(|d| ScoredDecision::from_review(d, pot_weighted))
        .collect()
}

/// One action category in a calibration row.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoryCell {
    pub category: String,
    /// Weighted share the solver's mixes give it.
    pub expected: f64,
    /// Weighted share of decisions where the hero chose it.
    pub observed: f64,
    /// `(observed − expected) / sd` for a player sampling from the mixes
    /// (infinite when the hero chose an action the solver never takes and
    /// nothing else varies).
    pub z: f64,
}

/// One spot type of the calibration table.
#[derive(Clone, Debug, PartialEq)]
pub struct SpotRow {
    pub spot: SpotKey,
    pub decisions: usize,
    pub weight: f64,
    /// Categories, passive to aggressive.
    pub cells: Vec<CategoryCell>,
    /// Total-variation distance between observed and expected shares (0-1).
    pub deviation: f64,
    /// Average deviation of a player sampling from the mixes.
    pub noise: f64,
    /// Chance that such a player deviates at least this much.
    pub p_value: f64,
}

impl SpotRow {
    /// Deviation beyond what luck explains, weighted by the row's share.
    pub fn excess(&self) -> f64 {
        (self.deviation - self.noise).max(0.0)
    }

    /// The category furthest from its expected share (by z).
    pub fn worst(&self) -> Option<&CategoryCell> {
        self.cells.iter().max_by(|a, b| a.z.abs().total_cmp(&b.z.abs()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// Weighted average of the rows' deviations.
    pub deviation: f64,
    /// Mean and 95th percentile of that average for a player sampling from
    /// the mixes.
    pub noise: f64,
    pub noise_hi: f64,
    pub p_value: f64,
}

impl Calibration {
    /// 0-100: the share of (weighted) decisions whose action frequencies
    /// match the solver's, after allowing for sampling luck.
    pub fn score(&self) -> f64 {
        (100.0 * (1.0 - (self.deviation - self.noise).max(0.0))).clamp(0.0, 100.0)
    }

    /// How much the score could move from luck alone (95%).
    pub fn uncertainty(&self) -> f64 {
        100.0 * (self.noise_hi - self.noise).max(0.0)
    }
}

/// A statistic and its z-score against a player sampling from the mixes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measure {
    pub value: f64,
    pub z: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyReport {
    pub decisions: usize,
    pub weight: f64,
    pub calibration: Calibration,
    /// Mean of `u − 1/2`: above 0 too aggressive, below 0 too passive.
    pub aggression: Measure,
    /// Observed / expected spread of `u` around 1/2: below 1 the solver's
    /// main option too often (under-mixing), above 1 its rare options too
    /// often.
    pub mixing: Measure,
    /// Rows, largest weight first.
    pub rows: Vec<SpotRow>,
}

/// Options of [`FrequencyReport::compute`].
#[derive(Clone, Copy, Debug)]
pub struct ScoreOptions {
    /// Monte Carlo samples of a player sampling from the mixes (fewer on
    /// long sessions: at most about two million draws in all, and at
    /// least 200 samples).
    pub simulations: usize,
    pub seed: u64,
}

impl Default for ScoreOptions {
    fn default() -> Self {
        ScoreOptions { simulations: 1000, seed: 0x5eed_5c0e }
    }
}

/// Midpoints `m_a = F(a−) + p_a/2` of a mix's actions.
fn midpoints(probs: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    probs
        .iter()
        .map(|&p| {
            let m = acc + p / 2.0;
            acc += p;
            m
        })
        .collect()
}

/// `(E[(u−½)²], E[(u−½)⁴])` for a player sampling from `probs`.
fn moments(probs: &[f64], mids: &[f64]) -> (f64, f64) {
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for (&p, &m) in probs.iter().zip(mids) {
        let d2 = (m - 0.5) * (m - 0.5);
        m2 += p * d2;
        m4 += p * d2 * d2;
    }
    (m2, m4)
}

/// Rank of a category for display order: fold, check/call, sizes, all-in.
fn class_rank(c: &str) -> u8 {
    match c {
        "Fold" => 0,
        "Check" | "Call" => 1,
        "All-in" => 3,
        _ => 2,
    }
}

fn z_score(diff: f64, var: f64) -> f64 {
    if var > 1e-12 {
        diff / var.sqrt()
    } else if diff.abs() < 1e-9 {
        0.0
    } else {
        f64::INFINITY.copysign(diff)
    }
}

/// Decisions of one row with their actions mapped to the row's categories.
struct RowData {
    spot: SpotKey,
    /// Per decision: weight, cumulative probabilities, category index of
    /// each action, chosen category.
    items: Vec<(f64, Vec<f64>, Vec<usize>, usize)>,
    categories: Vec<String>,
    weight: f64,
}

/// A small, fast generator for the simulations (wyrand).
struct WyRand(u64);

impl WyRand {
    #[inline]
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0xa076_1d64_78bd_642f);
        let t = (self.0 as u128).wrapping_mul((self.0 ^ 0xe703_7ed1_a0b4_28db) as u128);
        let x = ((t >> 64) as u64) ^ (t as u64);
        (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The mixed decisions, flattened: each has a weight and a run of
/// `(cumulative probability, global category)` entries.
#[derive(Default)]
struct FlatMixes {
    weights: Vec<f64>,
    /// End of each decision's run in `cum`/`cat`.
    ends: Vec<u32>,
    cum: Vec<f64>,
    cat: Vec<u32>,
}

impl FlatMixes {
    fn push(&mut self, w: f64, cum: &[f64], cats: impl Iterator<Item = u32>) {
        self.weights.push(w);
        self.cum.extend_from_slice(cum);
        self.cat.extend(cats);
        self.ends.push(self.cum.len() as u32);
    }

    /// Adds one sampled choice per decision to `counts`.
    #[inline]
    fn sample_into(&self, rng: &mut WyRand, counts: &mut [f64]) {
        let mut start = 0usize;
        for (&w, &end) in self.weights.iter().zip(&self.ends) {
            let end = end as usize;
            let cum = &self.cum[start..end];
            let x = rng.next_f64() * cum[cum.len() - 1];
            // Branch-free: the choice is how many cumulative bounds x passed
            // (a data-dependent loop mispredicts about every other draw).
            let a: usize = cum[..cum.len() - 1].iter().map(|&c| (x >= c) as usize).sum();
            counts[self.cat[start + a] as usize] += w;
            start = end;
        }
    }
}

fn tv(observed: &[f64], expected: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    0.5 * observed.iter().zip(expected).map(|(o, e)| (o - e).abs()).sum::<f64>() / total
}

impl FrequencyReport {
    /// The report for `decisions` (`None` when there are none).
    pub fn compute(decisions: &[ScoredDecision], opts: ScoreOptions) -> Option<FrequencyReport> {
        let decisions: Vec<&ScoredDecision> =
            decisions.iter().filter(|d| d.weight > 0.0 && d.chosen < d.probs.len()).collect();
        if decisions.is_empty() {
            return None;
        }
        let weight: f64 = decisions.iter().map(|d| d.weight).sum();

        // B: aggression and mixing, with exact null moments.
        let (mut bias, mut bias_var, mut spread, mut spread_exp, mut spread_var) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for d in &decisions {
            let mids = midpoints(&d.probs);
            let (v, m4) = moments(&d.probs, &mids);
            let du = mids[d.chosen] - 0.5;
            let w = d.weight;
            bias += w * du;
            bias_var += w * w * v;
            spread += w * du * du;
            spread_exp += w * v;
            spread_var += w * w * (m4 - v * v).max(0.0);
        }
        let aggression = Measure { value: bias / weight, z: z_score(bias, bias_var) };
        let mixing = if spread_exp > 1e-12 {
            Measure { value: spread / spread_exp, z: z_score(spread - spread_exp, spread_var) }
        } else {
            // Only pure spots: nothing to mix.
            Measure { value: 1.0, z: 0.0 }
        };

        // A: calibration rows.
        let mut by_spot: BTreeMap<SpotKey, Vec<&ScoredDecision>> = BTreeMap::new();
        for d in &decisions {
            by_spot.entry(d.spot).or_default().push(d);
        }
        let mut rows: Vec<RowData> = Vec::new();
        for (spot, ds) in by_spot {
            // Category order: class, then average position among the actions.
            let mut pos: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
            for d in &ds {
                for (i, c) in d.categories.iter().enumerate() {
                    let e = pos.entry(c.as_str()).or_default();
                    e.0 += i as f64;
                    e.1 += 1.0;
                }
            }
            let mut cats: Vec<(&str, f64)> = pos.into_iter().map(|(c, (s, n))| (c, s / n)).collect();
            cats.sort_by(|a, b| {
                class_rank(a.0).cmp(&class_rank(b.0)).then(a.1.total_cmp(&b.1)).then(a.0.cmp(b.0))
            });
            let categories: Vec<String> = cats.iter().map(|c| c.0.to_string()).collect();
            let index = |c: &str| categories.iter().position(|x| x == c).unwrap();
            let items = ds
                .iter()
                .map(|d| {
                    let mut cum = Vec::with_capacity(d.probs.len());
                    let mut acc = 0.0;
                    for &p in &d.probs {
                        acc += p;
                        cum.push(acc);
                    }
                    let ci: Vec<usize> = d.categories.iter().map(|c| index(c)).collect();
                    (d.weight, cum, ci.clone(), ci[d.chosen])
                })
                .collect();
            let w = ds.iter().map(|d| d.weight).sum();
            rows.push(RowData { spot, items, categories, weight: w });
        }

        // Observed and expected weights per row and category.
        let mut out_rows = Vec::with_capacity(rows.len());
        let mut observed_dev = 0.0;
        let mut expected: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
        for r in &rows {
            let k = r.categories.len();
            let (mut obs, mut exp, mut var) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
            for (w, cum, ci, chosen) in &r.items {
                obs[*chosen] += w;
                let mut prev = 0.0;
                // A category can't repeat within a decision, but be safe.
                let mut pc = vec![0.0; k];
                for (a, &c) in cum.iter().enumerate() {
                    pc[ci[a]] += c - prev;
                    prev = c;
                }
                for j in 0..k {
                    exp[j] += w * pc[j];
                    var[j] += w * w * pc[j] * (1.0 - pc[j]);
                }
            }
            let dev = tv(&obs, &exp, r.weight);
            observed_dev += r.weight * dev;
            let cells = (0..k)
                .map(|j| CategoryCell {
                    category: r.categories[j].clone(),
                    expected: exp[j] / r.weight,
                    observed: obs[j] / r.weight,
                    z: z_score(obs[j] - exp[j], var[j]),
                })
                .collect();
            out_rows.push(SpotRow {
                spot: r.spot,
                decisions: r.items.len(),
                weight: r.weight,
                cells,
                deviation: dev,
                noise: 0.0,
                p_value: 1.0,
            });
            expected.push(exp);
        }
        let observed_dev = observed_dev / weight;

        // Monte Carlo: a player sampling from the mixes on these decisions.
        // Flattened for speed: decisions with a pure mix always add the same
        // counts, so only the mixed ones are drawn.
        let offsets: Vec<usize> = rows
            .iter()
            .scan(0, |acc, r| {
                let o = *acc;
                *acc += r.categories.len();
                Some(o)
            })
            .collect();
        let num_cats = rows.iter().map(|r| r.categories.len()).sum();
        let mut base = vec![0.0; num_cats];
        let mut flat = FlatMixes::default();
        for (ri, r) in rows.iter().enumerate() {
            for (w, cum, ci, _) in &r.items {
                let total = cum.last().copied().unwrap_or(1.0);
                let mut prev = 0.0;
                let support: Vec<usize> = (0..cum.len())
                    .filter(|&a| {
                        let p = cum[a] - prev;
                        prev = cum[a];
                        p > 1e-12 * total
                    })
                    .collect();
                if support.len() <= 1 {
                    let a = support.first().copied().unwrap_or(0);
                    base[offsets[ri] + ci[a]] += w;
                } else {
                    flat.push(*w, cum, ci.iter().map(|&c| (offsets[ri] + c) as u32));
                }
            }
        }
        // Enough samples for a stable noise level without stalling on long
        // sessions (about two million draws at most).
        let mixed = flat.weights.len().max(1);
        let sims = opts.simulations.max(1).min((2_000_000 / mixed).max(200));
        let mut rng = WyRand(opts.seed);
        let mut totals = Vec::with_capacity(sims);
        let mut row_noise = vec![0.0; rows.len()];
        let mut row_hits = vec![0usize; rows.len()];
        let mut counts = vec![0.0; num_cats];
        for _ in 0..sims {
            counts.copy_from_slice(&base);
            flat.sample_into(&mut rng, &mut counts);
            let mut total = 0.0;
            for (ri, r) in rows.iter().enumerate() {
                let k = r.categories.len();
                let dev = tv(&counts[offsets[ri]..offsets[ri] + k], &expected[ri], r.weight);
                row_noise[ri] += dev;
                if dev >= out_rows[ri].deviation - 1e-12 {
                    row_hits[ri] += 1;
                }
                total += r.weight * dev;
            }
            totals.push(total / weight);
        }
        for (ri, row) in out_rows.iter_mut().enumerate() {
            row.noise = row_noise[ri] / sims as f64;
            row.p_value = (row_hits[ri] + 1) as f64 / (sims + 1) as f64;
        }
        let noise = totals.iter().sum::<f64>() / sims as f64;
        let hits = totals.iter().filter(|&&t| t >= observed_dev - 1e-12).count();
        totals.sort_by(f64::total_cmp);
        let noise_hi = totals[((0.95 * sims as f64) as usize).min(sims - 1)];
        out_rows.sort_by(|a, b| b.weight.total_cmp(&a.weight).then(a.spot.cmp(&b.spot)));

        Some(FrequencyReport {
            decisions: decisions.len(),
            weight,
            calibration: Calibration {
                deviation: observed_dev,
                noise,
                noise_hi,
                p_value: (hits + 1) as f64 / (sims + 1) as f64,
            },
            aggression,
            mixing,
            rows: out_rows,
        })
    }

    /// Rows whose deviation luck doesn't explain (p < 5%), biggest leak
    /// (weighted excess deviation) first.
    pub fn leaks(&self) -> Vec<&SpotRow> {
        let mut v: Vec<&SpotRow> = self.rows.iter().filter(|r| r.p_value < 0.05).collect();
        v.sort_by(|a, b| (b.weight * b.excess()).total_cmp(&(a.weight * a.excess())));
        v
    }
}

/// The action a random number (0-99) picks from a mix: low numbers pick
/// the most aggressive option (actions are ordered passive to aggressive),
/// e.g. with 30% bet / 70% check, 0-29 bets and 30-99 checks.
pub fn rng_action(probs: &[f32], roll: u8) -> usize {
    let x = roll as f64 + 0.5;
    let total: f64 = probs.iter().map(|&p| p.max(0.0) as f64).sum();
    let mut acc = 0.0;
    for i in (0..probs.len()).rev() {
        acc += 100.0 * probs[i].max(0.0) as f64 / total.max(1e-12);
        if x < acc {
            return i;
        }
    }
    // Rounding: the most passive action with any weight.
    (0..probs.len()).find(|&i| probs[i] > 0.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn spot(street: Street, facing: Facing, ip: bool) -> SpotKey {
        SpotKey { street, facing, in_position: ip }
    }

    fn decision(spot: SpotKey, probs: &[f64], cats: &[&str], chosen: usize, weight: f64) -> ScoredDecision {
        ScoredDecision {
            spot,
            weight,
            probs: probs.to_vec(),
            categories: cats.iter().map(|s| s.to_string()).collect(),
            chosen,
        }
    }

    fn sample(probs: &[f64], rng: &mut StdRng) -> usize {
        let x: f64 = rng.random();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if x < acc {
                return i;
            }
        }
        probs.len() - 1
    }

    fn opts() -> ScoreOptions {
        ScoreOptions { simulations: 400, seed: 7 }
    }

    /// A random spot with a random mix over its menu's categories.
    fn random_spot(rng: &mut StdRng) -> (SpotKey, Vec<f64>, Vec<&'static str>) {
        let street = [Street::Flop, Street::Turn, Street::River][rng.random_range(0..3)];
        let ip = rng.random();
        let (facing, cats): (Facing, Vec<&str>) = match rng.random_range(0..3) {
            0 => (Facing::Nothing, vec!["Check", "33%", "75%", "All-in"]),
            1 => (Facing::Bet, vec!["Fold", "Call", "3x", "All-in"]),
            _ => (Facing::Raise, vec!["Fold", "Call", "All-in"]),
        };
        // Sparse mixes, like real ones: some actions unused.
        let mut p: Vec<f64> =
            cats.iter().map(|_| if rng.random::<f64>() < 0.4 { 0.0 } else { rng.random() }).collect();
        if p.iter().sum::<f64>() == 0.0 {
            p[0] = 1.0;
        }
        let s: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= s);
        (spot(street, facing, ip), p, cats)
    }

    fn perfect_session(n: usize, seed: u64) -> Vec<ScoredDecision> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let (s, p, cats) = random_spot(&mut rng);
                let chosen = sample(&p, &mut rng);
                let w = rng.random_range(1.0..60.0);
                decision(s, &p, &cats, chosen, w)
            })
            .collect()
    }

    #[test]
    fn rng_picks_aggressive_first() {
        let mix = [0.7f32, 0.3];
        assert_eq!(rng_action(&mix, 0), 1);
        assert_eq!(rng_action(&mix, 29), 1);
        assert_eq!(rng_action(&mix, 30), 0);
        assert_eq!(rng_action(&mix, 99), 0);
        assert_eq!(rng_action(&[1.0, 0.0], 0), 0, "never an unused action");
        // Every roll maps to an action with weight; counts match the mix.
        let mix = [0.1f32, 0.0, 0.45, 0.45];
        let mut n = [0; 4];
        for r in 0..100 {
            n[rng_action(&mix, r)] += 1;
        }
        assert_eq!(n, [10, 0, 45, 45]);
    }

    #[test]
    fn midpoint_null_moments_are_exact() {
        // E[u] = 1/2 for any mix; the variance matches enumeration.
        for p in [vec![1.0], vec![0.7, 0.3], vec![0.1, 0.0, 0.6, 0.3], vec![0.25; 4]] {
            let m = midpoints(&p);
            let mean: f64 = p.iter().zip(&m).map(|(p, m)| p * m).sum();
            assert!((mean - 0.5).abs() < 1e-12, "{p:?}");
            let (v, _) = moments(&p, &m);
            let direct: f64 = p.iter().zip(&m).map(|(p, m)| p * (m - mean).powi(2)).sum();
            assert!((v - direct).abs() < 1e-12);
        }
        // Two actions: v = pq/4.
        let (v, _) = moments(&[0.7, 0.3], &midpoints(&[0.7, 0.3]));
        assert!((v - 0.7 * 0.3 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn perfect_mixer_looks_like_luck() {
        let ds = perfect_session(800, 1);
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        assert_eq!(r.decisions, 800);
        assert!(r.calibration.score() > 97.0, "{:?}", r.calibration);
        assert!(r.calibration.p_value > 0.01, "{:?}", r.calibration);
        assert!(r.aggression.z.abs() < 3.5, "{:?}", r.aggression);
        assert!((r.mixing.value - 1.0).abs() < 0.2 && r.mixing.z.abs() < 3.5, "{:?}", r.mixing);
        assert!(r.leaks().len() <= 3, "{} false leaks", r.leaks().len());
    }

    #[test]
    fn always_taking_the_main_option_is_caught() {
        // Flop IP, no bet: the solver bets 33% 30% of the time with every hand.
        let s = spot(Street::Flop, Facing::Nothing, true);
        let ds: Vec<_> = (0..120).map(|_| decision(s, &[0.7, 0.3], &["Check", "33%"], 0, 1.0)).collect();
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        let row = &r.rows[0];
        assert!((row.deviation - 0.3).abs() < 1e-9, "{row:?}");
        assert!(row.p_value < 0.01);
        assert!((r.calibration.score() - 100.0 * (1.0 - (0.3 - r.calibration.noise))).abs() < 1e-9);
        assert!(r.calibration.score() < 75.0);
        assert!(r.mixing.value < 0.5 && r.mixing.z < -3.0, "{:?}", r.mixing);
        assert!(r.aggression.z < -3.0, "checking more than the mix is passive: {:?}", r.aggression);
        let bet = &row.cells[1];
        assert_eq!(bet.category, "33%");
        assert!(bet.observed == 0.0 && (bet.expected - 0.3).abs() < 1e-9 && bet.z < -5.0);
        assert_eq!(r.leaks().len(), 1);
    }

    #[test]
    fn rare_bluffs_taken_rarely_are_fine() {
        // A 5% bluff, taken once in 20 such spots.
        let s = spot(Street::River, Facing::Nothing, false);
        let ds: Vec<_> =
            (0..20).map(|i| decision(s, &[0.95, 0.05], &["Check", "75%"], (i == 7) as usize, 1.0)).collect();
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        assert!(r.rows[0].deviation < 1e-9);
        assert!(r.calibration.score() >= 99.9);
        assert!(r.leaks().is_empty());
        // Bluffing half of them is a leak.
        let ds: Vec<_> = (0..20).map(|i| decision(s, &[0.95, 0.05], &["Check", "75%"], i % 2, 1.0)).collect();
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        assert!(r.rows[0].p_value < 0.01 && r.aggression.z > 3.0 && r.mixing.value > 2.0);
    }

    #[test]
    fn balanced_pure_play_fools_calibration_but_not_mixing() {
        // Hand A checks 70%, hand B bets 70%. Always taking each hand's main
        // option matches the row's totals exactly, but never mixes.
        let s = spot(Street::Turn, Facing::Nothing, true);
        let mut ds = Vec::new();
        for _ in 0..60 {
            ds.push(decision(s, &[0.7, 0.3], &["Check", "50%"], 0, 1.0));
            ds.push(decision(s, &[0.3, 0.7], &["Check", "50%"], 1, 1.0));
        }
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        assert!(r.rows[0].deviation < 1e-9);
        assert!(r.mixing.value < 0.5 && r.mixing.z < -3.0, "{:?}", r.mixing);
        assert!(r.aggression.z.abs() < 1e-9);
    }

    #[test]
    fn rows_split_by_spot_and_order_categories() {
        let a = spot(Street::Flop, Facing::Bet, false);
        let b = spot(Street::Flop, Facing::Nothing, false);
        let ds = vec![
            decision(a, &[0.2, 0.5, 0.2, 0.1], &["Fold", "Call", "3x", "All-in"], 1, 3.0),
            decision(a, &[0.3, 0.7], &["Fold", "Call"], 0, 3.0),
            decision(b, &[0.5, 0.25, 0.25], &["Check", "33%", "75%"], 2, 1.0),
        ];
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        assert_eq!(r.rows.len(), 2);
        assert_eq!(r.rows[0].spot, a, "heaviest row first");
        let cats: Vec<&str> = r.rows[0].cells.iter().map(|c| c.category.as_str()).collect();
        assert_eq!(cats, ["Fold", "Call", "3x", "All-in"]);
        let e: Vec<f64> = r.rows[0].cells.iter().map(|c| c.expected).collect();
        for (x, y) in e.iter().zip([0.25, 0.6, 0.1, 0.05]) {
            assert!((x - y).abs() < 1e-12, "{e:?}");
        }
        assert_eq!(a.category_label("3x"), "Raise 3x");
        assert_eq!(b.category_label("33%"), "Bet 33%");
        assert_eq!(a.label(), "Flop OOP vs bet");
    }

    #[test]
    fn off_support_choice_has_infinite_z() {
        let s = spot(Street::River, Facing::Bet, true);
        let ds = vec![decision(s, &[1.0, 0.0], &["Fold", "Call"], 1, 1.0)];
        let r = FrequencyReport::compute(&ds, opts()).unwrap();
        assert_eq!(r.rows[0].cells[1].z, f64::INFINITY);
        assert!(r.rows[0].p_value < 0.01);
        assert_eq!(r.mixing.value, 1.0, "pure spots have nothing to mix");
    }

    #[test]
    fn empty_and_deterministic() {
        assert!(FrequencyReport::compute(&[], opts()).is_none());
        let ds = perfect_session(200, 3);
        assert_eq!(FrequencyReport::compute(&ds, opts()), FrequencyReport::compute(&ds, opts()));
    }

    #[test]
    fn pot_weighting_counts_big_pots_more() {
        let s = spot(Street::Flop, Facing::Nothing, true);
        // Correct in small pots, always checking in big ones.
        let mut ds = Vec::new();
        for i in 0..40 {
            ds.push(decision(s, &[0.5, 0.5], &["Check", "33%"], i % 2, 1.0));
            ds.push(decision(s, &[0.5, 0.5], &["Check", "33%"], 0, 20.0));
        }
        let weighted = FrequencyReport::compute(&ds, opts()).unwrap();
        let flat: Vec<_> = ds
            .iter()
            .cloned()
            .map(|mut d| {
                d.weight = 1.0;
                d
            })
            .collect();
        let flat = FrequencyReport::compute(&flat, opts()).unwrap();
        assert!(weighted.rows[0].deviation > flat.rows[0].deviation + 0.1);
        assert!(weighted.aggression.value < flat.aggression.value);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// Scaling every weight changes nothing.
        #[test]
        fn weight_scale_invariant(seed in any::<u64>(), scale in 0.01f64..100.0) {
            let ds = perfect_session(60, seed);
            let scaled: Vec<_> = ds.iter().cloned().map(|mut d| { d.weight *= scale; d }).collect();
            let a = FrequencyReport::compute(&ds, opts()).unwrap();
            let b = FrequencyReport::compute(&scaled, opts()).unwrap();
            prop_assert!((a.calibration.deviation - b.calibration.deviation).abs() < 1e-9);
            prop_assert!((a.aggression.z - b.aggression.z).abs() < 1e-6);
            prop_assert!((a.mixing.value - b.mixing.value).abs() < 1e-9);
            for (x, y) in a.rows.iter().zip(&b.rows) {
                prop_assert!((x.deviation - y.deviation).abs() < 1e-9);
            }
        }

        /// Shares add up, deviations and p-values stay in range, and every
        /// decision lands in exactly one row.
        #[test]
        fn report_invariants(seed in any::<u64>(), n in 1usize..150) {
            let mut rng = StdRng::seed_from_u64(seed);
            // Any choices at all, not only samples from the mixes.
            let ds: Vec<_> = perfect_session(n, seed)
                .into_iter()
                .map(|mut d| { if rng.random::<f64>() < 0.5 { d.chosen = rng.random_range(0..d.probs.len()); } d })
                .collect();
            let r = FrequencyReport::compute(&ds, opts()).unwrap();
            prop_assert_eq!(r.rows.iter().map(|x| x.decisions).sum::<usize>(), n);
            for row in &r.rows {
                let e: f64 = row.cells.iter().map(|c| c.expected).sum();
                let o: f64 = row.cells.iter().map(|c| c.observed).sum();
                prop_assert!((e - 1.0).abs() < 1e-9 && (o - 1.0).abs() < 1e-9);
                prop_assert!((0.0..=1.0 + 1e-12).contains(&row.deviation));
                prop_assert!(row.p_value > 0.0 && row.p_value <= 1.0);
            }
            let c = &r.calibration;
            prop_assert!(c.noise <= c.noise_hi + 1e-12);
            prop_assert!((0.0..=100.0).contains(&c.score()));
            prop_assert!(r.aggression.value.abs() <= 0.5);
            prop_assert!(r.mixing.value >= 0.0 && r.mixing.value.is_finite());
        }
    }

    /// Over many perfect sessions the aggression z is roughly standard
    /// normal (its null moments are exact).
    #[test]
    fn aggression_z_is_calibrated() {
        let zs: Vec<f64> = (0..200)
            .map(|s| FrequencyReport::compute(&perfect_session(40, 100 + s), opts()).unwrap().aggression.z)
            .collect();
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let var = zs.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / zs.len() as f64;
        assert!(mean.abs() < 0.25, "mean {mean}");
        assert!((0.75..1.3).contains(&var), "var {var}");
        let over = zs.iter().filter(|z| z.abs() > 1.96).count();
        assert!(over < 25, "{over} of 200 beyond 1.96");
    }
}
