//! How good a solve is: what each seat earns with the average strategy,
//! and how much a player could gain by changing one decision.

use super::{sample, Flat, Solver};
use crate::deal::Deal;
use crate::rng::Rng;
use std::collections::HashMap;

/// Measurements of the average strategy on fresh deals.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnosis {
    pub deals: u64,
    /// What each seat nets per hand, in big blinds, when everyone plays the
    /// average strategy.
    pub ev: Vec<f64>,
    /// Per seat, the big blinds per hand gained by switching, at every
    /// decision and hand class where one action is clearly better than the
    /// average strategy's mix, to that action. "Clearly" is judged on half
    /// of the deals and the gain measured on the other half, so luck
    /// doesn't count as gain. At an equilibrium no single change gains
    /// anything and this is zero; a solve that hasn't converged shows what
    /// is still on the table (at least: unclear improvements aren't
    /// counted).
    pub deviation_gain: Vec<f64>,
}

/// A hand class must reach a decision this often on the choosing half of
/// the deals before an action can count as clearly better there.
const MIN_SAMPLES: u32 = 30;

/// How many standard errors above zero an action's advantage must be on
/// the choosing half.
const SIGMAS: f64 = 3.0;

/// What one `(decision, hand class)` collected on each half of the deals.
#[derive(Clone, Copy, Default)]
struct Samples {
    seat: u8,
    num_actions: u8,
    count: [u32; 2],
    /// Sum, and sum of squares, of each action's advantage over the
    /// average strategy's own mix (both valued on the same deal).
    gain: [[f64; 4]; 2],
    gain_squared: [[f64; 4]; 2],
}

impl Samples {
    /// The action whose advantage on half `half` is clearly positive (more
    /// than `SIGMAS` standard errors above zero), if any; the largest such.
    fn clear_improvement(&self, half: usize) -> Option<usize> {
        let n = self.count[half] as f64;
        if self.count[half] < MIN_SAMPLES {
            return None;
        }
        let lower_bound = |a: usize| {
            let mean = self.gain[half][a] / n;
            let variance = (self.gain_squared[half][a] / n - mean * mean).max(0.0);
            mean - SIGMAS * (variance / n).sqrt()
        };
        (0..self.num_actions as usize)
            .map(|a| (a, lower_bound(a)))
            .filter(|&(_, bound)| bound > 0.0)
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(a, _)| a)
    }
}

impl Solver {
    /// Plays `deals` hands with the average strategy. At every decision on
    /// the way each action is also tried once, the rest of the hand played
    /// out by the average strategy, to value the alternatives.
    pub fn diagnose(&self, deals: u64, seed: u64) -> Diagnosis {
        let players = self.tree.spec.players;
        let mut rng = Rng::stream(seed, u64::MAX);
        let mut ev = vec![0f64; players];
        let mut samples: HashMap<usize, Samples> = HashMap::new();
        for t in 0..deals {
            let half = (t % 2) as usize;
            let deal = Deal::sample(players, &mut rng);
            let mut node = 0u32;
            loop {
                match &self.nodes[node as usize] {
                    Flat::Leaf { live, prize, commit, context } => {
                        for seat in 0..players {
                            ev[seat] +=
                                self.payoff(*live, *prize, commit[seat], context[seat], seat, &deal) as f64;
                        }
                        break;
                    }
                    &Flat::Decision { seat, num_actions, has_fold, fold_value, kids, cells } => {
                        let n = num_actions as usize;
                        let base = cells + deal.class[seat as usize] as usize * n;
                        let mut sigma = [0f32; 4];
                        self.average_into(base, &mut sigma[..n]);
                        let mut value = [0f32; 4];
                        let mut mix = 0.0;
                        for a in 0..n {
                            value[a] = if a == 0 && has_fold {
                                fold_value
                            } else {
                                self.rollout(kids[a], seat as usize, &deal, &mut rng)
                            };
                            mix += sigma[a] * value[a];
                        }
                        let entry = samples.entry(base).or_insert(Samples {
                            seat,
                            num_actions,
                            ..Samples::default()
                        });
                        entry.count[half] += 1;
                        for a in 0..n {
                            let gain = (value[a] - mix) as f64;
                            entry.gain[half][a] += gain;
                            entry.gain_squared[half][a] += gain * gain;
                        }
                        node = kids[sample(&sigma[..n], &mut rng)];
                    }
                }
            }
        }
        let mut deviation_gain = vec![0f64; players];
        for s in samples.values() {
            // Choose on one half, measure on the other, both ways round.
            // Each half estimates the gain per hand from half the deals;
            // averaging the two estimates is the plain sum over all deals.
            for (choose, measure) in [(0, 1), (1, 0)] {
                if let Some(best) = s.clear_improvement(choose) {
                    deviation_gain[s.seat as usize] += s.gain[measure][best];
                }
            }
        }
        let per_hand = |total: f64| if deals > 0 { total / deals as f64 } else { 0.0 };
        Diagnosis {
            deals,
            ev: ev.into_iter().map(per_hand).collect(),
            deviation_gain: deviation_gain.into_iter().map(per_hand).collect(),
        }
    }

    /// The value to `seat` of `node` when everyone plays the average
    /// strategy from there.
    fn rollout(&self, mut node: u32, seat: usize, deal: &Deal, rng: &mut Rng) -> f32 {
        loop {
            match &self.nodes[node as usize] {
                Flat::Leaf { live, prize, commit, context } => {
                    return self.payoff(*live, *prize, commit[seat], context[seat], seat, deal);
                }
                &Flat::Decision { seat: actor, num_actions, kids, cells, .. } => {
                    let n = num_actions as usize;
                    let base = cells + deal.class[actor as usize] as usize * n;
                    let mut sigma = [0f32; 4];
                    self.average_into(base, &mut sigma[..n]);
                    node = kids[sample(&sigma[..n], rng)];
                }
            }
        }
    }
}
