//! Turning a solve into the trainer's charts, and measuring how far two
//! chart sets are apart on the spots that matter.

use crate::solver::Solver;
use crate::tree::{Decision, Node, Tree, NO_CHILD};
use holdem::{HandClass, NUM_CLASSES};
use std::collections::HashMap;
use trainer::preflop::{fallback_freqs, Chart, ChartSet, Freqs, PreAction};

/// The solve's average strategy as charts. A hand class with fewer than
/// `min_samples` samples at a spot (it practically never gets there) uses
/// the trainer's position-aware rules instead of noise, as imported charts
/// do.
pub fn to_charts(solver: &Solver, min_samples: u64) -> ChartSet {
    let players = solver.tree().spec.players;
    let charts = solver
        .tree()
        .decisions()
        .map(|d| {
            let freqs = (0..NUM_CLASSES)
                .map(|class| match solver.average(d, class) {
                    (freqs, samples) if samples >= min_samples.max(1) => freqs,
                    _ => d.avail.remap(fallback_freqs(&d.key, class, players)),
                })
                .collect();
            (d.key.clone(), Chart { freqs })
        })
        .collect();
    ChartSet { charts, players }
}

/// How a model's chart differs from a reference chart at one spot.
#[derive(Clone, Debug, PartialEq)]
pub struct SpotDiff {
    pub key: String,
    /// Probability that a hand reaches the spot when everyone follows the
    /// reference.
    pub reach: f32,
    /// Average over the hands that get here (by the reference) of the
    /// total-variation distance between the two strategies: 0 is the same
    /// play, 1 never the same action.
    pub distance: f32,
    /// How often the range that gets here takes each action.
    pub model: Freqs,
    pub reference: Freqs,
}

/// Compares `model` with `reference` on every spot of `tree` that both
/// have and that the reference reaches, most reached first.
pub fn compare(tree: &Tree, model: &ChartSet, reference: &ChartSet) -> Vec<SpotDiff> {
    let mut prior = [0f32; NUM_CLASSES];
    for class in HandClass::all() {
        prior[class.grid_index()] = class.num_combos() as f32 / holdem::NUM_COMBOS as f32;
    }
    let mut out = Vec::new();
    let mut ranges = vec![prior; tree.spec.players];
    descend(tree, 0, 1.0, &mut ranges, model, reference, &mut out);
    out.sort_by(|a, b| b.reach.total_cmp(&a.reach).then_with(|| a.key.cmp(&b.key)));
    out
}

fn descend(
    tree: &Tree,
    node: u32,
    reach: f32,
    ranges: &mut [[f32; NUM_CLASSES]],
    model: &ChartSet,
    reference: &ChartSet,
    out: &mut Vec<SpotDiff>,
) {
    let Node::Decision(d) = &tree.nodes[node as usize] else { return };
    let (Some(theirs), Some(ours)) = (reference.get(&d.key), model.get(&d.key)) else { return };
    let range = ranges[d.seat as usize];
    let total: f32 = range.iter().sum();
    if reach < 1e-7 || total <= 0.0 {
        return;
    }
    let (mut mine, mut refs, mut distance) = ([0f32; 4], [0f32; 4], 0.0);
    for class in 0..NUM_CLASSES {
        let (m, r) = (d.avail.remap(ours.class_freqs(class)), d.avail.remap(theirs.class_freqs(class)));
        let w = range[class] / total;
        for a in 0..4 {
            mine[a] += w * m[a];
            refs[a] += w * r[a];
            distance += w * 0.5 * (m[a] - r[a]).abs();
        }
    }
    out.push(SpotDiff { key: d.key.clone(), reach, distance, model: mine, reference: refs });
    for action in PreAction::ALL {
        let child = d.children[action.index()];
        if child == NO_CHILD || refs[action.index()] <= 0.0 {
            continue;
        }
        for class in 0..NUM_CLASSES {
            ranges[d.seat as usize][class] =
                range[class] * d.avail.remap(theirs.class_freqs(class))[action.index()];
        }
        descend(tree, child, reach * refs[action.index()], ranges, model, reference, out);
    }
    ranges[d.seat as usize] = range;
}

/// The reach-weighted mean distance of a comparison (0 when empty).
pub fn mean_distance(diffs: &[SpotDiff]) -> f32 {
    let reach: f32 = diffs.iter().map(|d| d.reach).sum();
    if reach <= 0.0 {
        return 0.0;
    }
    diffs.iter().map(|d| d.reach * d.distance).sum::<f32>() / reach
}

/// Range-weighted action frequencies of every decision, by spot key (for
/// summaries such as "how often does each position open").
pub fn first_in_frequencies(tree: &Tree, charts: &ChartSet) -> HashMap<String, Freqs> {
    let first_in = |d: &&Decision| !d.key.contains(':');
    tree.decisions()
        .filter(first_in)
        .filter_map(|d| {
            let chart = charts.get(&d.key)?;
            let mut freqs = [0f32; 4];
            for class in HandClass::all() {
                let w = class.num_combos() as f32 / holdem::NUM_COMBOS as f32;
                let f = d.avail.remap(chart.class_freqs(class.grid_index()));
                (0..4).for_each(|a| freqs[a] += w * f[a]);
            }
            Some((d.key.clone(), freqs))
        })
        .collect()
}

#[cfg(test)]
mod tests;
