//! Grading the hero's preflop decisions against the charts.
//!
//! Charts give how often each action is played, not EVs, so a preflop grade
//! is the chart's frequency of the action the hero chose with their hand
//! class (see [`verdict_of`]), not EV lost like a postflop grade. Spots
//! without a chart (where only the fallback rules apply) aren't graded.

use crate::preflop::{ChartSet, Freqs, PreAction, PreDecision};
use crate::review::Verdict;
use holdem::{Card, HandClass};

/// The grade of one hero preflop decision.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PreflopGrade {
    /// Index of the decision in the hand's preflop decisions.
    pub decision: usize,
    /// The hero's hand class ([`HandClass::grid_index`]).
    pub class: usize,
    pub chosen: PreAction,
    /// The chart's frequencies for the class at the spot, on the actions
    /// that were available; `None` where no chart covers the spot.
    pub chart: Option<Freqs>,
    /// `None` where no chart covers the spot.
    pub verdict: Option<Verdict>,
}

impl PreflopGrade {
    /// How often the chart plays the chosen action (`None` without chart).
    pub fn chosen_freq(&self) -> Option<f32> {
        self.chart.map(|f| f[self.chosen.index()])
    }

    /// The chart's most frequent action (`None` without chart).
    pub fn chart_action(&self) -> Option<PreAction> {
        let f = self.chart?;
        PreAction::ALL.into_iter().max_by(|a, b| f[a.index()].total_cmp(&f[b.index()]))
    }
}

/// The verdict for choosing `chosen` where the chart plays `freqs`:
/// ✓ Best for the chart's most frequent action (or one played at least half
/// the time), ✓ Good at 15% or more, ?! Inaccuracy at 5%, ? Mistake at 1%,
/// ?? Blunder below.
pub fn verdict_of(freqs: &Freqs, chosen: PreAction) -> Verdict {
    let f = freqs[chosen.index()];
    let top = freqs.iter().copied().fold(0.0f32, f32::max);
    if f >= 0.5 || (f > 0.0 && f >= top - 1e-6) {
        Verdict::Best
    } else if f >= 0.15 {
        Verdict::Good
    } else if f >= 0.05 {
        Verdict::Inaccuracy
    } else if f >= 0.01 {
        Verdict::Mistake
    } else {
        Verdict::Blunder
    }
}

/// Grades every decision of `hero` in `decisions` (a hand's preflop
/// decisions, in order), holding `hole`.
pub fn grade(
    charts: &ChartSet,
    decisions: &[PreDecision],
    hero: usize,
    hole: [Card; 2],
) -> Vec<PreflopGrade> {
    let class = HandClass::of_cards(hole[0], hole[1]).grid_index();
    decisions
        .iter()
        .enumerate()
        .filter(|(_, d)| d.seat == hero)
        .map(|(i, d)| {
            let (freqs, from_chart) = charts.freqs(&d.key, class, &d.avail);
            let chart = from_chart.then_some(freqs);
            PreflopGrade {
                decision: i,
                class,
                chosen: d.action,
                chart,
                verdict: chart.map(|f| verdict_of(&f, d.action)),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
