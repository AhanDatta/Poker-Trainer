//! Preflop charts and the abstract preflop action model.
//!
//! The preflop menu offers at most four actions per decision: fold,
//! check/call ("passive"), the single menu raise, and all-in. Charts give a
//! frequency for each of them per hand class, keyed by the preflop line
//! (`"-"`, `"R"`, `"RR"`, `"C"`, `"CR"`, ...).

use crate::strength;
use holdem::combo::{HandClass, COMBO_CLASS, NUM_CLASSES, NUM_COMBOS};
use holdem::{Action, HandState, Range, Situation};
use std::collections::HashMap;

/// Abstract preflop action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PreAction {
    Fold,
    Passive,
    Raise,
    AllIn,
}

impl PreAction {
    pub const ALL: [PreAction; 4] = [PreAction::Fold, PreAction::Passive, PreAction::Raise, PreAction::AllIn];

    pub fn index(self) -> usize {
        self as usize
    }

    /// Letter used in line keys.
    pub fn code(self) -> char {
        match self {
            PreAction::Fold => 'F',
            PreAction::Passive => 'C',
            PreAction::Raise => 'R',
            PreAction::AllIn => 'A',
        }
    }
}

/// Frequencies `[fold, passive, raise, allin]`.
pub type Freqs = [f32; 4];

/// Which abstract actions exist at a decision, and the concrete action each
/// maps to.
#[derive(Clone, Debug, PartialEq)]
pub struct Availability {
    pub actions: [Option<Action>; 4],
}

impl Availability {
    /// Classifies the menu actions of a preflop decision.
    pub fn from_actions(actions: &[Action], sit: &Situation) -> Availability {
        let mut out: [Option<Action>; 4] = [None; 4];
        for &a in actions {
            let slot = match a {
                Action::Fold => PreAction::Fold,
                Action::Check | Action::Call => PreAction::Passive,
                Action::Raise(to) if to >= sit.max_to => PreAction::AllIn,
                Action::Raise(_) => PreAction::Raise,
            };
            if out[slot.index()].is_none() {
                out[slot.index()] = Some(a);
            }
        }
        Availability { actions: out }
    }

    pub fn has(&self, a: PreAction) -> bool {
        self.actions[a.index()].is_some()
    }

    pub fn concrete(&self, a: PreAction) -> Option<Action> {
        self.actions[a.index()]
    }

    /// Abstract class of a concrete action.
    pub fn classify(&self, action: Action) -> Option<PreAction> {
        PreAction::ALL.into_iter().find(|&p| self.actions[p.index()] == Some(action))
    }

    /// Moves probability mass of unavailable actions to available ones and
    /// renormalizes.
    pub fn remap(&self, f: Freqs) -> Freqs {
        let [mut fold, mut pas, mut raise, mut allin] = f;
        if !self.has(PreAction::Raise) {
            if self.has(PreAction::AllIn) {
                allin += raise;
            } else {
                pas += raise;
            }
            raise = 0.0;
        }
        if !self.has(PreAction::AllIn) {
            if self.has(PreAction::Raise) {
                raise += allin;
            } else {
                pas += allin;
            }
            allin = 0.0;
        }
        if !self.has(PreAction::Passive) {
            fold += pas;
            pas = 0.0;
        }
        if !self.has(PreAction::Fold) {
            pas += fold;
            fold = 0.0;
        }
        let total = fold + pas + raise + allin;
        if total <= 0.0 {
            return if self.has(PreAction::Fold) { [1.0, 0.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0, 0.0] };
        }
        [fold / total, pas / total, raise / total, allin / total]
    }
}

/// Per-class action frequencies for one spot.
#[derive(Clone, Debug)]
pub struct Chart {
    pub freqs: Vec<Freqs>,
}

impl Chart {
    pub fn class_freqs(&self, class: usize) -> Freqs {
        self.freqs[class]
    }
}

/// A set of charts keyed by preflop line.
#[derive(Clone, Debug)]
pub struct ChartSet {
    pub charts: HashMap<String, Chart>,
}

const HU_100BB: &str = include_str!("../charts/hu_100bb.txt");

impl ChartSet {
    /// Built-in heads-up 100bb charts.
    pub fn hu_100bb() -> ChartSet {
        ChartSet::parse(HU_100BB).expect("built-in charts parse")
    }

    pub fn parse(text: &str) -> Result<ChartSet, String> {
        let mut charts = HashMap::new();
        let mut current: Option<(String, [Option<Range>; 3])> = None;
        let finish = |cur: Option<(String, [Option<Range>; 3])>, charts: &mut HashMap<String, Chart>| {
            if let Some((key, [allin, raise, call])) = cur {
                let cw =
                    |r: &Option<Range>| r.as_ref().map(|r| r.class_weights()).unwrap_or([0.0; NUM_CLASSES]);
                let (a, r, c) = (cw(&allin), cw(&raise), cw(&call));
                let freqs = (0..NUM_CLASSES)
                    .map(|g| {
                        let allin = a[g].clamp(0.0, 1.0);
                        let raise = r[g].clamp(0.0, 1.0 - allin);
                        let call = c[g].clamp(0.0, 1.0 - allin - raise);
                        [1.0 - allin - raise - call, call, raise, allin]
                    })
                    .collect();
                charts.insert(key, Chart { freqs });
            }
        };
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if let Some(key) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                finish(current.take(), &mut charts);
                current = Some((key.trim().to_ascii_uppercase(), [None, None, None]));
                continue;
            }
            let Some((kind, range)) = line.split_once(':') else {
                return Err(format!("line {}: expected `action: range`", lineno + 1));
            };
            let slot = match kind.trim().to_ascii_lowercase().as_str() {
                "allin" | "all-in" | "jam" => 0,
                "raise" => 1,
                "call" | "limp" | "check" => 2,
                other => return Err(format!("line {}: unknown action {other:?}", lineno + 1)),
            };
            let range = Range::parse(range).map_err(|e| format!("line {}: {e}", lineno + 1))?;
            let Some((_, slots)) = current.as_mut() else {
                return Err(format!("line {}: action outside of a [section]", lineno + 1));
            };
            slots[slot] = Some(range);
        }
        finish(current.take(), &mut charts);
        Ok(ChartSet { charts })
    }

    pub fn get(&self, key: &str) -> Option<&Chart> {
        self.charts.get(key)
    }

    /// Chart frequencies for a class at a spot, remapped to the available
    /// actions. Falls back to a strength-based default when the spot has no
    /// chart. Returns `(freqs, from_chart)`.
    pub fn freqs(&self, key: &str, class: usize, avail: &Availability) -> (Freqs, bool) {
        match self.get(key) {
            Some(chart) => (avail.remap(chart.class_freqs(class)), true),
            None => (avail.remap(fallback_freqs(key, class)), false),
        }
    }
}

/// Strength-threshold strategy for spots without a chart.
fn fallback_freqs(key: &str, class: usize) -> Freqs {
    let rank = strength::class_rank()[class];
    if key.ends_with('A') {
        // Facing an all-in: call with roughly the top 10%.
        return if rank > 0.90 { [0.0, 1.0, 0.0, 0.0] } else { [1.0, 0.0, 0.0, 0.0] };
    }
    let raises = key.chars().filter(|&c| c == 'R').count();
    let (raise_cut, call_cut) = match raises {
        0 => (0.75, 0.25),
        1 => (0.90, 0.45),
        2 => (0.95, 0.80),
        _ => (0.97, 0.90),
    };
    if rank > raise_cut {
        [0.0, 0.0, 1.0, 0.0]
    } else if rank > call_cut {
        [0.0, 1.0, 0.0, 0.0]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// One preflop decision of the hand.
#[derive(Clone, Debug)]
pub struct PreDecision {
    pub seat: usize,
    /// Line key at the time of the decision.
    pub key: String,
    pub avail: Availability,
    pub action: PreAction,
}

/// Line key from the abstract actions taken so far.
pub fn line_key(decisions: &[PreDecision]) -> String {
    if decisions.is_empty() {
        "-".to_string()
    } else {
        decisions.iter().map(|d| d.action.code()).collect()
    }
}

/// Availability for the player to act in `state`.
pub fn availability(state: &HandState, menu: &holdem::BetMenu) -> Option<Availability> {
    let p = state.to_act?;
    let actions = state.legal_actions(menu);
    Some(Availability::from_actions(&actions, &state.situation(p)))
}

/// Range implied by a seat's preflop decisions under `policy`, which gives
/// the per-class frequencies of a decision.
///
/// When the policy never takes the action that was actually taken (e.g. a
/// limp when the charts only raise or fold), the range is narrowed to the
/// hands the policy *continues* with at that decision instead; if even that
/// is empty the decision is ignored. Returns the range and the indices (into
/// `decisions`) of such uncovered decisions.
pub fn line_range<F>(decisions: &[PreDecision], seat: usize, mut policy: F) -> (Range, Vec<usize>)
where
    F: FnMut(&PreDecision) -> Vec<Freqs>,
{
    let mut class_w = [1f32; NUM_CLASSES];
    let mut uncovered = Vec::new();
    for (idx, d) in decisions.iter().enumerate().filter(|(_, d)| d.seat == seat) {
        let freqs = policy(d);
        let total_with = |factor: &[f32]| -> f32 { factor.iter().zip(&class_w).map(|(f, w)| f * w).sum() };
        let mut factor: Vec<f32> = freqs.iter().map(|f| f[d.action.index()]).collect();
        if total_with(&factor) <= 1e-6 {
            uncovered.push(idx);
            if d.action == PreAction::Fold {
                continue;
            }
            factor = freqs.iter().map(|f| 1.0 - f[PreAction::Fold.index()]).collect();
            if total_with(&factor) <= 1e-6 {
                continue;
            }
        }
        for (w, f) in class_w.iter_mut().zip(factor) {
            *w *= f;
        }
    }
    let mut weights = vec![0f32; NUM_COMBOS];
    for (c, w) in weights.iter_mut().enumerate() {
        *w = class_w[COMBO_CLASS[c] as usize];
    }
    (Range::from_weights(weights), uncovered)
}

/// Words for an abstract preflop action.
pub fn describe(a: PreAction, avail: &Availability) -> &'static str {
    match a {
        PreAction::Fold => "fold",
        PreAction::Passive => match avail.concrete(PreAction::Passive) {
            Some(Action::Check) => "check",
            _ => "call/limp",
        },
        PreAction::Raise => "raise",
        PreAction::AllIn => "all-in",
    }
}

/// All 169 class frequencies of a chart decision.
pub fn chart_policy(charts: &ChartSet, d: &PreDecision) -> Vec<Freqs> {
    (0..NUM_CLASSES).map(|g| charts.freqs(&d.key, g, &d.avail).0).collect()
}

/// Human-readable label of a class (used in summaries).
pub fn class_name(class: usize) -> String {
    HandClass::from_grid_index(class).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(s: &str) -> usize {
        s.parse::<HandClass>().unwrap().grid_index()
    }

    #[test]
    fn builtin_charts_parse() {
        let cs = ChartSet::hu_100bb();
        for key in ["-", "R", "RR", "RRR", "C", "CR", "A"] {
            assert!(cs.get(key).is_some(), "{key}");
        }
        let open = cs.get("-").unwrap();
        assert_eq!(open.class_freqs(class("AA"))[2], 1.0);
        assert_eq!(open.class_freqs(class("72o"))[0], 1.0);
        // Open frequency ~82%.
        let total: f32 =
            HandClass::all().map(|c| open.class_freqs(c.grid_index())[2] * c.num_combos() as f32).sum();
        assert!((total / 1326.0 - 0.82).abs() < 0.02, "{}", total / 1326.0);
        // BB vs open: A5s mixes raise/call.
        let f = cs.get("R").unwrap().class_freqs(class("A5s"));
        assert!((f[2] - 0.5).abs() < 1e-6 && (f[1] - 0.5).abs() < 1e-6, "{f:?}");
    }

    #[test]
    fn off_chart_limp_uses_continue_range() {
        let cs = ChartSet::hu_100bb();
        let avail = Availability {
            actions: [
                Some(Action::Fold),
                Some(Action::Call),
                Some(Action::Raise(250)),
                Some(Action::Raise(10000)),
            ],
        };
        let d = PreDecision { seat: 0, key: "-".into(), avail, action: PreAction::Passive };
        let (r, uncovered) = line_range(&[d], 0, |d| chart_policy(&cs, d));
        assert_eq!(uncovered, vec![0]);
        // Limping is never in the chart, so the range is the SB's opening range.
        let n = r.num_combos();
        assert!((n / 1326.0 - 0.82).abs() < 0.02, "{n}");
        let class = |s: &str| s.parse::<HandClass>().unwrap();
        assert_eq!(r.class_weights()[class("72o").grid_index()], 0.0);
        assert_eq!(r.class_weights()[class("AA").grid_index()], 1.0);
    }

    #[test]
    fn remap_moves_mass() {
        let avail = Availability { actions: [None, Some(Action::Check), Some(Action::Raise(350)), None] };
        // BB facing a limp: fold isn't possible -> check.
        assert_eq!(avail.remap([0.6, 0.0, 0.4, 0.0]), [0.0, 0.6, 0.4, 0.0]);
    }
}
