//! Preflop charts and the abstract preflop action model.
//!
//! The preflop menu offers at most four actions per decision: fold,
//! check/call ("passive"), the single menu raise, and all-in. Charts give a
//! frequency for each of them per hand class, keyed by the *spot*: the
//! position-labelled line so far and the deciding position, e.g.
//! `"UTG:R BTN:C>BB"` (UTG opened, BTN called, BB to act) or `">SB"`
//! (SB first to act heads-up). Folds are only listed for players who had
//! already put money in voluntarily, so folds "to the opener" don't change
//! the key. Spots without a chart fall back to position-aware rules.

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

/// A set of charts keyed by spot, for one table size.
#[derive(Clone, Debug)]
pub struct ChartSet {
    pub charts: HashMap<String, Chart>,
    /// Table size the charts are meant for (used by the fallback rules).
    pub players: usize,
}

const HU_100BB: &str = include_str!("../charts/hu_100bb.txt");
const HU_100BB_SOLVER: &str = include_str!("../charts/hu_100bb_solver.txt");
const SIX_MAX_100BB_SOLVER: &str = include_str!("../charts/6max_100bb_solver.txt");

impl ChartSet {
    /// The hand-written heads-up 100bb charts.
    pub fn hu_100bb() -> ChartSet {
        ChartSet::parse(HU_100BB, 2).expect("built-in charts parse")
    }

    /// Built-in charts for a table size:
    /// * heads-up: solver charts for the first decisions (SB open, BB vs
    ///   limp, BB vs open), hand-written charts for 3-bets and beyond;
    /// * 3-6 players: solver-generated 6-max 100bb charts;
    /// * larger tables: none (position-aware fallback rules).
    ///
    /// Spots without a chart always use the fallback rules.
    pub fn builtin(players: usize) -> ChartSet {
        match players {
            0..=2 => ChartSet::hu_100bb()
                .overlay(ChartSet::parse(HU_100BB_SOLVER, 2).expect("built-in charts parse")),
            3..=6 => ChartSet::parse(SIX_MAX_100BB_SOLVER, players).expect("built-in charts parse"),
            _ => ChartSet { charts: HashMap::new(), players },
        }
    }

    /// Adds `other`'s spots, replacing spots that both define.
    pub fn overlay(mut self, other: ChartSet) -> ChartSet {
        self.charts.extend(other.charts);
        self
    }

    /// Writes the charts in the text format read by [`ChartSet::parse`].
    pub fn to_text(&self, header: &str) -> String {
        let mut out = String::new();
        for line in header.lines() {
            out.push_str(&format!("# {line}\n"));
        }
        let mut keys: Vec<&String> = self.charts.keys().collect();
        keys.sort_by_key(|k| (k.matches(':').count(), k.to_string()));
        for key in keys {
            let chart = &self.charts[key];
            out.push_str(&format!("\n[{key}]\n"));
            for (label, idx) in [("allin", 3), ("raise", 2), ("call", 1)] {
                let parts: Vec<String> = chart
                    .freqs
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f[idx] > 1e-4)
                    .map(|(g, f)| {
                        let class = HandClass::from_grid_index(g);
                        if f[idx] > 0.9999 {
                            class.to_string()
                        } else {
                            format!("{class}:{:.4}", f[idx])
                        }
                    })
                    .collect();
                if !parts.is_empty() {
                    out.push_str(&format!("{label}: {}\n", parts.join(", ")));
                }
            }
        }
        out
    }

    pub fn parse(text: &str, players: usize) -> Result<ChartSet, String> {
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
        Ok(ChartSet { charts, players })
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
            None => (avail.remap(fallback_freqs(key, class, self.players)), false),
        }
    }
}

/// A parsed spot key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spot {
    /// `(position, action code)` for each action in the line.
    pub line: Vec<(String, char)>,
    /// Deciding position.
    pub position: String,
}

impl Spot {
    pub fn parse(key: &str) -> Option<Spot> {
        let (line, pos) = key.rsplit_once('>')?;
        let line = line
            .split_whitespace()
            .map(|t| {
                let (p, a) = t.rsplit_once(':')?;
                Some((p.to_string(), a.chars().next()?))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Spot { line, position: pos.trim().to_string() })
    }
}

/// Typical open-raise frequency by position (fallback rules).
pub fn rfi_pct(players: usize, pos: &str) -> f32 {
    match players {
        0..=2 => {
            if pos == "SB" {
                0.82
            } else {
                0.5
            }
        }
        3..=6 => match pos {
            "UTG" => 0.17,
            "HJ" => 0.21,
            "CO" => 0.28,
            "BTN" => 0.44,
            "SB" => 0.40,
            _ => 0.25,
        },
        _ => match pos {
            "UTG" => 0.11,
            "UTG+1" => 0.12,
            "UTG+2" => 0.13,
            "MP" => 0.14,
            "LJ" => 0.16,
            "HJ" => 0.19,
            "CO" => 0.26,
            "BTN" => 0.42,
            "SB" => 0.38,
            _ => 0.2,
        },
    }
}

/// Smooth 0..1 step: the share of `[threshold - w, threshold + w]` that
/// `rank` has passed. Mixes hands at the edge of a range.
fn ramp(rank: f32, threshold: f32) -> f32 {
    const W: f32 = 0.012;
    ((rank - (threshold - W)) / (2.0 * W)).clamp(0.0, 1.0)
}

/// Position-aware rules for spots without a chart: hands are ranked by
/// preflop strength and each spot type continues with a fraction of hands
/// that depends on positions and the action so far.
pub fn fallback_freqs(key: &str, class: usize, players: usize) -> Freqs {
    let rank = strength::class_rank()[class];
    let Some(spot) = Spot::parse(key) else {
        return [1.0, 0.0, 0.0, 0.0];
    };
    let aggressive: Vec<&(String, char)> = spot.line.iter().filter(|(_, a)| *a == 'R' || *a == 'A').collect();
    let n_raises = aggressive.len();
    let facing_allin = aggressive.last().is_some_and(|(_, a)| *a == 'A');
    let limpers = spot.line.iter().take_while(|(_, a)| *a == 'C').count();
    let last_raise = spot.line.iter().rposition(|(_, a)| *a == 'R' || *a == 'A');
    let callers_after =
        last_raise.map_or(0, |i| spot.line[i + 1..].iter().filter(|(_, a)| *a == 'C').count());
    let pos = spot.position.as_str();
    // (raise share, call share) of all hands, best hands raising.
    let (raise, call, raise_is_allin) = if facing_allin {
        let x = match n_raises {
            1 => 0.10,
            2 => 0.06,
            _ => 0.04,
        };
        (0.0, x * if callers_after > 0 { 0.7 } else { 1.0 }, false)
    } else {
        match n_raises {
            0 if limpers == 0 => (rfi_pct(players, pos), 0.0, false),
            0 if pos == "BB" => (0.25, 1.0, false),
            0 => (0.10, 0.10, false),
            1 => {
                let opener = aggressive[0].0.as_str();
                let o = rfi_pct(players, opener);
                let mut t = (0.3 * o + 0.02).clamp(0.03, 0.14);
                let mut c = match pos {
                    "BB" => (1.1 * o).clamp(0.10, 0.50),
                    "SB" => 0.25 * o,
                    _ => (0.45 * o).clamp(0.03, 0.18),
                };
                if callers_after > 0 {
                    t *= 0.8;
                    c *= 0.8;
                }
                (t, c, false)
            }
            2 => (0.025, 0.07, false),
            3 => (0.02, 0.012, true),
            _ => (0.015, 0.0, true),
        }
    };
    let r = if raise > 0.0 { ramp(rank, 1.0 - raise) } else { 0.0 };
    let c = if call > 0.0 { (ramp(rank, 1.0 - raise - call) - r).max(0.0) } else { 0.0 };
    let f = (1.0 - r - c).max(0.0);
    if raise_is_allin {
        [f, c, 0.0, r]
    } else {
        [f, c, r, 0.0]
    }
}

/// One preflop decision of the hand.
#[derive(Clone, Debug)]
pub struct PreDecision {
    pub seat: usize,
    /// Position name of the deciding seat (e.g. "BTN").
    pub position: String,
    /// Spot key at the time of the decision.
    pub key: String,
    pub avail: Availability,
    pub action: PreAction,
}

/// Spot key for `position` to act after `decisions`.
pub fn spot_key(decisions: &[PreDecision], position: &str) -> String {
    let mut entered: Vec<usize> = Vec::new();
    let mut tokens = Vec::new();
    for d in decisions {
        let was_in = entered.contains(&d.seat);
        if d.action != PreAction::Fold {
            if !was_in {
                entered.push(d.seat);
            }
            tokens.push(format!("{}:{}", d.position, d.action.code()));
        } else if was_in {
            tokens.push(format!("{}:F", d.position));
        }
    }
    format!("{}>{}", tokens.join(" "), position)
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
        for key in
            [">SB", "SB:R>BB", "SB:R BB:R>SB", "SB:R BB:R SB:R>BB", "SB:C>BB", "SB:C BB:R>SB", "SB:A>BB"]
        {
            assert!(cs.get(key).is_some(), "{key}");
        }
        let open = cs.get(">SB").unwrap();
        assert_eq!(open.class_freqs(class("AA"))[2], 1.0);
        assert_eq!(open.class_freqs(class("72o"))[0], 1.0);
        // Open frequency ~82%.
        let total: f32 =
            HandClass::all().map(|c| open.class_freqs(c.grid_index())[2] * c.num_combos() as f32).sum();
        assert!((total / 1326.0 - 0.82).abs() < 0.02, "{}", total / 1326.0);
        // BB vs open: A5s mixes raise/call.
        let f = cs.get("SB:R>BB").unwrap().class_freqs(class("A5s"));
        assert!((f[2] - 0.5).abs() < 1e-6 && (f[1] - 0.5).abs() < 1e-6, "{f:?}");
    }

    #[test]
    fn builtin_solver_charts() {
        // Heads-up: solver spots for the first decisions, hand-written beyond.
        let hu = ChartSet::builtin(2);
        let hand = ChartSet::hu_100bb();
        let solver = ChartSet::parse(HU_100BB_SOLVER, 2).unwrap();
        assert_eq!(solver.charts.len(), 3);
        for key in [">SB", "SB:R>BB", "SB:C>BB"] {
            let (a, b) = (hu.get(key).unwrap(), solver.get(key).unwrap());
            assert_eq!(a.freqs, b.freqs, "{key}");
        }
        assert_eq!(hu.get("SB:R BB:R>SB").unwrap().freqs, hand.get("SB:R BB:R>SB").unwrap().freqs);
        // The solver's SB opens (raise or limp) almost every hand.
        let open = hu.get(">SB").unwrap();
        let play: f32 = HandClass::all()
            .map(|c| (1.0 - open.class_freqs(c.grid_index())[0]) * c.num_combos() as f32)
            .sum::<f32>()
            / 1326.0;
        assert!((play - 0.905).abs() < 0.01, "{play}");
        // 6-max: complete solver charts.
        let six = ChartSet::builtin(6);
        assert_eq!(six.charts.len(), 506);
        assert!(ChartSet::builtin(9).charts.is_empty());
        // Every built-in spot is a distribution, and aces never fold.
        for cs in [&hu, &six] {
            for (key, chart) in &cs.charts {
                for (g, f) in chart.freqs.iter().enumerate() {
                    assert!(
                        f.iter().all(|&x| x >= 0.0) && (f.iter().sum::<f32>() - 1.0).abs() < 1e-4,
                        "{key} {g}"
                    );
                }
                assert!(chart.class_freqs(class("AA"))[0] <= 0.05, "{key}: AA folds");
            }
        }
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
        let d = PreDecision {
            seat: 0,
            position: "SB".into(),
            key: ">SB".into(),
            avail,
            action: PreAction::Passive,
        };
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
    fn spot_keys() {
        let avail = Availability {
            actions: [Some(Action::Fold), Some(Action::Call), Some(Action::Raise(250)), None],
        };
        let d = |seat: usize, pos: &str, a: PreAction| PreDecision {
            seat,
            position: pos.into(),
            key: String::new(),
            avail: avail.clone(),
            action: a,
        };
        // UTG folds, HJ opens, CO folds, BTN calls -> SB to act.
        let line = vec![
            d(3, "UTG", PreAction::Fold),
            d(4, "HJ", PreAction::Raise),
            d(5, "CO", PreAction::Fold),
            d(0, "BTN", PreAction::Passive),
        ];
        assert_eq!(spot_key(&line, "SB"), "HJ:R BTN:C>SB");
        // ... SB 3-bets, BB folds, HJ folds: BTN to act.
        let mut more = line.clone();
        more.push(d(1, "SB", PreAction::Raise));
        more.push(d(2, "BB", PreAction::Fold));
        more.push(d(4, "HJ", PreAction::Fold));
        assert_eq!(spot_key(&more, "BTN"), "HJ:R BTN:C SB:R HJ:F>BTN");
        assert_eq!(spot_key(&[], "SB"), ">SB");
        let spot = Spot::parse("HJ:R BTN:C>SB").unwrap();
        assert_eq!(spot.position, "SB");
        assert_eq!(spot.line, vec![("HJ".to_string(), 'R'), ("BTN".to_string(), 'C')]);
    }

    #[test]
    fn fallback_rules_are_positional() {
        let total = |key: &str, players: usize, idx: usize| -> f32 {
            HandClass::all()
                .map(|c| fallback_freqs(key, c.grid_index(), players)[idx] * c.num_combos() as f32)
                .sum::<f32>()
                / 1326.0
        };
        let utg = total(">UTG", 6, 2);
        let btn = total(">BTN", 6, 2);
        assert!((utg - 0.17).abs() < 0.02 && (btn - 0.44).abs() < 0.02, "{utg} {btn}");
        assert!(total(">UTG", 9, 2) < utg);
        // BB defends wider against a BTN open than against UTG.
        assert!(total("BTN:R>BB", 6, 1) > total("UTG:R>BB", 6, 1));
        // Facing an all-in nobody raises.
        assert_eq!(total("UTG:A>BB", 6, 2), 0.0);
    }

    #[test]
    fn remap_moves_mass() {
        let avail = Availability { actions: [None, Some(Action::Check), Some(Action::Raise(350)), None] };
        // BB facing a limp: fold isn't possible -> check.
        assert_eq!(avail.remap([0.6, 0.0, 0.4, 0.0]), [0.0, 0.6, 0.4, 0.0]);
    }
}
