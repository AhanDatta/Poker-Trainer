//! Importers for published preflop range collections.
//!
//! Two layouts are supported (both used by e.g. the `ranges/` folder of
//! github.com/bupticybee/TexasSolverGPU):
//!
//! * **Per-decision files** ("line files"): every file holds the frequency
//!   of one action at one decision, and its name spells the preflop line,
//!   e.g. `UTG_2.5bb_BTN_Call_BB_13.0bb.txt` = BB's 13bb squeeze after a
//!   UTG open and a BTN call. Players who fold without having entered are
//!   omitted from names.
//! * **Leaf trees**: a directory tree `SB/2.6bb/BB/Call/` whose leaf
//!   directories hold each player's range at the end of preflop
//!   (`SB_range.txt`, `BB_range.txt`). Decision frequencies are recovered by
//!   dividing a player's range after a decision by the range before it.
//!
//! Range files contain comma-separated `class:weight` entries (e.g.
//! `AKs:0.75,QQ:1`). Any raise size maps to the chart's single raise action
//! (or all-in), since the trainer's preflop menu has one size per level.

use crate::preflop::{fallback_freqs, Chart, ChartSet, Freqs};
use crate::strength;
use holdem::combo::{HandClass, NUM_CLASSES};
use holdem::Range;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

/// Abstract action code of a token such as `2.5bb`, `Call`, `FOLD`, `AllIn`.
/// Raises of at least `allin_bb` big blinds count as all-in.
fn action_code(token: &str, allin_bb: f64) -> Option<char> {
    let t = token.to_ascii_lowercase();
    match t.as_str() {
        "fold" => Some('F'),
        "call" | "check" | "limp" => Some('C'),
        "allin" | "all-in" | "jam" | "shove" => Some('A'),
        _ => {
            let v: f64 = t.strip_suffix("bb")?.parse().ok()?;
            Some(if v >= allin_bb { 'A' } else { 'R' })
        }
    }
}

fn action_index(code: char) -> usize {
    match code {
        'F' => 0,
        'C' => 1,
        'R' => 2,
        _ => 3,
    }
}

/// Maps dataset position names to the trainer's names for a table size.
pub fn position_alias(players: usize, pos: &str) -> String {
    let p = pos.to_ascii_uppercase();
    let mapped = match (players, p.as_str()) {
        (3..=6, "MP") => "HJ",
        (3..=6, "LJ") => "UTG",
        (_, "B" | "BU" | "BUTTON") => "BTN",
        (_, other) => other,
    };
    mapped.to_string()
}

/// Leaf line -> each player's class weights at the end of preflop.
type Leaves = BTreeMap<Vec<(String, String)>, HashMap<String, [f32; NUM_CLASSES]>>;

fn read_class_weights(path: &Path) -> Result<[f32; NUM_CLASSES], String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let range = Range::parse(text.trim()).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(range.class_weights())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "txt") {
            out.push(path);
        }
    }
    Ok(())
}

/// Per-decision accumulators: class frequencies per action (fold, call,
/// raise, all-in), and which classes the data covers at all.
type Accum = HashMap<String, ([Option<[f32; NUM_CLASSES]>; 4], [bool; NUM_CLASSES])>;

/// An imported chart set.
pub struct Import {
    pub charts: ChartSet,
    /// Spots left out, with the reason. Spots without a chart use the
    /// built-in charts or the position-aware rules.
    pub dropped: Vec<String>,
}

/// Builds charts from per-decision accumulators. Missing folds are the
/// remainder. Classes the data doesn't cover (hands that never reach the
/// spot under the solver's strategy) get the position-aware fallback
/// rules, so e.g. an AI that slowplayed aces doesn't fold them later.
///
/// Spots where aces fold more than 5% are dropped: solver output for
/// lines that are (almost) never reached can be untrained noise.
fn finish(actions: Accum, players: usize, mut dropped: Vec<String>) -> Import {
    let aces = HandClass::from_grid_index(0);
    debug_assert_eq!(aces.to_string(), "AA");
    let mut charts = HashMap::new();
    for (key, (acts, covered)) in actions {
        let freqs: Vec<Freqs> = (0..NUM_CLASSES)
            .map(|g| {
                if !covered[g] {
                    return fallback_freqs(&key, g, players);
                }
                let mut f = [0f32; 4];
                for a in 1..4 {
                    f[a] = acts[a].map_or(0.0, |w| w[g].clamp(0.0, 1.0));
                }
                let rest = 1.0 - f[1] - f[2] - f[3];
                f[0] = match acts[0] {
                    Some(w) if rest < -1e-3 => w[g],
                    Some(w) => w[g].max(rest.max(0.0)),
                    None => rest.max(0.0),
                };
                let total: f32 = f.iter().sum();
                if total > 0.0 {
                    f.iter_mut().for_each(|x| *x /= total);
                } else {
                    f = [1.0, 0.0, 0.0, 0.0];
                }
                f
            })
            .collect();
        if freqs[aces.grid_index()][0] > 0.05 {
            dropped.push(format!("{key} (aces fold)"));
            continue;
        }
        charts.insert(key, Chart { freqs });
    }
    dropped.sort();
    Import { charts: ChartSet { charts, players }, dropped }
}

/// Imports a directory of per-decision line files.
pub fn import_line_files(dir: &Path, players: usize, allin_bb: f64) -> Result<Import, String> {
    let mut files = Vec::new();
    collect_files(dir, &mut files)?;
    if files.is_empty() {
        return Err(format!("no .txt files under {}", dir.display()));
    }
    let mut actions: HashMap<String, [Option<[f32; NUM_CLASSES]>; 4]> = HashMap::new();
    let mut skipped = 0;
    for path in files {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let tokens: Vec<&str> = stem.split('_').collect();
        if tokens.len() < 2 || !tokens.len().is_multiple_of(2) {
            skipped += 1;
            continue;
        }
        let mut line = Vec::new();
        let mut ok = true;
        for pair in tokens.chunks(2) {
            match action_code(pair[1], allin_bb) {
                Some(c) => line.push((position_alias(players, pair[0]), c)),
                None => ok = false,
            }
        }
        if !ok {
            skipped += 1;
            continue;
        }
        let (pos, code) = line.pop().unwrap();
        let key =
            format!("{}>{}", line.iter().map(|(p, c)| format!("{p}:{c}")).collect::<Vec<_>>().join(" "), pos);
        let w = read_class_weights(&path)?;
        let slot = &mut actions.entry(key).or_insert([None; 4])[action_index(code)];
        // Several raise sizes at one decision merge into one raise action.
        *slot = Some(match slot {
            Some(prev) => {
                let mut sum = *prev;
                for g in 0..NUM_CLASSES {
                    sum[g] += w[g];
                }
                sum
            }
            None => w,
        });
    }
    if actions.is_empty() {
        return Err(format!("no usable range files under {} ({skipped} skipped)", dir.display()));
    }
    // With a fold file, a class missing from every file never reaches the
    // spot; without one, missing classes are folds.
    let accum = actions
        .into_iter()
        .map(|(key, acts)| {
            let covered: [bool; NUM_CLASSES] =
                std::array::from_fn(|g| acts[0].is_none() || acts.iter().flatten().any(|w| w[g] > 1e-6));
            (key, (acts, covered))
        })
        .collect();
    Ok(finish(accum, players, Vec::new()))
}

/// Combo-weighted size of a class-weight vector.
fn mass(w: &[f32; NUM_CLASSES]) -> f32 {
    (0..NUM_CLASSES).map(|g| w[g] * HandClass::from_grid_index(g).num_combos() as f32).sum()
}

/// Imports a leaf tree (`SB/2.6bb/BB/Call/SB_range.txt` ...) for heads-up.
///
/// A leaf tree only contains the lines somebody exported, usually those
/// that reach a flop. A decision's frequencies are its child ranges divided
/// by the range before it, so actions without exported lines (e.g. an
/// all-in) would silently turn into folds. Spots where the deciding
/// player's strongest hands aren't (almost) fully accounted for are
/// therefore dropped, and listed in [`Import::dropped`].
///
/// When several histories map to the same spot key (e.g. facing a 5.2bb or
/// a 7.8bb 3-bet), their frequencies are averaged, weighted by how often
/// the opponent takes each line and by the decider's range in each.
pub fn import_leaf_tree(dir: &Path, allin_bb: f64) -> Result<Import, String> {
    let mut leaves: Leaves = BTreeMap::new();
    fn walk(dir: &Path, path: &mut Vec<String>, leaves: &mut Leaves) -> Result<(), String> {
        let mut ranges = HashMap::new();
        for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let p = entry.map_err(|e| e.to_string())?.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default().to_string();
            if p.is_dir() {
                path.push(name);
                walk(&p, path, leaves)?;
                path.pop();
            } else if let Some(pos) = name.strip_suffix("_range.txt") {
                ranges.insert(pos.to_ascii_uppercase(), read_class_weights(&p)?);
            }
        }
        if !ranges.is_empty() && path.len().is_multiple_of(2) {
            let line = path.chunks(2).map(|c| (c[0].to_ascii_uppercase(), c[1].clone())).collect();
            leaves.insert(line, ranges);
        }
        Ok(())
    }
    walk(dir, &mut Vec::new(), &mut leaves)?;
    if leaves.is_empty() {
        return Err(format!("no *_range.txt leaves under {}", dir.display()));
    }
    // For a decision (prefix + action by `pos`), find a leaf below it where
    // `pos` doesn't act again: there, `pos`'s range is its range right after
    // the decision.
    let range_after = |prefix: &[(String, String)]| -> Option<[f32; NUM_CLASSES]> {
        let pos = &prefix.last()?.0;
        leaves.iter().find_map(|(line, ranges)| {
            (line.len() >= prefix.len()
                && line[..prefix.len()] == *prefix
                && line[prefix.len()..].iter().all(|(p, _)| p != pos))
            .then(|| ranges.get(pos).copied())
            .flatten()
        })
    };
    // The range of the player acting at `line[i]` before that decision.
    let range_before = |line: &[(String, String)], i: usize| -> Option<[f32; NUM_CLASSES]> {
        let pos = &line[i].0;
        match line[..i].iter().rposition(|(p, _)| p == pos) {
            Some(j) => range_after(&line[..=j]),
            None => Some([1.0; NUM_CLASSES]),
        }
    };
    // Probability (ignoring card removal) that the opponents of `pos` took
    // their actions in `hist`.
    let opponent_weight = |hist: &[(String, String)], pos: &str| -> f32 {
        let mut w = 1.0f32;
        for j in 0..hist.len() {
            if hist[j].0 == pos {
                continue;
            }
            if let (Some(after), Some(before)) = (range_after(&hist[..=j]), range_before(hist, j)) {
                let b = mass(&before);
                if b > 0.0 {
                    w *= mass(&after) / b;
                }
            }
        }
        w
    };

    struct Decision {
        before: [f32; NUM_CLASSES],
        weight: f32,
        acts: [Option<[f32; NUM_CLASSES]>; 4],
    }
    let mut spots: BTreeMap<String, BTreeMap<Vec<(String, String)>, Decision>> = BTreeMap::new();
    let mut seen = std::collections::HashSet::new();
    for line in leaves.keys() {
        for i in 0..line.len() {
            let prefix = &line[..=i];
            if !seen.insert(prefix.to_vec()) {
                continue;
            }
            let (pos, act) = &line[i];
            let Some(code) = action_code(act, allin_bb) else { continue };
            let (Some(after), Some(before)) = (range_after(prefix), range_before(line, i)) else { continue };
            let key = format!(
                "{}>{}",
                line[..i]
                    .iter()
                    .filter_map(|(p, a)| action_code(a, allin_bb).map(|c| format!("{p}:{c}")))
                    .collect::<Vec<_>>()
                    .join(" "),
                pos
            );
            let hist = line[..i].to_vec();
            let d = spots.entry(key).or_default().entry(hist.clone()).or_insert_with(|| Decision {
                before,
                weight: opponent_weight(&hist, pos),
                acts: [None; 4],
            });
            // Several sizes of one action at the same decision add up.
            let slot = d.acts[action_index(code)].get_or_insert([0.0; NUM_CLASSES]);
            for g in 0..NUM_CLASSES {
                if before[g] > 1e-6 {
                    slot[g] += (after[g] / before[g]).clamp(0.0, 1.0);
                }
            }
        }
    }

    let score = strength::class_score();
    let mut actions: Accum = HashMap::new();
    let mut dropped = Vec::new();
    for (key, hists) in spots {
        // Merge histories: weight each class by P(history) * P(class reaches it).
        let mut before = [0f32; NUM_CLASSES];
        let mut acc = [[0f32; NUM_CLASSES]; 4];
        let mut present = [false; 4];
        for d in hists.values() {
            for g in 0..NUM_CLASSES {
                let w = d.weight * d.before[g];
                before[g] += w;
                for a in 0..4 {
                    if let Some(f) = &d.acts[a] {
                        acc[a][g] += w * f[g];
                        present[a] = true;
                    }
                }
            }
        }
        let mut merged: [Option<[f32; NUM_CLASSES]>; 4] = [None; 4];
        for a in 0..4 {
            if present[a] {
                let mut f = [0f32; NUM_CLASSES];
                for g in 0..NUM_CLASSES {
                    if before[g] > 1e-9 {
                        f[g] = (acc[a][g] / before[g]).clamp(0.0, 1.0);
                    }
                }
                merged[a] = Some(f);
            }
        }
        // Completeness: the strongest 10% of the decider's range (by
        // combos) never folds, so its frequencies must add up to ~1.
        let mut order: Vec<usize> = (0..NUM_CLASSES).filter(|&g| before[g] > 1e-9).collect();
        order.sort_by(|&x, &y| score[y].total_cmp(&score[x]));
        let total = mass(&before);
        let (mut cum, mut covered) = (0f32, 0f32);
        for &g in &order {
            if cum >= 0.1 * total {
                break;
            }
            let m = before[g] * HandClass::from_grid_index(g).num_combos() as f32;
            let known: f32 = (1..4).map(|a| merged[a].map_or(0.0, |f| f[g])).sum();
            cum += m;
            covered += m * known.min(1.0);
        }
        if cum > 0.0 && covered / cum < 0.9 {
            dropped.push(format!("{key} (lines missing)"));
            continue;
        }
        let covered = std::array::from_fn(|g| before[g] > 1e-9);
        actions.insert(key, (merged, covered));
    }
    Ok(finish(actions, 2, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use holdem::combo::HandClass;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pt-import-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn g(s: &str) -> usize {
        s.parse::<HandClass>().unwrap().grid_index()
    }

    #[test]
    fn line_files() {
        let d = tmpdir("lines");
        fs::create_dir_all(d.join("UTG")).unwrap();
        fs::write(d.join("UTG/UTG_2.5bb.txt"), "AA:1,KK:1,72o:0").unwrap();
        fs::write(d.join("UTG/UTG_FOLD.txt"), "72o:1").unwrap();
        fs::write(d.join("UTG_2.5bb_MP_Call_BB_13.0bb.txt"), "AA:0.5,KK:1").unwrap();
        fs::write(d.join("UTG_2.5bb_MP_Call_BB_Call.txt"), "AA:0.5,QQ:1").unwrap();
        let cs = import_line_files(&d, 6, 90.0).unwrap().charts;
        let open = cs.get(">UTG").unwrap();
        assert_eq!(open.class_freqs(g("AA")), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(open.class_freqs(g("72o")), [1.0, 0.0, 0.0, 0.0]);
        // MP is renamed HJ for 6-max tables.
        let sq = cs.get("UTG:R HJ:C>BB").unwrap();
        assert_eq!(sq.class_freqs(g("AA")), [0.0, 0.5, 0.5, 0.0]);
        assert_eq!(sq.class_freqs(g("QQ")), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(sq.class_freqs(g("JJ")), [1.0, 0.0, 0.0, 0.0]);
        // Round trip through the text format.
        let back = ChartSet::parse(&cs.to_text("test"), 6).unwrap();
        assert_eq!(back.get("UTG:R HJ:C>BB").unwrap().class_freqs(g("AA")), [0.0, 0.5, 0.5, 0.0]);
        let _ = fs::remove_dir_all(&d);
    }

    fn leaf(d: &Path, path: &str, sb: &str, bb: &str) {
        let p = d.join(path);
        fs::create_dir_all(&p).unwrap();
        fs::write(p.join("SB_range.txt"), sb).unwrap();
        fs::write(p.join("BB_range.txt"), bb).unwrap();
    }

    /// A small but complete tree: SB opens AA, KK and half of QQ and limps
    /// the rest; BB 3-bets AA to 7.8bb, KK/QQ to 5.2bb and calls a wide
    /// range; SB calls 3-bets with AA/KK, plus QQ against the small size.
    fn tree(d: &Path, sb_calls_big_3bet: &str) {
        let bb_call = "JJ-22, A2s+, A2o+, K2s+, K2o+, Q2s+, Q2o+, J2s+, T2s+";
        leaf(d, "SB/Call/BB/Check", "any, AA:0, KK:0, QQ:0.5", "any");
        leaf(d, "SB/2.6bb/BB/Call", "AA, KK, QQ:0.5", bb_call);
        leaf(d, "SB/2.6bb/BB/7.8bb/SB/Call", sb_calls_big_3bet, "AA");
        leaf(d, "SB/2.6bb/BB/5.2bb/SB/Call", "AA, KK, QQ:0.5", "KK, QQ");
    }

    #[test]
    fn leaf_tree() {
        let d = tmpdir("leaves");
        tree(&d, "AA, KK");
        let imp = import_leaf_tree(&d, 90.0).unwrap();
        assert!(imp.dropped.is_empty(), "{:?}", imp.dropped);
        let cs = imp.charts;
        let close = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
        assert!(close(cs.get(">SB").unwrap().class_freqs(g("QQ")), [0.0, 0.5, 0.5, 0.0]));
        assert!(close(cs.get(">SB").unwrap().class_freqs(g("72o")), [0.0, 1.0, 0.0, 0.0]));
        let vs_open = cs.get("SB:R>BB").unwrap();
        assert!(close(vs_open.class_freqs(g("AA")), [0.0, 0.0, 1.0, 0.0]));
        assert!(close(vs_open.class_freqs(g("QQ")), [0.0, 0.0, 1.0, 0.0]));
        assert!(close(vs_open.class_freqs(g("JJ")), [0.0, 1.0, 0.0, 0.0]));
        assert!(close(vs_open.class_freqs(g("72o")), [1.0, 0.0, 0.0, 0.0]));
        // SB vs a 3-bet: BB uses the small size twice as often (12 combos
        // vs 6), so QQ (which only calls the small size) calls 2/3.
        let vs3 = cs.get("SB:R BB:R>SB").unwrap();
        assert!(close(vs3.class_freqs(g("KK")), [0.0, 1.0, 0.0, 0.0]));
        assert!(
            close(vs3.class_freqs(g("QQ")), [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]),
            "{:?}",
            vs3.class_freqs(g("QQ"))
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn leaf_tree_drops_spots_with_missing_lines() {
        // SB's AA goes all-in against the big 3-bet, a line the export
        // doesn't contain: AA would look like a fold, so the spot is dropped.
        let d = tmpdir("leaves-missing");
        tree(&d, "KK");
        let imp = import_leaf_tree(&d, 90.0).unwrap();
        assert_eq!(imp.dropped, vec!["SB:R BB:R>SB (lines missing)".to_string()]);
        assert!(imp.charts.get("SB:R BB:R>SB").is_none());
        assert!(imp.charts.get("SB:R>BB").is_some());
        let _ = fs::remove_dir_all(&d);
    }
}
