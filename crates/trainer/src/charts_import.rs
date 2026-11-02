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

use crate::preflop::{Chart, ChartSet, Freqs};
use holdem::combo::NUM_CLASSES;
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

/// Builds charts from per-decision accumulators: `actions[key][a]` holds
/// class frequencies; missing folds are the remainder.
fn finish(actions: HashMap<String, [Option<[f32; NUM_CLASSES]>; 4]>, players: usize) -> ChartSet {
    let mut charts = HashMap::new();
    for (key, acts) in actions {
        let freqs: Vec<Freqs> = (0..NUM_CLASSES)
            .map(|g| {
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
        charts.insert(key, Chart { freqs });
    }
    ChartSet { charts, players }
}

/// Imports a directory of per-decision line files.
pub fn import_line_files(dir: &Path, players: usize, allin_bb: f64) -> Result<ChartSet, String> {
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
    Ok(finish(actions, players))
}

/// Imports a leaf tree (`SB/2.6bb/BB/Call/SB_range.txt` ...) for heads-up.
pub fn import_leaf_tree(dir: &Path, allin_bb: f64) -> Result<ChartSet, String> {
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
        if !ranges.is_empty() && path.len() % 2 == 0 {
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
    let mut actions: HashMap<String, [Option<[f32; NUM_CLASSES]>; 4]> = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    for line in leaves.keys() {
        for i in 0..line.len() {
            let prefix = &line[..=i];
            if !seen.insert(prefix.to_vec()) {
                continue;
            }
            let (pos, act) = &line[i];
            let Some(code) = action_code(act, allin_bb) else { continue };
            let Some(after) = range_after(prefix) else { continue };
            // The player's range before this decision.
            let before = match line[..i].iter().rposition(|(p, _)| p == pos) {
                Some(j) => match range_after(&line[..=j]) {
                    Some(r) => r,
                    None => continue,
                },
                None => [1.0; NUM_CLASSES],
            };
            let mut freq = [0f32; NUM_CLASSES];
            for g in 0..NUM_CLASSES {
                freq[g] = if before[g] > 1e-6 { (after[g] / before[g]).clamp(0.0, 1.0) } else { 0.0 };
            }
            let key = format!(
                "{}>{}",
                line[..i]
                    .iter()
                    .filter_map(|(p, a)| action_code(a, allin_bb).map(|c| format!("{p}:{c}")))
                    .collect::<Vec<_>>()
                    .join(" "),
                pos
            );
            let slot = &mut actions.entry(key).or_insert([None; 4])[action_index(code)];
            *slot = Some(match slot {
                Some(prev) => {
                    let mut sum = *prev;
                    for g in 0..NUM_CLASSES {
                        sum[g] += freq[g];
                    }
                    sum
                }
                None => freq,
            });
        }
    }
    Ok(finish(actions, 2))
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
        let cs = import_line_files(&d, 6, 90.0).unwrap();
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

    #[test]
    fn leaf_tree() {
        let d = tmpdir("leaves");
        let leaf = |path: &str, sb: &str, bb: &str| {
            let p = d.join(path);
            fs::create_dir_all(&p).unwrap();
            fs::write(p.join("SB_range.txt"), sb).unwrap();
            fs::write(p.join("BB_range.txt"), bb).unwrap();
        };
        // SB opens AA,KK,QQ (QQ half the time); BB calls with QQ, 3-bets AA
        // (to 7.8bb) and KK (to 5.2bb); SB calls the 3-bet with everything
        // it opened except QQ.
        leaf("SB/2.6bb/BB/Call", "AA:1,KK:1,QQ:0.5", "QQ:1,JJ:1");
        leaf("SB/2.6bb/BB/7.8bb/SB/Call", "AA:1,KK:1", "AA:1");
        leaf("SB/2.6bb/BB/5.2bb/SB/Call", "AA:1,KK:1", "KK:1");
        let cs = import_leaf_tree(&d, 90.0).unwrap();
        assert_eq!(cs.get(">SB").unwrap().class_freqs(g("QQ")), [0.5, 0.0, 0.5, 0.0]);
        let vs_open = cs.get("SB:R>BB").unwrap();
        assert_eq!(vs_open.class_freqs(g("AA")), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(vs_open.class_freqs(g("KK")), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(vs_open.class_freqs(g("QQ")), [0.0, 1.0, 0.0, 0.0]);
        // SB vs 3-bet: AA/KK always call (range after = range before).
        let vs3 = cs.get("SB:R BB:R>SB").unwrap();
        assert_eq!(vs3.class_freqs(g("KK")), [0.0, 1.0, 0.0, 0.0]);
        let _ = fs::remove_dir_all(&d);
    }
}
