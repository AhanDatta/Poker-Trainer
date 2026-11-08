//! CSV export of saved sessions: one row per hand (results, for bb/100 by
//! format) and one row per graded decision (for leak analysis). The two
//! files join on `session` + `hand`.
//!
//! Everything is derived from the saved hands; the save format itself is
//! unchanged. Net results come from settling each hand, exactly as the
//! session's own stats do.

use crate::review::{DecisionReview, Variant, Verdict};
use crate::save::{self, LoadedSession, SavedHand, SessionMeta, SessionStore};
use holdem::{Card, Chips};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// File names written by [`export`].
pub const HANDS_FILE: &str = "poker-trainer-hands.csv";
pub const DECISIONS_FILE: &str = "poker-trainer-decisions.csv";

pub const HAND_COLUMNS: [&str; 18] = [
    "session",
    "session_start_utc",
    "hand",
    "players",
    "format",
    "stack_bb",
    "big_blind",
    "position",
    "hero_cards",
    "board",
    "opponents",
    "net_chips",
    "net_bb",
    "showdown",
    "review",
    "decisions",
    "ev_loss_bb",
    "worst_verdict",
];

pub const DECISION_COLUMNS: [&str; 19] = [
    "session",
    "hand",
    "decision",
    "variant",
    "counted",
    "street",
    "board",
    "history",
    "hero_cards",
    "pot_bb",
    "to_call_bb",
    "chosen",
    "best",
    "ev_loss_bb",
    "ev_loss_pct",
    "verdict",
    "exploitability_pct",
    "solver",
    "notes",
];

// ---------------------------------------------------------------- CSV ---

/// One CSV field (RFC 4180): quoted when it holds a comma, quote or line
/// break, with quotes doubled.
pub fn csv_field(s: &str) -> Cow<'_, str> {
    if s.contains([',', '"', '\n', '\r']) {
        Cow::Owned(format!("\"{}\"", s.replace('"', "\"\"")))
    } else {
        Cow::Borrowed(s)
    }
}

/// Writes one record (fields joined by commas, then a newline).
pub fn write_record<W: Write, S: AsRef<str>>(w: &mut W, fields: &[S]) -> io::Result<()> {
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            w.write_all(b",")?;
        }
        w.write_all(csv_field(f.as_ref()).as_bytes())?;
    }
    w.write_all(b"\n")
}

/// `x` with at most `decimals` decimals, trailing zeros dropped ("1.5",
/// "-2", never "-0").
fn num(x: f64, decimals: usize) -> String {
    let s = format!("{x:.decimals$}");
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { &s };
    if s == "-0" {
        "0".into()
    } else {
        s.to_string()
    }
}

fn cards(cs: &[Card], sep: &str) -> String {
    cs.iter().map(Card::to_string).collect::<Vec<_>>().join(sep)
}

// --------------------------------------------------------------- rows ---

/// A table size and stack depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Format {
    pub players: usize,
    /// Hero's starting stack, in chips.
    pub stack: Chips,
    pub big_blind: Chips,
}

impl Format {
    /// "HU", "6-max", "9-max".
    pub fn table(&self) -> String {
        if self.players == 2 {
            "HU".into()
        } else {
            format!("{}-max", self.players)
        }
    }

    pub fn stack_bb(&self) -> f64 {
        self.stack as f64 / self.big_blind as f64
    }
}

impl fmt::Display for Format {
    /// "6-max 100bb".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}bb", self.table(), num(self.stack_bb(), 2))
    }
}

/// The decisions that count in the session's stats and frequency score:
/// each decision once, as its GTO-range variant when both were solved
/// (the rule `Stats` and the score use).
fn counted(decisions: &[DecisionReview]) -> impl Iterator<Item = (bool, &DecisionReview)> {
    let has_gto = decisions.iter().any(|d| d.variant == Variant::Gto);
    decisions.iter().map(move |d| (!has_gto || d.variant == Variant::Gto, d))
}

/// One finished hand.
#[derive(Clone, Debug, PartialEq)]
pub struct HandRow {
    pub session: String,
    /// Session start, Unix seconds (hands have no time of their own).
    pub session_start: u64,
    pub hand: u64,
    pub format: Format,
    pub position: &'static str,
    pub hero_cards: String,
    pub board: String,
    /// Each opponent as "position style", in seat order.
    pub opponents: String,
    pub net_chips: Chips,
    /// The hero reached showdown.
    pub showdown: bool,
    /// "done", "cancelled", "error" or "none" (no postflop review).
    pub review: &'static str,
    /// Graded decisions, counted as in the session's stats.
    pub decisions: usize,
    /// Sum of their EV losses, in chips.
    pub ev_loss_chips: f64,
    pub worst: Option<Verdict>,
}

impl HandRow {
    pub fn new(meta: &SessionMeta, h: &SavedHand) -> HandRow {
        let st = &h.state;
        let settle = st.settle();
        let n = st.num_players();
        let opponents = (1..n)
            .map(|k| (h.hero + k) % n)
            .filter_map(|p| {
                let name = h.profiles.get(p)?.as_ref()?.name.as_str();
                Some(format!("{} {name}", st.position_name(p)))
            })
            .collect::<Vec<_>>()
            .join("; ");
        let review = match &h.review {
            None => "none",
            Some(r) if r.error.is_some() => "error",
            Some(r) if r.cancelled => "cancelled",
            Some(_) => "done",
        };
        let graded: Vec<&DecisionReview> =
            h.review.iter().flat_map(|r| counted(&r.decisions)).filter_map(|(c, d)| c.then_some(d)).collect();
        HandRow {
            session: meta.id.clone(),
            session_start: meta.created,
            hand: h.id,
            format: Format { players: n, stack: st.starting_stacks[h.hero], big_blind: st.config.big_blind },
            position: st.position_name(h.hero),
            hero_cards: cards(&st.hole[h.hero], ""),
            board: cards(&st.board, " "),
            opponents,
            net_chips: settle.net[h.hero],
            showdown: settle.showdown && !st.folded[h.hero],
            review,
            decisions: graded.len(),
            ev_loss_chips: graded.iter().map(|d| d.ev_loss as f64).sum(),
            worst: graded.iter().map(|d| d.verdict).max(),
        }
    }

    pub fn net_bb(&self) -> f64 {
        self.net_chips as f64 / self.format.big_blind as f64
    }

    /// The CSV record, in [`HAND_COLUMNS`] order.
    pub fn fields(&self) -> Vec<String> {
        let bb = self.format.big_blind as f64;
        vec![
            self.session.clone(),
            save::format_time(self.session_start),
            self.hand.to_string(),
            self.format.players.to_string(),
            self.format.table(),
            num(self.format.stack_bb(), 2),
            self.format.big_blind.to_string(),
            self.position.to_string(),
            self.hero_cards.clone(),
            self.board.clone(),
            self.opponents.clone(),
            self.net_chips.to_string(),
            num(self.net_bb(), 4),
            self.showdown.to_string(),
            self.review.to_string(),
            self.decisions.to_string(),
            num(self.ev_loss_chips / bb, 3),
            self.worst.map_or("", Verdict::label).to_string(),
        ]
    }
}

/// One graded decision (both variants when the review solved both; the
/// one the stats count has `counted` set).
#[derive(Clone, Debug)]
pub struct DecisionRow<'a> {
    pub session: &'a str,
    pub hand: u64,
    /// 1-based, in review order.
    pub index: usize,
    pub counted: bool,
    pub hero_cards: String,
    pub big_blind: Chips,
    pub d: &'a DecisionReview,
}

impl DecisionRow<'_> {
    /// The CSV record, in [`DECISION_COLUMNS`] order.
    pub fn fields(&self) -> Vec<String> {
        let d = self.d;
        let bb = self.big_blind as f64;
        let label = |i: usize| d.labels.get(i).cloned().unwrap_or_default();
        vec![
            self.session.to_string(),
            self.hand.to_string(),
            self.index.to_string(),
            match d.variant {
                Variant::Gto => "gto",
                Variant::Actual => "actual",
            }
            .to_string(),
            self.counted.to_string(),
            d.street.name().to_string(),
            cards(&d.board, " "),
            d.history.clone(),
            self.hero_cards.clone(),
            num(d.pot as f64 / bb, 2),
            num(d.to_call as f64 / bb, 2),
            label(d.chosen),
            label(d.best),
            num(d.ev_loss as f64 / bb, 3),
            num(d.ev_loss_pct as f64, 2),
            d.verdict.label().to_string(),
            num(d.exploitability_pct, 3),
            d.solver.describe(),
            d.notes.join("; "),
        ]
    }
}

/// The session's hands, in hand order.
pub fn hand_rows(s: &LoadedSession) -> Vec<HandRow> {
    let mut rows: Vec<HandRow> = s.hands.iter().map(|h| HandRow::new(&s.meta, h)).collect();
    rows.sort_by_key(|r| r.hand);
    rows
}

/// The session's graded decisions, in hand order.
pub fn decision_rows(s: &LoadedSession) -> Vec<DecisionRow<'_>> {
    let mut hands: Vec<&SavedHand> = s.hands.iter().collect();
    hands.sort_by_key(|h| h.id);
    let mut rows = Vec::new();
    for h in hands {
        let Some(r) = &h.review else { continue };
        let hero_cards = cards(&h.state.hole[h.hero], "");
        for (i, (c, d)) in counted(&r.decisions).enumerate() {
            rows.push(DecisionRow {
                session: &s.meta.id,
                hand: h.id,
                index: i + 1,
                counted: c,
                hero_cards: hero_cards.clone(),
                big_blind: h.state.config.big_blind,
                d,
            });
        }
    }
    rows
}

// ------------------------------------------------------------ bb/100 ---

/// Results in one format.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatTotals {
    pub format: Format,
    pub hands: usize,
    pub net_chips: Chips,
}

impl FormatTotals {
    pub fn net_bb(&self) -> f64 {
        self.net_chips as f64 / self.format.big_blind as f64
    }

    pub fn bb_per_100(&self) -> f64 {
        bb_per_100(self.net_bb(), self.hands)
    }
}

/// Big blinds won per 100 hands (0 without hands).
pub fn bb_per_100(net_bb: f64, hands: usize) -> f64 {
    if hands == 0 {
        0.0
    } else {
        100.0 * net_bb / hands as f64
    }
}

/// Totals per format, in format order.
pub fn totals_by_format(rows: &[HandRow]) -> Vec<FormatTotals> {
    let mut by: BTreeMap<Format, (usize, Chips)> = BTreeMap::new();
    for r in rows {
        let e = by.entry(r.format).or_default();
        e.0 += 1;
        e.1 += r.net_chips;
    }
    by.into_iter().map(|(format, (hands, net_chips))| FormatTotals { format, hands, net_chips }).collect()
}

/// Hands and net big blinds over all formats.
pub fn overall(totals: &[FormatTotals]) -> (usize, f64) {
    (totals.iter().map(|t| t.hands).sum(), totals.iter().map(FormatTotals::net_bb).sum())
}

/// A plain-text table of results by format, with the overall total.
pub fn summary_table(totals: &[FormatTotals]) -> String {
    let mut out = format!("{:<16} {:>7} {:>11} {:>9}\n", "Format", "Hands", "Net (bb)", "bb/100");
    let line = |name: &str, hands: usize, net: f64| {
        format!("{name:<16} {hands:>7} {:>11} {:>9}\n", signed(net, 2), signed(bb_per_100(net, hands), 1))
    };
    for t in totals {
        out += &line(&t.format.to_string(), t.hands, t.net_bb());
    }
    if totals.len() > 1 {
        let (hands, net) = overall(totals);
        out += &line("All", hands, net);
    }
    out
}

fn signed(x: f64, decimals: usize) -> String {
    let s = format!("{x:+.decimals$}");
    if s.trim_start_matches(['+', '-']).chars().all(|c| c == '0' || c == '.') {
        s.trim_start_matches(['+', '-']).to_string()
    } else {
        s
    }
}

// ------------------------------------------------------------- export ---

/// What [`export`] wrote.
#[derive(Clone, Debug, Default)]
pub struct ExportReport {
    pub sessions: usize,
    pub hands: usize,
    pub decisions: usize,
    /// Lines of `hands.jsonl` files that couldn't be read.
    pub skipped_lines: usize,
    /// Sessions that couldn't be read, with the reason.
    pub unreadable: Vec<String>,
    pub totals: Vec<FormatTotals>,
}

/// Writes both CSV files for `sessions` (in that order) into `out_dir`.
pub fn write_csvs(sessions: &[LoadedSession], out_dir: &Path) -> io::Result<ExportReport> {
    let mut hands_csv = Vec::new();
    let mut decisions_csv = Vec::new();
    write_record(&mut hands_csv, &HAND_COLUMNS)?;
    write_record(&mut decisions_csv, &DECISION_COLUMNS)?;
    let mut all = Vec::new();
    let mut report = ExportReport { sessions: sessions.len(), ..Default::default() };
    for s in sessions {
        report.skipped_lines += s.skipped;
        for r in hand_rows(s) {
            write_record(&mut hands_csv, &r.fields())?;
            all.push(r);
        }
        for r in decision_rows(s) {
            write_record(&mut decisions_csv, &r.fields())?;
            report.decisions += 1;
        }
    }
    report.hands = all.len();
    report.totals = totals_by_format(&all);
    fs::create_dir_all(out_dir)?;
    replace_file(&out_dir.join(HANDS_FILE), &hands_csv)?;
    replace_file(&out_dir.join(DECISIONS_FILE), &decisions_csv)?;
    Ok(report)
}

/// Exports the saved sessions under `root` (or just session `only`),
/// oldest first, replacing the CSV files in `out_dir`.
pub fn export(root: &Path, only: Option<&str>, out_dir: &Path) -> Result<ExportReport, String> {
    let mut unreadable = Vec::new();
    let sessions = match only {
        Some(id) => {
            let store = SessionStore::open(root, id).map_err(|e| e.to_string())?;
            vec![store.load()?]
        }
        None => {
            let mut metas = save::list_sessions(root);
            metas.sort_by(|a, b| a.created.cmp(&b.created).then_with(|| a.id.cmp(&b.id)));
            let mut v = Vec::new();
            for m in metas {
                match SessionStore::open(root, &m.id).map_err(|e| e.to_string()).and_then(|s| s.load()) {
                    Ok(s) => v.push(s),
                    Err(e) => unreadable.push(e),
                }
            }
            v
        }
    };
    let mut report = write_csvs(&sessions, out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    report.unreadable = unreadable;
    Ok(report)
}

/// Writes via a temporary file, so a failed export never leaves a half
/// written CSV behind.
fn replace_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests;
