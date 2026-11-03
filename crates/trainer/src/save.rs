//! Saved sessions: each session is a directory holding `meta.json` (settings,
//! table state and a summary; rewritten atomically) and `hands.jsonl` (one
//! finished hand with its review per line, appended as hands finish).
//!
//! A hand is written once its review is done, so a crash loses at most the
//! hands still being reviewed. A partly written last line (from a crash
//! mid-write) is skipped on load.

use crate::config::TrainerConfig;
use crate::hand::LogEntry;
use crate::preflop::PreDecision;
use crate::profiles::Profile;
use crate::review::{DecisionReview, ReviewStatus};
use holdem::{Chips, HandState};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

/// Version of the save format.
pub const FORMAT_VERSION: u32 = 1;

const META: &str = "meta.json";
const HANDS: &str = "hands.jsonl";

/// Totals shown in the session list.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionSummary {
    pub hands: usize,
    pub net_chips: Chips,
    /// Graded decisions.
    pub decisions: usize,
    /// Frequency score (0-100), once there are scored decisions.
    pub score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionMeta {
    pub version: u32,
    /// Directory name.
    pub id: String,
    /// Unix seconds.
    pub created: u64,
    pub updated: u64,
    pub config: TrainerConfig,
    pub lineup: Vec<Option<Profile>>,
    pub button: usize,
    pub next_id: u64,
    pub summary: SessionSummary,
}

/// The parts of a [`ReviewStatus`] worth keeping.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedReview {
    pub stage: String,
    pub decisions: Vec<DecisionReview>,
    pub notes: Vec<String>,
    pub cancelled: bool,
    pub error: Option<String>,
}

impl SavedReview {
    pub fn from_status(st: &ReviewStatus) -> SavedReview {
        SavedReview {
            stage: st.stage.clone(),
            decisions: st.decisions.clone(),
            notes: st.notes.clone(),
            // A review saved before it finished (on quit) was cut short.
            cancelled: st.cancelled || !st.finished,
            error: st.error.clone(),
        }
    }

    pub fn into_status(self) -> ReviewStatus {
        ReviewStatus {
            stage: self.stage,
            decisions: self.decisions,
            notes: self.notes,
            finished: true,
            cancelled: self.cancelled,
            error: self.error,
            ..Default::default()
        }
    }
}

/// One finished hand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedHand {
    pub id: u64,
    pub hero: usize,
    pub state: HandState,
    pub profiles: Vec<Option<Profile>>,
    pub preflop: Vec<PreDecision>,
    pub log: Vec<LogEntry>,
    pub roll_seed: u64,
    pub review: Option<SavedReview>,
}

/// A session read back from disk.
#[derive(Clone, Debug)]
pub struct LoadedSession {
    pub meta: SessionMeta,
    pub hands: Vec<SavedHand>,
    /// Lines of `hands.jsonl` that couldn't be read.
    pub skipped: usize,
}

/// Unix time in seconds.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// `YYYY-MM-DD HH:MM` (UTC) of a Unix time.
pub fn format_time(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", rem / 3600, rem % 3600 / 60)
}

/// "5 min ago", "3 h ago", "2 days ago".
pub fn format_age(secs: u64, now: u64) -> String {
    let a = now.saturating_sub(secs);
    match a {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", a / 60),
        3600..=86_399 => format!("{} h ago", a / 3600),
        _ => {
            let d = a / 86_400;
            format!("{d} day{} ago", if d == 1 { "" } else { "s" })
        }
    }
}

/// Session directories are plain names, never paths.
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Where one session is saved.
#[derive(Clone, Debug)]
pub struct SessionStore {
    dir: PathBuf,
    pub id: String,
}

impl SessionStore {
    /// A new, empty session directory under `root` named after the time.
    pub fn create(root: &Path, created: u64) -> io::Result<SessionStore> {
        fs::create_dir_all(root)?;
        let base = format_time(created).replace([' ', ':'], "-");
        for k in 1.. {
            let id = if k == 1 { base.clone() } else { format!("{base}-{k}") };
            let dir = root.join(&id);
            match fs::create_dir(&dir) {
                Ok(()) => return Ok(SessionStore { dir, id }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    /// An existing session.
    pub fn open(root: &Path, id: &str) -> io::Result<SessionStore> {
        if !valid_id(id) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("bad session id {id:?}")));
        }
        Ok(SessionStore { dir: root.join(id), id: id.to_string() })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Appends hands (one JSON line each) in a single write.
    pub fn append_hands(&self, hands: &[SavedHand]) -> io::Result<()> {
        if hands.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::new();
        for h in hands {
            serde_json::to_writer(&mut buf, h).map_err(io::Error::other)?;
            buf.push(b'\n');
        }
        // No fsync: it can stall the UI for tens of milliseconds, and the OS
        // writes the data out even if the trainer crashes; after a power
        // cut, a partial last line is skipped on load.
        let mut f = fs::OpenOptions::new().create(true).append(true).open(self.dir.join(HANDS))?;
        f.write_all(&buf)
    }

    /// Replaces `meta.json` (via a temporary file, so it is never half
    /// written).
    pub fn write_meta(&self, meta: &SessionMeta) -> io::Result<()> {
        let tmp = self.dir.join(format!("{META}.tmp"));
        let text = serde_json::to_vec_pretty(meta).map_err(io::Error::other)?;
        fs::write(&tmp, text)?;
        fs::rename(tmp, self.dir.join(META))
    }

    pub fn read_meta(&self) -> Result<SessionMeta, String> {
        let path = self.dir.join(META);
        let text = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let meta: SessionMeta =
            serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.version > FORMAT_VERSION {
            return Err(format!("{}: saved by a newer version of the trainer", path.display()));
        }
        Ok(meta)
    }

    /// Reads the whole session.
    pub fn load(&self) -> Result<LoadedSession, String> {
        let mut meta = self.read_meta()?;
        meta.id = self.id.clone();
        let mut hands = Vec::new();
        let mut skipped = 0;
        let path = self.dir.join(HANDS);
        match fs::File::open(&path) {
            Ok(f) => {
                for line in io::BufReader::new(f).lines() {
                    let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<SavedHand>(&line) {
                        Ok(h) => hands.push(h),
                        Err(_) => skipped += 1,
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
        // A crash between writing hands and the meta file leaves the meta
        // behind: hand numbers continue after the last saved hand.
        if let Some(last) = hands.iter().map(|h| h.id).max() {
            meta.next_id = meta.next_id.max(last + 1);
        }
        Ok(LoadedSession { meta, hands, skipped })
    }
}

/// Saved sessions under `root`, most recently played first. Directories
/// that aren't readable sessions are left out.
pub fn list_sessions(root: &Path) -> Vec<SessionMeta> {
    let Ok(entries) = fs::read_dir(root) else { return Vec::new() };
    let mut v: Vec<SessionMeta> = entries
        .flatten()
        .filter_map(|e| {
            let id = e.file_name().to_str()?.to_string();
            let store = SessionStore::open(root, &id).ok()?;
            let mut m = store.read_meta().ok()?;
            m.id = id;
            Some(m)
        })
        .collect();
    v.sort_by(|a, b| b.updated.cmp(&a.updated).then(b.id.cmp(&a.id)));
    v
}

/// Deletes a saved session.
pub fn delete_session(root: &Path, id: &str) -> io::Result<()> {
    let store = SessionStore::open(root, id)?;
    // Only delete what looks like a session.
    if !store.dir.join(META).exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound, format!("no saved session {id:?}")));
    }
    fs::remove_dir_all(store.dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scratch directory for one test.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("poker-trainer-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn meta(id: &str, updated: u64) -> SessionMeta {
        SessionMeta {
            version: FORMAT_VERSION,
            id: id.into(),
            created: updated,
            updated,
            config: TrainerConfig::default(),
            lineup: vec![None, Profile::by_name("Reg")],
            button: 1,
            next_id: 5,
            summary: SessionSummary { hands: 4, net_chips: -250, decisions: 3, score: Some(91.5) },
        }
    }

    #[test]
    fn time_formatting() {
        assert_eq!(format_time(0), "1970-01-01 00:00");
        assert_eq!(format_time(951_782_400 + 3_723), "2000-02-29 01:02");
        assert_eq!(format_time(1_793_712_300), "2026-11-03 13:25");
        assert_eq!(format_age(100, 100), "just now");
        assert_eq!(format_age(0, 7200), "2 h ago");
        assert_eq!(format_age(0, 86_400), "1 day ago");
    }

    #[test]
    fn create_list_delete() {
        let root = scratch("list");
        let a = SessionStore::create(&root, 1_000_000).unwrap();
        let b = SessionStore::create(&root, 1_000_000).unwrap();
        assert_ne!(a.id, b.id, "same second, distinct directories");
        a.write_meta(&meta(&a.id, 10)).unwrap();
        b.write_meta(&meta(&b.id, 20)).unwrap();
        fs::create_dir_all(root.join("not-a-session")).unwrap();
        let list = list_sessions(&root);
        assert_eq!(list.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), [b.id.as_str(), a.id.as_str()]);
        assert_eq!(list[0].summary.score, Some(91.5));
        delete_session(&root, &a.id).unwrap();
        assert_eq!(list_sessions(&root).len(), 1);
        assert!(delete_session(&root, "not-a-session").is_err(), "only sessions are deleted");
        assert!(delete_session(&root, "../etc").is_err());
        assert!(root.join("not-a-session").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn truncated_last_line_is_skipped() {
        let root = scratch("truncated");
        let s = SessionStore::create(&root, 0).unwrap();
        s.write_meta(&meta(&s.id, 1)).unwrap();
        fs::write(s.dir().join(HANDS), "{\"id\": 3, \"hero\"").unwrap();
        let loaded = s.load().unwrap();
        assert!(loaded.hands.is_empty());
        assert_eq!(loaded.skipped, 1);
        assert_eq!(loaded.meta.next_id, 5);
        fs::remove_dir_all(root).unwrap();
    }
}
