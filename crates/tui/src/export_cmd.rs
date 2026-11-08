//! The `export` command.

use anyhow::{anyhow, Result};
use std::path::Path;
use trainer::export::{self, DECISIONS_FILE, HANDS_FILE};
use trainer::TrainerConfig;

/// Writes the saved sessions as CSV and prints bb/100 by format.
pub fn run(cfg: &TrainerConfig, out: &str, session: Option<&str>) -> Result<()> {
    let root = cfg
        .sessions_root()
        .ok_or_else(|| anyhow!("no sessions directory: set sessions_dir in the config"))?;
    let report = export::export(&root, session, Path::new(out)).map_err(|e| anyhow!(e))?;
    for e in &report.unreadable {
        eprintln!("Skipped a session that couldn't be read: {e}");
    }
    if report.skipped_lines > 0 {
        eprintln!("Skipped {} unreadable hand lines", report.skipped_lines);
    }
    let dir = Path::new(out);
    println!(
        "Wrote {} hands from {} sessions to {}",
        report.hands,
        report.sessions,
        dir.join(HANDS_FILE).display()
    );
    println!("Wrote {} graded decisions to {}", report.decisions, dir.join(DECISIONS_FILE).display());
    if report.hands > 0 {
        println!("\n{}", export::summary_table(&report.totals));
    }
    Ok(())
}
