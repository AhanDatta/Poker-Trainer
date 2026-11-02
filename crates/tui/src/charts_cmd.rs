//! The `import-charts` command.

use anyhow::{anyhow, Context, Result};
use std::path::Path;
use trainer::charts_import::{import_leaf_tree, import_line_files, Import};

/// Imports a directory of ranges and writes it in the trainer's chart format.
pub fn import(dir: &str, format: &str, players: usize, allin_bb: f64, out: &str) -> Result<()> {
    let path = Path::new(dir);
    let Import { charts, dropped } = match format {
        "lines" => import_line_files(path, players, allin_bb),
        "tree" => import_leaf_tree(path, allin_bb),
        other => return Err(anyhow!("unknown format {other:?} (use \"lines\" or \"tree\")")),
    }
    .map_err(|e| anyhow!(e))?;
    let mut header = format!(
        "Imported by `poker-trainer import-charts` from {dir}\n({} spots, {players}-handed)",
        charts.charts.len()
    );
    if !dropped.is_empty() {
        header.push_str(&format!(
            "\nLeft out (built-in charts or position-aware rules apply there):\n  {}",
            dropped.join("\n  ")
        ));
        println!("Left out {} spots:\n  {}", dropped.len(), dropped.join("\n  "));
    }
    std::fs::write(out, charts.to_text(&header)).with_context(|| format!("writing {out}"))?;
    println!("Wrote {} spots to {out}", charts.charts.len());
    Ok(())
}
