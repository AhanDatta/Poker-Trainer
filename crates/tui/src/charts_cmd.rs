//! `import-charts` and `fetch-charts` commands.

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use trainer::charts_import::{import_leaf_tree, import_line_files};
use trainer::TrainerConfig;

/// Range collections fetched by `fetch-charts`.
const REPO: &str = "https://github.com/bupticybee/TexasSolverGPU";
const HU_DIR: &str = "ranges/headsup/100bb";
const SIX_MAX_DIR: &str = "ranges/qb_ranges/100bb 2.5x 500rake";

/// Imports a directory of ranges and writes it in the trainer's chart format.
pub fn import(dir: &str, format: &str, players: usize, allin_bb: f64, out: &str) -> Result<()> {
    let path = Path::new(dir);
    let charts = match format {
        "lines" => import_line_files(path, players, allin_bb),
        "tree" => import_leaf_tree(path, allin_bb),
        other => return Err(anyhow!("unknown format {other:?} (use \"lines\" or \"tree\")")),
    }
    .map_err(|e| anyhow!(e))?;
    let header = format!(
        "Imported by `poker-trainer import-charts` from {dir}\n({} spots, {players}-handed)",
        charts.charts.len()
    );
    std::fs::write(out, charts.to_text(&header)).with_context(|| format!("writing {out}"))?;
    println!("Wrote {} spots to {out}", charts.charts.len());
    Ok(())
}

fn git(args: &[&str], cwd: Option<&Path>) -> Result<()> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let status = cmd.status().context("running git (is it installed?)")?;
    if !status.success() {
        bail!("git {} failed", args.join(" "));
    }
    Ok(())
}

/// Downloads public range collections (sparse git checkout, ranges only),
/// imports them and points the config at the results.
pub fn fetch(dest: &str, config_path: &str) -> Result<()> {
    let dest = PathBuf::from(dest);
    std::fs::create_dir_all(&dest)?;
    let checkout = dest.join("source");
    if !checkout.join(".git").exists() {
        println!("Downloading ranges from {REPO} (sparse checkout of the ranges folder)...");
        git(
            &["clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, checkout.to_str().unwrap()],
            None,
        )?;
    }
    git(&["sparse-checkout", "set", HU_DIR, SIX_MAX_DIR], Some(&checkout))?;
    let hu_out = dest.join("hu_100bb.txt");
    let six_out = dest.join("6max_100bb.txt");
    import(checkout.join(HU_DIR).to_str().unwrap(), "tree", 2, 90.0, hu_out.to_str().unwrap())?;
    import(checkout.join(SIX_MAX_DIR).to_str().unwrap(), "lines", 6, 90.0, six_out.to_str().unwrap())?;
    let mut cfg = if Path::new(config_path).exists() {
        TrainerConfig::load(config_path).map_err(|e| anyhow!(e))?
    } else {
        TrainerConfig::default()
    };
    cfg.charts_hu = Some(hu_out.to_string_lossy().into_owned());
    cfg.charts_6max = Some(six_out.to_string_lossy().into_owned());
    std::fs::write(config_path, cfg.to_toml())?;
    println!("Updated {config_path}: charts_hu and charts_6max now point to the imported charts.");
    println!(
        "Sources: {REPO} ({HU_DIR}; {SIX_MAX_DIR}). Check that repository's terms before redistributing the data."
    );
    Ok(())
}
