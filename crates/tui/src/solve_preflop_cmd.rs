//! The `solve-preflop` command.

use anyhow::{anyhow, Context, Result};
use preflop::generate::{self, Job};
use preflop::{GameSpec, Realization};

/// Most deals `--deals` accepts, in millions.
pub const MAX_MILLION_DEALS: f64 = 100_000.0;

/// Settings of `solve-preflop`.
pub struct Settings<'a> {
    pub players: usize,
    pub stack_bb: f64,
    /// Percent of the pot.
    pub rake_pct: f64,
    pub rake_cap_bb: f64,
    /// Millions of deals.
    pub million_deals: f64,
    /// 0 = every core.
    pub threads: usize,
    pub seed: u64,
    pub cold_callers: usize,
    /// A factors file, or the built-in measured factors.
    pub factors: Option<&'a str>,
    pub out: &'a str,
}

/// Checks the settings and reads the factors. Errors on bad settings or an
/// unreadable or invalid factors file.
pub fn job(s: &Settings) -> Result<Job> {
    if !(0.0..=MAX_MILLION_DEALS).contains(&s.million_deals) {
        return Err(anyhow!("--deals {}: millions of deals, from 0 to {MAX_MILLION_DEALS}", s.million_deals));
    }
    let mut spec =
        GameSpec::new(s.players).with_stack(s.stack_bb).with_rake(s.rake_pct / 100.0, s.rake_cap_bb);
    spec.rules.max_cold_callers = s.cold_callers;
    spec.validate().map_err(|e| anyhow!(e))?;
    let (factors, factors_name) = match s.factors {
        Some(path) => {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
            (Realization::parse(&text).map_err(|e| anyhow!("{path}: {e}"))?, path.to_string())
        }
        None => (Realization::builtin(), Realization::BUILTIN_NAME.to_string()),
    };
    let threads =
        if s.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { s.threads };
    Ok(Job {
        spec,
        factors,
        factors_name,
        deals: (s.million_deals * 1e6).round() as u64,
        threads,
        seed: s.seed,
    })
}

/// Solves the preflop game with the model and writes its charts.
pub fn run(s: &Settings) -> Result<()> {
    let job = job(s)?;
    println!(
        "Solving {}-handed preflop, {}bb, rake {}% capped at {}bb, {} million deals on {} threads...",
        s.players, s.stack_bb, s.rake_pct, s.rake_cap_bb, s.million_deals, job.threads
    );
    let out = generate::solve(&job, |p| {
        let left = job.deals.saturating_sub(p.deals) as f64;
        let eta = if p.deals > 0 { p.seconds / p.deals as f64 * left } else { 0.0 };
        println!("  {:>6.1}M deals, {:>6.0} s (about {:.0} s left)", p.deals as f64 / 1e6, p.seconds, eta);
    })
    .map_err(|e| anyhow!(e))?;
    let header = generate::header(&job, &out);
    println!("{header}");
    std::fs::write(s.out, out.charts.to_text(&header)).with_context(|| format!("writing {}", s.out))?;
    println!("Wrote {} spots to {}", out.charts.charts.len(), s.out);
    Ok(())
}

#[cfg(test)]
mod tests;
