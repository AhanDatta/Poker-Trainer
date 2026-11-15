//! Calibrates the realization factors with the postflop solver.
//!
//! ```text
//! cargo run --release -p preflop --features gpu --example calibrate -- <mode> [options]
//!
//!   spots                       list the spots that would be solved
//!   run <checkpoint.jsonl>      solve them, appending to the checkpoint (resumes)
//!   aggregate <checkpoint.jsonl> <factors.txt>
//!                               turn the checkpoint into a factors file
//!
//! options (spots and run):
//!   --spots N        most reached spots per type (default 4)
//!   --flops N        flops per spot (default 12)
//!   --seed N         flop seed (default 1)
//!   --chance exact|fast      exact runouts or 49 turns x 10 rivers (default exact)
//!   --backend gpu|cpu        (default gpu)
//!   --target PCT     exploitability target, % of the pot (default 0.5)
//!   --iterations N   iteration cap (default 400)
//!   --limit N        stop after N solves (default: all)
//!
//! options (all modes):
//!   --players N      table size, played by its built-in charts (default 6)
//!   --rake PCT       rake in percent, capped at 3bb (default 5)
//!   --limp never|sb  open limps in the tree (default: sb heads-up, never otherwise)
//! ```
//!
//! The game is 100bb with one cold caller per raise. Use the same table
//! options for `run` and `aggregate` as the checkpoint was made with.

use preflop::calibrate::{self, aggregate, parse_checkpoint, Chance, PostflopSetup, SpotType};
use preflop::realize::Context;
use preflop::{GameSpec, Limp, Tree};
use solver::Backend;
use std::io::Write;
use std::time::Instant;
use trainer::preflop::ChartSet;

struct Args {
    mode: String,
    files: Vec<String>,
    spots: usize,
    flops: usize,
    seed: u64,
    chance: Chance,
    backend: Backend,
    target: f64,
    iterations: u32,
    limit: usize,
    players: usize,
    rake_pct: f64,
    limp: Option<Limp>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        mode: String::new(),
        files: Vec::new(),
        spots: 4,
        flops: 12,
        seed: 1,
        chance: Chance::Exact,
        backend: Backend::Gpu,
        target: 0.5,
        iterations: 400,
        limit: usize::MAX,
        players: 6,
        rake_pct: 5.0,
        limp: None,
    };
    let mut it = std::env::args().skip(1);
    args.mode = it.next().ok_or("missing mode: spots, run or aggregate")?;
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or(format!("{a} needs a value"));
        let number = |v: String| v.parse::<f64>().map_err(|_| format!("{a}: {v} is not a number"));
        match a.as_str() {
            "--spots" => args.spots = number(value()?)? as usize,
            "--flops" => args.flops = number(value()?)? as usize,
            "--seed" => args.seed = number(value()?)? as u64,
            "--target" => args.target = number(value()?)?,
            "--iterations" => args.iterations = number(value()?)? as u32,
            "--limit" => args.limit = number(value()?)? as usize,
            "--players" => args.players = number(value()?)? as usize,
            "--rake" => args.rake_pct = number(value()?)?,
            "--limp" => {
                args.limp = Some(match value()?.as_str() {
                    "never" => Limp::Never,
                    "sb" => Limp::SmallBlind,
                    other => return Err(format!("--limp {other}: never or sb")),
                })
            }
            "--chance" => {
                args.chance = match value()?.as_str() {
                    "exact" => Chance::Exact,
                    "fast" => Chance::Sampled { turn: 49, river: 10 },
                    other => return Err(format!("--chance {other}: exact or fast")),
                }
            }
            "--backend" => {
                args.backend = match value()?.as_str() {
                    "gpu" => Backend::Gpu,
                    "cpu" => Backend::Cpu,
                    other => return Err(format!("--backend {other}: gpu or cpu")),
                }
            }
            _ if a.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => args.files.push(a),
        }
    }
    Ok(args)
}

impl Args {
    fn spec(&self) -> Result<GameSpec, String> {
        let mut spec = GameSpec::new(self.players).with_rake(self.rake_pct / 100.0, 3.0);
        spec.rules.max_cold_callers = 1;
        spec.rules.limp = self.limp.unwrap_or(if self.players == 2 { Limp::SmallBlind } else { Limp::Never });
        spec.validate().map_err(|e| e.to_string())?;
        Ok(spec)
    }

    /// The game in words, for the factors file's header.
    fn game(&self) -> String {
        let table = if self.players == 2 { "heads-up".to_string() } else { format!("{}-max", self.players) };
        format!("{table} 100bb, {}% rake capped at 3bb", self.rake_pct)
    }
}

fn main() {
    if let Err(e) = parse_args().and_then(|args| match args.mode.as_str() {
        "spots" => list_spots(&args),
        "run" => run(&args),
        "aggregate" => aggregate_file(&args),
        other => Err(format!("unknown mode {other}: spots, run or aggregate")),
    }) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn chosen_spots(args: &Args) -> Result<Vec<calibrate::Spot>, String> {
    let tree = Tree::build(&args.spec()?).map_err(|e| e.to_string())?;
    let all = calibrate::heads_up_flops(&tree, &ChartSet::builtin(args.players));
    Ok(calibrate::pick(&all, args.spots))
}

fn list_spots(args: &Args) -> Result<(), String> {
    let tree = Tree::build(&args.spec()?).map_err(|e| e.to_string())?;
    let all = calibrate::heads_up_flops(&tree, &ChartSet::builtin(args.players));
    for t in SpotType::all() {
        let of_type: Vec<_> = all.iter().filter(|s| s.kind() == Some(t)).collect();
        let total: f64 = of_type.iter().map(|s| s.reach).sum();
        println!("{} ({} spots, reach {:.4}):", t.name(), of_type.len(), total);
        for s in of_type.iter().take(args.spots) {
            println!(
                "  {:<28} reach {:.4} ({:>4.1}% of type)  pot {:>5.1}bb  behind {:>5.1}bb",
                s.line,
                s.reach,
                100.0 * s.reach / total,
                s.pot as f64 / 100.0,
                s.behind as f64 / 100.0
            );
        }
    }
    Ok(())
}

fn setup(args: &Args) -> Result<PostflopSetup, String> {
    Ok(PostflopSetup {
        chance: args.chance,
        backend: args.backend,
        target_exploitability_pct: args.target,
        max_iterations: args.iterations,
        ..PostflopSetup::review(&args.spec()?)
    })
}

fn run(args: &Args) -> Result<(), String> {
    let [path] = &args.files[..] else { return Err("run needs one checkpoint file".into()) };
    let text = std::fs::read_to_string(path).or_else(|e| match e.kind() {
        std::io::ErrorKind::NotFound => Ok(String::new()),
        _ => Err(format!("{path}: {e}")),
    })?;
    let (done, complete) = parse_checkpoint(&text)?;
    if complete < text.len() {
        eprintln!("dropping a cut-off last line of {path}");
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.set_len(complete as u64).map_err(|e| e.to_string())?;
    let mut file = std::fs::OpenOptions::new().append(true).open(path).map_err(|e| e.to_string())?;

    let spots = chosen_spots(args)?;
    let jobs = calibrate::plan(&spots, args.flops, args.seed);
    let total = jobs.len();
    let mut jobs = calibrate::pending(&spots, jobs, &done);
    jobs.truncate(args.limit);
    println!(
        "{} spots x {} flops = {total} solves; {} done, {} to run now",
        spots.len(),
        args.flops,
        done.len(),
        jobs.len()
    );
    let setup = setup(args)?;
    let start = Instant::now();
    let mut count = 0;
    let saved = calibrate::run(
        &spots,
        &jobs,
        &setup,
        |r| {
            writeln!(file, "{}", r.to_line()).and_then(|()| file.flush()).map_err(|e| e.to_string())?;
            count += 1;
            let elapsed = start.elapsed().as_secs_f64();
            let eta = elapsed / count as f64 * (jobs.len() - count) as f64;
            println!(
                "[{}/{}] {:<26} {}  {:>3} it  {:.2}%  {:>5.1} s on {}{}  (eta {:.0} min)",
                done.len() + count,
                total,
                r.spot,
                r.flop,
                r.iterations,
                r.exploitability_pct,
                r.seconds,
                r.solved_on,
                r.gpu_error.as_ref().map_or(String::new(), |e| format!(" ({e})")),
                eta / 60.0
            );
            Ok(())
        },
        |job, why| {
            eprintln!("failed: {} on {}: {why}", spots[job.spot].line, calibrate::flop_name(&job.flop))
        },
    )?;
    println!("{saved} solves saved in {:.1} min", start.elapsed().as_secs_f64() / 60.0);
    Ok(())
}

fn aggregate_file(args: &Args) -> Result<(), String> {
    let [input, output] = &args.files[..] else {
        return Err("aggregate needs the checkpoint and the output file".into());
    };
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let (records, _) = parse_checkpoint(&text)?;
    let calibration = aggregate(&records)?;
    let r = &calibration.realization;
    println!("context: measured by; mean factor, AA KK AKs T9s 72o (with standard errors)");
    for ctx in Context::all() {
        let source = calibration.source[ctx.index()];
        let show = |class: &str| {
            let c = class.parse::<holdem::HandClass>().expect("a class").grid_index();
            format!("{class} {:.3}±{:.3}", r.factor(ctx.index(), c), calibration.stderr[ctx.index()][c])
        };
        if matches!(source, calibrate::Source::Measured(_)) {
            let mean = r.factors(ctx).iter().sum::<f32>() / 169.0;
            println!(
                "  [{}] {source}; mean {mean:.3}  {}",
                ctx.name(),
                ["AA", "KK", "AKs", "T9s", "72o"].map(show).join("  ")
            );
            let errors = &calibration.stderr[ctx.index()];
            let mut sorted: Vec<usize> = (0..169).collect();
            sorted.sort_by(|&a, &b| errors[a].total_cmp(&errors[b]));
            let worst: Vec<String> = sorted[166..]
                .iter()
                .rev()
                .map(|&c| {
                    format!(
                        "{} {:.3}±{:.3}",
                        holdem::HandClass::from_grid_index(c),
                        r.factor(ctx.index(), c),
                        errors[c]
                    )
                })
                .collect();
            println!(
                "      standard error: median {:.3}, 90% {:.3}, max {:.3}; noisiest {}",
                errors[sorted[84]],
                errors[sorted[152]],
                errors[sorted[168]],
                worst.join(", ")
            );
        }
    }
    let header = format!(
        "Realization factors for {}, measured with\n\
         heads-up postflop solves of the built-in chart ranges for that table\n\
         ({} solves; generated by `cargo run -p preflop --example calibrate`).\n\
         Each factor is what a hand class wins with betting over what it wins checking\n\
         down, so 1 is its equity share of the pot after the preflop pot's rake.\n\
         Sources per context:\n{}",
        args.game(),
        records.len(),
        Context::all()
            .map(|c| format!("  [{}] {}", c.name(), calibration.source[c.index()]))
            .collect::<Vec<_>>()
            .join("\n")
    );
    std::fs::write(output, r.to_text(&header)).map_err(|e| format!("{output}: {e}"))?;
    println!("wrote {output}");
    Ok(())
}
