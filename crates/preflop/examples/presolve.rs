//! Solves a preflop game with the model and compares the result with the
//! built-in charts for that table size.
//!
//! `cargo run --release -p preflop --example presolve -- [players] [million deals] [threads] [ip] [oop]`
//!
//! `ip` and `oop` are flat realization factors by position (default 1 and 1).

use preflop::charts::{compare, first_in_frequencies, mean_distance, to_charts};
use preflop::{GameSpec, Options, Realization, Solver, Tree};
use std::time::Instant;
use trainer::preflop::ChartSet;

fn main() {
    let arg =
        |i: usize, default: f64| std::env::args().nth(i).and_then(|a| a.parse().ok()).unwrap_or(default);
    let players = arg(1, 6.0) as usize;
    let millions = arg(2, 20.0);
    let threads = arg(3, 4.0) as usize;
    let (ip, oop) = (arg(4, 1.0) as f32, arg(5, 1.0) as f32);

    let mut spec = GameSpec::new(players).with_rake(0.05, 3.0);
    spec.rules.max_cold_callers = 1;
    let tree = Tree::build(&spec).expect("valid spec");
    println!("{players}-max: {} decisions, {} leaves", tree.num_decisions, tree.num_leaves());
    let model = Realization::positional(ip, oop, (ip + oop) / 2.0);
    let mut solver = Solver::new(tree, model, Options { threads, ..Options::default() });
    println!("{} cells ({} MB)", solver.num_cells(), solver.num_cells() * 8 / (1 << 20));

    let reference = ChartSet::builtin(players);
    let epochs = 20;
    let per_epoch = (millions * 1e6 / epochs as f64) as u64;
    let start = Instant::now();
    for epoch in 1..=epochs {
        solver.run(per_epoch);
        if epoch % 5 == 0 {
            let charts = to_charts(&solver, 200);
            let diffs = compare(solver.tree(), &charts, &reference);
            println!(
                "{:>6.1}M deals, {:>5.1} s: mean distance to the built-in charts {:.3} over {} spots",
                solver.deals() as f64 / 1e6,
                start.elapsed().as_secs_f64(),
                mean_distance(&diffs),
                diffs.len()
            );
        }
    }
    let diagnosis = solver.diagnose(2_000_000, 99);
    let positions = holdem::game::positions(players);
    let seat_of =
        |name: &str| holdem::game::position_offset(players, name).expect("a position of this table");
    println!("per position: result and gain from changing one decision, in bb/100");
    for name in &positions {
        let seat = seat_of(name);
        println!(
            "  {name:<6} {:>7.1} {:>6.2}",
            100.0 * diagnosis.ev[seat],
            100.0 * diagnosis.deviation_gain[seat]
        );
    }
    let charts = to_charts(&solver, 200);
    let ours = first_in_frequencies(solver.tree(), &charts);
    let theirs = first_in_frequencies(solver.tree(), &reference);
    let mut keys: Vec<&String> = ours.keys().collect();
    keys.sort();
    println!("first in: raise % (model / built-in)");
    for key in keys {
        let reference = theirs.get(key).map_or("-".to_string(), |f| format!("{:.1}", 100.0 * (f[2] + f[3])));
        println!("  {key:<8} {:>5.1} / {reference}", 100.0 * (ours[key][2] + ours[key][3]));
    }
    println!("most reached spots: distance, then fold/call/raise/all-in % (model | built-in)");
    for d in compare(solver.tree(), &charts, &reference).iter().take(25) {
        let pct = |f: &[f32; 4]| {
            format!("{:>3.0} {:>3.0} {:>3.0} {:>3.0}", 100.0 * f[0], 100.0 * f[1], 100.0 * f[2], 100.0 * f[3])
        };
        println!(
            "  {:<28} reach {:.4}  {:.3}  {} | {}",
            d.key,
            d.reach,
            d.distance,
            pct(&d.model),
            pct(&d.reference)
        );
    }
}
