//! Simulates all-AI tables and prints HUD-style stats per AI style.
//!
//! Usage:
//!   cargo run --release -p trainer --example population -- [hands] [style,style,...]
//!   cargo run --release -p trainer --example population -- [hands] game:<tough|typical|soft|wild> [players]
//! Default: 6-max with one of each preset style.

use rand::Rng;
use trainer::population::{simulate, simulate_lineups, StyleStats};
use trainer::preflop::ChartSet;
use trainer::{GameType, Profile};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hands: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let presets = Profile::presets();
    let find = |name: &str| {
        presets.iter().find(|p| p.name.eq_ignore_ascii_case(name.trim())).cloned().expect("style")
    };
    let t = std::time::Instant::now();
    let (stats, players) = match args.get(2).map(|s| s.as_str()) {
        Some(g) if g.starts_with("game:") => {
            let game: GameType = match &g[5..] {
                "tough" => GameType::Tough,
                "soft" => GameType::Soft,
                "wild" => GameType::Wild,
                _ => GameType::Typical,
            };
            let players: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(6);
            let weights = game.weights();
            let total: f64 = weights.iter().map(|w| w.1).sum();
            let mut seats: Vec<trainer::Profile> = Vec::new();
            let lineup = |i: usize, rng: &mut rand::rngs::StdRng| {
                // A new lineup every 50 hands.
                if i.is_multiple_of(50) || seats.is_empty() {
                    seats = (0..players)
                        .map(|_| {
                            let mut x = rng.random::<f64>() * total;
                            for &(name, w) in weights {
                                if x < w {
                                    return find(name);
                                }
                                x -= w;
                            }
                            find(weights[0].0)
                        })
                        .collect();
                }
                seats.clone()
            };
            (simulate_lineups(lineup, &ChartSet::builtin(players), hands, 1), players)
        }
        Some(list) => {
            let seats: Vec<Profile> = list.split(',').map(find).collect();
            let n = seats.len();
            (simulate(&seats, &ChartSet::builtin(n), hands, 1), n)
        }
        None => (simulate(&presets[..6], &ChartSet::builtin(6), hands, 1), 6),
    };
    println!("{} hands, {}-handed, {:.1}s", hands, players, t.elapsed().as_secs_f64());
    println!("{}", StyleStats::header());
    for (name, s) in &stats {
        println!("{}", s.row(name));
    }
    let total_mw: u64 = stats.values().map(|s| s.multiway_flops).sum();
    println!("\nshare of multiway-flop seats (3+ players):");
    for (name, s) in &stats {
        println!("  {name:<16} {:5.1}%", 100.0 * s.multiway_flops as f64 / total_mw.max(1) as f64);
    }
}
