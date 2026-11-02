//! Prints aggregate action frequencies of a chart file.
//! Usage: cargo run -p trainer --example chartstats -- <file> <players> [key...]
use holdem::combo::HandClass;
use trainer::preflop::ChartSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let players: usize = args[2].parse().unwrap();
    let cs = ChartSet::parse(&text, players).unwrap();
    let mut keys: Vec<String> =
        if args.len() > 3 { args[3..].to_vec() } else { cs.charts.keys().cloned().collect() };
    keys.sort();
    for key in keys {
        let Some(chart) = cs.get(&key) else {
            println!("{key}: missing");
            continue;
        };
        let mut tot = [0f32; 4];
        for c in HandClass::all() {
            let f = chart.class_freqs(c.grid_index());
            for a in 0..4 {
                tot[a] += f[a] * c.num_combos() as f32 / 1326.0;
            }
        }
        println!(
            "{key:<28} fold {:5.1}%  call {:5.1}%  raise {:5.1}%  allin {:5.1}%",
            tot[0] * 100.0,
            tot[1] * 100.0,
            tot[2] * 100.0,
            tot[3] * 100.0
        );
    }
}
