//! Prints the size of the preflop tree for each table size and a few tree
//! rules, and how many built-in chart spots each tree covers.
//!
//! `cargo run --release -p preflop --example treesize`

use preflop::{GameSpec, Limp, Tree, TreeRules};
use std::collections::HashSet;
use trainer::preflop::ChartSet;

fn main() {
    let rules = [
        ("no limps, all-in after 2 raises", TreeRules::default()),
        ("no limps, all-in after 1 raise", TreeRules { allin_after_raises: 1, ..TreeRules::default() }),
        ("no limps, at most 2 cold callers", TreeRules { max_cold_callers: 2, ..TreeRules::default() }),
        ("no limps, at most 1 cold caller", TreeRules { max_cold_callers: 1, ..TreeRules::default() }),
        ("no limps, no cold callers", TreeRules { max_cold_callers: 0, ..TreeRules::default() }),
        ("SB limps", TreeRules { limp: Limp::SmallBlind, ..TreeRules::default() }),
    ];
    for players in [2, 6, 9] {
        let charts = ChartSet::builtin(players);
        for (name, rules) in &rules {
            let spec = GameSpec { rules: rules.clone(), ..GameSpec::new(players) };
            match Tree::build(&spec) {
                Ok(tree) => {
                    let keys: HashSet<&str> = tree.decisions().map(|d| d.key.as_str()).collect();
                    let covered = charts.charts.keys().filter(|k| keys.contains(k.as_str())).count();
                    let cells: usize = tree.decisions().map(|d| d.num_actions() * 169).sum();
                    println!(
                        "{players}-max, {name}: {} decisions, {} leaves, {} cells, {covered}/{} chart spots",
                        tree.num_decisions,
                        tree.num_leaves(),
                        cells,
                        charts.charts.len()
                    );
                }
                Err(e) => println!("{players}-max, {name}: {e}"),
            }
        }
    }
}
