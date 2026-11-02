//! Plays hands with a scripted hero and prints the log and solver review.
//! Usage: cargo run --release -p trainer --example demo -- [seed] [style]

#![allow(clippy::field_reassign_with_default, clippy::while_let_loop)]

use holdem::Action;
use std::time::Duration;
use trainer::fmt::bb;
use trainer::{Session, StepResult, TrainerConfig};

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let style = std::env::args().nth(2).unwrap_or_else(|| "LAG".into());
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(seed);
    cfg.opponent = style;
    let mut session = Session::new(cfg).unwrap();
    for _ in 0..20 {
        let mut hand = session.new_hand();
        let mut n = 0;
        loop {
            match hand.run_until_hero() {
                StepResult::HeroToAct => {
                    let actions = hand.hero_actions();
                    let pick = actions
                        .iter()
                        .copied()
                        .filter(|a| !matches!(a, Action::Fold))
                        .nth(n % 2)
                        .unwrap_or(actions[0]);
                    n += 1;
                    hand.hero_act(pick).unwrap();
                }
                _ => break,
            }
        }
        let h = hand.state.hole[hand.hero];
        println!(
            "\n=== Hand #{} — you hold {} {} vs {} ===",
            hand.id,
            h[0],
            h[1],
            hand.villain_profile().unwrap().name
        );
        for l in &hand.log {
            println!("  {}", l.text);
        }
        println!("  result: {}", bb(hand.hero_net().unwrap()));
        if let Some(r) = hand.review() {
            while !r.is_finished() {
                std::thread::sleep(Duration::from_millis(200));
            }
            let st = r.status();
            for note in &st.notes {
                println!("  note: {note}");
            }
            for d in &st.decisions {
                println!(
                    "  [{}] {} {} | {} | you: {} -> {} {} (loss {:.2}bb = {:.1}% pot, expl {:.2}%)",
                    d.variant.label(),
                    d.street,
                    d.board.iter().map(|c| c.to_string()).collect::<String>(),
                    if d.history.is_empty() { "first to act".to_string() } else { d.history.clone() },
                    d.labels[d.chosen],
                    d.verdict.label(),
                    d.verdict.symbol(),
                    d.ev_loss / 100.0,
                    d.ev_loss_pct,
                    d.exploitability_pct
                );
                for (i, l) in d.labels.iter().enumerate() {
                    println!(
                        "      {:<22} mix {:5.1}%  EV {:>8}   range {:5.1}%",
                        l,
                        100.0 * d.hero_strategy[i],
                        bb(d.hero_ev[i].round() as i64),
                        100.0 * d.range_strategy[i]
                    );
                }
                for note in &d.notes {
                    println!("      note: {note}");
                }
            }
            if !st.decisions.is_empty() {
                session.finish_hand(hand);
                break;
            }
        }
        session.finish_hand(hand);
    }
}
