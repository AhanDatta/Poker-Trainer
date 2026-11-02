//! Headless mode: a scripted hero plays hands and the solver review is
//! printed. Handy for checking solver speed on your machine.

use anyhow::{anyhow, Result};
use holdem::Action;
use std::time::{Duration, Instant};
use trainer::fmt::bb;
use trainer::{Session, StepResult, TrainerConfig};

pub fn run(cfg: TrainerConfig, hands: usize) -> Result<()> {
    let mut session = Session::new(cfg).map_err(|e| anyhow!(e))?;
    let mut done = 0;
    let mut attempts = 0;
    while done < hands && attempts < hands * 50 {
        attempts += 1;
        let mut hand = session.new_hand();
        let mut k = 0usize;
        while hand.run_until_hero() == StepResult::HeroToAct {
            let acts = hand.hero_actions();
            // Mostly passive, sometimes the smallest bet/raise.
            let passive = acts.iter().copied().find(|a| matches!(a, Action::Check | Action::Call));
            let raise = acts.iter().copied().find(|a| matches!(a, Action::Raise(_)));
            let a = if k % 3 == 1 { raise.or(passive) } else { passive.or(raise) }.unwrap_or(acts[0]);
            k += 1;
            hand.hero_act(a).map_err(|e| anyhow!(e))?;
        }
        let Some(review) = hand.review() else {
            session.finish_hand(hand);
            continue;
        };
        let h = hand.state.hole[hand.hero];
        println!(
            "\nHand #{}: you ({}) held {}{} vs {}",
            hand.id,
            hand.state.position_name(hand.hero),
            h[0],
            h[1],
            hand.opponents_label()
        );
        for l in &hand.log {
            println!("   {}", l.text);
        }
        let t = Instant::now();
        while !review.is_finished() {
            std::thread::sleep(Duration::from_millis(100));
        }
        let st = review.status();
        println!("   review finished {:.1}s after the hand ({})", t.elapsed().as_secs_f64(), st.stage);
        if let Some(e) = &st.error {
            println!("   error: {e}");
        }
        for n in &st.notes {
            println!("   note: {n}");
        }
        for d in &st.decisions {
            println!(
                "   [{}] {} — you: {} → {} {} (−{}, {:.1}% pot; best: {})",
                d.variant.label(),
                d.street,
                d.labels[d.chosen],
                d.verdict.symbol(),
                d.verdict.label(),
                bb(d.ev_loss.round() as i64),
                d.ev_loss_pct,
                d.labels[d.best]
            );
        }
        session.finish_hand(hand);
        done += 1;
    }
    let s = session.stats();
    println!(
        "\n{} hands, net {}, {} decisions graded, accuracy {:.0}%, avg loss {:.1}% pot",
        s.hands,
        bb(s.net_chips),
        s.decisions,
        100.0 * s.accuracy(),
        s.avg_loss_pct()
    );
    Ok(())
}
