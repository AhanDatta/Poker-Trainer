//! Display helpers (chip amounts in big blinds, action labels).

use holdem::{Action, Chips};

/// Chips per big blind used throughout the trainer.
pub const BB: Chips = 100;

/// Formats chips as big blinds, e.g. `2.5bb`.
pub fn bb(chips: Chips) -> String {
    let v = chips as f64 / BB as f64;
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s}bb")
}

/// Context needed to label an action.
#[derive(Clone, Copy, Debug)]
pub struct ActionContext {
    pub to_call: Chips,
    /// Pot before the action (including current street bets).
    pub pot: Chips,
    /// Raise-to amount that puts the player all-in.
    pub all_in_to: Chips,
    /// Whether a bet has been made on this street (raise vs bet wording).
    pub facing_bet: bool,
    pub preflop: bool,
}

/// Short label such as "Bet 2bb (33%)", "Raise to 9bb", "Call 1.5bb".
pub fn action_label(a: Action, ctx: &ActionContext) -> String {
    match a {
        Action::Fold => "Fold".into(),
        Action::Check => "Check".into(),
        Action::Call => format!("Call {}", bb(ctx.to_call)),
        Action::Raise(to) => {
            let all_in = to >= ctx.all_in_to;
            if !ctx.facing_bet && !ctx.preflop {
                let pct = if ctx.pot > 0 { 100.0 * to as f64 / ctx.pot as f64 } else { 0.0 };
                if all_in {
                    format!("All-in {} ({pct:.0}%)", bb(to))
                } else {
                    format!("Bet {} ({pct:.0}%)", bb(to))
                }
            } else if all_in {
                format!("All-in {}", bb(to))
            } else {
                format!("Raise to {}", bb(to))
            }
        }
    }
}

/// Past-tense description for the hand log, e.g. "bets 2bb".
pub fn action_verb(a: Action, ctx: &ActionContext) -> String {
    match a {
        Action::Fold => "folds".into(),
        Action::Check => "checks".into(),
        Action::Call => format!("calls {}", bb(ctx.to_call)),
        Action::Raise(to) => {
            let all_in = if to >= ctx.all_in_to { " (all-in)" } else { "" };
            if !ctx.facing_bet && !ctx.preflop {
                format!("bets {}{all_in}", bb(to))
            } else {
                format!("raises to {}{all_in}", bb(to))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bb_format() {
        assert_eq!(bb(250), "2.5bb");
        assert_eq!(bb(1000), "10bb");
        assert_eq!(bb(165), "1.65bb");
        assert_eq!(bb(0), "0bb");
    }
}
