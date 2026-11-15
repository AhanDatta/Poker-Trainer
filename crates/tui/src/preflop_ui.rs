//! The review of the hero's preflop decisions: each one graded against the
//! charts, with the chart's mix for the hero's hand and the range at the
//! spot.

use crate::widgets::{action_color, bar, cards_spans, draw_grid, legend, truncate, GridCell, ACCENT};
use holdem::combo::NUM_CLASSES;
use holdem::{Action, Card, HandClass};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use trainer::preflop::{chart_policy, describe, line_range, ChartSet, Freqs, PreAction, PreDecision};
use trainer::preflop_grade::PreflopGrade;
use trainer::{LiveHand, Verdict};

/// One graded (or ungraded) hero preflop decision, with what the review
/// shows of it.
#[derive(Clone, Debug)]
pub struct PreflopItem {
    pub grade: PreflopGrade,
    pub decision: PreDecision,
    pub hole: [Card; 2],
    /// The available actions, in [`PreAction::ALL`] order, with their engine
    /// actions (for colors).
    pub actions: Vec<(PreAction, Action)>,
    /// The hero's range when it got here, under the charts (class weights).
    pub range: [f32; NUM_CLASSES],
    /// The charts' strategy at the spot for every class.
    pub policy: Vec<Freqs>,
}

/// The hero's preflop decisions of `hand`, in order.
pub fn preflop_items(hand: &LiveHand, charts: &ChartSet) -> Vec<PreflopItem> {
    let hole = hand.state.hole[hand.hero];
    hand.preflop_grades()
        .iter()
        .filter_map(|g| {
            let d = hand.preflop.get(g.decision)?.clone();
            let actions = PreAction::ALL.iter().filter_map(|&a| Some((a, d.avail.concrete(a)?))).collect();
            let (range, _) = line_range(&hand.preflop[..g.decision], hand.hero, |x| chart_policy(charts, x));
            Some(PreflopItem {
                grade: g.clone(),
                hole,
                actions,
                range: range.class_weights(),
                policy: chart_policy(charts, &d),
                decision: d,
            })
        })
        .collect()
}

/// "Raise", "Call/limp", ... for an abstract action at a decision.
pub fn action_label(a: PreAction, d: &PreDecision) -> String {
    let w = describe(a, &d.avail);
    let mut c = w.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// The line before the decision in words: `"UTG:R HJ:C>BTN"` gives
/// `"UTG raises · HJ calls"` (a call before any raise is a limp); the first
/// decision gives `"Folded to you"`.
pub fn line_words(key: &str) -> String {
    let before = key.split_once('>').map_or("", |(b, _)| b);
    if before.trim().is_empty() {
        return "Folded to you".to_string();
    }
    // A call before anyone raised is a limp.
    let mut raised = false;
    before
        .split_whitespace()
        .map(|t| {
            let (pos, code) = t.rsplit_once(':').unwrap_or((t, ""));
            let verb = match code {
                "R" => "raises",
                "C" if raised => "calls",
                "C" => "limps",
                "A" => "goes all-in",
                "F" => "folds",
                _ => "acts",
            };
            raised |= matches!(code, "R" | "A");
            format!("{pos} {verb}")
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// What the verdict line says about the chart's frequency of the choice.
pub fn verdict_text(item: &PreflopItem) -> String {
    let Some(f) = item.grade.chosen_freq() else {
        return "No chart covers this spot (only rough fallback rules), so it isn't graded.".to_string();
    };
    let main = item.grade.chart_action().map(|a| action_label(a, &item.decision)).unwrap_or_default();
    if f <= 0.0 {
        format!("the chart never does this here ({main} instead)")
    } else if f >= 0.995 {
        "the chart always does this here".to_string()
    } else {
        format!("the chart does this {:.0}% of the time here (mostly: {main})", 100.0 * f)
    }
}

/// A list row: `("Pre", "Raise", Some(verdict))`.
pub fn row(item: &PreflopItem) -> (String, Option<Verdict>) {
    (action_label(item.grade.chosen, &item.decision), item.grade.verdict)
}

/// The decision's details: the spot, your hand and choice, its grade, the
/// chart's mix for your hand, and the chart's strategy for your range.
pub fn draw_preflop_decision(
    f: &mut Frame,
    area: Rect,
    item: &PreflopItem,
    verdict_color: fn(Verdict) -> Color,
) {
    let block = Block::default().borders(Borders::ALL).title(" Preflop — against the charts ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    const TEXT_MIN: u16 = 45;
    let side_by_side = inner.width > TEXT_MIN + 52;
    let parts = if side_by_side {
        let grid_w = (inner.width / 2).clamp(52, inner.width - TEXT_MIN - 1);
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(TEXT_MIN), Constraint::Length(1), Constraint::Length(grid_w)])
            .split(inner)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(6), Constraint::Length(0), Constraint::Length(15)])
            .split(inner)
    };
    let text_w = parts[0].width as usize;
    let bar_w = text_w.saturating_sub(28 + 14).min(10);
    let label_w = text_w.saturating_sub(28 + bar_w).clamp(10, 19);
    let engine: Vec<Action> = item.actions.iter().map(|(_, a)| *a).collect();

    let mut lines = vec![
        Line::from(format!("{} — {} to act", line_words(&item.decision.key), item.decision.position)),
        Line::from(Span::styled(
            format!("Chart spot [{}]", item.decision.key),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];
    let mut you = vec![Span::raw("You ")];
    you.extend(cards_spans(&item.hole));
    you.push(Span::raw(format!(" ({}) chose ", HandClass::from_grid_index(item.grade.class))));
    you.push(Span::styled(
        action_label(item.grade.chosen, &item.decision),
        Style::default().add_modifier(Modifier::BOLD),
    ));
    lines.push(Line::from(you));
    lines.push(Line::from(match item.grade.verdict {
        Some(v) => vec![
            Span::styled(
                format!("{} {}", v.symbol(), v.label()),
                Style::default().fg(verdict_color(v)).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("  {}", verdict_text(item))),
        ],
        None => vec![Span::styled(verdict_text(item), Style::default().fg(Color::DarkGray))],
    }));
    if let Some(chart) = item.grade.chart {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Chart, your hand:", Style::default().fg(ACCENT))));
        let main = item.grade.chart_action();
        for (i, (a, _)) in item.actions.iter().enumerate() {
            let mark = if *a == item.grade.chosen {
                "◀ you"
            } else if Some(*a) == main {
                "★ chart"
            } else {
                ""
            };
            let label = action_label(*a, &item.decision);
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {:<w$}", truncate(&label, label_w - 1), w = label_w),
                    Style::default().fg(action_color(&engine, i)),
                ),
                Span::styled(bar(chart[a.index()], bar_w), Style::default().fg(action_color(&engine, i))),
                Span::raw(format!(" {:5.1}% {mark}", 100.0 * chart[a.index()])),
            ]));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Preflop grades say how often the chart plays your action with your hand (Best: its main \
         action or at least half the time; Good 15%+; ?! 5%+; ? 1%+; ?? less), not EV lost.",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), parts[0]);

    let cells: Vec<GridCell> = (0..NUM_CLASSES)
        .map(|c| GridCell {
            fill: item.range[c].clamp(0.0, 1.0),
            mix: item
                .actions
                .iter()
                .enumerate()
                .map(|(i, (a, _))| (item.policy[c][a.index()], action_color(&engine, i)))
                .collect(),
            sub: None,
        })
        .collect();
    let entries: Vec<(String, Color)> = item
        .actions
        .iter()
        .enumerate()
        .map(|(i, (a, _))| (action_label(*a, &item.decision), action_color(&engine, i)))
        .collect();
    let title =
        Line::from(Span::styled("Your range here, colored by the chart's mix:", Style::default().fg(ACCENT)));
    draw_grid(f, parts[2], title, &cells, legend(&entries, 16), Some(item.grade.class), None);
}

#[cfg(test)]
mod tests;
