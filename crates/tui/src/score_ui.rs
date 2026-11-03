//! The frequency-score and saved-sessions screens.

use crate::app::App;
use crate::ui::title_bar;
use crate::widgets::{truncate, ACCENT};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use trainer::config::GameType;
use trainer::fmt::bb;
use trainer::save;
use trainer::scoring::{CategoryCell, FrequencyReport, Measure, SpotRow};

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn z_text(z: f64) -> String {
    if z.is_infinite() {
        "z ∞".into()
    } else {
        format!("z {z:+.1}")
    }
}

/// Words for the aggression measure.
fn aggression_words(m: &Measure) -> &'static str {
    if m.z.abs() < 2.0 {
        "balanced: no clear lean either way"
    } else if m.value > 0.0 {
        "too aggressive: you pick the more aggressive options more than the mixes do"
    } else {
        "too passive: you pick the more passive options more than the mixes do"
    }
}

fn mixing_words(m: &Measure) -> &'static str {
    if m.z.abs() < 2.0 {
        "mixes like the solver"
    } else if m.value < 1.0 {
        "too pure: you take the solver's main option more often than its mix says"
    } else {
        "too loose: you take the solver's rare options more often than its mix says"
    }
}

/// Style and arrow of a category cell: red ▲ too often, blue ▼ too rarely.
fn cell_style(c: &CategoryCell) -> (Style, &'static str) {
    if c.z >= 2.0 {
        (Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD), "▲")
    } else if c.z <= -2.0 {
        (Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD), "▼")
    } else {
        (Style::default(), " ")
    }
}

fn decisions(n: usize) -> String {
    format!("{n} decision{}", if n == 1 { "" } else { "s" })
}

fn p_text(p: f64) -> String {
    if p < 0.001 {
        "< 0.001".into()
    } else {
        format!("{p:.3}")
    }
}

fn pct(x: f64) -> String {
    format!("{:.0}", 100.0 * x)
}

/// A leak in one line: the row's furthest-off category.
fn leak_line(row: &SpotRow) -> Line<'static> {
    let mut spans =
        vec![Span::raw("  "), Span::styled(format!("{:<20}", row.spot.label()), Style::default().fg(ACCENT))];
    if let Some(c) = row.worst() {
        let (st, arrow) = cell_style(c);
        spans.push(Span::styled(
            format!(
                "{} {}% vs solver {}% {arrow}",
                row.spot.category_label(&c.category),
                pct(c.observed),
                pct(c.expected)
            ),
            st,
        ));
        spans.push(Span::styled(format!("  ({}, {})", z_text(c.z), decisions(row.decisions)), dim()));
    }
    Line::from(spans)
}

/// Everything on the score screen, as lines (the screen scrolls).
pub fn score_lines(r: &FrequencyReport, pot_weighted: bool, width: usize) -> Vec<Line<'static>> {
    let c = &r.calibration;
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Frequency score  ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:.0}", c.score()), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" ± {:.0}", c.uncertainty())),
            Span::styled(
                "   100 = your frequencies differ from the solver's no more than luck explains",
                dim(),
            ),
        ]),
        Line::from(format!(
            "  Your choices deviate {:.1}% from the solver's mixes for your hands; a player sampling from \
             those mixes would show {:.1}% on these decisions by luck alone (95% of the time under {:.1}%).",
            100.0 * c.deviation,
            100.0 * c.noise,
            100.0 * c.noise_hi
        )),
        Line::from(Span::styled(
            if c.p_value < 0.05 {
                format!("  p {}: more than luck explains — see the leaks below.", p_text(c.p_value))
            } else {
                format!("  p {}: consistent with the solver's frequencies so far.", p_text(c.p_value))
            },
            if c.p_value < 0.05 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            },
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{:<13}", "Aggression"), Style::default().fg(ACCENT)),
            Span::styled(
                format!("{:+.3}", r.aggression.value),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  ({})  ", z_text(r.aggression.z)), dim()),
            Span::raw(aggression_words(&r.aggression)),
        ]),
        Line::from(vec![
            Span::styled(format!("{:<13}", "Mixing"), Style::default().fg(ACCENT)),
            Span::styled(format!("{:.2}", r.mixing.value), Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(format!("  ({})  ", z_text(r.mixing.z)), dim()),
            Span::raw(mixing_words(&r.mixing)),
        ]),
        Line::from(Span::styled(
            format!(
                "{} heads-up decisions{} · EV grades still flag actions the solver doesn't take at all.",
                r.decisions,
                if pot_weighted { ", weighted by pot size" } else { ", weighted equally" }
            ),
            dim(),
        )),
        Line::from(""),
    ];
    let leaks = r.leaks();
    lines.push(Line::from(Span::styled(
        if leaks.is_empty() { "No significant leaks yet".to_string() } else { "Biggest leaks".to_string() },
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    )));
    for row in leaks.iter().take(5) {
        lines.push(leak_line(row));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("By spot", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("   you % / solver %  ·  ", dim()),
        Span::styled("▲ too often", Style::default().fg(Color::LightRed)),
        Span::styled("  ", dim()),
        Span::styled("▼ too rarely", Style::default().fg(Color::LightBlue)),
        Span::styled("  (|z| ≥ 2)", dim()),
    ]));
    for row in &r.rows {
        let flag = row.p_value < 0.05;
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<20}", row.spot.label()),
                if flag {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(ACCENT)
                },
            ),
            Span::styled(
                format!(
                    "{} · deviation {:.0}% (luck {:.0}%) · p {}{}",
                    decisions(row.decisions),
                    100.0 * row.deviation,
                    100.0 * row.noise,
                    p_text(row.p_value),
                    if flag { "  !" } else { "" }
                ),
                dim(),
            ),
        ]));
        let mut spans = vec![Span::raw("   ")];
        let mut x = 3;
        let mut row_lines = Vec::new();
        for c in &row.cells {
            let (st, arrow) = cell_style(c);
            let text = format!(
                "{} {}/{}{arrow}   ",
                row.spot.category_label(&c.category),
                pct(c.observed),
                pct(c.expected)
            );
            let w = text.chars().count();
            if x + w > width && x > 3 {
                row_lines.push(Line::from(std::mem::replace(&mut spans, vec![Span::raw("   ")])));
                x = 3;
            }
            spans.push(Span::styled(text, st));
            x += w;
        }
        row_lines.push(Line::from(spans));
        lines.extend(row_lines);
    }
    lines
}

pub fn draw_score(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(5), Constraint::Length(1)])
        .split(area);
    f.render_widget(title_bar(app, "Frequency score"), rows[0]);
    let inner_w = rows[1].width.saturating_sub(2) as usize;
    let lines = match app.session.report() {
        Some(r) => score_lines(r, app.session.config.scoring.pot_weighted, inner_w),
        None => vec![
            Line::from("No scored decisions yet."),
            Line::from(""),
            Line::from(
                "Your heads-up postflop decisions are scored once their reviews finish: the score compares \
                 how often you take each option (fold, check, call, each bet size, all-in) with the solver's \
                 mixes for the hands you held, spot by spot, and tells leaks from luck.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Tip: the RNG number at each decision helps you play mixes (e.g. bet 30%: bet on 0-29).",
                dim(),
            )),
        ],
    };
    let visible = rows[1].height.saturating_sub(2) as usize;
    let scroll = app.score_scroll.min(lines.len().saturating_sub(visible.max(1)));
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll as u16, 0))
            .block(Block::default().borders(Borders::ALL).title(" How you mix ")),
        rows[1],
    );
    f.render_widget(
        Paragraph::new(truncate("↑/↓ PgUp/PgDn scroll · any other key back · Q quit", area.width as usize)),
        rows[2],
    );
}

fn table_label(players: usize) -> String {
    match players {
        2 => "Heads-up".into(),
        n => format!("{n}-max"),
    }
}

pub fn draw_sessions(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(4),
            Constraint::Length(1),
        ])
        .split(area);
    f.render_widget(title_bar(app, "Sessions"), rows[0]);
    let now = save::now_secs();
    let current = app.session.save_id();
    let mut items = vec![ListItem::new(Line::from(vec![
        Span::styled("+ Start a new session", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled("  (with the current settings)", dim()),
    ]))];
    for m in &app.sessions {
        let s = &m.summary;
        let game = if m.config.opponent.eq_ignore_ascii_case("random") {
            if m.config.game == GameType::Custom {
                "custom table".to_string()
            } else {
                m.config.game.label().to_lowercase()
            }
        } else {
            format!("all {}", m.config.opponent)
        };
        let mut spans = vec![
            Span::styled(
                format!("{} UTC", save::format_time(m.created)),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  played {:<12}", save::format_age(m.updated, now)), dim()),
            Span::raw(format!("{:<9} {:<16}", table_label(m.config.players), truncate(&game, 16))),
            Span::raw(format!("{:>5} hands  ", s.hands)),
            Span::styled(
                format!("{:>9}", format!("{}{}", if s.net_chips >= 0 { "+" } else { "" }, bb(s.net_chips))),
                Style::default().fg(if s.net_chips >= 0 { Color::Green } else { Color::Red }),
            ),
            Span::raw(format!("  {:>4} graded", s.decisions)),
        ];
        if let Some(sc) = s.score {
            spans.push(Span::raw(format!("  freq score {sc:.0}")));
        }
        if current == Some(m.id.as_str()) {
            spans.push(Span::styled("  ◀ playing now", Style::default().fg(ACCENT)));
        }
        items.push(ListItem::new(Line::from(spans)));
    }
    let mut state = ListState::default().with_selected(Some(app.sessions_cursor.min(app.sessions.len())));
    f.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::ALL).title(" Continue a session or start a new one "))
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol("▶ "),
        rows[1],
        &mut state,
    );
    let info: Vec<Line> = if let Some(id) = &app.confirm_delete {
        vec![Line::from(Span::styled(
            format!(
                "Delete the session {id} and all its hands? Press Y to delete, any other key to keep it."
            ),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ))]
    } else {
        let root = app.sessions_root.as_ref().map(|r| r.display().to_string()).unwrap_or_default();
        vec![
            Line::from(
                "Hands are saved as you play (once their review is done), so a session can be continued \
                 later with its stats, reviews and frequency score.",
            ),
            Line::from(Span::styled(format!("Saved in {root}"), dim())),
        ]
    };
    f.render_widget(
        Paragraph::new(info).wrap(Wrap { trim: true }).block(Block::default().borders(Borders::TOP)),
        rows[2],
    );
    f.render_widget(
        Paragraph::new(truncate(
            "↑/↓ select · Enter continue / start · D delete · Esc back · Q quit",
            area.width as usize,
        )),
        rows[3],
    );
}
