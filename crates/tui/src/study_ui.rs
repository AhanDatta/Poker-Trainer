//! Rendering of the solver study screen.

use crate::app::App;
use crate::study_app::{EditorRow, GridView, Phase, SpotEditor, StudyState};
use crate::ui::title_bar;
use crate::widgets::{
    action_color, bar, big_cards, cards_spans, combo_spans, draw_grid, legend, truncate, GridCell, Hit,
    ACCENT,
};
use holdem::combo::{HandClass, COMBO_CLASS, NUM_CLASSES};
use holdem::{Action, Card, Street};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use std::time::Instant;
use trainer::fmt::{bb, BB};
use trainer::study::{class_fill, live_combos, range_combos, LineItem, NodeView, StudyPos};

/// Colors of players' ranges (by order in the hand).
const PLAYER_COLORS: [Color; 3] = [Color::Cyan, Color::LightYellow, Color::LightMagenta];

/// Width of the right-hand panel.
fn panel_width(total: u16) -> u16 {
    (total * 2 / 5).clamp(44, 72).min(total.saturating_sub(52))
}

pub fn draw_study(f: &mut Frame, app: &App) {
    let Some(st) = &app.study else { return };
    let area = f.area();
    let footer_h = 2;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(footer_h),
        ])
        .split(area);
    let street = st.street();
    let mode = if street == Street::Preflop {
        format!("Solver study — preflop charts ({})", table_name(st.settings.players))
    } else {
        "Solver study".to_string()
    };
    f.render_widget(title_bar(app, &mode), rows[0]);
    if let Some(ed) = &st.editor {
        draw_editor(f, app, rows[1].union(rows[2]), ed);
        draw_footer(f, rows[3], st, true);
        return;
    }
    draw_info(f, rows[1], st);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(52), Constraint::Length(panel_width(area.width))])
        .split(rows[2]);
    match &st.phase {
        Phase::Preflop(_) => draw_preflop(f, app, st, cols[0], cols[1]),
        Phase::Postflop(_) => draw_postflop(f, app, st, cols[0], cols[1]),
    }
    draw_footer(f, rows[3], st, false);
}

fn table_name(n: usize) -> String {
    match n {
        2 => "heads-up".into(),
        n => format!("{n}-max"),
    }
}

/// Spot summary and the line so far.
fn draw_info(f: &mut Frame, area: Rect, st: &StudyState) {
    let width = area.width as usize;
    let mut first = Vec::new();
    let mut second: Vec<Span> = Vec::new();
    match &st.phase {
        Phase::Preflop(nav) => {
            let s = nav.state();
            first.push(Span::raw(format!(
                "{} · {}bb stacks · pot {} · ",
                table_name(st.settings.players),
                st.settings.stack_bb,
                bb(s.pot())
            )));
            first.push(Span::styled(
                "preflop ranges from the built-in solver charts",
                Style::default().fg(Color::DarkGray),
            ));
            let line = nav.line_text();
            second.push(Span::styled("Line: ", Style::default().fg(ACCENT)));
            second.push(Span::raw(if line.is_empty() { "(start)".to_string() } else { line }));
        }
        Phase::Postflop(p) => {
            let spot = p.study.spot();
            first.push(Span::raw("Board "));
            first.extend(cards_spans(&p.study.board()));
            let (pot, stacks) = match &p.pos {
                StudyPos::Decision(v) => (v.pot, Some(v.stacks.clone())),
                StudyPos::NeedCard { view: Some(v), .. } | StudyPos::Over { view: Some(v), .. } => {
                    (v.pot, Some(v.stacks.clone()))
                }
                _ => (spot.pot, None),
            };
            first.push(Span::raw(format!("   Pot {}", bb(pot))));
            let stacks = stacks.unwrap_or_else(|| spot.stacks.clone());
            for (i, n) in spot.names.iter().enumerate() {
                first.push(Span::raw(" · "));
                first.push(Span::styled(n.clone(), Style::default().fg(PLAYER_COLORS[i % 3])));
                first.push(Span::raw(format!(" {}", bb(stacks[i]))));
            }
            if !spot.preflop.is_empty() {
                first.push(Span::styled(
                    format!("   Preflop: {}", spot.preflop),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            // The postflop line, keeping its end visible.
            let mut items: Vec<Vec<Span>> = Vec::new();
            for item in p.study.line() {
                match item {
                    LineItem::Street(s, cards) => {
                        let mut v = vec![Span::styled(format!("{s} "), Style::default().fg(Color::Yellow))];
                        v.extend(cards_spans(&cards));
                        items.push(v);
                    }
                    LineItem::Action { player, label } => items.push(vec![
                        Span::styled(spot.names[player].clone(), Style::default().fg(PLAYER_COLORS[player % 3])),
                        Span::raw(format!(" {label}")),
                    ]),
                }
            }
            let len = |v: &Vec<Span>| v.iter().map(|s| s.content.chars().count()).sum::<usize>() + 3;
            let mut start = 0;
            while start + 1 < items.len() && items[start..].iter().map(len).sum::<usize>() > width {
                start += 1;
            }
            if start > 0 {
                second.push(Span::styled("… ", Style::default().fg(Color::DarkGray)));
            }
            for (k, it) in items[start..].iter().enumerate() {
                if k > 0 {
                    second.push(Span::styled(" · ", Style::default().fg(Color::DarkGray)));
                }
                second.extend(it.iter().cloned());
            }
        }
    }
    f.render_widget(
        Paragraph::new(vec![Line::from(first), Line::from(second)]).block(Block::default().borders(Borders::BOTTOM)),
        area,
    );
}

fn draw_footer(f: &mut Frame, area: Rect, st: &StudyState, editor: bool) {
    let mut lines = Vec::new();
    if let Some(i) = &st.input {
        lines.push(Line::from(vec![
            Span::styled(format!("{}: ", i.prompt), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{}▏", i.text), Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                st.message.as_ref().map(|m| format!("   {m}")).unwrap_or_default(),
                Style::default().fg(Color::LightRed),
            ),
        ]));
    } else if let Some(m) = &st.message {
        lines.push(Line::from(Span::styled(m.clone(), Style::default().fg(Color::LightRed))));
    } else {
        lines.push(Line::from(""));
    }
    let keys = if editor {
        "↑/↓ field · Enter edit/confirm · ←/→ players · Ctrl+U clear · S solve · Esc cancel"
    } else if st.input.is_some() {
        "Type cards · Enter confirm (empty = random) · Backspace back · Tab ranges · arrows hand · Esc menu"
    } else {
        "F/C/1-9 act · arrows hand · Tab ranges · Backspace back · G start · N new board · E edit spot · P preflop · Esc menu · ? help"
    };
    lines.push(Line::from(Span::styled(truncate(keys, area.width as usize), Style::default().fg(Color::DarkGray))));
    f.render_widget(Paragraph::new(lines), area);
}

/// Grid cells of a range (one color).
fn range_cells(weights: &[f32], board: &[Card], color: Color) -> Vec<GridCell> {
    class_fill(weights, board)
        .into_iter()
        .map(|fill| GridCell {
            fill,
            mix: vec![(1.0, color)],
            sub: (fill > 0.0).then(|| format!("{:.0}%", 100.0 * fill)),
        })
        .collect()
}

fn draw_preflop(f: &mut Frame, app: &App, st: &StudyState, grid_area: Rect, panel: Rect) {
    let Phase::Preflop(nav) = &st.phase else { return };
    let node = nav.node();
    let players = st.range_players();
    let mut lines: Vec<Line> = Vec::new();
    match (st.view, &node) {
        (GridView::Strategy, Some(node)) => {
            let cells: Vec<GridCell> = node
                .grid
                .iter()
                .map(|c| match c {
                    Some(c) => GridCell {
                        fill: c.weight,
                        mix: c
                            .strategy
                            .iter()
                            .enumerate()
                            .map(|(a, &w)| (w, action_color(&node.actions, a)))
                            .collect(),
                        sub: None,
                    },
                    None => GridCell::default(),
                })
                .collect();
            let entries: Vec<(String, Color)> =
                node.labels.iter().enumerate().map(|(i, l)| (l.clone(), action_color(&node.actions, i))).collect();
            let title = Line::from(vec![
                Span::styled(format!("{} ", node.position), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
                Span::raw("to act · chart mix per hand"),
                Span::styled(" (width = share still in range)", Style::default().fg(Color::DarkGray)),
            ]);
            draw_grid(f, grid_area, title, &cells, legend(&entries, 18), Some(st.cursor), Some(&app.hits));
        }
        (view, _) => {
            let (seat, name) = match view {
                GridView::Range(s) => players.iter().find(|p| p.0 == s).cloned().unwrap_or((s, String::new())),
                _ => players.first().cloned().unwrap_or((0, String::new())),
            };
            let k = players.iter().position(|p| p.0 == seat).unwrap_or(0);
            let r = nav.range_of(seat);
            let cells = range_cells(r.weights(), &[], PLAYER_COLORS[k % 3]);
            let title = Line::from(vec![
                Span::styled(format!("{name}'s range "), Style::default().fg(PLAYER_COLORS[k % 3]).add_modifier(Modifier::BOLD)),
                Span::raw(format!("({:.0} combos, {:.1}% of hands)", r.num_combos(), 100.0 * r.num_combos() / 1326.0)),
            ]);
            draw_grid(f, grid_area, title, &cells, Vec::new(), Some(st.cursor), Some(&app.hits));
        }
    }

    // Panel.
    let class = HandClass::from_grid_index(st.cursor);
    if let Some(node) = &node {
        let s = nav.state();
        let to_call = s.to_call(node.seat);
        lines.push(Line::from(vec![
            Span::styled(node.position.clone(), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::raw(format!(" to act · pot {} · to call {}", bb(s.pot()), bb(to_call))),
        ]));
        lines.push(Line::from(""));
        let cell = node.grid[st.cursor].as_ref();
        let layout = RowLayout::new(panel.width.saturating_sub(1) as usize, true, false);
        lines.push(layout.header(&class.to_string()));
        let y0 = panel.y + lines.len() as u16;
        for (i, label) in node.labels.iter().enumerate() {
            let color = action_color(&node.actions, i);
            let key = action_key(&node.actions, i);
            lines.push(layout.row(key, label, color, node.range_strategy[i], cell.map(|c| c.strategy[i]), None));
            app.hits.add(Rect { x: panel.x, y: y0 + i as u16, width: panel.width, height: 1 }, Hit::Action(i));
        }
        lines.push(Line::from(""));
        match cell {
            Some(c) => lines.push(Line::from(format!(
                "{class}: {:.0}% of its combos are in {}'s range here.",
                100.0 * c.weight,
                node.position
            ))),
            None => lines.push(Line::from(Span::styled(
                format!("{class} isn't in {}'s range here.", node.position),
                Style::default().fg(Color::DarkGray),
            ))),
        }
    } else {
        lines.push(Line::from(Span::styled("Preflop is over.", Style::default().add_modifier(Modifier::BOLD))));
        lines.push(Line::from(""));
        if st.input.is_some() {
            lines.push(Line::from("Type the flop below (or press Enter for a random one) to solve it."));
            lines.push(Line::from("Tab shows each player's range going to the flop."));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Preflop strategies come from the charts (heads-up and 6-max: TexasSolverGPU solver output; other spots: position-aware rules).",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(Block::default().borders(Borders::LEFT)),
        panel,
    );
}

/// Column widths of the action rows for a panel of `inner` columns.
struct RowLayout {
    label: usize,
    bar: usize,
    class: bool,
    ev: bool,
}

impl RowLayout {
    fn new(inner: usize, class: bool, ev: bool) -> RowLayout {
        let extra = if class { 8 } else { 0 } + if ev { 10 } else { 0 };
        let mut bar = 6;
        let mut label = inner.saturating_sub(4 + bar + 7 + extra);
        if label < 12 {
            bar = 0;
            label = inner.saturating_sub(4 + 7 + extra);
        }
        RowLayout { label: label.clamp(6, 22), bar, class, ev }
    }

    fn header(&self, class: &str) -> Line<'static> {
        let mut t = format!("{:<w$}{:>r$}", "    Action", "range", w = self.label + 4, r = self.bar + 7);
        if self.class {
            t.push_str(&format!("{:>8}", class));
        }
        if self.ev {
            t.push_str(&format!("{:>10}", "EV"));
        }
        Line::from(Span::styled(t, Style::default().fg(Color::DarkGray)))
    }

    #[allow(clippy::too_many_arguments)]
    fn row(&self, key: char, label: &str, color: Color, freq: f64, class: Option<f32>, ev: Option<f32>) -> Line<'static> {
        let mut spans = vec![
            Span::styled(format!(" {key} "), Style::default().fg(Color::Black).bg(color)),
            Span::styled(format!(" {:<w$}", truncate(label, self.label), w = self.label), Style::default().fg(color)),
        ];
        if self.bar > 0 {
            spans.push(Span::styled(bar(freq as f32, self.bar), Style::default().fg(color)));
        }
        spans.push(Span::raw(format!(" {:5.1}%", 100.0 * freq)));
        if self.class {
            spans.push(Span::raw(match class {
                Some(c) => format!(" {:6.1}%", 100.0 * c),
                None => " ".repeat(8),
            }));
        }
        if self.ev {
            spans.push(Span::raw(match ev {
                Some(e) => format!("{:>10}", ev_text(e)),
                None => " ".repeat(10),
            }));
        }
        Line::from(spans)
    }
}

fn action_key(actions: &[Action], i: usize) -> char {
    let n = actions[..=i].iter().filter(|a| matches!(a, Action::Raise(_))).count();
    match actions[i] {
        Action::Fold => 'F',
        Action::Check | Action::Call => 'C',
        Action::Raise(_) => char::from_digit(n as u32, 10).unwrap_or('9'),
    }
}

fn elapsed(since: Option<Instant>) -> String {
    match since {
        Some(t) => {
            let s = t.elapsed().as_secs();
            format!("{}:{:02}", s / 60, s % 60)
        }
        None => String::new(),
    }
}

fn draw_postflop(f: &mut Frame, app: &App, st: &StudyState, grid_area: Rect, panel: Rect) {
    let Phase::Postflop(p) = &st.phase else { return };
    let spot = p.study.spot();
    let board = p.study.board();
    let view: Option<&NodeView> = match &p.pos {
        StudyPos::Decision(v) => Some(v),
        StudyPos::NeedCard { view: Some(v), .. } | StudyPos::Over { view: Some(v), .. } => Some(v),
        _ => None,
    };
    // Grid.
    let decision = matches!(p.pos, StudyPos::Decision(_));
    match (st.view, view) {
        (GridView::Strategy, Some(v)) if decision => {
            let actor = v.actor.unwrap_or(0);
            let evs = class_evs(v);
            let cells: Vec<GridCell> = v
                .grid
                .iter()
                .enumerate()
                .map(|(g, c)| match c {
                    Some(c) => GridCell {
                        fill: (c.weight / live_combos(g, &board).max(1) as f32).clamp(0.0, 1.0),
                        mix: c.strategy.iter().enumerate().map(|(a, &w)| (w, action_color(&v.actions, a))).collect(),
                        sub: evs[g].map(|e| format!("{:+.1}", e / BB as f32)),
                    },
                    None => GridCell::default(),
                })
                .collect();
            let entries: Vec<(String, Color)> =
                v.labels.iter().enumerate().map(|(i, l)| (l.clone(), action_color(&v.actions, i))).collect();
            let title = Line::from(vec![
                Span::styled(
                    format!("{} ", spot.names[actor]),
                    Style::default().fg(PLAYER_COLORS[actor % 3]).add_modifier(Modifier::BOLD),
                ),
                Span::raw("to act · solver mix per hand"),
                Span::styled(" (width = share in range; 2nd row = EV in bb)", Style::default().fg(Color::DarkGray)),
            ]);
            draw_grid(f, grid_area, title, &cells, legend(&entries, 18), Some(st.cursor), Some(&app.hits));
        }
        (view_kind, _) => {
            let players = st.range_players();
            let who = match view_kind {
                GridView::Range(i) => i,
                GridView::Strategy => players.first().map(|p| p.0).unwrap_or(0),
            };
            let (ranges, rboard) = match (view, &st.last_ranges) {
                (Some(v), _) => (v.ranges.clone(), board.clone()),
                (None, Some((r, b))) => (r.clone(), b.clone()),
                (None, None) => (spot.ranges.clone(), spot.board.clone()),
            };
            let cells = range_cells(&ranges[who], &rboard, PLAYER_COLORS[who % 3]);
            let combos = range_combos(&ranges[who], &rboard);
            let start = range_combos(&spot.ranges[who], &spot.board);
            let mut title = vec![
                Span::styled(
                    format!("{}'s range ", spot.names[who]),
                    Style::default().fg(PLAYER_COLORS[who % 3]).add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!("({combos:.1} combos, {:.0}% of the starting range)", 100.0 * combos / start.max(1e-9))),
            ];
            if view.is_none() {
                title.push(Span::styled(" — solving…", Style::default().fg(Color::Yellow)));
            }
            draw_grid(f, grid_area, Line::from(title), &cells, Vec::new(), Some(st.cursor), Some(&app.hits));
        }
    }

    // Panel.
    let mut lines: Vec<Line> = Vec::new();
    let wide = panel.width as usize;
    if panel.height >= 24 {
        let mut cards: Vec<Option<Card>> = board.iter().copied().map(Some).collect();
        while cards.len() < 5 {
            cards.push(None);
        }
        lines.extend(big_cards(&cards));
        lines.push(Line::from(""));
    }
    match &p.pos {
        StudyPos::Solving(s) => {
            lines.push(Line::from(Span::styled(
                format!("Solving the {}…", s.label.to_lowercase()),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(s.stage.clone()));
            if let Some(pr) = s.progress {
                lines.push(Line::from(format!(
                    "Iteration {} · exploitability {:.2}% of the pot (target {:.2}%)",
                    pr.iteration,
                    pr.exploitability_pct,
                    st.settings.solver.target_exploitability_pct
                )));
            }
            lines.push(Line::from(Span::styled(
                format!("Elapsed {}", elapsed(s.since)),
                Style::default().fg(Color::DarkGray),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Tab shows the ranges going into this street. Backspace goes back.",
                Style::default().fg(Color::DarkGray),
            )));
        }
        StudyPos::Loading => {
            lines.push(Line::from(Span::styled("Reading the solution…", Style::default().fg(Color::Yellow))));
        }
        StudyPos::Failed(e) => {
            lines.push(Line::from(Span::styled(format!("Solve failed: {e}"), Style::default().fg(Color::LightRed))));
            lines.push(Line::from("Backspace goes back; E edits the spot."));
        }
        StudyPos::NeedCard { next, .. } => {
            lines.push(Line::from(Span::styled(
                format!("The {} betting is over.", Street::from_board_len(board.len())),
                Style::default().add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(format!("Type the {} card below (Enter = random).", next.name().to_lowercase())));
            lines.push(Line::from("Tab: the ranges going into the next street."));
        }
        StudyPos::Over { text, .. } => {
            lines.push(Line::from(Span::styled(text.clone(), Style::default().add_modifier(Modifier::BOLD))));
            lines.push(Line::from("Backspace goes back; Tab shows the final ranges."));
        }
        StudyPos::Decision(v) => {
            let actor = v.actor.unwrap_or(0);
            lines.push(Line::from(vec![
                Span::styled(
                    spot.names[actor].clone(),
                    Style::default().fg(PLAYER_COLORS[actor % 3]).add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(" to act · pot {} · to call {}", bb(v.pot), bb(v.to_call))),
            ]));
            let class = HandClass::from_grid_index(st.cursor);
            let detail = class_detail(v, st.cursor);
            let layout = RowLayout::new(panel.width.saturating_sub(1) as usize, true, true);
            lines.push(layout.header(&class.to_string()));
            let y0 = panel.y + lines.len() as u16;
            for (i, label) in v.labels.iter().enumerate() {
                let color = action_color(&v.actions, i);
                let key = action_key(&v.actions, i);
                let (cf, ce) = match &detail {
                    Some(d) => (Some(d.freq[i]), Some(d.ev[i])),
                    None => (None, None),
                };
                lines.push(layout.row(key, label, color, v.range_strategy[i], cf, ce));
                app.hits.add(Rect { x: panel.x, y: y0 + i as u16, width: panel.width, height: 1 }, Hit::Action(i));
            }
            lines.push(Line::from(""));
            match &detail {
                Some(d) => {
                    lines.push(Line::from(format!(
                        "{class}: {:.1} combos in range · EV {}",
                        d.combos,
                        ev_text(d.total_ev)
                    )));
                    for (combo, w, mix, ev) in &d.rows {
                        let mut spans = vec![Span::raw(" ")];
                        spans.extend(combo_spans(*combo));
                        spans.push(Span::raw(format!(" {:>4.0}% ", 100.0 * w)));
                        let slots = crate::widgets::mix_slots(mix, 10);
                        for (a, &n) in slots.iter().enumerate() {
                            if n > 0 {
                                spans.push(Span::styled(
                                    "█".repeat(n),
                                    Style::default().fg(action_color(&v.actions, a)),
                                ));
                            }
                        }
                        spans.push(Span::raw(format!(" EV {}", ev_text(*ev))));
                        lines.push(Line::from(spans));
                    }
                }
                None => lines.push(Line::from(Span::styled(
                    format!("{class} isn't in {}'s range here.", spot.names[actor]),
                    Style::default().fg(Color::DarkGray),
                ))),
            }
        }
    }
    // Solve accuracy and notes.
    let status = p.study.solve_status();
    lines.push(Line::from(""));
    if status.done && status.error.is_none() {
        let what = if view.is_some_and(|v| v.multiway) { "convergence" } else { "exploitability" };
        lines.push(Line::from(Span::styled(
            format!("{} solve {what}: {:.2}% of the pot", status.label, status.exploitability_pct),
            Style::default().fg(Color::DarkGray),
        )));
    }
    for n in p.study.notes() {
        lines.push(Line::from(Span::styled(format!("• {}", truncate(&n, wide * 3)), Style::default().fg(Color::DarkGray))));
    }
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(Block::default().borders(Borders::LEFT)),
        panel,
    );
}

fn ev_text(chips: f32) -> String {
    let v = chips / BB as f32;
    if v.abs() < 0.005 {
        "0bb".into()
    } else {
        format!("{v:+.2}bb")
    }
}

/// Average EV (chips) of each class's combos in range, playing the solver's mix.
fn class_evs(v: &NodeView) -> Vec<Option<f32>> {
    let Some(info) = v.info.as_ref() else { return vec![None; NUM_CLASSES] };
    let n = info.num_hands();
    let na = info.actions.len();
    let mut sum = vec![(0f32, 0f32); NUM_CLASSES];
    for h in 0..n {
        let w = info.reach[h];
        if w <= 0.0 {
            continue;
        }
        let ev: f32 = (0..na).map(|a| info.strategy[a * n + h] * info.ev[a * n + h]).sum();
        let g = COMBO_CLASS[info.combos[h] as usize] as usize;
        sum[g].0 += w * ev;
        sum[g].1 += w;
    }
    sum.into_iter().map(|(e, w)| (w > 0.0).then(|| e / w)).collect()
}

/// A class's strategy and EVs at a decision.
struct ClassDetail {
    combos: f32,
    freq: Vec<f32>,
    ev: Vec<f32>,
    total_ev: f32,
    /// (combo, weight, mix, EV) per combo in range.
    rows: Vec<(usize, f32, Vec<f32>, f32)>,
}

fn class_detail(v: &NodeView, class: usize) -> Option<ClassDetail> {
    let info = v.info.as_ref()?;
    let n = info.num_hands();
    let na = info.actions.len();
    let mut d = ClassDetail { combos: 0.0, freq: vec![0.0; na], ev: vec![0.0; na], total_ev: 0.0, rows: Vec::new() };
    for h in 0..n {
        let c = info.combos[h] as usize;
        let w = info.reach[h];
        if COMBO_CLASS[c] as usize != class || w <= 0.0 {
            continue;
        }
        let mix = info.hand_strategy(h);
        let ev = info.hand_ev(h);
        let hand_ev: f32 = mix.iter().zip(&ev).map(|(s, e)| s * e).sum();
        d.combos += w;
        for a in 0..na {
            d.freq[a] += w * mix[a];
            d.ev[a] += w * ev[a];
        }
        d.total_ev += w * hand_ev;
        d.rows.push((c, w, mix, hand_ev));
    }
    if d.combos <= 0.0 {
        return None;
    }
    for a in 0..na {
        d.freq[a] /= d.combos;
        d.ev[a] /= d.combos;
    }
    d.total_ev /= d.combos;
    d.rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    Some(d)
}

fn draw_editor(f: &mut Frame, app: &App, area: Rect, ed: &SpotEditor) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(40), Constraint::Length(54.max(area.width / 2).min(area.width.saturating_sub(40)))])
        .split(area);
    let mut lines = vec![
        Line::from(Span::styled("Type a spot", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(
            "Ranges use the usual notation: 22+, A2s+, KTo+, 99-55, AKs:0.5 (weight), AsKs.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];
    let label_w = 12;
    let text_w = (cols[0].width as usize).saturating_sub(label_w + 4).max(10);
    let rows = ed.rows();
    let y0 = cols[0].y + lines.len() as u16;
    let mut y = y0;
    for (i, row) in rows.iter().enumerate() {
        let selected = i == ed.cursor;
        let (label, value) = match row {
            EditorRow::Players => (
                "Players".to_string(),
                format!("◂ {} ({}) ▸", ed.names.len(), ed.names.join(", ")),
            ),
            EditorRow::Range(k) => (format!("{} range", ed.names[*k]), ed.ranges[*k].clone()),
            EditorRow::Pot => ("Pot (bb)".into(), ed.pot.clone()),
            EditorRow::Stack => ("Stacks (bb)".into(), ed.stack.clone()),
            EditorRow::Board => (
                "Board".into(),
                if ed.board.trim().is_empty() && !(selected && ed.editing) {
                    "(random flop)".into()
                } else {
                    ed.board.clone()
                },
            ),
            EditorRow::Solve => (String::new(), "[ Solve this spot ]".into()),
        };
        let editing = selected && ed.editing;
        let style = if selected { Style::default().bg(Color::Indexed(238)) } else { Style::default() };
        let mut spans = vec![
            Span::styled(if selected { "▶ " } else { "  " }, style),
            Span::styled(format!("{:<w$}", truncate(&label, label_w), w = label_w), style.fg(ACCENT)),
        ];
        // While editing, show the end of long texts with a cursor.
        let shown = if editing {
            let t: Vec<char> = value.chars().collect();
            let start = t.len().saturating_sub(text_w.saturating_sub(1));
            format!("{}▏", t[start..].iter().collect::<String>())
        } else {
            truncate(&value, text_w)
        };
        spans.push(Span::styled(
            format!(" {shown}"),
            if editing { style.fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { style.add_modifier(Modifier::BOLD) },
        ));
        lines.push(Line::from(spans));
        app.hits.add(Rect { x: cols[0].x, y, width: cols[0].width, height: 1 }, Hit::Field(i));
        y += 1;
    }
    lines.push(Line::from(""));
    if let Some(e) = &ed.error {
        lines.push(Line::from(Span::styled(e.clone(), Style::default().fg(Color::LightRed))));
    }
    lines.push(Line::from(Span::styled(
        "Players act in the listed order (out of position first). Stacks are the chips behind at the start.",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), cols[0]);

    // Range preview.
    let (i, parsed) = ed.preview();
    let board = holdem::card::parse_cards(&ed.board).unwrap_or_default();
    match parsed {
        Ok(r) => {
            let cells = range_cells(r.weights(), &board, PLAYER_COLORS[i % 3]);
            let combos = range_combos(r.weights(), &board);
            let title = Line::from(vec![
                Span::styled(format!("{}'s range ", ed.names[i]), Style::default().fg(PLAYER_COLORS[i % 3])),
                Span::raw(format!("({combos:.1} combos)")),
            ]);
            draw_grid(f, cols[1], title, &cells, Vec::new(), None, None);
        }
        Err(e) => f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(format!("{}'s range", ed.names[i]), Style::default().fg(ACCENT))),
                Line::from(Span::styled(e, Style::default().fg(Color::LightRed))),
            ])
            .wrap(Wrap { trim: false }),
            cols[1],
        ),
    }
}
