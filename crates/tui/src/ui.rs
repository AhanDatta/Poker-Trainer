//! Rendering.

use crate::app::{App, OpponentChoice, Screen, SetupItem, SEAT_CHANGE_PRESETS};
use holdem::combo::{combo_cards, HandClass, NUM_CLASSES};
use holdem::{Action, Card, Street};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use trainer::config::{FlopMode, HeroPosition, VillainRange};
use trainer::fmt::bb;
use trainer::review::ClassCell;
use trainer::{DecisionReview, LiveHand, Variant, Verdict};

const ACCENT: Color = Color::Cyan;

pub fn draw(f: &mut Frame, app: &App) {
    match app.screen {
        Screen::Setup => draw_setup(f, app),
        Screen::Table => draw_table(f, app),
        Screen::Review => draw_review(f, app),
        Screen::History => draw_history(f, app),
        Screen::Help => draw_help(f),
    }
    if let Some((msg, _)) = &app.message {
        let area = f.area();
        let w = (msg.chars().count() as u16 + 4).min(area.width);
        let rect = Rect {
            x: area.x + (area.width - w) / 2,
            y: area.y + area.height.saturating_sub(4),
            width: w,
            height: 3,
        };
        f.render_widget(Clear, rect);
        f.render_widget(
            Paragraph::new(msg.as_str()).alignment(Alignment::Center).block(
                Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Yellow)),
            ),
            rect,
        );
    }
}

fn suit_style(c: Card) -> (char, Color) {
    match c.suit() {
        0 => ('♣', Color::Green),
        1 => ('♦', Color::LightBlue),
        2 => ('♥', Color::LightRed),
        _ => ('♠', Color::White),
    }
}

fn card_span(c: Card) -> Span<'static> {
    let (sym, color) = suit_style(c);
    Span::styled(format!("{}{}", c.rank_char(), sym), Style::default().fg(color).add_modifier(Modifier::BOLD))
}

fn cards_spans(cards: &[Card]) -> Vec<Span<'static>> {
    let mut v = Vec::new();
    for (i, &c) in cards.iter().enumerate() {
        if i > 0 {
            v.push(Span::raw(" "));
        }
        v.push(card_span(c));
    }
    v
}

fn combo_spans(combo: usize) -> Vec<Span<'static>> {
    let (a, b) = combo_cards(combo);
    cards_spans(&[a, b])
}

fn verdict_color(v: Verdict) -> Color {
    match v {
        Verdict::Best => Color::Green,
        Verdict::Good => Color::LightGreen,
        Verdict::Inaccuracy => Color::Yellow,
        Verdict::Mistake => Color::LightRed,
        Verdict::Blunder => Color::Red,
    }
}

/// Color of the `i`-th action in a node's action list.
fn action_color(actions: &[Action], i: usize) -> Color {
    match actions[i] {
        Action::Fold => Color::DarkGray,
        Action::Check | Action::Call => Color::Green,
        Action::Raise(_) => {
            let raises: Vec<usize> =
                (0..actions.len()).filter(|&j| matches!(actions[j], Action::Raise(_))).collect();
            let k = raises.iter().position(|&j| j == i).unwrap_or(0);
            let palette = [Color::Yellow, Color::LightRed, Color::Red, Color::Magenta];
            if k + 1 == raises.len() && raises.len() > 1 {
                palette[(raises.len() - 1).min(3)]
            } else {
                palette[k.min(3)]
            }
        }
    }
}

fn title_bar(app: &App, extra: &str) -> Paragraph<'static> {
    let st = app.session.stats();
    let mut spans = vec![
        Span::styled(
            " ♠ Poker Trainer ",
            Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(" {extra}  ")),
    ];
    let net = st.net_chips;
    spans.push(Span::raw(format!("Hands {} · Net ", st.hands)));
    spans.push(Span::styled(
        format!("{}{}", if net >= 0 { "+" } else { "" }, bb(net)),
        Style::default().fg(if net >= 0 { Color::Green } else { Color::Red }),
    ));
    if st.decisions > 0 {
        spans.push(Span::raw(format!(
            " · Graded {} · Accuracy {:.0}% · Avg loss {:.1}% pot",
            st.decisions,
            100.0 * st.accuracy(),
            st.avg_loss_pct()
        )));
    }
    if st.reviews_pending > 0 {
        spans.push(Span::styled(
            format!(" · {} review(s) solving", st.reviews_pending),
            Style::default().fg(Color::Yellow),
        ));
    }
    Paragraph::new(Line::from(spans))
}

fn draw_setup(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(10), Constraint::Length(3)])
        .split(area);
    f.render_widget(title_bar(app, "Setup"), rows[0]);
    let cfg = &app.session.config;
    let value = |item: SetupItem| -> String {
        match item {
            SetupItem::Table => match cfg.players {
                2 => "Heads-up (2 players)".into(),
                6 => "6-max".into(),
                9 => "9-max (full ring)".into(),
                n => format!("{n} players"),
            },
            SetupItem::Opponent => match app.current_opponent_choice() {
                OpponentChoice::Game(g) if cfg.players == 2 => {
                    format!("{} — opponent drawn from: {}", g.label(), g.description())
                }
                OpponentChoice::Game(g) => format!("{} — {}", g.label(), g.description()),
                OpponentChoice::Style(name) => {
                    let d = cfg.profile_by_name(&name).map(|p| p.description).unwrap_or_default();
                    format!("All {name} — {d}")
                }
            },
            SetupItem::Count(i) => {
                let name = cfg.profiles().get(i).map(|p| p.name.clone()).unwrap_or_default();
                let n = cfg.custom_table.get(&name).copied().unwrap_or(0);
                format!("{n} at the table")
            }
            SetupItem::SeatChanges => SEAT_CHANGE_PRESETS
                .iter()
                .find(|&&(p, _)| (p - cfg.seat_change_pct).abs() < 1e-9)
                .map(|&(_, l)| l.to_string())
                .unwrap_or_else(|| format!("{}% chance per player and hand", cfg.seat_change_pct)),
            SetupItem::Position => match (cfg.hero_position, cfg.players) {
                (HeroPosition::Alternate, 2) => "Alternate SB / BB".into(),
                (HeroPosition::Alternate, _) => "Rotate (the button moves every hand)".into(),
                (HeroPosition::Random, _) => "Random".into(),
                (HeroPosition::Sb, 2) => "Always SB (button)".into(),
                (HeroPosition::Sb, _) => "Always SB".into(),
                (HeroPosition::Bb, _) => "Always BB".into(),
            },
            SetupItem::Stack => format!("{}bb", cfg.stack_bb),
            SetupItem::Rake => {
                if cfg.rake_pct <= 0.0 {
                    "None".into()
                } else {
                    format!(
                        "{}% of the pot, capped at {}bb (no flop, no drop)",
                        cfg.rake_pct, cfg.rake_cap_bb
                    )
                }
            }
            SetupItem::FlopSolve => match cfg.solver.flop_mode {
                FlopMode::Fast => format!(
                    "Fast — {} turn × {} sampled river cards (~4x faster, approximate)",
                    cfg.solver.flop_turn_samples, cfg.solver.flop_river_samples
                ),
                FlopMode::Exact => {
                    format!("Exact — every runout (up to {} GB of RAM, minutes)", cfg.solver.max_memory_gb)
                }
            },
            SetupItem::VillainRange => match cfg.solver.villain_range {
                VillainRange::Gto => "GTO ranges (villain plays the charts)".into(),
                VillainRange::Actual => "Villain's actual range (its style's preflop range)".into(),
                VillainRange::Both => "Both, side by side (2× solving time)".into(),
            },
            SetupItem::RevealStyle => {
                if cfg.show_opponent_style { "On" } else { "Off (shown in the review)" }.into()
            }
        }
    };
    let label = |item: SetupItem| -> String {
        match item {
            SetupItem::Table => "Table",
            SetupItem::Opponent => {
                if cfg.players == 2 {
                    "Opponent"
                } else {
                    "Opponents"
                }
            }
            SetupItem::Count(i) => {
                return format!("    {}", cfg.profiles().get(i).map(|p| p.name.clone()).unwrap_or_default())
            }
            SetupItem::SeatChanges => "New players sit down",
            SetupItem::Position => "Your position",
            SetupItem::Stack => "Stacks",
            SetupItem::Rake => "Rake",
            SetupItem::FlopSolve => "Flop solving",
            SetupItem::VillainRange => "Villain range in review",
            SetupItem::RevealStyle => "Reveal style during hand",
        }
        .to_string()
    };
    let setup_items = app.setup_items();
    let items: Vec<ListItem> = setup_items
        .iter()
        .map(|&it| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<26}", label(it)), Style::default().fg(ACCENT)),
                Span::raw("◂ "),
                Span::styled(value(it), Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" ▸"),
            ]))
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(app.setup_cursor));
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Settings "))
        .highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");
    let mid = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(setup_items.len() as u16 + 2), Constraint::Min(4)])
        .split(rows[1]);
    f.render_stateful_widget(list, mid[0], &mut state);
    let about = vec![
        Line::from("Play NLHE hands (heads-up, 6-max or 9-max) from preflop to the river against AI"),
        Line::from("opponents with different styles. After each hand, press R to see what the solver"),
        Line::from("would have done in every postflop spot you played, with the EV of each option."),
        Line::from(""),
        Line::from(Span::styled(
            "Preflop decisions are not graded yet; the charts set the AIs' play and the starting ranges.",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    f.render_widget(
        Paragraph::new(about)
            .block(Block::default().borders(Borders::ALL).title(" About "))
            .wrap(Wrap { trim: true }),
        mid[1],
    );
    f.render_widget(
        Paragraph::new("↑/↓ select · ←/→ change · Enter play · W save settings · ? help · Q quit")
            .block(Block::default().borders(Borders::TOP)),
        rows[2],
    );
}

fn seat_line(hand: &LiveHand, seat: usize, show_cards: bool, app: &App) -> Line<'static> {
    let st = &hand.state;
    let mut spans = Vec::new();
    let is_hero = seat == hand.hero;
    let name = if is_hero {
        "You".to_string()
    } else {
        let style = hand.profiles[seat].as_ref().map(|p| p.name.clone()).unwrap_or_default();
        let who = if st.num_players() == 2 { "Villain".to_string() } else { format!("Player {}", seat + 1) };
        if app.session.config.show_opponent_style || hand.is_finished() {
            format!("{who} ({style})")
        } else {
            who
        }
    };
    let to_act = st.to_act == Some(seat);
    spans.push(Span::styled(
        format!("{} {:<20}", if to_act { "▶" } else { " " }, name),
        Style::default().fg(if is_hero { ACCENT } else { Color::White }).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(format!("{:<4}", st.position_name(seat)), Style::default().fg(Color::Yellow)));
    spans.push(Span::raw(format!(" stack {:<9}", bb(st.stacks[seat]))));
    spans.push(Span::raw("  "));
    if st.folded[seat] {
        spans.push(Span::styled("folded", Style::default().fg(Color::DarkGray)));
    } else if show_cards {
        spans.extend(cards_spans(&st.hole[seat]));
    } else {
        spans.push(Span::styled("▒▒ ▒▒", Style::default().fg(Color::Blue)));
    }
    if st.street_commit[seat] > 0 {
        spans.push(Span::styled(
            format!("   bet {}", bb(st.street_commit[seat])),
            Style::default().fg(Color::LightYellow),
        ));
    }
    if st.is_all_in(seat) {
        spans.push(Span::styled("  ALL-IN", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)));
    }
    Line::from(spans)
}

fn draw_table(f: &mut Frame, app: &App) {
    let area = f.area();
    let Some(hand) = app.current() else { return };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(10),
            Constraint::Length(4),
            Constraint::Length(3),
        ])
        .split(area);
    f.render_widget(title_bar(app, &format!("Hand #{}", hand.id)), rows[0]);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[1]);

    // Felt.
    let st = &hand.state;
    let n = st.num_players();
    let showdown = hand.is_finished() && hand.settlement().is_some_and(|s| s.showdown);
    let mut board_spans = vec![Span::raw("      Board  ")];
    for i in 0..5 {
        if i > 0 {
            board_spans.push(Span::raw(" "));
        }
        match st.board.get(i) {
            Some(&c) => board_spans.push(card_span(c)),
            None => board_spans.push(Span::styled("··", Style::default().fg(Color::DarkGray))),
        }
    }
    // Opponents in seating order starting left of the hero; cards are shown
    // at showdown for players still in the hand.
    let mut felt = vec![Line::from("")];
    for k in 1..n {
        let seat = (hand.hero + k) % n;
        felt.push(seat_line(hand, seat, showdown && !st.folded[seat], app));
    }
    let spacer = if n <= 3 { 2 } else { 1 };
    felt.extend(std::iter::repeat_n(Line::from(""), spacer));
    felt.extend(vec![
        Line::from(board_spans),
        Line::from(vec![
            Span::raw("      Pot    "),
            Span::styled(bb(st.pot()), Style::default().fg(Color::LightYellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("   {}", st.street), Style::default().fg(Color::DarkGray)),
        ]),
    ]);
    felt.extend(std::iter::repeat_n(Line::from(""), spacer));
    felt.push(seat_line(hand, hand.hero, true, app));
    f.render_widget(
        Paragraph::new(felt).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(" Table "),
        ),
        cols[0],
    );

    // Log.
    let log_height = cols[1].height.saturating_sub(2) as usize;
    let start = hand.log.len().saturating_sub(log_height);
    let log: Vec<Line> = hand.log[start..]
        .iter()
        .map(|e| {
            let style = match e.seat {
                Some(s) if s == hand.hero => Style::default().fg(ACCENT),
                Some(_) => Style::default(),
                None => Style::default().fg(Color::Yellow),
            };
            Line::from(Span::styled(e.text.clone(), style))
        })
        .collect();
    f.render_widget(
        Paragraph::new(log)
            .block(Block::default().borders(Borders::ALL).title(" Hand log "))
            .wrap(Wrap { trim: false }),
        cols[1],
    );

    // Actions.
    let mut lines = Vec::new();
    if hand.hero_to_act() {
        let ctx = hand.action_context().unwrap();
        let mut spans = Vec::new();
        for (key, a) in app.action_keys() {
            spans.push(Span::styled(
                format!(" [{}] ", key.to_ascii_uppercase()),
                Style::default().fg(Color::Black).bg(ACCENT),
            ));
            spans.push(Span::raw(format!(" {}   ", trainer::fmt::action_label(a, &ctx))));
        }
        lines.push(Line::from(spans));
        let mut info = format!("To call {} · pot {}", bb(ctx.to_call), bb(ctx.pot));
        if st.street == Street::Preflop {
            info.push_str(" · [X]/[C] check/call · [A] max raise");
        }
        lines.push(Line::from(Span::styled(info, Style::default().fg(Color::DarkGray))));
    } else if hand.is_finished() {
        let net = hand.hero_net().unwrap_or(0);
        let result = if net >= 0 { format!("You won {}", bb(net)) } else { format!("You lost {}", bb(-net)) };
        lines.push(Line::from(vec![
            Span::styled(
                result,
                Style::default()
                    .fg(if net >= 0 { Color::Green } else { Color::Red })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("   [N] next hand   [R] solver review   [H] history   [S] settings"),
        ]));
    } else {
        lines.push(Line::from(Span::styled("Villain is thinking…", Style::default().fg(Color::DarkGray))));
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Your move ")),
        rows[2],
    );

    // Solver status.
    let status = match hand.review() {
        None => Line::from(Span::styled(
            if hand.is_finished() {
                "No postflop decisions to review in this hand."
            } else if hand.hero_folded() {
                "You folded: nothing to review."
            } else {
                "Solver review starts when the flop is dealt."
            },
            Style::default().fg(Color::DarkGray),
        )),
        Some(r) => {
            let s = r.status();
            let mut text = format!("Solver: {}", s.stage);
            if let Some(p) = s.progress {
                text.push_str(&format!(
                    " — iteration {} · exploitability {:.2}% pot",
                    p.iteration, p.exploitability_pct
                ));
            }
            text.push_str(&elapsed(&s));
            if s.finished && s.error.is_none() {
                text = format!(
                    "Solver: review ready ({} decisions graded) — press R after the hand",
                    s.decisions.len()
                );
            }
            Line::from(Span::styled(text, Style::default().fg(Color::LightMagenta)))
        }
    };
    f.render_widget(Paragraph::new(status).block(Block::default().borders(Borders::TOP)), rows[3]);
}

/// " · 1:23" for a running solve stage, so a long solve visibly moves
/// before its first exploitability check.
fn elapsed(s: &trainer::ReviewStatus) -> String {
    match s.stage_since {
        Some(t) if !s.finished && s.stage.contains("solving") => {
            let secs = t.elapsed().as_secs();
            format!(" · {}:{:02}", secs / 60, secs % 60)
        }
        _ => String::new(),
    }
}

fn bar(frac: f32, width: usize) -> String {
    let filled = (frac.clamp(0.0, 1.0) * width as f32).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn draw_review(f: &mut Frame, app: &App) {
    let area = f.area();
    let Some(hand) = app.review_hand() else { return };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);
    f.render_widget(title_bar(app, &format!("Review — hand #{}", hand.id)), rows[0]);

    let status = hand.review().map(|r| r.status()).unwrap_or_default();
    let villain_style = hand.opponents_label();
    let net = hand.hero_net().unwrap_or(0);
    let mut head = vec![Span::raw(" You held ")];
    head.extend(cards_spans(&hand.state.hole[hand.hero]));
    let variants: Vec<Variant> = {
        let mut v: Vec<Variant> = status.decisions.iter().map(|d| d.variant).collect();
        v.dedup();
        v.sort_by_key(|x| *x as u8);
        v.dedup();
        v
    };
    head.push(Span::raw(format!(
        " as {} vs {} · result {}{}{}",
        hand.state.position_name(hand.hero),
        villain_style,
        if net >= 0 { "+" } else { "" },
        bb(net),
        if variants.is_empty() { "" } else { " · " }
    )));
    for v in &variants {
        let active = *v == app.review_variant || variants.len() == 1;
        head.push(Span::styled(
            format!(" {} ", v.label()),
            if active {
                Style::default().fg(Color::Black).bg(ACCENT)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ));
        head.push(Span::raw(" "));
    }
    let mut second = String::new();
    if !status.finished {
        second = format!("Solving… {}", status.stage);
        if let Some(p) = status.progress {
            second.push_str(&format!(" (iteration {}, {:.2}% pot)", p.iteration, p.exploitability_pct));
        }
        second.push_str(&elapsed(&status));
    } else if let Some(e) = &status.error {
        second = format!("Review error: {e}");
    } else if status.cancelled {
        second = "Review cancelled: you moved on to the next hand before it finished.".into();
    }
    let header = vec![Line::from(head), Line::from(Span::styled(second, Style::default().fg(Color::Yellow)))];
    f.render_widget(Paragraph::new(header).block(Block::default().borders(Borders::BOTTOM)), rows[1]);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(38), Constraint::Min(40)])
        .split(rows[2]);
    let decisions = app.visible_decisions();
    let items: Vec<ListItem> = decisions
        .iter()
        .map(|d| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<6}", d.street.name()), Style::default().fg(Color::Yellow)),
                Span::raw(format!("{:<17}", truncate(&d.labels[d.chosen], 16))),
                Span::styled(
                    format!("{:>3} {}", d.verdict.symbol(), d.verdict.label()),
                    Style::default().fg(verdict_color(d.verdict)).add_modifier(Modifier::BOLD),
                ),
            ]))
        })
        .collect();
    let mut state = ListState::default().with_selected(if decisions.is_empty() {
        None
    } else {
        Some(app.review_selected.min(decisions.len() - 1))
    });
    let empty_msg = if decisions.is_empty() && status.finished {
        " No postflop decisions to grade "
    } else {
        " Your decisions "
    };
    f.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::ALL).title(empty_msg))
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol("▶ "),
        cols[0],
        &mut state,
    );

    if let Some(d) = decisions.get(app.review_selected.min(decisions.len().saturating_sub(1))) {
        draw_decision(f, cols[1], d, &status.notes);
    } else {
        let mut lines: Vec<Line> = status.notes.iter().map(|n| Line::from(format!("• {n}"))).collect();
        if lines.is_empty() {
            lines.push(Line::from("Waiting for the solver…"));
        }
        f.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: true }).block(Block::default().borders(Borders::ALL)),
            cols[1],
        );
    }
    f.render_widget(
        Paragraph::new("↑/↓ decision · Tab switch range variant · N next hand · Esc back · Q quit"),
        rows[3],
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

fn draw_decision(f: &mut Frame, area: Rect, d: &DecisionReview, hand_notes: &[String]) {
    let block =
        Block::default().borders(Borders::ALL).title(format!(" {} — {} ", d.street, d.variant.label()));
    let inner = block.inner(area);
    f.render_widget(block, area);
    const GRID_W: u16 = 54;
    const GRID_H: u16 = 15;
    let side_by_side = inner.width >= 56 + GRID_W;
    let parts = if side_by_side {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(56), Constraint::Length(GRID_W)])
            .split(inner)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(GRID_H)])
            .split(inner)
    };

    let mut lines = Vec::new();
    let mut board = vec![Span::raw("Board ")];
    board.extend(cards_spans(&d.board));
    board.push(Span::raw(format!("   Pot {}", bb(d.pot))));
    if d.to_call > 0 {
        board.push(Span::raw(format!(" · to call {}", bb(d.to_call))));
    }
    lines.push(Line::from(board));
    lines.push(Line::from(Span::styled(
        if d.history.is_empty() { "First action on this street".to_string() } else { d.history.clone() },
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));
    let mut you = vec![Span::raw("You ")];
    you.extend(combo_spans(d.hero_combo));
    you.push(Span::raw(" chose "));
    you.push(Span::styled(d.labels[d.chosen].clone(), Style::default().add_modifier(Modifier::BOLD)));
    lines.push(Line::from(you));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{} {}", d.verdict.symbol(), d.verdict.label()),
            Style::default().fg(verdict_color(d.verdict)).add_modifier(Modifier::BOLD),
        ),
        Span::raw(if d.ev_loss > 0.0 {
            format!(
                "  −{} ({:.1}% of pot) vs. {}",
                bb(d.ev_loss.round() as i64),
                d.ev_loss_pct,
                d.labels[d.best]
            )
        } else {
            "  no EV lost".to_string()
        }),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Solver, your hand:", Style::default().fg(ACCENT))));
    for i in 0..d.actions.len() {
        let mark = if i == d.chosen {
            "◀ you"
        } else if i == d.best {
            "★ best"
        } else {
            ""
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:<19}", truncate(&d.labels[i], 18)),
                Style::default().fg(action_color(&d.actions, i)),
            ),
            Span::styled(bar(d.hero_strategy[i], 10), Style::default().fg(action_color(&d.actions, i))),
            Span::raw(format!(
                " {:5.1}%  EV {:>8} {mark}",
                100.0 * d.hero_strategy[i],
                bb(d.hero_ev[i].round() as i64)
            )),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Whole range:", Style::default().fg(ACCENT))));
    for i in 0..d.actions.len() {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:<19}", truncate(&d.labels[i], 18)),
                Style::default().fg(action_color(&d.actions, i)),
            ),
            Span::styled(
                bar(d.range_strategy[i] as f32, 10),
                Style::default().fg(action_color(&d.actions, i)),
            ),
            Span::raw(format!(" {:5.1}%", 100.0 * d.range_strategy[i])),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("Solve accuracy: exploitability {:.2}% of pot", d.exploitability_pct),
        Style::default().fg(Color::DarkGray),
    )));
    for n in d.notes.iter().chain(hand_notes) {
        lines.push(Line::from(Span::styled(format!("• {n}"), Style::default().fg(Color::DarkGray))));
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), parts[0]);
    draw_grid(f, parts[1], d);
}

/// Splits `width` character cells among actions in proportion to `strategy`
/// (largest remainders), so a cell shows its mix rather than one action.
fn mix_slots(strategy: &[f32], width: usize) -> Vec<usize> {
    let total: f32 = strategy.iter().sum();
    if total <= 0.0 {
        return vec![0; strategy.len()];
    }
    let exact: Vec<f32> = strategy.iter().map(|&v| v / total * width as f32).collect();
    let mut slots: Vec<usize> = exact.iter().map(|&x| x.floor() as usize).collect();
    let mut left = width - slots.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..exact.len()).collect();
    order.sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())));
    for &i in &order {
        if left == 0 {
            break;
        }
        slots[i] += 1;
        left -= 1;
    }
    slots
}

/// 13x13 grid; each class's 4-character cell is colored by its action mix
/// (in quarters). Coloring by the single most frequent action would hide
/// low-frequency bluffs and make betting ranges look value-heavy.
fn draw_grid(f: &mut Frame, area: Rect, d: &DecisionReview) {
    let mut lines = vec![Line::from(Span::styled(
        "Your range at this spot (each cell colored by its action mix):",
        Style::default().fg(ACCENT),
    ))];
    for row in 0..13 {
        let mut spans = Vec::new();
        for col in 0..13 {
            let g = row * 13 + col;
            let class = HandClass::from_grid_index(g);
            let label: Vec<char> = format!("{:<4}", class.to_string()).chars().collect();
            match &d.grid[g] {
                Some(ClassCell { strategy, .. }) => {
                    let slots = mix_slots(strategy, label.len());
                    let mut k = 0;
                    for (a, &n) in slots.iter().enumerate() {
                        if n == 0 {
                            continue;
                        }
                        let text: String = label[k..k + n].iter().collect();
                        k += n;
                        spans.push(Span::styled(
                            text,
                            Style::default().fg(Color::Black).bg(action_color(&d.actions, a)),
                        ));
                    }
                }
                None => {
                    let text: String = label.iter().collect();
                    spans.push(Span::styled(text, Style::default().fg(Color::DarkGray)));
                }
            }
        }
        lines.push(Line::from(spans));
    }
    let mut legend = vec![Span::raw(" ")];
    for (i, l) in d.labels.iter().enumerate() {
        legend.push(Span::styled("  ", Style::default().bg(action_color(&d.actions, i))));
        legend.push(Span::raw(format!(" {}  ", truncate(l, 16))));
    }
    lines.push(Line::from(legend));
    debug_assert_eq!(d.grid.len(), NUM_CLASSES);
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
}

fn draw_history(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(5), Constraint::Length(1)])
        .split(area);
    f.render_widget(title_bar(app, "History"), rows[0]);
    let items: Vec<ListItem> = app
        .session
        .history
        .iter()
        .map(|h| {
            let net = h.hero_net().unwrap_or(0);
            let mut spans = vec![Span::raw(format!("#{:<4} {:<3} ", h.id, h.state.position_name(h.hero)))];
            spans.extend(cards_spans(&h.state.hole[h.hero]));
            spans.push(Span::raw(format!("  vs {:<16}", h.opponents_label())));
            spans.push(Span::styled(
                format!("{:>9}  ", format!("{}{}", if net >= 0 { "+" } else { "" }, bb(net))),
                Style::default().fg(if net >= 0 { Color::Green } else { Color::Red }),
            ));
            match h.review() {
                None => spans.push(Span::styled("no postflop", Style::default().fg(Color::DarkGray))),
                Some(r) => {
                    let st = r.status();
                    if !st.finished {
                        spans.push(Span::styled("solving…", Style::default().fg(Color::Yellow)));
                    } else if st.cancelled {
                        spans.push(Span::styled("cancelled ", Style::default().fg(Color::DarkGray)));
                    }
                    let first = st.decisions.first().map(|d| d.variant);
                    for d in st.decisions.iter().filter(|d| Some(d.variant) == first) {
                        spans.push(Span::styled(
                            format!("{} ", d.verdict.symbol()),
                            Style::default().fg(verdict_color(d.verdict)),
                        ));
                    }
                }
            }
            ListItem::new(Line::from(spans))
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(app.history_selected));
    f.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::ALL).title(" Finished hands "))
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol("▶ "),
        rows[1],
        &mut state,
    );
    f.render_widget(Paragraph::new("↑/↓ select · Enter open review · Esc back · Q quit"), rows[2]);
}

fn draw_help(f: &mut Frame) {
    let text = vec![
        Line::from(Span::styled("Poker Trainer — keys", Style::default().add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from("At the table"),
        Line::from("  F fold · C or X check/call · 1-9 bet/raise sizes from the menu · A largest raise"),
        Line::from("  N next hand · R solver review (after the hand) · H history · S settings · Q quit"),
        Line::from(""),
        Line::from("Review"),
        Line::from("  ↑/↓ choose a decision · Tab switch between GTO and villain's-actual-range solves"),
        Line::from(""),
        Line::from("Grades (EV lost vs. the best action, as % of the pot)"),
        Line::from("  ✓ Best ≤1% · ✓ Good ≤3% · ?! Inaccuracy ≤6% · ? Mistake ≤12% · ?? Blunder >12%"),
        Line::from(""),
        Line::from("EV is measured from the decision onwards: folding is worth 0."),
        Line::from(""),
        Line::from("Press any key to go back."),
    ];
    f.render_widget(
        Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title(" Help "))
            .wrap(Wrap { trim: false }),
        f.area(),
    );
}

#[cfg(test)]
mod tests {
    use super::mix_slots;

    #[test]
    fn grid_cells_show_mixes() {
        assert_eq!(mix_slots(&[1.0, 0.0], 4), vec![4, 0]);
        assert_eq!(mix_slots(&[0.75, 0.25], 4), vec![3, 1]);
        // A 14% bluff still shows up as one quarter of the cell.
        assert_eq!(mix_slots(&[0.86, 0.14], 4), vec![3, 1]);
        assert_eq!(mix_slots(&[0.5, 0.3, 0.2], 4), vec![2, 1, 1]);
        assert_eq!(mix_slots(&[0.0, 0.0], 4), vec![0, 0]);
        for s in [[0.1f32, 0.2, 0.7], [0.33, 0.33, 0.34], [0.05, 0.05, 0.9]] {
            assert_eq!(mix_slots(&s, 4).iter().sum::<usize>(), 4);
        }
    }
}
