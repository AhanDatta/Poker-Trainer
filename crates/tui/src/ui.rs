//! Rendering.

use crate::app::{App, Mode, OpponentChoice, Screen, SetupItem, SEAT_CHANGE_PRESETS};
use crate::widgets::{
    action_color, bar, big_cards, card_span, cards_spans, combo_spans, draw_grid, legend, truncate, GridCell,
    Hit, ACCENT,
};
use holdem::{Card, Street};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use trainer::config::{FlopMode, HeroPosition, VillainRange};
use trainer::fmt::bb;
use trainer::{DecisionReview, LiveHand, Variant, Verdict};

pub fn draw(f: &mut Frame, app: &App) {
    app.hits.clear();
    match app.screen {
        Screen::Setup => draw_setup(f, app),
        Screen::Table => draw_table(f, app),
        Screen::Review => draw_review(f, app),
        Screen::History => draw_history(f, app),
        Screen::Study => crate::study_ui::draw_study(f, app),
        Screen::Sizes => draw_sizes(f, app),
        Screen::Sessions => crate::score_ui::draw_sessions(f, app),
        Screen::Score => crate::score_ui::draw_score(f, app),
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

fn verdict_color(v: Verdict) -> Color {
    match v {
        Verdict::Best => Color::Green,
        Verdict::Good => Color::LightGreen,
        Verdict::Inaccuracy => Color::Yellow,
        Verdict::Mistake => Color::LightRed,
        Verdict::Blunder => Color::Red,
    }
}

pub(crate) fn title_bar(app: &App, extra: &str) -> Paragraph<'static> {
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
    if let Some(r) = app.session.report() {
        spans.push(Span::raw(format!(" · Freq score {:.0} [G]", r.calibration.score())));
    }
    if st.reviews_pending > 0 {
        spans.push(Span::styled(
            format!(" · {} review(s) solving", st.reviews_pending),
            Style::default().fg(Color::Yellow),
        ));
    }
    Paragraph::new(Line::from(spans))
}

/// Short value of a setup row and a longer explanation (shown for the
/// selected row).
fn setup_value(app: &App, item: SetupItem) -> (String, String) {
    let cfg = &app.session.config;
    let s = |a: &str, b: &str| (a.to_string(), b.to_string());
    match item {
        SetupItem::Session => {
            let hands = app.session.history.len();
            match (&app.sessions_root, app.session.save_id()) {
                (None, _) => s("Not saved", "Saving is off (save_sessions = false in the config)."),
                (Some(root), Some(id)) => (
                    format!("Saved · {hands} hands"),
                    format!(
                        "Saved as {} after every hand. Enter: start a new session, continue or delete another.",
                        root.join(id).display()
                    ),
                ),
                (Some(root), None) if hands > 0 => (
                    format!("Saving · {hands} hands"),
                    format!("Hands are saved in {} as their reviews finish.", root.display()),
                ),
                (Some(root), None) => (
                    if app.sessions.is_empty() { "New".into() } else { format!("New ({} saved)", app.sessions.len()) },
                    format!(
                        "Saved in {} once a hand is finished. Enter: continue or delete saved sessions.",
                        root.display()
                    ),
                ),
            }
        }
        SetupItem::Rng => {
            if cfg.show_rng {
                s(
                    "Shown",
                    "A random number 0-99 at each of your decisions, to play mixes: with a 30% bet, \
                     bet on 0-29. The review shows what the solver's mix makes of it.",
                )
            } else {
                s("Hidden", "No random number at your decisions.")
            }
        }
        SetupItem::ScoreWeight => {
            if cfg.scoring.pot_weighted {
                s("By pot size", "Frequency score: decisions count in proportion to the pot, so leaks in big pots weigh more.")
            } else {
                s("Equal", "Frequency score: every decision counts the same.")
            }
        }
        SetupItem::Mode => match app.mode {
            Mode::Play => s(
                "Play hands",
                "Play against the AI; after each hand the solver grades your postflop decisions.",
            ),
            Mode::Study => s(
                "Solver study",
                "Browse the preflop charts and the solver's postflop strategy in any spot, GTO Wizard style.",
            ),
        },
        SetupItem::Table if app.mode == Mode::Study => match cfg.players {
            2 => s("Heads-up", "Preflop lines come from the heads-up solver charts."),
            6 => s("6-max", "Preflop lines come from the 6-max solver charts."),
            9 => s("9-max", "9-max preflop lines use position-aware rules (no 9-max solver charts)."),
            n => (format!("{n} players"), String::new()),
        },
        SetupItem::Table => match cfg.players {
            2 => s("Heads-up", "Two players."),
            6 => s("6-max", "Six players."),
            9 => s("9-max", "Full ring; preflop uses position-aware rules (no 9-max solver charts)."),
            n => (format!("{n} players"), String::new()),
        },
        SetupItem::Opponent => match app.current_opponent_choice() {
            OpponentChoice::Game(g) if cfg.players == 2 => {
                (g.label().to_string(), format!("Your opponent is drawn from: {}.", g.description()))
            }
            OpponentChoice::Game(g) => {
                (g.label().to_string(), format!("Seats are filled with {}.", g.description()))
            }
            OpponentChoice::Style(name) => {
                let d = cfg.profile_by_name(&name).map(|p| p.description).unwrap_or_default();
                (format!("All {name}"), d)
            }
        },
        SetupItem::Count(i) => {
            let name = cfg.profiles().get(i).map(|p| p.name.clone()).unwrap_or_default();
            let n = cfg.custom_table.get(&name).copied().unwrap_or(0);
            let d = cfg.profile_by_name(&name).map(|p| p.description).unwrap_or_default();
            (format!("{n} at the table"), format!("{name}: {d}"))
        }
        SetupItem::SeatChanges => {
            let label = SEAT_CHANGE_PRESETS
                .iter()
                .find(|&&(p, _)| (p - cfg.seat_change_pct).abs() < 1e-9)
                .map(|&(_, l)| l.to_string())
                .unwrap_or_else(|| format!("{}% chance per player and hand", cfg.seat_change_pct));
            match label.split_once(": ") {
                Some((a, b)) => (a.to_string(), format!("{b}.")),
                None => (label, String::new()),
            }
        }
        SetupItem::Position => match (cfg.hero_position, cfg.players) {
            (HeroPosition::Alternate, 2) => s("Alternate", "Alternate between SB (button) and BB."),
            (HeroPosition::Alternate, _) => s("Rotate", "The button moves every hand."),
            (HeroPosition::Random, _) => s("Random", "A random seat every hand."),
            (HeroPosition::Sb, 2) => s("Always SB", "Always the small blind (the button heads-up)."),
            (HeroPosition::Sb, _) => s("Always SB", "Always the small blind."),
            (HeroPosition::Bb, _) => s("Always BB", "Always the big blind."),
        },
        SetupItem::Stack => (
            format!("{}bb", cfg.stack_bb),
            if (cfg.stack_bb - 100.0).abs() > 1e-9 {
                "The built-in preflop charts assume 100bb stacks.".into()
            } else {
                "Starting stacks.".into()
            },
        ),
        SetupItem::Rake => {
            if cfg.rake_pct <= 0.0 {
                s("None", "No rake.")
            } else {
                (
                    format!("{}%, cap {}bb", cfg.rake_pct, cfg.rake_cap_bb),
                    format!(
                        "{}% of the pot, capped at {}bb (no flop, no drop).",
                        cfg.rake_pct, cfg.rake_cap_bb
                    ),
                )
            }
        }
        SetupItem::FlopSolve => match cfg.solver.flop_mode {
            FlopMode::Fast => (
                "Fast".into(),
                format!(
                    "Every turn card and {} sampled river cards per turn: ~4x faster, close to exact.",
                    cfg.solver.flop_river_samples
                ),
            ),
            FlopMode::Exact => (
                "Exact".into(),
                format!("Every runout: up to {} GB of RAM and minutes per flop.", cfg.solver.max_memory_gb),
            ),
        },
        SetupItem::ThreeWay => {
            if cfg.solver.multiway_exact_removal {
                s(
                    "Exact card removal",
                    "3-way pots account for the two opponents blocking each other's cards (about 2.5x slower \
                     than approximate; a 3-way flop takes under a minute on a 4-core machine).",
                )
            } else {
                s(
                    "Approximate",
                    "3-way pots treat the two opponents' hands as independent: close with wide ranges, \
                     but off by several % of the pot when narrow ranges overlap (squeeze pots).",
                )
            }
        }
        SetupItem::BetSizes => {
            let p = &cfg.menu.postflop;
            let st =
                |s: &holdem::StreetSizes| s.bet.iter().map(|b| b.to_string()).collect::<Vec<_>>().join("/");
            (
                if cfg.menu == holdem::BetMenu::default() { "Default".into() } else { "Custom".into() },
                format!(
                    "Open {}bb · flop {} · turn {} · river {}. Press Enter to edit.",
                    cfg.menu.preflop.open_bb,
                    st(&p.flop),
                    st(&p.turn),
                    st(&p.river)
                ),
            )
        }
        SetupItem::VillainRange => match cfg.solver.villain_range {
            VillainRange::Gto => s("GTO ranges", "The review assumes villains play the charts preflop."),
            VillainRange::Actual => {
                s("Actual range", "The review uses each villain style's own preflop range.")
            }
            VillainRange::Both => s("Both", "Both solves side by side (twice the solving time)."),
        },
        SetupItem::RevealStyle => {
            if cfg.show_opponent_style {
                s("On", "Opponents' styles are shown during the hand.")
            } else {
                s("Off", "Opponents' styles are shown in the review.")
            }
        }
    }
}

fn draw_setup(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(10), Constraint::Length(2)])
        .split(area);
    f.render_widget(title_bar(app, "Setup"), rows[0]);
    let cfg = &app.session.config;
    let label = |item: SetupItem| -> String {
        match item {
            SetupItem::Session => "Session",
            SetupItem::Rng => "Random number",
            SetupItem::ScoreWeight => "Frequency score weighting",
            SetupItem::Mode => "Mode",
            SetupItem::Table => "Table",
            SetupItem::Opponent => {
                if cfg.players == 2 {
                    "Opponent"
                } else {
                    "Opponents"
                }
            }
            SetupItem::Count(i) => {
                return format!("  {}", cfg.profiles().get(i).map(|p| p.name.clone()).unwrap_or_default())
            }
            SetupItem::SeatChanges => "New players sit down",
            SetupItem::Position => "Your position",
            SetupItem::Stack => "Stacks",
            SetupItem::Rake => "Rake",
            SetupItem::FlopSolve => "Flop solving",
            SetupItem::ThreeWay => "3-way pots",
            SetupItem::BetSizes => "Bet sizes",
            SetupItem::VillainRange => "Villain range in review",
            SetupItem::RevealStyle => "Reveal style during hand",
        }
        .to_string()
    };
    let setup_items = app.setup_items();
    let label_w = 26;
    let items: Vec<ListItem> = setup_items
        .iter()
        .map(|&it| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<label_w$}", label(it)), Style::default().fg(ACCENT)),
                Span::raw("◂ "),
                Span::styled(setup_value(app, it).0, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" ▸"),
            ]))
        })
        .collect();
    let cursor = app.setup_cursor.min(setup_items.len() - 1);
    let mut state = ListState::default().with_selected(Some(cursor));
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Settings "))
        .highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");
    let mid = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(setup_items.len() as u16 + 2), Constraint::Min(4)])
        .split(rows[1]);
    f.render_stateful_widget(list, mid[0], &mut state);
    let (_, detail) = setup_value(app, setup_items[cursor]);
    let mut about = vec![
        Line::from(vec![
            Span::styled(format!("{}: ", label(setup_items[cursor]).trim()), Style::default().fg(ACCENT)),
            Span::raw(detail),
        ]),
        Line::from(""),
    ];
    about.extend(match app.mode {
        Mode::Play => vec![
            Line::from(
                "Play NLHE hands (heads-up, 6-max or 9-max) from preflop to the river against AI \
                 opponents with different styles. After each hand, press R to see what the solver \
                 would have done in every postflop spot you played, with the EV of each option.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Preflop decisions are not graded yet; the charts set the AIs' play and the starting ranges. \
                 Text too small? Zoom your terminal (Ctrl/Cmd +): every screen adapts down to about 100×30.",
                Style::default().fg(Color::DarkGray),
            )),
        ],
        Mode::Study => vec![
            Line::from(
                "Click through a preflop line in the charts (seeing each player's range), deal a flop, \
                 and browse the solver's strategy: every hand's mix and EV, both ranges, any bet or \
                 raise. Turn and river are solved when you deal them. Press E there to type your own \
                 ranges, pot and board instead.",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Heads-up and 3-way pots (3-way solves sample runouts and have no equilibrium guarantee). Fast flop solving is recommended here.",
                Style::default().fg(Color::DarkGray),
            )),
        ],
    });
    f.render_widget(
        Paragraph::new(about)
            .block(Block::default().borders(Borders::ALL).title(" About "))
            .wrap(Wrap { trim: true }),
        mid[1],
    );
    let keys = match app.mode {
        Mode::Play => {
            "↑/↓ select · ←/→ change · Enter play · O sessions · G score · W save settings · ? help · Q quit"
        }
        Mode::Study => "↑/↓ select · ←/→ change · Enter study · W save settings · ? help · Q quit",
    };
    f.render_widget(
        Paragraph::new(truncate(keys, area.width as usize)).block(Block::default().borders(Borders::TOP)),
        rows[2],
    );
}

/// One seat on the felt; `width` is the felt's inner width.
fn seat_line(hand: &LiveHand, seat: usize, show_cards: bool, app: &App, width: u16) -> Line<'static> {
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
    let name_w = (width as usize).saturating_sub(44).clamp(10, 26);
    let to_act = st.to_act == Some(seat);
    spans.push(Span::styled(
        format!("{} {:<name_w$} ", if to_act { "▶" } else { " " }, truncate(&name, name_w)),
        Style::default().fg(if is_hero { ACCENT } else { Color::White }).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(format!("{:<4}", st.position_name(seat)), Style::default().fg(Color::Yellow)));
    spans.push(Span::raw(format!(" {:>8}  ", bb(st.stacks[seat]))));
    if st.folded[seat] {
        spans.push(Span::styled("folded", Style::default().fg(Color::DarkGray)));
    } else if show_cards {
        spans.extend(cards_spans(&st.hole[seat]));
    } else {
        spans.push(Span::styled("▒▒ ▒▒", Style::default().fg(Color::Blue)));
    }
    if st.street_commit[seat] > 0 {
        spans.push(Span::styled(
            format!("  bet {}", bb(st.street_commit[seat])),
            Style::default().fg(Color::LightYellow),
        ));
    }
    if st.is_all_in(seat) {
        spans.push(Span::styled(" ALL-IN", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)));
    }
    Line::from(spans)
}

/// Action buttons laid out in lines of at most `width` columns; returns the
/// lines and each button's (line, column, width).
fn action_buttons(app: &App, width: u16) -> (Vec<Line<'static>>, Vec<(u16, u16, u16)>) {
    let Some(hand) = app.current() else { return (Vec::new(), Vec::new()) };
    let Some(ctx) = hand.action_context() else { return (Vec::new(), Vec::new()) };
    let mut lines: Vec<Vec<Span>> = vec![Vec::new()];
    let mut boxes = Vec::new();
    let mut x = 0u16;
    for (key, a) in app.action_keys() {
        let k = format!(" [{}] ", key.to_ascii_uppercase());
        let l = format!(" {}  ", trainer::fmt::action_label(a, &ctx));
        let w = (k.chars().count() + l.chars().count()) as u16;
        if x > 0 && x + w > width {
            lines.push(Vec::new());
            x = 0;
        }
        boxes.push(((lines.len() - 1) as u16, x, w));
        let line = lines.last_mut().unwrap();
        line.push(Span::styled(k, Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD)));
        line.push(Span::styled(l, Style::default().add_modifier(Modifier::BOLD)));
        line.push(Span::raw(" "));
        x += w + 1;
    }
    (lines.into_iter().map(Line::from).collect(), boxes)
}

fn draw_table(f: &mut Frame, app: &App) {
    let area = f.area();
    let Some(hand) = app.current() else { return };
    let (buttons, boxes) = if hand.hero_to_act() {
        action_buttons(app, area.width.saturating_sub(2))
    } else {
        (Vec::new(), Vec::new())
    };
    let actions_h = 3 + buttons.len().max(1) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(10),
            Constraint::Length(actions_h),
            Constraint::Length(2),
        ])
        .split(area);
    f.render_widget(title_bar(app, &format!("Hand #{}", hand.id)), rows[0]);
    let log_w = (area.width * 38 / 100).clamp(30, 60);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(50), Constraint::Length(log_w)])
        .split(rows[1]);

    // Felt: opponents in seating order starting left of the hero (their
    // cards are shown at showdown), the board and pot, then the hero.
    let st = &hand.state;
    let n = st.num_players();
    let showdown = hand.is_finished() && hand.settlement().is_some_and(|s| s.showdown);
    let inner_w = cols[0].width.saturating_sub(2);
    let inner_h = cols[0].height.saturating_sub(2) as usize;
    let base = n + 2; // seats + board + pot
    let big_board = inner_h >= base + 2 + 2;
    let big_hero = big_board && inner_h >= base + 2 + 3 + 3;
    let used = base + if big_board { 2 } else { 0 } + if big_hero { 3 } else { 0 };
    let spare = inner_h.saturating_sub(used);
    let gap = (spare / 3).min(3);
    let mut felt = vec![Line::from(""); (spare - 2 * gap) / 2];
    for k in 1..n {
        let seat = (hand.hero + k) % n;
        felt.push(seat_line(hand, seat, showdown && !st.folded[seat], app, inner_w));
    }
    felt.extend(std::iter::repeat_n(Line::from(""), gap));
    let board: Vec<Option<Card>> = (0..5).map(|i| st.board.get(i).copied()).collect();
    if big_board {
        let pad = "       ";
        for (i, l) in big_cards(&board).into_iter().enumerate() {
            let mut spans = vec![Span::raw(if i == 1 { "  Board" } else { "       " }), Span::raw(pad)];
            spans.extend(l.spans);
            felt.push(Line::from(spans));
        }
    } else {
        let mut spans = vec![Span::raw("  Board  ")];
        for (i, c) in board.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw(" "));
            }
            match c {
                Some(c) => spans.push(card_span(*c)),
                None => spans.push(Span::styled("··", Style::default().fg(Color::DarkGray))),
            }
        }
        felt.push(Line::from(spans));
    }
    felt.push(Line::from(vec![
        Span::raw("  Pot    "),
        Span::styled(bb(st.pot()), Style::default().fg(Color::LightYellow).add_modifier(Modifier::BOLD)),
        Span::styled(format!("   {}", st.street), Style::default().fg(Color::DarkGray)),
    ]));
    felt.extend(std::iter::repeat_n(Line::from(""), gap));
    felt.push(seat_line(hand, hand.hero, true, app, inner_w));
    if big_hero {
        let hole: Vec<Option<Card>> = st.hole[hand.hero].iter().copied().map(Some).collect();
        for l in big_cards(&hole) {
            let mut spans = vec![Span::raw("    ")];
            spans.extend(l.spans);
            felt.push(Line::from(spans));
        }
    }
    f.render_widget(
        Paragraph::new(felt).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(" Table "),
        ),
        cols[0],
    );

    // Log: the latest lines that fit (long lines wrap).
    let log_w = cols[1].width.saturating_sub(2).max(1) as usize;
    let log_height = cols[1].height.saturating_sub(2) as usize;
    let mut used_rows = 0;
    let mut start = hand.log.len();
    while start > 0 {
        let rows_needed = hand.log[start - 1].text.chars().count().div_ceil(log_w).max(1);
        if used_rows + rows_needed > log_height {
            break;
        }
        used_rows += rows_needed;
        start -= 1;
    }
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
        for (i, &(line, x, w)) in boxes.iter().enumerate() {
            app.hits.add(
                Rect { x: rows[2].x + 1 + x, y: rows[2].y + 1 + line, width: w, height: 1 },
                Hit::Action(i),
            );
        }
        lines.extend(buttons);
        let mut info = format!("To call {} · pot {}", bb(ctx.to_call), bb(ctx.pot));
        if st.street == Street::Preflop {
            info.push_str(" · X/C check/call · A max raise");
        }
        info.push_str(" · or click a button");
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
            Span::raw("   [N] next hand   [R] solver review   [H] history   [G] score   [S] settings"),
        ]));
    } else {
        lines.push(Line::from(Span::styled("Villain is thinking…", Style::default().fg(Color::DarkGray))));
    }
    let mut title = vec![Span::raw(" Your move ")];
    if let Some(roll) = hand.current_roll().filter(|_| app.session.config.show_rng) {
        title.push(Span::styled(
            format!(" RNG {roll:02} "),
            Style::default().fg(Color::Black).bg(Color::LightYellow).add_modifier(Modifier::BOLD),
        ));
        title.push(Span::raw(" "));
    }
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(Line::from(title))),
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
            let text = r.with_status(|s| {
                let mut text = format!("Solver: {}", s.stage);
                if let Some(p) = s.progress {
                    text.push_str(&format!(
                        " — iteration {} · exploitability {:.2}% pot",
                        p.iteration, p.exploitability_pct
                    ));
                }
                text.push_str(&elapsed(s));
                if s.finished && s.error.is_none() {
                    text = format!(
                        "Solver: review ready ({} decisions graded) — press R after the hand",
                        s.decisions.len()
                    );
                }
                text
            });
            Line::from(Span::styled(
                truncate(&text, area.width as usize),
                Style::default().fg(Color::LightMagenta),
            ))
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

    let decisions = app.visible_decisions();
    let selected = app.review_selected.min(decisions.len().saturating_sub(1));
    let empty_msg = if decisions.is_empty() && status.finished {
        " No postflop decisions to grade "
    } else {
        " Your decisions "
    };
    // Wide windows list the decisions on the left; narrower ones show them
    // as a strip above the details, leaving the full width to the grid.
    let detail_area = if area.width >= 150 {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(38), Constraint::Min(40)])
            .split(rows[2]);
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
        let mut state = ListState::default().with_selected((!decisions.is_empty()).then_some(selected));
        f.render_stateful_widget(
            List::new(items)
                .block(Block::default().borders(Borders::ALL).title(empty_msg))
                .highlight_style(Style::default().bg(Color::DarkGray))
                .highlight_symbol("▶ "),
            cols[0],
            &mut state,
        );
        for i in 0..decisions.len().min(cols[0].height.saturating_sub(2) as usize) {
            app.hits.add(
                Rect { x: cols[0].x, y: cols[0].y + 1 + i as u16, width: cols[0].width, height: 1 },
                Hit::Decision(i),
            );
        }
        cols[1]
    } else {
        let mut lines: Vec<Vec<Span>> =
            vec![vec![Span::styled(format!("{}: ", empty_msg.trim()), Style::default().fg(Color::DarkGray))]];
        let mut x = lines[0][0].content.chars().count() as u16;
        let mut boxes = Vec::new();
        for (i, d) in decisions.iter().enumerate() {
            let text =
                format!(" {} {} {} ", d.street.name(), truncate(&d.labels[d.chosen], 14), d.verdict.symbol());
            let w = text.chars().count() as u16;
            if x + w + 1 > area.width && x > 0 {
                lines.push(Vec::new());
                x = 0;
            }
            let mut st = Style::default().fg(verdict_color(d.verdict)).add_modifier(Modifier::BOLD);
            if i == selected {
                st = st.bg(Color::Indexed(238)).add_modifier(Modifier::UNDERLINED);
            }
            boxes.push(((lines.len() - 1) as u16, x, w));
            let line = lines.last_mut().unwrap();
            line.push(Span::styled(text, st));
            line.push(Span::raw(" "));
            x += w + 1;
        }
        let strip_h = lines.len() as u16 + 1;
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(strip_h), Constraint::Min(6)])
            .split(rows[2]);
        for (i, &(line, x, w)) in boxes.iter().enumerate() {
            app.hits
                .add(Rect { x: parts[0].x + x, y: parts[0].y + line, width: w, height: 1 }, Hit::Decision(i));
        }
        f.render_widget(
            Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
                .block(Block::default().borders(Borders::BOTTOM)),
            parts[0],
        );
        parts[1]
    };
    if let Some(d) = decisions.get(app.review_selected.min(decisions.len().saturating_sub(1))) {
        let roll = (app.session.config.show_rng && d.spot.categories.len() == d.actions.len())
            .then(|| hand.roll(d.spot.history_index));
        draw_decision(f, detail_area, d, &status.notes, roll);
    } else {
        let mut lines: Vec<Line> = status.notes.iter().map(|n| Line::from(format!("• {n}"))).collect();
        if lines.is_empty() {
            lines.push(Line::from("Waiting for the solver…"));
        }
        f.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: true }).block(Block::default().borders(Borders::ALL)),
            detail_area,
        );
    }
    f.render_widget(
        Paragraph::new(truncate(
            "↑/↓ or click a decision · Tab switch range variant · N next hand · Esc back · Q quit",
            area.width as usize,
        )),
        rows[3],
    );
}

fn draw_decision(f: &mut Frame, area: Rect, d: &DecisionReview, hand_notes: &[String], roll: Option<u8>) {
    let block =
        Block::default().borders(Borders::ALL).title(format!(" {} — {} ", d.street, d.variant.label()));
    let inner = block.inner(area);
    f.render_widget(block, area);
    // Grid beside the text when both fit (the grid grows with the space),
    // otherwise below it.
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
    if let Some(roll) = roll {
        let pick = trainer::scoring::rng_action(&d.hero_strategy, roll);
        let same = pick == d.chosen;
        lines.push(Line::from(vec![
            Span::styled(format!("RNG {roll:02}"), Style::default().fg(Color::LightYellow)),
            Span::raw(" → the solver's mix plays "),
            Span::styled(d.labels[pick].clone(), Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(if same { "  (as you did)" } else { "" }, Style::default().fg(Color::DarkGray)),
        ]));
    }
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
                format!(" {:<w$}", truncate(&d.labels[i], label_w - 1), w = label_w),
                Style::default().fg(action_color(&d.actions, i)),
            ),
            Span::styled(bar(d.hero_strategy[i], bar_w), Style::default().fg(action_color(&d.actions, i))),
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
                format!(" {:<w$}", truncate(&d.labels[i], label_w - 1), w = label_w),
                Style::default().fg(action_color(&d.actions, i)),
            ),
            Span::styled(
                bar(d.range_strategy[i] as f32, bar_w),
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
    draw_review_grid(f, parts[2], d);
}

/// 13x13 grid of the hero's range at a decision; each cell is colored by
/// its action mix (coloring by the most frequent action alone would hide
/// low-frequency bluffs and make betting ranges look value-heavy).
fn draw_review_grid(f: &mut Frame, area: Rect, d: &DecisionReview) {
    let cells: Vec<GridCell> = d
        .grid
        .iter()
        .map(|c| match c {
            Some(c) => GridCell {
                fill: 1.0,
                mix: c.strategy.iter().enumerate().map(|(a, &w)| (w, action_color(&d.actions, a))).collect(),
                sub: None,
            },
            None => GridCell::default(),
        })
        .collect();
    let entries: Vec<(String, Color)> =
        d.labels.iter().enumerate().map(|(i, l)| (l.clone(), action_color(&d.actions, i))).collect();
    let title = Line::from(Span::styled(
        "Your range here (cells colored by action mix):",
        Style::default().fg(ACCENT),
    ));
    draw_grid(f, area, title, &cells, legend(&entries, 16), None, None);
}

fn draw_sizes(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(10), Constraint::Length(1)])
        .split(area);
    f.render_widget(title_bar(app, "Bet sizes"), rows[0]);
    if let Some(ed) = &app.sizes {
        ed.draw(f, rows[1], &app.session.config, &app.hits);
    }
    f.render_widget(
        Paragraph::new(truncate(
            "↑/↓ field · Enter edit/confirm · ←/→ change · Ctrl+U clear · Esc back (W on the setup screen saves)",
            area.width as usize,
        )),
        rows[2],
    );
}

fn draw_history(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(5), Constraint::Length(1)])
        .split(area);
    f.render_widget(title_bar(app, "History"), rows[0]);
    // Only the rows that fit are built (long sessions have thousands).
    let window = rows[1].height.saturating_sub(2).max(1) as usize;
    let n = app.session.history.len();
    let selected = app.history_selected.min(n.saturating_sub(1));
    let mut offset = app.history_offset.get().min(n.saturating_sub(1));
    if selected < offset {
        offset = selected;
    } else if selected >= offset + window {
        offset = selected + 1 - window;
    }
    app.history_offset.set(offset);
    let items: Vec<ListItem> = app.session.history[offset..(offset + window).min(n)]
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
                Some(r) => r.with_status(|st| {
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
                }),
            }
            ListItem::new(Line::from(spans))
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(selected - offset));
    f.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(" Finished hands ({n}) ")))
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
        Line::from("  N next hand · R solver review (after the hand) · H history · G frequency score · S settings · Q quit"),
        Line::from("  RNG nn: a random number 0-99 for mixing (low numbers: the most aggressive option)"),
        Line::from(""),
        Line::from("Review"),
        Line::from("  ↑/↓ choose a decision · Tab switch between GTO and villain's-actual-range solves"),
        Line::from(""),
        Line::from("Grades (EV lost vs. the best action, as % of the pot)"),
        Line::from("  ✓ Best ≤1% · ✓ Good ≤3% · ?! Inaccuracy ≤6% · ? Mistake ≤12% · ?? Blunder >12%"),
        Line::from("  The frequency score (G) checks how often you take each option against the solver's mixes."),
        Line::from(""),
        Line::from("Sessions"),
        Line::from("  Hands are saved as you play. O on the setup screen: continue a session, start a new one, delete one (D)."),
        Line::from(""),
        Line::from("Solver study"),
        Line::from("  F / C / 1-9 take an action · arrow keys pick a hand (or click it) · Tab cycles strategy / ranges"),
        Line::from("  Backspace goes back · G back to the start · N new random board · E type a spot · P preflop"),
        Line::from("  At the end of a street, type the next card (e.g. 9h) or press Enter for a random one."),
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
