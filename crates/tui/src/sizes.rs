//! Bet-size editor (opened from the setup screen's "Bet sizes" row).
//!
//! Edits the menu the table, the AI and every solve use, plus the
//! simplified turn/river sizes used inside flop solves. Changes apply when
//! a field is confirmed; W on the setup screen saves them to the config.

use crate::widgets::{truncate, Hit, HitMap, ACCENT};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use holdem::menu::postflop_actions;
use holdem::{Action, BetMenu, BetSize, PostflopMenu, Range, Situation, Street};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use solver::{BettingState, ChanceMode, SolveConfig, Solver};
use trainer::fmt::{action_label, ActionContext, BB};
use trainer::TrainerConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    Open,
    Iso,
    Reraises,
    Limp,
    Bet(Street),
    Raise(Street),
    MaxRaises,
    AddAllin,
    FutureBet(Street),
    Reset,
}

pub const ROWS: [Row; 15] = [
    Row::Open,
    Row::Iso,
    Row::Reraises,
    Row::Limp,
    Row::Bet(Street::Flop),
    Row::Raise(Street::Flop),
    Row::Bet(Street::Turn),
    Row::Raise(Street::Turn),
    Row::Bet(Street::River),
    Row::Raise(Street::River),
    Row::MaxRaises,
    Row::AddAllin,
    Row::FutureBet(Street::Turn),
    Row::FutureBet(Street::River),
    Row::Reset,
];

const ADD_ALLIN: [f64; 5] = [0.0, 1.0, 1.5, 2.0, 3.0];

/// Sizes of a reference heads-up single-raised pot's trees (turn with every
/// river; flop with every runout and the flop-solve sizes later).
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    turn_nodes: u64,
    flop_gb: f64,
}

pub struct SizesEditor {
    pub cursor: usize,
    /// Text being edited (`None` when not editing).
    pub editing: Option<String>,
    pub error: Option<String>,
    cost: std::cell::Cell<Option<Cost>>,
    default_cost: Cost,
}

fn sizes_text(v: &[BetSize]) -> String {
    if v.is_empty() {
        "none".into()
    } else {
        v.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
    }
}

fn num(v: f64) -> String {
    let s = format!("{v:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Parses "33%, 75%, allin" (a plain number is a % of the pot for bets and
/// a multiple for raises).
fn parse_sizes(text: &str, raises: bool) -> Result<Vec<BetSize>, String> {
    let t = text.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for tok in t.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()) {
        let tok = if tok.parse::<f64>().is_ok() {
            if raises {
                format!("{tok}x")
            } else {
                format!("{tok}%")
            }
        } else {
            tok.to_string()
        };
        let size: BetSize = tok.parse().map_err(|e: holdem::ParseError| e.to_string())?;
        if !raises && matches!(size, BetSize::Mult(_)) {
            return Err("bets are a % of the pot (e.g. 75%) or allin; multiples (3x) are for raises".into());
        }
        out.push(size);
    }
    Ok(out)
}

impl SizesEditor {
    pub fn new() -> SizesEditor {
        let default_cost = cost(&BetMenu::default(), &trainer::SolverSettings::default().flop_future_menu);
        SizesEditor { cursor: 0, editing: None, error: None, cost: std::cell::Cell::new(None), default_cost }
    }

    fn row(&self) -> Row {
        ROWS[self.cursor.min(ROWS.len() - 1)]
    }

    /// Current text of a text row.
    fn text(cfg: &TrainerConfig, row: Row) -> Option<String> {
        let m = &cfg.menu;
        Some(match row {
            Row::Open => num(m.preflop.open_bb),
            Row::Iso => num(m.preflop.iso_bb),
            Row::Reraises => {
                m.preflop.reraise_mult.iter().map(|x| format!("{}x", num(*x))).collect::<Vec<_>>().join(", ")
            }
            Row::Bet(s) => sizes_text(&m.postflop.street(s).bet),
            Row::Raise(s) => sizes_text(&m.postflop.street(s).raise),
            Row::FutureBet(s) => sizes_text(&cfg.solver.flop_future_menu.street(s).bet),
            _ => return None,
        })
    }

    /// Applies an edited text; `Err` keeps the old value.
    fn apply(cfg: &mut TrainerConfig, row: Row, text: &str) -> Result<(), String> {
        let bb = |t: &str, min: f64| -> Result<f64, String> {
            let v: f64 = t
                .trim()
                .trim_end_matches("bb")
                .trim()
                .parse()
                .map_err(|_| "enter a number of big blinds".to_string())?;
            if v < min || v > 50.0 {
                return Err(format!("must be between {min} and 50 big blinds"));
            }
            Ok(v)
        };
        let m = &mut cfg.menu;
        match row {
            Row::Open => m.preflop.open_bb = bb(text, 2.0)?,
            Row::Iso => m.preflop.iso_bb = bb(text, 2.0)?,
            Row::Reraises => {
                let mut v = Vec::new();
                for tok in text.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()) {
                    let x: f64 = tok
                        .trim_end_matches('x')
                        .parse()
                        .map_err(|_| format!("{tok:?}: use multiples like 3x"))?;
                    if !(1.5..=10.0).contains(&x) {
                        return Err("re-raise multiples must be between 1.5x and 10x".into());
                    }
                    v.push(x);
                }
                m.preflop.reraise_mult = v;
            }
            Row::Bet(s) => m.postflop.street_mut(s).bet = parse_sizes(text, false)?,
            Row::Raise(s) => m.postflop.street_mut(s).raise = parse_sizes(text, true)?,
            Row::FutureBet(s) => cfg.solver.flop_future_menu.street_mut(s).bet = parse_sizes(text, false)?,
            _ => {}
        }
        Ok(())
    }

    fn step(cfg: &mut TrainerConfig, row: Row, dir: i32) {
        let m = &mut cfg.menu;
        match row {
            Row::Limp => m.preflop.allow_limp = !m.preflop.allow_limp,
            Row::MaxRaises => m.postflop.max_raises = (m.postflop.max_raises as i32 + dir).clamp(1, 6) as u32,
            Row::AddAllin => {
                let cur = ADD_ALLIN
                    .iter()
                    .position(|&x| (x - m.postflop.add_allin_threshold).abs() < 1e-9)
                    .unwrap_or(2);
                let next = (cur as i32 + dir).rem_euclid(ADD_ALLIN.len() as i32) as usize;
                m.postflop.add_allin_threshold = ADD_ALLIN[next];
            }
            _ => {}
        }
    }

    pub fn on_click(&mut self, hit: Hit) {
        if let Hit::Field(i) = hit {
            self.cursor = i;
            self.editing = None;
        }
    }

    /// Handles a key; returns `true` when the editor should close.
    pub fn on_key(&mut self, key: KeyEvent, cfg: &mut TrainerConfig) -> bool {
        if let Some(text) = &mut self.editing {
            match key.code {
                KeyCode::Enter | KeyCode::Tab => {
                    let text = text.clone();
                    match Self::apply(cfg, self.row(), &text) {
                        Ok(()) => {
                            self.editing = None;
                            self.error = None;
                            self.cost.set(None);
                        }
                        Err(e) => self.error = Some(e),
                    }
                }
                KeyCode::Esc => {
                    self.editing = None;
                    self.error = None;
                }
                KeyCode::Backspace => {
                    text.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => text.clear(),
                KeyCode::Char(c) => text.push(c),
                _ => {}
            }
            return false;
        }
        let n = ROWS.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.cursor = self.cursor.checked_sub(1).unwrap_or(n - 1),
            KeyCode::Down | KeyCode::Char('j') => self.cursor = (self.cursor + 1) % n,
            KeyCode::Esc | KeyCode::Char('b') => return true,
            KeyCode::Left | KeyCode::Char('h') => {
                Self::step(cfg, self.row(), -1);
                self.cost.set(None);
            }
            KeyCode::Right | KeyCode::Char('l') => {
                Self::step(cfg, self.row(), 1);
                self.cost.set(None);
            }
            KeyCode::Enter | KeyCode::Char(' ') => match self.row() {
                Row::Reset => {
                    cfg.menu = BetMenu::default();
                    cfg.solver.flop_future_menu = trainer::SolverSettings::default().flop_future_menu;
                    self.cost.set(None);
                    self.error = None;
                }
                Row::Limp | Row::MaxRaises | Row::AddAllin => {
                    Self::step(cfg, self.row(), 1);
                    self.cost.set(None);
                }
                row => {
                    let t = Self::text(cfg, row).unwrap_or_default();
                    self.editing = Some(if t == "none" { String::new() } else { t });
                    self.error = None;
                }
            },
            _ => {}
        }
        false
    }

    pub fn draw(&self, f: &mut Frame, area: Rect, cfg: &TrainerConfig, hits: &HitMap) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(46), Constraint::Length((area.width / 2).clamp(40, 70))])
            .split(area);
        let m = &cfg.menu;
        let mut lines = vec![
            Line::from(Span::styled("Bet sizes", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))),
            Line::from(Span::styled(
                "Used at the table, by the AI and by every solve. Enter edits, ←/→ change, Esc back.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
        ];
        let label_w = 26;
        let text_w = (cols[0].width as usize).saturating_sub(label_w + 4).max(8);
        let y0 = cols[0].y + lines.len() as u16;
        let mut section = None;
        let mut y = y0;
        for (i, &row) in ROWS.iter().enumerate() {
            let head = match row {
                Row::Open => Some("Preflop"),
                Row::Bet(Street::Flop) => Some("Postflop"),
                Row::FutureBet(Street::Turn) => Some("Inside flop solves (later streets, simplified)"),
                _ => None,
            };
            if let Some(h) = head.filter(|_| head != section) {
                section = head;
                lines.push(Line::from(Span::styled(h, Style::default().fg(Color::Yellow))));
                y += 1;
            }
            let label = match row {
                Row::Open => "Open raise (bb)".to_string(),
                Row::Iso => "Raise over a limper (bb)".to_string(),
                Row::Reraises => "3-bet, 4-bet (x the bet)".to_string(),
                Row::Limp => "Open-limping".to_string(),
                Row::Bet(s) => format!("{} bets", s.name()),
                Row::Raise(s) => format!("{} raises", s.name()),
                Row::MaxRaises => "Max bets + raises / street".to_string(),
                Row::AddAllin => "Add all-in as an option".to_string(),
                Row::FutureBet(s) => format!("{} bet", s.name()),
                Row::Reset => String::new(),
            };
            let value = match row {
                Row::Limp => if m.preflop.allow_limp { "◂ allowed ▸" } else { "◂ not allowed ▸" }.to_string(),
                Row::MaxRaises => format!("◂ {} ▸", m.postflop.max_raises),
                Row::AddAllin => {
                    if m.postflop.add_allin_threshold <= 0.0 {
                        "◂ never ▸".into()
                    } else {
                        format!("◂ if ≤ {}× the pot ▸", num(m.postflop.add_allin_threshold))
                    }
                }
                Row::Reset => "[ Reset to defaults ]".into(),
                r => Self::text(cfg, r).unwrap_or_default(),
            };
            let selected = i == self.cursor;
            let style = if selected { Style::default().bg(Color::Indexed(238)) } else { Style::default() };
            let shown = match (&self.editing, selected) {
                (Some(t), true) => format!("{}▏", truncate(t, text_w.saturating_sub(1))),
                _ => truncate(&value, text_w),
            };
            lines.push(Line::from(vec![
                Span::styled(if selected { "▶ " } else { "  " }, style),
                Span::styled(format!("{:<label_w$}", truncate(&label, label_w)), style.fg(ACCENT)),
                Span::styled(
                    format!(" {shown}"),
                    if selected && self.editing.is_some() {
                        style.fg(Color::Yellow).add_modifier(Modifier::BOLD)
                    } else {
                        style.add_modifier(Modifier::BOLD)
                    },
                ),
            ]));
            hits.add(Rect { x: cols[0].x, y, width: cols[0].width, height: 1 }, Hit::Field(i));
            y += 1;
        }
        lines.push(Line::from(""));
        if let Some(e) = &self.error {
            lines.push(Line::from(Span::styled(e.clone(), Style::default().fg(Color::LightRed))));
        }
        f.render_widget(Paragraph::new(lines), cols[0]);

        // Explanation, preview and cost.
        let mut info = vec![Line::from(Span::styled(self.help(), Style::default())), Line::from("")];
        info.extend(preview(cfg, self.row()));
        let c = self.cost.get().unwrap_or_else(|| {
            let c = cost(&cfg.menu, &cfg.solver.flop_future_menu);
            self.cost.set(Some(c));
            c
        });
        let d = self.default_cost;
        info.push(Line::from(""));
        info.push(Line::from(Span::styled(
            "Solve cost (reference heads-up single-raised pot)",
            Style::default().fg(ACCENT),
        )));
        info.push(Line::from(format!(
            "Turn tree: {:.1}× the default sizes' ({} decisions)",
            c.turn_nodes as f64 / d.turn_nodes.max(1) as f64,
            c.turn_nodes
        )));
        info.push(Line::from(format!(
            "Exact flop: ~{:.1} GB ({:.1}× the default); fast flop is ~5× smaller",
            c.flop_gb,
            c.flop_gb / d.flop_gb.max(1e-9)
        )));
        if c.flop_gb > cfg.solver.max_memory_gb {
            info.push(Line::from(Span::styled(
                format!(
                    "Over the {} GB limit: exact flop solves will fall back to fast mode.",
                    num(cfg.solver.max_memory_gb)
                ),
                Style::default().fg(Color::Yellow),
            )));
        }
        f.render_widget(
            Paragraph::new(info).wrap(Wrap { trim: false }).block(Block::default().borders(Borders::LEFT)),
            cols[1],
        );
    }

    fn help(&self) -> String {
        match self.row() {
            Row::Open => "First raise when nobody has limped. The built-in preflop charts were solved with 2.5bb opens; with other sizes the AI and the review still use the charts' frequencies.",
            Row::Iso => "Raise over one limper; each extra limper adds 1bb.",
            Row::Reraises => "Re-raise sizes as multiples of the current bet: the 3-bet, then the 4-bet. Beyond the listed levels only all-in remains.",
            Row::Limp => "Whether players may open by calling the big blind.",
            Row::Bet(_) => "Bet sizes when nobody has bet yet on this street, as % of the pot (e.g. 33%, 75%, 150% for an overbet), or allin. Separate several with commas.",
            Row::Raise(_) => "Raise sizes facing a bet: multiples of the bet (3x), % of the pot after calling (100%), or allin.",
            Row::MaxRaises => "Bets plus raises allowed on one street; after that only fold or call remain.",
            Row::AddAllin => "Also offer all-in when it's at most this many times the pot.",
            Row::FutureBet(_) => "Flop solves use one bet size (plus all-in) on the turn and river to stay in memory. Reviews and study mode re-solve the turn and river with the full sizes above.",
            Row::Reset => "Back to the default sizes.",
        }
        .into()
    }
}

/// Options in a sample spot for the selected row.
fn preview(cfg: &TrainerConfig, row: Row) -> Vec<Line<'static>> {
    let (street, facing, menu) = match row {
        Row::Bet(s) => (s, false, cfg.menu.postflop.clone()),
        Row::Raise(s) => (s, true, cfg.menu.postflop.clone()),
        Row::MaxRaises | Row::AddAllin => (Street::Flop, false, cfg.menu.postflop.clone()),
        Row::FutureBet(s) => (s, false, cfg.solver.flop_future_menu.clone()),
        _ => return Vec::new(),
    };
    let pot = 1000;
    let stack = 9000;
    let bet = if facing { 500 } else { 0 };
    let sit = Situation {
        street,
        pot: pot + bet,
        to_call: bet,
        current_bet: bet,
        max_to: stack,
        min_raise_to: if facing { 2 * bet } else { BB },
        num_raises: u32::from(facing),
        limpers: 0,
        big_blind: BB,
    };
    let ctx =
        ActionContext { to_call: bet, pot: pot + bet, all_in_to: stack, facing_bet: facing, preflop: false };
    let labels: Vec<String> = postflop_actions(&sit, &menu)
        .into_iter()
        .filter(|a| matches!(a, Action::Raise(_)))
        .map(|a| action_label(a, &ctx))
        .collect();
    let title = if facing {
        format!("Example: {street}, 10bb pot, facing a 5bb bet, 90bb behind")
    } else {
        format!("Example: {street}, 10bb pot, 90bb behind")
    };
    vec![
        Line::from(Span::styled(title, Style::default().fg(Color::DarkGray))),
        Line::from(if labels.is_empty() { "(no bets or raises)".to_string() } else { labels.join(" · ") }),
    ]
}

/// Tree sizes of a reference heads-up single-raised pot.
fn cost(menu: &BetMenu, future: &PostflopMenu) -> Cost {
    let ip =
        Range::parse("22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,85s+,75s+,64s+,54s,A2o+,K8o+,Q9o+,J9o+,T9o").unwrap();
    let oop = Range::parse(
        "22-99,A2s-AJs,K2s-KJs,Q6s+,J7s+,T7s+,96s+,85s+,74s+,63s+,53s+,43s,A8o-AJo,K9o+,QTo+,JTo",
    )
    .unwrap();
    let ranges = [oop.into_weights(), ip.into_weights()];
    let est = |board: &str, street: Street, future: Option<&PostflopMenu>| {
        let board = holdem::card::parse_cards(board).unwrap();
        let mut c = SolveConfig::new(
            board,
            ranges.clone(),
            BettingState::street_start(street, 550, [9725, 9725], BB),
            menu.postflop.clone(),
        );
        c.future_menu = future.cloned();
        c.chance = ChanceMode::Exact;
        c.max_memory_bytes = u64::MAX;
        Solver::estimate(&c).ok()
    };
    let turn = est("Ks7d2c9h", Street::Turn, None);
    let flop = est("Ks7d2c", Street::Flop, Some(future));
    Cost {
        turn_nodes: turn.map(|t| t.player_nodes).unwrap_or(0),
        flop_gb: flop.map(|t| t.memory_bytes() as f64 / 1e9).unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse() {
        assert_eq!(parse_sizes("33%, 75%", false).unwrap().len(), 2);
        assert_eq!(parse_sizes("50 150 allin", false).unwrap()[1], BetSize::Pot(1.5));
        assert_eq!(parse_sizes("3", true).unwrap()[0], BetSize::Mult(3.0));
        assert!(parse_sizes("3x", false).is_err());
        assert!(parse_sizes("none", false).unwrap().is_empty());
        assert!(parse_sizes("abc", false).is_err());
    }

    #[test]
    fn editing_applies_and_validates() {
        let mut cfg = TrainerConfig::default();
        let mut ed = SizesEditor::new();
        let key = |c| KeyEvent::new(c, KeyModifiers::NONE);
        // River bets: add an overbet.
        ed.cursor = ROWS.iter().position(|&r| r == Row::Bet(Street::River)).unwrap();
        ed.on_key(key(KeyCode::Enter), &mut cfg);
        for c in ", 150%".chars() {
            ed.on_key(key(KeyCode::Char(c)), &mut cfg);
        }
        ed.on_key(key(KeyCode::Enter), &mut cfg);
        assert_eq!(cfg.menu.postflop.river.bet.len(), 3, "{:?}", ed.error);
        // A bad value is rejected and the old one kept.
        ed.cursor = 0;
        ed.on_key(key(KeyCode::Enter), &mut cfg);
        ed.on_key(key(KeyCode::Char('x')), &mut cfg);
        ed.on_key(key(KeyCode::Enter), &mut cfg);
        assert!(ed.error.is_some());
        assert_eq!(cfg.menu.preflop.open_bb, 2.5);
        ed.on_key(key(KeyCode::Esc), &mut cfg);
        // Reset.
        ed.cursor = ROWS.len() - 1;
        ed.on_key(key(KeyCode::Enter), &mut cfg);
        assert_eq!(cfg.menu, BetMenu::default());
        assert!(ed.on_key(key(KeyCode::Esc), &mut cfg));
    }
}
