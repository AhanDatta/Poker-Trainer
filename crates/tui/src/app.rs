//! Application state and input handling.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use holdem::Action;
use std::time::{Duration, Instant};
use trainer::config::{FlopMode, HeroPosition, VillainRange};
use trainer::{LiveHand, Session, Variant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Setup,
    Table,
    Review,
    History,
    Help,
}

/// Rows of the setup screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupItem {
    Table,
    Opponent,
    Position,
    Stack,
    Rake,
    FlopSolve,
    VillainRange,
    RevealStyle,
}

/// Rake presets: (percent, cap in big blinds).
pub const RAKE_PRESETS: [(f64, f64); 5] = [(0.0, 3.0), (2.5, 1.0), (5.0, 1.0), (5.0, 3.0), (5.0, 5.0)];

pub const SETUP_ITEMS: [SetupItem; 8] = [
    SetupItem::Table,
    SetupItem::Opponent,
    SetupItem::Position,
    SetupItem::Stack,
    SetupItem::Rake,
    SetupItem::FlopSolve,
    SetupItem::VillainRange,
    SetupItem::RevealStyle,
];

pub struct App {
    pub session: Session,
    pub screen: Screen,
    pub prev_screen: Screen,
    pub hand: Option<LiveHand>,
    pub setup_cursor: usize,
    /// Which hand the review screen shows: `None` = current hand.
    pub review_target: Option<usize>,
    pub review_selected: usize,
    pub review_variant: Variant,
    pub history_selected: usize,
    pub message: Option<(String, Instant)>,
    pub quit: bool,
    pub config_path: String,
    /// The last finished hand (now in the history) is still on the table.
    showing_last: bool,
    last_step: Instant,
}

impl App {
    pub fn new(session: Session, config_path: String) -> App {
        App {
            session,
            screen: Screen::Setup,
            prev_screen: Screen::Setup,
            hand: None,
            setup_cursor: 1,
            review_target: None,
            review_selected: 0,
            review_variant: Variant::Gto,
            history_selected: 0,
            message: None,
            quit: false,
            config_path,
            showing_last: false,
            last_step: Instant::now(),
        }
    }

    /// The hand on the table: the live hand, or the just-finished one.
    pub fn current(&self) -> Option<&LiveHand> {
        match &self.hand {
            Some(h) => Some(h),
            None if self.showing_last => self.session.history.last(),
            None => None,
        }
    }

    /// Moves a finished live hand into the session history.
    fn archive_if_finished(&mut self) {
        if self.hand.as_ref().is_some_and(|h| h.is_finished()) {
            let h = self.hand.take().unwrap();
            self.session.finish_hand(h);
            self.showing_last = true;
        }
    }

    fn hand_finished(&self) -> bool {
        self.hand.as_ref().is_none_or(|h| h.is_finished())
    }

    pub fn flash(&mut self, msg: impl Into<String>) {
        self.message = Some((msg.into(), Instant::now()));
    }

    /// Opponent choices for the setup screen.
    pub fn opponent_choices(&self) -> Vec<String> {
        let mut v = vec!["random".to_string()];
        v.extend(self.session.config.profiles().into_iter().map(|p| p.name));
        v
    }

    /// Advances AI actions / dealing at the configured pace.
    pub fn tick(&mut self) {
        if self.screen != Screen::Table {
            return;
        }
        let delay = Duration::from_millis(self.session.config.ai_delay_ms);
        if let Some(hand) = &mut self.hand {
            if !hand.is_finished() && !hand.hero_to_act() && self.last_step.elapsed() >= delay {
                hand.step();
                self.last_step = Instant::now();
            }
        }
        self.archive_if_finished();
        if let Some((_, t)) = &self.message {
            if t.elapsed() > Duration::from_secs(4) {
                self.message = None;
            }
        }
    }

    fn start_hand(&mut self) {
        if let Some(h) = self.hand.take() {
            if h.is_finished() {
                self.session.finish_hand(h);
            } else if let Some(r) = h.review() {
                r.cancel();
            }
        }
        self.showing_last = false;
        self.hand = Some(self.session.new_hand());
        self.review_target = None;
        self.last_step = Instant::now();
    }

    /// The hand shown on the review screen.
    pub fn review_hand(&self) -> Option<&LiveHand> {
        match self.review_target {
            None => self.current(),
            Some(i) => self.session.history.get(i),
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        match self.screen {
            Screen::Setup => self.setup_key(key),
            Screen::Table => self.table_key(key),
            Screen::Review => self.review_key(key),
            Screen::History => self.history_key(key),
            Screen::Help => {
                self.screen = self.prev_screen;
            }
        }
    }

    fn goto(&mut self, s: Screen) {
        self.prev_screen = self.screen;
        self.screen = s;
    }

    fn setup_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.setup_cursor = self.setup_cursor.checked_sub(1).unwrap_or(SETUP_ITEMS.len() - 1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.setup_cursor = (self.setup_cursor + 1) % SETUP_ITEMS.len();
            }
            KeyCode::Left | KeyCode::Char('h') => self.change_setting(-1),
            KeyCode::Right | KeyCode::Char('l') => self.change_setting(1),
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.goto(Screen::Table);
                if self.hand_finished() {
                    self.start_hand();
                }
            }
            KeyCode::Char('w') => {
                let text = self.session.config.to_toml();
                match std::fs::write(&self.config_path, text) {
                    Ok(()) => self.flash(format!("Saved settings to {}", self.config_path)),
                    Err(e) => self.flash(format!("Couldn't save {}: {e}", self.config_path)),
                }
            }
            KeyCode::Char('?') => self.goto(Screen::Help),
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            _ => {}
        }
    }

    fn change_setting(&mut self, dir: i32) {
        let cycle = |i: usize, n: usize| -> usize { ((i as i32 + dir).rem_euclid(n as i32)) as usize };
        let cfg = &mut self.session.config;
        match SETUP_ITEMS[self.setup_cursor] {
            SetupItem::Table => self.flash("6-max and 9-max are planned; heads-up is available now."),
            SetupItem::Opponent => {
                let choices = self.opponent_choices();
                let cur = choices
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case(&self.session.config.opponent))
                    .unwrap_or(0);
                let next = choices[cycle(cur, choices.len())].clone();
                if let Err(e) = self.session.set_opponent(&next) {
                    self.flash(e);
                }
            }
            SetupItem::Position => {
                let all = [HeroPosition::Alternate, HeroPosition::Random, HeroPosition::Sb, HeroPosition::Bb];
                let cur = all.iter().position(|&p| p == cfg.hero_position).unwrap_or(0);
                cfg.hero_position = all[cycle(cur, all.len())];
            }
            SetupItem::Stack => {
                cfg.stack_bb = (cfg.stack_bb + 10.0 * dir as f64).clamp(20.0, 300.0);
                if (cfg.stack_bb - 100.0).abs() > 1e-9 {
                    self.flash("Note: the built-in preflop charts assume 100bb stacks.");
                }
            }
            SetupItem::Rake => {
                let cur = RAKE_PRESETS
                    .iter()
                    .position(|&(p, c)| (p - cfg.rake_pct).abs() < 1e-9 && (c - cfg.rake_cap_bb).abs() < 1e-9)
                    .unwrap_or(0);
                let (p, c) = RAKE_PRESETS[cycle(cur, RAKE_PRESETS.len())];
                cfg.rake_pct = p;
                cfg.rake_cap_bb = c;
            }
            SetupItem::FlopSolve => {
                cfg.solver.flop_mode = match cfg.solver.flop_mode {
                    FlopMode::Fast => FlopMode::Exact,
                    FlopMode::Exact => FlopMode::Fast,
                };
                if cfg.solver.flop_mode == FlopMode::Exact {
                    self.flash("Exact flop solves need several GB of RAM and minutes per hand.");
                }
            }
            SetupItem::VillainRange => {
                let all = [VillainRange::Gto, VillainRange::Actual, VillainRange::Both];
                let cur = all.iter().position(|&p| p == cfg.solver.villain_range).unwrap_or(0);
                cfg.solver.villain_range = all[cycle(cur, all.len())];
            }
            SetupItem::RevealStyle => cfg.show_opponent_style = !cfg.show_opponent_style,
        }
    }

    /// Hero actions paired with their key.
    pub fn action_keys(&self) -> Vec<(char, Action)> {
        let Some(hand) = &self.hand else { return Vec::new() };
        let mut out = Vec::new();
        let mut n = 0;
        for a in hand.hero_actions() {
            let key = match a {
                Action::Fold => 'f',
                Action::Check | Action::Call => 'c',
                Action::Raise(_) => {
                    n += 1;
                    char::from_digit(n, 10).unwrap_or('9')
                }
            };
            out.push((key, a));
        }
        out
    }

    fn table_key(&mut self, key: KeyEvent) {
        let finished = self.hand_finished();
        if let KeyCode::Char(ch) = key.code {
            let ch = ch.to_ascii_lowercase();
            let acts = self.action_keys();
            let alias = match ch {
                'x' | 'k' => 'c',
                'a' => {
                    acts.iter().rev().find(|(_, a)| matches!(a, Action::Raise(_))).map_or('a', |(k, _)| *k)
                }
                other => other,
            };
            if let Some(&(_, a)) = acts.iter().find(|(k, _)| *k == alias) {
                if let Some(hand) = &mut self.hand {
                    if let Err(e) = hand.hero_act(a) {
                        self.flash(e.to_string());
                    }
                    self.last_step = Instant::now();
                }
                self.archive_if_finished();
                return;
            }
        }
        match key.code {
            KeyCode::Char('n') | KeyCode::Enter if finished => self.start_hand(),
            KeyCode::Char('r') if finished => {
                if self.current().is_some_and(|h| h.review().is_some()) {
                    self.review_target = None;
                    self.review_selected = 0;
                    self.goto(Screen::Review);
                } else {
                    self.flash("Nothing to review: the hand ended before the flop.");
                }
            }
            KeyCode::Char('r') => self.flash("Finish the hand first, then press R for the solver review."),
            KeyCode::Char('h') => {
                if self.session.history.is_empty() {
                    self.flash("No finished hands yet.");
                } else {
                    self.history_selected = self.session.history.len() - 1;
                    self.goto(Screen::History);
                }
            }
            KeyCode::Char('?') => self.goto(Screen::Help),
            KeyCode::Esc | KeyCode::Char('s') => {
                if !finished {
                    self.flash("Finish this hand before changing settings.");
                } else {
                    self.goto(Screen::Setup);
                }
            }
            KeyCode::Char('q') => self.quit = true,
            _ => {}
        }
    }

    fn review_key(&mut self, key: KeyEvent) {
        let count = self.visible_decisions().len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.review_selected = self.review_selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if count > 0 {
                    self.review_selected = (self.review_selected + 1).min(count - 1);
                }
            }
            KeyCode::Tab | KeyCode::Char('v') => {
                self.review_variant = match self.review_variant {
                    Variant::Gto => Variant::Actual,
                    Variant::Actual => Variant::Gto,
                };
                self.review_selected = 0;
            }
            KeyCode::Char('n') => {
                self.goto(Screen::Table);
                if self.hand_finished() {
                    self.start_hand();
                }
            }
            KeyCode::Char('?') => self.goto(Screen::Help),
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Esc | KeyCode::Char('b') => {
                self.screen = if self.review_target.is_some() { Screen::History } else { Screen::Table };
            }
            _ => {}
        }
    }

    fn history_key(&mut self, key: KeyEvent) {
        let n = self.session.history.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.history_selected = self.history_selected.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if n > 0 {
                    self.history_selected = (self.history_selected + 1).min(n - 1);
                }
            }
            KeyCode::Enter | KeyCode::Char('r') => {
                if n > 0 && self.session.history[self.history_selected].review().is_some() {
                    self.review_target = Some(self.history_selected);
                    self.review_selected = 0;
                    self.goto(Screen::Review);
                } else {
                    self.flash("That hand has no solver review (it ended preflop).");
                }
            }
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Esc | KeyCode::Char('b') => self.screen = Screen::Table,
            _ => {}
        }
    }

    /// Decisions of the selected variant for the review screen.
    pub fn visible_decisions(&self) -> Vec<trainer::DecisionReview> {
        let Some(hand) = self.review_hand() else { return Vec::new() };
        let Some(r) = hand.review() else { return Vec::new() };
        let st = r.status();
        let has_variant = st.decisions.iter().any(|d| d.variant == self.review_variant);
        st.decisions.into_iter().filter(|d| !has_variant || d.variant == self.review_variant).collect()
    }
}
