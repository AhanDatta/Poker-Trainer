//! Application state and input handling.

use crate::study_app::{self, StudyState};
use crate::widgets::{Hit, HitMap};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use holdem::Action;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use trainer::config::{FlopMode, GameType, HeroPosition, SolverBackend, VillainRange};
use trainer::save::{self, SessionMeta, SessionStore};
use trainer::study::StudySettings;
use trainer::{LiveHand, Session, Variant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Setup,
    Table,
    Review,
    History,
    Study,
    /// Bet-size editor.
    Sizes,
    /// Saved sessions: continue, start new, delete.
    Sessions,
    /// The session's frequency score.
    Score,
    Help,
}

/// What Enter on the setup screen starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Play hands against the AI, with solver reviews.
    Play,
    /// Browse the preflop charts and solver strategies.
    Study,
}

/// Rows of the setup screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupItem {
    /// Which saved session this is (Enter: the sessions screen).
    Session,
    Mode,
    Table,
    Opponent,
    /// Custom table: how many players of style `i` (index into the style
    /// list) sit at the table.
    Count(usize),
    SeatChanges,
    Position,
    Stack,
    Rake,
    FlopSolve,
    /// Where heads-up solves run (CPU, GPU or automatic).
    SolveOn,
    /// Exact card removal between opponents in 3-way solves.
    ThreeWay,
    /// Opens the bet-size editor.
    BetSizes,
    VillainRange,
    RevealStyle,
    /// Show a random number at each decision.
    Rng,
    /// Frequency score: weight decisions by the pot.
    ScoreWeight,
}

/// Rake presets: (percent, cap in big blinds).
pub const RAKE_PRESETS: [(f64, f64); 5] = [(0.0, 3.0), (2.5, 1.0), (5.0, 1.0), (5.0, 3.0), (5.0, 5.0)];

/// Seat-change presets: (percent chance per opponent and hand, label).
pub const SEAT_CHANGE_PRESETS: [(f64, &str); 5] = [
    (0.0, "Never: same players all session"),
    (0.5, "Rarely: a new player every ~40 hands (6-max)"),
    (2.0, "Sometimes: a new player every ~10 hands (6-max)"),
    (5.0, "Often: a new player every ~4 hands (6-max)"),
    (100.0, "Every hand: a whole new lineup"),
];

/// What the opponents row can be set to.
#[derive(Clone, Debug, PartialEq)]
pub enum OpponentChoice {
    Game(GameType),
    Style(String),
}

/// Time to press Q a second time to quit.
pub const QUIT_WINDOW: Duration = Duration::from_secs(2);

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
    pub mode: Mode,
    pub study: Option<StudyState>,
    /// Clickable regions of the last frame.
    pub hits: HitMap,
    /// When Q was pressed once: a second press within [`QUIT_WINDOW`] quits.
    quit_armed: Option<Instant>,
    pub sizes: Option<crate::sizes::SizesEditor>,
    /// Where sessions are saved (`None`: saving is off).
    pub sessions_root: Option<PathBuf>,
    /// Saved sessions as last listed, and the selected row (0 = new).
    pub sessions: Vec<SessionMeta>,
    pub sessions_cursor: usize,
    /// A saved session waiting for "y" to be deleted.
    pub confirm_delete: Option<String>,
    /// First line shown on the score screen.
    pub score_scroll: usize,
    /// First hand shown on the history screen (kept by drawing).
    pub history_offset: std::cell::Cell<usize>,
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
            setup_cursor: 0,
            review_target: None,
            review_selected: 0,
            review_variant: Variant::Gto,
            history_selected: 0,
            message: None,
            quit: false,
            config_path,
            mode: Mode::Play,
            study: None,
            hits: HitMap::default(),
            quit_armed: None,
            sizes: None,
            sessions_root: None,
            sessions: Vec::new(),
            sessions_cursor: 0,
            confirm_delete: None,
            score_scroll: 0,
            history_offset: std::cell::Cell::new(0),
            showing_last: false,
            last_step: Instant::now(),
        }
    }

    /// Saves finished hands under `root` and lists the saved sessions.
    pub fn enable_saving(&mut self, root: PathBuf) {
        self.session.start_saving(root.clone());
        self.sessions = save::list_sessions(&root);
        self.sessions_root = Some(root);
    }

    /// Opens the saved-sessions screen.
    pub fn open_sessions(&mut self) {
        if let Some(root) = &self.sessions_root {
            self.sessions = save::list_sessions(root);
        }
        self.sessions_cursor = 0;
        self.confirm_delete = None;
        if self.screen != Screen::Sessions {
            self.goto(Screen::Sessions);
        }
    }

    /// Replaces the session (after saving the current one).
    fn switch_session(&mut self, s: Session) {
        self.session = s;
        self.hand = None;
        self.showing_last = false;
        self.review_target = None;
        self.history_selected = 0;
        self.review_selected = 0;
        self.score_scroll = 0;
        self.screen = Screen::Setup;
    }

    /// Continues saved session `id`.
    fn continue_session(&mut self, id: &str) {
        let Some(root) = self.sessions_root.clone() else { return };
        if self.session.save_id() == Some(id) {
            self.flash("That's the session you're playing.");
            self.screen = Screen::Setup;
            return;
        }
        let loaded = match SessionStore::open(&root, id).map_err(|e| e.to_string()).and_then(|s| s.load()) {
            Ok(l) => l,
            Err(e) => return self.flash(format!("Couldn't read that session: {e}")),
        };
        let skipped = loaded.skipped;
        self.session.flush(Duration::from_secs(3));
        match Session::resume(loaded, root, &self.session.config) {
            Ok(s) => {
                let n = s.history.len();
                self.switch_session(s);
                let mut msg = format!("Continuing the session: {n} hands so far.");
                if skipped > 0 {
                    msg.push_str(&format!(" {skipped} damaged hand(s) couldn't be read."));
                }
                self.flash(msg);
            }
            Err(e) => self.flash(format!("Couldn't continue that session: {e}")),
        }
    }

    /// Starts a new session with the current settings.
    fn new_session(&mut self) {
        if self.session.history.is_empty() && self.session.save_id().is_none() {
            self.screen = Screen::Setup;
            return;
        }
        self.session.flush(Duration::from_secs(3));
        match Session::new(self.session.config.clone()) {
            Ok(mut s) => {
                if let Some(root) = &self.sessions_root {
                    s.start_saving(root.clone());
                }
                self.switch_session(s);
                self.flash("New session started.");
            }
            Err(e) => self.flash(e),
        }
    }

    fn sessions_key(&mut self, key: KeyEvent) {
        if let Some(id) = self.confirm_delete.clone() {
            if matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
                if let Some(root) = &self.sessions_root {
                    match save::delete_session(root, &id) {
                        Ok(()) => self.flash("Session deleted."),
                        Err(e) => self.flash(format!("Couldn't delete it: {e}")),
                    }
                }
                self.open_sessions();
            }
            self.confirm_delete = None;
            return;
        }
        let n = self.sessions.len() + 1;
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.sessions_cursor = self.sessions_cursor.checked_sub(1).unwrap_or(n - 1)
            }
            KeyCode::Down | KeyCode::Char('j') => self.sessions_cursor = (self.sessions_cursor + 1) % n,
            KeyCode::Enter | KeyCode::Char(' ') => match self.sessions_cursor {
                0 => self.new_session(),
                i => {
                    if let Some(id) = self.sessions.get(i - 1).map(|m| m.id.clone()) {
                        self.continue_session(&id);
                    }
                }
            },
            KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Delete | KeyCode::Backspace => {
                match self.sessions_cursor.checked_sub(1).and_then(|i| self.sessions.get(i)) {
                    Some(m) if self.session.save_id() == Some(m.id.as_str()) => {
                        self.flash("That's the session you're playing: start a new one first to delete it.")
                    }
                    Some(m) => {
                        self.confirm_delete = Some(m.id.clone());
                        // The prompt must not hide behind an old message.
                        self.message = None;
                    }
                    None => {}
                }
            }
            KeyCode::Char('?') => self.goto(Screen::Help),
            KeyCode::Char('q') => self.request_quit(),
            KeyCode::Esc | KeyCode::Char('b') => self.screen = Screen::Setup,
            _ => {}
        }
    }

    fn score_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.score_scroll = self.score_scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.score_scroll += 1,
            KeyCode::PageUp => self.score_scroll = self.score_scroll.saturating_sub(10),
            KeyCode::PageDown => self.score_scroll += 10,
            KeyCode::Char('?') => self.goto(Screen::Help),
            KeyCode::Char('q') => self.request_quit(),
            _ => {
                self.screen =
                    if self.prev_screen == Screen::Score { Screen::Table } else { self.prev_screen };
            }
        }
    }

    fn open_score(&mut self) {
        self.score_scroll = 0;
        self.goto(Screen::Score);
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
    /// Game types, then "everyone plays this style" for each style.
    pub fn opponent_choices(&self) -> Vec<OpponentChoice> {
        let mut v: Vec<OpponentChoice> = GameType::ALL.into_iter().map(OpponentChoice::Game).collect();
        v.extend(self.session.config.profiles().into_iter().map(|p| OpponentChoice::Style(p.name)));
        v
    }

    pub fn current_opponent_choice(&self) -> OpponentChoice {
        let cfg = &self.session.config;
        if cfg.opponent.eq_ignore_ascii_case("random") {
            OpponentChoice::Game(cfg.game)
        } else {
            OpponentChoice::Style(cfg.opponent.clone())
        }
    }

    /// Rows of the setup screen (custom tables get one row per style).
    pub fn setup_items(&self) -> Vec<SetupItem> {
        if self.mode == Mode::Study {
            return vec![
                SetupItem::Mode,
                SetupItem::Table,
                SetupItem::Stack,
                SetupItem::Rake,
                SetupItem::FlopSolve,
                SetupItem::SolveOn,
                SetupItem::ThreeWay,
                SetupItem::BetSizes,
            ];
        }
        let mut v = vec![SetupItem::Mode, SetupItem::Session, SetupItem::Table, SetupItem::Opponent];
        if self.current_opponent_choice() == OpponentChoice::Game(GameType::Custom) {
            v.extend((0..self.session.config.profiles().len()).map(SetupItem::Count));
        }
        v.extend([
            SetupItem::SeatChanges,
            SetupItem::Position,
            SetupItem::Stack,
            SetupItem::Rake,
            SetupItem::FlopSolve,
            SetupItem::SolveOn,
            SetupItem::ThreeWay,
            SetupItem::BetSizes,
            SetupItem::VillainRange,
            SetupItem::RevealStyle,
            SetupItem::Rng,
            SetupItem::ScoreWeight,
        ]);
        v
    }

    /// Advances AI actions / dealing at the configured pace.
    pub fn tick(&mut self) {
        if let Some((_, t)) = &self.message {
            if t.elapsed() > Duration::from_secs(4) {
                self.message = None;
            }
        }
        // Settles hands whose reviews finished (stats, score, autosave).
        self.session.refresh();
        if let Some(e) = self.session.take_save_error() {
            self.flash(e);
        }
        if self.screen == Screen::Study {
            if let Some(st) = &mut self.study {
                st.tick();
            }
            return;
        }
        if self.screen != Screen::Table {
            return;
        }
        let delay = Duration::from_millis(self.session.config.ai_delay_ms);
        if let Some(hand) = &mut self.hand {
            // After the hero folds, play the rest of the hand out instantly.
            let delay = if hand.hero_folded() { Duration::ZERO } else { delay };
            if !hand.is_finished() && !hand.hero_to_act() && self.last_step.elapsed() >= delay {
                hand.step();
                self.last_step = Instant::now();
            }
        }
        self.archive_if_finished();
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

    /// Quits on the second Q within [`QUIT_WINDOW`], so a stray key press
    /// doesn't end a long session (Ctrl+C still quits at once).
    pub fn request_quit(&mut self) {
        if self.quit_armed.is_some_and(|t| t.elapsed() <= QUIT_WINDOW) {
            self.quit = true;
        } else {
            self.quit_armed = Some(Instant::now());
            if self.session.is_saving() {
                self.flash("Press Q again to quit (the session is saved: continue it next time).");
            } else {
                self.flash("Press Q again to quit (this session isn't saved: saving is off).");
            }
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        // Any other key cancels a pending quit.
        let quit_key = matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q'))
            || (self.screen == Screen::Setup && key.code == KeyCode::Esc);
        if !quit_key && self.quit_armed.take().is_some() {
            self.message = None;
        }
        match self.screen {
            Screen::Setup => self.setup_key(key),
            Screen::Table => self.table_key(key),
            Screen::Review => self.review_key(key),
            Screen::History => self.history_key(key),
            Screen::Study => self.study_key(key),
            Screen::Sizes => {
                let ed = self.sizes.get_or_insert_with(crate::sizes::SizesEditor::new);
                if ed.on_key(key, &mut self.session.config) {
                    self.screen = Screen::Setup;
                }
            }
            Screen::Sessions => self.sessions_key(key),
            Screen::Score => self.score_key(key),
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
                let n = self.setup_items().len();
                self.setup_cursor = self.setup_cursor.checked_sub(1).unwrap_or(n - 1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let n = self.setup_items().len();
                self.setup_cursor = (self.setup_cursor + 1) % n;
            }
            KeyCode::Left | KeyCode::Char('h') => self.change_setting(-1),
            KeyCode::Right | KeyCode::Char('l') => self.change_setting(1),
            KeyCode::Enter | KeyCode::Char(' ')
                if self.setup_items().get(self.setup_cursor) == Some(&SetupItem::BetSizes) =>
            {
                self.sizes.get_or_insert_with(crate::sizes::SizesEditor::new);
                self.goto(Screen::Sizes);
            }
            KeyCode::Enter | KeyCode::Char(' ')
                if self.setup_items().get(self.setup_cursor) == Some(&SetupItem::Session) =>
            {
                self.open_session_screen_or_explain();
            }
            KeyCode::Char('g') => self.open_score(),
            KeyCode::Char('o') => self.open_session_screen_or_explain(),
            KeyCode::Enter | KeyCode::Char(' ') => match self.mode {
                Mode::Play => {
                    self.goto(Screen::Table);
                    if self.hand_finished() {
                        self.start_hand();
                    }
                }
                Mode::Study => self.open_study(),
            },
            KeyCode::Char('w') => {
                let text = self.session.config.to_toml();
                match std::fs::write(&self.config_path, text) {
                    Ok(()) => self.flash(format!("Saved settings to {}", self.config_path)),
                    Err(e) => self.flash(format!("Couldn't save {}: {e}", self.config_path)),
                }
            }
            KeyCode::Char('?') => self.goto(Screen::Help),
            KeyCode::Char('q') | KeyCode::Esc => self.request_quit(),
            _ => {}
        }
    }

    fn change_setting(&mut self, dir: i32) {
        let cycle = |i: usize, n: usize| -> usize { ((i as i32 + dir).rem_euclid(n as i32)) as usize };
        let items = self.setup_items();
        let item = items[self.setup_cursor.min(items.len() - 1)];
        let cfg = &mut self.session.config;
        match item {
            SetupItem::Session => self.open_session_screen_or_explain(),
            SetupItem::Rng => cfg.show_rng = !cfg.show_rng,
            SetupItem::ScoreWeight => cfg.scoring.pot_weighted = !cfg.scoring.pot_weighted,
            SetupItem::Mode => {
                self.mode = match self.mode {
                    Mode::Play => Mode::Study,
                    Mode::Study => Mode::Play,
                };
                self.setup_cursor = 0;
            }
            SetupItem::Table => {
                let sizes = [2usize, 6, 9];
                let cur = sizes.iter().position(|&n| n == self.session.config.players).unwrap_or(0);
                let next = sizes[cycle(cur, sizes.len())];
                if let Err(e) = self.session.set_players(next) {
                    self.flash(e);
                } else if next > 6 && self.session.config.charts_for(next).is_none() {
                    self.flash("9-max uses position-aware preflop rules (there are no 9-max solver charts).");
                }
            }
            SetupItem::Opponent => {
                let choices = self.opponent_choices();
                let cur = choices.iter().position(|c| *c == self.current_opponent_choice()).unwrap_or(0);
                match choices[cycle(cur, choices.len())].clone() {
                    OpponentChoice::Game(g) => self.session.set_game(g),
                    OpponentChoice::Style(name) => {
                        if let Err(e) = self.session.set_opponent(&name) {
                            self.flash(e);
                        }
                    }
                }
            }
            SetupItem::Count(i) => {
                let Some(style) = self.session.config.profiles().get(i).map(|p| p.name.clone()) else {
                    return;
                };
                let cur = self.session.config.custom_table.get(&style).copied().unwrap_or(0);
                let max = self.session.config.players.saturating_sub(1);
                let next = (cur as i32 + dir).clamp(0, max as i32) as usize;
                self.session.set_custom_count(&style, next);
            }
            SetupItem::SeatChanges => {
                let cur = SEAT_CHANGE_PRESETS
                    .iter()
                    .position(|&(p, _)| (p - cfg.seat_change_pct).abs() < 1e-9)
                    .unwrap_or(2);
                cfg.seat_change_pct = SEAT_CHANGE_PRESETS[cycle(cur, SEAT_CHANGE_PRESETS.len())].0;
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
            SetupItem::SolveOn => {
                let all = [SolverBackend::Cpu, SolverBackend::Auto, SolverBackend::Gpu];
                let cur = all.iter().position(|&b| b == cfg.solver.backend).unwrap_or(0);
                cfg.solver.backend = all[cycle(cur, all.len())];
                if cfg.solver.backend != SolverBackend::Cpu && !solver::GPU_SUPPORT {
                    self.flash("This build has no GPU support (build with --features gpu); solves stay on the CPU.");
                }
            }
            SetupItem::ThreeWay => {
                cfg.solver.multiway_exact_removal = !cfg.solver.multiway_exact_removal;
            }
            SetupItem::BetSizes => {
                self.sizes.get_or_insert_with(crate::sizes::SizesEditor::new);
                self.goto(Screen::Sizes);
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
            KeyCode::Char('g') => self.open_score(),
            KeyCode::Esc | KeyCode::Char('s') => {
                if !finished {
                    self.flash("Finish this hand before changing settings.");
                } else {
                    self.goto(Screen::Setup);
                }
            }
            KeyCode::Char('q') => self.request_quit(),
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
            KeyCode::Char('g') => self.open_score(),
            KeyCode::Char('q') => self.request_quit(),
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
            KeyCode::Char('g') => self.open_score(),
            KeyCode::Char('q') => self.request_quit(),
            KeyCode::Esc | KeyCode::Char('b') => self.screen = Screen::Table,
            _ => {}
        }
    }

    fn open_session_screen_or_explain(&mut self) {
        if self.sessions_root.is_some() {
            self.open_sessions();
        } else {
            self.flash(
                "Saving is off (save_sessions = false in the config), so there are no saved sessions.",
            );
        }
    }

    /// Opens the study screen, resuming the last study unless the settings
    /// it depends on changed.
    pub fn open_study(&mut self) {
        let settings =
            Arc::new(StudySettings::from_config(&self.session.config, Arc::clone(self.session.charts())));
        let fresh = study_app::fingerprint(&settings);
        if self.study.as_ref().is_none_or(|s| s.fingerprint != fresh) {
            self.study = Some(StudyState::new(settings));
        }
        if self.session.stats().reviews_pending > 0 {
            self.session.cancel_pending_reviews();
            self.flash("Unfinished hand reviews were cancelled to free the solver for study mode.");
        }
        self.mode = Mode::Study;
        self.goto(Screen::Study);
    }

    fn study_key(&mut self, key: KeyEvent) {
        let Some(st) = &mut self.study else {
            self.screen = Screen::Setup;
            return;
        };
        match st.on_key(key) {
            study_app::Outcome::None => {}
            study_app::Outcome::Leave => self.screen = Screen::Setup,
            study_app::Outcome::Help => self.goto(Screen::Help),
            study_app::Outcome::Quit => self.request_quit(),
        }
    }

    pub fn on_mouse(&mut self, ev: MouseEvent) {
        if ev.kind != MouseEventKind::Down(MouseButton::Left) {
            return;
        }
        let Some(hit) = self.hits.at(ev.column, ev.row) else { return };
        match self.screen {
            Screen::Study => {
                if let Some(st) = &mut self.study {
                    st.on_click(hit);
                }
            }
            Screen::Sizes => {
                if let Some(ed) = &mut self.sizes {
                    ed.on_click(hit);
                }
            }
            Screen::Review => {
                if let Hit::Decision(i) = hit {
                    self.review_selected = i;
                }
            }
            Screen::Table => {
                if let Hit::Action(i) = hit {
                    let acts = self.action_keys();
                    if let Some(&(_, a)) = acts.get(i) {
                        if let Some(hand) = &mut self.hand {
                            if let Err(e) = hand.hero_act(a) {
                                self.flash(e.to_string());
                            }
                            self.last_step = Instant::now();
                        }
                        self.archive_if_finished();
                    }
                }
            }
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
