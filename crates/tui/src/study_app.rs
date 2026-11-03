//! Solver study mode: state and input handling.

use crate::widgets::Hit;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use holdem::card::parse_cards;
use holdem::{Action, Card, CardSet, Range, Street};
use rand::seq::SliceRandom;
use std::sync::Arc;
use trainer::fmt::BB;
use trainer::study::{PreflopNav, PreflopOutcome, Spot, Study, StudyPos, StudySettings};

/// What the grid shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridView {
    /// The acting player's strategy.
    Strategy,
    /// A player's range (index into the spot's players, or the preflop seat).
    Range(usize),
}

/// A one-line text prompt (card entry).
#[derive(Clone, Debug, Default)]
pub struct TextInput {
    pub prompt: String,
    pub text: String,
}

pub enum Phase {
    Preflop(Box<PreflopNav>),
    Postflop(Box<Postflop>),
}

pub struct Postflop {
    pub study: Study,
    /// Position as of the last tick.
    pub pos: StudyPos,
    /// The preflop line that led here (`None` for typed spots).
    pub preflop: Option<PreflopNav>,
}

/// Rows of the spot editor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorRow {
    Players,
    Range(usize),
    Pot,
    Stack,
    Board,
    Solve,
}

/// Typed-spot form.
#[derive(Clone, Debug)]
pub struct SpotEditor {
    pub names: Vec<String>,
    pub ranges: Vec<String>,
    pub pot: String,
    pub stack: String,
    pub board: String,
    pub cursor: usize,
    pub editing: bool,
    pub error: Option<String>,
}

/// Example ranges for a fresh editor: a button open called by the big blind.
const EXAMPLE_IP: &str = "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,85s+,75s+,64s+,54s,A2o+,K8o+,Q9o+,J9o+,T9o";
const EXAMPLE_OOP: &str =
    "22-99,A2s-AJs,K2s-KJs,Q6s+,J7s+,T7s+,96s+,85s+,74s+,63s+,53s+,43s,A8o-AJo,K9o+,QTo+,JTo,T9o";

impl SpotEditor {
    pub fn example() -> SpotEditor {
        SpotEditor {
            names: Spot::default_names(2),
            ranges: vec![EXAMPLE_OOP.into(), EXAMPLE_IP.into()],
            pot: "5.5".into(),
            stack: "97.5".into(),
            board: String::new(),
            cursor: 1,
            editing: false,
            error: None,
        }
    }

    pub fn from_spot(spot: &Spot) -> SpotEditor {
        let num = |c: holdem::Chips| {
            let s = format!("{:.2}", c as f64 / BB as f64);
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        };
        SpotEditor {
            names: spot.names.clone(),
            ranges: spot.ranges.iter().map(|r| range_text(r)).collect(),
            pot: num(spot.pot),
            stack: num(spot.stacks.iter().copied().min().unwrap_or(0)),
            board: spot.board.iter().map(|c| c.to_string()).collect(),
            cursor: 1,
            editing: false,
            error: None,
        }
    }

    pub fn rows(&self) -> Vec<EditorRow> {
        let mut v = vec![EditorRow::Players];
        v.extend((0..self.names.len()).map(EditorRow::Range));
        v.extend([EditorRow::Pot, EditorRow::Stack, EditorRow::Board, EditorRow::Solve]);
        v
    }

    pub fn row(&self) -> EditorRow {
        let rows = self.rows();
        rows[self.cursor.min(rows.len() - 1)]
    }

    /// The range the cursor is on (or the first one), parsed.
    pub fn preview(&self) -> (usize, Result<Range, String>) {
        let i = match self.row() {
            EditorRow::Range(i) => i,
            _ => 0,
        };
        (i, Range::parse(&self.ranges[i]).map_err(|e| e.to_string()))
    }

    fn text_mut(&mut self) -> Option<&mut String> {
        match self.row() {
            EditorRow::Range(i) => Some(&mut self.ranges[i]),
            EditorRow::Pot => Some(&mut self.pot),
            EditorRow::Stack => Some(&mut self.stack),
            EditorRow::Board => Some(&mut self.board),
            _ => None,
        }
    }

    fn toggle_players(&mut self) {
        if self.names.len() == 2 {
            self.names = Spot::default_names(3);
            let first = self.ranges[0].clone();
            self.ranges.insert(1, first);
        } else {
            self.names = Spot::default_names(2);
            self.ranges.remove(1);
        }
    }

    /// Builds the spot (a random flop when the board is empty).
    fn spot(&self) -> Result<Spot, String> {
        let pot: f64 =
            self.pot.trim().parse().map_err(|_| "pot: enter a number of big blinds".to_string())?;
        let stack: f64 =
            self.stack.trim().parse().map_err(|_| "stacks: enter a number of big blinds".to_string())?;
        let board = if self.board.trim().is_empty() {
            random_cards(CardSet::EMPTY, 3).iter().map(|c| c.to_string()).collect()
        } else {
            self.board.clone()
        };
        Spot::from_text(self.names.clone(), &self.ranges, pot, stack, &board)
    }
}

/// A range in notation, per class (`AKo:0.5`), for editing.
pub fn range_text(weights: &[f32]) -> String {
    Range::from_weights(weights.to_vec()).to_class_string().replace(", ", ",")
}

fn random_cards(dead: CardSet, n: usize) -> Vec<Card> {
    let mut deck: Vec<Card> = Card::all().filter(|c| !dead.contains(*c)).collect();
    deck.shuffle(&mut rand::rng());
    deck.truncate(n);
    deck
}

/// What the app should do after a key in study mode.
pub enum Outcome {
    None,
    Leave,
    Help,
    Quit,
}

pub struct StudyState {
    pub settings: Arc<StudySettings>,
    /// Settings the state was built with (a change starts over).
    pub fingerprint: String,
    pub phase: Phase,
    pub view: GridView,
    pub cursor: usize,
    pub input: Option<TextInput>,
    pub editor: Option<SpotEditor>,
    /// Inline message (bad card, unsupported line…).
    pub message: Option<String>,
    /// Ranges (and board) from the last solved view, shown while the next
    /// solve runs.
    pub last_ranges: Option<(Vec<Vec<f32>>, Vec<Card>)>,
}

pub fn fingerprint(s: &StudySettings) -> String {
    format!(
        "{}|{}|{}|{}|{:?}|{:?}|{:?}",
        s.players,
        s.stack_bb,
        s.rake_rate,
        s.rake_cap,
        s.solver,
        s.menu,
        Arc::as_ptr(&s.charts)
    )
}

impl StudyState {
    pub fn new(settings: Arc<StudySettings>) -> StudyState {
        StudyState {
            fingerprint: fingerprint(&settings),
            phase: Phase::Preflop(Box::new(PreflopNav::new(Arc::clone(&settings)))),
            settings,
            view: GridView::Strategy,
            cursor: 0,
            input: None,
            editor: None,
            message: None,
            last_ranges: None,
        }
    }

    /// Advances solves and keeps the card prompt in sync with the position.
    pub fn tick(&mut self) {
        if let Phase::Postflop(p) = &mut self.phase {
            p.pos = p.study.poll();
            let view = match &p.pos {
                StudyPos::Decision(v) => Some(v),
                StudyPos::NeedCard { view: Some(v), .. } | StudyPos::Over { view: Some(v), .. } => Some(v),
                _ => None,
            };
            if let Some(v) = view {
                self.last_ranges = Some((v.ranges.clone(), p.study.board()));
            }
        }
        let want = self.wanted_prompt();
        match (&want, &self.input) {
            (Some(p), Some(i)) if *p == i.prompt => {}
            (Some(p), _) => self.input = Some(TextInput { prompt: p.clone(), text: String::new() }),
            (None, Some(_)) => self.input = None,
            (None, None) => {}
        }
    }

    /// The card prompt the position needs, if any.
    fn wanted_prompt(&self) -> Option<String> {
        if self.editor.is_some() {
            return None;
        }
        match &self.phase {
            Phase::Preflop(nav) => match nav.outcome() {
                PreflopOutcome::Flop(seats) if (2..=3).contains(&seats.len()) => {
                    Some("Flop cards (e.g. Ks7d2c), Enter = random flop".into())
                }
                _ => None,
            },
            Phase::Postflop(p) => match &p.pos {
                StudyPos::NeedCard { next, .. } => Some(format!("{next} card (e.g. 9h), Enter = random")),
                _ => None,
            },
        }
    }

    /// Players whose ranges the grid can show.
    pub fn range_players(&self) -> Vec<(usize, String)> {
        match &self.phase {
            Phase::Preflop(nav) => {
                let st = nav.state();
                st.players_in_hand().into_iter().map(|s| (s, st.position_name(s).to_string())).collect()
            }
            Phase::Postflop(p) => {
                let spot = p.study.spot();
                let folded = match &p.pos {
                    StudyPos::Decision(v) => v.folded.clone(),
                    StudyPos::NeedCard { view: Some(v), .. } | StudyPos::Over { view: Some(v), .. } => {
                        v.folded.clone()
                    }
                    _ => vec![false; spot.names.len()],
                };
                spot.names
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !folded[*i])
                    .map(|(i, n)| (i, n.clone()))
                    .collect()
            }
        }
    }

    fn cycle_view(&mut self, dir: i32) {
        let players = self.range_players();
        let mut views = vec![GridView::Strategy];
        views.extend(players.iter().map(|&(i, _)| GridView::Range(i)));
        let cur = views.iter().position(|&v| v == self.view).unwrap_or(0) as i32;
        self.view = views[(cur + dir).rem_euclid(views.len() as i32) as usize];
    }

    /// Actions of the current decision.
    pub fn actions(&self) -> Vec<Action> {
        match &self.phase {
            Phase::Preflop(nav) => nav.node().map(|n| n.actions).unwrap_or_default(),
            Phase::Postflop(p) => match &p.pos {
                StudyPos::Decision(v) => v.actions.clone(),
                _ => Vec::new(),
            },
        }
    }

    /// Key letter of each action: F fold, C check/call, 1-9 bets and raises.
    pub fn action_keys(&self) -> Vec<char> {
        let mut n = 0;
        self.actions()
            .iter()
            .map(|a| match a {
                Action::Fold => 'f',
                Action::Check | Action::Call => 'c',
                Action::Raise(_) => {
                    n += 1;
                    char::from_digit(n, 10).unwrap_or('9')
                }
            })
            .collect()
    }

    fn take(&mut self, i: usize) {
        self.message = None;
        match &mut self.phase {
            Phase::Preflop(nav) => {
                if let Err(e) = nav.act(i) {
                    self.message = Some(e);
                }
                self.check_preflop();
            }
            Phase::Postflop(p) => {
                let StudyPos::Decision(v) = &p.pos else { return };
                let Some(&a) = v.actions.get(i) else { return };
                if let Err(e) = p.study.act(a) {
                    self.message = Some(e);
                }
                p.pos = p.study.poll();
            }
        }
        if let GridView::Range(_) = self.view {
            // Keep showing ranges, unless that player folded.
            if !self.range_players().iter().any(|&(i, _)| GridView::Range(i) == self.view) {
                self.view = GridView::Strategy;
            }
        }
    }

    fn check_preflop(&mut self) {
        let Phase::Preflop(nav) = &self.phase else { return };
        self.message = match nav.outcome() {
            PreflopOutcome::Won(s) => {
                Some(format!("{} wins the blinds. Backspace goes back.", nav.state().position_name(s)))
            }
            PreflopOutcome::AllIn => {
                Some("All-in preflop: nothing left to decide. Backspace goes back.".into())
            }
            PreflopOutcome::Flop(seats) if seats.len() > 3 => Some(format!(
                "{} players see the flop: study mode solves 2 or 3. Backspace goes back.",
                seats.len()
            )),
            _ => None,
        };
    }

    fn back(&mut self) {
        self.message = None;
        let to_preflop = match &mut self.phase {
            Phase::Preflop(nav) => {
                nav.back();
                self.check_preflop();
                false
            }
            Phase::Postflop(p) => {
                if p.study.back() {
                    p.pos = p.study.poll();
                    false
                } else {
                    p.preflop.is_some()
                }
            }
        };
        if to_preflop {
            let old = std::mem::replace(
                &mut self.phase,
                Phase::Preflop(Box::new(PreflopNav::new(Arc::clone(&self.settings)))),
            );
            if let Phase::Postflop(p) = old {
                self.phase = Phase::Preflop(Box::new(p.preflop.expect("checked")));
            }
            self.view = GridView::Strategy;
        }
    }

    /// Back to the start of the current street's browser (preflop: the
    /// first decision; postflop: the spot's root).
    fn home(&mut self) {
        self.message = None;
        match &mut self.phase {
            Phase::Preflop(nav) => nav.reset(),
            Phase::Postflop(p) => {
                p.study.home();
                p.pos = p.study.poll();
            }
        }
    }

    /// Starts browsing a spot.
    fn open(&mut self, spot: Spot, preflop: Option<PreflopNav>) {
        match Study::new(Arc::clone(&self.settings), spot) {
            Ok(mut study) => {
                self.last_ranges = Some((study.spot().ranges.clone(), study.spot().board.clone()));
                let pos = study.poll();
                self.phase = Phase::Postflop(Box::new(Postflop { study, pos, preflop }));
                self.view = GridView::Strategy;
                self.input = None;
                self.message = None;
            }
            Err(e) => self.message = Some(e),
        }
    }

    fn submit_input(&mut self) {
        let Some(input) = self.input.clone() else { return };
        let text = input.text.trim().to_string();
        let board_now: Vec<Card> = match &self.phase {
            Phase::Preflop(_) => Vec::new(),
            Phase::Postflop(p) => p.study.board(),
        };
        let need = match &self.phase {
            Phase::Preflop(_) => 3,
            Phase::Postflop(_) => 1,
        };
        let cards = if text.is_empty() {
            random_cards(CardSet::from_cards(&board_now), need)
        } else {
            match parse_cards(&text) {
                Ok(c) => c,
                Err(e) => {
                    self.message = Some(e.to_string());
                    return;
                }
            }
        };
        if cards.len() != need || CardSet::from_cards(&cards).len() != need {
            self.message = Some(format!("Enter {need} different card{}.", if need == 1 { "" } else { "s" }));
            return;
        }
        self.message = None;
        match &mut self.phase {
            Phase::Preflop(nav) => {
                let Some(mut spot) = nav.spot() else { return };
                spot.board = cards;
                let pre = (**nav).clone();
                self.open(spot, Some(pre));
            }
            Phase::Postflop(p) => {
                if let Err(e) = p.study.deal(cards[0]) {
                    self.message = Some(e);
                }
                p.pos = p.study.poll();
            }
        }
        self.input = None;
    }

    /// A new random board for the current spot (same ranges and line).
    fn new_board(&mut self) {
        let Phase::Postflop(p) = &mut self.phase else {
            self.message = Some("Reach the flop first (or press E to type a spot).".into());
            return;
        };
        let mut spot = p.study.spot().clone();
        spot.board = random_cards(CardSet::EMPTY, spot.board.len());
        let pre = p.preflop.clone();
        self.open(spot, pre);
    }

    fn open_editor(&mut self) {
        let ed = match &self.phase {
            Phase::Postflop(p) => SpotEditor::from_spot(p.study.spot()),
            Phase::Preflop(nav) => match nav.spot() {
                Some(spot) => SpotEditor::from_spot(&spot),
                None => SpotEditor::example(),
            },
        };
        self.editor = Some(ed);
        self.input = None;
    }

    pub fn on_click(&mut self, hit: Hit) {
        match hit {
            Hit::Cell(g) => self.cursor = g,
            Hit::Action(i) => {
                if self.editor.is_none() && self.input.is_none() {
                    self.take(i)
                }
            }
            Hit::Decision(_) => {}
            Hit::Field(i) => {
                if let Some(ed) = &mut self.editor {
                    ed.cursor = i;
                    ed.editing = false;
                }
            }
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Outcome {
        if self.editor.is_some() {
            return self.editor_key(key);
        }
        // Grid cursor and views work everywhere.
        match key.code {
            KeyCode::Up => {
                self.cursor = self.cursor.checked_sub(13).unwrap_or(self.cursor);
                return Outcome::None;
            }
            KeyCode::Down => {
                if self.cursor + 13 < 169 {
                    self.cursor += 13;
                }
                return Outcome::None;
            }
            KeyCode::Left => {
                if !self.cursor.is_multiple_of(13) {
                    self.cursor -= 1;
                }
                return Outcome::None;
            }
            KeyCode::Right => {
                if self.cursor % 13 < 12 {
                    self.cursor += 1;
                }
                return Outcome::None;
            }
            KeyCode::Tab => {
                self.cycle_view(1);
                return Outcome::None;
            }
            KeyCode::BackTab => {
                self.cycle_view(-1);
                return Outcome::None;
            }
            KeyCode::Esc => return Outcome::Leave,
            _ => {}
        }
        if self.input.is_some() {
            match key.code {
                KeyCode::Home => {
                    self.input = None;
                    self.home();
                }
                KeyCode::Enter => self.submit_input(),
                KeyCode::Backspace => {
                    let empty = self.input.as_ref().is_some_and(|i| i.text.is_empty());
                    if empty {
                        self.input = None;
                        self.back();
                    } else if let Some(i) = &mut self.input {
                        i.text.pop();
                    }
                }
                KeyCode::Char(c)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && (c.is_ascii_alphanumeric() || c == ' ') =>
                {
                    if let Some(i) = &mut self.input {
                        if i.text.len() < 16 {
                            i.text.push(c);
                        }
                    }
                }
                _ => {}
            }
            return Outcome::None;
        }
        match key.code {
            KeyCode::Backspace | KeyCode::Char('b') => self.back(),
            KeyCode::Home | KeyCode::Char('g') => self.home(),
            KeyCode::Char('e') => self.open_editor(),
            KeyCode::Char('n') => self.new_board(),
            KeyCode::Char('p') => {
                let has_pre = matches!(&self.phase, Phase::Postflop(p) if p.preflop.is_some());
                if has_pre {
                    while matches!(&self.phase, Phase::Postflop(_)) {
                        self.back();
                    }
                } else if let Phase::Postflop(_) = self.phase {
                    self.phase = Phase::Preflop(Box::new(PreflopNav::new(Arc::clone(&self.settings))));
                    self.view = GridView::Strategy;
                }
            }
            KeyCode::Char('?') => return Outcome::Help,
            KeyCode::Char('q') => return Outcome::Quit,
            KeyCode::Char(ch) => {
                let ch = match ch.to_ascii_lowercase() {
                    'x' | 'k' => 'c',
                    other => other,
                };
                let keys = self.action_keys();
                let idx = if ch == 'a' {
                    keys.iter().rposition(|k| k.is_ascii_digit())
                } else {
                    keys.iter().position(|&k| k == ch)
                };
                if let Some(i) = idx {
                    self.take(i);
                }
            }
            _ => {}
        }
        Outcome::None
    }

    fn editor_key(&mut self, key: KeyEvent) -> Outcome {
        let Some(ed) = &mut self.editor else { return Outcome::None };
        if ed.editing {
            match key.code {
                KeyCode::Enter | KeyCode::Esc | KeyCode::Tab => ed.editing = false,
                KeyCode::Backspace => {
                    if let Some(t) = ed.text_mut() {
                        t.pop();
                    }
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    if let Some(t) = ed.text_mut() {
                        t.clear();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(t) = ed.text_mut() {
                        t.push(c);
                    }
                }
                _ => {}
            }
            ed.error = None;
            return Outcome::None;
        }
        let n = ed.rows().len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => ed.cursor = ed.cursor.checked_sub(1).unwrap_or(n - 1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => ed.cursor = (ed.cursor + 1) % n,
            KeyCode::Esc => self.editor = None,
            KeyCode::Char('q') => return Outcome::Quit,
            KeyCode::Char('s') => self.solve_editor(),
            KeyCode::Left | KeyCode::Right if ed.row() == EditorRow::Players => ed.toggle_players(),
            KeyCode::Enter | KeyCode::Char(' ') => match ed.row() {
                EditorRow::Players => ed.toggle_players(),
                EditorRow::Solve => self.solve_editor(),
                _ => ed.editing = true,
            },
            KeyCode::Backspace | KeyCode::Delete => {
                if let Some(t) = ed.text_mut() {
                    t.clear();
                    ed.editing = true;
                }
            }
            _ => {}
        }
        Outcome::None
    }

    fn solve_editor(&mut self) {
        let Some(ed) = &mut self.editor else { return };
        match ed.spot() {
            Ok(spot) => {
                self.editor = None;
                self.open(spot, None);
            }
            Err(e) => ed.error = Some(e),
        }
    }

    /// Street of the current position (for titles).
    pub fn street(&self) -> Street {
        match &self.phase {
            Phase::Preflop(_) => Street::Preflop,
            Phase::Postflop(p) => Street::from_board_len(p.study.board().len()),
        }
    }
}
