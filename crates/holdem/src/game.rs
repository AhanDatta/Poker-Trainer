//! N-player No-Limit Hold'em hand state machine.
//!
//! The engine tracks blinds/antes, betting rounds, all-ins, side pots and the
//! showdown. It does not deal cards itself: the caller provides hole cards up
//! front and calls [`HandState::deal_board`] whenever the state is
//! [`Phase::AwaitingBoard`]. Chip amounts are integers (the trainer uses
//! 100 chips = 1 big blind).

use crate::card::Card;
use crate::eval::{EvalMask, HandValue};
use crate::menu::{BetMenu, Situation};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Chip amount.
pub type Chips = i64;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    pub fn next(self) -> Option<Street> {
        match self {
            Street::Preflop => Some(Street::Flop),
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }

    /// Number of board cards once this street is dealt.
    pub fn board_len(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }

    /// Street corresponding to a board with `n` cards.
    pub fn from_board_len(n: usize) -> Street {
        match n {
            0 => Street::Preflop,
            3 => Street::Flop,
            4 => Street::Turn,
            _ => Street::River,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Street::Preflop => "Preflop",
            Street::Flop => "Flop",
            Street::Turn => "Turn",
            Street::River => "River",
        }
    }
}

impl fmt::Display for Street {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A player decision. `Raise(to)` is a bet or raise *to* a total street
/// commitment of `to` chips.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Action {
    Fold,
    Check,
    Call,
    Raise(Chips),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableConfig {
    pub num_players: usize,
    pub small_blind: Chips,
    pub big_blind: Chips,
    pub ante: Chips,
    /// Rake as a fraction of the pot ("no flop, no drop": only pots that
    /// see a flop are raked; uncalled bets are never raked).
    #[serde(default)]
    pub rake_rate: f64,
    /// Maximum rake per hand, in chips.
    #[serde(default)]
    pub rake_cap: Chips,
}

impl TableConfig {
    pub fn new(num_players: usize, small_blind: Chips, big_blind: Chips) -> TableConfig {
        TableConfig { num_players, small_blind, big_blind, ante: 0, rake_rate: 0.0, rake_cap: 0 }
    }

    pub fn with_rake(mut self, rate: f64, cap: Chips) -> TableConfig {
        self.rake_rate = rate;
        self.rake_cap = cap;
        self
    }
}

/// A resolved action in the hand history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionRecord {
    pub player: usize,
    pub street: Street,
    pub action: Action,
    /// Chips added to the pot by this action.
    pub put_in: Chips,
    /// The player's street commitment after the action.
    pub street_commit: Chips,
    /// Whether the action put the player all-in.
    pub all_in: bool,
    /// Pot size before the action.
    pub pot_before: Chips,
    /// Chips needed to call before the action.
    pub to_call_before: Chips,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Waiting for `to_act` to choose an action.
    Betting,
    /// The betting round is over; the next street must be dealt.
    AwaitingBoard,
    /// The hand is over (everyone else folded, or showdown).
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    NotBetting,
    NotAwaitingBoard,
    IllegalAction(String),
    BadBoard(String),
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::NotBetting => f.write_str("no player is to act"),
            GameError::NotAwaitingBoard => f.write_str("the board can't be dealt now"),
            GameError::IllegalAction(s) => write!(f, "illegal action: {s}"),
            GameError::BadBoard(s) => write!(f, "bad board: {s}"),
        }
    }
}

impl std::error::Error for GameError {}

/// Outcome of a finished hand.
#[derive(Clone, Debug, PartialEq)]
pub struct Settlement {
    /// Chips each player receives from the pot.
    pub payouts: Vec<Chips>,
    /// Net result per player (`payout - total contribution`).
    pub net: Vec<Chips>,
    /// Hand values of players who reached showdown.
    pub hand_values: Vec<Option<HandValue>>,
    pub showdown: bool,
    /// Rake taken from the pot.
    pub rake: Chips,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandState {
    pub config: TableConfig,
    pub button: usize,
    pub hole: Vec<[Card; 2]>,
    pub board: Vec<Card>,
    pub street: Street,
    pub starting_stacks: Vec<Chips>,
    /// Chips behind.
    pub stacks: Vec<Chips>,
    /// Chips committed on the current street.
    pub street_commit: Vec<Chips>,
    /// Chips committed over the whole hand (including antes).
    pub total_commit: Vec<Chips>,
    pub folded: Vec<bool>,
    pub current_bet: Chips,
    pub last_raise_inc: Chips,
    pub num_raises: u32,
    pub limpers: u32,
    pub to_act: Option<usize>,
    needs_action: Vec<bool>,
    pub history: Vec<ActionRecord>,
    pub phase: Phase,
}

impl HandState {
    /// Starts a hand: posts antes and blinds and sets the first player to act.
    pub fn new(config: TableConfig, button: usize, stacks: Vec<Chips>, hole: Vec<[Card; 2]>) -> HandState {
        let n = config.num_players;
        assert!((2..=10).contains(&n), "2..=10 players supported");
        assert_eq!(stacks.len(), n);
        assert_eq!(hole.len(), n);
        let mut s = HandState {
            button: button % n,
            hole,
            board: Vec::new(),
            street: Street::Preflop,
            starting_stacks: stacks.clone(),
            stacks,
            street_commit: vec![0; n],
            total_commit: vec![0; n],
            folded: vec![false; n],
            current_bet: 0,
            last_raise_inc: config.big_blind,
            num_raises: 0,
            limpers: 0,
            to_act: None,
            needs_action: vec![false; n],
            history: Vec::new(),
            phase: Phase::Betting,
            config,
        };
        if s.config.ante > 0 {
            for p in 0..n {
                let a = s.config.ante.min(s.stacks[p]);
                s.stacks[p] -= a;
                s.total_commit[p] += a;
            }
        }
        let (sb, bb) = (s.sb_seat(), s.bb_seat());
        s.post(sb, s.config.small_blind);
        s.post(bb, s.config.big_blind);
        s.current_bet = s.street_commit.iter().copied().max().unwrap_or(0).max(s.config.big_blind);
        for p in 0..n {
            s.needs_action[p] = s.stacks[p] > 0;
        }
        let first = if n == 2 { sb } else { (bb + 1) % n };
        s.to_act = s.next_needing_action_from(first);
        if s.to_act.is_none() || s.num_can_act() == 0 {
            s.finish_round();
        }
        s
    }

    fn post(&mut self, p: usize, amount: Chips) {
        let a = amount.min(self.stacks[p]);
        self.stacks[p] -= a;
        self.street_commit[p] += a;
        self.total_commit[p] += a;
    }

    pub fn num_players(&self) -> usize {
        self.config.num_players
    }

    pub fn sb_seat(&self) -> usize {
        if self.num_players() == 2 {
            self.button
        } else {
            (self.button + 1) % self.num_players()
        }
    }

    pub fn bb_seat(&self) -> usize {
        if self.num_players() == 2 {
            (self.button + 1) % 2
        } else {
            (self.button + 2) % self.num_players()
        }
    }

    /// Position name of `seat` (BTN, SB, BB, UTG, ..., CO).
    pub fn position_name(&self, seat: usize) -> &'static str {
        position_name(self.num_players(), (seat + self.num_players() - self.button) % self.num_players())
    }

    pub fn pot(&self) -> Chips {
        self.total_commit.iter().sum()
    }

    pub fn players_in_hand(&self) -> Vec<usize> {
        (0..self.num_players()).filter(|&p| !self.folded[p]).collect()
    }

    pub fn is_all_in(&self, p: usize) -> bool {
        !self.folded[p] && self.stacks[p] == 0
    }

    fn num_can_act(&self) -> usize {
        (0..self.num_players()).filter(|&p| !self.folded[p] && self.stacks[p] > 0).count()
    }

    fn next_needing_action_from(&self, start: usize) -> Option<usize> {
        let n = self.num_players();
        (0..n).map(|i| (start + i) % n).find(|&p| self.needs_action[p] && !self.folded[p])
    }

    /// Chips `p` needs to call.
    pub fn to_call(&self, p: usize) -> Chips {
        (self.current_bet - self.street_commit[p]).max(0).min(self.stacks[p])
    }

    /// The betting situation for `p` (used by the bet menu).
    pub fn situation(&self, p: usize) -> Situation {
        let opp_max = (0..self.num_players())
            .filter(|&q| q != p && !self.folded[q])
            .map(|q| self.street_commit[q] + self.stacks[q])
            .max()
            .unwrap_or(0);
        let max_to = (self.street_commit[p] + self.stacks[p]).min(opp_max);
        let inc = self.last_raise_inc.max(self.config.big_blind);
        Situation {
            street: self.street,
            pot: self.pot(),
            to_call: self.to_call(p),
            current_bet: self.current_bet,
            max_to,
            min_raise_to: self.current_bet + inc,
            num_raises: self.num_raises,
            limpers: self.limpers,
            big_blind: self.config.big_blind,
        }
    }

    /// Menu actions for the player to act (empty if nobody is to act).
    pub fn legal_actions(&self, menu: &BetMenu) -> Vec<Action> {
        match (self.phase, self.to_act) {
            (Phase::Betting, Some(p)) => menu.actions(&self.situation(p)),
            _ => Vec::new(),
        }
    }

    /// Applies an action for the player to act. Any legal NLHE action is
    /// accepted (not only menu actions).
    pub fn apply(&mut self, action: Action) -> Result<ActionRecord, GameError> {
        let p = match (self.phase, self.to_act) {
            (Phase::Betting, Some(p)) => p,
            _ => return Err(GameError::NotBetting),
        };
        let to_call = self.to_call(p);
        let pot_before = self.pot();
        let mut put_in = 0;
        match action {
            Action::Fold => {
                self.folded[p] = true;
            }
            Action::Check => {
                if to_call > 0 {
                    return Err(GameError::IllegalAction("can't check facing a bet".into()));
                }
            }
            Action::Call => {
                if to_call == 0 {
                    return Err(GameError::IllegalAction("nothing to call".into()));
                }
                put_in = to_call;
                if self.street == Street::Preflop && self.num_raises == 0 {
                    self.limpers += 1;
                }
            }
            Action::Raise(to) => {
                let sit = self.situation(p);
                if !sit.can_raise() {
                    return Err(GameError::IllegalAction("raising isn't possible".into()));
                }
                let all_in_to = self.street_commit[p] + self.stacks[p];
                if to <= self.current_bet || to > sit.max_to {
                    return Err(GameError::IllegalAction(format!("raise to {to} out of range")));
                }
                if to < sit.min_raise_to && to != all_in_to && to != sit.max_to {
                    return Err(GameError::IllegalAction(format!(
                        "raise to {to} is below the minimum {}",
                        sit.min_raise_to
                    )));
                }
                let inc = to - self.current_bet;
                if inc >= self.last_raise_inc {
                    self.last_raise_inc = inc;
                }
                self.current_bet = to;
                self.num_raises += 1;
                put_in = to - self.street_commit[p];
                for q in 0..self.num_players() {
                    if q != p && !self.folded[q] && self.stacks[q] > 0 {
                        self.needs_action[q] = true;
                    }
                }
            }
        }
        self.stacks[p] -= put_in;
        self.street_commit[p] += put_in;
        self.total_commit[p] += put_in;
        self.needs_action[p] = false;
        let record = ActionRecord {
            player: p,
            street: self.street,
            action,
            put_in,
            street_commit: self.street_commit[p],
            all_in: self.stacks[p] == 0 && !self.folded[p],
            pot_before,
            to_call_before: to_call,
        };
        self.history.push(record.clone());

        if self.players_in_hand().len() == 1 {
            self.phase = Phase::Finished;
            self.to_act = None;
        } else {
            match self.next_needing_action_from((p + 1) % self.num_players()) {
                Some(q) => self.to_act = Some(q),
                None => self.finish_round(),
            }
        }
        debug_assert_eq!(self.invariants(), Ok(()), "after {action:?}");
        Ok(record)
    }

    fn finish_round(&mut self) {
        self.to_act = None;
        self.phase = if self.street == Street::River || self.players_in_hand().len() <= 1 {
            Phase::Finished
        } else {
            Phase::AwaitingBoard
        };
    }

    /// Whether no further betting is possible this hand (all-in runout).
    pub fn is_runout(&self) -> bool {
        self.phase != Phase::Finished && self.num_can_act() < 2 && self.to_act.is_none()
    }

    /// Number of cards to deal for the next street.
    pub fn cards_needed(&self) -> usize {
        match self.street.next() {
            Some(s) => s.board_len() - self.board.len(),
            None => 0,
        }
    }

    /// Deals the next street's card(s) and starts its betting round (or skips
    /// it when fewer than two players can still bet).
    pub fn deal_board(&mut self, cards: &[Card]) -> Result<(), GameError> {
        if self.phase != Phase::AwaitingBoard {
            return Err(GameError::NotAwaitingBoard);
        }
        if cards.len() != self.cards_needed() {
            return Err(GameError::BadBoard(format!("expected {} card(s)", self.cards_needed())));
        }
        self.street = self.street.next().expect("not river");
        self.board.extend_from_slice(cards);
        self.street_commit.iter_mut().for_each(|c| *c = 0);
        self.current_bet = 0;
        self.last_raise_inc = self.config.big_blind;
        self.num_raises = 0;
        self.limpers = 0;
        for p in 0..self.num_players() {
            self.needs_action[p] = !self.folded[p] && self.stacks[p] > 0;
        }
        if self.num_can_act() < 2 {
            self.needs_action.iter_mut().for_each(|x| *x = false);
            self.finish_round();
            return Ok(());
        }
        self.phase = Phase::Betting;
        self.to_act = self.next_needing_action_from((self.button + 1) % self.num_players());
        debug_assert_eq!(self.invariants(), Ok(()), "after dealing {cards:?}");
        Ok(())
    }

    /// Internal consistency checks. They run after every state change in
    /// debug builds (and in the property tests).
    pub fn invariants(&self) -> Result<(), String> {
        let n = self.num_players();
        let start: Chips = self.starting_stacks.iter().sum();
        let now: Chips = self.stacks.iter().sum::<Chips>() + self.pot();
        if start != now {
            return Err(format!("chips not conserved: started with {start}, now {now}"));
        }
        for p in 0..n {
            if self.stacks[p] < 0 || self.street_commit[p] < 0 {
                return Err(format!("negative chips for seat {p}"));
            }
            if self.street_commit[p] > self.total_commit[p] {
                return Err(format!("street commitment exceeds total for seat {p}"));
            }
            if self.street_commit[p] > self.current_bet {
                return Err(format!("seat {p} committed more than the current bet"));
            }
        }
        match (self.phase, self.to_act) {
            (Phase::Betting, Some(p)) => {
                if self.folded[p] || self.stacks[p] == 0 {
                    return Err(format!("seat {p} can't act (folded or all-in)"));
                }
            }
            (Phase::Betting, None) => return Err("betting with nobody to act".into()),
            (_, Some(_)) => return Err("someone to act outside a betting round".into()),
            _ => {}
        }
        if self.players_in_hand().is_empty() {
            return Err("everyone folded".into());
        }
        if self.board.len() != self.street.board_len() {
            return Err("board doesn't match the street".into());
        }
        Ok(())
    }

    /// Settles a finished hand (side pots, split pots, uncalled bets).
    pub fn settle(&self) -> Settlement {
        let n = self.num_players();
        let live = self.players_in_hand();
        let showdown = live.len() > 1;
        let hand_values: Vec<Option<HandValue>> = (0..n)
            .map(|p| {
                if showdown && !self.folded[p] && self.board.len() == 5 {
                    let m = EvalMask::from_cards(&self.board).add(self.hole[p][0]).add(self.hole[p][1]);
                    Some(m.evaluate())
                } else {
                    None
                }
            })
            .collect();
        // Split the pot into layers by contribution level (side pots).
        let mut levels: Vec<Chips> = self.total_commit.clone();
        levels.sort_unstable();
        levels.dedup();
        let mut pots: Vec<(Chips, usize, Vec<usize>)> = Vec::new(); // (amount, contributors, eligible)
        let mut prev = 0;
        for level in levels {
            if level <= prev {
                continue;
            }
            let slice: Chips = self.total_commit.iter().map(|&c| c.min(level) - c.min(prev)).sum();
            let contributors = self.total_commit.iter().filter(|&&c| c >= level).count();
            let eligible: Vec<usize> =
                live.iter().copied().filter(|&p| self.total_commit[p] >= level).collect();
            prev = level;
            if slice > 0 {
                pots.push((slice, contributors, eligible));
            }
        }
        // Rake: only pots that saw a flop, only matched chips (layers with
        // at least two contributors), capped; taken from the main pot first.
        let mut rake = 0;
        if self.config.rake_rate > 0.0 && self.board.len() >= 3 {
            let matched: Chips = pots.iter().filter(|(_, c, _)| *c >= 2).map(|(a, _, _)| *a).sum();
            rake =
                ((matched as f64 * self.config.rake_rate).floor() as Chips).min(self.config.rake_cap).max(0);
            let mut left = rake;
            for (amount, contributors, _) in pots.iter_mut() {
                if left == 0 {
                    break;
                }
                if *contributors >= 2 {
                    let take = left.min(*amount);
                    *amount -= take;
                    left -= take;
                }
            }
        }
        let mut payouts = vec![0; n];
        for (slice, _, eligible) in pots {
            if slice == 0 {
                continue;
            }
            let winners: Vec<usize> = if eligible.len() <= 1 {
                if eligible.is_empty() {
                    // Everyone eligible folded: give it to the remaining live
                    // player(s).
                    live.clone()
                } else {
                    eligible
                }
            } else {
                let best = eligible.iter().filter_map(|&p| hand_values[p]).max();
                eligible.into_iter().filter(|&p| hand_values[p] == best).collect()
            };
            let share = slice / winners.len() as Chips;
            let mut rem = slice - share * winners.len() as Chips;
            // Odd chips go to the first winner(s) left of the button.
            let mut ordered = winners.clone();
            ordered.sort_by_key(|&p| (p + n - self.button - 1) % n);
            for p in ordered {
                payouts[p] += share;
                if rem > 0 {
                    payouts[p] += 1;
                    rem -= 1;
                }
            }
        }
        let net = (0..n).map(|p| payouts[p] - self.total_commit[p]).collect();
        Settlement { payouts, net, hand_values, showdown, rake }
    }
}

/// Position name for a seat at `offset` seats after the button.
pub fn position_name(num_players: usize, offset: usize) -> &'static str {
    if num_players == 2 {
        return if offset == 0 { "SB" } else { "BB" };
    }
    match offset {
        0 => "BTN",
        1 => "SB",
        2 => "BB",
        _ => {
            let early: &[&'static str] = match num_players - 3 {
                1 => &["CO"],
                2 => &["HJ", "CO"],
                3 => &["UTG", "HJ", "CO"],
                4 => &["UTG", "LJ", "HJ", "CO"],
                5 => &["UTG", "UTG+1", "LJ", "HJ", "CO"],
                6 => &["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"],
                _ => &["UTG", "UTG+1", "UTG+2", "MP", "LJ", "HJ", "CO"],
            };
            early.get(offset - 3).copied().unwrap_or("MP")
        }
    }
}

/// Offset after the button of the position called `name` (any case) at a
/// table of `num_players`, or `None` if the table has no such position.
/// Heads-up, "BTN" is the small blind (who has the button).
pub fn position_offset(num_players: usize, name: &str) -> Option<usize> {
    if num_players == 2 && name.eq_ignore_ascii_case("BTN") {
        return Some(0);
    }
    (0..num_players).find(|&o| position_name(num_players, o).eq_ignore_ascii_case(name))
}

/// The positions of a table in preflop action order: UTG first, the big
/// blind last (heads-up: SB, BB).
pub fn positions(num_players: usize) -> Vec<&'static str> {
    if num_players == 2 {
        return vec!["SB", "BB"];
    }
    (3..num_players).chain(0..3).map(|o| position_name(num_players, o)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn holes(n: usize) -> Vec<[Card; 2]> {
        let cards = parse_cards("AsAhKsKhQsQhJsJhTsTh9s9h8s8h7s7h6s6h5s5h").unwrap();
        (0..n).map(|i| [cards[2 * i], cards[2 * i + 1]]).collect()
    }

    fn hu() -> HandState {
        HandState::new(TableConfig::new(2, 50, 100), 0, vec![10000, 10000], holes(2))
    }

    #[test]
    fn heads_up_blinds_and_order() {
        let s = hu();
        assert_eq!(s.sb_seat(), 0);
        assert_eq!(s.bb_seat(), 1);
        assert_eq!(s.to_act, Some(0)); // SB acts first preflop
        assert_eq!(s.pot(), 150);
        assert_eq!(s.to_call(0), 50);
        assert_eq!(s.position_name(0), "SB");
        assert_eq!(s.position_name(1), "BB");
    }

    #[test]
    fn limp_check_goes_to_flop_with_bb_first() {
        let mut s = hu();
        s.apply(Action::Call).unwrap();
        assert_eq!(s.to_act, Some(1)); // BB option
        assert!(s.apply(Action::Call).is_err());
        s.apply(Action::Check).unwrap();
        assert_eq!(s.phase, Phase::AwaitingBoard);
        s.deal_board(&parse_cards("2c3c4d").unwrap()).unwrap();
        assert_eq!(s.street, Street::Flop);
        assert_eq!(s.to_act, Some(1)); // BB first postflop
        assert_eq!(s.pot(), 200);
    }

    #[test]
    fn raise_reopens_action() {
        let mut s = hu();
        s.apply(Action::Raise(250)).unwrap();
        assert_eq!(s.to_act, Some(1));
        s.apply(Action::Raise(875)).unwrap();
        assert_eq!(s.to_act, Some(0));
        assert_eq!(s.situation(0).min_raise_to, 875 + 625);
        s.apply(Action::Call).unwrap();
        assert_eq!(s.phase, Phase::AwaitingBoard);
        assert_eq!(s.pot(), 1750);
    }

    #[test]
    fn fold_ends_hand() {
        let mut s = hu();
        s.apply(Action::Fold).unwrap();
        assert_eq!(s.phase, Phase::Finished);
        let st = s.settle();
        assert_eq!(st.net, vec![-50, 50]);
        assert!(!st.showdown);
    }

    #[test]
    fn all_in_runout_and_showdown() {
        let mut s = hu();
        s.apply(Action::Raise(10000)).unwrap();
        s.apply(Action::Call).unwrap();
        assert_eq!(s.phase, Phase::AwaitingBoard);
        s.deal_board(&parse_cards("2c3c4d").unwrap()).unwrap();
        assert_eq!(s.phase, Phase::AwaitingBoard);
        s.deal_board(&parse_cards("8d").unwrap()).unwrap();
        s.deal_board(&parse_cards("9c").unwrap()).unwrap();
        assert_eq!(s.phase, Phase::Finished);
        let st = s.settle();
        assert!(st.showdown);
        assert_eq!(st.net, vec![10000, -10000]); // AA beats KK
    }

    #[test]
    fn side_pots() {
        // 3 players: seat 0 (BTN) AA short stack, seat 1 (SB) KK, seat 2 (BB) QQ.
        let mut s = HandState::new(TableConfig::new(3, 50, 100), 0, vec![1000, 5000, 5000], holes(3));
        assert_eq!(s.to_act, Some(0)); // BTN is first in 3-handed
        s.apply(Action::Raise(1000)).unwrap(); // all-in
        s.apply(Action::Raise(3000)).unwrap(); // SB re-raises
        s.apply(Action::Call).unwrap(); // BB calls 3000
        assert_eq!(s.phase, Phase::AwaitingBoard);
        s.deal_board(&parse_cards("2c3c4d").unwrap()).unwrap();
        assert_eq!(s.phase, Phase::Betting);
        assert_eq!(s.to_act, Some(1));
        s.apply(Action::Check).unwrap();
        s.apply(Action::Check).unwrap();
        s.deal_board(&parse_cards("8d").unwrap()).unwrap();
        s.apply(Action::Check).unwrap();
        s.apply(Action::Check).unwrap();
        s.deal_board(&parse_cards("9c").unwrap()).unwrap();
        s.apply(Action::Check).unwrap();
        s.apply(Action::Check).unwrap();
        assert_eq!(s.phase, Phase::Finished);
        let st = s.settle();
        // Main pot 3000 to AA; side pot 4000 to KK.
        assert_eq!(st.payouts, vec![3000, 4000, 0]);
        assert_eq!(st.net.iter().sum::<Chips>(), 0);
    }

    #[test]
    fn rake_no_flop_no_drop() {
        let cfg = TableConfig::new(2, 50, 100).with_rake(0.05, 300);
        // Preflop fold: no rake.
        let mut s = HandState::new(cfg.clone(), 0, vec![10000, 10000], holes(2));
        s.apply(Action::Raise(250)).unwrap();
        s.apply(Action::Fold).unwrap();
        assert_eq!(s.settle().rake, 0);
        // Flop bet folded to: the matched 5bb pot is raked, the bet returned.
        let mut s = HandState::new(cfg.clone(), 0, vec![10000, 10000], holes(2));
        s.apply(Action::Raise(250)).unwrap();
        s.apply(Action::Call).unwrap();
        s.deal_board(&parse_cards("2c3c4d").unwrap()).unwrap();
        s.apply(Action::Check).unwrap();
        s.apply(Action::Raise(300)).unwrap();
        s.apply(Action::Fold).unwrap();
        let st = s.settle();
        assert_eq!(st.rake, 25);
        assert_eq!(st.net, vec![250 - 25, -250]);
        // Big all-in: rake capped.
        let mut s = HandState::new(cfg, 0, vec![10000, 10000], holes(2));
        s.apply(Action::Raise(10000)).unwrap();
        s.apply(Action::Call).unwrap();
        for c in ["2c3c4d", "8d", "9c"] {
            s.deal_board(&parse_cards(c).unwrap()).unwrap();
        }
        let st = s.settle();
        assert_eq!(st.rake, 300);
        assert_eq!(st.net.iter().sum::<Chips>(), -300);
    }

    #[test]
    fn menu_actions_from_state() {
        let s = hu();
        let menu = BetMenu::default();
        assert_eq!(
            s.legal_actions(&menu),
            vec![Action::Fold, Action::Call, Action::Raise(250), Action::Raise(10000)]
        );
    }

    #[test]
    fn position_names() {
        let names: Vec<_> = (0..6).map(|o| position_name(6, o)).collect();
        assert_eq!(names, vec!["BTN", "SB", "BB", "UTG", "HJ", "CO"]);
        let names: Vec<_> = (0..9).map(|o| position_name(9, o)).collect();
        assert_eq!(names, vec!["BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"]);
    }

    #[test]
    fn position_offsets_invert_position_names() {
        for n in 2..=10 {
            for o in 0..n {
                let name = position_name(n, o);
                assert_eq!(position_offset(n, name), Some(o), "{n} players, {name}");
                assert_eq!(position_offset(n, &name.to_lowercase()), Some(o), "{n} players, {name}");
            }
            let all = positions(n);
            assert_eq!(all.len(), n);
            let mut offsets: Vec<usize> = all.iter().map(|p| position_offset(n, p).unwrap()).collect();
            assert_eq!(*all.last().unwrap(), "BB", "{n} players");
            offsets.sort_unstable();
            assert_eq!(offsets, (0..n).collect::<Vec<_>>(), "{n} players: every position once");
        }
        assert_eq!(positions(6), vec!["UTG", "HJ", "CO", "BTN", "SB", "BB"]);
        assert_eq!(positions(9), vec!["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]);
        assert_eq!(positions(2), vec!["SB", "BB"]);
        // Heads-up the button is the small blind.
        assert_eq!(position_offset(2, "btn"), Some(0));
        assert_eq!(position_offset(6, "UTG+2"), None);
        assert_eq!(position_offset(6, "LJ"), None);
        assert_eq!(position_offset(2, "CO"), None);
        assert_eq!(position_offset(9, "nope"), None);
    }
}
