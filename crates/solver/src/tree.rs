//! Heads-up postflop betting state and game-tree construction.

use crate::sampling::sample_cards;
use holdem::menu::postflop_actions;
use holdem::{Action, Card, CardSet, Chips, PostflopMenu, Situation, Street};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Betting state of a heads-up postflop street. Player 0 is out of position
/// (acts first on every street), player 1 is in position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BettingState {
    pub street: Street,
    /// Total pot including current street commitments.
    pub pot: Chips,
    /// Street commitments.
    pub commit: [Chips; 2],
    /// Chips behind.
    pub stack: [Chips; 2],
    pub current_bet: Chips,
    pub last_raise_inc: Chips,
    pub num_raises: u32,
    pub to_act: usize,
    /// Whether each player still has to act in this betting round.
    pub needs: [bool; 2],
}

pub(crate) enum Transition {
    Continue(BettingState),
    Fold(usize),
    StreetOver(BettingState),
}

impl BettingState {
    /// State at the start of a street (OOP to act, nothing committed).
    pub fn street_start(street: Street, pot: Chips, stack: [Chips; 2], big_blind: Chips) -> BettingState {
        BettingState {
            street,
            pot,
            commit: [0, 0],
            stack,
            current_bet: 0,
            last_raise_inc: big_blind,
            num_raises: 0,
            to_act: 0,
            needs: [stack[0] > 0, stack[1] > 0],
        }
    }

    pub fn situation(&self, big_blind: Chips) -> Situation {
        let p = self.to_act;
        let o = p ^ 1;
        Situation {
            street: self.street,
            pot: self.pot,
            to_call: (self.current_bet - self.commit[p]).max(0).min(self.stack[p]),
            current_bet: self.current_bet,
            max_to: (self.commit[p] + self.stack[p]).min(self.commit[o] + self.stack[o]),
            min_raise_to: self.current_bet + self.last_raise_inc.max(big_blind),
            num_raises: self.num_raises,
            limpers: 0,
            big_blind,
        }
    }

    /// The state after `action` when the betting round continues, `None`
    /// when the action folds or closes the street.
    pub fn after(&self, action: Action) -> Option<BettingState> {
        match self.apply(action).0 {
            Transition::Continue(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn apply(&self, action: Action) -> (Transition, Chips) {
        let p = self.to_act;
        let o = p ^ 1;
        let mut s = *self;
        let mut put = 0;
        match action {
            Action::Fold => return (Transition::Fold(p), 0),
            Action::Check => {}
            Action::Call => {
                put = (s.current_bet - s.commit[p]).max(0).min(s.stack[p]);
            }
            Action::Raise(to) => {
                put = to - s.commit[p];
                let inc = to - s.current_bet;
                if inc >= s.last_raise_inc {
                    s.last_raise_inc = inc;
                }
                s.current_bet = to;
                s.num_raises += 1;
                s.needs[o] = s.stack[o] > 0;
            }
        }
        s.commit[p] += put;
        s.stack[p] -= put;
        s.pot += put;
        s.needs[p] = false;
        if s.needs[o] {
            s.to_act = o;
            (Transition::Continue(s), put)
        } else {
            (Transition::StreetOver(s), put)
        }
    }
}

/// How turn and river cards are enumerated in the game tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChanceMode {
    /// Every possible card.
    Exact,
    /// A stratified sample of `turn` turn cards and `river` river cards per
    /// branch (all-in runouts are always enumerated exactly).
    Sampled { turn: usize, river: usize, seed: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Player(u8),
    Chance,
    Fold(u8),
    Showdown,
}

/// A game-tree node. Storage vectors are laid out action-major:
/// `regrets[a * num_hands + i]`.
#[derive(Debug)]
pub(crate) struct Node {
    pub kind: Kind,
    /// Player node: the actions; chance node: empty.
    pub actions: Vec<Action>,
    /// Chance node: the card dealt to reach each child.
    pub cards: Vec<Card>,
    pub children: Vec<Node>,
    /// Chips each player has invested since the root.
    pub invest: [Chips; 2],
    /// Chance node: reach multiplier for each child.
    pub chance_factor: f32,
    /// Showdown node: strength table index.
    pub table: u32,
    pub regrets: Vec<f32>,
    pub strat_sum: Vec<f32>,
}

impl Node {
    fn new(kind: Kind, invest: [Chips; 2]) -> Node {
        Node {
            kind,
            actions: Vec::new(),
            cards: Vec::new(),
            children: Vec::new(),
            invest,
            chance_factor: 1.0,
            table: 0,
            regrets: Vec::new(),
            strat_sum: Vec::new(),
        }
    }

    pub fn num_actions(&self) -> usize {
        self.actions.len()
    }
}

/// Sizes of a tree (as computed by the counting pass).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreeStats {
    pub player_nodes: u64,
    pub chance_nodes: u64,
    pub terminal_nodes: u64,
    /// Number of `f32` storage cells (regrets + strategy sums).
    pub storage_floats: u64,
}

impl TreeStats {
    fn add_scaled(&mut self, other: &TreeStats, k: u64) {
        self.player_nodes += other.player_nodes * k;
        self.chance_nodes += other.chance_nodes * k;
        self.terminal_nodes += other.terminal_nodes * k;
        self.storage_floats += other.storage_floats * k;
    }

    pub fn total_nodes(&self) -> u64 {
        self.player_nodes + self.chance_nodes + self.terminal_nodes
    }

    /// Approximate memory in bytes.
    pub fn memory_bytes(&self) -> u64 {
        self.storage_floats * 4 + self.total_nodes() * std::mem::size_of::<Node>() as u64
    }
}

pub(crate) struct TreeParams<'a> {
    pub root_street: Street,
    pub menu: &'a PostflopMenu,
    pub future_menu: &'a PostflopMenu,
    pub big_blind: Chips,
    pub chance: ChanceMode,
    /// Number of hands per player (for storage sizing).
    pub num_hands: [usize; 2],
}

impl TreeParams<'_> {
    fn menu_for(&self, street: Street) -> &PostflopMenu {
        if street == self.root_street {
            self.menu
        } else {
            self.future_menu
        }
    }

    fn actions(&self, st: &BettingState) -> Vec<Action> {
        postflop_actions(&st.situation(self.big_blind), self.menu_for(st.street))
    }

    fn sample_size(&self, next: Street) -> Option<usize> {
        match (self.chance, next) {
            (ChanceMode::Exact, _) => None,
            (ChanceMode::Sampled { turn, .. }, Street::Turn) => Some(turn),
            (ChanceMode::Sampled { river, .. }, _) => Some(river),
        }
    }
}

/// Counts nodes and storage without building the tree.
pub(crate) fn count_tree(params: &TreeParams, root: &BettingState, board_len: usize) -> TreeStats {
    count_player(params, root, board_len)
}

fn count_player(params: &TreeParams, st: &BettingState, board_len: usize) -> TreeStats {
    let mut stats = TreeStats { player_nodes: 1, ..Default::default() };
    let actions = params.actions(st);
    stats.storage_floats += 2 * (actions.len() * params.num_hands[st.to_act]) as u64;
    for a in actions {
        let (tr, _) = st.apply(a);
        match tr {
            Transition::Continue(s2) => stats.add_scaled(&count_player(params, &s2, board_len), 1),
            Transition::Fold(_) => stats.terminal_nodes += 1,
            Transition::StreetOver(s2) => stats.add_scaled(&count_after_street(params, &s2, board_len), 1),
        }
    }
    stats
}

fn count_after_street(params: &TreeParams, st: &BettingState, board_len: usize) -> TreeStats {
    let mut stats = TreeStats::default();
    if st.street == Street::River {
        stats.terminal_nodes = 1;
        return stats;
    }
    let remaining = (52 - board_len) as u64;
    stats.chance_nodes = 1;
    if st.stack[0] == 0 || st.stack[1] == 0 {
        // All-in runout: chance nodes down to showdown.
        let mut next = *st;
        next.street = st.street.next().unwrap();
        let child = count_after_street(params, &next, board_len + 1);
        stats.add_scaled(&child, remaining);
        return stats;
    }
    let next_street = st.street.next().unwrap();
    let k = params.sample_size(next_street).map_or(remaining, |k| (k as u64).min(remaining));
    let next = BettingState::street_start(next_street, st.pot, st.stack, params.big_blind);
    let child = count_player(params, &next, board_len + 1);
    stats.add_scaled(&child, k);
    stats
}

/// Builds the full game tree. Returns the root and the list of final boards
/// (dealt cards beyond the root board) that need strength tables.
pub(crate) struct Builder<'a> {
    pub params: &'a TreeParams<'a>,
    pub root_board: Vec<Card>,
    pub boards: Vec<Vec<Card>>,
    board_index: HashMap<u64, u32>,
    rng: StdRng,
}

impl<'a> Builder<'a> {
    pub fn new(params: &'a TreeParams<'a>, root_board: Vec<Card>) -> Builder<'a> {
        let seed = match params.chance {
            ChanceMode::Sampled { seed, .. } => seed,
            ChanceMode::Exact => 0,
        };
        Builder {
            params,
            root_board,
            boards: Vec::new(),
            board_index: HashMap::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn build(&mut self, root: &BettingState) -> Node {
        self.build_player(root, [0, 0], &[])
    }

    fn build_player(&mut self, st: &BettingState, invest: [Chips; 2], dealt: &[Card]) -> Node {
        let mut node = Node::new(Kind::Player(st.to_act as u8), invest);
        let actions = self.params.actions(st);
        let n = self.params.num_hands[st.to_act];
        node.regrets = vec![0.0; actions.len() * n];
        node.strat_sum = vec![0.0; actions.len() * n];
        for &a in &actions {
            let (tr, put) = st.apply(a);
            let mut inv = invest;
            inv[st.to_act] += put;
            let child = match tr {
                Transition::Continue(s2) => self.build_player(&s2, inv, dealt),
                Transition::Fold(f) => Node::new(Kind::Fold(f as u8), inv),
                Transition::StreetOver(s2) => self.build_after_street(&s2, inv, dealt),
            };
            node.children.push(child);
        }
        node.actions = actions;
        node
    }

    fn build_after_street(&mut self, st: &BettingState, invest: [Chips; 2], dealt: &[Card]) -> Node {
        if st.street == Street::River {
            let mut node = Node::new(Kind::Showdown, invest);
            node.table = self.table_for(dealt);
            return node;
        }
        let used: CardSet = CardSet::from_cards(&self.root_board).union(CardSet::from_cards(dealt));
        let available: Vec<Card> = Card::all().filter(|c| !used.contains(*c)).collect();
        let board_len = self.root_board.len() + dealt.len();
        let compatible = (52 - board_len - 4) as f32;
        let mut node = Node::new(Kind::Chance, invest);
        let next_street = st.street.next().unwrap();
        let all_in = st.stack[0] == 0 || st.stack[1] == 0;
        let cards = match (all_in, self.params.sample_size(next_street)) {
            (false, Some(k)) if k < available.len() => sample_cards(&available, k, &mut self.rng),
            _ => available.clone(),
        };
        node.chance_factor = 1.0 / (cards.len() as f32 * compatible / available.len() as f32);
        let mut next_dealt = dealt.to_vec();
        next_dealt.push(Card::from_index(0));
        for &c in &cards {
            *next_dealt.last_mut().unwrap() = c;
            let child = if all_in {
                let mut next = *st;
                next.street = next_street;
                self.build_after_street(&next, invest, &next_dealt)
            } else {
                let next = BettingState::street_start(next_street, st.pot, st.stack, self.params.big_blind);
                self.build_player(&next, invest, &next_dealt)
            };
            node.children.push(child);
        }
        node.cards = cards;
        node
    }

    fn table_for(&mut self, dealt: &[Card]) -> u32 {
        let key = CardSet::from_cards(dealt).0;
        if let Some(&i) = self.board_index.get(&key) {
            return i;
        }
        let i = self.boards.len() as u32;
        let mut board = self.root_board.clone();
        board.extend_from_slice(dealt);
        self.boards.push(board);
        self.board_index.insert(key, i);
        i
    }
}

trait Union {
    fn union(self, other: CardSet) -> CardSet;
}

impl Union for CardSet {
    fn union(self, other: CardSet) -> CardSet {
        CardSet(self.0 | other.0)
    }
}
