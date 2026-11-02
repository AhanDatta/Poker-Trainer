//! Heads-up postflop betting state and game-tree construction.
//!
//! The tree is stored as a flat arena of compact [`Node`]s; the children of
//! a node occupy a contiguous index range. Regrets and strategy sums live in
//! a separate [`crate::storage::Storage`].

use crate::iso::{IsoRef, Isomorphism};
use crate::sampling::sample_cards;
use crate::storage::{Block, Storage};
use holdem::menu::postflop_actions;
use holdem::{Action, Card, CardSet, Chips, PostflopMenu, Situation, Street};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::ops::Range;

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

/// How a node was reached from its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edge {
    Root,
    Action(Action),
    Card(Card),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub kind: Kind,
    pub street: Street,
    pub num_children: u16,
    pub first_child: u32,
    /// Chips each player has invested since the root.
    pub invest: [i32; 2],
    /// Player node: storage offset. Showdown: strength-table index.
    pub offset: u64,
    /// Player node: storage slot (scales). Chance node: `chance` index.
    pub slot: u32,
    /// Terminal nodes: rake taken from the pot.
    pub rake: f32,
    /// Chance node: reach multiplier for each child.
    pub chance_factor: f32,
    pub edge: Edge,
}

impl Node {
    fn placeholder() -> Node {
        Node {
            kind: Kind::Showdown,
            street: Street::River,
            num_children: 0,
            first_child: 0,
            invest: [0, 0],
            offset: 0,
            slot: 0,
            rake: 0.0,
            chance_factor: 1.0,
            edge: Edge::Root,
        }
    }

    #[inline]
    pub fn children(&self) -> Range<usize> {
        let f = self.first_child as usize;
        f..f + self.num_children as usize
    }

    #[inline]
    pub fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    /// Storage block of a player node with `num_hands` hands.
    #[inline]
    pub fn block(&self, num_hands: usize) -> Block {
        Block { offset: self.offset as usize, len: self.num_actions() * num_hands, slot: self.slot as usize }
    }
}

/// Isomorphic outcomes of a chance node.
#[derive(Clone, Debug, Default)]
pub(crate) struct ChanceInfo {
    /// `(index of the canonical child, suit pair)` for each isomorphic card.
    pub iso: Vec<(u16, u8)>,
    /// The isomorphic cards themselves (parallel to `iso`).
    pub iso_cards: Vec<Card>,
}

pub(crate) struct Tree {
    pub nodes: Vec<Node>,
    pub chance: Vec<ChanceInfo>,
    /// Full board of every showdown strength table.
    pub boards: Vec<Vec<Card>>,
    pub storage_len: u64,
    pub num_player_nodes: u32,
}

impl Tree {
    /// Actions of a player node.
    pub fn actions(&self, id: usize) -> Vec<Action> {
        self.nodes[id]
            .children()
            .map(|c| match self.nodes[c].edge {
                Edge::Action(a) => a,
                _ => unreachable!("player-node child reached by a card"),
            })
            .collect()
    }

    /// Card of a chance child.
    #[inline]
    pub fn card(&self, id: usize) -> Card {
        match self.nodes[id].edge {
            Edge::Card(c) => c,
            _ => unreachable!("chance child reached by an action"),
        }
    }
}

/// Sizes of a tree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreeStats {
    pub player_nodes: u64,
    pub chance_nodes: u64,
    pub terminal_nodes: u64,
    /// Number of `(action, hand)` storage cells.
    pub storage_cells: u64,
    /// Whether storage would be compressed.
    pub compressed: bool,
}

impl TreeStats {
    pub fn total_nodes(&self) -> u64 {
        self.player_nodes + self.chance_nodes + self.terminal_nodes
    }

    /// Approximate memory use in bytes.
    pub fn memory_bytes(&self) -> u64 {
        self.storage_cells * Storage::bytes_per_value(self.compressed)
            + self.total_nodes() * std::mem::size_of::<Node>() as u64
    }

    /// Memory if storage is uncompressed `f32`.
    pub fn memory_bytes_f32(&self) -> u64 {
        TreeStats { compressed: false, ..*self }.memory_bytes()
    }
}

pub(crate) struct TreeParams<'a> {
    pub root_street: Street,
    pub root_board: &'a [Card],
    pub menu: &'a PostflopMenu,
    pub future_menu: &'a PostflopMenu,
    pub big_blind: Chips,
    pub chance: ChanceMode,
    pub num_hands: [usize; 2],
    pub rake_rate: f64,
    pub rake_cap: f64,
    pub iso: &'a Isomorphism,
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

    fn rake(&self, pot: Chips) -> f32 {
        (pot as f64 * self.rake_rate).min(self.rake_cap) as f32
    }

    /// Cards dealt at a chance node: canonical cards and isomorphic refs.
    fn chance_cards(
        &self,
        dealt: &[Card],
        all_in: bool,
        next: Street,
        rng: &mut StdRng,
    ) -> (Vec<Card>, Vec<IsoRef>, usize) {
        let mut board = self.root_board.to_vec();
        board.extend_from_slice(dealt);
        let used = CardSet::from_cards(&board);
        let available: Vec<Card> = Card::all().filter(|c| !used.contains(*c)).collect();
        let n_avail = available.len();
        match (all_in, self.sample_size(next)) {
            (false, Some(k)) if k < n_avail => (sample_cards(&available, k, rng), Vec::new(), n_avail),
            _ => {
                let (canon, refs) = self.iso.chance_cards(&board, &available);
                (canon, refs, n_avail)
            }
        }
    }
}

/// Counts nodes and storage without building the tree.
pub(crate) fn count_tree(params: &TreeParams, root: &BettingState) -> TreeStats {
    let mut rng = StdRng::seed_from_u64(0);
    let mut stats = TreeStats::default();
    count_player(params, root, &mut Vec::new(), &mut rng, &mut stats);
    stats
}

fn count_player(
    params: &TreeParams,
    st: &BettingState,
    dealt: &mut Vec<Card>,
    rng: &mut StdRng,
    stats: &mut TreeStats,
) {
    stats.player_nodes += 1;
    let actions = params.actions(st);
    stats.storage_cells += (actions.len() * params.num_hands[st.to_act]) as u64;
    for a in actions {
        match st.apply(a).0 {
            Transition::Continue(s2) => count_player(params, &s2, dealt, rng, stats),
            Transition::Fold(_) => stats.terminal_nodes += 1,
            Transition::StreetOver(s2) => count_after_street(params, &s2, dealt, rng, stats),
        }
    }
}

fn count_after_street(
    params: &TreeParams,
    st: &BettingState,
    dealt: &mut Vec<Card>,
    rng: &mut StdRng,
    stats: &mut TreeStats,
) {
    if st.street == Street::River {
        stats.terminal_nodes += 1;
        return;
    }
    stats.chance_nodes += 1;
    let all_in = st.stack[0] == 0 || st.stack[1] == 0;
    let next = st.street.next().unwrap();
    let (cards, refs, _) = params.chance_cards(dealt, all_in, next, rng);
    // Subtrees only differ (in size) through isomorphism further down, so
    // without isomorphism one child can stand for all.
    let uniform = refs.is_empty() && !params.iso.enabled;
    let reps: &[Card] = if uniform { &cards[..1] } else { &cards };
    let mut sub = TreeStats::default();
    for &c in reps {
        dealt.push(c);
        if all_in {
            let mut s2 = *st;
            s2.street = next;
            count_after_street(params, &s2, dealt, rng, &mut sub);
        } else {
            let s2 = BettingState::street_start(next, st.pot, st.stack, params.big_blind);
            count_player(params, &s2, dealt, rng, &mut sub);
        }
        dealt.pop();
    }
    let k = if uniform { cards.len() as u64 } else { 1 };
    stats.player_nodes += sub.player_nodes * k;
    stats.chance_nodes += sub.chance_nodes * k;
    stats.terminal_nodes += sub.terminal_nodes * k;
    stats.storage_cells += sub.storage_cells * k;
}

/// Builds the tree into an arena.
pub(crate) struct Builder<'a> {
    params: &'a TreeParams<'a>,
    nodes: Vec<Node>,
    chance: Vec<ChanceInfo>,
    boards: Vec<Vec<Card>>,
    board_index: HashMap<u64, u32>,
    storage_len: u64,
    player_slots: u32,
    rng: StdRng,
}

impl<'a> Builder<'a> {
    pub fn new(params: &'a TreeParams<'a>) -> Builder<'a> {
        let seed = match params.chance {
            ChanceMode::Sampled { seed, .. } => seed,
            ChanceMode::Exact => 0,
        };
        Builder {
            params,
            nodes: Vec::new(),
            chance: Vec::new(),
            boards: Vec::new(),
            board_index: HashMap::new(),
            storage_len: 0,
            player_slots: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn build(mut self, root: &BettingState) -> Tree {
        self.nodes.push(Node::placeholder());
        self.fill_player(0, root, [0, 0], &mut Vec::new());
        Tree {
            nodes: self.nodes,
            chance: self.chance,
            boards: self.boards,
            storage_len: self.storage_len,
            num_player_nodes: self.player_slots,
        }
    }

    fn alloc_children(&mut self, k: usize) -> usize {
        let first = self.nodes.len();
        self.nodes.resize(first + k, Node::placeholder());
        first
    }

    fn fill_player(&mut self, id: usize, st: &BettingState, invest: [i32; 2], dealt: &mut Vec<Card>) {
        let actions = self.params.actions(st);
        let k = actions.len();
        let first = self.alloc_children(k);
        let edge = self.nodes[id].edge;
        self.nodes[id] = Node {
            kind: Kind::Player(st.to_act as u8),
            street: st.street,
            num_children: k as u16,
            first_child: first as u32,
            invest,
            offset: self.storage_len,
            slot: self.player_slots,
            rake: 0.0,
            chance_factor: 1.0,
            edge,
        };
        self.storage_len += (k * self.params.num_hands[st.to_act]) as u64;
        self.player_slots += 1;
        for (i, &a) in actions.iter().enumerate() {
            let child = first + i;
            let (tr, put) = st.apply(a);
            let mut inv = invest;
            inv[st.to_act] += put as i32;
            match tr {
                Transition::Continue(s2) => {
                    self.nodes[child].edge = Edge::Action(a);
                    self.fill_player(child, &s2, inv, dealt);
                }
                Transition::Fold(f) => {
                    // Uncalled chips are returned and not raked.
                    let o = f ^ 1;
                    let uncalled = (st.commit[o] - st.commit[f]).max(0);
                    self.nodes[child] = Node {
                        kind: Kind::Fold(f as u8),
                        street: st.street,
                        invest: inv,
                        rake: self.params.rake(st.pot - uncalled),
                        edge: Edge::Action(a),
                        ..Node::placeholder()
                    };
                }
                Transition::StreetOver(s2) => {
                    self.nodes[child].edge = Edge::Action(a);
                    self.fill_after_street(child, &s2, inv, dealt);
                }
            }
        }
    }

    fn fill_after_street(&mut self, id: usize, st: &BettingState, invest: [i32; 2], dealt: &mut Vec<Card>) {
        let edge = self.nodes[id].edge;
        if st.street == Street::River {
            let table = self.table_for(dealt);
            self.nodes[id] = Node {
                kind: Kind::Showdown,
                street: Street::River,
                invest,
                offset: table as u64,
                rake: self.params.rake(st.pot),
                edge,
                ..Node::placeholder()
            };
            return;
        }
        let all_in = st.stack[0] == 0 || st.stack[1] == 0;
        let next = st.street.next().unwrap();
        let (cards, refs, n_avail) = self.params.chance_cards(dealt, all_in, next, &mut self.rng);
        let board_len = self.params.root_board.len() + dealt.len();
        let compatible = (52 - board_len - 4) as f32;
        let total_cards = cards.len() + refs.len();
        let first = self.alloc_children(cards.len());
        let slot = self.chance.len() as u32;
        let info = ChanceInfo {
            iso: refs
                .iter()
                .map(|r| {
                    let k = cards.iter().position(|&c| c == r.canonical).expect("canonical card present");
                    (k as u16, r.pair)
                })
                .collect(),
            iso_cards: refs.iter().map(|r| r.card).collect(),
        };
        self.chance.push(info);
        self.nodes[id] = Node {
            kind: Kind::Chance,
            street: st.street,
            num_children: cards.len() as u16,
            first_child: first as u32,
            invest,
            slot,
            chance_factor: 1.0 / (total_cards as f32 * compatible / n_avail as f32),
            edge,
            ..Node::placeholder()
        };
        for (i, &c) in cards.iter().enumerate() {
            let child = first + i;
            self.nodes[child].edge = Edge::Card(c);
            dealt.push(c);
            if all_in {
                let mut s2 = *st;
                s2.street = next;
                self.fill_after_street(child, &s2, invest, dealt);
            } else {
                let s2 = BettingState::street_start(next, st.pot, st.stack, self.params.big_blind);
                self.fill_player(child, &s2, invest, dealt);
            }
            dealt.pop();
        }
    }

    fn table_for(&mut self, dealt: &[Card]) -> u32 {
        let key = CardSet::from_cards(dealt).0;
        if let Some(&i) = self.board_index.get(&key) {
            return i;
        }
        let i = self.boards.len() as u32;
        let mut board = self.params.root_board.to_vec();
        board.extend_from_slice(dealt);
        self.boards.push(board);
        self.board_index.insert(key, i);
        i
    }
}
