//! The three-player game tree as a flat arena (children of a node are
//! contiguous), built in the same depth-first order as before so sampled
//! runouts are reproducible from the seed.

use super::hands::Hands;
use super::{MultiConfig, MultiState, Next, N};
use crate::sampling::sample_cards;
use crate::storage::Block;
use crate::{ChanceMode, SolverError};
use holdem::menu::postflop_actions;
use holdem::{Action, Card, CardSet, Chips, PostflopMenu, Street};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Player(u8),
    Chance,
    /// Everyone else folded: this player takes the pot.
    Won(u8),
    Showdown,
}

/// How a node was reached from its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Edge {
    Root,
    Action(Action),
    Card(Card),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Node {
    pub kind: Kind,
    pub street: Street,
    pub folded: [bool; N],
    pub num_children: u16,
    pub first_child: u32,
    /// Chips each player has invested since the root.
    pub invest: [i32; N],
    /// Player node: storage offset.
    pub offset: u64,
    /// Player node: storage slot. Showdown: strength-table index.
    pub slot: u32,
    /// Terminal: rake taken from the pot.
    pub rake: f32,
    /// Chance node: reach multiplier for each child.
    pub factor: f32,
    pub edge: Edge,
}

impl Node {
    fn new(kind: Kind, street: Street, invest: [i32; N], folded: [bool; N], edge: Edge) -> Node {
        Node {
            kind,
            street,
            folded,
            num_children: 0,
            first_child: 0,
            invest,
            offset: 0,
            slot: 0,
            rake: 0.0,
            factor: 1.0,
            edge,
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

    /// Storage block of a player node whose player has `num_hands` hands.
    #[inline]
    pub fn block(&self, num_hands: usize) -> Block {
        Block { offset: self.offset as usize, len: self.num_actions() * num_hands, slot: self.slot as usize }
    }
}

pub(super) struct Tree {
    pub nodes: Vec<Node>,
    /// Full board of every showdown strength table.
    pub boards: Vec<Vec<Card>>,
    /// Number of `(action, hand)` storage cells.
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
    pub fn card(&self, id: usize) -> Card {
        match self.nodes[id].edge {
            Edge::Card(c) => c,
            _ => unreachable!("chance child reached by an action"),
        }
    }

    pub fn build(cfg: &MultiConfig, hands: &Hands, max_cells: u64) -> Result<Tree, SolverError> {
        let seed = match cfg.chance {
            ChanceMode::Sampled { seed, .. } => seed,
            ChanceMode::Exact => 0,
        };
        let mut b = Builder {
            cfg,
            hands,
            nodes: vec![Node::new(Kind::Showdown, cfg.root.street, [0; N], cfg.root.folded, Edge::Root)],
            boards: Vec::new(),
            board_index: Default::default(),
            cells: 0,
            max_cells,
            player_nodes: 0,
            rng: StdRng::seed_from_u64(seed),
        };
        b.player(0, &cfg.root, [0; N], &mut Vec::new())?;
        Ok(Tree { nodes: b.nodes, boards: b.boards, storage_len: b.cells, num_player_nodes: b.player_nodes })
    }
}

struct Builder<'a> {
    cfg: &'a MultiConfig,
    hands: &'a Hands,
    nodes: Vec<Node>,
    boards: Vec<Vec<Card>>,
    board_index: std::collections::HashMap<u64, u32>,
    cells: u64,
    max_cells: u64,
    player_nodes: u32,
    rng: StdRng,
}

impl Builder<'_> {
    fn menu(&self, street: Street) -> &PostflopMenu {
        if street == self.cfg.root.street {
            &self.cfg.menu
        } else {
            self.cfg.future_menu.as_ref().unwrap_or(&self.cfg.menu)
        }
    }

    fn rake(&self, pot: Chips) -> f32 {
        (pot as f64 * self.cfg.rake_rate).min(self.cfg.rake_cap) as f32
    }

    /// Reserves `k` contiguous child slots of node `id`.
    fn reserve(&mut self, id: usize, k: usize) -> usize {
        let first = self.nodes.len();
        let placeholder = Node::new(Kind::Showdown, Street::River, [0; N], [false; N], Edge::Root);
        self.nodes.resize(first + k, placeholder);
        self.nodes[id].first_child = first as u32;
        self.nodes[id].num_children = k as u16;
        first
    }

    /// Fills node `id` (already allocated) as a decision of `st.to_act`.
    fn player(
        &mut self,
        id: usize,
        st: &MultiState,
        invest: [i32; N],
        dealt: &mut Vec<Card>,
    ) -> Result<(), SolverError> {
        let actions = postflop_actions(&st.situation(self.cfg.big_blind), self.menu(st.street));
        let p = st.to_act;
        let len = (actions.len() * self.hands.len(p)) as u64;
        {
            let node = &mut self.nodes[id];
            node.kind = Kind::Player(p as u8);
            node.street = st.street;
            node.invest = invest;
            node.folded = st.folded;
            node.offset = self.cells;
            node.slot = self.player_nodes;
        }
        self.cells += len;
        self.player_nodes += 1;
        if self.cells > self.max_cells {
            return Err(SolverError::TooLarge { needed: self.cells, limit: self.max_cells });
        }
        let first = self.reserve(id, actions.len());
        for (k, &a) in actions.iter().enumerate() {
            let c = first + k;
            let (next, put) = st.apply(a);
            let mut inv = invest;
            inv[p] += put as i32;
            self.nodes[c].edge = Edge::Action(a);
            match next {
                Next::Continue(s2) => self.player(c, &s2, inv, dealt)?,
                Next::Won(w) => {
                    let mut s2 = *st;
                    s2.folded[p] = true;
                    s2.commit[p] += put;
                    // The uncalled part of the winner's bet isn't raked.
                    let others = (0..N).filter(|&j| j != w).map(|j| s2.commit[j]).max().unwrap_or(0);
                    let uncalled = (s2.commit[w] - others).max(0);
                    let rake = self.rake(st.pot + put - uncalled);
                    let edge = self.nodes[c].edge;
                    self.nodes[c] = Node::new(Kind::Won(w as u8), st.street, inv, s2.folded, edge);
                    self.nodes[c].rake = rake;
                }
                Next::StreetOver(s2) => self.after_street(c, &s2, inv, dealt)?,
            }
        }
        Ok(())
    }

    fn after_street(
        &mut self,
        id: usize,
        st: &MultiState,
        invest: [i32; N],
        dealt: &mut Vec<Card>,
    ) -> Result<(), SolverError> {
        let edge = self.nodes[id].edge;
        if st.street == Street::River {
            let table = self.table_for(dealt);
            self.nodes[id] = Node::new(Kind::Showdown, st.street, invest, st.folded, edge);
            self.nodes[id].rake = self.rake(st.pot);
            self.nodes[id].slot = table;
            return Ok(());
        }
        let can_bet = (0..N).filter(|&q| !st.folded[q] && st.stack[q] > 0).count();
        let runout = can_bet < 2;
        let next = st.street.next().unwrap();
        let mut board = self.cfg.board.clone();
        board.extend_from_slice(dealt);
        let used = CardSet::from_cards(&board);
        let available: Vec<Card> = Card::all().filter(|c| !used.contains(*c)).collect();
        // All-in runouts are sampled like the other chance nodes: with three
        // players they hold most of a sampled flop tree's showdowns if every
        // card is dealt, and each costs a full sweep on its own board.
        let k = match (self.cfg.chance, next) {
            (ChanceMode::Sampled { turn, .. }, Street::Turn) => turn.max(1),
            (ChanceMode::Sampled { river, .. }, _) => river.max(1),
            _ => available.len(),
        };
        let cards =
            if k < available.len() { sample_cards(&available, k, &mut self.rng) } else { available.clone() };
        let compatible = (52 - board.len() - 2 * N) as f32;
        let factor = 1.0 / (cards.len() as f32 * compatible / available.len() as f32);
        self.nodes[id] = Node::new(Kind::Chance, st.street, invest, st.folded, edge);
        self.nodes[id].factor = factor;
        let first = self.reserve(id, cards.len());
        for (i, &c) in cards.iter().enumerate() {
            let child = first + i;
            self.nodes[child].edge = Edge::Card(c);
            dealt.push(c);
            if runout {
                let mut s2 = *st;
                s2.street = next;
                self.after_street(child, &s2, invest, dealt)?;
            } else {
                let s2 = MultiState::street_start(next, st.pot, st.stack, st.folded, self.cfg.big_blind);
                self.player(child, &s2, invest, dealt)?;
            }
            dealt.pop();
        }
        Ok(())
    }

    fn table_for(&mut self, dealt: &[Card]) -> u32 {
        let key = CardSet::from_cards(dealt).0;
        if let Some(&i) = self.board_index.get(&key) {
            return i;
        }
        let i = self.boards.len() as u32;
        let mut board = self.cfg.board.clone();
        board.extend_from_slice(dealt);
        self.boards.push(board);
        self.board_index.insert(key, i);
        i
    }
}
