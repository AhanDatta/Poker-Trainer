//! The preflop betting tree of a [`GameSpec`], built with the trainer's own
//! game engine, bet menu and spot keys, so every decision of the tree is a
//! spot the trainer can look up by the same key.

use crate::spec::{GameSpec, Limp, SpecError, BB, MAX_PLAYERS};
use holdem::game::Phase;
use holdem::{Action, Card, Chips, HandState, Situation, TableConfig};
use trainer::preflop::{spot_key, Availability, PreAction, PreDecision};

/// "No such action" in [`Decision::children`].
pub const NO_CHILD: u32 = u32::MAX;

/// How the preflop betting ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// Everyone else folded: this seat takes the pot, unraked.
    Won(u8),
    /// At most one player has chips left: the board runs out.
    AllIn,
    /// Two or more players see a flop with chips behind.
    Flop,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Leaf {
    pub kind: LeafKind,
    /// All chips in the pot, folded players' included.
    pub pot: Chips,
    /// Rake on a pot of this size (not taken when everyone folded).
    pub rake: Chips,
    /// Chips each seat put in.
    pub commit: [Chips; MAX_PLAYERS],
    /// Bit `s` is set when seat `s` is still in the hand.
    pub live: u16,
    /// Raises made (the open is 1, a 3-bet 2, ...).
    pub raises: u8,
    /// Seat of the last raiser.
    pub aggressor: Option<u8>,
}

impl Leaf {
    pub fn is_live(&self, seat: usize) -> bool {
        self.live & (1 << seat) != 0
    }

    pub fn num_live(&self) -> usize {
        self.live.count_ones() as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// Seat to act (its offset after the button, so seat 0 is the button).
    pub seat: u8,
    /// The trainer's spot key, e.g. `"UTG:R BTN:C>BB"`.
    pub key: String,
    /// The actions of this decision, after the tree rules.
    pub avail: Availability,
    /// Child node per abstract action ([`PreAction::index`]), or [`NO_CHILD`].
    pub children: [u32; 4],
    /// Chips the seat has put in so far (what folding here costs).
    pub commit: Chips,
    /// Index of this decision among the tree's decisions.
    pub slot: u32,
}

impl Decision {
    /// The abstract actions this decision has, in [`PreAction::ALL`] order.
    pub fn actions(&self) -> impl Iterator<Item = PreAction> + '_ {
        PreAction::ALL.into_iter().filter(|a| self.children[a.index()] != NO_CHILD)
    }

    pub fn num_actions(&self) -> usize {
        self.children.iter().filter(|&&c| c != NO_CHILD).count()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Decision(Decision),
    Leaf(Leaf),
}

/// The whole betting tree; node 0 is the first decision.
#[derive(Clone, Debug)]
pub struct Tree {
    pub spec: GameSpec,
    pub nodes: Vec<Node>,
    pub num_decisions: usize,
}

impl Tree {
    pub fn build(spec: &GameSpec) -> Result<Tree, SpecError> {
        spec.validate()?;
        let n = spec.players;
        let config = TableConfig::new(n, BB / 2, BB).with_rake(spec.rake_rate, spec.rake_cap());
        // The cards don't matter to the betting; any distinct ones do.
        let hole =
            (0..n).map(|p| [Card::from_index(2 * p as u8), Card::from_index(2 * p as u8 + 1)]).collect();
        let state = HandState::new(config, 0, vec![spec.stack(); n], hole);
        let mut b = Builder { spec, nodes: Vec::new(), line: Vec::new(), num_decisions: 0 };
        b.node(&state, 0)?;
        Ok(Tree { spec: spec.clone(), nodes: b.nodes, num_decisions: b.num_decisions })
    }

    pub fn decisions(&self) -> impl Iterator<Item = &Decision> {
        self.nodes.iter().filter_map(|n| match n {
            Node::Decision(d) => Some(d),
            Node::Leaf(_) => None,
        })
    }

    pub fn leaves(&self) -> impl Iterator<Item = &Leaf> {
        self.nodes.iter().filter_map(|n| match n {
            Node::Leaf(l) => Some(l),
            Node::Decision(_) => None,
        })
    }

    pub fn num_leaves(&self) -> usize {
        self.nodes.len() - self.num_decisions
    }

    /// The decision with spot key `key`.
    pub fn find(&self, key: &str) -> Option<&Decision> {
        self.decisions().find(|d| d.key == key)
    }
}

struct Builder<'a> {
    spec: &'a GameSpec,
    nodes: Vec<Node>,
    /// Decisions on the path to the node being built.
    line: Vec<PreDecision>,
    num_decisions: usize,
}

impl Builder<'_> {
    /// Adds the node for `state`, where `callers` players have called the
    /// current raise, and returns its index.
    fn node(&mut self, state: &HandState, callers: usize) -> Result<u32, SpecError> {
        if self.nodes.len() >= self.spec.rules.max_nodes {
            return Err(SpecError::TooLarge { limit: self.spec.rules.max_nodes });
        }
        let id = self.nodes.len() as u32;
        let kind = match state.phase {
            Phase::Betting => return self.decision(id, state, callers),
            Phase::Finished => LeafKind::Won(state.players_in_hand()[0] as u8),
            Phase::AwaitingBoard if state.is_runout() => LeafKind::AllIn,
            Phase::AwaitingBoard => LeafKind::Flop,
        };
        let leaf = self.leaf(kind, state)?;
        self.nodes.push(Node::Leaf(leaf));
        Ok(id)
    }

    fn leaf(&self, kind: LeafKind, state: &HandState) -> Result<Leaf, SpecError> {
        let mut commit = [0; MAX_PLAYERS];
        commit[..state.num_players()].copy_from_slice(&state.total_commit);
        let live = state.players_in_hand();
        if !matches!(kind, LeafKind::Won(_)) && live.iter().any(|&p| commit[p] != commit[live[0]]) {
            let line = self.line.last().map_or(String::new(), |d| format!("{} {:?}", d.key, d.action));
            return Err(SpecError::UnequalCommitments(line));
        }
        let aggressor = state.history.iter().rev().find(|r| matches!(r.action, Action::Raise(_)));
        Ok(Leaf {
            kind,
            pot: state.pot(),
            rake: self.spec.rake(state.pot()),
            commit,
            live: live.iter().fold(0, |m, &p| m | 1 << p),
            raises: state.num_raises as u8,
            aggressor: aggressor.map(|r| r.player as u8),
        })
    }

    fn decision(&mut self, id: u32, state: &HandState, callers: usize) -> Result<u32, SpecError> {
        let p = state.to_act.expect("betting with a player to act");
        let sit = state.situation(p);
        let position = state.position_name(p).to_string();
        let full = Availability::from_actions(&state.legal_actions(&self.spec.menu), &sit);
        let acted = self.line.iter().any(|d| d.seat == p);
        let avail = prune(full, &self.spec.rules, &sit, &position, callers, acted);
        let key = spot_key(&self.line, &position);
        let slot = self.num_decisions as u32;
        self.num_decisions += 1;
        self.nodes.push(Node::Decision(Decision {
            seat: p as u8,
            key: key.clone(),
            avail: avail.clone(),
            children: [NO_CHILD; 4],
            commit: state.total_commit[p],
            slot,
        }));
        for action in PreAction::ALL {
            let Some(concrete) = avail.concrete(action) else { continue };
            let mut next = state.clone();
            next.apply(concrete).expect("menu actions are legal");
            let callers = match action {
                PreAction::Passive if sit.to_call > 0 => callers + 1,
                PreAction::Raise | PreAction::AllIn => 0,
                _ => callers,
            };
            self.line.push(PreDecision {
                seat: p,
                position: position.clone(),
                key: key.clone(),
                avail: avail.clone(),
                action,
            });
            let child = self.node(&next, callers);
            self.line.pop();
            let Node::Decision(d) = &mut self.nodes[id as usize] else {
                unreachable!("a decision was pushed")
            };
            d.children[action.index()] = child?;
        }
        Ok(id)
    }
}

/// Applies the tree rules to the actions of a decision where `callers`
/// players called the current raise and the player to act has `acted`
/// before or not. Folding, and checking in the big blind, are never
/// removed, and a decision always keeps a way to continue.
fn prune(
    mut avail: Availability,
    rules: &crate::spec::TreeRules,
    sit: &Situation,
    position: &str,
    callers: usize,
    acted: bool,
) -> Availability {
    let drop = |avail: &mut Availability, a: PreAction| avail.actions[a.index()] = None;
    if avail.has(PreAction::Raise) && sit.num_raises < rules.allin_after_raises {
        drop(&mut avail, PreAction::AllIn);
    }
    let can_raise = avail.has(PreAction::Raise) || avail.has(PreAction::AllIn);
    let calling = sit.to_call > 0 && avail.has(PreAction::Passive);
    if calling && can_raise {
        let limp = sit.num_raises == 0;
        let allowed = if limp {
            match rules.limp {
                Limp::Never => false,
                Limp::SmallBlind => position == "SB" && sit.limpers == 0,
                Limp::Any => true,
            }
        } else {
            acted || position == "BB" || callers < rules.max_cold_callers
        };
        if !allowed {
            drop(&mut avail, PreAction::Passive);
        }
    }
    avail
}

#[cfg(test)]
mod tests;
