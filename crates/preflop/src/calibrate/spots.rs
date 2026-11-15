//! The heads-up pots that see a flop when everyone plays by a chart set,
//! with how often each happens and the two ranges that get there.

use crate::realize::{Context, Position, PotKind, Role};
use crate::tree::{Leaf, LeafKind, Node, Tree, NO_CHILD};
use holdem::{Chips, HandClass, NUM_CLASSES};
use trainer::preflop::{ChartSet, PreAction};

/// A heads-up pot that sees a flop, as the charts play to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Spot {
    /// The line in the charts' tokens, e.g. `"CO:R BB:C"` (folds of players
    /// who never entered are left out, as in spot keys).
    pub line: String,
    /// The leaf's node in the tree.
    pub leaf: u32,
    /// The decisions from the root: `(node, action taken)`.
    pub path: Vec<(u32, PreAction)>,
    /// Probability that the line happens with random hands.
    pub reach: f64,
    /// Seats of the two players: `[out of position, in position]` after the
    /// flop, which is also the postflop solver's player order.
    pub seats: [usize; 2],
    pub contexts: [Context; 2],
    /// Per-combo weight of every hand class in each player's range (the
    /// product of the chart frequencies along the line), in seat order.
    pub classes: [[f32; NUM_CLASSES]; 2],
    /// Chips in the pot, folded players' included.
    pub pot: Chips,
    /// Chips each player has behind.
    pub behind: Chips,
}

/// What a spot measures: the pot type and where the last raiser sits. Each
/// type gives two contexts, the raiser's and the caller's; a limped pot has
/// no raiser (`aggressor_in_position` is false) and gives the two callers'.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpotType {
    pub pot: PotKind,
    pub aggressor_in_position: bool,
}

impl SpotType {
    /// The six types of heads-up raised pots, then the limped pot (which
    /// only the heads-up game, with open limps, has).
    pub fn all() -> [SpotType; 7] {
        let t = |pot, aggressor_in_position| SpotType { pot, aggressor_in_position };
        [
            t(PotKind::Raised, true),
            t(PotKind::Raised, false),
            t(PotKind::ThreeBet, true),
            t(PotKind::ThreeBet, false),
            t(PotKind::FourBet, true),
            t(PotKind::FourBet, false),
            t(PotKind::Limped, false),
        ]
    }

    pub fn name(self) -> String {
        let pot = match self.pot {
            PotKind::Limped => "limped",
            PotKind::Raised => "single raised",
            PotKind::ThreeBet => "3-bet",
            PotKind::FourBet => "4-bet",
        };
        if self.pot == PotKind::Limped {
            return pot.to_string();
        }
        format!("{pot}, raiser {}", if self.aggressor_in_position { "IP" } else { "OOP" })
    }
}

impl Spot {
    /// The spot's type (`None` only if the contexts are inconsistent).
    pub fn kind(&self) -> Option<SpotType> {
        let Some(raiser) = self.contexts.iter().find(|c| c.role == Role::Aggressor) else {
            let limped = self.contexts.iter().all(|c| c.pot == PotKind::Limped);
            return limped.then_some(SpotType { pot: PotKind::Limped, aggressor_in_position: false });
        };
        Some(SpotType { pot: raiser.pot, aggressor_in_position: raiser.position == Position::InPosition })
    }
}

/// A leaf the charts reach.
pub struct LeafVisit<'a> {
    pub node: u32,
    pub leaf: &'a Leaf,
    pub path: &'a [(u32, PreAction)],
    /// Probability of the line with random hands.
    pub reach: f64,
    /// Per-combo class weights of every seat's range along the line.
    pub ranges: &'a [[f32; NUM_CLASSES]],
}

/// Share of all combos that each class has (its prior probability).
pub fn class_prior() -> [f64; NUM_CLASSES] {
    let mut prior = [0f64; NUM_CLASSES];
    for class in HandClass::all() {
        prior[class.grid_index()] = class.num_combos() as f64 / holdem::NUM_COMBOS as f64;
    }
    prior
}

/// Calls `visit` for every leaf of `tree` that the charts reach, with every
/// player playing by `charts` (the trainer's fallback rules where a spot has
/// no chart, as [`trainer::preflop::chart_policy`] does). The reaches of all
/// visited leaves add up to 1; branches the charts never take are skipped.
pub fn walk_leaves(tree: &Tree, charts: &ChartSet, mut visit: impl FnMut(&LeafVisit)) {
    let prior = class_prior();
    let mut ranges = vec![[1f32; NUM_CLASSES]; tree.spec.players];
    let mut path = Vec::new();
    descend(tree, charts, &prior, 0, &mut ranges, &mut path, &mut visit);
}

fn descend(
    tree: &Tree,
    charts: &ChartSet,
    prior: &[f64; NUM_CLASSES],
    node: u32,
    ranges: &mut [[f32; NUM_CLASSES]],
    path: &mut Vec<(u32, PreAction)>,
    visit: &mut impl FnMut(&LeafVisit),
) {
    let d = match &tree.nodes[node as usize] {
        Node::Leaf(leaf) => {
            let reach = ranges
                .iter()
                .map(|r| r.iter().zip(prior).map(|(&w, &p)| w as f64 * p).sum::<f64>())
                .product();
            visit(&LeafVisit { node, leaf, path, reach, ranges });
            return;
        }
        Node::Decision(d) => d,
    };
    let seat = d.seat as usize;
    let before = ranges[seat];
    let freqs: Vec<[f32; 4]> = (0..NUM_CLASSES).map(|c| charts.freqs(&d.key, c, &d.avail).0).collect();
    for action in PreAction::ALL {
        let child = d.children[action.index()];
        if child == NO_CHILD {
            continue;
        }
        let mut any = false;
        for c in 0..NUM_CLASSES {
            ranges[seat][c] = before[c] * freqs[c][action.index()];
            any |= ranges[seat][c] > 0.0;
        }
        if any {
            path.push((node, action));
            descend(tree, charts, prior, child, ranges, path, visit);
            path.pop();
        }
    }
    ranges[seat] = before;
}

/// The line of a leaf in chart tokens: the last decision's key with its
/// action, e.g. key `"CO:R>BB"` and a call give `"CO:R BB:C"`.
fn line_of(tree: &Tree, path: &[(u32, PreAction)]) -> String {
    let Some(&(node, action)) = path.last() else { return String::new() };
    let Node::Decision(d) = &tree.nodes[node as usize] else { unreachable!("paths hold decisions") };
    let (before, position) = d.key.split_once('>').expect("spot keys have a decider");
    let last = format!("{position}:{}", action.code());
    if before.is_empty() {
        last
    } else {
        format!("{before} {last}")
    }
}

/// Every heads-up pot that sees a flop under `charts`, most reached first.
pub fn heads_up_flops(tree: &Tree, charts: &ChartSet) -> Vec<Spot> {
    let players = tree.spec.players;
    let stack = tree.spec.stack();
    let mut out = Vec::new();
    walk_leaves(tree, charts, |v| {
        if v.leaf.kind != LeafKind::Flop || v.leaf.num_live() != 2 || v.reach <= 0.0 {
            return;
        }
        let mut seats: Vec<usize> = (0..players).filter(|&s| v.leaf.is_live(s)).collect();
        // After the flop seat 1 (the small blind) acts first, the button last.
        seats.sort_by_key(|&s| (s + players - 1) % players);
        let seats = [seats[0], seats[1]];
        out.push(Spot {
            line: line_of(tree, v.path),
            leaf: v.node,
            path: v.path.to_vec(),
            reach: v.reach,
            seats,
            contexts: seats.map(|s| Context::of(v.leaf, s, players)),
            classes: seats.map(|s| v.ranges[s]),
            pot: v.leaf.pot,
            behind: stack - v.leaf.commit[seats[0]],
        });
    });
    out.sort_by(|a, b| b.reach.total_cmp(&a.reach).then_with(|| a.line.cmp(&b.line)));
    out
}

/// The `per_type` most reached spots of every [`SpotType`], in the order of
/// [`SpotType::all`] and then by reach. `spots` must be sorted by reach, as
/// [`heads_up_flops`] returns them.
pub fn pick(spots: &[Spot], per_type: usize) -> Vec<Spot> {
    SpotType::all()
        .into_iter()
        .flat_map(|t| spots.iter().filter(move |s| s.kind() == Some(t)).take(per_type))
        .cloned()
        .collect()
}
