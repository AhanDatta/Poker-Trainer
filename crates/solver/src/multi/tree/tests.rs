//! `MultiConfig::share_runouts`: every chance node deals from one shared
//! sample. Valid cards and chance factors, one deck per tree, the same
//! betting tree as without sharing, and trees without it unchanged.

use super::*;
use crate::multi::tests::config;
use crate::multi::{MultiSolver, MultiState};
use crate::SolveOptions;
use proptest::prelude::*;
use std::collections::HashMap;

const RANGES: [&str; N] = [
    "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
    "AA, KK, AJ, QJ, JT, 99, 77, KQs",
    "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
];

/// A chance node: the cards dealt before it, the cards it deals, its factor.
struct Deal {
    dealt: Vec<Card>,
    cards: Vec<Card>,
    factor: f32,
}

/// Every chance node of `t` in tree order, and every showdown's dealt cards
/// with its strength-table index.
fn deals(t: &Tree) -> (Vec<Deal>, Vec<(Vec<Card>, u32)>) {
    fn walk(t: &Tree, id: usize, dealt: &mut Vec<Card>, out: &mut (Vec<Deal>, Vec<(Vec<Card>, u32)>)) {
        let node = &t.nodes[id];
        match node.kind {
            Kind::Chance => {
                let cards = node.children().map(|c| t.card(c)).collect();
                out.0.push(Deal { dealt: dealt.clone(), cards, factor: node.factor });
                for c in node.children() {
                    dealt.push(t.card(c));
                    walk(t, c, dealt, out);
                    dealt.pop();
                }
            }
            Kind::Showdown => out.1.push((dealt.clone(), node.slot)),
            _ => {
                for c in node.children() {
                    walk(t, c, dealt, out);
                }
            }
        }
    }
    let mut out = (Vec::new(), Vec::new());
    walk(t, 0, &mut Vec::new(), &mut out);
    out
}

fn solver(cfg: &MultiConfig) -> MultiSolver {
    MultiSolver::new(cfg.clone()).unwrap()
}

fn shared(cfg: &MultiConfig) -> MultiConfig {
    MultiConfig { share_runouts: true, ..cfg.clone() }
}

/// The checks every tree passes, shared or not: each chance node deals
/// distinct live cards, as many as the mode samples, with the factor that
/// makes its children's weights sum to one over the compatible hands; each
/// showdown's table is its board.
fn check_cards(cfg: &MultiConfig, t: &Tree, what: &str) {
    let (deals, showdowns) = deals(t);
    for d in &deals {
        let used = CardSet(CardSet::from_cards(&cfg.board).0 | CardSet::from_cards(&d.dealt).0);
        assert_eq!(used.len(), cfg.board.len() + d.dealt.len(), "{what}: a card dealt twice before {:?}", d.dealt);
        assert!(d.cards.windows(2).all(|w| w[0] < w[1]), "{what}: duplicate or unsorted cards {:?}", d.cards);
        assert!(d.cards.iter().all(|&c| !used.contains(c)), "{what}: dead card in {:?} after {:?}", d.cards, d.dealt);
        let available = 52 - used.len();
        let street = cfg.root.street as usize + d.dealt.len() + 1;
        let k = match cfg.chance {
            ChanceMode::Sampled { turn, .. } if street == Street::Turn as usize => turn.max(1),
            ChanceMode::Sampled { river, .. } => river.max(1),
            ChanceMode::Exact => available,
        };
        assert_eq!(d.cards.len(), k.min(available), "{what}: card count after {:?}", d.dealt);
        let compatible = (available - 2 * N) as f64;
        let total = d.cards.len() as f64 * f64::from(d.factor) * compatible / available as f64;
        assert!((total - 1.0).abs() < 1e-6, "{what}: chance weights sum to {total} after {:?}", d.dealt);
    }
    for (dealt, slot) in showdowns {
        let mut board = cfg.board.clone();
        board.extend_from_slice(&dealt);
        assert_eq!(board.len(), 5, "{what}");
        let table = &t.boards[slot as usize];
        assert_eq!(CardSet::from_cards(table), CardSet::from_cards(&board), "{what}: showdown table");
        assert_eq!(CardSet::from_cards(table).len(), 5, "{what}: board with a duplicate card");
    }
}

/// A shared tree: every chance node after the same cards deals the same
/// cards, so the tree has at most turns × rivers boards.
fn check_shared(cfg: &MultiConfig, t: &Tree, what: &str) {
    check_cards(cfg, t, what);
    let mut deck: HashMap<u64, &Vec<Card>> = HashMap::new();
    let (deals, _) = deals(t);
    for d in &deals {
        let key = CardSet::from_cards(&d.dealt).0;
        let first = *deck.entry(key).or_insert(&d.cards);
        assert_eq!(first, &d.cards, "{what}: two chance nodes after {:?} deal different cards", d.dealt);
    }
    if let ChanceMode::Sampled { turn, river, .. } = cfg.chance {
        let bound = match cfg.root.street {
            Street::Flop => turn.max(1) * river.max(1),
            Street::Turn => river.max(1),
            _ => 1,
        };
        assert!(t.boards.len() <= bound, "{what}: {} boards, more than {bound}", t.boards.len());
    }
}

/// The same betting tree: the same nodes in the same order, only the cards
/// (and so the strength tables) differ.
fn check_same_shape(a: &Tree, b: &Tree, what: &str) {
    assert_eq!(a.nodes.len(), b.nodes.len(), "{what}: node count");
    assert_eq!(a.storage_len, b.storage_len, "{what}: cells");
    for (i, (x, y)) in a.nodes.iter().zip(&b.nodes).enumerate() {
        // Player nodes keep their storage slot; showdowns' tables differ.
        let player = matches!(x.kind, Kind::Player(_));
        let edges = match (x.edge, y.edge) {
            (Edge::Card(_), Edge::Card(_)) => true,
            (a, b) => a == b,
        };
        let same = x.kind == y.kind
            && x.street == y.street
            && x.folded == y.folded
            && x.children() == y.children()
            && x.invest == y.invest
            && x.offset == y.offset
            && x.rake.to_bits() == y.rake.to_bits()
            && x.factor.to_bits() == y.factor.to_bits()
            && (!player || x.slot == y.slot)
            && edges;
        assert!(same, "{what}: node {i} differs: {x:?} vs {y:?}");
    }
}

/// A sampled flop with short stacks (all-in runouts on the flop and the
/// turn), a sampled turn, an all-in turn, and a turn with a player folded.
fn spots() -> Vec<(&'static str, MultiConfig)> {
    let mut flop = config("Ks7d2c", RANGES, [false; N]);
    flop.root = MultiState::street_start(Street::Flop, 900, [700, 2500, 3000], [false; N], 100);
    flop.chance = ChanceMode::Sampled { turn: 8, river: 4, seed: 1 };
    let mut turn = config("Jh8d4c2s", RANGES, [false; N]);
    turn.chance = ChanceMode::Sampled { turn: 0, river: 12, seed: 1 };
    let mut allin = turn.clone();
    allin.root = MultiState::street_start(Street::Turn, 900, [400, 400, 3000], [false; N], 100);
    allin.chance = ChanceMode::Sampled { turn: 0, river: 5, seed: 3 };
    let mut folded = config("Jh8d4c2s", RANGES, [true, false, false]);
    folded.chance = ChanceMode::Sampled { turn: 0, river: 6, seed: 2 };
    vec![("short-stack flop", flop), ("sampled turn", turn), ("all-in turn", allin), ("folded turn", folded)]
}

#[test]
fn shared_trees_deal_one_deck_of_live_cards() {
    for (name, cfg) in spots() {
        let m = solver(&shared(&cfg));
        check_shared(&m.config, &m.tree, name);
        let k = match (cfg.root.street, cfg.chance) {
            (Street::Flop, ChanceMode::Sampled { turn, river, .. }) => turn * river,
            (_, ChanceMode::Sampled { river, .. }) => river,
            _ => unreachable!(),
        };
        assert_eq!(m.tree.boards.len(), k, "{name}: boards");
    }
}

#[test]
fn trees_without_sharing_deal_live_cards() {
    for (name, cfg) in spots() {
        let m = solver(&cfg);
        check_cards(&m.config, &m.tree, name);
    }
}

/// The short-stack flop has both kinds of runout and many betting lines:
/// without sharing they see far more boards than with it.
#[test]
fn sharing_reduces_the_boards_and_keeps_the_betting_tree() {
    for (name, cfg) in spots() {
        let (own, shared) = (solver(&cfg), solver(&shared(&cfg)));
        check_same_shape(&own.tree, &shared.tree, name);
        assert!(shared.tree.boards.len() < own.tree.boards.len(), "{name}: no fewer boards");
    }
}

/// With every card dealt the flag changes nothing.
#[test]
fn exact_chance_ignores_the_flag() {
    let mut cfg = config("Jh8d4c2s", RANGES, [false; N]);
    cfg.root = MultiState::street_start(Street::Turn, 900, [400, 3000, 3000], [false; N], 100);
    let (own, shared) = (solver(&cfg), solver(&shared(&cfg)));
    check_same_shape(&own.tree, &shared.tree, "exact");
    let (a, b) = (deals(&own.tree), deals(&shared.tree));
    assert!(a.0.iter().zip(&b.0).all(|(x, y)| x.cards == y.cards), "exact: different cards");
    assert_eq!(own.tree.boards, shared.tree.boards);
}

/// The deck comes from the tree's seed: the same seed, the same tree.
#[test]
fn the_deck_follows_the_seed() {
    let cfg = shared(&spots()[0].1);
    let cards = |seed| {
        let mut cfg = cfg.clone();
        cfg.chance = ChanceMode::Sampled { turn: 8, river: 4, seed };
        deals(&solver(&cfg).tree).0.into_iter().map(|d| d.cards).collect::<Vec<_>>()
    };
    assert_eq!(cards(1), cards(1));
    assert_ne!(cards(1), cards(2));
}

/// FNV-1a over a tree's cards and factors, in node order, and its boards.
fn fingerprint(t: &Tree) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for n in &t.nodes {
        if let Edge::Card(c) = n.edge {
            eat(c.index() as u64);
        }
        eat(u64::from(n.factor.to_bits()));
    }
    for b in &t.boards {
        for c in b {
            eat(c.index() as u64);
        }
    }
    h
}

/// Without the flag, trees are the ones every chance node sampled on its
/// own before the option existed (fingerprints taken on the commit before
/// it, d60dfa4).
#[test]
fn trees_without_sharing_are_unchanged() {
    let got: Vec<u64> = spots().iter().map(|(_, cfg)| fingerprint(&solver(cfg).tree)).collect();
    assert_eq!(got, GOLDEN, "{got:#x?}");
}

const GOLDEN: [u64; 4] = [0xe2e1_18f6_306a_2f3a, 0xda01_508c_ffc7_1eb1, 0xd14c_f2c1_eb0b_b778, 0x328e_549a_56e0_3fd0];

/// A shared tree solves like any other: convergence falls, values finite.
#[test]
fn shared_trees_solve() {
    for (name, cfg) in spots() {
        let mut m = solver(&shared(&cfg));
        let opts = SolveOptions { max_iterations: 60, target_exploitability_pct: 0.0, check_every: 20 };
        let mut seen = Vec::new();
        m.solve(&opts, |p| {
            seen.push(p.exploitability);
            true
        });
        assert!(seen.iter().all(|x| x.is_finite()), "{name}: {seen:?}");
        assert!(seen.last() < seen.first(), "{name}: no progress {seen:?}");
        assert!(m.values().iter().all(|v| v.is_finite()), "{name}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random flops and turns, short stacks, a folded player, sample sizes
    /// from one card to every card, any seed: a valid shared deck and the
    /// same betting tree as without sharing.
    #[test]
    fn random_shared_trees_are_valid(
        cards in prop::sample::subsequence((0u8..52).collect::<Vec<u8>>(), 4).prop_shuffle(),
        flop in any::<bool>(),
        stacks in [100i32..4000, 100i32..4000, 100i32..4000],
        folded in prop::option::of(0usize..N),
        turn in 0usize..10,
        river in 0usize..50,
        seed in any::<u64>(),
    ) {
        let len = if flop { 3 } else { 4 };
        let board: Vec<Card> = cards.into_iter().take(len).map(Card::from_index).collect();
        let mut fold = [false; N];
        if let Some(f) = folded {
            fold[f] = true;
        }
        let mut cfg = config("2c3c4c", RANGES, fold);
        cfg.board = board;
        let stacks = stacks.map(|s| s as Chips);
        cfg.root = MultiState::street_start(Street::from_board_len(len), 900, stacks, fold, 100);
        // Fewer rivers on flops keep the trees small.
        let river = if flop { river % 6 } else { river };
        cfg.chance = ChanceMode::Sampled { turn, river, seed };
        let (Ok(own), Ok(sh)) = (MultiSolver::new(cfg.clone()), MultiSolver::new(shared(&cfg))) else {
            return Ok(());
        };
        check_shared(&sh.config, &sh.tree, "random");
        check_cards(&own.config, &own.tree, "random, not shared");
        check_same_shape(&own.tree, &sh.tree, "random");
    }
}
