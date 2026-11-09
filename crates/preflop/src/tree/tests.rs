use super::*;
use crate::spec::TreeRules;
use holdem::BetMenu;
use proptest::prelude::*;
use std::collections::HashSet;
use trainer::preflop::ChartSet;

fn spec(players: usize, max_cold_callers: usize) -> GameSpec {
    let mut spec = GameSpec::new(players);
    spec.rules.max_cold_callers = max_cold_callers;
    spec
}

#[test]
fn heads_up_tree_has_the_chart_spots() {
    let mut spec = GameSpec::new(2);
    spec.rules.limp = Limp::SmallBlind;
    let tree = Tree::build(&spec).unwrap();
    let Node::Decision(root) = &tree.nodes[0] else { panic!("the root is a decision") };
    assert_eq!((root.key.as_str(), root.seat, root.commit), (">SB", 0, BB / 2));
    for key in [">SB", "SB:R>BB", "SB:C>BB", "SB:R BB:R>SB", "SB:C BB:R>SB", "SB:R BB:R SB:R>BB"] {
        assert!(tree.find(key).is_some(), "{key}");
    }
    // Without limps the small blind raises or folds.
    let tree = Tree::build(&GameSpec::new(2)).unwrap();
    assert!(tree.find("SB:C>BB").is_none());
    assert_eq!(tree.find(">SB").unwrap().actions().collect::<Vec<_>>(), [PreAction::Fold, PreAction::Raise]);
}

/// Every spot of the built-in 6-max solver charts is a decision of the
/// tree, even with a single cold caller allowed.
#[test]
fn six_max_tree_covers_the_built_in_charts() {
    let tree = Tree::build(&spec(6, 1)).unwrap();
    let keys: HashSet<&str> = tree.decisions().map(|d| d.key.as_str()).collect();
    let charts = ChartSet::builtin(6);
    assert_eq!(charts.charts.len(), 506);
    for key in charts.charts.keys() {
        assert!(keys.contains(key.as_str()), "{key}");
    }
}

fn check_invariants(tree: &Tree) {
    let n = tree.spec.players;
    let stack = tree.spec.stack();
    let mut keys = HashSet::new();
    let mut referenced = vec![0u32; tree.nodes.len()];
    let mut slots = 0;
    for node in &tree.nodes {
        match node {
            Node::Decision(d) => {
                assert!(keys.insert(d.key.as_str()), "duplicate key {}", d.key);
                assert_eq!(d.slot, slots);
                slots += 1;
                assert!((d.seat as usize) < n && d.commit >= 0 && d.commit < stack, "{}", d.key);
                assert!(d.num_actions() >= 1, "{}: no actions", d.key);
                for a in PreAction::ALL {
                    let child = d.children[a.index()];
                    assert_eq!(child != NO_CHILD, d.avail.has(a), "{} {a:?}", d.key);
                    if child != NO_CHILD {
                        assert!((child as usize) < tree.nodes.len());
                        referenced[child as usize] += 1;
                    }
                }
                assert!(d.key.ends_with(&format!(">{}", holdem::game::position_name(n, d.seat as usize))));
            }
            Node::Leaf(l) => {
                assert_eq!(l.pot, l.commit.iter().sum::<Chips>());
                assert!(l.commit[..n].iter().all(|&c| (0..=stack).contains(&c)));
                assert!(l.commit[n..].iter().all(|&c| c == 0) && l.live >> n == 0);
                assert_eq!(l.rake, tree.spec.rake(l.pot));
                let live: Vec<usize> = (0..n).filter(|&s| l.is_live(s)).collect();
                match l.kind {
                    LeafKind::Won(w) => assert_eq!(live, [w as usize]),
                    LeafKind::AllIn | LeafKind::Flop => {
                        assert!(live.len() >= 2);
                        assert!(live.iter().all(|&s| l.commit[s] == l.commit[live[0]]));
                        let all_in = live.iter().filter(|&&s| l.commit[s] == stack).count();
                        assert_eq!(l.kind == LeafKind::AllIn, all_in + 1 >= live.len());
                    }
                }
                if let Some(a) = l.aggressor {
                    assert!((a as usize) < n && l.raises > 0);
                } else {
                    assert_eq!(l.raises, 0);
                }
            }
        }
    }
    assert_eq!(slots as usize, tree.num_decisions);
    assert_eq!(referenced[0], 0, "the root has no parent");
    assert!(referenced[1..].iter().all(|&r| r == 1), "every other node has exactly one parent");
}

#[test]
fn trees_of_every_table_size_are_consistent() {
    for players in 2..=MAX_PLAYERS {
        let cold_callers = if players <= 6 { 1 } else { 0 };
        for limp in [Limp::Never, Limp::SmallBlind] {
            let mut spec = spec(players, cold_callers);
            spec.rules.limp = limp;
            spec.rake_rate = 0.05;
            spec.rake_cap_bb = 3.0;
            check_invariants(&Tree::build(&spec).unwrap());
        }
    }
    let mut short = spec(3, usize::MAX);
    short.stack_bb = 12.5;
    short.rules = TreeRules { limp: Limp::Any, allin_after_raises: 0, ..TreeRules::default() };
    check_invariants(&Tree::build(&short).unwrap());
}

#[test]
fn tree_rules_prune_only_what_they_say() {
    let avail = |fold: bool, call: bool, raise: bool, allin: bool| Availability {
        actions: [
            fold.then_some(Action::Fold),
            call.then_some(Action::Call),
            raise.then_some(Action::Raise(250)),
            allin.then_some(Action::Raise(10_000)),
        ],
    };
    let sit = |to_call, num_raises, limpers| Situation {
        street: holdem::Street::Preflop,
        pot: 150,
        to_call,
        current_bet: 100,
        max_to: 10_000,
        min_raise_to: 200,
        num_raises,
        limpers,
        big_blind: BB,
    };
    let has = |a: &Availability| PreAction::ALL.map(|x| a.has(x));
    let rules = TreeRules { max_cold_callers: 1, ..TreeRules::default() };
    // First in: no limp, no open jam.
    assert_eq!(
        has(&prune(avail(true, true, true, true), &rules, &sit(100, 0, 0), "CO", 0, false)),
        [true, false, true, false]
    );
    // The small blind may limp under that rule, but not over a limper.
    let sb = TreeRules { limp: Limp::SmallBlind, ..rules.clone() };
    assert!(prune(avail(true, true, true, true), &sb, &sit(50, 0, 0), "SB", 0, false).has(PreAction::Passive));
    assert!(
        !prune(avail(true, true, true, true), &sb, &sit(50, 0, 1), "SB", 0, false).has(PreAction::Passive)
    );
    assert!(
        !prune(avail(true, true, true, true), &sb, &sit(100, 0, 0), "BTN", 0, false).has(PreAction::Passive)
    );
    // Facing a raise: the second cold caller is cut, unless it's the big
    // blind or a player who already acted.
    let facing = sit(250, 1, 0);
    assert!(prune(avail(true, true, true, true), &rules, &facing, "BTN", 0, false).has(PreAction::Passive));
    assert!(!prune(avail(true, true, true, true), &rules, &facing, "BTN", 1, false).has(PreAction::Passive));
    assert!(prune(avail(true, true, true, true), &rules, &facing, "BB", 1, false).has(PreAction::Passive));
    assert!(prune(avail(true, true, true, true), &rules, &facing, "BTN", 1, true).has(PreAction::Passive));
    // All-in appears after two raises, and stays when it's the only raise.
    assert!(
        prune(avail(true, true, true, true), &rules, &sit(875, 2, 0), "BTN", 0, false).has(PreAction::AllIn)
    );
    assert!(prune(avail(true, true, false, true), &rules, &facing, "BTN", 0, false).has(PreAction::AllIn));
    // A check is never removed, and a call stays when there's no raise.
    assert!(
        prune(avail(false, true, true, true), &rules, &sit(0, 0, 1), "BB", 0, false).has(PreAction::Passive)
    );
    assert!(prune(avail(true, true, false, false), &rules, &sit(100, 0, 0), "CO", 0, false)
        .has(PreAction::Passive));
}

/// A re-raise starts a new count of cold callers, and the last raiser is
/// the aggressor of the pot.
#[test]
fn raises_reset_the_callers_and_set_the_aggressor() {
    let tree = Tree::build(&spec(6, 1)).unwrap();
    // UTG opens, HJ calls: the cutoff can't cold call any more...
    assert!(!tree.find("UTG:R HJ:C>CO").unwrap().avail.has(PreAction::Passive));
    // ...but once it 3-bets, the button may call the 3-bet.
    assert!(tree.find("UTG:R HJ:C CO:R>BTN").unwrap().avail.has(PreAction::Passive));
    // UTG opens, the button (seat 0) 3-bets, UTG (seat 3) calls.
    let facing = tree.find("UTG:R BTN:R>UTG").unwrap();
    let Node::Leaf(flop) = &tree.nodes[facing.children[PreAction::Passive.index()] as usize] else {
        panic!("calling the 3-bet closes the betting")
    };
    assert_eq!((flop.kind, flop.raises, flop.aggressor), (LeafKind::Flop, 2, Some(0)));
    assert_eq!(flop.live, 0b1001);
}

#[test]
fn oversized_trees_are_refused() {
    let mut spec = GameSpec::new(9);
    spec.rules.max_nodes = 10_000;
    assert_eq!(Tree::build(&spec).unwrap_err(), SpecError::TooLarge { limit: 10_000 });
    assert!(Tree::build(&GameSpec::new(12)).is_err());
}

fn arb_spec() -> impl Strategy<Value = GameSpec> {
    (2usize..=MAX_PLAYERS, 5u32..=400, 0usize..=2, 0u32..=3, 0usize..3, any::<bool>(), 20u32..=40, 15u32..=40)
        .prop_map(|(players, stack_half_bb, cold, allin_after, limp, allow_limp, open, reraise)| {
            let mut spec = GameSpec::new(players).with_stack(stack_half_bb as f64 / 2.0).with_rake(0.05, 3.0);
            // Keep big tables small enough to build in a test.
            spec.rules.max_cold_callers =
                if players > 6 { 0 } else { cold.min(if players > 4 { 1 } else { 2 }) };
            spec.rules.allin_after_raises = allin_after;
            spec.rules.limp =
                [Limp::Never, Limp::SmallBlind, Limp::Any][if players > 5 { limp.min(1) } else { limp }];
            let mut menu = BetMenu::default();
            menu.preflop.allow_limp = allow_limp;
            menu.preflop.open_bb = open as f64 / 10.0;
            menu.preflop.reraise_mult = vec![reraise as f64 / 10.0, 2.3];
            spec.menu = menu;
            spec
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// A random walk down the tree, replayed on a fresh game state with the
    /// trainer's own bookkeeping, sees the same seats, keys, actions and
    /// chips at every step, and the same pot at the end.
    #[test]
    fn walks_match_an_independent_replay(spec in arb_spec(), picks in proptest::collection::vec(0usize..4, 64)) {
        let tree = Tree::build(&spec).unwrap();
        check_invariants(&tree);
        let n = spec.players;
        let config = TableConfig::new(n, BB / 2, BB).with_rake(spec.rake_rate, spec.rake_cap());
        let hole = (0..n).map(|p| [Card::from_index(51 - 2 * p as u8), Card::from_index(50 - 2 * p as u8)]).collect();
        let mut state = HandState::new(config, 0, vec![spec.stack(); n], hole);
        let mut line: Vec<PreDecision> = Vec::new();
        let mut node = 0u32;
        let mut picks = picks.into_iter();
        loop {
            match &tree.nodes[node as usize] {
                Node::Decision(d) => {
                    let p = state.to_act.expect("the replay is at a decision too");
                    let position = state.position_name(p).to_string();
                    prop_assert_eq!(d.seat as usize, p);
                    prop_assert_eq!(&d.key, &spot_key(&line, &position));
                    prop_assert_eq!(d.commit, state.total_commit[p]);
                    let full = trainer::preflop::availability(&state, &spec.menu).unwrap();
                    let actions: Vec<PreAction> = d.actions().collect();
                    for &a in &actions {
                        prop_assert_eq!(d.avail.concrete(a), full.concrete(a), "{}: {:?}", d.key, a);
                    }
                    let action = actions[picks.next().unwrap_or(0) % actions.len()];
                    state.apply(d.avail.concrete(action).unwrap()).unwrap();
                    line.push(PreDecision { seat: p, position, key: d.key.clone(), avail: full, action });
                    node = d.children[action.index()];
                }
                Node::Leaf(l) => {
                    prop_assert!(state.to_act.is_none());
                    prop_assert_eq!(l.pot, state.pot());
                    prop_assert_eq!(&l.commit[..n], &state.total_commit[..]);
                    let live = state.players_in_hand();
                    prop_assert_eq!(l.live, live.iter().fold(0u16, |m, &p| m | 1 << p));
                    prop_assert_eq!(l.raises as u32, state.num_raises);
                    match l.kind {
                        LeafKind::Won(w) => {
                            prop_assert_eq!(state.phase, Phase::Finished);
                            let settled = state.settle();
                            prop_assert_eq!(settled.net[w as usize], l.pot - l.commit[w as usize]);
                            prop_assert_eq!(settled.rake, 0);
                        }
                        LeafKind::AllIn => prop_assert!(state.is_runout()),
                        LeafKind::Flop => prop_assert!(state.phase == Phase::AwaitingBoard && !state.is_runout()),
                    }
                    break;
                }
            }
        }
    }
}
