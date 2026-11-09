use super::*;
use crate::spec::{GameSpec, Limp};
use holdem::{Card, HandClass, HandState, TableConfig};

fn class(s: &str) -> usize {
    s.parse::<HandClass>().unwrap().grid_index()
}

/// Heads-up with `stack_bb` where the only raise is all-in: push or fold.
fn push_fold(stack_bb: f64) -> Tree {
    let mut spec = GameSpec::new(2).with_stack(stack_bb);
    spec.menu.preflop.open_bb = stack_bb;
    spec.rules.allin_after_raises = 0;
    let tree = Tree::build(&spec).unwrap();
    assert_eq!(tree.num_decisions, 2, "small blind: fold or jam; big blind: fold or call");
    tree
}

fn solved(tree: Tree, model: Realization, deals: u64, options: Options) -> Solver {
    let mut solver = Solver::new(tree, model, options);
    for _ in 0..20 {
        solver.run(deals / 20);
    }
    solver
}

/// Share of all hands that takes `action` at the decision `key`.
fn frequency(solver: &Solver, key: &str, action: PreAction) -> f32 {
    let d = solver.tree().find(key).unwrap();
    HandClass::all()
        .map(|c| solver.average(d, c.grid_index()).0[action.index()] * c.num_combos() as f32)
        .sum::<f32>()
        / holdem::NUM_COMBOS as f32
}

#[test]
fn sampling_follows_the_distribution() {
    let mut rng = Rng::new(9);
    let sigma = [0.1, 0.0, 0.6, 0.3];
    let mut counts = [0u32; 4];
    for _ in 0..100_000 {
        counts[sample(&sigma, &mut rng)] += 1;
    }
    assert_eq!(counts[1], 0);
    for (a, &p) in sigma.iter().enumerate() {
        assert!((counts[a] as f32 / 100_000.0 - p).abs() < 0.01, "{counts:?}");
    }
    // Weights that sum to slightly less than 1 still return a real action.
    for _ in 0..1000 {
        assert_ne!(sample(&[0.5, 0.4999, 0.0], &mut rng), 2);
    }
}

#[test]
fn one_thread_is_reproducible_from_the_seed() {
    let run = |seed| {
        let solver =
            solved(push_fold(10.0), Realization::neutral(), 20_000, Options { seed, ..Options::default() });
        let d = solver.tree().find(">SB").unwrap().clone();
        (0..NUM_CLASSES).map(|c| solver.average(&d, c)).collect::<Vec<_>>()
    };
    assert_eq!(run(3), run(3));
    assert_ne!(run(3), run(4));
}

/// Push-or-fold heads-up has a well-known equilibrium: at 10 big blinds
/// the small blind jams roughly 58% of hands and the big blind calls
/// roughly 37%. The solve gets there, nobody can gain by changing a
/// decision, and what one player wins the other loses.
#[test]
fn push_fold_reaches_the_known_equilibrium() {
    let solver = solved(push_fold(10.0), Realization::neutral(), 3_000_000, Options::default());
    let jam = frequency(&solver, ">SB", PreAction::AllIn);
    let call = frequency(&solver, "SB:A>BB", PreAction::Passive);
    assert!((jam - 0.58).abs() < 0.04, "small blind jams {jam}");
    assert!((call - 0.37).abs() < 0.04, "big blind calls {call}");
    let sb = solver.tree().find(">SB").unwrap();
    let bb = solver.tree().find("SB:A>BB").unwrap();
    assert!(solver.average(sb, class("AA")).0[3] > 0.99);
    assert!(solver.average(sb, class("32o")).0[0] > 0.9);
    assert!(solver.average(bb, class("AA")).0[1] > 0.99);
    assert!(solver.average(bb, class("72o")).0[0] > 0.99);

    let d = solver.diagnose(400_000, 77);
    assert!((d.ev[0] + d.ev[1]).abs() < 1e-3, "zero sum: {:?}", d.ev);
    assert!(d.deviation_gain.iter().all(|&g| g < 0.01), "{:?}", d.deviation_gain);
    // An untrained strategy (uniform) leaves a lot to gain.
    let untrained = Solver::new(push_fold(10.0), Realization::neutral(), Options::default());
    let u = untrained.diagnose(400_000, 77);
    assert!(u.deviation_gain.iter().all(|&g| g > 0.2), "{:?}", u.deviation_gain);
}

/// A game with a flop: the solve converges (no decision change gains
/// much), threads don't change the answer much, and rake comes out of
/// the players' results.
#[test]
fn heads_up_with_flops_converges_on_several_threads() {
    let mut spec = GameSpec::new(2).with_stack(30.0).with_rake(0.05, 3.0);
    spec.rules.limp = Limp::SmallBlind;
    let tree = Tree::build(&spec).unwrap();
    let one = solved(tree.clone(), Realization::neutral(), 2_000_000, Options::default());
    let four = solved(tree, Realization::neutral(), 2_000_000, Options { threads: 4, ..Options::default() });
    assert_eq!((one.deals(), four.deals()), (2_000_000, 2_000_000));
    for action in [PreAction::Fold, PreAction::Passive, PreAction::Raise] {
        let (a, b) = (frequency(&one, ">SB", action), frequency(&four, ">SB", action));
        assert!((a - b).abs() < 0.05, "{action:?}: {a} with one thread, {b} with four");
    }
    let d = one.diagnose(400_000, 5);
    assert!(d.deviation_gain.iter().all(|&g| g < 0.03), "{:?}", d.deviation_gain);
    let lost = -(d.ev[0] + d.ev[1]);
    assert!(lost > 0.01 && lost < 0.3, "rake per hand: {lost}");
}

/// Realization factors do what they say: a big blind that realizes less
/// of its equity out of position defends less.
#[test]
fn a_lower_factor_out_of_position_tightens_the_defense() {
    let tree = Tree::build(&GameSpec::new(2).with_stack(40.0)).unwrap();
    let defend = |model: Realization| {
        let solver = solved(tree.clone(), model, 1_500_000, Options::default());
        1.0 - frequency(&solver, "SB:R>BB", PreAction::Fold)
    };
    let neutral = defend(Realization::neutral());
    let harsh = defend(Realization::positional(1.0, 0.7, 1.0));
    assert!(harsh < neutral - 0.1, "defends {harsh} with a 0.7 factor, {neutral} with 1");
}

#[test]
fn warm_up_discounts_regrets_and_counts() {
    let mut solver = Solver::new(push_fold(10.0), Realization::neutral(), Options::default());
    *solver.regret[0].get_mut() = 8.0f32.to_bits();
    *solver.regret[1].get_mut() = (-4.0f32).to_bits();
    *solver.visits[0].get_mut() = 9;
    solver.discount(0.5);
    assert_eq!(f32::from_bits(*solver.regret[0].get_mut()), 4.0);
    assert_eq!(f32::from_bits(*solver.regret[1].get_mut()), -2.0);
    assert_eq!(*solver.visits[0].get_mut(), 5);
    // Regrets never drop below the floor.
    let floor = Options { regret_floor: -1.0, ..Options::default() };
    let solver = solved(push_fold(10.0), Realization::neutral(), 50_000, floor);
    assert!(solver.regret.iter().all(|r| f32::from_bits(r.load(Relaxed)) >= -1.0));
}

#[test]
fn averages_are_distributions_over_the_available_actions() {
    let mut spec = GameSpec::new(3).with_stack(25.0);
    spec.rules.max_cold_callers = 1;
    let solver = solved(Tree::build(&spec).unwrap(), Realization::neutral(), 100_000, Options::default());
    assert_eq!(
        solver.num_cells(),
        solver.tree().decisions().map(|d| d.num_actions() * NUM_CLASSES).sum::<usize>()
    );
    for d in solver.tree().decisions() {
        for c in 0..NUM_CLASSES {
            let (freqs, _) = solver.average(d, c);
            assert!((freqs.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{} {c}", d.key);
            for a in PreAction::ALL {
                assert!(freqs[a.index()] >= 0.0 && (d.avail.has(a) || freqs[a.index()] == 0.0));
            }
        }
    }
}

/// What the solver pays at a leaf is what the game engine pays for the
/// same cards and betting (with every later street checked through),
/// rake included, for every seat. The neutral model realizes exactly the
/// showdown, so this covers flop leaves too.
#[test]
fn leaf_payoffs_match_the_engine_settlement() {
    let mut rng = Rng::new(21);
    for players in [2, 3, 6] {
        let mut spec = GameSpec::new(players).with_stack(40.0).with_rake(0.05, 3.0);
        spec.rules.max_cold_callers = 1;
        spec.rules.limp = Limp::SmallBlind;
        let solver = Solver::new(Tree::build(&spec).unwrap(), Realization::neutral(), Options::default());
        let mut kinds = [0u32; 3];
        for _ in 0..3000 {
            let mut deck: Vec<Card> = Card::all().collect();
            for i in 0..deck.len() {
                let j = i + rng.below((deck.len() - i) as u32) as usize;
                deck.swap(i, j);
            }
            let cards = &deck[..2 * players + 5];
            let deal = Deal::from_cards(players, cards);
            let hole = (0..players).map(|p| [cards[2 * p], cards[2 * p + 1]]).collect();
            let config = TableConfig::new(players, BB / 2, BB).with_rake(spec.rake_rate, spec.rake_cap());
            let mut state = HandState::new(config, 0, vec![spec.stack(); players], hole);
            // A random line through the tree, replayed on the engine.
            let mut node = 0usize;
            let leaf = loop {
                match &solver.tree().nodes[node] {
                    Node::Decision(d) => {
                        let actions: Vec<PreAction> = d.actions().collect();
                        let action = actions[rng.below(actions.len() as u32) as usize];
                        state.apply(d.avail.concrete(action).unwrap()).unwrap();
                        node = d.children[action.index()] as usize;
                    }
                    Node::Leaf(l) => break l,
                }
            };
            kinds[match leaf.kind {
                LeafKind::Won(_) => 0,
                LeafKind::AllIn => 1,
                LeafKind::Flop => 2,
            }] += 1;
            let mut dealt = 2 * players;
            while state.phase != holdem::game::Phase::Finished {
                if state.to_act.is_some() {
                    state.apply(holdem::Action::Check).unwrap();
                } else {
                    let n = state.cards_needed();
                    state.deal_board(&cards[dealt..dealt + n]).unwrap();
                    dealt += n;
                }
            }
            let settled = state.settle();
            let Flat::Leaf { live, prize, commit, context } = &solver.nodes[node] else { panic!("a leaf") };
            for seat in 0..players {
                let paid = solver.payoff(*live, *prize, commit[seat], context[seat], seat, &deal);
                let expected = settled.net[seat] as f32 / BB as f32;
                // Odd chips of a split pot go to one of the winners.
                assert!(
                    (paid - expected).abs() <= 0.021,
                    "{players} players, seat {seat}: {paid} vs {expected}"
                );
            }
        }
        assert!(kinds.iter().all(|&k| k > 50), "every kind of leaf was reached: {kinds:?}");
    }
}
