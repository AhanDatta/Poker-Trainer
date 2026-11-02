#![allow(clippy::needless_range_loop)]
//! Property-based tests: invariants that must hold for *any* input.

use holdem::card::{parse_cards, Card};
use holdem::combo::{combo_index, HandClass, NUM_CLASSES};
use holdem::eval::{evaluate_cards, HandCategory};
use holdem::game::Phase;
use holdem::{Action, BetMenu, Chips, Deck, HandState, Range, Street, TableConfig};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `n` distinct cards in random order.
fn cards(n: usize) -> impl Strategy<Value = Vec<Card>> {
    prop::sample::subsequence((0u8..52).collect::<Vec<_>>(), n)
        .prop_shuffle()
        .prop_map(|v| v.into_iter().map(Card::from_index).collect())
}

fn best_five(seven: &[Card]) -> u32 {
    let mut best = 0;
    for a in 0..seven.len() {
        for b in a + 1..seven.len() {
            let five: Vec<Card> = (0..seven.len()).filter(|&i| i != a && i != b).map(|i| seven[i]).collect();
            best = best.max(evaluate_cards(&five));
        }
    }
    best
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// A 7-card hand is worth exactly its best 5-card subset.
    #[test]
    fn seven_cards_equal_best_five(c in cards(7)) {
        prop_assert_eq!(evaluate_cards(&c), best_five(&c));
    }

    /// Relabeling suits never changes a hand's value.
    #[test]
    fn suit_relabeling_is_invariant(c in cards(7), perm in Just([0u8, 1, 2, 3]).prop_shuffle()) {
        let mapped: Vec<Card> = c.iter().map(|x| Card::new(x.rank(), perm[x.suit() as usize])).collect();
        prop_assert_eq!(evaluate_cards(&c), evaluate_cards(&mapped));
    }

    /// Adding a card can only improve a hand.
    #[test]
    fn more_cards_never_hurt(c in cards(7)) {
        prop_assert!(evaluate_cards(&c[..5]) <= evaluate_cards(&c[..6]));
        prop_assert!(evaluate_cards(&c[..6]) <= evaluate_cards(&c));
    }

    /// The category encoded in a value matches a direct classification.
    #[test]
    fn category_counts_ranks(c in cards(5)) {
        let v = evaluate_cards(&c);
        let mut counts = [0u8; 13];
        for x in &c { counts[x.rank() as usize] += 1; }
        let max = *counts.iter().max().unwrap();
        let pairs = counts.iter().filter(|&&k| k == 2).count();
        let cat = HandCategory::of(v);
        match max {
            4 => prop_assert_eq!(cat, HandCategory::Quads),
            3 if pairs == 1 => prop_assert_eq!(cat, HandCategory::FullHouse),
            3 => prop_assert!(cat == HandCategory::Trips),
            2 if pairs == 2 => prop_assert!(cat == HandCategory::TwoPair),
            2 => prop_assert!(cat == HandCategory::Pair),
            _ => prop_assert!(matches!(cat, HandCategory::HighCard | HandCategory::Straight
                | HandCategory::Flush | HandCategory::StraightFlush)),
        }
    }

    /// Ranges written per class parse back to the same class weights, and
    /// `to_class_string` round-trips.
    #[test]
    fn range_notation_roundtrip(classes in prop::collection::btree_map(0usize..NUM_CLASSES, 1u8..=20, 1..40)) {
        let text: Vec<String> = classes
            .iter()
            .map(|(&g, &w)| format!("{}:{}", HandClass::from_grid_index(g), w as f32 / 20.0))
            .collect();
        let r = Range::parse(&text.join(", ")).unwrap();
        let cw = r.class_weights();
        for g in 0..NUM_CLASSES {
            let expected = classes.get(&g).map_or(0.0, |&w| w as f32 / 20.0);
            prop_assert!((cw[g] - expected).abs() < 1e-6);
        }
        let back = Range::parse(&r.to_class_string()).unwrap();
        for (a, b) in back.class_weights().iter().zip(cw.iter()) {
            prop_assert!((a - b).abs() < 1e-3);
        }
    }
}

/// Plays a random hand: menu actions most of the time, arbitrary legal
/// raise sizes sometimes. Checks the engine invariants after every step and
/// the settlement invariants at the end.
fn random_hand(
    seed: u64,
    n: usize,
    stacks: Vec<Chips>,
    rake_rate: f64,
    rake_cap: Chips,
) -> Result<(), TestCaseError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut deck = Deck::shuffled(&mut rng);
    let hole: Vec<[Card; 2]> = (0..n).map(|_| [deck.deal(), deck.deal()]).collect();
    let cfg = TableConfig::new(n, 50, 100).with_rake(rake_rate, rake_cap);
    let button = rng.random_range(0..n);
    let mut s = HandState::new(cfg, button, stacks.clone(), hole);
    let menu = BetMenu::default();
    for _ in 0..500 {
        prop_assert_eq!(s.invariants(), Ok(()));
        match s.phase {
            Phase::Finished => break,
            Phase::AwaitingBoard => {
                let k = s.cards_needed();
                s.deal_board(&deck.deal_n(k)).unwrap();
            }
            Phase::Betting => {
                let p = s.to_act.unwrap();
                let sit = s.situation(p);
                let actions = s.legal_actions(&menu);
                // Menu invariants.
                prop_assert!(!actions.is_empty());
                prop_assert_eq!(actions.contains(&Action::Fold), sit.to_call > 0);
                prop_assert_eq!(actions.contains(&Action::Check), sit.to_call == 0);
                let raises: Vec<Chips> = actions
                    .iter()
                    .filter_map(|a| if let Action::Raise(t) = a { Some(*t) } else { None })
                    .collect();
                prop_assert!(raises.windows(2).all(|w| w[0] < w[1]), "raises sorted: {:?}", raises);
                for &t in &raises {
                    prop_assert!(t > sit.current_bet && t <= sit.max_to);
                    prop_assert!(t >= sit.min_raise_to.min(sit.max_to));
                }
                let action = if rng.random_bool(0.85) || !sit.can_raise() || sit.min_raise_to >= sit.max_to {
                    actions[rng.random_range(0..actions.len())]
                } else {
                    Action::Raise(rng.random_range(sit.min_raise_to..=sit.max_to))
                };
                s.apply(action).map_err(|e| TestCaseError::fail(format!("{action:?} rejected: {e}")))?;
            }
        }
    }
    prop_assert_eq!(s.phase, Phase::Finished, "hand didn't finish");
    let st = s.settle();
    let pot = s.pot();
    prop_assert_eq!(st.payouts.iter().sum::<Chips>() + st.rake, pot);
    prop_assert_eq!(st.net.iter().sum::<Chips>(), -st.rake);
    prop_assert!(st.rake <= rake_cap && st.rake as f64 <= rake_rate * pot as f64 + 1e-9);
    if s.board.len() < 3 {
        prop_assert_eq!(st.rake, 0, "no flop, no drop");
    }
    for p in 0..n {
        prop_assert!(st.payouts[p] >= 0);
        prop_assert_eq!(st.net[p], st.payouts[p] - s.total_commit[p]);
        if s.folded[p] {
            prop_assert_eq!(st.payouts[p], 0);
        }
        // Nobody wins more than the chips they matched from everyone.
        let cap: Chips = (0..n).map(|q| s.total_commit[q].min(s.total_commit[p])).sum();
        prop_assert!(st.payouts[p] <= cap);
    }
    // A single remaining player collects everything (minus rake).
    let live = s.players_in_hand();
    if live.len() == 1 {
        prop_assert_eq!(st.payouts[live[0]], pot - st.rake);
    } else if s.street == Street::River && s.board.len() == 5 {
        prop_assert!(st.showdown);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn random_hands_keep_invariants(
        seed in any::<u64>(),
        n in 2usize..=9,
        stacks in prop::collection::vec(150i64..30_000, 9),
        rake in prop_oneof![Just(0.0f64), 0.01f64..0.1],
        cap in 0i64..1000,
    ) {
        random_hand(seed, n, stacks[..n].to_vec(), rake, cap)?;
    }
}

#[test]
fn parse_cards_rejects_garbage() {
    assert!(parse_cards("Xx").is_err());
    assert_eq!(combo_index(Card::new(0, 0), Card::new(0, 1)), 0);
}
