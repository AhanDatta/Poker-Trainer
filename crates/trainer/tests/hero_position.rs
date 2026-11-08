//! Choosing the hero's position: over many hands the hero is only dealt
//! the chosen positions, in rotation order, with the blinds and the action
//! order right; the default (rotation) is unchanged.

#![allow(clippy::field_reassign_with_default)]

use holdem::game::{position_offset, positions};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use trainer::save::SessionStore;
use trainer::{HeroPosition, LiveHand, Session, StepResult, TrainerConfig};

const BB: i64 = trainer::fmt::BB;

fn config(seed: u64, players: usize, pos: HeroPosition) -> TrainerConfig {
    let mut cfg = TrainerConfig::default();
    cfg.seed = Some(seed);
    cfg.players = players;
    cfg.hero_position = pos;
    cfg.solver.enabled = false;
    cfg
}

/// Offset of the hero's seat after the button.
fn hero_offset(h: &LiveHand) -> usize {
    let n = h.state.num_players();
    (h.hero + n - h.state.button) % n
}

/// The blinds are posted by the two seats after the button (heads-up: the
/// button posts the small blind) and the first to act preflop is the seat
/// after the big blind (heads-up: the button).
fn check_blinds_and_order(h: &LiveHand) -> Result<(), TestCaseError> {
    let st = &h.state;
    let n = st.num_players();
    let (sb, bb) =
        if n == 2 { (st.button, (st.button + 1) % 2) } else { ((st.button + 1) % n, (st.button + 2) % n) };
    prop_assert_eq!((st.sb_seat(), st.bb_seat()), (sb, bb));
    for p in 0..n {
        let want = if p == sb {
            BB / 2
        } else if p == bb {
            BB
        } else {
            0
        };
        prop_assert_eq!(st.total_commit[p], want, "seat {} of {}", p, n);
    }
    let first = if n == 2 { sb } else { (bb + 1) % n };
    prop_assert_eq!(st.to_act, Some(first));
    prop_assert_eq!(st.position_name(st.to_act.unwrap()), positions(n)[0]);
    Ok(())
}

/// Plays a hand to the end with random hero actions.
fn play_out(hand: &mut LiveHand, rng: &mut StdRng) -> Result<(), TestCaseError> {
    for _ in 0..300 {
        match hand.step() {
            StepResult::HeroToAct => {
                let acts = hand.hero_actions();
                let a = acts[rng.random_range(0..acts.len())];
                hand.hero_act(a).unwrap();
            }
            StepResult::Finished => {
                let st = hand.settlement().unwrap();
                prop_assert_eq!(st.net.iter().sum::<i64>(), -st.rake);
                return Ok(());
            }
            _ => {}
        }
        prop_assert_eq!(hand.state.invariants(), Ok(()));
    }
    Err(TestCaseError::fail("hand did not finish"))
}

/// A table size and a non-empty set of its positions.
fn table_and_seats() -> impl Strategy<Value = (usize, Vec<&'static str>)> {
    (2usize..=9).prop_flat_map(|n| {
        let all = positions(n);
        (Just(n), proptest::sample::subsequence(all, 1..=n))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Over many hands the hero only plays the chosen positions, each new
    /// hand moving the button on to the next one in rotation order, with
    /// the blinds and the action order right.
    #[test]
    fn hero_plays_only_the_chosen_positions(seed in any::<u64>(), (n, seats) in table_and_seats()) {
        let pos = HeroPosition::seats(&seats).unwrap();
        let mut session = Session::new(config(seed, n, pos.clone())).unwrap();
        let chosen: Vec<usize> = seats.iter().map(|s| position_offset(n, s).unwrap()).collect();
        let mut prev_button: Option<usize> = None;
        let mut seen = vec![false; n];
        for _ in 0..3 * n {
            let hand = session.new_hand();
            let off = hero_offset(&hand);
            prop_assert!(chosen.contains(&off), "{} at {}-max, chose {:?}", hand.state.position_name(hand.hero), n, seats);
            prop_assert!(seats.contains(&hand.state.position_name(hand.hero)));
            prop_assert!(pos.allows(n, off));
            check_blinds_and_order(&hand)?;
            if let Some(b) = prev_button {
                // The button moved on by the fewest seats (at least one,
                // or a full lap with one position) that seat the hero in a
                // chosen position.
                let k = (hand.state.button + n - b) % n;
                let k = if k == 0 { n } else { k };
                let skipped_ok = (1..k).all(|j| !chosen.contains(&((hand.hero + 2 * n - b - j) % n)));
                prop_assert!(skipped_ok, "button {} -> {} skipped a chosen position", b, hand.state.button);
            }
            prev_button = Some(hand.state.button);
            seen[off] = true;
        }
        // Every chosen position comes up.
        for &o in &chosen {
            prop_assert!(seen[o], "{} never dealt", positions(n).iter().find(|p| position_offset(n, p) == Some(o)).unwrap());
        }
    }

    /// Hands at a chosen position play out correctly (chips conserved, the
    /// state's invariants hold at every step).
    #[test]
    fn hands_at_chosen_positions_complete(seed in any::<u64>(), (n, seats) in table_and_seats()) {
        let mut session = Session::new(config(seed, n, HeroPosition::seats(&seats).unwrap())).unwrap();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37);
        for _ in 0..3 {
            let mut hand = session.new_hand();
            prop_assert!(seats.contains(&hand.state.position_name(hand.hero)));
            play_out(&mut hand, &mut rng)?;
            session.finish_hand(hand);
        }
    }

    /// Choosing every position deals exactly as rotation does.
    #[test]
    fn every_position_chosen_is_rotation(seed in any::<u64>(), n in 2usize..=9) {
        let mut all = Session::new(config(seed, n, HeroPosition::seats(positions(n)).unwrap())).unwrap();
        let mut rotate = Session::new(config(seed, n, HeroPosition::Alternate)).unwrap();
        for _ in 0..2 * n {
            let (a, b) = (all.new_hand(), rotate.new_hand());
            prop_assert_eq!(a.state.button, b.state.button);
            prop_assert_eq!(&a.state.hole, &b.state.hole);
        }
    }

    /// The default still rotates: the button moves one seat every hand.
    #[test]
    fn default_rotates(seed in any::<u64>(), n in 2usize..=9) {
        let mut cfg = config(seed, n, HeroPosition::Alternate);
        cfg.hero_position = TrainerConfig::default().hero_position;
        let mut session = Session::new(cfg).unwrap();
        let mut prev = session.new_hand().state.button;
        for _ in 0..2 * n {
            let hand = session.new_hand();
            prop_assert_eq!(hand.state.button, (prev + 1) % n);
            check_blinds_and_order(&hand)?;
            prev = hand.state.button;
        }
    }
}

#[test]
fn default_is_rotation() {
    assert_eq!(TrainerConfig::default().hero_position, HeroPosition::Alternate);
}

/// The old `sb` and `bb` settings deal the same button as before.
#[test]
fn blinds_are_dealt_as_before() {
    for n in 2..=9 {
        for (name, old_button) in
            [("sb", if n == 2 { 0 } else { n - 1 }), ("bb", if n == 2 { 1 } else { n - 2 })]
        {
            let cfg =
                TrainerConfig::from_toml(&format!("players = {n}\nhero_position = \"{name}\"\nseed = 1\n"))
                    .unwrap();
            let mut session = Session::new(cfg).unwrap();
            for _ in 0..5 {
                assert_eq!(session.new_hand().state.button, old_button, "{name} at {n} players");
            }
        }
    }
}

/// 6-max BTN only and 9-max UTG+2 only, by name.
#[test]
fn fixed_positions_by_name() {
    for (n, name) in [(6, "BTN"), (6, "UTG"), (6, "HJ"), (9, "UTG+2"), (9, "LJ"), (9, "CO"), (2, "BB")] {
        let mut session = Session::new(config(7, n, HeroPosition::seats([name]).unwrap())).unwrap();
        for _ in 0..20 {
            let h = session.new_hand();
            assert_eq!(h.state.position_name(h.hero), name);
        }
    }
    // Heads-up "btn" is the small blind (the button).
    let mut session = Session::new(config(7, 2, HeroPosition::seats(["btn"]).unwrap())).unwrap();
    let h = session.new_hand();
    assert_eq!((h.state.button, h.state.position_name(h.hero)), (h.hero, "SB"));
}

/// A position the table doesn't have is an error; switching the table size
/// drops it (or falls back to rotation).
#[test]
fn table_size_changes() {
    assert!(Session::new(config(1, 6, HeroPosition::seats(["UTG+2"]).unwrap())).is_err());
    let mut session = Session::new(config(1, 9, HeroPosition::seats(["UTG+2", "BTN"]).unwrap())).unwrap();
    session.set_players(6).unwrap();
    assert_eq!(session.config.hero_position, HeroPosition::seats(["BTN"]).unwrap());
    for _ in 0..10 {
        let h = session.new_hand();
        assert_eq!(h.state.position_name(h.hero), "BTN");
    }
    session.set_players(2).unwrap();
    assert_eq!(session.config.hero_position, HeroPosition::seats(["BTN"]).unwrap(), "heads-up BTN = SB");
    let h = session.new_hand();
    assert_eq!(h.state.position_name(h.hero), "SB");
    let mut session = Session::new(config(1, 9, HeroPosition::seats(["LJ"]).unwrap())).unwrap();
    session.set_players(6).unwrap();
    assert_eq!(session.config.hero_position, HeroPosition::Alternate);
}

/// The choice is saved with the session and kept when it's continued.
#[test]
fn saved_sessions_keep_the_choice() {
    let root = std::env::temp_dir().join(format!("poker-trainer-it-{}-hero-position", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let pos = HeroPosition::seats(["CO", "BTN"]).unwrap();
    let mut session = Session::new(config(3, 6, pos.clone())).unwrap();
    session.start_saving(root.clone());
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..3 {
        let mut hand = session.new_hand();
        play_out(&mut hand, &mut rng).unwrap();
        session.finish_hand(hand);
        session.refresh();
    }
    let id = session.save_id().expect("saved").to_string();
    let loaded = SessionStore::open(&root, &id).unwrap().load().unwrap();
    let mut resumed = Session::resume(loaded, root.clone(), &config(4, 2, HeroPosition::Alternate)).unwrap();
    assert_eq!(resumed.config.hero_position, pos);
    for h in &resumed.history {
        assert!(["CO", "BTN"].contains(&h.state.position_name(h.hero)));
    }
    for _ in 0..6 {
        let h = resumed.new_hand();
        assert!(["CO", "BTN"].contains(&h.state.position_name(h.hero)));
    }
    let _ = std::fs::remove_dir_all(&root);
}
