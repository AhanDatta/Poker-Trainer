use super::*;
use crate::storage::StorageMode;
use holdem::card::parse_cards;
use holdem::{Range, Street, StreetSizes};

fn menu_one(bet: &[&str], raise: &[&str]) -> PostflopMenu {
    let s = StreetSizes::new(bet, raise);
    PostflopMenu {
        flop: s.clone(),
        turn: s.clone(),
        river: s,
        max_raises: 3,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merge_threshold: 0.0,
    }
}

fn config(board: &str, oop: &str, ip: &str, pot: Chips, stack: Chips, menu: PostflopMenu) -> SolveConfig {
    let board = parse_cards(board).unwrap();
    let street = Street::from_board_len(board.len());
    SolveConfig::new(
        board,
        [Range::parse(oop).unwrap().into_weights(), Range::parse(ip).unwrap().into_weights()],
        BettingState::street_start(street, pot, [stack, stack], 100),
        menu,
    )
}

fn solve(cfg: SolveConfig, iters: u32, target: f64) -> (Solver, Progress) {
    let mut s = Solver::new(cfg).unwrap();
    let opts = SolveOptions { max_iterations: iters, target_exploitability_pct: target, check_every: 25 };
    let p = s.solve(&opts, |_| true);
    (s, p)
}

fn combo(s: &str) -> usize {
    let c = parse_cards(s).unwrap();
    holdem::combo_index(c[0], c[1])
}

/// Classic polarized river: OOP has 6 combos of nuts (AA) and 6 combos of
/// air against IP's bluff-catchers. With a pot-sized bet OOP bluffs half its
/// air, IP calls half the time, and bluffing/checking air are both worth 0.
#[test]
fn polarized_river_toy_game() {
    let cfg = config("Kc8d5h3s2c", "AA, JTs, J9s:0.5", "KQ", 1000, 5000, menu_one(&["100%"], &[]));
    let (s, p) = solve(cfg, 2000, 0.05);
    assert!(p.exploitability_pct < 0.2, "exploitability {}", p.exploitability_pct);

    let root = s.query(&[]).unwrap();
    assert_eq!(root.actions, vec![Action::Check, Action::Raise(1000)]);
    let aa = root.hand_index(combo("AsAh")).unwrap();
    assert!(root.hand_strategy(aa)[1] > 0.98);
    let ev = root.hand_ev(aa);
    assert!((ev[1] - 1500.0).abs() < 20.0, "AA bet EV {}", ev[1]);
    assert!((ev[0] - 1000.0).abs() < 20.0, "AA check EV {}", ev[0]);

    // Air bets about half the time overall.
    let n = root.num_hands();
    let (mut bet, mut tot) = (0.0, 0.0);
    for i in 0..n {
        let c = root.combos[i] as usize;
        if holdem::HandClass::of_combo(c).hi == 9 {
            // J-high air
            bet += root.reach[i] as f64 * root.strategy[n + i] as f64;
            tot += root.reach[i] as f64;
            let ev = root.hand_ev(i);
            assert!(ev[0].abs() < 20.0 && ev[1].abs() < 40.0, "air EVs {ev:?}");
        }
    }
    assert!((bet / tot - 0.5).abs() < 0.05, "bluff frequency {}", bet / tot);

    // IP calls half the time.
    let facing = s.query(&[Step::Action(Action::Raise(1000))]).unwrap();
    assert_eq!(facing.actions, vec![Action::Fold, Action::Call]);
    let call = facing.range_strategy()[1];
    assert!((call - 0.5).abs() < 0.05, "call frequency {call}");
    let kq = facing.hand_index(combo("KdQd")).unwrap();
    let ev = facing.hand_ev(kq);
    assert!(ev[0].abs() < 1e-3 && ev[1].abs() < 30.0, "KQ EVs {ev:?}");

    // Range narrowing: after a bet OOP keeps all AA and ~half the air.
    let ranges = s.ranges_after(&[Step::Action(Action::Raise(1000))]).unwrap();
    assert!(ranges[0][combo("AsAh")] > 0.98);
    let jt = ranges[0][combo("JsTs")];
    assert!(jt > 0.2 && jt < 0.8, "JT weight {jt}");
}

/// AA vs KK all-in on the turn: KK has exactly 2 outs among 44 rivers.
/// Checks chance factors and card removal.
#[test]
fn all_in_equity_on_turn() {
    let cfg = config("2c7d9hTs", "AhAd", "KhKd", 1000, 1000, menu_one(&["allin"], &[]));
    let (s, _) = solve(cfg, 50, 0.0);
    let facing = s.query(&[Step::Action(Action::Raise(1000))]).unwrap();
    assert_eq!(facing.actions, vec![Action::Fold, Action::Call]);
    let ev = facing.hand_ev(0);
    let expected = 2.0 / 44.0 * 3000.0 - 1000.0;
    assert!((ev[1] as f64 - expected).abs() < 0.5, "call EV {} vs {expected}", ev[1]);
    assert!(ev[0].abs() < 1e-3);
    // KK should fold.
    assert!(facing.hand_strategy(0)[0] > 0.99);
}

/// Mid-street root (facing a bet) on a board where every hand chops.
#[test]
fn mid_street_root_with_ties() {
    let mut cfg = config("AsKsQsJsTs", "22", "33", 1500, 5000, menu_one(&["100%"], &["3x"]));
    cfg.root = BettingState {
        street: Street::River,
        pot: 1500,
        commit: [0, 500],
        stack: [5000, 4500],
        current_bet: 500,
        last_raise_inc: 500,
        num_raises: 1,
        to_act: 0,
        needs: [true, false],
    };
    let (s, _) = solve(cfg, 200, 0.0);
    let root = s.query(&[]).unwrap();
    assert_eq!(root.actions[..2], [Action::Fold, Action::Call]);
    let ev = root.hand_ev(0);
    assert!(ev[0].abs() < 1e-3, "fold EV {}", ev[0]);
    assert!((ev[1] - 500.0).abs() < 1.0, "call EV {}", ev[1]);
}

/// A real turn spot converges.
#[test]
fn turn_spot_converges() {
    let cfg = config(
        "Qh9d4c2s",
        "AQ, KQ, QJs, JTs, 99, 44, T9s, 87s, A5s, KJs",
        "AA, KK, AQ, QT, 99, JJ, TT, 98s, KJs, 65s, AJs",
        600,
        4000,
        PostflopMenu::default(),
    );
    let mut s = Solver::new(cfg).unwrap();
    let opts = SolveOptions { max_iterations: 300, target_exploitability_pct: 0.3, check_every: 20 };
    let mut history = Vec::new();
    let p = s.solve(&opts, |p| {
        history.push(p.exploitability_pct);
        true
    });
    assert!(p.exploitability_pct < 0.5, "{history:?}");
    assert!(history.first().unwrap() > history.last().unwrap());
}

/// Sampled flop solves build, solve and answer root queries.
#[test]
fn sampled_flop_solve() {
    let mut cfg = config(
        "Kh7d2c",
        "AK, KQ, 77, 22, AJs, QJs, T9s, 65s, A5s",
        "AK, KJ, QQ, JJ, 88, A7s, 98s, JTs, 76s",
        600,
        9700,
        PostflopMenu::default(),
    );
    cfg.chance = ChanceMode::Sampled { turn: 8, river: 6, seed: 3 };
    cfg.future_menu = Some(menu_one(&["66%"], &["3x"]));
    let est = Solver::estimate(&cfg).unwrap();
    assert!(est.player_nodes > 0);
    let (s, p) = solve(cfg, 300, 0.5);
    assert!(p.exploitability_pct < 1.0, "{}", p.exploitability_pct);
    let root = s.query(&[]).unwrap();
    let total: f64 = root.range_strategy().iter().sum();
    assert!((total - 1.0).abs() < 1e-4);
}

#[test]
fn too_large_is_rejected() {
    let mut cfg = config("Kh7d2c", "any", "any", 600, 9700, PostflopMenu::default());
    cfg.max_memory_bytes = 1 << 20;
    match Solver::new(cfg) {
        Err(SolverError::TooLarge { .. }) => {}
        other => panic!("expected TooLarge, got {:?}", other.err()),
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn assert_same_node(a: &NodeInfo, b: &NodeInfo, strat_tol: f32, ev_tol: f32) {
    assert_eq!(a.actions, b.actions);
    assert_eq!(a.combos, b.combos);
    let ds = max_diff(&a.strategy, &b.strategy);
    let de = max_diff(&a.ev, &b.ev);
    assert!(ds <= strat_tol, "strategy differs by {ds}");
    assert!(de <= ev_tol, "EV differs by {de}");
}

/// The breadth-first batched river evaluator must match plain recursion.
#[test]
fn batched_river_matches_recursion() {
    let make = || {
        let mut c = config(
            "Qh9d4c2s",
            "AQ, KQ, QJs, JTs, 99, 44, T9s, 87s, A5s, KJs",
            "AA, KK, AQ, QT, 99, JJ, TT, 98s, KJs, 65s, AJs",
            600,
            4000,
            PostflopMenu::default(),
        );
        c.storage = StorageMode::F32;
        c
    };
    // Few iterations: the two paths only differ by floating-point rounding,
    // which CFR's dynamics amplify over many iterations.
    let opts = SolveOptions { max_iterations: 5, target_exploitability_pct: 0.0, check_every: 5 };
    let mut a = Solver::new(make()).unwrap();
    let mut b = Solver::new(make()).unwrap();
    b.set_batch_river(false);
    let pa = a.solve(&opts, |_| true);
    let pb = b.solve(&opts, |_| true);
    assert!((pa.exploitability - pb.exploitability).abs() < 1e-4 * (1.0 + pb.exploitability.abs()));
    // The batched path sums some fold values in strength order rather than
    // hand order. After one iteration the paths differ by f32 rounding
    // (5e-7 in strategy, 1e-4 chips in EV, with EVs up to ~2500 chips);
    // five iterations amplify that to ~1e-5 and ~0.1 chips.
    for path in [vec![], vec![Step::Action(Action::Check)]] {
        assert_same_node(&a.query(&path).unwrap(), &b.query(&path).unwrap(), 5e-5, 0.25);
    }
    let river = [Step::Action(Action::Check), Step::Action(Action::Check), Step::Card("Ks".parse().unwrap())];
    assert_same_node(&a.query(&river).unwrap(), &b.query(&river).unwrap(), 5e-5, 0.25);
}

/// Solving with suit isomorphism must give the same strategies and EVs,
/// including for queries that go through a mirrored (non-canonical) card.
#[test]
fn isomorphism_matches_full_tree() {
    // Monotone turn: clubs, diamonds and hearts are interchangeable.
    let make = |iso: bool| {
        let mut c = config(
            "Ks9s5s2s",
            "AA, KQ, QJ, T9, 87, 66, A5",
            "AK, KJ, JT, 98, 76, 55, Q2",
            1000,
            3000,
            menu_one(&["66%"], &["3x"]),
        );
        c.isomorphism = iso;
        c.storage = StorageMode::F32;
        c
    };
    let opts = SolveOptions { max_iterations: 60, target_exploitability_pct: 0.0, check_every: 60 };
    let mut a = Solver::new(make(true)).unwrap();
    let mut b = Solver::new(make(false)).unwrap();
    assert!(a.isomorphic_pairs() >= 3);
    assert!(a.stats().total_nodes() < b.stats().total_nodes());
    let pa = a.solve(&opts, |_| true);
    let pb = b.solve(&opts, |_| true);
    assert!(
        (pa.exploitability - pb.exploitability).abs() < 1e-2 * (1.0 + pb.exploitability.abs()),
        "{pa:?} {pb:?}"
    );
    assert_same_node(&a.query(&[]).unwrap(), &b.query(&[]).unwrap(), 2e-3, 0.5);
    // Rivers in every suit, including mirrored ones.
    for river in ["7c", "7d", "7h", "As", "Ah"] {
        let path =
            [Step::Action(Action::Check), Step::Action(Action::Check), Step::Card(river.parse().unwrap())];
        assert_same_node(&a.query(&path).unwrap(), &b.query(&path).unwrap(), 2e-3, 0.5);
        let ra = a.ranges_after(&path).unwrap();
        let rb = b.ranges_after(&path).unwrap();
        assert!(max_diff(&ra[0], &rb[0]) < 1e-3 && max_diff(&ra[1], &rb[1]) < 1e-3);
    }
}

/// 16-bit compressed storage stays close to f32.
#[test]
fn compressed_storage_is_close() {
    let make = |mode: StorageMode| {
        let mut c = config(
            "Qh9d4c2s",
            "AQ, KQ, QJs, JTs, 99, 44, T9s, 87s, A5s, KJs",
            "AA, KK, AQ, QT, 99, JJ, TT, 98s, KJs, 65s, AJs",
            600,
            4000,
            PostflopMenu::default(),
        );
        c.storage = mode;
        c
    };
    let opts = SolveOptions { max_iterations: 300, target_exploitability_pct: 0.2, check_every: 20 };
    let mut a = Solver::new(make(StorageMode::Compressed)).unwrap();
    let mut b = Solver::new(make(StorageMode::F32)).unwrap();
    assert!(a.is_compressed() && !b.is_compressed());
    assert!((a.storage_bytes() as f64) < 0.55 * b.storage_bytes() as f64);
    let pa = a.solve(&opts, |_| true);
    let pb = b.solve(&opts, |_| true);
    assert!(pa.exploitability_pct < 0.5, "{pa:?}");
    assert!(pb.exploitability_pct < 0.5, "{pb:?}");
    let (na, nb) = (a.query(&[]).unwrap(), b.query(&[]).unwrap());
    let ra = na.range_strategy();
    let rb = nb.range_strategy();
    for (x, y) in ra.iter().zip(&rb) {
        assert!((x - y).abs() < 0.03, "{ra:?} vs {rb:?}");
    }
}

/// With rake, a chopped pot returns less than the call.
#[test]
fn rake_reduces_chops() {
    let mut cfg = config("AsKsQsJsTs", "22", "33", 1500, 5000, menu_one(&["100%"], &["3x"]));
    cfg.root = BettingState {
        street: Street::River,
        pot: 1500,
        commit: [0, 500],
        stack: [5000, 4500],
        current_bet: 500,
        last_raise_inc: 500,
        num_raises: 1,
        to_act: 0,
        needs: [true, false],
    };
    cfg.rake_rate = 0.05;
    cfg.rake_cap = 300.0;
    let (s, _) = solve(cfg, 200, 0.0);
    let root = s.query(&[]).unwrap();
    let ev = root.hand_ev(0);
    // Pot after the call is 2000, rake 100, each side gets 950.
    assert!(ev[0].abs() < 1e-3);
    assert!((ev[1] - 450.0).abs() < 1.0, "call EV {}", ev[1]);
}

/// Rake on a fold is taken from the matched pot only.
#[test]
fn rake_on_folds() {
    // OOP has the nuts; IP air always folds to a bet.
    let mut cfg = config("Kc8d5h3s2c", "AA", "QJs", 1000, 5000, menu_one(&["100%"], &[]));
    cfg.rake_rate = 0.05;
    cfg.rake_cap = 1e9;
    let (s, _) = solve(cfg, 300, 0.0);
    let root = s.query(&[]).unwrap();
    let aa = root.hand_index(combo("AsAh")).unwrap();
    let ev = root.hand_ev(aa);
    // Betting: IP folds, OOP collects the 1000 pot minus 5% rake (the
    // uncalled bet is returned unraked) = 950. Checking: showdown, same.
    assert!((ev[1] - 950.0).abs() < 2.0, "bet EV {}", ev[1]);
    assert!((ev[0] - 950.0).abs() < 2.0, "check EV {}", ev[0]);
}

#[test]
fn check_schedule_aims_at_predicted_convergence() {
    use crate::next_check_gap as gap;
    assert_eq!(gap(None, (20, 2.0), 0.5, 20), 20);
    // e ∝ 1/t: the target is predicted at t = 80, further than the max gap.
    assert_eq!(gap(Some((20, 2.0)), (40, 1.0), 0.5, 20), 20);
    // Close to the target: check again soon (but not sooner than max/4).
    assert_eq!(gap(Some((20, 1.0)), (40, 0.55), 0.5, 20), 5);
    let g = gap(Some((20, 1.0)), (40, 0.6), 0.5, 20);
    assert!((5..20).contains(&g), "{g}");
    // No progress: fall back to the max gap.
    assert_eq!(gap(Some((20, 1.0)), (40, 1.2), 0.5, 20), 20);
}

