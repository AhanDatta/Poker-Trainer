use super::*;
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
    SolveConfig {
        board,
        ranges: [Range::parse(oop).unwrap().into_weights(), Range::parse(ip).unwrap().into_weights()],
        root: BettingState::street_start(street, pot, [stack, stack], 100),
        menu,
        future_menu: None,
        big_blind: 100,
        chance: ChanceMode::Exact,
        prune: 0.0,
        keep: [vec![], vec![]],
        max_memory_bytes: 4 << 30,
    }
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
