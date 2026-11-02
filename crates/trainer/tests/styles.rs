//! The AI styles play like the player types they're named after (HUD-style
//! stats from all-AI simulations), and game types seat the right players.

#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use trainer::population::{simulate, StyleStats};
use trainer::preflop::ChartSet;
use trainer::{GameType, Profile, Session, TrainerConfig};

fn table(styles: &[&str], seed: u64) -> BTreeMap<String, StyleStats> {
    let seats: Vec<Profile> = styles.iter().map(|n| Profile::by_name(n).unwrap()).collect();
    simulate(&seats, &ChartSet::builtin(6), 2500, seed)
}

/// Checks every style present against its player type's ranges.
fn check_ranges(st: &BTreeMap<String, StyleStats>) {
    let within = |name: &str, what: &str, v: f64, lo: f64, hi: f64| {
        assert!((lo..=hi).contains(&v), "{name} {what} = {v:.1}, expected {lo}-{hi}");
    };
    // (style, VPIP range, PFR range)
    for (name, vpip, pfr) in [
        ("Reg", (16.0, 28.0), (12.0, 23.0)),
        ("TAG", (14.0, 25.0), (11.0, 21.0)),
        ("LAG", (25.0, 38.0), (19.0, 31.0)),
        ("Nit", (8.0, 16.0), (6.0, 13.0)),
        ("Rock", (11.0, 20.0), (4.0, 11.0)),
        ("Fish", (30.0, 45.0), (5.0, 15.0)),
        ("Calling Station", (42.0, 62.0), (2.0, 13.0)),
        ("Maniac", (45.0, 70.0), (33.0, 55.0)),
    ] {
        let Some(s) = st.get(name) else { continue };
        within(name, "VPIP", s.vpip_pct(), vpip.0, vpip.1);
        within(name, "PFR", s.pfr_pct(), pfr.0, pfr.1);
    }
    if let Some(s) = st.get("Nit") {
        within("Nit", "3-bet", s.three_bet_pct(), 0.0, 6.0);
    }
    if let Some(s) = st.get("LAG") {
        within("LAG", "3-bet", s.three_bet_pct(), 9.0, 18.0);
    }
    if let Some(s) = st.get("Maniac") {
        within("Maniac", "3-bet", s.three_bet_pct(), 15.0, 40.0);
        within("Maniac", "c-bet", s.cbet_pct(), 70.0, 100.0);
        within("Maniac", "AF", s.af(), 3.0, 20.0);
    }
    for name in ["Reg", "TAG", "LAG"] {
        if let Some(s) = st.get(name) {
            within(name, "c-bet", s.cbet_pct(), 50.0, 85.0);
        }
    }
    if let Some(s) = st.get("Calling Station") {
        within("Calling Station", "AF", s.af(), 0.0, 1.0);
    }
}

#[test]
fn aggressive_table_styles_match_player_types() {
    let st = table(&["Reg", "TAG", "LAG", "Nit", "Fish", "Maniac"], 11);
    check_ranges(&st);
    // Orderings that define the player types. (Win rates need far more
    // hands than a test can afford: the population example measures them.)
    let vpip = |n: &str| st[n].vpip_pct();
    let pfr = |n: &str| st[n].pfr_pct();
    assert!(vpip("Fish") > vpip("LAG") && vpip("LAG") > vpip("Reg") && vpip("Reg") > vpip("Nit"));
    assert!(pfr("Maniac") > pfr("LAG") && pfr("LAG") > pfr("Reg") && pfr("Reg") > pfr("Fish"));
    assert!(st["Maniac"].af() > st["Reg"].af());
}

#[test]
fn passive_table_styles_match_player_types() {
    let st = table(&["Reg", "Rock", "Calling Station", "Fish", "TAG", "Nit"], 12);
    check_ranges(&st);
    assert!(st["Calling Station"].vpip_pct() > st["Fish"].vpip_pct());
    assert!(st["Reg"].af() > st["Calling Station"].af());
}

fn session(game: GameType, players: usize, seat_change_pct: f64) -> Session {
    let mut cfg = TrainerConfig::default();
    cfg.players = players;
    cfg.seed = Some(5);
    cfg.game = game;
    cfg.seat_change_pct = seat_change_pct;
    cfg.solver.enabled = false;
    Session::new(cfg).unwrap()
}

fn names(s: &Session) -> Vec<String> {
    s.lineup().iter().flatten().map(|p| p.name.clone()).collect()
}

#[test]
fn game_types_seat_the_right_players() {
    let mut s = session(GameType::Tough, 9, 100.0);
    for _ in 0..50 {
        let _ = s.new_hand();
        for n in names(&s) {
            assert!(["Reg", "TAG", "LAG", "Nit"].contains(&n.as_str()), "tough game seated a {n}");
        }
    }
    // Custom table: exact counts (seats beyond them filled like a typical game).
    let mut s = session(GameType::Custom, 6, 100.0);
    s.config.custom_table = [("Maniac".to_string(), 3), ("Fish".to_string(), 2)].into_iter().collect();
    s.set_game(GameType::Custom);
    for _ in 0..20 {
        let _ = s.new_hand();
        let n = names(&s);
        assert_eq!(n.iter().filter(|x| *x == "Maniac").count(), 3, "{n:?}");
        assert_eq!(n.iter().filter(|x| *x == "Fish").count(), 2, "{n:?}");
    }
}

#[test]
fn lineups_persist_and_change_as_configured() {
    // Never: the same players all session.
    let mut s = session(GameType::Typical, 6, 0.0);
    let _ = s.new_hand();
    let first = names(&s);
    for _ in 0..100 {
        let _ = s.new_hand();
        assert_eq!(names(&s), first);
    }
    // Sometimes (2%): players come and go, but most hands keep the lineup.
    let mut s = session(GameType::Typical, 6, 2.0);
    let mut prev = {
        let _ = s.new_hand();
        names(&s)
    };
    let (mut same, mut notes) = (0, 0);
    for _ in 0..300 {
        let h = s.new_hand();
        notes += h.log.iter().filter(|l| l.text.contains("new player sat down")).count();
        let now = names(&s);
        if now == prev {
            same += 1;
        }
        prev = now;
    }
    assert!(same > 200, "lineup changed in {} of 300 hands", 300 - same);
    assert!((10..=60).contains(&notes), "{notes} seat changes in 300 hands (expected ~30)");
}
