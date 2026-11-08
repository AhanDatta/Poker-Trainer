use super::*;
use crate::config::TrainerConfig;
use crate::fmt::BB;
use crate::profiles::Profile;
use crate::review::SpotInfo;
use crate::save::{SavedReview, SessionSummary, FORMAT_VERSION};
use crate::session::Session;
use crate::solver_tag::{SolvedBy, SolverTag};
use crate::StepResult;
use holdem::card::parse_cards;
use holdem::{Action, HandState, Street, TableConfig};
use proptest::prelude::*;

/// RFC 4180 reader (fields may be quoted; quotes doubled inside).
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            (true, '"') => quoted = false,
            (true, c) => field.push(c),
            (false, '"') => quoted = true,
            (false, ',') => record.push(std::mem::take(&mut field)),
            (false, '\n') => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            (false, c) => field.push(c),
        }
    }
    assert!(field.is_empty() && record.is_empty() && !quoted, "records end with a newline");
    records
}

fn csv_text(records: &[Vec<String>]) -> String {
    let mut buf = Vec::new();
    for r in records {
        write_record(&mut buf, r).unwrap();
    }
    String::from_utf8(buf).unwrap()
}

#[test]
fn fields_are_escaped() {
    assert_eq!(csv_field("plain"), "plain");
    assert_eq!(csv_field(""), "");
    assert_eq!(csv_field("a,b"), "\"a,b\"");
    assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    assert_eq!(csv_field("two\nlines"), "\"two\nlines\"");
    assert_eq!(csv_field("cr\r"), "\"cr\r\"");
    assert_eq!(
        csv_text(&[vec!["BB Check · SB Bet 2bb (33%)".into(), "x,y".into()]]),
        "BB Check · SB Bet 2bb (33%),\"x,y\"\n"
    );
}

#[test]
fn numbers_are_trimmed() {
    assert_eq!(num(1.5, 2), "1.5");
    assert_eq!(num(2.0, 2), "2");
    assert_eq!(num(100.0, 2), "100");
    assert_eq!(num(100.0, 0), "100");
    assert_eq!(num(-0.0001, 2), "0");
    assert_eq!(num(-12.346, 2), "-12.35");
    assert_eq!(signed(0.0, 1), "0.0");
    assert_eq!(signed(-0.01, 1), "0.0");
    assert_eq!(signed(2.26, 1), "+2.3");
    assert_eq!(signed(-3.0, 2), "-3.00");
}

#[test]
fn format_labels() {
    let f = |players, stack| Format { players, stack, big_blind: 100 };
    assert_eq!(f(2, 10_000).to_string(), "HU 100bb");
    assert_eq!(f(6, 10_000).to_string(), "6-max 100bb");
    assert_eq!(f(9, 4_050).to_string(), "9-max 40.5bb");
}

// ------------------------------------------------------ fixture hands ---

fn holes(n: usize) -> Vec<[Card; 2]> {
    let cards = parse_cards("AsAhKsKhQsQhJsJhTsTh9s9h8s8h7s7h6s6h5s5h").unwrap();
    (0..n).map(|i| [cards[2 * i], cards[2 * i + 1]]).collect()
}

/// Heads-up, hero in the SB (seat 0, AA) vs BB (KK): all-in preflop and a
/// runout when `showdown`, else the hero folds.
fn hu_state(showdown: bool) -> HandState {
    let mut s = HandState::new(TableConfig::new(2, 50, 100), 0, vec![10_000, 10_000], holes(2));
    if showdown {
        s.apply(Action::Raise(10_000)).unwrap();
        s.apply(Action::Call).unwrap();
        s.deal_board(&parse_cards("2c3c4d").unwrap()).unwrap();
        s.deal_board(&parse_cards("8d").unwrap()).unwrap();
        s.deal_board(&parse_cards("9c").unwrap()).unwrap();
    } else {
        s.apply(Action::Fold).unwrap();
    }
    s
}

fn decision(variant: Variant, ev_loss: f32, verdict: Verdict) -> DecisionReview {
    DecisionReview {
        variant,
        street: Street::Flop,
        board: parse_cards("2c3c4d").unwrap(),
        history: "BB Check · SB Bet 2bb (33%)".into(),
        pot: 600,
        to_call: 200,
        hero_combo: 0,
        actions: vec![Action::Fold, Action::Call],
        labels: vec!["Fold".into(), "Call 2bb".into()],
        chosen: 0,
        best: 1,
        hero_strategy: vec![0.2, 0.8],
        hero_ev: vec![0.0, ev_loss],
        ev_loss,
        ev_loss_pct: 100.0 * ev_loss / 600.0,
        verdict,
        range_strategy: vec![0.3, 0.7],
        grid: Vec::new(),
        exploitability_pct: 0.4,
        notes: vec!["a note, with a comma".into()],
        spot: SpotInfo::default(),
        solver: SolverTag { by: SolvedBy::GpuFast, gpu_fallback: None },
    }
}

fn saved(id: u64, state: HandState, decisions: Vec<DecisionReview>) -> SavedHand {
    SavedHand {
        id,
        hero: 0,
        state,
        profiles: vec![None, Profile::by_name("Reg")],
        preflop: Vec::new(),
        log: Vec::new(),
        roll_seed: 0,
        review: Some(SavedReview {
            stage: "done".into(),
            decisions,
            notes: Vec::new(),
            cancelled: false,
            error: None,
        }),
    }
}

fn meta(id: &str) -> SessionMeta {
    SessionMeta {
        version: FORMAT_VERSION,
        id: id.into(),
        created: 1_793_712_300,
        updated: 1_793_712_300,
        config: TrainerConfig::default(),
        lineup: vec![None, Profile::by_name("Reg")],
        button: 0,
        next_id: 3,
        summary: SessionSummary::default(),
    }
}

fn fixture() -> LoadedSession {
    let both = vec![
        decision(Variant::Gto, 50.0, Verdict::Mistake),
        decision(Variant::Actual, 300.0, Verdict::Blunder),
        decision(Variant::Gto, 10.0, Verdict::Good),
    ];
    LoadedSession {
        meta: meta("2026-11-03-13-25"),
        // Out of order on purpose: rows come out in hand order.
        hands: vec![saved(2, hu_state(false), Vec::new()), saved(1, hu_state(true), both)],
        skipped: 0,
    }
}

#[test]
fn hand_rows_from_saved_hands() {
    let s = fixture();
    let rows = hand_rows(&s);
    assert_eq!(rows.iter().map(|r| r.hand).collect::<Vec<_>>(), [1, 2]);
    let (won, folded) = (&rows[0], &rows[1]);
    assert_eq!(won.format, Format { players: 2, stack: 10_000, big_blind: 100 });
    assert_eq!(won.position, "SB");
    assert_eq!(won.hero_cards, "AsAh");
    assert_eq!(won.board, "2c 3c 4d 8d 9c");
    assert_eq!(won.opponents, "BB Reg");
    assert_eq!((won.net_chips, won.showdown), (10_000, true));
    // GTO decisions count; the actual-range one doesn't.
    assert_eq!((won.decisions, won.ev_loss_chips, won.worst), (2, 60.0, Some(Verdict::Mistake)));
    assert_eq!((folded.net_chips, folded.showdown, folded.decisions, folded.worst), (-50, false, 0, None));

    let f = won.fields();
    assert_eq!(f.len(), HAND_COLUMNS.len());
    let col = |name: &str| HAND_COLUMNS.iter().position(|c| *c == name).unwrap();
    assert_eq!(f[col("session_start_utc")], "2026-11-03 13:25");
    assert_eq!(f[col("format")], "HU");
    assert_eq!(f[col("stack_bb")], "100");
    assert_eq!(f[col("net_bb")], "100");
    assert_eq!(f[col("ev_loss_bb")], "0.6");
    assert_eq!(f[col("worst_verdict")], "Mistake");
    assert_eq!(folded.fields()[col("net_bb")], "-0.5");
    assert_eq!(folded.fields()[col("worst_verdict")], "");
}

#[test]
fn review_states() {
    let mut h = saved(1, hu_state(false), Vec::new());
    let m = meta("x");
    assert_eq!(HandRow::new(&m, &h).review, "done");
    h.review.as_mut().unwrap().cancelled = true;
    assert_eq!(HandRow::new(&m, &h).review, "cancelled");
    h.review.as_mut().unwrap().error = Some("boom".into());
    assert_eq!(HandRow::new(&m, &h).review, "error");
    h.review = None;
    assert_eq!(HandRow::new(&m, &h).review, "none");
}

#[test]
fn decision_rows_mark_counted_variant() {
    let s = fixture();
    let rows = decision_rows(&s);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter().map(|r| (r.index, r.counted)).collect::<Vec<_>>(),
        [(1, true), (2, false), (3, true)]
    );
    let f = rows[0].fields();
    assert_eq!(f.len(), DECISION_COLUMNS.len());
    let col = |name: &str| DECISION_COLUMNS.iter().position(|c| *c == name).unwrap();
    assert_eq!(f[col("variant")], "gto");
    assert_eq!(rows[1].fields()[col("variant")], "actual");
    assert_eq!(f[col("pot_bb")], "6");
    assert_eq!(f[col("to_call_bb")], "2");
    assert_eq!((f[col("chosen")].as_str(), f[col("best")].as_str()), ("Fold", "Call 2bb"));
    assert_eq!(f[col("ev_loss_bb")], "0.5");
    assert_eq!(f[col("solver")], "GPU (fast)");
    // An actual-range-only review counts every decision.
    let only_actual = vec![decision(Variant::Actual, 1.0, Verdict::Best)];
    let s2 =
        LoadedSession { meta: meta("y"), hands: vec![saved(1, hu_state(true), only_actual)], skipped: 0 };
    assert!(decision_rows(&s2)[0].counted);
    assert_eq!(hand_rows(&s2)[0].decisions, 1);
}

/// Hands saved before reviews recorded their solver and spot still load
/// and export (the fields default).
#[test]
fn old_saves_export() {
    let s = fixture();
    let mut lines = String::new();
    for h in &s.hands {
        let mut v = serde_json::to_value(h).unwrap();
        for d in v["review"]["decisions"].as_array_mut().unwrap() {
            let d = d.as_object_mut().unwrap();
            assert!(d.remove("solver").is_some() && d.remove("spot").is_some());
        }
        lines += &format!("{v}\n");
    }
    let root = std::env::temp_dir().join(format!("poker-trainer-test-{}-old-export", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let store = SessionStore::create(&root, s.meta.created).unwrap();
    store.write_meta(&s.meta).unwrap();
    fs::write(store.dir().join("hands.jsonl"), lines).unwrap();
    let loaded = store.load().unwrap();
    assert_eq!((loaded.hands.len(), loaded.skipped), (2, 0));
    let mut want = fixture();
    want.meta.id = store.id.clone();
    assert_eq!(hand_rows(&loaded), hand_rows(&want));
    let col = DECISION_COLUMNS.iter().position(|c| *c == "solver").unwrap();
    assert!(decision_rows(&loaded).iter().all(|r| r.fields()[col] == "CPU"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn export_writes_both_files() {
    let root = std::env::temp_dir().join(format!("poker-trainer-test-{}-export", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let s = fixture();
    let a = SessionStore::create(&root.join("sessions"), 2_000).unwrap();
    let b = SessionStore::create(&root.join("sessions"), 1_000).unwrap();
    for (store, created) in [(&a, 2_000), (&b, 1_000)] {
        let mut m = s.meta.clone();
        m.id = store.id.clone();
        m.created = created;
        store.write_meta(&m).unwrap();
        store.append_hands(&s.hands).unwrap();
    }
    let out = root.join("out");
    let report = export(&root.join("sessions"), None, &out).unwrap();
    assert_eq!((report.sessions, report.hands, report.decisions), (2, 4, 6));
    assert!(report.unreadable.is_empty());
    assert_eq!(
        report.totals,
        vec![FormatTotals { format: hand_rows(&s)[0].format, hands: 4, net_chips: 2 * 9_950 }]
    );
    let hands = parse_csv(&fs::read_to_string(out.join(HANDS_FILE)).unwrap());
    assert_eq!(hands[0], HAND_COLUMNS);
    // Oldest session first.
    assert_eq!(hands[1][0], b.id);
    assert_eq!(hands[3][0], a.id);
    let decisions = parse_csv(&fs::read_to_string(out.join(DECISIONS_FILE)).unwrap());
    assert_eq!(decisions.len(), 7);
    assert!(decisions.iter().all(|r| r.len() == DECISION_COLUMNS.len()));
    assert!(!out.join(format!("{HANDS_FILE}.tmp")).exists());

    // One session; the files are replaced.
    let report = export(&root.join("sessions"), Some(&a.id), &out).unwrap();
    assert_eq!((report.sessions, report.hands), (1, 2));
    assert_eq!(parse_csv(&fs::read_to_string(out.join(HANDS_FILE)).unwrap()).len(), 3);
    assert!(export(&root.join("sessions"), Some("../x"), &out).is_err());
    assert!(export(&root.join("sessions"), Some("missing"), &out).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn summary_table_lists_formats_and_total() {
    let f = |players| Format { players, stack: 10_000, big_blind: 100 };
    let totals = vec![
        FormatTotals { format: f(2), hands: 200, net_chips: 1_000 },
        FormatTotals { format: f(6), hands: 50, net_chips: -2_550 },
    ];
    let t = summary_table(&totals);
    let lines: Vec<&str> = t.lines().collect();
    assert_eq!(lines.len(), 4);
    assert!(lines[1].starts_with("HU 100bb") && lines[1].ends_with("+5.0"), "{t}");
    assert!(
        lines[2].starts_with("6-max 100bb") && lines[2].contains("-25.50") && lines[2].ends_with("-51.0")
    );
    assert!(lines[3].starts_with("All") && lines[3].contains("250") && lines[3].ends_with("-6.2"), "{t}");
    assert_eq!(summary_table(&totals[..1]).lines().count(), 2, "no total for one format");
    assert_eq!(bb_per_100(5.0, 0), 0.0);
}

// --------------------------------------------------------- properties ---

/// Plays `hands` hands (the table size changes halfway) with a hero that
/// picks actions from `seed`; reviews are off so hands finish at once.
fn play_session(seed: u64, players: [usize; 2], hands: usize) -> Session {
    let mut cfg = TrainerConfig { seed: Some(seed), players: players[0], ..TrainerConfig::default() };
    cfg.solver.enabled = false;
    let mut session = Session::new(cfg).unwrap();
    let mut rng = seed | 1;
    for i in 0..hands {
        if i == hands / 2 {
            session.set_players(players[1]).unwrap();
        }
        let mut hand = session.new_hand();
        while hand.run_until_hero() == StepResult::HeroToAct {
            let acts = hand.hero_actions();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            hand.hero_act(acts[(rng % acts.len() as u64) as usize]).unwrap();
        }
        session.finish_hand(hand);
    }
    session.refresh();
    session
}

fn arb_field() -> impl Strategy<Value = String> {
    prop_oneof![
        "[a-z0-9 .%·()+-]{0,12}",
        proptest::collection::vec(
            prop_oneof![Just(','), Just('"'), Just('\n'), Just('\r'), Just('a'), Just(' ')],
            0..8
        )
        .prop_map(|v| v.into_iter().collect()),
        any::<String>(),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Any records survive writing and reading back.
    #[test]
    fn csv_round_trips(records in proptest::collection::vec(proptest::collection::vec(arb_field(), 1..6), 0..6)) {
        prop_assert_eq!(parse_csv(&csv_text(&records)), records);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// The export's results (rows, per-format totals, bb/100 and the CSV
    /// itself) match the session's own stats.
    #[test]
    fn export_matches_session_stats(seed in any::<u64>(), a in 2usize..=9, b in 2usize..=9, hands in 1usize..30) {
        let session = play_session(seed, [a, b], hands);
        let stats = session.stats();
        let loaded = LoadedSession {
            meta: session.meta(),
            hands: session.history.iter().map(|h| h.to_saved()).collect(),
            skipped: 0,
        };
        let rows = hand_rows(&loaded);
        prop_assert_eq!(rows.len(), stats.hands);
        for (r, h) in rows.iter().zip(&session.history) {
            prop_assert_eq!(r.hand, h.id);
            prop_assert_eq!(Some(r.net_chips), h.hero_net());
            prop_assert_eq!(r.format.players, h.state.num_players());
            prop_assert_eq!(r.position, h.state.position_name(h.hero));
            prop_assert!(!r.showdown || !h.hero_folded());
            prop_assert_eq!(r.review, "none");
        }
        let totals = totals_by_format(&rows);
        let expected_formats = if a == b || hands < 2 { 1 } else { 2 };
        prop_assert_eq!(totals.len(), expected_formats);
        let (n, net_bb) = overall(&totals);
        prop_assert_eq!(n, stats.hands);
        prop_assert_eq!(totals.iter().map(|t| t.net_chips).sum::<Chips>(), stats.net_chips);
        let in_app = 100.0 * stats.net_chips as f64 / BB as f64 / stats.hands as f64;
        prop_assert!((bb_per_100(net_bb, n) - in_app).abs() < 1e-9);
        // The CSV says the same.
        let mut buf = Vec::new();
        for r in &rows {
            write_record(&mut buf, &r.fields()).unwrap();
        }
        let csv = parse_csv(&String::from_utf8(buf).unwrap());
        let col = |name: &str| HAND_COLUMNS.iter().position(|c| *c == name).unwrap();
        let chips: Chips = csv.iter().map(|r| r[col("net_chips")].parse::<Chips>().unwrap()).sum();
        prop_assert_eq!(chips, stats.net_chips);
        let bbs: f64 = csv.iter().map(|r| r[col("net_bb")].parse::<f64>().unwrap()).sum();
        prop_assert!((bbs - stats.net_chips as f64 / BB as f64).abs() < 1e-6);
        prop_assert!(decision_rows(&loaded).is_empty());
    }
}
