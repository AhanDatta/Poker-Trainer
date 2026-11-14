use super::*;
use crate::spec::{GameSpec, MAX_PLAYERS};
use crate::tree::{LeafKind, Tree};

#[test]
fn context_index_round_trips() {
    let mut seen = [false; Context::COUNT];
    for ctx in Context::all() {
        assert_eq!(Context::from_index(ctx.index()), ctx);
        assert!(!std::mem::replace(&mut seen[ctx.index()], true));
    }
    assert!(seen.iter().all(|&s| s));
}

fn leaf(live: &[usize], raises: u8, aggressor: Option<u8>) -> Leaf {
    Leaf {
        kind: LeafKind::Flop,
        pot: 0,
        rake: 0,
        commit: [0; MAX_PLAYERS],
        live: live.iter().fold(0, |m, &s| m | 1 << s),
        raises,
        aggressor,
    }
}

#[test]
fn contexts_of_typical_pots() {
    // 6-max: BTN (seat 0) opens, BB (seat 2) calls.
    let l = leaf(&[0, 2], 1, Some(0));
    let btn = Context::of(&l, 0, 6);
    let bb = Context::of(&l, 2, 6);
    assert_eq!(
        btn,
        Context {
            pot: PotKind::Raised,
            role: Role::Aggressor,
            position: Position::InPosition,
            multiway: false
        }
    );
    assert_eq!(
        bb,
        Context {
            pot: PotKind::Raised,
            role: Role::Caller,
            position: Position::OutOfPosition,
            multiway: false
        }
    );
    // SB (seat 1) 3-bets a CO (seat 5) open, BB and CO call.
    let l = leaf(&[1, 2, 5], 2, Some(1));
    assert_eq!(Context::of(&l, 1, 6).position, Position::OutOfPosition);
    assert_eq!(Context::of(&l, 2, 6).position, Position::Middle);
    assert_eq!(Context::of(&l, 5, 6).position, Position::InPosition);
    assert!(Context::of(&l, 5, 6).multiway);
    assert_eq!(Context::of(&l, 1, 6).role, Role::Aggressor);
    assert_eq!(Context::of(&l, 5, 6).pot, PotKind::ThreeBet);
    // Heads-up the small blind has the button and acts last.
    let l = leaf(&[0, 1], 0, None);
    assert_eq!(Context::of(&l, 0, 2).position, Position::InPosition);
    assert_eq!(Context::of(&l, 1, 2).position, Position::OutOfPosition);
    assert_eq!(Context::of(&l, 1, 2).pot, PotKind::Limped);
}

/// In every flop leaf of a real tree exactly one player is in position
/// and one out of position, and at most one is the aggressor.
#[test]
fn every_flop_leaf_has_one_player_in_and_one_out_of_position() {
    let mut spec = GameSpec::new(6);
    spec.rules.max_cold_callers = 1;
    let tree = Tree::build(&spec).unwrap();
    let mut flops = 0;
    for l in tree.leaves().filter(|l| l.kind == LeafKind::Flop) {
        flops += 1;
        let ctxs: Vec<Context> = (0..6).filter(|&s| l.is_live(s)).map(|s| Context::of(l, s, 6)).collect();
        let count = |p: Position| ctxs.iter().filter(|c| c.position == p).count();
        assert_eq!((count(Position::InPosition), count(Position::OutOfPosition)), (1, 1));
        assert_eq!(count(Position::Middle), ctxs.len() - 2);
        assert!(ctxs.iter().filter(|c| c.role == Role::Aggressor).count() <= 1);
        assert!(ctxs.iter().all(|c| c.multiway == (ctxs.len() > 2)));
    }
    assert!(flops > 100);
}

#[test]
#[should_panic(expected = "finite")]
fn negative_factors_are_rejected() {
    Realization::neutral().set(Context::from_index(0), [-0.1; NUM_CLASSES]);
}

#[test]
fn context_names_round_trip() {
    let mut names = std::collections::HashSet::new();
    for ctx in Context::all() {
        assert_eq!(Context::parse_name(&ctx.name()), Some(ctx));
        assert!(names.insert(ctx.name()));
    }
    assert_eq!(Context::parse_name("raised aggressor ip"), None);
    assert_eq!(Context::parse_name("raised aggressor ip headsup extra"), None);
    assert_eq!(Context::parse_name("raised boss ip headsup"), None);
}

/// Factors with distinct values in every cell, to catch mixed-up contexts
/// or classes.
fn distinct_factors() -> Realization {
    let mut r = Realization::neutral();
    r.includes_rake = true;
    for ctx in Context::all() {
        let mut f = [0f32; NUM_CLASSES];
        for (c, f) in f.iter_mut().enumerate() {
            let k = ctx.index() * NUM_CLASSES + c;
            *f = format!("{:.4}", 0.5 + k as f64 / 10_000.0).parse().unwrap();
        }
        r.set(ctx, f);
    }
    r
}

#[test]
fn factors_round_trip_through_the_text_format() {
    let r = distinct_factors();
    let text = r.to_text("measured factors\n\nsecond line");
    assert!(text.starts_with("# measured factors\n#\n# second line\nincludes_rake true\n"));
    assert_eq!(Realization::parse(&text), Ok(r));
    let neutral = Realization::neutral();
    assert_eq!(Realization::parse(&neutral.to_text("")), Ok(neutral));
}

#[test]
fn text_keeps_four_decimals() {
    let mut r = Realization::neutral();
    r.set(Context::from_index(5), [0.123_456; NUM_CLASSES]);
    let back = Realization::parse(&r.to_text("")).unwrap();
    assert_eq!(back.factor(5, 100), 0.1235);
}

#[test]
fn bad_factor_files_are_errors() {
    let good = distinct_factors().to_text("");
    let first = Context::from_index(0).name();
    let cases: Vec<(String, &str)> = vec![
        (good.replace("includes_rake true\n", ""), "includes_rake"),
        (good.replace("includes_rake true", "includes_rake maybe"), "true or false"),
        (good.replace("includes_rake true\n", "") + "includes_rake true\n", "before the factors"),
        (format!("includes_rake true\nincludes_rake true\n{}", &good[good.find('[').unwrap()..]), "once"),
        (good.replacen(&format!("[{first}]"), "[raised boss ip headsup]", 1), "unknown context"),
        (good.replacen(&format!("[{first}]\n"), "", 1), "before any"),
        (good.replace("0.5000", "-0.5000"), "at least 0"),
        (good.replace("0.5000", "NaN"), "at least 0"),
        (good.replace("0.5000", "x"), "not a number"),
        (good.replace("0.5000 ", ""), "has 168 factors"),
        (good.replace("0.5000", "0.5000 1.0"), "has 170 factors"),
    ];
    for (text, expected) in cases {
        let err = Realization::parse(&text).unwrap_err();
        assert!(err.contains(expected), "{expected:?} not in {err:?}");
    }
    // A repeated context, and therefore a missing one.
    let second = Context::from_index(1).name();
    let err = Realization::parse(&good.replace(&format!("[{second}]"), &format!("[{first}]"))).unwrap_err();
    assert!(err.contains("twice"), "{err}");
    let last = Context::from_index(Context::COUNT - 1).name();
    let cut = &good[..good.find(&format!("[{last}]")).unwrap()];
    assert!(Realization::parse(cut).unwrap_err().contains("missing"));
    assert!(Realization::parse("").is_err());
}

/// The shipped factors parse, keep the copying rules of the calibration, and
/// look like poker: aces beat 72o, and position helps.
#[test]
fn builtin_factors_parse_and_follow_the_fill_rules() {
    let r = Realization::builtin();
    assert!(!r.includes_rake, "measured against equity times the raked pot");
    let class = |name: &str| name.parse::<holdem::HandClass>().unwrap().grid_index();
    let (aces, trash) = (class("AA"), class("72o"));
    // The file rounds to 4 decimals, so the average is off by at most 1e-4.
    let near = |a: f32, b: f32| (a - b).abs() <= 1.5e-4;
    for ctx in Context::all() {
        let f = r.factors(ctx);
        assert!(f.iter().all(|&x| x.is_finite() && (0.2..4.0).contains(&x)), "{}", ctx.name());
        assert!(f[aces] > f[trash], "{}", ctx.name());
        if ctx.multiway {
            assert_eq!(f, r.factors(Context { multiway: false, ..ctx }), "{}", ctx.name());
        }
        if ctx.pot == PotKind::Limped {
            let raised = Context { pot: PotKind::Raised, role: Role::Caller, ..ctx };
            assert_eq!(f, r.factors(raised), "{}", ctx.name());
        }
        if ctx.position == Position::Middle {
            let ip = r.factors(Context { position: Position::InPosition, ..ctx });
            let oop = r.factors(Context { position: Position::OutOfPosition, ..ctx });
            assert!((0..NUM_CLASSES).all(|c| near(f[c], (ip[c] + oop[c]) / 2.0)), "{}", ctx.name());
        }
        if ctx.position == Position::InPosition {
            let oop = r.factors(Context { position: Position::OutOfPosition, ..ctx });
            let mean = |f: &[f32; NUM_CLASSES]| f.iter().sum::<f32>() / NUM_CLASSES as f32;
            assert!(mean(f) > mean(oop) + 0.05, "{}", ctx.name());
        }
    }
    assert!(Realization::BUILTIN_NAME.contains("6-max"));
}

proptest::proptest! {
    /// Arbitrary text never panics the parser.
    #[test]
    fn parsing_never_panics(text in r"(\[|\]|[a-z0-9 .#\n-]){0,400}") {
        let _ = Realization::parse(&text);
    }

    /// Damaged copies of a good file never panic the parser either.
    #[test]
    fn damaged_files_never_panic(edits in proptest::collection::vec((0usize..1_000_000, r"[\[\]a-z0-9 .\n-]{0,3}"), 1..6)) {
        let mut text = distinct_factors().to_text("x");
        for (at, insert) in edits {
            let at = at % text.len();
            let at = (at..text.len()).find(|&i| text.is_char_boundary(i)).unwrap_or(text.len());
            let end = (at + 2).min(text.len());
            let end = (end..=text.len()).find(|&i| text.is_char_boundary(i)).unwrap_or(text.len());
            text.replace_range(at..end, &insert);
        }
        let _ = Realization::parse(&text);
    }

    /// Factors in the text's precision round-trip exactly.
    #[test]
    fn rounded_factors_round_trip(cells in proptest::collection::vec(0u32..40_000, NUM_CLASSES), ctx in 0..Context::COUNT) {
        let mut r = Realization::neutral();
        let mut f = [0f32; NUM_CLASSES];
        for (f, &c) in f.iter_mut().zip(&cells) {
            *f = format!("{:.4}", c as f32 / 10_000.0).parse().unwrap();
        }
        r.set(Context::from_index(ctx), f);
        proptest::prop_assert_eq!(Realization::parse(&r.to_text("x")), Ok(r));
    }
}
