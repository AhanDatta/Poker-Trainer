use super::measure::{check_only_menu, solve_ranges, SeatSums, SolveInfo};
use super::spots::walk_leaves;
use super::*;
use crate::charts::compare;
use crate::realize::{Context, Position, PotKind, Role};
use crate::spec::GameSpec;
use crate::tree::{Node, Tree};
use holdem::combo::COMBO_CLASS;
use holdem::{combo_cards, evaluate_cards, HandClass, PostflopMenu, StreetSizes, NUM_CLASSES, NUM_COMBOS};
use proptest::prelude::*;
use std::sync::OnceLock;
use trainer::preflop::{line_range, ChartSet, PreDecision};

fn spec() -> GameSpec {
    let mut spec = GameSpec::new(6).with_rake(0.05, 3.0);
    spec.rules.max_cold_callers = 1;
    spec
}

/// The 6-max tree and its heads-up flop spots under the built-in charts.
fn six_max() -> &'static (Tree, Vec<Spot>) {
    static CELL: OnceLock<(Tree, Vec<Spot>)> = OnceLock::new();
    CELL.get_or_init(|| {
        let tree = Tree::build(&spec()).unwrap();
        let spots = heads_up_flops(&tree, &ChartSet::builtin(6));
        (tree, spots)
    })
}

fn class(s: &str) -> usize {
    s.parse::<HandClass>().unwrap().grid_index()
}

fn cards(s: &str) -> Vec<holdem::Card> {
    holdem::card::parse_cards(s).unwrap()
}

/// A small postflop game that solves in a blink on the CPU.
fn quick_setup() -> PostflopSetup {
    let half = StreetSizes::new(&["50%"], &[]);
    let mut setup = PostflopSetup::review(&spec());
    setup.menu = PostflopMenu {
        flop: half.clone(),
        turn: half.clone(),
        river: half,
        max_raises: 1,
        add_allin_threshold: 0.0,
        ..PostflopMenu::default()
    };
    setup.future_menu = None;
    // Exact runouts: tiny samples mis-weight the hands that hold a sampled
    // card (values below 0 or above the pot).
    setup.chance = Chance::Exact;
    setup.max_iterations = 30;
    setup.target_exploitability_pct = 1.0;
    setup
}

// ---- spots ----

#[test]
fn the_reach_of_all_leaves_adds_up_to_one() {
    let (tree, _) = six_max();
    let mut total = 0.0;
    let mut leaves = 0;
    walk_leaves(tree, &ChartSet::builtin(6), |v| {
        assert!(v.reach >= 0.0);
        total += v.reach;
        leaves += 1;
    });
    assert!((total - 1.0).abs() < 1e-6, "{total}");
    assert!(leaves > 500, "{leaves}");
}

#[test]
fn spots_are_heads_up_flops_with_their_contexts() {
    let (tree, spots) = six_max();
    assert!(spots.len() > 50);
    assert!(spots.windows(2).all(|w| w[0].reach >= w[1].reach));
    for s in spots {
        let Node::Leaf(leaf) = &tree.nodes[s.leaf as usize] else { panic!("{} is not a leaf", s.line) };
        assert_eq!(leaf.num_live(), 2, "{}", s.line);
        assert!(s.seats.iter().all(|&seat| leaf.is_live(seat)));
        assert!(s.reach > 0.0);
        assert_eq!(s.contexts, s.seats.map(|seat| Context::of(leaf, seat, 6)));
        assert_eq!(s.contexts.map(|c| c.position), [Position::OutOfPosition, Position::InPosition]);
        assert!(s.contexts.iter().all(|c| !c.multiway && c.pot != PotKind::Limped));
        assert_eq!(s.contexts.iter().filter(|c| c.role == Role::Aggressor).count(), 1, "{}", s.line);
        assert_eq!(s.pot, leaf.pot);
        assert_eq!(s.behind, spec().stack() - leaf.commit[s.seats[1]]);
        assert!(s.kind().is_some());
    }
    let first = &spots[0];
    assert_eq!(first.line, "BTN:R BB:C");
    // The button is seat 0 and in position; the big blind is seat 2.
    assert_eq!(first.seats, [2, 0]);
    assert_eq!(first.contexts[1].role, Role::Aggressor);
    let lines: std::collections::HashSet<&str> = spots.iter().map(|s| s.line.as_str()).collect();
    assert_eq!(lines.len(), spots.len(), "lines identify spots");
}

/// The ranges along a spot's line are the trainer's own `line_range` with
/// the same charts.
#[test]
fn spot_ranges_are_the_trainers_line_ranges() {
    let (tree, spots) = six_max();
    let charts = ChartSet::builtin(6);
    for s in pick(spots, 3) {
        let decisions: Vec<PreDecision> = s
            .path
            .iter()
            .map(|&(node, action)| {
                let Node::Decision(d) = &tree.nodes[node as usize] else { panic!("a decision") };
                PreDecision {
                    seat: d.seat as usize,
                    position: holdem::game::position_name(6, d.seat as usize).to_string(),
                    key: d.key.clone(),
                    avail: d.avail.clone(),
                    action,
                }
            })
            .collect();
        for (p, &seat) in s.seats.iter().enumerate() {
            let (range, uncovered) =
                line_range(&decisions, seat, |d| trainer::preflop::chart_policy(&charts, d));
            assert!(uncovered.is_empty());
            for c in 0..NUM_COMBOS {
                let ours = s.classes[p][COMBO_CLASS[c] as usize];
                assert!((range.weight(c) - ours).abs() < 1e-6, "{} seat {seat} combo {c}", s.line);
            }
        }
    }
}

/// A spot's reach is the reach of its last decision (as the chart
/// comparison computes it) times the range's frequency of the last action.
#[test]
fn spot_reach_agrees_with_the_chart_comparison() {
    let (tree, spots) = six_max();
    let charts = ChartSet::builtin(6);
    let diffs = compare(tree, &charts, &charts);
    for s in pick(spots, 2) {
        let &(node, action) = s.path.last().unwrap();
        let Node::Decision(d) = &tree.nodes[node as usize] else { panic!("a decision") };
        let diff = diffs.iter().find(|x| x.key == d.key).unwrap();
        let expected = diff.reach as f64 * diff.reference[action.index()] as f64;
        assert!(
            (s.reach - expected).abs() < 1e-5 * expected.max(1e-3),
            "{}: {} vs {expected}",
            s.line,
            s.reach
        );
    }
}

#[test]
fn picking_keeps_the_most_reached_spots_of_each_type() {
    let (_, spots) = six_max();
    let picked = pick(spots, 4);
    assert_eq!(picked.len(), 24);
    for (i, t) in SpotType::all().into_iter().enumerate() {
        let of_type: Vec<&Spot> = spots.iter().filter(|s| s.kind() == Some(t)).collect();
        let chosen = &picked[4 * i..4 * i + 4];
        assert!(chosen.iter().all(|s| s.kind() == Some(t)));
        assert_eq!(
            chosen.iter().map(|s| &s.line).collect::<Vec<_>>(),
            of_type[..4].iter().map(|s| &s.line).collect::<Vec<_>>()
        );
    }
    assert!(pick(spots, 0).is_empty());
    assert_eq!(pick(spots, 1000).len(), spots.len());
    let names: std::collections::HashSet<String> = SpotType::all().iter().map(|t| t.name()).collect();
    assert_eq!(names.len(), 6);
}

// ---- ranges and solves ----

#[test]
fn solve_ranges_floor_every_class_and_drop_board_cards() {
    let (_, spots) = six_max();
    let board = cards("AsKd7h");
    let dead = holdem::CardSet::from_cards(&board);
    for s in pick(spots, 1) {
        let ranges = solve_ranges(&s, &board, 0.01);
        for p in 0..2 {
            let max = s.classes[p].iter().copied().fold(0f32, f32::max);
            for c in 0..NUM_COMBOS {
                let (a, b) = combo_cards(c);
                let w = ranges[p][c];
                if dead.contains(a) || dead.contains(b) {
                    assert_eq!(w, 0.0);
                } else {
                    let chart = s.classes[p][COMBO_CLASS[c] as usize];
                    assert_eq!(w, chart.max(0.01 * max), "{} {c}", s.line);
                    assert!(w > 0.0);
                }
            }
        }
        // Without a floor, classes the charts never play stay out.
        let bare = solve_ranges(&s, &board, 0.0);
        let c72 = HandClass::from_grid_index(class("72o")).combos()[0];
        assert_eq!(bare[0][c72], s.classes[0][class("72o")]);
    }
}

/// Every hand's check-down value is its equity against the other range
/// times the pot after rake, computed here by enumerating the runouts.
#[test]
fn check_down_values_are_equity_times_the_raked_pot() {
    let mut classes = [[0f32; NUM_CLASSES]; 2];
    classes[0][class("AA")] = 1.0;
    classes[1][class("KK")] = 1.0;
    classes[1][class("AKs")] = 0.5;
    let spot = Spot {
        line: "test".into(),
        leaf: 0,
        path: Vec::new(),
        reach: 0.25,
        seats: [2, 0],
        contexts: [Context::from_index(10), Context::from_index(8)],
        classes,
        pot: 550,
        behind: 9750,
    };
    let board = cards("Kh7c2d");
    let mut setup = PostflopSetup::review(&spec());
    setup.floor = 0.0;
    let (sums, _) = measure(&spot, &board, &setup.check_down()).unwrap();
    let ranges = solve_ranges(&spot, &board, 0.0);
    let prize = 550.0 - (550.0f64 * 0.05).min(300.0);
    let ranges = &ranges;
    let live = |p: usize| (0..NUM_COMBOS).filter(move |&c| ranges[p][c] > 0.0);
    for p in 0..2 {
        let mut expected = vec![0f64; NUM_CLASSES];
        for h in live(p) {
            let (a, b) = combo_cards(h);
            let (mut share, mut weight) = (0.0, 0.0);
            for o in live(1 - p) {
                let (c, d) = combo_cards(o);
                if [c, d].iter().any(|x| *x == a || *x == b) {
                    continue;
                }
                let used = holdem::CardSet::from_cards(&[board[0], board[1], board[2], a, b, c, d]);
                let rest: Vec<_> = holdem::Card::all().filter(|x| !used.contains(*x)).collect();
                let (mut won, mut runouts) = (0.0, 0.0);
                for i in 0..rest.len() {
                    for j in i + 1..rest.len() {
                        let full = [board[0], board[1], board[2], rest[i], rest[j]];
                        let mine = evaluate_cards(&[&full[..], &[a, b]].concat());
                        let theirs = evaluate_cards(&[&full[..], &[c, d]].concat());
                        won += if mine > theirs {
                            1.0
                        } else if mine == theirs {
                            0.5
                        } else {
                            0.0
                        };
                        runouts += 1.0;
                    }
                }
                share += ranges[1 - p][o] as f64 * won / runouts;
                weight += ranges[1 - p][o] as f64;
            }
            expected[COMBO_CLASS[h] as usize] += share / weight * prize;
        }
        for c in 0..NUM_CLASSES {
            let got = sums[p].check_down[c];
            assert!(
                (got - expected[c]).abs() < 1e-3 * prize,
                "player {p} class {c}: {got} vs {}",
                expected[c]
            );
        }
    }
    // Each combo weighs the reach times its share of the range.
    assert_eq!(sums[0].combos[class("AA")], 6);
    assert_eq!(sums[1].combos[class("KK")], 3);
    assert!((sums[1].weight[class("KK")] - 0.25 * 1.0 / 4.5).abs() < 1e-12);
    assert!((sums[1].weight[class("AKs")] - 0.25 * 0.5 / 4.5).abs() < 1e-12);
}

#[test]
fn measured_sums_are_consistent() {
    let (_, spots) = six_max();
    let spot = &pick(spots, 1)[0];
    let board = cards("Qs8h3c");
    let (sums, info) = measure(spot, &board, &quick_setup()).unwrap();
    assert!(info.iterations > 0 && info.exploitability_pct.is_finite());
    assert_eq!(info.solved_on, "Cpu");
    let dead = holdem::CardSet::from_cards(&board);
    for s in &sums {
        let mut live = [0u32; NUM_CLASSES];
        for c in (0..NUM_COMBOS).filter(|&c| {
            let (a, b) = combo_cards(c);
            !dead.contains(a) && !dead.contains(b)
        }) {
            live[COMBO_CLASS[c] as usize] += 1;
        }
        assert_eq!(s.combos, live, "every live combo of every class is solved");
        let total: f64 = s.weight.iter().zip(&s.combos).map(|(w, &n)| w * n as f64).sum();
        assert!((total - spot.reach).abs() < 1e-9 * spot.reach);
        assert!(s.value.iter().chain(&s.check_down).all(|v| v.is_finite()));
        assert!(s.check_down.iter().all(|&v| v > 0.0));
        // Checking down a hand never wins more than the whole pot.
        for c in 0..NUM_CLASSES {
            assert!(s.check_down[c] <= s.combos[c] as f64 * spot.pot as f64 + 1e-6);
        }
    }
    // Aces win more than their equity share with bets; 72o less.
    let r = aggregate(&[Record::new(spot, "Qs8h3c", &sums, &info)]).unwrap();
    let ip = spot.contexts[1].index();
    assert!(r.realization.factor(ip, class("AA")) > 1.0);
}

#[test]
fn bad_flops_are_errors() {
    let (_, spots) = six_max();
    let spot = &spots[0];
    for board in ["AsKd", "AsKdAs", "AsKd7h2c"] {
        assert!(measure(spot, &cards(board), &quick_setup()).is_err(), "{board}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 6, ..ProptestConfig::default() })]

    /// With checks only, betting wins exactly what checking down does: every
    /// measured factor is 1.
    #[test]
    fn a_check_only_menu_gives_factors_of_one(spot in 0usize..24, seed in 0u64..1000) {
        let (_, spots) = six_max();
        let picked = pick(spots, 4);
        let spot = &picked[spot];
        let flop = flops(seed, &spot.line, 1)[0];
        let mut setup = quick_setup();
        setup.menu = check_only_menu();
        let (sums, info) = measure(spot, &flop, &setup).unwrap();
        for s in &sums {
            prop_assert_eq!(&s.value, &s.check_down);
        }
        let r = aggregate(&[Record::new(spot, &flop_name(&flop), &sums, &info)]).unwrap();
        for ctx in spot.contexts {
            prop_assert!(r.realization.factors(ctx).iter().all(|&f| f == 1.0), "{:?}", r.realization.factors(ctx));
        }
    }
}

// ---- flops, plans and checkpoints ----

#[test]
fn flops_are_reproducible_and_extend() {
    let a = flops(7, "BTN:R BB:C", 30);
    assert_eq!(a, flops(7, "BTN:R BB:C", 30));
    assert_eq!(a[..10], flops(7, "BTN:R BB:C", 10)[..]);
    assert_ne!(a, flops(8, "BTN:R BB:C", 30));
    assert_ne!(a, flops(7, "CO:R BB:C", 30));
    for f in &a {
        assert_eq!(holdem::CardSet::from_cards(f).len(), 3);
    }
    assert_eq!(flop_name(&cards("As7d2c")), "As7d2c");
}

#[test]
fn plans_go_flop_by_flop_and_resume_skips_finished_solves() {
    let (_, spots) = six_max();
    let spots = pick(spots, 1);
    let jobs = plan(&spots, 3, 5);
    assert_eq!(jobs.len(), 18);
    for (i, job) in jobs.iter().enumerate() {
        assert_eq!(job.spot, i % 6);
        assert_eq!(job.flop, flops(5, &spots[job.spot].line, 3)[i / 6]);
    }
    // Run the first three with checks only (fast), as the runner does.
    let mut setup = quick_setup();
    setup.menu = check_only_menu();
    let mut saved = Vec::new();
    let n = run(
        &spots,
        &jobs[..3],
        &setup,
        |r| {
            saved.push(r.clone());
            Ok(())
        },
        |_, why| panic!("{why}"),
    )
    .unwrap();
    assert_eq!(n, 3);
    let text: String = saved.iter().map(|r| r.to_line() + "\n").collect();
    let (back, complete) = parse_checkpoint(&text).unwrap();
    assert_eq!((back.clone(), complete), (saved.clone(), text.len()));
    let left = pending(&spots, jobs.clone(), &back);
    assert_eq!(left, jobs[3..].to_vec());
    // A failing save stops the run.
    let err = run(&spots, &jobs[3..5], &setup, |_| Err("disk full".into()), |_, _| ()).unwrap_err();
    assert_eq!(err, "disk full");
    // A failing solve is reported and skipped.
    let bad = [Job { spot: 0, flop: [jobs[0].flop[0]; 3] }, jobs[3].clone()];
    let mut failures = 0;
    let n = run(&spots, &bad, &setup, |_| Ok(()), |_, _| failures += 1).unwrap();
    assert_eq!((n, failures), (1, 1));
}

fn sample_record(contexts: [Context; 2], value: f64, check_down: f64) -> Record {
    let sums = [0, 1].map(|_| SeatSums {
        weight: vec![0.5; NUM_CLASSES],
        value: vec![value; NUM_CLASSES],
        check_down: vec![check_down; NUM_CLASSES],
        combos: vec![6; NUM_CLASSES],
    });
    let info = SolveInfo {
        iterations: 3,
        exploitability_pct: 0.4,
        solved_on: "Cpu".into(),
        gpu_error: None,
        seconds: 1.5,
    };
    let spot = Spot {
        line: "BTN:R BB:C".into(),
        leaf: 0,
        path: Vec::new(),
        reach: 0.5,
        seats: [2, 0],
        contexts,
        classes: [[1.0; NUM_CLASSES]; 2],
        pot: 550,
        behind: 9750,
    };
    Record::new(&spot, "As7d2c", &sums, &info)
}

fn hu(pot: PotKind, role: Role, position: Position) -> Context {
    Context { pot, role, position, multiway: false }
}

#[test]
fn cut_off_and_bad_checkpoint_lines() {
    let ctx = [
        hu(PotKind::Raised, Role::Caller, Position::OutOfPosition),
        hu(PotKind::Raised, Role::Aggressor, Position::InPosition),
    ];
    let line = sample_record(ctx, 3.0, 2.0).to_line();
    let text = format!("{line}\n{line}\n{}", &line[..line.len() / 2]);
    let (records, complete) = parse_checkpoint(&text).unwrap();
    assert_eq!((records.len(), complete), (2, 2 * line.len() + 2));
    assert_eq!(parse_checkpoint("").unwrap(), (Vec::new(), 0));
    assert_eq!(parse_checkpoint("\n\n").unwrap().0.len(), 0);
    let bad = format!("{line}\n{}\n{line}\n", &line[..line.len() / 2]);
    assert!(parse_checkpoint(&bad).unwrap_err().contains("line 2"));
    let mut r = sample_record(ctx, 3.0, 2.0);
    r.contexts[0] = "nowhere".into();
    assert!(parse_checkpoint(&(r.to_line() + "\n")).unwrap_err().contains("unknown context"));
    let mut r = sample_record(ctx, 3.0, 2.0);
    r.value[1].pop();
    assert!(r.validate().is_err());
    let mut r = sample_record(ctx, 3.0, 2.0);
    r.weight[0][5] = -1.0;
    assert!(r.validate().is_err());
    let mut r = sample_record(ctx, 3.0, 2.0);
    r.reach = -1.0;
    assert!(r.validate().is_err());
    let mut r = sample_record(ctx, 3.0, 2.0);
    r.value[0][5] = f64::NAN;
    assert!(r.validate().is_err());
    assert!(aggregate(&[r]).is_err());
}

// ---- aggregation ----

#[test]
fn aggregation_is_a_weighted_ratio_of_sums() {
    let oop = hu(PotKind::Raised, Role::Caller, Position::OutOfPosition);
    let ip = hu(PotKind::Raised, Role::Aggressor, Position::InPosition);
    let mut a = sample_record([oop, ip], 3.0, 2.0);
    let mut b = sample_record([oop, ip], 1.0, 2.0);
    b.weight = [vec![1.5; NUM_CLASSES], vec![0.5; NUM_CLASSES]];
    // One class has no check-down value: it gets the context's ratio.
    a.check_down[1][7] = 0.0;
    b.check_down[1][7] = 0.0;
    let cal = aggregate(&[a.clone(), b.clone()]).unwrap();
    let r = &cal.realization;
    // OOP: (0.5*3 + 1.5*1) / (0.5*2 + 1.5*2) = 3 / 4.
    assert!(r.factors(oop).iter().all(|&f| (f - 0.75).abs() < 1e-6));
    // IP: equal weights, (3 + 1) / (2 + 2) = 1.
    assert!(r.factors(ip).iter().all(|&f| (f - 1.0).abs() < 1e-6));
    assert_eq!(cal.source[oop.index()], Source::Measured(2));
    // Jackknife: leaving out a gives 1/2, leaving out b gives 3/2; their
    // mean is 1, and the variance is (n-1)/n * ((1/2)^2 + (1/2)^2) = 1/4.
    for (c, &s) in cal.stderr[ip.index()].iter().enumerate() {
        // The class without check-down values has none.
        assert_eq!(s, if c == 7 { 0.0 } else { 0.5 }, "class {c}");
    }
    // A single solve has no standard error.
    let one = aggregate(&[a]).unwrap();
    assert!(one.stderr[oop.index()].iter().all(|&s| s == 0.0));
    // Values below 0 (a hand that loses chips on average) give factor 0.
    let negative = aggregate(&[sample_record([oop, ip], -1.0, 2.0)]).unwrap();
    assert!(negative.realization.factors(oop).iter().all(|&f| f == 0.0));
}

#[test]
fn unmeasured_contexts_are_filled_from_measured_ones() {
    let ip = hu(PotKind::Raised, Role::Aggressor, Position::InPosition);
    let oop = hu(PotKind::Raised, Role::Caller, Position::OutOfPosition);
    let ip3 = hu(PotKind::ThreeBet, Role::Caller, Position::InPosition);
    let oop3 = hu(PotKind::ThreeBet, Role::Aggressor, Position::OutOfPosition);
    let records = [sample_record([oop, ip], 3.0, 2.0), sample_record([oop3, ip3], 1.0, 2.0)];
    let cal = aggregate(&records).unwrap();
    let r = &cal.realization;
    let f = |ctx: Context| r.factors(ctx)[0];
    assert_eq!((f(ip), f(oop), f(ip3), f(oop3)), (1.5, 1.5, 0.5, 0.5));
    for ctx in [ip, oop, ip3, oop3] {
        let multi = Context { multiway: true, ..ctx };
        assert_eq!(f(multi), f(ctx));
        assert_eq!(cal.source[multi.index()], Source::HeadsUp(ctx));
    }
    // Middle: the average of in and out of position with the same pot and
    // role (one of them unmeasured here, so 1).
    let middle = Context { position: Position::Middle, multiway: true, ..ip };
    assert_eq!(f(middle), (1.5 + 1.0) / 2.0);
    // Limped pots copy the single-raised caller in the same position.
    for ctx in Context::all().filter(|c| c.pot == PotKind::Limped) {
        let from = Context { pot: PotKind::Raised, role: Role::Caller, ..ctx };
        assert_eq!(r.factors(ctx), r.factors(from), "{}", ctx.name());
        assert_eq!(cal.source[ctx.index()], Source::Limped(from));
    }
    // Nothing measured in 4-bet pots: factor 1.
    let four = hu(PotKind::FourBet, Role::Caller, Position::InPosition);
    assert_eq!((f(four), cal.source[four.index()]), (1.0, Source::Neutral));
    assert!(cal.source.iter().all(|s| s.to_string().len() > 3));
    // No records: everything neutral.
    assert_eq!(aggregate(&[]).unwrap().realization, crate::Realization::neutral());
}
