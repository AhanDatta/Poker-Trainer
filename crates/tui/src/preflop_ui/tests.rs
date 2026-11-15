use super::*;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use trainer::preflop::Availability;
use trainer::preflop_grade::grade;
use trainer::{Session, StepResult, TrainerConfig};

fn first_in(key: &str, action: PreAction) -> PreDecision {
    PreDecision {
        seat: 3,
        position: "UTG".into(),
        key: key.into(),
        avail: Availability {
            actions: [
                Some(Action::Fold),
                Some(Action::Call),
                Some(Action::Raise(250)),
                Some(Action::Raise(10_000)),
            ],
        },
        action,
    }
}

fn item(hole: &str, action: PreAction) -> PreflopItem {
    let charts = ChartSet::builtin(6);
    let d = first_in(">UTG", action);
    let cards: Vec<Card> = hole.split_whitespace().map(|c| c.parse().unwrap()).collect();
    let hole = [cards[0], cards[1]];
    let g = grade(&charts, std::slice::from_ref(&d), 3, hole).remove(0);
    PreflopItem {
        grade: g,
        actions: PreAction::ALL.iter().filter_map(|&a| Some((a, d.avail.concrete(a)?))).collect(),
        range: [1.0; NUM_CLASSES],
        policy: chart_policy(&charts, &d),
        decision: d,
        hole,
    }
}

fn render(item: &PreflopItem, w: u16, h: u16) -> String {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| draw_preflop_decision(f, f.area(), item, |_| Color::Green)).unwrap();
    let buf = term.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

#[test]
fn lines_and_actions_in_words() {
    assert_eq!(line_words(">UTG"), "Folded to you");
    assert_eq!(line_words("UTG:R HJ:C>BTN"), "UTG raises · HJ calls");
    assert_eq!(line_words("SB:R BB:A SB:F>BB"), "SB raises · BB goes all-in · SB folds");
    assert_eq!(line_words("weird"), "Folded to you");
    assert_eq!(line_words("X:Q>Y"), "X acts");
    assert_eq!(line_words("SB:C>BB"), "SB limps");
    assert_eq!(line_words("UTG:C HJ:R UTG:C>CO"), "UTG limps · HJ raises · UTG calls");
    let d = first_in(">UTG", PreAction::Raise);
    assert_eq!(action_label(PreAction::Raise, &d), "Raise");
    assert_eq!(action_label(PreAction::Passive, &d), "Call/limp");
    assert_eq!(action_label(PreAction::AllIn, &d), "All-in");
}

#[test]
fn verdict_texts_say_how_often_the_chart_does_it() {
    let aa = item("Ah As", PreAction::Raise);
    assert_eq!(aa.grade.verdict, Some(Verdict::Best));
    assert_eq!(verdict_text(&aa), "the chart always does this here");
    let trash = item("7h 2c", PreAction::Raise);
    assert_eq!(trash.grade.verdict, Some(Verdict::Blunder));
    assert_eq!(verdict_text(&trash), "the chart never does this here (Fold instead)");
    let mut mixed = item("Ah As", PreAction::Raise);
    mixed.grade.chart = Some([0.6, 0.0, 0.4, 0.0]);
    assert_eq!(verdict_text(&mixed), "the chart does this 40% of the time here (mostly: Fold)");
    let mut none = item("Ah As", PreAction::Raise);
    none.grade.chart = None;
    none.grade.verdict = None;
    assert!(verdict_text(&none).contains("No chart covers this spot"));
    assert_eq!(row(&trash), ("Raise".to_string(), Some(Verdict::Blunder)));
}

#[test]
fn the_detail_view_renders_at_every_size() {
    let trash = item("7h 2c", PreAction::Raise);
    let s = render(&trash, 120, 30);
    for needle in
        ["Preflop", "Folded to you", "UTG to act", "72o", "Blunder", "Chart, your hand", "◀ you", "★ chart"]
    {
        assert!(s.contains(needle), "{needle:?} not in\n{s}");
    }
    let mut none = trash.clone();
    none.grade.chart = None;
    none.grade.verdict = None;
    let s = render(&none, 120, 30);
    assert!(s.contains("isn't graded") && !s.contains("Chart, your hand"), "{s}");
    // Narrow, short and tiny terminals: no panic.
    for (w, h) in [(60, 40), (100, 30), (20, 6), (1, 1)] {
        let _ = render(&trash, w, h);
    }
}

/// Played hands give one item per hero preflop decision, with the hero's
/// chart range at the spot.
#[test]
fn items_come_from_played_hands() {
    let mut cfg = TrainerConfig { seed: Some(4), players: 6, ..TrainerConfig::default() };
    cfg.solver.enabled = false;
    let mut session = Session::new(cfg).unwrap();
    let charts = session.charts().clone();
    let mut seen = 0;
    for _ in 0..30 {
        let mut hand = session.new_hand();
        while hand.run_until_hero() == StepResult::HeroToAct {
            let acts = hand.hero_actions();
            let pick =
                acts.iter().copied().find(|a| matches!(a, Action::Check | Action::Call)).unwrap_or(acts[0]);
            hand.hero_act(pick).unwrap();
        }
        let items = preflop_items(&hand, &charts);
        assert_eq!(items.len(), hand.preflop_grades().len());
        for it in &items {
            assert_eq!(it.decision.seat, hand.hero);
            assert_eq!(it.policy.len(), NUM_CLASSES);
            assert!(!it.actions.is_empty() && it.actions.iter().all(|(a, _)| it.decision.avail.has(*a)));
            // The hero's class is in its own range at the spot.
            assert!(it.range[it.grade.class] > 0.0 || it.grade.decision > 0);
            seen += 1;
        }
        session.finish_hand(hand);
    }
    assert!(seen >= 30);
}
