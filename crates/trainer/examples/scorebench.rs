//! Benchmarks the frequency score and saved sessions on a large synthetic
//! session: hands are really played (reviews off), then given synthetic
//! graded decisions shaped like real ones (13x13 grids included).
//!
//! `cargo run --release -p trainer --example scorebench -- [hands] [decisions per hand]`

use holdem::{Action, Street};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};
use trainer::review::{ClassCell, SpotInfo};
use trainer::save::{SavedReview, SessionStore};
use trainer::scoring::{hand_decisions, FrequencyReport, ScoreOptions};
use trainer::{DecisionReview, Session, StepResult, TrainerConfig, Variant, Verdict};

fn decision(rng: &mut StdRng, street: Street, idx: usize) -> DecisionReview {
    let facing = rng.random_range(0..3u8);
    let (actions, cats): (Vec<Action>, Vec<&str>) = match facing {
        0 => (vec![Action::Check, Action::Raise(200), Action::Raise(450)], vec!["Check", "33%", "75%"]),
        _ => (
            vec![Action::Fold, Action::Call, Action::Raise(1350), Action::Raise(9750)],
            vec!["Fold", "Call", "3x", "All-in"],
        ),
    };
    let na = actions.len();
    let mut mix: Vec<f32> = (0..na).map(|_| rng.random::<f32>().powi(2)).collect();
    let s: f32 = mix.iter().sum();
    mix.iter_mut().for_each(|x| *x /= s);
    let grid = (0..169)
        .map(|_| {
            rng.random_bool(0.6).then(|| {
                let mut st: Vec<f32> = (0..na).map(|_| rng.random()).collect();
                let t: f32 = st.iter().sum();
                st.iter_mut().for_each(|x| *x /= t);
                ClassCell { weight: rng.random::<f32>() * 6.0, strategy: st }
            })
        })
        .collect();
    let chosen = rng.random_range(0..na);
    DecisionReview {
        variant: Variant::Gto,
        street,
        board: Vec::new(),
        history: "BB Check · SB Bet 2bb (33%)".into(),
        pot: rng.random_range(200..8000),
        to_call: 0,
        hero_combo: rng.random_range(0..1326),
        actions,
        labels: cats.iter().map(|c| c.to_string()).collect(),
        chosen,
        best: 0,
        hero_strategy: mix,
        hero_ev: (0..na).map(|_| rng.random::<f32>() * 500.0).collect(),
        ev_loss: 0.0,
        ev_loss_pct: rng.random::<f32>() * 5.0,
        verdict: Verdict::Good,
        range_strategy: (0..na).map(|_| 1.0 / na as f64).collect(),
        grid,
        exploitability_pct: 0.4,
        notes: Vec::new(),
        spot: SpotInfo {
            players: 2,
            in_position: rng.random(),
            facing,
            history_index: idx,
            categories: cats.iter().map(|c| c.to_string()).collect(),
        },
        solver: Default::default(),
    }
}

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let x = f();
    println!("{label:<44} {:>10.2?}", t.elapsed());
    x
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hands: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let per_hand: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut cfg = TrainerConfig { seed: Some(1), ..Default::default() };
    cfg.solver.enabled = false;
    let mut session = Session::new(cfg.clone()).unwrap();
    let mut rng = StdRng::seed_from_u64(2);
    let saved: Vec<_> = time(&format!("play {hands} hands (no reviews)"), || {
        (0..hands)
            .map(|_| {
                let mut h = session.new_hand();
                while h.run_until_hero() == StepResult::HeroToAct {
                    let a = h.hero_actions();
                    let pick =
                        a.iter().copied().find(|x| matches!(x, Action::Check | Action::Call)).unwrap_or(a[0]);
                    h.hero_act(pick).unwrap();
                }
                h.to_saved()
            })
            .collect()
    });
    let saved: Vec<_> = saved
        .into_iter()
        .map(|mut h| {
            let decisions = (0..per_hand)
                .map(|k| decision(&mut rng, [Street::Flop, Street::Turn, Street::River][k % 3], k))
                .collect();
            h.review = Some(SavedReview {
                stage: "Review complete".into(),
                decisions,
                notes: Vec::new(),
                cancelled: false,
                error: None,
            });
            h
        })
        .collect();
    let n_dec = hands * per_hand;

    // Score computation.
    let scored: Vec<_> =
        saved.iter().flat_map(|h| hand_decisions(&h.review.as_ref().unwrap().decisions, true)).collect();
    for sims in [200, 1000] {
        for n in [500, 2000, n_dec] {
            let n = n.min(scored.len());
            let opts = ScoreOptions { simulations: sims, seed: 3 };
            let r = time(&format!("score {n} decisions, {sims} simulations"), || {
                FrequencyReport::compute(&scored[..n], opts).unwrap()
            });
            assert_eq!(r.decisions, n);
        }
    }

    // Saving and loading.
    let root = std::env::temp_dir().join(format!("poker-trainer-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let store = SessionStore::create(&root, 0).unwrap();
    let mut meta = session.meta();
    meta.id = store.id.clone();
    store.write_meta(&meta).unwrap();
    // As in play: one hand per write (each synced to disk).
    let t = Instant::now();
    let mut worst = Duration::ZERO;
    for h in &saved {
        let t1 = Instant::now();
        store.append_hands(std::slice::from_ref(h)).unwrap();
        store.write_meta(&meta).unwrap();
        worst = worst.max(t1.elapsed());
    }
    println!(
        "{:<44} {:>10.2?}  (avg {:.2?}, worst {:.2?})",
        format!("save {hands} hands one by one (+meta)"),
        t.elapsed(),
        t.elapsed() / hands as u32,
        worst
    );
    let size = std::fs::metadata(store.dir().join("hands.jsonl")).unwrap().len();
    println!(
        "{:<44} {:>10.1} MB ({:.1} KB per hand)",
        "hands.jsonl",
        size as f64 / 1e6,
        size as f64 / 1e3 / hands as f64
    );
    let loaded = time("load the session file", || store.load().unwrap());
    assert_eq!(loaded.hands.len(), hands);
    let mut resumed =
        time("resume (rebuild hands, stats, score)", || Session::resume(loaded, root.clone(), &cfg).unwrap());
    let st = resumed.stats();
    assert_eq!(st.hands, hands);
    assert_eq!(st.decisions, n_dec);
    // Per-frame costs.
    let k = 1000;
    let t = Instant::now();
    for _ in 0..k {
        std::hint::black_box(resumed.stats());
    }
    println!("{:<44} {:>10.2?}", "stats() per frame", t.elapsed() / k);
    let t = Instant::now();
    for _ in 0..k {
        resumed.refresh();
    }
    println!("{:<44} {:>10.2?}", "refresh() per tick", t.elapsed() / k);
    let t = Instant::now();
    for _ in 0..k {
        std::hint::black_box(resumed.report().map(|r| r.calibration.score()));
    }
    println!("{:<44} {:>10.2?}", "report() per frame", t.elapsed() / k);
    std::fs::remove_dir_all(root).unwrap();
}
