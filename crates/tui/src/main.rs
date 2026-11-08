//! `poker-trainer`: terminal NLHE trainer.

mod app;
mod charts_cmd;
mod score_ui;
mod selfplay;
mod sizes;
mod study_app;
mod study_ui;
mod ui;
mod widgets;

use anyhow::{anyhow, Result};
use app::App;
use clap::{Parser, Subcommand};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind};
use crossterm::execute;
use std::path::Path;
use std::time::Duration;
use trainer::{Session, TrainerConfig};

const DEFAULT_CONFIG: &str = "poker-trainer.toml";

#[derive(Parser)]
#[command(
    name = "poker-trainer",
    version,
    about = "Play NLHE hands against AI styles and check your decisions with a solver"
)]
struct Cli {
    /// Config file (default: ./poker-trainer.toml if it exists).
    #[arg(short, long, global = true)]
    config: Option<String>,
    /// Random seed for reproducible hands.
    #[arg(long, global = true)]
    seed: Option<u64>,
    /// Don't save this session (or offer to continue saved ones).
    #[arg(long, global = true)]
    no_save: bool,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play in the terminal UI (default).
    Play,
    /// Open the solver study mode directly.
    Study,
    /// Write a config file with all settings and their defaults.
    InitConfig {
        /// Output path.
        #[arg(default_value = DEFAULT_CONFIG)]
        path: String,
    },
    /// Convert a preflop range collection into a chart file (set it as
    /// charts_hu / charts_6max / charts_9max in the config to use it).
    ImportCharts {
        /// Directory with the ranges.
        dir: String,
        /// "lines" (one file per decision, named by the line) or "tree"
        /// (leaf directories with *_range.txt files).
        #[arg(long, default_value = "lines")]
        format: String,
        /// Table size the ranges are for.
        #[arg(long, default_value_t = 6)]
        players: usize,
        /// Raises of at least this many big blinds count as all-in.
        #[arg(long, default_value_t = 90.0)]
        allin_bb: f64,
        /// Output chart file.
        #[arg(short, long, default_value = "charts.txt")]
        out: String,
    },
    /// Headless: play hands with a scripted hero and print the solver review
    /// (useful to check performance on your machine).
    Selfplay {
        /// Hands that reach the flop to play.
        #[arg(long, default_value_t = 3)]
        hands: usize,
        /// Opponent style (or "random").
        #[arg(long)]
        style: Option<String>,
    },
}

fn load_config(cli: &Cli) -> Result<(TrainerConfig, String)> {
    let path = cli.config.clone().unwrap_or_else(|| DEFAULT_CONFIG.to_string());
    let mut cfg = if Path::new(&path).exists() {
        TrainerConfig::load(&path).map_err(|e| anyhow!(e))?
    } else if cli.config.is_some() {
        return Err(anyhow!("config file {path} not found"));
    } else {
        TrainerConfig::default()
    };
    if cli.seed.is_some() {
        cfg.seed = cli.seed;
    }
    Ok((cfg, path))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Some(Cmd::InitConfig { path }) => {
            if Path::new(path).exists() {
                return Err(anyhow!("{path} already exists"));
            }
            std::fs::write(path, TrainerConfig::default().to_toml())?;
            println!("Wrote {path}");
            Ok(())
        }
        Some(Cmd::ImportCharts { dir, format, players, allin_bb, out }) => {
            charts_cmd::import(dir, format, *players, *allin_bb, out)
        }
        Some(Cmd::Selfplay { hands, style }) => {
            let (mut cfg, _) = load_config(&cli)?;
            if let Some(s) = style {
                cfg.opponent = s.clone();
            }
            selfplay::run(cfg, *hands)
        }
        Some(Cmd::Play) | None => {
            let (cfg, path) = load_config(&cli)?;
            let mut app = App::new(Session::new(cfg).map_err(|e| anyhow!(e))?, path);
            setup_saving(&mut app, cli.no_save);
            // Offer to continue a saved session.
            if !app.sessions.is_empty() {
                app.open_sessions();
            }
            run_tui(app)
        }
        Some(Cmd::Study) => {
            let (cfg, path) = load_config(&cli)?;
            let mut app = App::new(Session::new(cfg).map_err(|e| anyhow!(e))?, path);
            setup_saving(&mut app, cli.no_save);
            app.open_study();
            run_tui(app)
        }
    }
}

/// Saves sessions where the config says (unless turned off).
fn setup_saving(app: &mut App, no_save: bool) {
    let cfg = &app.session.config;
    if no_save || !cfg.save_sessions {
        return;
    }
    if let Some(root) = cfg.sessions_root() {
        app.enable_saving(root);
    }
}

fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = ratatui::init();
    // Clicks select hands and actions (hold Shift to select text instead).
    let _ = execute!(std::io::stdout(), EnableMouseCapture);
    let result = (|| -> Result<()> {
        while !app.quit {
            terminal.draw(|f| ui::draw(f, &app))?;
            if event::poll(Duration::from_millis(50))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => app.on_key(key),
                    Event::Mouse(m) => app.on_mouse(m),
                    _ => {}
                }
            }
            app.tick();
        }
        Ok(())
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    // Save every finished hand (cutting running reviews short).
    if app.session.is_saving() {
        app.session.flush(Duration::from_secs(3));
        if let Some(e) = app.session.take_save_error() {
            eprintln!("{e}");
        } else if let Some(id) = app.session.save_id() {
            eprintln!("Session saved ({id}): run poker-trainer again to continue it.");
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn render(app: &App) -> String {
        render_at(app, 150, 40)
    }

    fn render_at(app: &App, w: u16, h: u16) -> String {
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| ui::draw(f, app)).unwrap();
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

    /// Drives the UI through setup, a hand and the review screen.
    #[test]
    fn ui_flow_renders() {
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(3);
        cfg.ai_delay_ms = 0;
        cfg.opponent = "Calling Station".into();
        cfg.solver.flop_mode = trainer::FlopMode::Fast;
        cfg.solver.flop_turn_samples = 3;
        cfg.solver.flop_river_samples = 2;
        cfg.solver.max_iterations = 30;
        cfg.solver.target_exploitability_pct = 5.0;
        let mut app = App::new(Session::new(cfg).unwrap(), "/nonexistent/x.toml".into());
        let s = render(&app);
        assert!(s.contains("Opponent") && s.contains("All Calling Station"), "{s}");
        let small = render_at(&app, 100, 30);
        assert!(small.contains("Reveal style during hand  ◂ Off ▸"), "{small}");
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let mut reviewed = false;
        for _ in 0..4000 {
            app.tick();
            let hand = app.current().unwrap();
            if hand.hero_to_act() {
                let s = render(&app);
                assert!(s.contains("Your move"), "{s}");
                let small = render_at(&app, 100, 30);
                assert!(small.contains("[F]") || small.contains("[C]"), "{small}");
                assert!(small.contains("Board") && small.contains("Pot"), "{small}");
                // Check/call when possible.
                app.on_key(key('c'));
            } else if hand.is_finished() {
                let has_review = hand.review().is_some();
                if has_review {
                    while !hand.review().unwrap().is_finished() {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    app.on_key(key('r'));
                    let s = render(&app);
                    assert!(s.contains("Review"), "{s}");
                    if !app.visible_decisions().is_empty() {
                        assert!(s.contains("Solver, your hand"), "{s}");
                        assert!(s.contains("Whole range"), "{s}");
                        // Zoomed-in terminal: everything important still shows.
                        let small = render_at(&app, 100, 30);
                        assert!(
                            small.contains("Solver, your hand") && small.contains("Your decisions"),
                            "{small}"
                        );
                        assert!(small.contains("A2o") && small.contains("22"), "grid cut off:\n{small}");
                        reviewed = true;
                        break;
                    }
                    app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
                }
                app.on_key(key('n'));
            }
        }
        assert!(reviewed);
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        app.on_key(key('h'));
        let s = render(&app);
        assert!(s.contains("Finished hands"), "{s}");
    }

    /// The position row and its per-position rows: choose positions, keep
    /// one ticked, switch table sizes, and get dealt the chosen position.
    #[test]
    fn setup_chooses_positions() {
        use app::SetupItem;
        use trainer::HeroPosition;
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(5);
        cfg.players = 6;
        cfg.solver.enabled = false;
        let mut app = App::new(Session::new(cfg).unwrap(), "/nonexistent/x.toml".into());
        let right = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);
        let goto = |app: &mut App, item: SetupItem| {
            app.setup_cursor = app.setup_items().iter().position(|&i| i == item).expect("row shown");
        };
        // Rotate is the default and shows no position rows.
        assert!(render(&app).contains("Your position             ◂ Rotate ▸"));
        assert!(!app.setup_items().contains(&SetupItem::Seat(0)));
        goto(&mut app, SetupItem::Position);
        app.on_key(right);
        assert_eq!(app.session.config.hero_position, HeroPosition::Random);
        app.on_key(right);
        assert_eq!(app.session.config.hero_position, HeroPosition::seats(["BTN"]).unwrap());
        let s = render(&app);
        assert!(s.contains("◂ Only BTN ▸") && s.contains("  BTN ") && s.contains("  UTG "), "{s}");
        assert_eq!(app.setup_items().iter().filter(|i| matches!(i, SetupItem::Seat(_))).count(), 6);
        // Tick UTG (the first position row), then untick BTN.
        goto(&mut app, SetupItem::Seat(0));
        app.on_key(right);
        assert_eq!(app.session.config.hero_position, HeroPosition::seats(["UTG", "BTN"]).unwrap());
        assert!(render(&app).contains("◂ Only UTG, BTN ▸"));
        goto(&mut app, SetupItem::Seat(3));
        app.on_key(right);
        assert_eq!(app.session.config.hero_position, HeroPosition::seats(["UTG"]).unwrap());
        // The last ticked position stays.
        goto(&mut app, SetupItem::Seat(0));
        app.on_key(right);
        assert_eq!(app.session.config.hero_position, HeroPosition::seats(["UTG"]).unwrap());
        assert!(app.message.as_ref().is_some_and(|(m, _)| m.contains("At least one")));
        // 9-max keeps UTG; heads-up has none, so it rotates.
        goto(&mut app, SetupItem::Table);
        app.on_key(right);
        assert_eq!(app.session.config.players, 9);
        assert_eq!(app.session.config.hero_position, HeroPosition::seats(["UTG"]).unwrap());
        for _ in 0..5 {
            let h = app.session.new_hand();
            assert_eq!(h.state.position_name(h.hero), "UTG");
        }
        app.on_key(right);
        assert_eq!(app.session.config.players, 2);
        assert_eq!(app.session.config.hero_position, HeroPosition::Alternate);
        assert!(app.message.as_ref().is_some_and(|(m, _)| m.contains("you rotate")));
    }

    /// Prints the setup, table and review screens (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn print_screens() {
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(11);
        cfg.ai_delay_ms = 0;
        cfg.opponent = "LAG".into();
        cfg.players = std::env::var("PLAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        cfg.solver.flop_mode = trainer::FlopMode::Fast;
        cfg.solver.flop_turn_samples = 4;
        cfg.solver.flop_river_samples = 3;
        let size: Vec<u16> =
            std::env::var("SIZE").unwrap_or("150x40".into()).split('x').map(|v| v.parse().unwrap()).collect();
        let render = |app: &App| render_at(app, size[0], size[1]);
        let mut app = App::new(Session::new(cfg).unwrap(), "/nonexistent/x.toml".into());
        println!("{}", render(&app));
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let mut printed_table = false;
        for _ in 0..100000 {
            app.tick();
            let hand = app.current().unwrap();
            if hand.hero_to_act() {
                if !printed_table && hand.state.board.len() >= 3 {
                    println!("{}", render(&app));
                    printed_table = true;
                }
                let k = if hand.state.board.len() == 3 { '1' } else { 'c' };
                let keys = app.action_keys();
                let k = if keys.iter().any(|(c, _)| *c == k) { k } else { 'c' };
                app.on_key(key(k));
            } else if hand.is_finished() {
                if let Some(r) = hand.review() {
                    while !r.is_finished() {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    if !r.status().decisions.is_empty() {
                        app.on_key(key('r'));
                        println!("{}", render(&app));
                        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
                        println!("{}", render(&app));
                        break;
                    }
                }
                app.on_key(key('n'));
            }
        }
    }

    fn study_config() -> TrainerConfig {
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(5);
        cfg.players = 6;
        cfg.solver.flop_mode = trainer::FlopMode::Fast;
        cfg.solver.flop_turn_samples = 4;
        cfg.solver.flop_river_samples = 3;
        cfg.solver.max_iterations = 60;
        cfg.solver.target_exploitability_pct = 2.0;
        cfg
    }

    fn press(app: &mut App, code: KeyCode) {
        app.on_key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            app.on_key(key(c));
        }
    }

    /// Ticks until the study screen has a decision (or `f` says stop).
    fn study_settle(app: &mut App) {
        for _ in 0..6000 {
            app.tick();
            let st = app.study.as_ref().unwrap();
            if let study_app::Phase::Postflop(p) = &st.phase {
                if !matches!(p.pos, trainer::study::StudyPos::Solving(_) | trainer::study::StudyPos::Loading)
                {
                    return;
                }
            } else {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("study didn't settle");
    }

    /// Prints the study screens at several sizes (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn print_study_screens() {
        let mut app = App::new(Session::new(study_config()).unwrap(), "/nonexistent/x.toml".into());
        app.open_study();
        app.tick();
        let sizes = [(100u16, 30u16), (150, 40), (200, 55)];
        for &(w, h) in &sizes {
            println!("{}", render_at(&app, w, h));
        }
        // UTG, HJ, CO fold; BTN raises; SB folds; BB calls.
        for c in ['f', 'f', 'f', '1', 'f', 'c'] {
            app.on_key(key(c));
            app.tick();
        }
        press(&mut app, KeyCode::Tab);
        app.tick();
        println!("{}", render_at(&app, 100, 30));
        press(&mut app, KeyCode::BackTab);
        type_text(&mut app, "Ks7d2c");
        println!("{}", render_at(&app, 100, 30));
        press(&mut app, KeyCode::Enter);
        app.tick();
        println!("{}", render_at(&app, 150, 40));
        study_settle(&mut app);
        app.study.as_mut().unwrap().cursor = 0; // AA
        for &(w, h) in &sizes {
            println!("{}", render_at(&app, w, h));
        }
        app.on_key(key('e'));
        println!("{}", render_at(&app, 150, 40));
    }

    /// Drives study mode: a preflop line, a flop, actions, a turn card,
    /// the range view, going back, and a typed spot.
    #[test]
    fn study_flow_works() {
        let mut app = App::new(Session::new(study_config()).unwrap(), "/nonexistent/x.toml".into());
        // From the setup screen: switch the mode row, then Enter.
        app.on_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(app.mode, app::Mode::Study);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.screen, app::Screen::Study);
        app.tick();
        let s = render_at(&app, 100, 30);
        assert!(s.contains("UTG to act") && s.contains("Raise to 2.5bb"), "{s}");
        // UTG, HJ, CO fold; BTN raises; SB folds; BB calls.
        for c in ['f', 'f', 'f', '1', 'f', 'c'] {
            app.on_key(key(c));
            app.tick();
        }
        let s = render_at(&app, 100, 30);
        assert!(s.contains("Flop cards"), "{s}");
        press(&mut app, KeyCode::Tab);
        let s = render_at(&app, 100, 30);
        assert!(s.contains("BB's range"), "{s}");
        press(&mut app, KeyCode::BackTab);
        type_text(&mut app, "Ks7d2c");
        press(&mut app, KeyCode::Enter);
        study_settle(&mut app);
        let s = render_at(&app, 120, 34);
        assert!(s.contains("BB to act") && s.contains("Check"), "{s}");
        // Clicking an action works like its key: check, then check back.
        app.on_key(key('c'));
        study_settle(&mut app);
        let s = render_at(&app, 120, 34);
        assert!(s.contains("BTN to act"), "{s}");
        app.on_key(key('c'));
        study_settle(&mut app);
        app.tick();
        let s = render_at(&app, 120, 34);
        assert!(s.contains("Turn card"), "{s}");
        type_text(&mut app, "9h");
        press(&mut app, KeyCode::Enter);
        study_settle(&mut app);
        let s = render_at(&app, 120, 34);
        assert!(s.contains("BB to act") && s.contains("Turn 9♥"), "{s}");
        // Ranges going into the turn.
        press(&mut app, KeyCode::Tab);
        let s = render_at(&app, 120, 34);
        assert!(s.contains("BB's range") && s.contains("of the starting range"), "{s}");
        // Back to the card prompt, then to BTN's flop decision.
        press(&mut app, KeyCode::Backspace);
        app.tick();
        press(&mut app, KeyCode::Backspace);
        study_settle(&mut app);
        app.tick();
        let s = render_at(&app, 120, 34);
        assert!(s.contains("BTN to act"), "{s}");
        // Type a spot: the editor starts from the current spot.
        app.on_key(key('e'));
        let s = render_at(&app, 120, 34);
        assert!(s.contains("Type a spot") && s.contains("BB range"), "{s}");
        app.on_key(key('s'));
        study_settle(&mut app);
        let s = render_at(&app, 120, 34);
        assert!(s.contains("BB to act"), "{s}");
        // Esc goes back to the menu; Enter resumes the same study.
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.screen, app::Screen::Setup);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.screen, app::Screen::Study);
        study_settle(&mut app);
        assert!(render_at(&app, 120, 34).contains("BB to act"));
    }

    /// Q quits only when pressed twice in a row.
    #[test]
    fn quit_needs_two_presses() {
        let mut app = App::new(Session::new(TrainerConfig::default()).unwrap(), "/nonexistent/x.toml".into());
        app.on_key(key('q'));
        assert!(!app.quit);
        assert!(render_at(&app, 120, 34).contains("Press Q again to quit"));
        // Another key in between cancels it.
        press(&mut app, KeyCode::Down);
        app.on_key(key('q'));
        assert!(!app.quit);
        app.on_key(key('q'));
        assert!(app.quit);
    }

    fn quick_cfg(seed: u64) -> TrainerConfig {
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(seed);
        cfg.ai_delay_ms = 0;
        cfg.opponent = "Calling Station".into();
        cfg.solver.flop_mode = trainer::FlopMode::Fast;
        cfg.solver.flop_turn_samples = 3;
        cfg.solver.flop_river_samples = 2;
        cfg.solver.max_iterations = 30;
        cfg.solver.target_exploitability_pct = 5.0;
        cfg
    }

    /// Plays hands through the UI (betting when the RNG is low) until
    /// `reviewed` hands have graded decisions and are settled.
    fn play_reviewed(app: &mut App, reviewed: usize) {
        let mut done = 0;
        for _ in 0..20_000 {
            app.tick();
            let Some(hand) = app.current() else { panic!("no hand") };
            if hand.hero_to_act() {
                let roll = hand.current_roll().unwrap();
                let s = render(app);
                assert!(s.contains(&format!("RNG {roll:02}")), "{s}");
                let bet = hand.state.street != holdem::Street::Preflop && roll < 35;
                let k = if bet && app.action_keys().iter().any(|(k, _)| *k == '1') { '1' } else { 'c' };
                app.on_key(key(k));
            } else if hand.is_finished() {
                if let Some(r) = hand.review() {
                    while !r.is_finished() {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    if !r.status().decisions.is_empty() {
                        done += 1;
                    }
                }
                app.tick();
                if done >= reviewed {
                    return;
                }
                app.on_key(key('n'));
            }
        }
        panic!("not enough reviewed hands");
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("poker-trainer-tui-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// Hands are saved; the sessions screen continues, starts and deletes
    /// sessions; the score screen and the review's RNG line render.
    #[test]
    fn sessions_and_score_screens() {
        let root = scratch("sessions");
        let mut app = App::new(Session::new(quick_cfg(4)).unwrap(), "/nonexistent/x.toml".into());
        app.enable_saving(root.clone());
        assert!(app.sessions.is_empty());
        press(&mut app, KeyCode::Enter);
        play_reviewed(&mut app, 2);
        let first = app.session.save_id().expect("saved after the first hand").to_string();
        let hands = app.session.history.len();

        // Review: the RNG line maps the number to the solver's mix.
        app.on_key(key('r'));
        let s = render(&app);
        assert!(s.contains("the solver's mix plays"), "{s}");
        // Score screen.
        app.on_key(key('g'));
        let s = render(&app);
        assert!(s.contains("Frequency score") && s.contains("By spot") && s.contains("Aggression"), "{s}");
        let small = render_at(&app, 100, 30);
        assert!(small.contains("Frequency score"), "{small}");
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.screen, app::Screen::Review);
        press(&mut app, KeyCode::Esc);

        // Sessions: start a new one, then continue the first.
        app.on_key(key('s'));
        app.on_key(key('o'));
        assert_eq!(app.screen, app::Screen::Sessions);
        let s = render(&app);
        assert!(s.contains("Start a new session") && s.contains("playing now"), "{s}");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.screen, app::Screen::Setup);
        assert!(app.session.history.is_empty() && app.session.save_id().is_none());
        app.on_key(key('o'));
        assert_eq!(app.sessions.len(), 1);
        assert_eq!(app.sessions[0].summary.hands, hands);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.session.save_id(), Some(first.as_str()));
        assert_eq!(app.session.history.len(), hands);
        assert!(app.session.report().is_some(), "the score survives");
        let s = render(&app);
        assert!(s.contains(&format!("Saved · {hands} hands")), "{s}");

        // The session in use can't be deleted; another one can.
        app.on_key(key('o'));
        press(&mut app, KeyCode::Down);
        app.on_key(key('d'));
        assert!(app.confirm_delete.is_none());
        press(&mut app, KeyCode::Up);
        press(&mut app, KeyCode::Enter); // new session
        press(&mut app, KeyCode::Enter); // play a hand in it
        play_reviewed(&mut app, 1);
        app.on_key(key('s'));
        app.on_key(key('o'));
        assert_eq!(
            app.sessions.len(),
            2,
            "{:?} now {:?} hist {}",
            app.sessions.iter().map(|m| &m.id).collect::<Vec<_>>(),
            app.session.save_id(),
            app.session.history.len()
        );
        let row = 1 + app.sessions.iter().position(|m| m.id == first).unwrap();
        for _ in 0..row {
            press(&mut app, KeyCode::Down);
        }
        app.on_key(key('d'));
        assert!(render(&app).contains("Press Y to delete"));
        app.on_key(key('y'));
        assert_eq!(app.sessions.len(), 1);
        assert_ne!(app.sessions[0].id, first);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Prints the score, sessions, setup and review screens (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn print_score_screens() {
        let root = scratch("print");
        let mut app = App::new(Session::new(quick_cfg(9)).unwrap(), "/nonexistent/x.toml".into());
        app.enable_saving(root.clone());
        press(&mut app, KeyCode::Enter);
        play_reviewed(&mut app, 6);
        app.on_key(key('r'));
        println!("{}", render_at(&app, 150, 40));
        println!("{}", render_at(&app, 100, 30));
        app.on_key(key('g'));
        println!("{}", render_at(&app, 150, 40));
        println!("{}", render_at(&app, 100, 30));
        press(&mut app, KeyCode::Esc);
        press(&mut app, KeyCode::Esc);
        app.on_key(key('s'));
        println!("{}", render_at(&app, 100, 30));
        app.on_key(key('o'));
        println!("{}", render_at(&app, 150, 40));
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Times drawing the screens of a 2000-hand session (run with
    /// --ignored --nocapture, in release mode for real numbers).
    #[test]
    #[ignore]
    fn bench_large_session_render() {
        use trainer::save::{SavedReview, SessionStore};
        let root = scratch("bench");
        let mut cfg = TrainerConfig::default();
        cfg.seed = Some(1);
        cfg.solver.enabled = false;
        let mut s = Session::new(cfg.clone()).unwrap();
        let mut saved = Vec::new();
        for _ in 0..2000 {
            let mut h = s.new_hand();
            while h.run_until_hero() == trainer::StepResult::HeroToAct {
                let a = h.hero_actions();
                h.hero_act(
                    *a.iter()
                        .find(|x| matches!(x, holdem::Action::Check | holdem::Action::Call))
                        .unwrap_or(&a[0]),
                )
                .unwrap();
            }
            let mut sh = h.to_saved();
            let d = trainer::DecisionReview {
                variant: trainer::Variant::Gto,
                street: holdem::Street::Flop,
                board: Vec::new(),
                history: String::new(),
                pot: 600,
                to_call: 0,
                hero_combo: 0,
                actions: vec![holdem::Action::Check, holdem::Action::Raise(200)],
                labels: vec!["Check".into(), "Bet".into()],
                chosen: 0,
                best: 0,
                hero_strategy: vec![0.6, 0.4],
                hero_ev: vec![1.0, 1.0],
                ev_loss: 0.0,
                ev_loss_pct: 0.0,
                verdict: trainer::Verdict::Best,
                range_strategy: vec![0.5, 0.5],
                grid: vec![None; 169],
                exploitability_pct: 0.3,
                notes: Vec::new(),
                spot: trainer::SpotInfo {
                    players: 2,
                    in_position: false,
                    facing: 0,
                    history_index: 0,
                    categories: vec!["Check".into(), "33%".into()],
                },
            };
            sh.review = Some(SavedReview {
                stage: "done".into(),
                decisions: vec![d.clone(), d.clone(), d],
                notes: Vec::new(),
                cancelled: false,
                error: None,
            });
            saved.push(sh);
        }
        let store = SessionStore::create(&root, 0).unwrap();
        let mut meta = s.meta();
        meta.id = store.id.clone();
        store.write_meta(&meta).unwrap();
        store.append_hands(&saved).unwrap();
        let loaded = store.load().unwrap();
        let session = Session::resume(loaded, root.clone(), &cfg).unwrap();
        let mut app = App::new(session, "/nonexistent/x.toml".into());
        app.sessions_root = Some(root.clone());
        let time = |app: &App, label: &str| {
            let mut term = Terminal::new(TestBackend::new(150, 40)).unwrap();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                term.draw(|f| ui::draw(f, app)).unwrap();
            }
            println!("{label:<12} {:?} per frame", t.elapsed() / 20);
        };
        time(&app, "setup");
        app.on_key(key('h'));
        app.screen = app::Screen::History;
        app.history_selected = 1999;
        time(&app, "history");
        app.screen = app::Screen::Score;
        time(&app, "score");
        app.open_sessions();
        time(&app, "sessions");
        std::fs::remove_dir_all(root).unwrap();
    }

    /// The "Solve on" row cycles CPU, Auto, GPU and shows the choice.
    #[test]
    fn solve_on_row_cycles_backends() {
        use trainer::SolverBackend;
        let mut app = App::new(Session::new(TrainerConfig::default()).unwrap(), "/nonexistent/x.toml".into());
        while app.setup_items()[app.setup_cursor] != app::SetupItem::SolveOn {
            press(&mut app, KeyCode::Down);
        }
        assert!(render(&app).contains("Solve on"));
        for want in [SolverBackend::Auto, SolverBackend::Gpu, SolverBackend::Cpu] {
            press(&mut app, KeyCode::Right);
            assert_eq!(app.session.config.solver.backend, want);
        }
        press(&mut app, KeyCode::Left);
        assert_eq!(app.session.config.solver.backend, SolverBackend::Gpu);
        assert!(render(&app).contains("GPU"));
    }

    /// Prints the bet-size editor (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn print_sizes_screen() {
        let mut app = App::new(Session::new(TrainerConfig::default()).unwrap(), "/nonexistent/x.toml".into());
        while app.setup_items()[app.setup_cursor] != app::SetupItem::BetSizes {
            press(&mut app, KeyCode::Down);
        }
        press(&mut app, KeyCode::Enter);
        let t = std::time::Instant::now();
        println!("{}", render_at(&app, 120, 34));
        println!("first draw {:?}", t.elapsed());
        for _ in 0..8 {
            press(&mut app, KeyCode::Down);
        }
        println!("{}", render_at(&app, 120, 34));
    }
}
