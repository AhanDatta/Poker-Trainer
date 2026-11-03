//! `poker-trainer`: terminal NLHE trainer.

mod app;
mod charts_cmd;
mod selfplay;
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
            let session = Session::new(cfg).map_err(|e| anyhow!(e))?;
            run_tui(App::new(session, path))
        }
        Some(Cmd::Study) => {
            let (cfg, path) = load_config(&cli)?;
            let session = Session::new(cfg).map_err(|e| anyhow!(e))?;
            let mut app = App::new(session, path);
            app.open_study();
            run_tui(app)
        }
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
                        assert!(small.contains("Solver, your hand") && small.contains("Your decisions"), "{small}");
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
        let size: Vec<u16> = std::env::var("SIZE")
            .unwrap_or("150x40".into())
            .split('x')
            .map(|v| v.parse().unwrap())
            .collect();
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
                if !matches!(p.pos, trainer::study::StudyPos::Solving(_) | trainer::study::StudyPos::Loading) {
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
}
