//! `poker-trainer`: terminal NLHE trainer.

mod app;
mod charts_cmd;
mod selfplay;
mod ui;

use anyhow::{anyhow, Result};
use app::App;
use clap::{Parser, Subcommand};
use crossterm::event::{self, Event, KeyEventKind};
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
    }
}

fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = ratatui::init();
    let result = (|| -> Result<()> {
        while !app.quit {
            terminal.draw(|f| ui::draw(f, &app))?;
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        app.on_key(key);
                    }
                }
            }
            app.tick();
        }
        Ok(())
    })();
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
        let mut term = Terminal::new(TestBackend::new(150, 40)).unwrap();
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
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let mut reviewed = false;
        for _ in 0..4000 {
            app.tick();
            let hand = app.current().unwrap();
            if hand.hero_to_act() {
                let s = render(&app);
                assert!(s.contains("Your move"), "{s}");
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
}
