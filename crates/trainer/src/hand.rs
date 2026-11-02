//! One hand between the hero and AI opponents.

use crate::ai::{concrete_preflop, AiPlayer};
use crate::fmt::{action_verb, bb, ActionContext};
use crate::preflop::{availability, chart_policy, line_range, spot_key, ChartSet, PreDecision};
use crate::profiles::Profile;
use crate::review::{Review, ReviewSetup, Snapshot};
use holdem::game::{Phase, Settlement};
use holdem::{Action, BetMenu, Card, CardSet, Chips, Deck, GameError, HandState, Street, TableConfig};
use rand::rngs::StdRng;
use std::sync::Arc;

/// A line of the hand log.
#[derive(Clone, Debug)]
pub struct LogEntry {
    pub street: Street,
    pub seat: Option<usize>,
    pub text: String,
}

/// What [`LiveHand::step`] did.
#[derive(Clone, Debug, PartialEq)]
pub enum StepResult {
    /// The hero must act; call [`LiveHand::hero_act`].
    HeroToAct,
    AiActed {
        seat: usize,
        action: Action,
    },
    Dealt(Street),
    Finished,
}

/// Static parameters of a hand.
pub struct HandSetup {
    pub id: u64,
    pub table: TableConfig,
    pub button: usize,
    pub stacks: Vec<Chips>,
    pub hero: usize,
    /// Profile of each seat (`None` for the hero).
    pub profiles: Vec<Option<Profile>>,
    pub menu: BetMenu,
    pub charts: Arc<ChartSet>,
    /// Review settings (`None` disables the solver review).
    pub review: Option<crate::config::SolverSettings>,
}

pub struct LiveHand {
    pub id: u64,
    pub state: HandState,
    pub hero: usize,
    pub profiles: Vec<Option<Profile>>,
    pub preflop: Vec<PreDecision>,
    pub log: Vec<LogEntry>,
    ais: Vec<Option<AiPlayer>>,
    deck: Deck,
    menu: BetMenu,
    charts: Arc<ChartSet>,
    review_settings: Option<crate::config::SolverSettings>,
    review: Option<Review>,
    rng: StdRng,
}

impl LiveHand {
    pub fn new(setup: HandSetup, mut rng: StdRng) -> LiveHand {
        let n = setup.table.num_players;
        let mut deck = Deck::shuffled(&mut rng);
        let hole: Vec<[Card; 2]> = (0..n).map(|_| [deck.deal(), deck.deal()]).collect();
        let state = HandState::new(setup.table.clone(), setup.button, setup.stacks.clone(), hole);
        let ais = setup
            .profiles
            .iter()
            .enumerate()
            .map(|(seat, p)| p.as_ref().map(|p| AiPlayer::new(seat, p.clone(), n)))
            .collect();
        let mut hand = LiveHand {
            id: setup.id,
            state,
            hero: setup.hero,
            profiles: setup.profiles,
            preflop: Vec::new(),
            log: Vec::new(),
            ais,
            deck,
            menu: setup.menu,
            charts: setup.charts,
            review_settings: setup.review,
            review: None,
            rng,
        };
        let (sb, bb_seat) = (hand.state.sb_seat(), hand.state.bb_seat());
        hand.push_log(
            None,
            format!(
                "Hand #{}: {} posts {}, {} posts {}",
                hand.id,
                hand.seat_name(sb),
                bb(hand.state.config.small_blind),
                hand.seat_name(bb_seat),
                bb(hand.state.config.big_blind)
            ),
        );
        hand
    }

    pub fn menu(&self) -> &BetMenu {
        &self.menu
    }

    /// "SB (you)" / "BB (Villain)".
    pub fn seat_name(&self, seat: usize) -> String {
        let pos = self.state.position_name(seat);
        if seat == self.hero {
            format!("{pos} (you)")
        } else {
            format!("{pos} (villain)")
        }
    }

    fn push_log(&mut self, seat: Option<usize>, text: String) {
        self.log.push(LogEntry { street: self.state.street, seat, text });
    }

    pub fn is_finished(&self) -> bool {
        self.state.phase == Phase::Finished
    }

    pub fn hero_to_act(&self) -> bool {
        self.state.phase == Phase::Betting && self.state.to_act == Some(self.hero)
    }

    /// Menu actions available to the hero (empty when it's not their turn).
    pub fn hero_actions(&self) -> Vec<Action> {
        if self.hero_to_act() {
            self.state.legal_actions(&self.menu)
        } else {
            Vec::new()
        }
    }

    /// Labeling context for the player to act.
    pub fn action_context(&self) -> Option<ActionContext> {
        let p = self.state.to_act?;
        let sit = self.state.situation(p);
        Some(ActionContext {
            to_call: sit.to_call,
            pot: sit.pot,
            all_in_to: self.state.street_commit[p] + self.state.stacks[p],
            facing_bet: self.state.current_bet > 0,
            preflop: self.state.street == Street::Preflop,
        })
    }

    /// Advances the hand by one event: an AI action, a dealt street, or a
    /// stop because the hero must act / the hand is over.
    pub fn step(&mut self) -> StepResult {
        match self.state.phase {
            Phase::Finished => StepResult::Finished,
            Phase::AwaitingBoard => {
                let n = self.state.cards_needed();
                let cards = self.deck.deal_n(n);
                self.state.deal_board(&cards).expect("valid board");
                let street = self.state.street;
                let shown: String =
                    self.state.board.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
                self.push_log(None, format!("{street}: {shown}  (pot {})", bb(self.state.pot())));
                if street == Street::Flop {
                    self.on_flop();
                }
                self.after_change();
                StepResult::Dealt(street)
            }
            Phase::Betting => {
                let p = self.state.to_act.expect("someone to act");
                if p == self.hero {
                    return StepResult::HeroToAct;
                }
                let action = self.ai_decision(p);
                self.apply(p, action).expect("AI chose a legal action");
                StepResult::AiActed { seat: p, action }
            }
        }
    }

    /// Applies the hero's action.
    pub fn hero_act(&mut self, action: Action) -> Result<(), GameError> {
        if !self.hero_to_act() {
            return Err(GameError::NotBetting);
        }
        if !self.hero_actions().contains(&action) {
            return Err(GameError::IllegalAction(format!("{action:?} is not in the menu")));
        }
        self.apply(self.hero, action)
    }

    fn ai_decision(&mut self, p: usize) -> Action {
        let ai = self.ais[p].as_ref().expect("AI seat");
        if self.state.street == Street::Preflop {
            let avail = availability(&self.state, &self.menu).expect("decision");
            let position = self.state.position_name(p).to_string();
            let d = PreDecision {
                seat: p,
                key: spot_key(&self.preflop, &position),
                position,
                avail: avail.clone(),
                action: crate::preflop::PreAction::Fold,
            };
            let a = ai.decide_preflop(&self.charts, &d, self.state.hole[p], &mut self.rng);
            concrete_preflop(&avail, a)
        } else {
            ai.decide_postflop(&self.state, &self.menu, &mut self.rng)
        }
    }

    fn apply(&mut self, p: usize, action: Action) -> Result<(), GameError> {
        let before = self.state.clone();
        let ctx = self.action_context().expect("player to act");
        if before.street == Street::Preflop {
            if let Some(avail) = availability(&before, &self.menu) {
                if let Some(abs) = avail.classify(action) {
                    let position = before.position_name(p).to_string();
                    let key = spot_key(&self.preflop, &position);
                    self.preflop.push(PreDecision { seat: p, position, key, avail, action: abs });
                }
            }
        }
        let rec = self.state.apply(action)?;
        for ai in self.ais.iter_mut().flatten() {
            ai.observe(&before, &rec);
        }
        let verb = action_verb(action, &ctx);
        self.push_log(Some(p), format!("{} {verb}", self.seat_name(p)));
        if self.state.phase == Phase::Finished {
            self.on_finish();
        }
        self.after_change();
        Ok(())
    }

    fn on_flop(&mut self) {
        // AI reads start from the chart range of each opponent's preflop line.
        let live = self.state.players_in_hand();
        for ai_seat in 0..self.ais.len() {
            if self.ais[ai_seat].is_none() {
                continue;
            }
            for &q in &live {
                if q == ai_seat {
                    continue;
                }
                let (range, _) = line_range(&self.preflop, q, |d| chart_policy(&self.charts, d));
                let mut w = range.into_weights();
                let dead = CardSet::from_cards(&self.state.board)
                    .with(self.state.hole[ai_seat][0])
                    .with(self.state.hole[ai_seat][1]);
                for (c, x) in w.iter_mut().enumerate() {
                    if holdem::combo::combo_mask(c) & dead.0 != 0 {
                        *x = 0.0;
                    }
                }
                self.ais[ai_seat].as_mut().unwrap().set_read(q, w);
            }
        }
        // Start the solver review (heads-up pots only).
        if let Some(settings) = self.review_settings.clone() {
            if settings.enabled && live.len() >= 2 && live.contains(&self.hero) {
                let setup = ReviewSetup {
                    hero: self.hero,
                    profiles: self.profiles.clone(),
                    charts: Arc::clone(&self.charts),
                    menu: self.menu.clone(),
                    settings,
                    rake_rate: self.state.config.rake_rate,
                    rake_cap: self.state.config.rake_cap,
                };
                self.review = Some(Review::start(setup, self.snapshot()));
            }
        }
    }

    fn on_finish(&mut self) {
        // Deal out remaining board cards only for display when all-in? The
        // engine already ran out the board for all-in hands; nothing to do.
        let st = self.state.settle();
        if st.showdown {
            for p in self.state.players_in_hand() {
                let h = self.state.hole[p];
                let cat = st.hand_values[p].map(holdem::HandCategory::of).map(|c| c.name()).unwrap_or("");
                let text = format!("{} shows {} {} ({cat})", self.seat_name(p), h[0], h[1]);
                self.push_log(Some(p), text);
            }
        }
        for p in 0..self.state.num_players() {
            if st.payouts[p] > 0 {
                let text = format!("{} wins {}", self.seat_name(p), bb(st.payouts[p]));
                self.push_log(Some(p), text);
            }
        }
    }

    fn after_change(&self) {
        if let Some(r) = &self.review {
            r.update(self.snapshot());
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot { state: self.state.clone(), preflop: self.preflop.clone() }
    }

    pub fn settlement(&self) -> Option<Settlement> {
        if self.is_finished() {
            Some(self.state.settle())
        } else {
            None
        }
    }

    /// Hero's net result in chips (once finished).
    pub fn hero_net(&self) -> Option<Chips> {
        self.settlement().map(|s| s.net[self.hero])
    }

    pub fn review(&self) -> Option<&Review> {
        self.review.as_ref()
    }

    pub fn villain_profile(&self) -> Option<&Profile> {
        self.profiles.iter().flatten().next()
    }

    /// Whether the hero has folded (the rest of the hand can be skipped).
    pub fn hero_folded(&self) -> bool {
        self.state.folded[self.hero]
    }

    /// Short description of the opposition: the villain's style heads-up or
    /// in a heads-up pot, otherwise the number of opponents.
    pub fn opponents_label(&self) -> String {
        let n = self.state.num_players();
        let style = |s: usize| self.profiles[s].as_ref().map_or("?".to_string(), |p| p.name.clone());
        if n == 2 {
            return style(1 - self.hero);
        }
        let live: Vec<usize> = self.state.players_in_hand().into_iter().filter(|&s| s != self.hero).collect();
        if live.len() == 1 && !self.hero_folded() && self.state.board.len() >= 3 {
            format!("{} {}", self.state.position_name(live[0]), style(live[0]))
        } else {
            format!("{} opponents", n - 1)
        }
    }

    /// Runs AI actions and deals cards until the hero must act or the hand
    /// ends (no animation). Returns the last step result.
    pub fn run_until_hero(&mut self) -> StepResult {
        loop {
            match self.step() {
                StepResult::HeroToAct => return StepResult::HeroToAct,
                StepResult::Finished => return StepResult::Finished,
                _ => {}
            }
        }
    }
}
