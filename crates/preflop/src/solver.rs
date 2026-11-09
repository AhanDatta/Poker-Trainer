//! External-sampling Monte Carlo CFR over the preflop tree.
//!
//! Each deal gives every seat real hole cards and a full board. For each
//! seat in turn (the traverser) the tree is walked once: at the traverser's
//! decisions every action is tried and its regret updated; at the other
//! seats' decisions one action is sampled from the current strategy and
//! counted. The counts are the average strategy, which is what converges.
//! A player's information is the decision and its hand class, so there are
//! 169 strategies per decision.
//!
//! Regrets and counts are shared between threads without locks; a lost
//! update now and then doesn't matter to a sampling method.

use crate::deal::Deal;
use crate::realize::{Context, Realization};
use crate::rng::Rng;
use crate::spec::{BB, MAX_PLAYERS};
use crate::tree::{Decision, LeafKind, Node, Tree, NO_CHILD};
use holdem::NUM_CLASSES;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use trainer::preflop::{Freqs, PreAction};

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Worker threads. One thread makes a solve reproducible from the seed.
    pub threads: usize,
    pub seed: u64,
    /// Regrets are kept above this (in big blinds), so an action that was
    /// bad for a long time can come back when the opponents change.
    pub regret_floor: f32,
    /// The first calls of [`Solver::run`] that are weighted linearly
    /// (earlier ones count less), which forgets the random early play.
    pub warmup_epochs: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options { threads: 1, seed: 1, regret_floor: -300_000.0, warmup_epochs: 10 }
    }
}

/// The tree in the compact form the walk uses, with payoffs in big blinds.
enum Flat {
    Decision {
        seat: u8,
        num_actions: u8,
        /// The first action is folding (worth `fold_value` to the seat).
        has_fold: bool,
        fold_value: f32,
        kids: [u32; 4],
        /// First cell of the decision: `cells + class * num_actions + action`.
        cells: usize,
    },
    Leaf {
        live: u16,
        /// What the players in the hand play for (rake already taken).
        prize: f32,
        commit: [f32; MAX_PLAYERS],
        /// Realization context per seat, or `NO_FACTOR`.
        context: [u8; MAX_PLAYERS],
    },
}

const NO_FACTOR: u8 = u8::MAX;

pub struct Solver {
    tree: Tree,
    model: Realization,
    options: Options,
    nodes: Vec<Flat>,
    /// First cell of each decision, by [`Decision::slot`].
    cells: Vec<usize>,
    /// Regret sums as `f32` bits, in big blinds.
    regret: Vec<AtomicU32>,
    /// How often each action was sampled: the average strategy.
    visits: Vec<AtomicU32>,
    deals: u64,
    epochs: u32,
}

impl Solver {
    pub fn new(tree: Tree, model: Realization, options: Options) -> Solver {
        let players = tree.spec.players;
        let bb = BB as f32;
        let mut cells = Vec::with_capacity(tree.num_decisions);
        let mut total = 0usize;
        let nodes = tree
            .nodes
            .iter()
            .map(|node| match node {
                Node::Decision(d) => {
                    let mut kids = [NO_CHILD; 4];
                    for (i, a) in d.actions().enumerate() {
                        kids[i] = d.children[a.index()];
                    }
                    let first = total;
                    cells.push(first);
                    total += NUM_CLASSES * d.num_actions();
                    Flat::Decision {
                        seat: d.seat,
                        num_actions: d.num_actions() as u8,
                        has_fold: d.children[PreAction::Fold.index()] != NO_CHILD,
                        fold_value: -(d.commit as f32) / bb,
                        kids,
                        cells: first,
                    }
                }
                Node::Leaf(l) => {
                    let flop = l.kind == LeafKind::Flop;
                    let raked = match l.kind {
                        LeafKind::Won(_) => false,
                        LeafKind::AllIn => true,
                        LeafKind::Flop => !model.includes_rake,
                    };
                    let mut context = [NO_FACTOR; MAX_PLAYERS];
                    for s in (0..players).filter(|&s| flop && l.is_live(s)) {
                        context[s] = Context::of(l, s, players).index() as u8;
                    }
                    Flat::Leaf {
                        live: l.live,
                        prize: (l.pot - if raked { l.rake } else { 0 }) as f32 / bb,
                        commit: l.commit.map(|c| c as f32 / bb),
                        context,
                    }
                }
            })
            .collect();
        let zeros = |n: usize| (0..n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        Solver {
            tree,
            model,
            options: Options { threads: options.threads.max(1), ..options },
            nodes,
            cells,
            regret: zeros(total),
            visits: zeros(total),
            deals: 0,
            epochs: 0,
        }
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Deals played so far.
    pub fn deals(&self) -> u64 {
        self.deals
    }

    /// Number of `(decision, hand class, action)` cells.
    pub fn num_cells(&self) -> usize {
        self.regret.len()
    }

    /// Plays `deals` more deals (one epoch).
    pub fn run(&mut self, deals: u64) {
        let threads = self.options.threads as u64;
        let epoch = self.epochs as u64;
        let this = &*self;
        let work = |thread: u64| {
            let share = deals / threads + u64::from(thread < deals % threads);
            let mut rng = Rng::stream(this.options.seed, epoch * 1024 + thread);
            for _ in 0..share {
                let deal = Deal::sample(this.tree.spec.players, &mut rng);
                for seat in 0..this.tree.spec.players {
                    this.walk(0, seat, &deal, &mut rng);
                }
            }
        };
        if threads == 1 {
            work(0);
        } else {
            std::thread::scope(|scope| {
                for thread in 0..threads {
                    scope.spawn(move || work(thread));
                }
            });
        }
        self.deals += deals;
        self.epochs += 1;
        if self.epochs <= self.options.warmup_epochs {
            let k = self.epochs as f32;
            self.discount(k / (k + 1.0));
        }
    }

    /// Scales every regret and count by `factor`.
    fn discount(&mut self, factor: f32) {
        for cell in &mut self.regret {
            *cell.get_mut() = (f32::from_bits(*cell.get_mut()) * factor).to_bits();
        }
        for cell in &mut self.visits {
            *cell.get_mut() = (*cell.get_mut() as f32 * factor).round() as u32;
        }
    }

    /// The traverser's value (in big blinds) of `node` in `deal`.
    fn walk(&self, node: u32, traverser: usize, deal: &Deal, rng: &mut Rng) -> f32 {
        match &self.nodes[node as usize] {
            Flat::Leaf { live, prize, commit, context } => {
                self.payoff(*live, *prize, commit[traverser], context[traverser], traverser, deal)
            }
            &Flat::Decision { seat, num_actions, has_fold, fold_value, kids, cells } => {
                let n = num_actions as usize;
                let base = cells + deal.class[seat as usize] as usize * n;
                let mut sigma = [0f32; 4];
                self.current(base, &mut sigma[..n]);
                if seat as usize != traverser {
                    let action = sample(&sigma[..n], rng);
                    let count = &self.visits[base + action];
                    count.store(count.load(Relaxed).saturating_add(1), Relaxed);
                    return self.walk(kids[action], traverser, deal, rng);
                }
                let mut value = [0f32; 4];
                let mut expected = 0.0;
                for a in 0..n {
                    value[a] = if a == 0 && has_fold {
                        fold_value
                    } else {
                        self.walk(kids[a], traverser, deal, rng)
                    };
                    expected += sigma[a] * value[a];
                }
                for a in 0..n {
                    let cell = &self.regret[base + a];
                    let regret = f32::from_bits(cell.load(Relaxed)) + value[a] - expected;
                    cell.store(regret.max(self.options.regret_floor).to_bits(), Relaxed);
                }
                expected
            }
        }
    }

    /// Regret matching: the current strategy of the cells from `base`.
    #[inline]
    fn current(&self, base: usize, sigma: &mut [f32]) {
        let mut total = 0.0;
        for (a, s) in sigma.iter_mut().enumerate() {
            *s = f32::from_bits(self.regret[base + a].load(Relaxed)).max(0.0);
            total += *s;
        }
        if total > 0.0 {
            sigma.iter_mut().for_each(|s| *s /= total);
        } else {
            sigma.fill(1.0 / sigma.len() as f32);
        }
    }

    /// The average strategy of `class` at `decision` as frequencies of
    /// `[fold, passive, raise, all-in]`, and how many samples it rests on.
    /// Without samples the frequencies are uniform over the actions.
    pub fn average(&self, decision: &Decision, class: usize) -> (Freqs, u64) {
        let n = decision.num_actions();
        let base = self.cells[decision.slot as usize] + class * n;
        let counts: Vec<u64> = (0..n).map(|a| self.visits[base + a].load(Relaxed) as u64).collect();
        let total: u64 = counts.iter().sum();
        let mut freqs = [0f32; 4];
        for (i, action) in decision.actions().enumerate() {
            freqs[action.index()] = if total > 0 { counts[i] as f32 / total as f32 } else { 1.0 / n as f32 };
        }
        (freqs, total)
    }
}

impl Solver {
    /// The average strategy of the cells from `base` (uniform without
    /// samples).
    fn average_into(&self, base: usize, sigma: &mut [f32]) {
        let mut total = 0.0;
        for (a, s) in sigma.iter_mut().enumerate() {
            *s = self.visits[base + a].load(Relaxed) as f32;
            total += *s;
        }
        if total > 0.0 {
            sigma.iter_mut().for_each(|s| *s /= total);
        } else {
            sigma.fill(1.0 / sigma.len() as f32);
        }
    }

    /// What `seat` nets (in big blinds) at a leaf.
    #[inline]
    fn payoff(&self, live: u16, prize: f32, commit: f32, context: u8, seat: usize, deal: &Deal) -> f32 {
        let factor = if context == NO_FACTOR {
            1.0
        } else {
            self.model.factor(context as usize, deal.class[seat] as usize)
        };
        factor * deal.share(live, seat) * prize - commit
    }
}

/// Samples an index from the distribution `sigma`.
#[inline]
fn sample(sigma: &[f32], rng: &mut Rng) -> usize {
    let mut u = rng.unit();
    for (a, &p) in sigma.iter().enumerate() {
        if u < p {
            return a;
        }
        u -= p;
    }
    // Rounding left a sliver: take the last action that has any weight.
    sigma.iter().rposition(|&p| p > 0.0).unwrap_or(0)
}

mod diagnose;
pub use diagnose::Diagnosis;

#[cfg(test)]
mod tests;
