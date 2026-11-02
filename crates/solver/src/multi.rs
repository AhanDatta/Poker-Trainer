//! Approximate three-player postflop solver for multiway pots.
//!
//! Exact multiway solving is out of reach on a desktop, so this module runs
//! DCFR on a three-player game with a few deliberate approximations:
//! * the two opponents of a hand are treated as independent: card removal
//!   between the traverser and each opponent is exact, but the opponents'
//!   hands are not checked against each other. Terminal values then factor
//!   into per-opponent sweeps: `P(win) = P(beat a) * P(beat b)`;
//! * ties against an opponent count half a win;
//! * callers usually build trees with sampled runouts and simplified later
//!   streets (see the trainer's review settings).
//!
//! CFR has no equilibrium guarantee with three players; in practice (as in
//! Pluribus) it produces sensible strategies. Convergence is reported as
//! the average best-response gain (NashConv / 3).
//!
//! A pot that loses a player is still a three-player game here (the folded
//! player's money is dead), so the tree handles 2 or 3 active players.

use crate::sampling::sample_cards;
use crate::{ChanceMode, Progress, SolveOptions, SolverError, Step};
use holdem::combo::{combo_cards, NUM_COMBOS};
use holdem::eval::EvalMask;
use holdem::menu::postflop_actions;
use holdem::{Action, Card, CardSet, Chips, PostflopMenu, Situation, Street};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Number of players.
pub const N: usize = 3;
const NONE: u32 = u32::MAX;

/// Betting state of a three-player street (player 0 acts first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiState {
    pub street: Street,
    pub pot: Chips,
    pub commit: [Chips; N],
    pub stack: [Chips; N],
    pub folded: [bool; N],
    pub current_bet: Chips,
    pub last_raise_inc: Chips,
    pub num_raises: u32,
    pub to_act: usize,
    pub needs: [bool; N],
}

enum Next {
    Continue(MultiState),
    Won(usize),
    StreetOver(MultiState),
}

impl MultiState {
    /// State at the start of a street: first active player with chips acts.
    pub fn street_start(
        street: Street,
        pot: Chips,
        stack: [Chips; N],
        folded: [bool; N],
        big_blind: Chips,
    ) -> MultiState {
        let needs = [0, 1, 2].map(|i| !folded[i] && stack[i] > 0);
        MultiState {
            street,
            pot,
            commit: [0; N],
            stack,
            folded,
            current_bet: 0,
            last_raise_inc: big_blind,
            num_raises: 0,
            to_act: (0..N).find(|&i| needs[i]).unwrap_or(0),
            needs,
        }
    }

    pub fn active(&self) -> usize {
        self.folded.iter().filter(|&&f| !f).count()
    }

    pub fn situation(&self, big_blind: Chips) -> Situation {
        let p = self.to_act;
        let opp_max = (0..N)
            .filter(|&q| q != p && !self.folded[q])
            .map(|q| self.commit[q] + self.stack[q])
            .max()
            .unwrap_or(0);
        Situation {
            street: self.street,
            pot: self.pot,
            to_call: (self.current_bet - self.commit[p]).max(0).min(self.stack[p]),
            current_bet: self.current_bet,
            max_to: (self.commit[p] + self.stack[p]).min(opp_max),
            min_raise_to: self.current_bet + self.last_raise_inc.max(big_blind),
            num_raises: self.num_raises,
            limpers: 0,
            big_blind,
        }
    }

    /// The state after `action` when the betting round continues.
    pub fn after(&self, action: Action) -> Option<MultiState> {
        match self.apply(action).0 {
            Next::Continue(s) => Some(s),
            _ => None,
        }
    }

    fn apply(&self, action: Action) -> (Next, Chips) {
        let p = self.to_act;
        let mut s = *self;
        let mut put = 0;
        match action {
            Action::Fold => s.folded[p] = true,
            Action::Check => {}
            Action::Call => put = (s.current_bet - s.commit[p]).max(0).min(s.stack[p]),
            Action::Raise(to) => {
                put = to - s.commit[p];
                let inc = to - s.current_bet;
                if inc >= s.last_raise_inc {
                    s.last_raise_inc = inc;
                }
                s.current_bet = to;
                s.num_raises += 1;
                for q in 0..N {
                    if q != p && !s.folded[q] && s.stack[q] > 0 {
                        s.needs[q] = true;
                    }
                }
            }
        }
        s.commit[p] += put;
        s.stack[p] -= put;
        s.pot += put;
        s.needs[p] = false;
        if s.active() == 1 {
            let w = (0..N).find(|&q| !s.folded[q]).unwrap();
            return (Next::Won(w), put);
        }
        match (1..N).map(|k| (p + k) % N).find(|&q| s.needs[q] && !s.folded[q]) {
            Some(q) => {
                s.to_act = q;
                (Next::Continue(s), put)
            }
            None => (Next::StreetOver(s), put),
        }
    }
}

/// Inputs of a three-player solve (players in postflop order).
#[derive(Clone, Debug)]
pub struct MultiConfig {
    pub board: Vec<Card>,
    pub ranges: [Vec<f32>; N],
    pub root: MultiState,
    pub menu: PostflopMenu,
    pub future_menu: Option<PostflopMenu>,
    pub big_blind: Chips,
    pub chance: ChanceMode,
    pub prune: f32,
    pub keep: [Vec<usize>; N],
    pub max_memory_bytes: u64,
    pub rake_rate: f64,
    pub rake_cap: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Player(u8),
    Chance,
    Won(u8),
    Showdown,
}

struct Node {
    kind: Kind,
    actions: Vec<Action>,
    cards: Vec<Card>,
    children: Vec<Node>,
    invest: [i32; N],
    folded: [bool; N],
    table: u32,
    rake: f32,
    factor: f32,
    regrets: Vec<f32>,
    strat: Vec<f32>,
}

impl Node {
    fn terminal(kind: Kind, invest: [i32; N], folded: [bool; N], rake: f32) -> Node {
        Node {
            kind,
            actions: Vec::new(),
            cards: Vec::new(),
            children: Vec::new(),
            invest,
            folded,
            table: 0,
            rake,
            factor: 1.0,
            regrets: Vec::new(),
            strat: Vec::new(),
        }
    }
}

/// Hand lists of the three players.
struct Hands {
    combos: [Vec<u16>; N],
    cards: [Vec<(u8, u8)>; N],
    weights: [Vec<f32>; N],
    lookup: [Vec<u32>; N],
    blocked: [Vec<Vec<u32>>; N],
    /// `same[p][q][i]`: index in `q`'s list of `p`'s hand `i` (or NONE).
    same: [[Vec<u32>; N]; N],
}

impl Hands {
    fn new(ranges: &[Vec<f32>; N], board: CardSet, prune: f32, keep: &[Vec<usize>; N]) -> Hands {
        let mut combos: [Vec<u16>; N] = Default::default();
        let mut cards: [Vec<(u8, u8)>; N] = Default::default();
        let mut weights: [Vec<f32>; N] = Default::default();
        for p in 0..N {
            let max_w = ranges[p].iter().copied().fold(0f32, f32::max);
            for c in 0..NUM_COMBOS {
                let (a, b) = combo_cards(c);
                if board.contains(a) || board.contains(b) {
                    continue;
                }
                let w = ranges[p][c];
                if w > 0.0 && (w >= prune * max_w || keep[p].contains(&c)) {
                    combos[p].push(c as u16);
                    cards[p].push((a.index() as u8, b.index() as u8));
                    weights[p].push(w);
                }
            }
        }
        let mut lookup = [vec![NONE; NUM_COMBOS], vec![NONE; NUM_COMBOS], vec![NONE; NUM_COMBOS]];
        let mut blocked: [Vec<Vec<u32>>; N] = Default::default();
        for p in 0..N {
            blocked[p] = vec![Vec::new(); 52];
            for (i, &c) in combos[p].iter().enumerate() {
                lookup[p][c as usize] = i as u32;
                let (a, b) = cards[p][i];
                blocked[p][a as usize].push(i as u32);
                blocked[p][b as usize].push(i as u32);
            }
        }
        let mut same: [[Vec<u32>; N]; N] = Default::default();
        for p in 0..N {
            for q in 0..N {
                same[p][q] = combos[p].iter().map(|&c| lookup[q][c as usize]).collect();
            }
        }
        Hands { combos, cards, weights, lookup, blocked, same }
    }

    fn len(&self, p: usize) -> usize {
        self.combos[p].len()
    }

    /// For each hand of `p`: `q`'s total reach over compatible hands.
    fn compat(&self, p: usize, q: usize, r: &[f32]) -> Vec<f64> {
        let mut total = 0f64;
        let mut minus = [0f64; 52];
        for (&(a, b), &w) in self.cards[q].iter().zip(r) {
            if w != 0.0 {
                let w = w as f64;
                total += w;
                minus[a as usize] += w;
                minus[b as usize] += w;
            }
        }
        (0..self.len(p))
            .map(|i| {
                let (a, b) = self.cards[p][i];
                let s = match self.same[p][q][i] {
                    NONE => 0.0,
                    j => r[j as usize] as f64,
                };
                (total - minus[a as usize] - minus[b as usize] + s).max(0.0)
            })
            .collect()
    }
}

/// Hands of the three players sorted by strength on one final board.
struct Table {
    lists: [Vec<(u32, u32, u8, u8)>; N],
}

impl Table {
    fn new(hands: &Hands, board: &[Card]) -> Table {
        let mask = EvalMask::from_cards(board);
        let set = CardSet::from_cards(board);
        let mut lists: [Vec<(u32, u32, u8, u8)>; N] = Default::default();
        for p in 0..N {
            let mut v: Vec<(u32, u32, u8, u8)> = hands.cards[p]
                .iter()
                .enumerate()
                .filter_map(|(i, &(a, b))| {
                    let (ca, cb) = (Card::from_index(a), Card::from_index(b));
                    (!set.contains(ca) && !set.contains(cb))
                        .then(|| (mask.add(ca).add(cb).evaluate(), i as u32, a, b))
                })
                .collect();
            v.sort_unstable();
            lists[p] = v;
        }
        Table { lists }
    }

    /// For each hand of `p`: `q`'s compatible reach over weaker hands plus
    /// half of the equal ones.
    fn beat(&self, hands: &Hands, p: usize, q: usize, r: &[f32]) -> Vec<f64> {
        let mut out = vec![0f64; hands.len(p)];
        let (mine, theirs) = (&self.lists[p], &self.lists[q]);
        let mut sum = 0f64;
        let mut minus = [0f64; 52];
        let (mut i, mut j) = (0, 0);
        while i < mine.len() {
            let s = mine[i].0;
            while j < theirs.len() && theirs[j].0 < s {
                let (_, idx, a, b) = theirs[j];
                let w = r[idx as usize] as f64;
                if w != 0.0 {
                    sum += w;
                    minus[a as usize] += w;
                    minus[b as usize] += w;
                }
                j += 1;
            }
            let mut eq = 0f64;
            let mut eq_minus = [0f64; 52];
            let mut jj = j;
            while jj < theirs.len() && theirs[jj].0 == s {
                let (_, idx, a, b) = theirs[jj];
                let w = r[idx as usize] as f64;
                eq += w;
                eq_minus[a as usize] += w;
                eq_minus[b as usize] += w;
                jj += 1;
            }
            while i < mine.len() && mine[i].0 == s {
                let (_, h, a, b) = mine[i];
                let same = match hands.same[p][q][h as usize] {
                    NONE => 0.0,
                    k => r[k as usize] as f64,
                };
                let weaker = sum - minus[a as usize] - minus[b as usize];
                let equal = eq - eq_minus[a as usize] - eq_minus[b as usize] + same;
                out[h as usize] = (weaker + 0.5 * equal).max(0.0);
                i += 1;
            }
        }
        out
    }
}

struct Builder<'a> {
    cfg: &'a MultiConfig,
    hands: &'a Hands,
    boards: Vec<Vec<Card>>,
    board_index: std::collections::HashMap<u64, u32>,
    cells: u64,
    rng: StdRng,
}

impl Builder<'_> {
    fn menu(&self, street: Street) -> &PostflopMenu {
        if street == self.cfg.root.street {
            &self.cfg.menu
        } else {
            self.cfg.future_menu.as_ref().unwrap_or(&self.cfg.menu)
        }
    }

    fn rake(&self, pot: Chips) -> f32 {
        (pot as f64 * self.cfg.rake_rate).min(self.cfg.rake_cap) as f32
    }

    fn player(
        &mut self,
        st: &MultiState,
        invest: [i32; N],
        dealt: &mut Vec<Card>,
    ) -> Result<Node, SolverError> {
        let actions = postflop_actions(&st.situation(self.cfg.big_blind), self.menu(st.street));
        let p = st.to_act;
        let len = actions.len() * self.hands.len(p);
        self.cells += len as u64;
        if self.cells * 8 > self.cfg.max_memory_bytes {
            return Err(SolverError::TooLarge { needed: self.cells * 8, limit: self.cfg.max_memory_bytes });
        }
        let mut children = Vec::with_capacity(actions.len());
        for &a in &actions {
            let (next, put) = st.apply(a);
            let mut inv = invest;
            inv[p] += put as i32;
            children.push(match next {
                Next::Continue(s2) => self.player(&s2, inv, dealt)?,
                Next::Won(w) => {
                    let mut s2 = *st;
                    s2.folded[p] = true;
                    s2.commit[p] += put;
                    // The uncalled part of the winner's bet isn't raked.
                    let others = (0..N).filter(|&j| j != w).map(|j| s2.commit[j]).max().unwrap_or(0);
                    let uncalled = (s2.commit[w] - others).max(0);
                    Node::terminal(Kind::Won(w as u8), inv, s2.folded, self.rake(st.pot + put - uncalled))
                }
                Next::StreetOver(s2) => self.after_street(&s2, inv, dealt)?,
            });
        }
        Ok(Node {
            kind: Kind::Player(p as u8),
            actions,
            cards: Vec::new(),
            children,
            invest,
            folded: st.folded,
            table: 0,
            rake: 0.0,
            factor: 1.0,
            regrets: vec![0.0; len],
            strat: vec![0.0; len],
        })
    }

    fn after_street(
        &mut self,
        st: &MultiState,
        invest: [i32; N],
        dealt: &mut Vec<Card>,
    ) -> Result<Node, SolverError> {
        if st.street == Street::River {
            let mut node = Node::terminal(Kind::Showdown, invest, st.folded, self.rake(st.pot));
            node.table = self.table_for(dealt);
            return Ok(node);
        }
        let can_bet = (0..N).filter(|&q| !st.folded[q] && st.stack[q] > 0).count();
        let runout = can_bet < 2;
        let next = st.street.next().unwrap();
        let mut board = self.cfg.board.clone();
        board.extend_from_slice(dealt);
        let used = CardSet::from_cards(&board);
        let available: Vec<Card> = Card::all().filter(|c| !used.contains(*c)).collect();
        let k = match (runout, self.cfg.chance, next) {
            (false, ChanceMode::Sampled { turn, .. }, Street::Turn) => turn,
            (false, ChanceMode::Sampled { river, .. }, _) => river,
            _ => available.len(),
        };
        let cards =
            if k < available.len() { sample_cards(&available, k, &mut self.rng) } else { available.clone() };
        let compatible = (52 - board.len() - 2 * N) as f32;
        let factor = 1.0 / (cards.len() as f32 * compatible / available.len() as f32);
        let mut children = Vec::with_capacity(cards.len());
        for &c in &cards {
            dealt.push(c);
            let child = if runout {
                let mut s2 = *st;
                s2.street = next;
                self.after_street(&s2, invest, dealt)?
            } else {
                let s2 = MultiState::street_start(next, st.pot, st.stack, st.folded, self.cfg.big_blind);
                self.player(&s2, invest, dealt)?
            };
            dealt.pop();
            children.push(child);
        }
        Ok(Node {
            kind: Kind::Chance,
            actions: Vec::new(),
            cards,
            children,
            invest,
            folded: st.folded,
            table: 0,
            rake: 0.0,
            factor,
            regrets: Vec::new(),
            strat: Vec::new(),
        })
    }

    fn table_for(&mut self, dealt: &[Card]) -> u32 {
        let key = CardSet::from_cards(dealt).0;
        if let Some(&i) = self.board_index.get(&key) {
            return i;
        }
        let i = self.boards.len() as u32;
        let mut board = self.cfg.board.clone();
        board.extend_from_slice(dealt);
        self.boards.push(board);
        self.board_index.insert(key, i);
        i
    }
}

#[derive(Clone, Copy)]
struct Discount {
    pos: f32,
    neg: f32,
    strat: f32,
}

struct Ctx<'a> {
    hands: &'a Hands,
    tables: &'a [Table],
    third_pot: f64,
    root_pot: f64,
    norm: [f64; N],
}

fn normalize(s: &mut [f32], na: usize, n: usize) {
    if n == 0 {
        return;
    }
    for h in 0..n {
        let d: f32 = (0..na).map(|a| s[a * n + h]).sum();
        for a in 0..na {
            s[a * n + h] = if d > 0.0 { s[a * n + h] / d } else { 1.0 / na as f32 };
        }
    }
}

fn current_strategy(node: &Node, n: usize) -> Vec<f32> {
    let mut s: Vec<f32> = node.regrets.iter().map(|&r| r.max(0.0)).collect();
    normalize(&mut s, node.actions.len(), n);
    s
}

fn average_strategy(node: &Node, n: usize) -> Vec<f32> {
    let mut s = node.strat.clone();
    normalize(&mut s, node.actions.len(), n);
    s
}

/// Terminal values of `p` (reach-weighted, normalized). With `unit`, every
/// outcome pays 1: the result is the probability mass reaching the node.
fn terminal(ctx: &Ctx, node: &Node, p: usize, reach: &[Vec<f32>; N], unit: bool, out: &mut [f32]) {
    let opps: Vec<usize> = (0..N).filter(|&q| q != p).collect();
    let compat: Vec<Vec<f64>> = opps.iter().map(|&q| ctx.hands.compat(p, q, &reach[q])).collect();
    let n = out.len();
    let (win, lose) = if unit {
        (1.0, 1.0)
    } else {
        (
            ctx.root_pot + (0..N).filter(|&j| j != p).map(|j| node.invest[j] as f64).sum::<f64>()
                - node.rake as f64
                - ctx.third_pot,
            -(node.invest[p] as f64 + ctx.third_pot),
        )
    };
    let norm = ctx.norm[p];
    match node.kind {
        Kind::Won(w) => {
            let pay = if w as usize == p { win } else { lose };
            for h in 0..n {
                out[h] = (pay * compat[0][h] * compat[1][h] * norm) as f32;
            }
        }
        Kind::Showdown => {
            if node.folded[p] {
                for h in 0..n {
                    out[h] = (lose * compat[0][h] * compat[1][h] * norm) as f32;
                }
                return;
            }
            let table = &ctx.tables[node.table as usize];
            let beat: Vec<Vec<f64>> =
                opps.iter()
                    .enumerate()
                    .map(|(k, &q)| {
                        if node.folded[q] {
                            compat[k].clone()
                        } else {
                            table.beat(ctx.hands, p, q, &reach[q])
                        }
                    })
                    .collect();
            for h in 0..n {
                let total = compat[0][h] * compat[1][h];
                let pwin = beat[0][h] * beat[1][h];
                out[h] = ((win * pwin + lose * (total - pwin)) * norm) as f32;
            }
        }
        _ => unreachable!(),
    }
}

/// Opponent reach after a chance card: blocked hands removed, the chance
/// factor applied once (to the first opponent).
fn chance_reach(ctx: &Ctx, p: usize, reach: &[Vec<f32>; N], factor: f32, card: usize) -> [Vec<f32>; N] {
    let mut r = reach.clone();
    let mut first = true;
    for q in (0..N).filter(|&q| q != p) {
        if first {
            r[q].iter_mut().for_each(|x| *x *= factor);
            first = false;
        }
        for &h in &ctx.hands.blocked[q][card] {
            r[q][h as usize] = 0.0;
        }
    }
    r
}

fn sum_rows(rows: Vec<Vec<f32>>, out: &mut [f32]) {
    out.fill(0.0);
    for row in rows {
        for (o, v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
}

/// One DCFR traversal for player `p`: updates `p`'s regrets and, in the
/// traversal of the player after it, each opponent's strategy sums.
fn cfr(ctx: &Ctx, node: &mut Node, p: usize, reach: &[Vec<f32>; N], d: Discount, out: &mut [f32]) {
    let n = out.len();
    match node.kind {
        Kind::Won(_) | Kind::Showdown => terminal(ctx, node, p, reach, false, out),
        Kind::Chance => {
            let factor = node.factor;
            let rows: Vec<Vec<f32>> = node
                .children
                .par_iter_mut()
                .zip(node.cards.par_iter())
                .map(|(child, &card)| {
                    let r = chance_reach(ctx, p, reach, factor, card.index());
                    let mut v = vec![0f32; n];
                    cfr(ctx, child, p, &r, d, &mut v);
                    for &h in &ctx.hands.blocked[p][card.index()] {
                        v[h as usize] = 0.0;
                    }
                    v
                })
                .collect();
            sum_rows(rows, out);
        }
        Kind::Player(q) if q as usize == p => {
            let na = node.actions.len();
            let mut cfv = vec![0f32; na * n];
            for (a, child) in node.children.iter_mut().enumerate() {
                cfr(ctx, child, p, reach, d, &mut cfv[a * n..(a + 1) * n]);
            }
            let s = current_strategy(node, n);
            for h in 0..n {
                out[h] = (0..na).map(|a| s[a * n + h] * cfv[a * n + h]).sum();
            }
            for a in 0..na {
                for h in 0..n {
                    let r = &mut node.regrets[a * n + h];
                    let disc = if *r > 0.0 { d.pos } else { d.neg };
                    *r = *r * disc + cfv[a * n + h] - out[h];
                }
            }
        }
        Kind::Player(q) => {
            let q = q as usize;
            let na = node.actions.len();
            let nq = reach[q].len();
            let s = current_strategy(node, nq);
            // Each player's average strategy is accumulated once per
            // iteration: in the traversal of the next player.
            if (q + 1) % N == p {
                for a in 0..na {
                    for h in 0..nq {
                        let x = &mut node.strat[a * nq + h];
                        *x = *x * d.strat + reach[q][h] * s[a * nq + h];
                    }
                }
            }
            out.fill(0.0);
            let mut tmp = vec![0f32; n];
            for (a, child) in node.children.iter_mut().enumerate() {
                let mut r = reach.clone();
                let mut any = false;
                for h in 0..nq {
                    r[q][h] *= s[a * nq + h];
                    any |= r[q][h] != 0.0;
                }
                if !any {
                    continue;
                }
                cfr(ctx, child, p, &r, d, &mut tmp);
                for (o, v) in out.iter_mut().zip(&tmp) {
                    *o += v;
                }
            }
        }
    }
}

/// Values of `p` when everyone plays the average strategy, or when `p`
/// best-responds (`best_response`). Read-only.
fn eval(ctx: &Ctx, node: &Node, p: usize, reach: &[Vec<f32>; N], best_response: bool, out: &mut [f32]) {
    eval_with(ctx, node, p, reach, best_response, false, out)
}

/// [`eval`] with unit payoffs (`unit`): the probability mass reaching the
/// terminals, used to normalize EVs consistently with the chance model.
fn eval_with(
    ctx: &Ctx,
    node: &Node,
    p: usize,
    reach: &[Vec<f32>; N],
    best_response: bool,
    unit: bool,
    out: &mut [f32],
) {
    let n = out.len();
    match node.kind {
        Kind::Won(_) | Kind::Showdown => terminal(ctx, node, p, reach, unit, out),
        Kind::Chance => {
            let factor = node.factor;
            let rows: Vec<Vec<f32>> = node
                .children
                .par_iter()
                .zip(node.cards.par_iter())
                .map(|(child, &card)| {
                    let r = chance_reach(ctx, p, reach, factor, card.index());
                    let mut v = vec![0f32; n];
                    eval_with(ctx, child, p, &r, best_response, unit, &mut v);
                    for &h in &ctx.hands.blocked[p][card.index()] {
                        v[h as usize] = 0.0;
                    }
                    v
                })
                .collect();
            sum_rows(rows, out);
        }
        Kind::Player(q) if q as usize == p => {
            let na = node.actions.len();
            let mut cfv = vec![0f32; na * n];
            for (a, child) in node.children.iter().enumerate() {
                eval_with(ctx, child, p, reach, best_response, unit, &mut cfv[a * n..(a + 1) * n]);
            }
            if best_response {
                for h in 0..n {
                    out[h] = (0..na).map(|a| cfv[a * n + h]).fold(f32::NEG_INFINITY, f32::max);
                }
            } else {
                let s = average_strategy(node, n);
                for h in 0..n {
                    out[h] = (0..na).map(|a| s[a * n + h] * cfv[a * n + h]).sum();
                }
            }
        }
        Kind::Player(q) => {
            let q = q as usize;
            let nq = reach[q].len();
            let s = average_strategy(node, nq);
            out.fill(0.0);
            let mut tmp = vec![0f32; n];
            for (a, child) in node.children.iter().enumerate() {
                let mut r = reach.clone();
                let mut any = false;
                for h in 0..nq {
                    r[q][h] *= s[a * nq + h];
                    any |= r[q][h] != 0.0;
                }
                if !any {
                    continue;
                }
                eval_with(ctx, child, p, &r, best_response, unit, &mut tmp);
                for (o, v) in out.iter_mut().zip(&tmp) {
                    *o += v;
                }
            }
        }
    }
}

fn make_ctx<'a>(hands: &'a Hands, tables: &'a [Table], config: &MultiConfig) -> Ctx<'a> {
    let mut norm = [0f64; N];
    for p in 0..N {
        let opps: Vec<usize> = (0..N).filter(|&q| q != p).collect();
        let c0 = hands.compat(p, opps[0], &hands.weights[opps[0]]);
        let c1 = hands.compat(p, opps[1], &hands.weights[opps[1]]);
        let total: f64 = (0..hands.len(p)).map(|h| hands.weights[p][h] as f64 * c0[h] * c1[h]).sum();
        norm[p] = if total > 0.0 { 1.0 / total } else { 0.0 };
    }
    Ctx {
        hands,
        tables,
        third_pot: config.root.pot as f64 / N as f64,
        root_pot: config.root.pot as f64,
        norm,
    }
}

/// Strategy and values at a decision node of a three-player solve.
pub type MultiNodeInfo = crate::NodeInfo;

pub struct MultiSolver {
    config: MultiConfig,
    hands: Hands,
    tables: Vec<Table>,
    root: Node,
    iterations: u32,
    convergence: f64,
}

impl MultiSolver {
    pub fn new(config: MultiConfig) -> Result<MultiSolver, SolverError> {
        let n = config.board.len();
        if !(3..=5).contains(&n) || Street::from_board_len(n) != config.root.street {
            return Err(SolverError::InvalidBoard(format!("{n} cards for {:?}", config.root.street)));
        }
        let hands =
            Hands::new(&config.ranges, CardSet::from_cards(&config.board), config.prune, &config.keep);
        for p in 0..N {
            if hands.len(p) == 0 {
                return Err(SolverError::EmptyRange(p));
            }
        }
        let seed = match config.chance {
            ChanceMode::Sampled { seed, .. } => seed,
            ChanceMode::Exact => 0,
        };
        let mut b = Builder {
            cfg: &config,
            hands: &hands,
            boards: Vec::new(),
            board_index: Default::default(),
            cells: 0,
            rng: StdRng::seed_from_u64(seed),
        };
        let root = b.player(&config.root, [0; N], &mut Vec::new())?;
        let boards = std::mem::take(&mut b.boards);
        let tables = boards.par_iter().map(|bd| Table::new(&hands, bd)).collect();
        Ok(MultiSolver { config, hands, tables, root, iterations: 0, convergence: f64::INFINITY })
    }

    fn ctx(&self) -> Ctx<'_> {
        make_ctx(&self.hands, &self.tables, &self.config)
    }

    fn root_values(&self, best_response: bool) -> [f64; N] {
        let ctx = self.ctx();
        let mut v = [0f64; N];
        for p in 0..N {
            let mut out = vec![0f32; self.hands.len(p)];
            eval(&ctx, &self.root, p, &self.hands.weights, best_response, &mut out);
            v[p] = out.iter().zip(&self.hands.weights[p]).map(|(&x, &w)| x as f64 * w as f64).sum();
        }
        v
    }

    /// Average best-response gain over the three players (chips).
    pub fn convergence(&self) -> f64 {
        let br = self.root_values(true);
        let ev = self.root_values(false);
        (0..N).map(|p| br[p] - ev[p]).sum::<f64>() / N as f64
    }

    /// Expected payoff of each player (chips, relative to an even split of
    /// the root pot) under the average strategies.
    pub fn values(&self) -> [f64; N] {
        self.root_values(false)
    }

    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    pub fn solve(&mut self, opts: &SolveOptions, on_progress: impl FnMut(&Progress) -> bool) -> Progress {
        self.solve_cancellable(opts, &std::sync::atomic::AtomicBool::new(false), on_progress)
    }

    /// Like [`Self::solve`], but stops within one iteration once `cancel` is
    /// set (`on_progress` only runs at exploitability checks, which can be
    /// many seconds apart on big trees).
    pub fn solve_cancellable(
        &mut self,
        opts: &SolveOptions,
        cancel: &std::sync::atomic::AtomicBool,
        mut on_progress: impl FnMut(&Progress) -> bool,
    ) -> Progress {
        let pot = self.config.root.pot as f64;
        let check = opts.check_every.max(1);
        let target = opts.target_exploitability_pct * pot / 100.0;
        let mut next_check = self.iterations + check;
        let mut last_check: Option<(u32, f64)> = None;
        let mut progress = Progress {
            iteration: self.iterations,
            max_iterations: opts.max_iterations,
            exploitability: self.convergence,
            exploitability_pct: 100.0 * self.convergence / pot,
            finished: false,
        };
        while self.iterations < opts.max_iterations {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                progress.finished = true;
                return progress;
            }
            self.iterations += 1;
            let t = self.iterations as f64;
            let prev = t - 1.0;
            let pa = prev.powf(1.5);
            let d = Discount { pos: (pa / (pa + 1.0)) as f32, neg: 0.5, strat: (prev / t).powi(2) as f32 };
            let ctx = make_ctx(&self.hands, &self.tables, &self.config);
            for p in 0..N {
                let mut out = vec![0f32; self.hands.len(p)];
                cfr(&ctx, &mut self.root, p, &self.hands.weights, d, &mut out);
            }
            let last = self.iterations >= opts.max_iterations;
            if self.iterations >= next_check || last {
                self.convergence = self.convergence();
                progress.iteration = self.iterations;
                progress.exploitability = self.convergence;
                progress.exploitability_pct = 100.0 * self.convergence / pot;
                let done = progress.exploitability_pct <= opts.target_exploitability_pct || last;
                progress.finished = done;
                if !on_progress(&progress) || done {
                    progress.finished = true;
                    return progress;
                }
                let (t, e) = (self.iterations, self.convergence);
                next_check = t + crate::next_check_gap(last_check, (t, e), target, check);
                last_check = Some((t, e));
            }
        }
        progress.finished = true;
        progress
    }

    /// Follows `path` (actions within the root street, or dealt cards).
    fn walk(&self, path: &[Step]) -> Result<(&Node, [Vec<f32>; N]), SolverError> {
        let mut node = &self.root;
        let mut reach = self.hands.weights.clone();
        for (depth, step) in path.iter().enumerate() {
            match (node.kind, step) {
                (Kind::Player(q), Step::Action(a)) => {
                    let ai = node.actions.iter().position(|x| x == a).ok_or_else(|| {
                        SolverError::BadPath(format!(
                            "action {a:?} not in the tree at depth {depth} ({:?})",
                            node.actions
                        ))
                    })?;
                    let q = q as usize;
                    let n = self.hands.len(q);
                    let s = average_strategy(node, n);
                    for h in 0..n {
                        reach[q][h] *= s[ai * n + h];
                    }
                    node = &node.children[ai];
                }
                (Kind::Chance, Step::Card(c)) => {
                    let ci = node.cards.iter().position(|x| x == c).ok_or_else(|| {
                        SolverError::BadPath(format!("card {c} not in the tree at depth {depth}"))
                    })?;
                    for (x, r) in reach.iter_mut().enumerate() {
                        for &h in &self.hands.blocked[x][c.index()] {
                            r[h as usize] = 0.0;
                        }
                    }
                    node = &node.children[ci];
                }
                _ => {
                    return Err(SolverError::BadPath(format!(
                        "step {step:?} doesn't fit the tree at depth {depth}"
                    )))
                }
            }
        }
        Ok((node, reach))
    }

    /// Strategy and per-action EVs (chips from the decision on) at a node.
    /// `reach_fraction` holds the acting player's and the least-reached
    /// opponent's share of their root range; `invest[0]` is the acting
    /// player's investment since the root.
    pub fn query(&self, path: &[Step]) -> Result<MultiNodeInfo, SolverError> {
        let ctx = self.ctx();
        let (node, reach) = self.walk(path)?;
        let Kind::Player(q) = node.kind else {
            return Err(SolverError::BadPath("not a decision node".into()));
        };
        let q = q as usize;
        let n = self.hands.len(q);
        let na = node.actions.len();
        let opps: Vec<usize> = (0..N).filter(|&x| x != q).collect();
        let mut ev = vec![0f32; na * n];
        for (a, child) in node.children.iter().enumerate() {
            let mut v = vec![0f32; n];
            let mut mass = vec![0f32; n];
            eval_with(&ctx, child, q, &reach, false, false, &mut v);
            eval_with(&ctx, child, q, &reach, false, true, &mut mass);
            for h in 0..n {
                ev[a * n + h] = if mass[h] > 0.0 {
                    (v[h] as f64 / mass[h] as f64 + ctx.third_pot + node.invest[q] as f64) as f32
                } else {
                    0.0
                };
            }
        }
        let frac = |p: usize| {
            let root: f64 = self.hands.weights[p].iter().map(|&x| x as f64).sum();
            let now: f64 = reach[p].iter().map(|&x| x as f64).sum();
            if root > 0.0 {
                now / root
            } else {
                0.0
            }
        };
        Ok(crate::NodeInfo {
            player: q,
            actions: node.actions.clone(),
            combos: self.hands.combos[q].clone(),
            strategy: average_strategy(node, n),
            ev,
            reach: reach[q].clone(),
            reach_fraction: [frac(q), opps.iter().map(|&x| frac(x)).fold(1.0, f64::min)],
            pot: self.config.root.pot + node.invest.iter().map(|&x| x as Chips).sum::<Chips>(),
            invest: [node.invest[q] as Chips, node.invest[opps[0]] as Chips],
        })
    }

    /// Ranges (1326 weights) of the three players after `path`.
    pub fn ranges_after(&self, path: &[Step]) -> Result<[Vec<f32>; N], SolverError> {
        let (_, reach) = self.walk(path)?;
        let mut out = [vec![0f32; NUM_COMBOS], vec![0f32; NUM_COMBOS], vec![0f32; NUM_COMBOS]];
        for p in 0..N {
            for (i, &c) in self.hands.combos[p].iter().enumerate() {
                out[p][c as usize] = reach[p][i];
            }
        }
        Ok(out)
    }

    /// Hand index lookup (combo -> index) for player `p`.
    pub fn has_combo(&self, p: usize, combo: usize) -> bool {
        self.hands.lookup[p][combo] != NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BettingState, SolveConfig, Solver};
    use holdem::card::parse_cards;
    use holdem::{Range, StreetSizes};

    fn menu() -> PostflopMenu {
        let s = StreetSizes::new(&["66%"], &["3x"]);
        PostflopMenu { flop: s.clone(), turn: s.clone(), river: s, max_raises: 2, ..PostflopMenu::default() }
    }

    fn config(board: &str, ranges: [&str; N], folded: [bool; N]) -> MultiConfig {
        let board = parse_cards(board).unwrap();
        let street = Street::from_board_len(board.len());
        MultiConfig {
            board,
            ranges: ranges.map(|r| Range::parse(r).unwrap().into_weights()),
            root: MultiState::street_start(street, 900, [3000, 3000, 3000], folded, 100),
            menu: menu(),
            future_menu: None,
            big_blind: 100,
            chance: ChanceMode::Exact,
            prune: 0.0,
            keep: Default::default(),
            max_memory_bytes: 1 << 30,
            rake_rate: 0.0,
            rake_cap: 0.0,
        }
    }

    /// With the third player already folded holding any two cards, the
    /// three-player solver is exactly the heads-up game.
    #[test]
    fn matches_heads_up_when_third_player_folded() {
        let r0 = "AA, KK, QJs, T9s, 76s, A5s";
        let r1 = "AK, QQ, JJ, KQs, 98s, 65s";
        let mut m = MultiSolver::new(config("Kd9s5h2c2d", [r0, r1, "any"], [false, false, true])).unwrap();
        let hu_cfg = SolveConfig::new(
            parse_cards("Kd9s5h2c2d").unwrap(),
            [Range::parse(r0).unwrap().into_weights(), Range::parse(r1).unwrap().into_weights()],
            BettingState::street_start(Street::River, 900, [3000, 3000], 100),
            menu(),
        );
        let mut h = Solver::new(hu_cfg).unwrap();
        // The update order differs (three traversals per iteration), so
        // compare converged solutions.
        let opts = SolveOptions { max_iterations: 600, target_exploitability_pct: 0.0, check_every: 600 };
        m.solve(&opts, |_| true);
        h.solve(&opts, |_| true);
        for path in [vec![], vec![Step::Action(Action::Check)]] {
            let (qm, qh) = (m.query(&path).unwrap(), h.query(&path).unwrap());
            assert_eq!(qm.actions, qh.actions);
            assert_eq!(qm.combos, qh.combos);
            for (a, b) in qm.range_strategy().iter().zip(qh.range_strategy()) {
                assert!((a - b).abs() < 0.01, "range frequency {a} vs {b}");
            }
            let ds = qm.strategy.iter().zip(&qh.strategy).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(ds < 0.05, "strategy differs by {ds}");
            // EVs of actions that are actually played (others are poorly
            // determined by any CFR solve).
            let de = (0..qm.ev.len())
                .filter(|&k| qm.strategy[k] > 0.05 && qh.strategy[k] > 0.05)
                .map(|k| (qm.ev[k] - qh.ev[k]).abs())
                .fold(0.0, f32::max);
            assert!(de < 1.5, "EV differs by {de} chips (pot 900)");
        }
    }

    /// A real three-way turn spot converges and gives well-formed answers.
    #[test]
    fn three_way_turn_spot() {
        let mut cfg = config(
            "Jh8d4c2s",
            [
                "JJ, 88, AJ, KJ, QTs, T9s, A4s, 65s",
                "AA, KK, AJ, QJ, JT, 99, 77, KQs",
                "TT, 99, A8s, 98s, 87s, 76s, KJ, QJ",
            ],
            [false; N],
        );
        cfg.chance = ChanceMode::Sampled { turn: 0, river: 12, seed: 1 };
        let mut m = MultiSolver::new(cfg).unwrap();
        let opts = SolveOptions { max_iterations: 150, target_exploitability_pct: 1.0, check_every: 25 };
        let mut history = Vec::new();
        let p = m.solve(&opts, |p| {
            history.push(p.exploitability_pct);
            true
        });
        assert!(history.first().unwrap() > history.last().unwrap(), "{history:?}");
        assert!(p.exploitability_pct < 5.0, "{history:?}");
        // Treating the opponents as independent makes the game only
        // approximately zero-sum.
        let v = m.values();
        assert!(v.iter().sum::<f64>().abs() < 0.03 * 900.0, "zero-sum: {v:?}");
        for path in [
            vec![],
            vec![Step::Action(Action::Check)],
            vec![Step::Action(Action::Check), Step::Action(Action::Check)],
        ] {
            let q = m.query(&path).unwrap();
            for h in 0..q.num_hands() {
                let mix: f32 = q.hand_strategy(h).iter().sum();
                assert!((mix - 1.0).abs() < 1e-3);
            }
        }
        // Facing a bet: folding is worth exactly zero.
        let bet = m.query(&[]).unwrap().actions[1];
        let q = m.query(&[Step::Action(bet)]).unwrap();
        assert_eq!(q.actions[0], Action::Fold);
        for h in 0..q.num_hands() {
            assert!(q.hand_ev(h)[0].abs() < 1e-3);
        }
        let r = m.ranges_after(&[Step::Action(bet)]).unwrap();
        assert!(r.iter().all(|v| v.iter().all(|&x| (0.0..=1.0 + 1e-6).contains(&x))));
    }
}

impl MultiSolver {
    /// Number of `(action, hand)` cells in the tree (memory ≈ 8 bytes each).
    pub fn cells(&self) -> u64 {
        fn walk(n: &Node) -> u64 {
            n.regrets.len() as u64 + n.children.iter().map(walk).sum::<u64>()
        }
        walk(&self.root)
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;
    use holdem::card::parse_cards;
    use holdem::{Range, StreetSizes};
    use std::time::Instant;

    /// Prints tree sizes and solve times of typical 3-way flop setups.
    #[test]
    #[ignore]
    fn three_way_flop_sizes() {
        let ranges = [
            "22-99, A2s-A9s, ATo-AJo, K9s-KJs, KTo+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s",
            "22+, A2s+, ATo+, K9s+, KJo+, Q9s+, QJo, J9s+, T9s, 98s, 87s, 76s",
            "22-JJ, A9s-AQs, AJo-AQo, KTs+, KQo, QTs+, JTs, T9s, 98s",
        ];
        let board = parse_cards("Ks7d2c").unwrap();
        for (label, future_raise, cap, chance) in [
            ("future 1 size, no raises, 6x4", "3x", 1, ChanceMode::Sampled { turn: 6, river: 4, seed: 1 }),
            (
                "future 1 size, allin raise, 8x6",
                "allin",
                2,
                ChanceMode::Sampled { turn: 8, river: 6, seed: 1 },
            ),
            ("future 1 size, no raises, 12x6", "3x", 1, ChanceMode::Sampled { turn: 12, river: 6, seed: 1 }),
        ] {
            let future = PostflopMenu {
                turn: StreetSizes::new(&["66%"], &[future_raise]),
                river: StreetSizes::new(&["75%"], &[future_raise]),
                max_raises: cap,
                ..PostflopMenu::default()
            };
            let cfg = MultiConfig {
                board: board.clone(),
                ranges: ranges.map(|r| Range::parse(r).unwrap().into_weights()),
                root: MultiState::street_start(Street::Flop, 800, [9750, 9750, 9750], [false; N], 100),
                menu: PostflopMenu { max_raises: 2, ..PostflopMenu::default() },
                future_menu: Some(future),
                big_blind: 100,
                chance,
                prune: 0.0,
                keep: Default::default(),
                max_memory_bytes: 8 << 30,
                rake_rate: 0.0,
                rake_cap: 0.0,
            };
            let t = Instant::now();
            let mut m = MultiSolver::new(cfg).unwrap();
            let cells = m.cells();
            let opts = SolveOptions { max_iterations: 300, target_exploitability_pct: 1.0, check_every: 20 };
            let p = m.solve(&opts, |_| true);
            println!(
                "{label:<36} cells {cells:>10} (~{:.2} GB) iters {} conv {:.2}% pot, {:.1}s",
                cells as f64 * 8.0 / 1e9,
                p.iteration,
                p.exploitability_pct,
                t.elapsed().as_secs_f64()
            );
        }
    }
}
