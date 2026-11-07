//! The leaf kernels' plan data: Rust models of the kernels' sweeps, built
//! only from the plan's lists the way `lanes.cuh`, `exact.cuh` and
//! `independent.cuh` read them, reproduce the CPU's leaf evaluators bit for
//! bit (`exact::showdown`, `exact::fold_values`, `independent_showdown`,
//! `pay_all`); the groups cover every leaf once; corruptions are rejected.

use super::*;
use crate::gpu::multi::plan::leaves::{combo_of, GROUP_FIELDS, NO_POSITION};
use crate::gpu::multi::plan::{SweepLists, LEAF_LANES};
use crate::multi::exact;

/// One list of [`SweepLists`] (the kernels' `HandList`).
struct List<'a> {
    l: &'a SweepLists,
    k: usize,
}

impl List<'_> {
    fn of(l: &SweepLists, order: usize, seat: usize) -> List<'_> {
        List { l, k: order * N + seat }
    }
    fn start(&self) -> usize {
        self.l.off[self.k] as usize
    }
    fn len(&self) -> usize {
        self.l.off[self.k + 1] as usize - self.start()
    }
    fn strength(&self, pos: usize) -> u32 {
        self.l.strength[self.start() + pos]
    }
    fn hand(&self, pos: usize) -> usize {
        self.l.hand[self.start() + pos] as usize
    }
    fn cards(&self, pos: usize) -> (usize, usize) {
        unpack_cards(self.l.cards[self.start() + pos])
    }
    /// `(position, other card)` of the entries holding card `c`.
    fn holding(&self, c: usize) -> Vec<(usize, usize)> {
        let o = &self.l.card_off[self.k * 53..];
        let entries = &self.l.holding[o[c] as usize..o[c + 1] as usize];
        entries.iter().map(|&e| ((e & 0xFFFF) as usize, (e >> 16) as usize)).collect()
    }
    fn combo_pos(&self, a: usize, b: usize) -> Option<usize> {
        if a == b {
            return None;
        }
        let pos = self.l.combo_pos[self.k * 1326 + combo_of(a, b)];
        (pos != NO_POSITION).then_some(pos as usize)
    }
}

fn positive(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// The exact model's per-lane state of one set (`exact::Set`), in f64.
struct Set<'a> {
    lists: [&'a List<'a>; 2],
    reach: [&'a [f32]; 2],
    lo: [usize; 2],
    hi: [usize; 2],
    t: [f64; 2],
    m: [[f64; 52]; 2],
    s: f64,
    same: f64,
    /// `[xqr, xrq]`.
    cross: [[f64; 52]; 2],
    same_card: [f64; 52],
}

impl<'a> Set<'a> {
    fn new(lists: [&'a List<'a>; 2], reach: [&'a [f32]; 2]) -> Set<'a> {
        Set {
            lists,
            reach,
            lo: [0; 2],
            hi: [0; 2],
            t: [0.0; 2],
            m: [[0.0; 52]; 2],
            s: 0.0,
            same: 0.0,
            cross: [[0.0; 52]; 2],
            same_card: [0.0; 52],
        }
    }

    fn clear_at(&mut self, lo: [usize; 2]) {
        let (lists, reach) = (self.lists, self.reach);
        *self = Set::new(lists, reach);
        self.lo = lo;
        self.hi = lo;
    }

    fn w(&self, k: usize, pos: usize) -> f64 {
        self.reach[k][self.lists[k].hand(pos)] as f64
    }

    /// `x_k(u, v)`: reach of opponent `k`'s combo `{u, v}` if in the set.
    fn x(&self, k: usize, u: usize, v: usize) -> f64 {
        match self.lists[k].combo_pos(u, v) {
            Some(pos) if pos >= self.lo[k] && pos < self.hi[k] => self.w(k, pos),
            _ => 0.0,
        }
    }

    /// `by_card`: the present entries of opponent `k` holding card `c`.
    fn holding(&self, k: usize, c: usize) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        for (pos, other) in self.lists[k].holding(c) {
            if pos >= self.hi[k] {
                break;
            }
            if pos < self.lo[k] {
                continue;
            }
            out.push((other, self.w(k, pos)));
        }
        out
    }

    /// `Pairs::add_q` / `add_r`: the entry at position `hi[k]` enters.
    fn add(&mut self, k: usize) {
        let pos = self.hi[k];
        let o = 1 - k;
        let (u, v) = self.lists[k].cards(pos);
        let w = self.w(k, pos);
        let xo = self.x(o, u, v);
        self.s += w * (self.m[o][u] + self.m[o][v]);
        self.cross[o][v] += self.m[o][u] * w;
        self.cross[o][u] += self.m[o][v] * w;
        let sm = w * xo;
        self.same += sm;
        self.same_card[u] += sm;
        self.same_card[v] += sm;
        for c in [u, v] {
            for (a, x) in self.holding(o, c) {
                self.cross[k][a] += w * x;
            }
        }
        self.t[k] += w;
        self.m[k][u] += w;
        self.m[k][v] += w;
        self.hi[k] = pos + 1;
    }

    /// `Pairs::pairs`.
    fn pairs(&self, a: usize, b: usize) -> f64 {
        let mut y = 0.0;
        for (c, x) in self.holding(0, a) {
            y += x * self.x(1, b, c);
        }
        for (c, x) in self.holding(0, b) {
            y += x * self.x(1, a, c);
        }
        let (qab, rab) = (self.x(0, a, b), self.x(1, a, b));
        let (qm, rm) = (&self.m[0], &self.m[1]);
        let tq = self.t[0] - qm[a] - qm[b] + qab;
        let tr = self.t[1] - rm[a] - rm[b] + rab;
        let shared = self.s - self.cross[0][a] - self.cross[0][b] - self.cross[1][a] - self.cross[1][b]
            + self.same_card[a]
            + self.same_card[b]
            + y
            - (qm[a] - qab) * (rm[a] - rab)
            - (qm[b] - qab) * (rm[b] - rab);
        let same = self.same - self.same_card[a] - self.same_card[b] + qab * rab;
        positive(tq * tr - shared + same)
    }
}

/// The exact model's showdown sweep for one lane.
#[allow(clippy::too_many_arguments)]
fn model_exact_showdown(
    lists: &SweepLists,
    order: usize,
    seats: [usize; 3],
    folded: u32,
    reach: [&[f32]; 2],
    pay: [f64; 4],
    hands: usize,
) -> Vec<f32> {
    let [p, q, r] = seats;
    let (lp, lq, lr) = (List::of(lists, order, p), List::of(lists, order, q), List::of(lists, order, r));
    let (q_in, r_in) = (folded >> q & 1 == 0, folded >> r & 1 == 0);
    let [win, lose, split2, split3] = pay;
    let mut lt = Set::new([&lq, &lr], reach);
    let mut eq = Set::new([&lq, &lr], reach);
    if !q_in {
        while lt.hi[0] < lq.len() {
            lt.add(0);
        }
    }
    if !r_in {
        while lt.hi[1] < lr.len() {
            lt.add(1);
        }
    }
    let n = lp.len();
    let (mut partial, mut le) = (vec![0.0; n], vec![0.0; n]);
    let mut i = 0;
    while i < n {
        let s = lp.strength(i);
        if q_in {
            while lt.hi[0] < lq.len() && lq.strength(lt.hi[0]) < s {
                lt.add(0);
            }
        }
        if r_in {
            while lt.hi[1] < lr.len() && lr.strength(lt.hi[1]) < s {
                lt.add(1);
            }
        }
        let start = i;
        let mut alone = Vec::new();
        while i < n && lp.strength(i) == s {
            let (a, b) = lp.cards(i);
            alone.push(lt.pairs(a, b));
            i += 1;
        }
        let count = |l: &List, from: usize, on: bool| if on { (from..l.len()).take_while(|&j| l.strength(j) == s).count() } else { 0 };
        let (eq_q, eq_r) = (count(&lq, lt.hi[0], q_in), count(&lr, lt.hi[1], r_in));
        let tie_both = eq_q > 0 && eq_r > 0;
        if eq_q + eq_r > 0 {
            eq.clear_at(lt.hi);
            for _ in 0..eq_q {
                lt.add(0);
            }
            for _ in 0..eq_r {
                lt.add(1);
            }
            if tie_both {
                for _ in 0..eq_q {
                    eq.add(0);
                }
                for _ in 0..eq_r {
                    eq.add(1);
                }
            }
        }
        for (j, &al) in (start..i).zip(&alone) {
            let (a, b) = lp.cards(j);
            let (l, tied) = if eq_q + eq_r == 0 {
                (al, 0.0)
            } else {
                (lt.pairs(a, b), if tie_both { eq.pairs(a, b) } else { 0.0 })
            };
            let (two, three) = if q_in && r_in { (l - al - tied, tied) } else { (l - al, 0.0) };
            partial[j] = win * al + split2 * two + split3 * three;
            le[j] = l;
        }
    }
    if q_in {
        while lt.hi[0] < lq.len() {
            lt.add(0);
        }
    }
    if r_in {
        while lt.hi[1] < lr.len() {
            lt.add(1);
        }
    }
    let mut out = vec![0f32; hands];
    for j in 0..n {
        let (a, b) = lp.cards(j);
        let total = lt.pairs(a, b);
        out[lp.hand(j)] = (partial[j] + lose * (total - le[j])) as f32;
    }
    out
}

/// The exact model's fixed-payoff sweep: pair mass per hand (hand order).
fn model_exact_pairs(lists: &SweepLists, seats: [usize; 3], reach: [&[f32]; 2]) -> Vec<f64> {
    let order = lists.orders as usize - 1;
    let [p, q, r] = seats;
    let (lp, lq, lr) = (List::of(lists, order, p), List::of(lists, order, q), List::of(lists, order, r));
    let mut lt = Set::new([&lq, &lr], reach);
    while lt.hi[0] < lq.len() {
        lt.add(0);
    }
    while lt.hi[1] < lr.len() {
        lt.add(1);
    }
    (0..lp.len()).map(|h| {
        let (a, b) = lp.cards(h);
        lt.pairs(a, b)
    }).collect()
}

/// The independent model's compatible mass of opponent `k` for every
/// traverser hand (`Hands::compat`): a hand-order sweep, then a lookup per
/// hand.
fn model_compat(lists: &SweepLists, seats: [usize; 3], k: usize, reach: &[f32]) -> Vec<f64> {
    let order = lists.orders as usize - 1;
    let (lp, lo) = (List::of(lists, order, seats[0]), List::of(lists, order, seats[1 + k]));
    let (mut total, mut minus) = (0.0, [0.0; 52]);
    for pos in 0..lo.len() {
        let w = reach[lo.hand(pos)] as f64;
        let (a, b) = lo.cards(pos);
        total += w;
        minus[a] += w;
        minus[b] += w;
    }
    (0..lp.len())
        .map(|h| {
            let (a, b) = lp.cards(h);
            let same = lo.combo_pos(a, b).map_or(0.0, |pos| reach[lo.hand(pos)] as f64);
            positive(total - minus[a] - minus[b] + same)
        })
        .collect()
}

/// The independent model's beaten mass of opponent `k` on board `order`
/// (`Table::beat`): a strength-order sweep, the tied hands summed per level.
fn model_beat(lists: &SweepLists, order: usize, seats: [usize; 3], k: usize, reach: &[f32], hands: usize) -> Vec<f64> {
    let (lp, lo) = (List::of(lists, order, seats[0]), List::of(lists, order, seats[1 + k]));
    let by_hand = List::of(lists, lists.orders as usize - 1, seats[1 + k]);
    let w = |pos: usize| reach[lo.hand(pos)] as f64;
    let mut out = vec![0.0; hands];
    let (mut sum, mut minus) = (0.0, [0.0; 52]);
    let (mut i, mut j) = (0, 0);
    while i < lp.len() {
        let s = lp.strength(i);
        while j < lo.len() && lo.strength(j) < s {
            let (a, b) = lo.cards(j);
            sum += w(j);
            minus[a] += w(j);
            minus[b] += w(j);
            j += 1;
        }
        let mut end = j;
        while end < lo.len() && lo.strength(end) == s {
            end += 1;
        }
        let mut eq = 0.0;
        for e in j..end {
            eq += w(e);
        }
        while i < lp.len() && lp.strength(i) == s {
            let (a, b) = lp.cards(i);
            let (mut ea, mut eb) = (0.0, 0.0);
            for e in j..end {
                let (x, y) = lo.cards(e);
                if x == a || y == a {
                    ea += w(e);
                }
                if x == b || y == b {
                    eb += w(e);
                }
            }
            let same = by_hand.combo_pos(a, b).map_or(0.0, |pos| reach[by_hand.hand(pos)] as f64);
            let weaker = sum - minus[a] - minus[b];
            let equal = eq - ea - eb + same;
            out[lp.hand(i)] = positive(weaker + 0.5 * equal);
            i += 1;
        }
    }
    out
}

/// The independent model's showdown for one lane (`independent_showdown`).
#[allow(clippy::too_many_arguments)]
fn model_independent_showdown(
    lists: &SweepLists,
    order: usize,
    seats: [usize; 3],
    folded: u32,
    reach: [&[f32]; 2],
    win: f64,
    lose: f64,
    hands: usize,
) -> Vec<f32> {
    let c = [model_compat(lists, seats, 0, reach[0]), model_compat(lists, seats, 1, reach[1])];
    let b: Vec<Vec<f64>> = (0..2)
        .map(|k| {
            if folded >> seats[1 + k] & 1 == 1 {
                c[k].clone()
            } else {
                model_beat(lists, order, seats, k, reach[k], hands)
            }
        })
        .collect();
    (0..hands)
        .map(|h| {
            let total = c[0][h] * c[1][h];
            let pwin = b[0][h] * b[1][h];
            (win * pwin + lose * (total - pwin)) as f32
        })
        .collect()
}

fn random_reach(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| if rng.random_bool(0.2) { 0.0 } else { rng.random_range(0.0..1.0f32) }).collect()
}

/// Every test spot under both terminal models.
fn model_spots() -> Vec<(String, MultiSolver)> {
    let mut out = Vec::new();
    for (name, cfg) in spots() {
        for exact in [false, true] {
            let mut cfg = cfg.clone();
            cfg.exact_removal = exact;
            out.push((format!("{name} (exact removal {exact})"), MultiSolver::new(cfg).unwrap()));
        }
    }
    out
}

/// The CPU leaf values for random reach, and the models' on the same:
/// fixed payoffs and every board's showdowns for each folded pattern.
fn check_models(name: &str, m: &MultiSolver, plan: &MultiPlan, rng: &mut StdRng, boards: usize) {
    let ctx = m.ctx();
    let lists = &plan.lists;
    for p in 0..N {
        let (q, r) = opponents(p);
        let nh = ctx.hands.len(p);
        let rq = random_reach(rng, ctx.hands.len(q));
        let rr = random_reach(rng, ctx.hands.len(r));
        let pay = rng.random_range(-300.0..300.0);
        let mut want = vec![0f32; nh];
        pay_all(&ctx, p, &rq, &rr, pay, &mut want);
        let got: Vec<f32> = if ctx.exact {
            model_exact_pairs(lists, [p, q, r], [&rq, &rr]).iter().map(|&v| (pay * v) as f32).collect()
        } else {
            let (cq, cr) = (model_compat(lists, [p, q, r], 0, &rq), model_compat(lists, [p, q, r], 1, &rr));
            (0..nh).map(|h| (pay * cq[h] * cr[h]) as f32).collect()
        };
        assert_eq!(bits(&got), bits(&want), "{name}: player {p}'s fixed payoffs");
        for (order, table) in ctx.tables.iter().enumerate().take(boards) {
            for folded in [0u32, 1 << q, 1 << r] {
                let (win, lose) = (rng.random_range(50.0..900.0), rng.random_range(-500.0..-10.0));
                let split = |k: f64| lose + (win - lose) / k;
                let flags = std::array::from_fn(|s| folded >> s & 1 == 1);
                let mut want = vec![0f32; nh];
                let got = if ctx.exact {
                    exact::showdown(table, flags, p, &rq, &rr, win, lose, &mut want);
                    model_exact_showdown(lists, order, [p, q, r], folded, [&rq, &rr], [win, lose, split(2.0), split(3.0)], nh)
                } else {
                    independent_showdown(ctx.hands, table, flags, p, &rq, &rr, win, lose, &mut want);
                    model_independent_showdown(lists, order, [p, q, r], folded, [&rq, &rr], win, lose, nh)
                };
                assert_eq!(bits(&got), bits(&want), "{name}: player {p}, board {order}, folded {folded:#b}");
            }
        }
    }
}

/// Every board's showdowns and the fixed payoffs of the models match the
/// CPU's evaluators bit for bit, for every traverser and folded pattern, on
/// random reach (with zeros, which the CPU skips).
#[test]
fn sweep_models_match_cpu_evaluators_bit_for_bit() {
    let mut rng = StdRng::seed_from_u64(23);
    for (name, m) in model_spots() {
        let plan = plan_for(&m);
        for _ in 0..3 {
            check_models(&name, &m, &plan, &mut rng, usize::MAX);
        }
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The lists are the CPU's: each board's `Table::lists`, then hand order.
#[test]
fn lists_are_the_cpu_tables() {
    for (name, m) in model_spots() {
        let ctx = m.ctx();
        let plan = plan_for(&m);
        let lists = &plan.lists;
        assert_eq!(lists.orders as usize, ctx.tables.len() + 1, "{name}");
        for seat in 0..N {
            for (order, table) in ctx.tables.iter().enumerate() {
                let l = List::of(lists, order, seat);
                let got: Vec<_> = (0..l.len())
                    .map(|i| (l.strength(i), l.hand(i) as u32, l.cards(i).0 as u8, l.cards(i).1 as u8))
                    .collect();
                assert_eq!(got, table.lists[seat], "{name}: seat {seat}, board {order}");
            }
            let l = List::of(lists, ctx.tables.len(), seat);
            let got: Vec<_> = (0..l.len()).map(|i| (l.cards(i).0 as u8, l.cards(i).1 as u8)).collect();
            assert_eq!(got, ctx.hands.cards[seat], "{name}: seat {seat}, hand order");
        }
    }
}

/// Showdown warps share a board and folded players; fixed jobs share both
/// reach slots; every leaf is in exactly one (`validate` checks the rest).
#[test]
fn groups_cover_every_leaf_once() {
    for (name, m) in model_spots() {
        let plan = plan_for(&m);
        for (p, pp) in plan.players.iter().enumerate() {
            let x = &pp.leaves;
            let mut lanes: Vec<u32> = x.showdown_lanes.clone();
            lanes.sort_unstable();
            assert_eq!(lanes, (0..pp.showdowns.len() as u32).collect::<Vec<_>>(), "{name}: player {p}");
            for g in x.groups.as_chunks::<GROUP_FIELDS>().0 {
                assert!((1..=LEAF_LANES as u32).contains(&g[3]));
            }
            let mut leaves: Vec<u32> = x.job_leaves.clone();
            leaves.sort_unstable();
            assert_eq!(leaves, (0..pp.fixed.len() as u32).collect::<Vec<_>>(), "{name}: player {p}");
            // Jobs are distinct reach pairs.
            let mut pairs: Vec<_> = x.job_slots[0].iter().zip(&x.job_slots[1]).collect();
            let jobs = pairs.len();
            pairs.sort_unstable();
            pairs.dedup();
            assert_eq!(pairs.len(), jobs, "{name}: player {p}: a reach pair in two jobs");
        }
    }
}

/// Every corruption of the leaf data is rejected.
#[test]
fn validate_rejects_corrupted_leaf_data() {
    let mut cfg = spots().into_iter().find(|s| s.0 == "sampled turn").unwrap().1;
    cfg.exact_removal = true;
    let good = plan_for(&MultiSolver::new(cfg).unwrap());
    good.validate().unwrap();
    type Mutation = (&'static str, fn(&mut MultiPlan));
    let mutations: &[Mutation] = &[
        ("orders", |pl| pl.lists.orders += 1),
        ("list offset", |pl| pl.lists.off[3] += 1),
        ("hand", |pl| pl.lists.hand[2] = pl.players[0].hands + 7),
        ("hand swapped", |pl| pl.lists.hand.swap(0, 1)),
        ("cards", |pl| pl.lists.cards[0] ^= 1),
        ("strength order", |pl| pl.lists.strength[0] = u32::MAX),
        ("hand-order strength", |pl| *pl.lists.strength.last_mut().unwrap() = 5),
        ("card offset", |pl| pl.lists.card_off[5] += 1),
        ("card position", |pl| pl.lists.holding[0] = 1000),
        ("card position order", |pl| pl.lists.holding.swap(0, 1)),
        ("card list's other card", |pl| pl.lists.holding[0] ^= 1 << 16),
        ("combo position", |pl| {
            let i = pl.lists.combo_pos.iter().position(|&x| x != NO_POSITION).unwrap();
            pl.lists.combo_pos[i] = NO_POSITION;
        }),
        ("extra combo position", |pl| {
            let i = pl.lists.combo_pos.iter().position(|&x| x == NO_POSITION).unwrap();
            pl.lists.combo_pos[i] = 0;
        }),
        ("group board", |pl| pl.players[0].leaves.groups[0] = pl.num_tables),
        ("group folded", |pl| pl.players[0].leaves.groups[1] ^= 1 << opponents(0).0),
        ("group count", |pl| pl.players[0].leaves.groups[3] = 33),
        ("group start", |pl| pl.players[1].leaves.groups[2] = 1),
        ("lane duplicate", |pl| pl.players[2].leaves.showdown_lanes[1] = pl.players[2].leaves.showdown_lanes[0]),
        ("lane out of range", |pl| pl.players[2].leaves.showdown_lanes[0] = 1 << 30),
        ("split", |pl| pl.players[0].leaves.split2[0] += 1.0),
        ("split3", |pl| pl.players[0].leaves.split3[0] = f64::NAN),
        ("job slot", |pl| pl.players[0].leaves.job_slots[1][0] = pl.players[0].num_slots[1]),
        ("job slot mismatch", |pl| {
            let x = &mut pl.players[1].leaves;
            let s = x.job_slots[0][0];
            x.job_slots[0][0] = if s == 0 { 1 } else { 0 };
        }),
        ("job offsets", |pl| pl.players[0].leaves.job_off[1] = 0),
        ("job leaf duplicate", |pl| pl.players[0].leaves.job_leaves[1] = pl.players[0].leaves.job_leaves[0]),
        ("rounding flag", |pl| pl.players[0].leaves.fixed_scalar[0] ^= 1),
    ];
    for (what, mutate) in mutations {
        let mut plan = good.clone();
        mutate(&mut plan);
        assert!(plan.validate().is_err(), "{what} was accepted");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Random spots (either terminal model, river batching on or off): the
    /// plan validates and the models of the kernels' sweeps match the CPU.
    #[test]
    fn random_plans_sweeps_match_the_cpu((cfg, batch) in arb_multi_config(), seed in any::<u64>()) {
        let Ok(mut m) = MultiSolver::new(cfg) else { return Ok(()) };
        m.set_batch_river(batch);
        let plan = plan_for(&m);
        let mut rng = StdRng::seed_from_u64(seed);
        check_models("random spot", &m, &plan, &mut rng, 4);
    }
}
