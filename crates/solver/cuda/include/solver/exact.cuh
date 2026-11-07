// Exact card removal between the two opponents of a three-player leaf
// (`multi::exact`), one leaf per lane (lanes.cuh).
//
// For a traverser hand h = {a, b} a leaf needs the reach of opponent hand
// pairs (i, j) with h, i and j pairwise disjoint. The CPU gets it by
// inclusion-exclusion from per-set statistics that it updates as hands
// enter two sets, one per opponent (`Pairs`):
//   pairs(h) = tQ'·tR' − Σ_c mQ'[c]·mR'[c] + Σ_k xQ'(k)·xR'(k)
// (t a set's total reach, m[c] that of its hands holding card c, x(k) that
// of combo k, ' restricting to hands disjoint from h). Hands enter in list
// order (a board's strength order at a showdown, hand order for a fixed
// payoff), so "is this combo in the set" is a position test: each set holds
// a contiguous range [lo, hi) of each opponent list's positions.
#pragma once

#include <solver/lanes.cuh>

namespace solver::exact {

using lanes::HandList;
using lanes::Lane;
using lanes::positive;

// ---- Per-lane state layout -------------------------------------------------

/// Per-card fields of one set (the CPU's `Pairs`; its scalars live in
/// registers, `Set`): per opponent k (0 = q, 1 = r) the per-card masses
/// `m[c]`; the cross sums `cross[0][a] = Σ_c mQ[c] xR(c, a)` (`xqr`) and
/// `cross[1][a] = Σ_c mR[c] xQ(c, a)` (`xrq`); `same_card[a]`, the part of
/// `same` over combos holding a.
namespace field {
[[nodiscard]] __device__ constexpr auto mass(u32 k, u32 c) noexcept -> u32 { return k * kCards + c; }
[[nodiscard]] __device__ constexpr auto cross(u32 k, u32 c) noexcept -> u32 { return (2u + k) * kCards + c; }
[[nodiscard]] __device__ constexpr auto same_card(u32 c) noexcept -> u32 { return 4u * kCards + c; }
inline constexpr u32 kSet = 5u * kCards;
}  // namespace field

/// Two sets per lane: hands below the current strength (`lt`, also the
/// whole sweep of a fixed payoff) and hands tied at it (`eq`).
inline constexpr u32 kStateFields = 2u * field::kSet;

static_assert(field::kSet == 260u, "the engine sizes the scratch with this (STATE_FIELDS)");

/// One set of a lane (the CPU's `Pairs`): which positions of each
/// opponent's list it holds, and its statistics in the lane's state.
template <Accumulator A>
class Set {
 public:
  __device__ Set(const Lane<A>& lane, u32 base, const HandList& q, const HandList& r) noexcept
      : lane_{lane}, base_{base}, lists_{&q, &r} {}

  /// Empty, positioned to take each opponent's entries from `lo_q`, `lo_r`.
  __device__ void clear_at(u32 lo_q, u32 lo_r) noexcept {
    for (u32 f = 0; f < field::kSet; ++f) {
      lane_.field(base_ + f) = A{};
    }
    t_[0] = t_[1] = s_ = same_ = A{};
    lo_[0] = hi_[0] = lo_q;
    lo_[1] = hi_[1] = lo_r;
  }

  [[nodiscard]] __device__ auto end(u32 k) const noexcept -> u32 { return hi_[k]; }

  /// `Pairs::add_q` (k = 0) / `add_r` (k = 1): opponent k's next entry
  /// (position `end(k)`) enters.
  __device__ void enter(u32 k) noexcept {
    const u32 o = 1u - k;
    const u32 pos = hi_[k];
    const auto [u, v] = lists_[k]->cards(pos);
    const A w = lane_.weight(*lists_[k], pos);
    const A xo = x(o, u, v);
    const A mu = f(field::mass(o, u));
    const A mv = f(field::mass(o, v));
    s_ = add(s_, mul(w, add(mu, mv)));
    A& cross_v = f(field::cross(o, v));
    cross_v = add(cross_v, mul(mu, w));
    A& cross_u = f(field::cross(o, u));
    cross_u = add(cross_u, mul(mv, w));
    const A sm = mul(w, xo);
    same_ = add(same_, sm);
    A& same_u = f(field::same_card(u));
    same_u = add(same_u, sm);
    A& same_v = f(field::same_card(v));
    same_v = add(same_v, sm);
    // m_k[u] and m_k[v] grow by w: Σ_c m_k[c] x_o(c, a) grows by w x_o(u, a)
    // and w x_o(v, a), for each a holding with u (then v) in o's set.
    const u32 held[2] = {u, v};
    for (const u32 c : held) {
      for_each_holding(o, c, [&](u32 a, A xa) {
        A& acc = f(field::cross(k, a));
        acc = add(acc, mul(w, xa));
      });
    }
    t_[k] = add(t_[k], w);
    A& m_u = f(field::mass(k, u));
    m_u = add(m_u, w);
    A& m_v = f(field::mass(k, v));
    m_v = add(m_v, w);
    hi_[k] = pos + 1u;
  }

  /// `Pairs::pairs`: reach of the set's disjoint opponent pairs that are
  /// also disjoint from `{a, b}`.
  [[nodiscard]] __device__ auto pairs(u32 a, u32 b) const noexcept -> A {
    // Y(a, b) + Y(b, a), Y(a, b) = Σ_c xQ(c, a) xR(c, b).
    A y{};
    for_each_holding(0u, a, [&](u32 c, A xc) { y = add(y, mul(xc, x(1u, b, c))); });
    for_each_holding(0u, b, [&](u32 c, A xc) { y = add(y, mul(xc, x(1u, a, c))); });
    const A qab = x(0u, a, b);
    const A rab = x(1u, a, b);
    const A qa = f(field::mass(0u, a));
    const A qb = f(field::mass(0u, b));
    const A ra = f(field::mass(1u, a));
    const A rb = f(field::mass(1u, b));
    const A tq = add(sub(sub(t_[0], qa), qb), qab);
    const A tr = add(sub(sub(t_[1], ra), rb), rab);
    // Σ_c (mQ[c] − xQ(c,a) − xQ(c,b))·(mR[c] − xR(c,a) − xR(c,b)) without
    // cards a and b, in the CPU's order.
    A shared = s_;
    shared = sub(shared, f(field::cross(0u, a)));
    shared = sub(shared, f(field::cross(0u, b)));
    shared = sub(shared, f(field::cross(1u, a)));
    shared = sub(shared, f(field::cross(1u, b)));
    shared = add(shared, f(field::same_card(a)));
    shared = add(shared, f(field::same_card(b)));
    shared = add(shared, y);
    shared = sub(shared, mul(sub(qa, qab), sub(ra, rab)));
    shared = sub(shared, mul(sub(qb, qab), sub(rb, rab)));
    const A same = add(sub(sub(same_, f(field::same_card(a))), f(field::same_card(b))), mul(qab, rab));
    return positive(add(sub(mul(tq, tr), shared), same));
  }

 private:
  [[nodiscard]] __device__ auto f(u32 x) const noexcept -> A& { return lane_.field(base_ + x); }

  /// `x_k(u, v)`: reach of opponent k's combo `{u, v}` if in the set, else 0.
  [[nodiscard]] __device__ auto x(u32 k, u32 u, u32 v) const noexcept -> A {
    const u32 pos = lists_[k]->position(u, v);
    if (pos == kNone || pos < lo_[k] || pos >= hi_[k]) {
      return A{};
    }
    return lane_.weight(*lists_[k], pos);
  }

  /// `by_card`: `fn(other card, reach)` for opponent k's entries in the set
  /// holding card `c`, in the order they entered.
  template <class F>
  __device__ void for_each_holding(u32 k, u32 c, F&& fn) const noexcept {
    for (const u32 e : lists_[k]->holding(c)) {
      const auto [pos, other] = lanes::unpack_holding(e);
      if (pos >= hi_[k]) {
        break;
      }
      if (pos < lo_[k]) {
        continue;
      }
      fn(other, lane_.weight(*lists_[k], pos));
    }
  }

  const Lane<A>& lane_;
  u32 base_;
  const HandList* lists_[2];
  u32 lo_[2] = {0u, 0u};
  u32 hi_[2] = {0u, 0u};
  /// Per opponent the set's total reach; `s = Σ_c mQ[c] mR[c]`; `same =
  /// Σ_k xQ(k) xR(k)`.
  A t_[2] = {A{}, A{}};
  A s_{};
  A same_{};
};

// ---- Sweeps ----------------------------------------------------------------

/// A showdown's payoffs for the traverser (`exact::showdown_lanes`): `win`
/// alone with the best hand, `split2`/`split3` tied with one or both
/// opponents, `lose` otherwise.
template <Accumulator A>
struct Payoffs {
  A win;
  A lose;
  A split2;
  A split3;
};

/// One lane's exact showdown (`exact::showdown_lanes`, one lane of it):
/// writes the traverser's values for the hands of its list `lp` into `out`
/// (hands not on this board are left to the caller to zero). `q_in`/`r_in`:
/// whether each opponent is still in the hand (a folded opponent's cards
/// only block, so all of its hands enter first).
template <Accumulator A>
__device__ void showdown(const Lane<A>& lane, const HandList& lp, const HandList& lq, const HandList& lr,
                         bool q_in, bool r_in, const Payoffs<A>& pay, Span<float> out) noexcept {
  Set<A> lt{lane, 0u, lq, lr};
  Set<A> eq{lane, field::kSet, lq, lr};
  lt.clear_at(0u, 0u);
  if (!q_in) {
    while (lt.end(0u) < lq.len()) {
      lt.enter(0u);
    }
  }
  if (!r_in) {
    while (lt.end(1u) < lr.len()) {
      lt.enter(1u);
    }
  }
  const u32 n = lp.len();
  const bool both = q_in && r_in;
  // Stash rows: [0, n) the pairs where the traverser alone is best, then
  // the partial value; [n, 2n) the pairs where it is best with others.
  u32 i = 0;
  while (i < n) {
    const u32 s = lp.strength(i);
    if (q_in) {
      while (lt.end(0u) < lq.len() && lq.strength(lt.end(0u)) < s) {
        lt.enter(0u);
      }
    }
    if (r_in) {
      while (lt.end(1u) < lr.len() && lr.strength(lt.end(1u)) < s) {
        lt.enter(1u);
      }
    }
    const u32 start = i;
    while (i < n && lp.strength(i) == s) {
      const auto [a, b] = lp.cards(i);
      lane.stashed(i) = lt.pairs(a, b);
      ++i;
    }
    // Opponent hands of equal strength.
    u32 eq_q = 0;
    u32 eq_r = 0;
    if (q_in) {
      while (lt.end(0u) + eq_q < lq.len() && lq.strength(lt.end(0u) + eq_q) == s) {
        ++eq_q;
      }
    }
    if (r_in) {
      while (lt.end(1u) + eq_r < lr.len() && lr.strength(lt.end(1u) + eq_r) == s) {
        ++eq_r;
      }
    }
    const bool ties = eq_q + eq_r > 0u;
    const bool tie_both = eq_q > 0u && eq_r > 0u;
    if (tie_both) {
      eq.clear_at(lt.end(0u), lt.end(1u));
    }
    for (u32 e = 0; e < eq_q; ++e) {
      lt.enter(0u);
    }
    for (u32 e = 0; e < eq_r; ++e) {
      lt.enter(1u);
    }
    if (tie_both) {
      for (u32 e = 0; e < eq_q; ++e) {
        eq.enter(0u);
      }
      for (u32 e = 0; e < eq_r; ++e) {
        eq.enter(1u);
      }
    }
    for (u32 j = start; j < i; ++j) {
      const auto [a, b] = lp.cards(j);
      const A alone = lane.stashed(j);
      const A le = ties ? lt.pairs(a, b) : alone;
      const A tied = tie_both ? eq.pairs(a, b) : A{};
      // `(two, three)`: tied with one opponent, with both (only when both
      // are in; with one folded a tie is a two-way split).
      const A two = both ? sub(sub(le, alone), tied) : sub(le, alone);
      const A three = both ? tied : A{};
      lane.stashed(j) = add(add(mul(pay.win, alone), mul(pay.split2, two)), mul(pay.split3, three));
      lane.stashed(n + j) = le;
    }
  }
  if (q_in) {
    while (lt.end(0u) < lq.len()) {
      lt.enter(0u);
    }
  }
  if (r_in) {
    while (lt.end(1u) < lr.len()) {
      lt.enter(1u);
    }
  }
  for (u32 j = 0; j < n; ++j) {
    const auto [a, b] = lp.cards(j);
    const A total = lt.pairs(a, b);
    out[lp.hand(j)] = to_f32(add(lane.stashed(j), mul(pay.lose, sub(total, lane.stashed(n + j)))));
  }
}

/// One lane's fixed-payoff job (`exact::fold_values`): every hand of both
/// opponents enters in hand order (q's first), then `fn(hand, pairs)` for
/// each traverser hand in hand order.
template <Accumulator A, class F>
__device__ void pair_masses(const Lane<A>& lane, const HandList& lp, const HandList& lq, const HandList& lr,
                            F&& fn) noexcept {
  Set<A> all{lane, 0u, lq, lr};
  all.clear_at(0u, 0u);
  while (all.end(0u) < lq.len()) {
    all.enter(0u);
  }
  while (all.end(1u) < lr.len()) {
    all.enter(1u);
  }
  for (u32 j = 0; j < lp.len(); ++j) {
    const auto [a, b] = lp.cards(j);
    fn(lp.hand(j), all.pairs(a, b));
  }
}

}  // namespace solver::exact
