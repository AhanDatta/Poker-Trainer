// Exact card removal at the fast precision: the sweep of exact.cuh with its
// sums merged and reordered where only their combination reaches a result
// (3-way `GpuPrecision::Fast` may sum in any order; validated within one
// f32 rounding of the CPU).
//
// `Pairs::pairs` only ever uses, per card a, `xqr[a] + xrq[a] - 2
// same_card[a]`, and `same - s`:
//   pairs(h) = tq' tr' + (same - s) + G[a] + G[b] - Y(a, b)
//              + (qa - qab)(ra - rab) + (qb - qab)(rb - rab) + qab rab
// with G = xqr + xrq - 2 same_card. So a set keeps three per-card arrays
// (each opponent's masses and G) instead of five, and one scalar for
// `same - s`. Ties with both opponents (the CPU's second set) are summed
// directly over the few hands of equal strength, and a showdown keeps one
// stash value per hand (`partial - lose * le`) instead of two: about a
// third of the exact path's per-lane memory traffic.
#pragma once

#include <solver/exact.cuh>

namespace solver::exact_fast {

using lanes::HandList;
using lanes::Lane;
using lanes::positive;

/// Per-card fields: each opponent's masses, then G.
namespace field {
[[nodiscard]] __device__ constexpr auto mass(u32 k, u32 c) noexcept -> u32 { return k * kCards + c; }
[[nodiscard]] __device__ constexpr auto g(u32 c) noexcept -> u32 { return 2u * kCards + c; }
}  // namespace field

inline constexpr u32 kStateFields = 3u * kCards;
static_assert(kStateFields == 156u, "the engine sizes the scratch with this (STATE_FIELDS)");

/// One set of a lane: every opponent entry below `end(k)` of each list.
template <Accumulator A>
class Set {
 public:
  __device__ Set(const Lane<A>& lane, const HandList& q, const HandList& r) noexcept : lane_{lane}, lists_{&q, &r} {
    for (u32 f = 0; f < kStateFields; ++f) {
      lane_.field(f) = A{};
    }
  }

  [[nodiscard]] __device__ auto end(u32 k) const noexcept -> u32 { return hi_[k]; }

  /// Opponent k's next entry enters (`Pairs::add_q` / `add_r`).
  __device__ void enter(u32 k) noexcept {
    const u32 o = 1u - k;
    const u32 pos = hi_[k];
    const auto [u, v] = lists_[k]->cards(pos);
    const A w = lane_.weight(*lists_[k], pos);
    const A mu = f(field::mass(o, u));
    const A mv = f(field::mass(o, v));
    // same += w x_o(u, v); s += w (m_o[u] + m_o[v]).
    const A sm = mul(w, x(o, u, v));
    base_ = sub(add(base_, sm), mul(w, add(mu, mv)));
    // G[v] += m_o[u] w - 2 sm; G[u] += m_o[v] w - 2 sm.
    const A two_sm = add(sm, sm);
    A& gv = f(field::g(v));
    gv = sub(add(gv, mul(mu, w)), two_sm);
    A& gu = f(field::g(u));
    gu = sub(add(gu, mul(mv, w)), two_sm);
    // m_k[u], m_k[v] grow: G[a] += w x_o(c, a) for a holding with c in o's set.
    const u32 held[2] = {u, v};
    for (const u32 c : held) {
      for (const u32 e : lists_[o]->holding(c)) {
        const auto [p, a] = lanes::unpack_holding(e);
        if (p >= hi_[o]) {
          break;
        }
        A& acc = f(field::g(a));
        acc = add(acc, mul(w, lane_.weight(*lists_[o], p)));
      }
    }
    t_[k] = add(t_[k], w);
    A& m_u = f(field::mass(k, u));
    m_u = add(m_u, w);
    A& m_v = f(field::mass(k, v));
    m_v = add(m_v, w);
    hi_[k] = pos + 1u;
  }

  /// Reach of the set's disjoint opponent pairs disjoint from `{a, b}`.
  [[nodiscard]] __device__ auto pairs(u32 a, u32 b) const noexcept -> A {
    A y{};
    const u32 cards[2] = {a, b};
    for (const u32 cc : cards) {
      const u32 other = cc == a ? b : a;
      for (const u32 e : lists_[0]->holding(cc)) {
        const auto [p, c] = lanes::unpack_holding(e);
        if (p >= hi_[0]) {
          break;
        }
        y = add(y, mul(lane_.weight(*lists_[0], p), x(1u, other, c)));
      }
    }
    const A qab = x(0u, a, b);
    const A rab = x(1u, a, b);
    const A qa = f(field::mass(0u, a));
    const A qb = f(field::mass(0u, b));
    const A ra = f(field::mass(1u, a));
    const A rb = f(field::mass(1u, b));
    const A tq = add(sub(t_[0], add(qa, qb)), qab);
    const A tr = add(sub(t_[1], add(ra, rb)), rab);
    A v = add(mul(tq, tr), base_);
    v = add(v, add(f(field::g(a)), f(field::g(b))));
    v = sub(v, y);
    v = add(v, mul(sub(qa, qab), sub(ra, rab)));
    v = add(v, mul(sub(qb, qab), sub(rb, rab)));
    v = add(v, mul(qab, rab));
    return positive(v);
  }

  /// `x_k(u, v)`: reach of opponent k's combo `{u, v}` if in the set.
  [[nodiscard]] __device__ auto x(u32 k, u32 u, u32 v) const noexcept -> A {
    const u32 pos = lists_[k]->position(u, v);
    return pos == kNone || pos >= hi_[k] ? A{} : lane_.weight(*lists_[k], pos);
  }

 private:
  [[nodiscard]] __device__ auto f(u32 x) const noexcept -> A& { return lane_.field(x); }

  const Lane<A>& lane_;
  const HandList* lists_[2];
  u32 hi_[2] = {0u, 0u};
  A t_[2] = {A{}, A{}};
  /// `same - s`.
  A base_{};
};

/// Reach of the pairs of tied opponent hands (q positions `[q0, q1)`, r
/// positions `[r0, r1)`) disjoint from each other and from `{a, b}`.
template <Accumulator A>
[[nodiscard]] __device__ auto tied_pairs(const Lane<A>& lane, const HandList& lq, const HandList& lr, u32 q0, u32 q1,
                                         u32 r0, u32 r1, CardPair h) noexcept -> A {
  const auto disjoint = [](CardPair x, CardPair y) {
    return x.first != y.first && x.first != y.second && x.second != y.first && x.second != y.second;
  };
  A sum{};
  for (u32 i = q0; i < q1; ++i) {
    const CardPair ci = lq.cards(i);
    if (!disjoint(ci, h)) {
      continue;
    }
    A inner{};
    for (u32 j = r0; j < r1; ++j) {
      const CardPair cj = lr.cards(j);
      if (disjoint(cj, h) && disjoint(cj, ci)) {
        inner = add(inner, lane.weight(lr, j));
      }
    }
    sum = add(sum, mul(lane.weight(lq, i), inner));
  }
  return sum;
}

/// One lane's exact showdown (fast): as `exact::showdown`, hands not on this
/// board left to the caller to zero.
template <Accumulator A>
__device__ void showdown(const Lane<A>& lane, const HandList& lp, const HandList& lq, const HandList& lr, bool q_in,
                         bool r_in, const exact::Payoffs<A>& pay, Span<float> out) noexcept {
  Set<A> lt{lane, lq, lr};
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
    const u32 q0 = lt.end(0u);
    const u32 r0 = lt.end(1u);
    u32 q1 = q0;
    u32 r1 = r0;
    if (q_in) {
      while (q1 < lq.len() && lq.strength(q1) == s) {
        ++q1;
      }
    }
    if (r_in) {
      while (r1 < lr.len() && lr.strength(r1) == s) {
        ++r1;
      }
    }
    while (lt.end(0u) < q1) {
      lt.enter(0u);
    }
    while (lt.end(1u) < r1) {
      lt.enter(1u);
    }
    const bool ties = q1 > q0 || r1 > r0;
    const bool tie_both = q1 > q0 && r1 > r0;
    for (u32 j = start; j < i; ++j) {
      const CardPair h = lp.cards(j);
      const A alone = lane.stashed(j);
      const A le = ties ? lt.pairs(h.first, h.second) : alone;
      const A tied = tie_both ? tied_pairs(lane, lq, lr, q0, q1, r0, r1, h) : A{};
      const A two = both ? sub(sub(le, alone), tied) : sub(le, alone);
      const A partial = add(add(mul(pay.win, alone), mul(pay.split2, two)), mul(pay.split3, tied));
      // v = partial + lose * (total - le): keep partial - lose * le.
      lane.stashed(j) = sub(partial, mul(pay.lose, le));
    }
  }
  while (lt.end(0u) < lq.len()) {
    lt.enter(0u);
  }
  while (lt.end(1u) < lr.len()) {
    lt.enter(1u);
  }
  for (u32 j = 0; j < n; ++j) {
    const auto [a, b] = lp.cards(j);
    out[lp.hand(j)] = to_f32(add(lane.stashed(j), mul(pay.lose, lt.pairs(a, b))));
  }
}

/// One lane's fixed-payoff job (fast): `fn(hand, pairs)` per traverser hand.
template <Accumulator A, class F>
__device__ void pair_masses(const Lane<A>& lane, const HandList& lp, const HandList& lq, const HandList& lr,
                            F&& fn) noexcept {
  Set<A> all{lane, lq, lr};
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

}  // namespace solver::exact_fast
