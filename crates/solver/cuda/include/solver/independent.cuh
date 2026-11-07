// Three-player leaves with independent opponents, one leaf per lane
// (lanes.cuh): the CPU's `multi::cfr::pay_all` and `independent_showdown`
// with `exact_removal` off.
//
// For each traverser hand h and opponent o the CPU needs
// * the compatible mass `c_o(h) = max(total - minus[c1] - minus[c2] +
//   same, 0)` (`Hands::compat`: a hand-order sweep of o's hands), and
// * at showdowns, the beaten mass `b_o(h) = max(weaker + equal / 2, 0)`
//   (`Table::beat`: a strength-order sweep of o's hands on the board; `c_o`
//   when o folded, 0 for traverser hands on the board's cards),
// and combines them as `pay * c_q * c_r` (fixed payoffs) or `win * p +
// lose * (c_q * c_r - p)` with `p = b_q * b_r` (showdowns), in f64,
// rounded to f32 at the end. Each sum here runs in the CPU's order.
#pragma once

#include <solver/lanes.cuh>

namespace solver::independent {

using lanes::HandList;
using lanes::Lane;
using lanes::positive;

/// Per-lane state: each opponent's hand-order total and per-card masses,
/// then the running sums of a strength sweep (one opponent at a time).
namespace field {
[[nodiscard]] __device__ constexpr auto total(u32 k) noexcept -> u32 { return k * (kCards + 1u); }
[[nodiscard]] __device__ constexpr auto minus(u32 k, u32 c) noexcept -> u32 { return k * (kCards + 1u) + 1u + c; }
inline constexpr u32 kWeaker = 2u * (kCards + 1u);
[[nodiscard]] __device__ constexpr auto weaker_minus(u32 c) noexcept -> u32 { return kWeaker + 1u + c; }
}  // namespace field

inline constexpr u32 kStateFields = field::kWeaker + 1u + kCards;
static_assert(kStateFields == 159u, "the engine sizes the scratch with this (STATE_FIELDS)");

/// `Hands::compat`'s sums for both opponents: every hand in hand order
/// (`by_hand[k]`, opponent k's hand-order list). The CPU skips hands with
/// zero reach; adding +0.0 changes nothing.
template <Accumulator A>
__device__ void compat_sums(const Lane<A>& lane, const HandList (&by_hand)[2]) noexcept {
  for (u32 k = 0; k < 2; ++k) {
    A& total = lane.field(field::total(k));
    total = A{};
    for (u32 c = 0; c < kCards; ++c) {
      lane.field(field::minus(k, c)) = A{};
    }
    for (u32 pos = 0; pos < by_hand[k].len(); ++pos) {
      const A w = lane.weight(by_hand[k], pos);
      const auto [a, b] = by_hand[k].cards(pos);
      total = add(total, w);
      A& ma = lane.field(field::minus(k, a));
      ma = add(ma, w);
      A& mb = lane.field(field::minus(k, b));
      mb = add(mb, w);
    }
  }
}

/// Reach of opponent k's combo `{a, b}` (`Hands::same`), 0 if it has none.
template <Accumulator A>
[[nodiscard]] __device__ auto same_reach(const Lane<A>& lane, const HandList& by_hand, CardPair h) noexcept -> A {
  const u32 pos = by_hand.position(h.first, h.second);
  return pos == kNone ? A{} : lane.weight(by_hand, pos);
}

/// `Hands::compat` of opponent k for traverser hand `h` (after `compat_sums`).
template <Accumulator A>
[[nodiscard]] __device__ auto compat(const Lane<A>& lane, const HandList& by_hand, u32 k, CardPair h) noexcept -> A {
  const A t = lane.field(field::total(k));
  const A m1 = lane.field(field::minus(k, h.first));
  const A m2 = lane.field(field::minus(k, h.second));
  return positive(add(sub(sub(t, m1), m2), same_reach(lane, by_hand, h)));
}

/// `Table::beat` against opponent k: writes `b_k` of the hands on the
/// traverser's board list `lp` to stash rows `base + hand` (the caller has
/// zeroed the others).
template <Accumulator A>
__device__ void beat(const Lane<A>& lane, const HandList& lp, const HandList& lo, const HandList& by_hand,
                     u32 base) noexcept {
  A& sum = lane.field(field::kWeaker);
  sum = A{};
  for (u32 c = 0; c < kCards; ++c) {
    lane.field(field::weaker_minus(c)) = A{};
  }
  const A half = narrow_from<A>(0.5);
  u32 j = 0;
  u32 i = 0;
  while (i < lp.len()) {
    const u32 s = lp.strength(i);
    while (j < lo.len() && lo.strength(j) < s) {
      const A w = lane.weight(lo, j);
      const auto [a, b] = lo.cards(j);
      sum = add(sum, w);
      A& ma = lane.field(field::weaker_minus(a));
      ma = add(ma, w);
      A& mb = lane.field(field::weaker_minus(b));
      mb = add(mb, w);
      ++j;
    }
    // Opponent hands of equal strength: `eq`, and per traverser hand the
    // part holding each of its cards (`eq_minus`), in list order.
    u32 end = j;
    while (end < lo.len() && lo.strength(end) == s) {
      ++end;
    }
    A eq{};
    for (u32 e = j; e < end; ++e) {
      eq = add(eq, lane.weight(lo, e));
    }
    while (i < lp.len() && lp.strength(i) == s) {
      const CardPair h = lp.cards(i);
      A eq_a{};
      A eq_b{};
      for (u32 e = j; e < end; ++e) {
        const auto [x, y] = lo.cards(e);
        const A w = lane.weight(lo, e);
        if (x == h.first || y == h.first) {
          eq_a = add(eq_a, w);
        }
        if (x == h.second || y == h.second) {
          eq_b = add(eq_b, w);
        }
      }
      const A weaker = sub(sub(sum, lane.field(field::weaker_minus(h.first))), lane.field(field::weaker_minus(h.second)));
      const A equal = add(sub(sub(eq, eq_a), eq_b), same_reach(lane, by_hand, h));
      lane.stashed(base + lp.hand(i)) = positive(add(weaker, mul(half, equal)));
      ++i;
    }
  }
}

/// One lane's independent showdown: the traverser's values for all of its
/// `hands` (hand order) into `out`. `lp`, `lo[k]`: the board's lists of
/// the traverser and its opponents; `by_hand`: the opponents' hand-order
/// lists; `in[k]`: whether opponent k is still in the hand.
template <Accumulator A>
__device__ void showdown(const Lane<A>& lane, const HandList& lp, const HandList (&lo)[2],
                         const HandList& my_hands, const HandList (&by_hand)[2], const bool (&in)[2], A win, A lose,
                         Span<float> out) noexcept {
  compat_sums(lane, by_hand);
  const u32 n = my_hands.len();
  // Stash rows k * n + h: b_k(h).
  for (u32 k = 0; k < 2; ++k) {
    if (in[k]) {
      for (u32 h = 0; h < n; ++h) {
        lane.stashed(k * n + h) = A{};
      }
      beat(lane, lp, lo[k], by_hand[k], k * n);
    } else {
      // A folded opponent's hand beats nothing: b = c.
      for (u32 h = 0; h < n; ++h) {
        lane.stashed(k * n + h) = compat(lane, by_hand[k], k, my_hands.cards(h));
      }
    }
  }
  for (u32 h = 0; h < n; ++h) {
    const CardPair cards = my_hands.cards(h);
    const A total = mul(compat(lane, by_hand[0], 0u, cards), compat(lane, by_hand[1], 1u, cards));
    const A pwin = mul(lane.stashed(h), lane.stashed(n + h));
    out[my_hands.hand(h)] = to_f32(add(mul(win, pwin), mul(lose, sub(total, pwin))));
  }
}

/// One lane's fixed-payoff job (`pay_all`): `fn(hand, c_q, c_r)` for each
/// traverser hand in hand order.
template <Accumulator A, class F>
__device__ void compat_masses(const Lane<A>& lane, const HandList& my_hands, const HandList (&by_hand)[2],
                              F&& fn) noexcept {
  compat_sums(lane, by_hand);
  for (u32 h = 0; h < my_hands.len(); ++h) {
    const CardPair cards = my_hands.cards(h);
    fn(my_hands.hand(h), compat(lane, by_hand[0], 0u, cards), compat(lane, by_hand[1], 1u, cards));
  }
}

}  // namespace solver::independent
