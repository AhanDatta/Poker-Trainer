// One three-player leaf per GPU lane: the hand lists the CPU's leaf
// evaluators sweep, and a lane's columns of the scratch arrays. Shared by
// both terminal models (independent.cuh, exact.cuh) and their kernels
// (multi_leaves.cu).
//
// The CPU evaluates a leaf with sweeps over the opponents' hands (in hand
// order for the compatible mass, in a board's strength order for
// showdowns), each traverser hand reading the running sums at its turn.
// The order of a sweep depends only on the board and the folded players,
// so the 32 lanes of a warp, each with its own leaf (its own reach and
// payoffs), run the same sweep in lockstep: no divergence, and each lane's
// arithmetic is the CPU's, operation for operation (`*_rn` with `double`),
// so the exact path is bit-identical to the CPU, rounding noise included.
// `DoubleFloat` gives the fast path.
//
// Per-lane state lives in global scratch, column `column` of a field-major
// array (field f of lane l at f * stride + l), so a warp's accesses to one
// field are one coalesced 256-byte transaction. A lane copies its reach
// into its column in the order of the lists it sweeps (`HandList::
// reach_row`), so a combo's reach is two loads away: its position, then
// the reach at that position. The host plan
// (src/gpu/multi/plan/leaves.rs) validates every list index; the Rust
// models in src/gpu/multi/plan/tests/leaves.rs mirror these sweeps.
#pragma once

#include <solver/core.cuh>
#include <solver/double_float.cuh>

namespace solver::lanes {

/// Leaves per warp (`LEAF_LANES` in the plan).
inline constexpr u32 kLanes = 32;

/// Threads per block of the leaf kernels (`BLOCK` in the engine).
inline constexpr u32 kBlock = 128;
static_assert(kBlock % kLanes == 0);

/// "Not in this list" in the 16-bit position arrays (`NO_POSITION`).
inline constexpr cuda::std::uint16_t kNoPosition = 0xFFFFu;

inline constexpr u32 kCombos = 1326;

/// Card offsets per list: one per card, plus the end.
inline constexpr u32 kCardOffsets = kCards + 1;

/// `holdem::combo::combo_index` of two distinct cards.
[[nodiscard]] __device__ inline auto combo_index(u32 a, u32 b) noexcept -> u32 {
  const u32 lo = a < b ? a : b;
  const u32 hi = a < b ? b : a;
  return hi * (hi - 1u) / 2u + lo;
}

/// The plan's `SweepLists`: every seat's hand list in every sweep order
/// (one per final board, strength-sorted; then hand order).
struct Lists {
  Span<const u32> off;
  Span<const u32> strength;
  Span<const u32> hand;
  Span<const u32> cards;
  Span<const u32> card_off;
  /// Per card, its entries by ascending position: `position | other << 16`.
  Span<const u32> holding;
  Span<const cuda::std::uint16_t> combo_pos;
};

/// An entry of `Lists::holding`: a position in the list and the other card
/// of the entry's combo.
struct Holding {
  u32 position;
  u32 other;
};

[[nodiscard]] __device__ inline auto unpack_holding(u32 e) noexcept -> Holding { return {e & 0xFFFFu, e >> 16}; }

/// One seat's list in one order, and (for an opponent's list) the first of
/// the lane's reach rows holding its entries' reach in list order.
class HandList {
 public:
  __device__ HandList(const Lists& lists, u32 order, u32 seat, u32 reach_row = 0u) noexcept
      : k_{order * 3u + seat},
        start_{lists.off[k_]},
        len_{lists.off[k_ + 1u] - start_},
        reach_row_{reach_row},
        lists_{lists} {}

  [[nodiscard]] __device__ auto reach_row() const noexcept -> u32 { return reach_row_; }
  [[nodiscard]] __device__ auto len() const noexcept -> u32 { return len_; }
  [[nodiscard]] __device__ auto strength(u32 pos) const noexcept -> u32 { return lists_.strength[start_ + pos]; }
  [[nodiscard]] __device__ auto hand(u32 pos) const noexcept -> u32 { return lists_.hand[start_ + pos]; }
  [[nodiscard]] __device__ auto cards(u32 pos) const noexcept -> CardPair {
    return unpack_cards(lists_.cards[start_ + pos]);
  }

  /// The entries holding card `c` (`unpack_holding`), by ascending position.
  [[nodiscard]] __device__ auto holding(u32 c) const noexcept -> Span<const u32> {
    const u32 b = lists_.card_off[k_ * kCardOffsets + c];
    const u32 e = lists_.card_off[k_ * kCardOffsets + c + 1u];
    return lists_.holding.subspan(b, e - b);
  }

  /// Position of combo `{u, v}`, or `kNone` (also for `u == v`).
  [[nodiscard]] __device__ auto position(u32 u, u32 v) const noexcept -> u32 {
    if (u == v) {
      return kNone;
    }
    const auto pos = lists_.combo_pos[k_ * kCombos + combo_index(u, v)];
    return pos == kNoPosition ? kNone : static_cast<u32>(pos);
  }

 private:
  u32 k_;
  u32 start_;
  u32 len_;
  u32 reach_row_;
  const Lists& lists_;
};

/// A lane's columns of the scratch arrays: the model's state fields, the
/// opponents' reach in the order of the lists it sweeps, and per-hand stash
/// rows.
template <Accumulator A>
struct Lane {
  Span<A> state;
  Span<float> reach;
  Span<A> stash;
  u32 stride;
  u32 column;

  [[nodiscard]] __device__ auto field(u32 f) const noexcept -> A& { return state[f * stride + column]; }
  [[nodiscard]] __device__ auto reach_at(u32 row) const noexcept -> float& { return reach[row * stride + column]; }
  /// Reach of the entry at `pos` of an opponent's list.
  [[nodiscard]] __device__ auto weight(const HandList& list, u32 pos) const noexcept -> A {
    return widen<A>(reach_at(list.reach_row() + pos));
  }
  /// Copies an opponent's reach `src` (by hand) into `list`'s rows.
  __device__ void load(const HandList& list, Span<const float> src) const noexcept {
    for (u32 pos = 0; pos < list.len(); ++pos) {
      reach_at(list.reach_row() + pos) = src[list.hand(pos)];
    }
  }
  [[nodiscard]] __device__ auto stashed(u32 r) const noexcept -> A& { return stash[r * stride + column]; }
};

/// `max(x, 0)` (`f64::max`; no -0.0 or NaN reaches it).
[[nodiscard]] __device__ inline auto positive(double x) noexcept -> double { return x > 0.0 ? x : 0.0; }
[[nodiscard]] __device__ inline auto positive(DoubleFloat x) noexcept -> DoubleFloat {
  return x.hi > 0.0f ? x : DoubleFloat{};
}

}  // namespace solver::lanes
