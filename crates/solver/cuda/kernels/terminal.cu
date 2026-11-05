// Terminal values: folds and showdowns, one thread block per terminal.
//
// CPU twins: `eval::fold_batch` / `eval::eval_fold` and
// `eval::showdown_batch` / `eval::eval_showdown` (+ `add_ties`). The CPU
// batches the terminals of river subtrees and evaluates the others (folds
// before the river, showdowns right after an all-in runout) one at a time;
// the two round differently, and the plan's `scalar` flag says which one a
// terminal follows. Card removal is inclusion-exclusion:
// the opponent reach compatible with hand {c1, c2} is
// `total - minus[c1] - minus[c2] + reach(identical combo)`, where
// `minus[c]` is the reach of opponent hands holding card c.
//
// Two precisions, chosen per launch (`exact`):
// * exact: sums in f64 like the CPU's, final combinations in the CPU's
//   operation order without fused multiply-adds. Bit-identical to the CPU
//   in every test so far.
// * fast: the same code with double-single accumulators (double_float.cuh,
//   about 44-48 bits on the f32 units). The GPU runs f64 at 1/64 of its f32
//   rate and the f64 pipe was the bottleneck of these kernels; results
//   agree with the exact path except for rare one-rounding-step differences.
// Either way, sums and prefix sums run sequentially per thread over
// contiguous chunks (or a quarter of one card's list), with only the chunk
// totals combined by a block-wide scan or reduction (or warp shuffles).
//
// Showdowns replace the CPU's sequential strength sweep by prefix sums over
// the board's strength-sorted opponent hands, plus one prefix sum per card
// over the opponent hands holding it. Where each of the traverser's hands
// sits in those lists depends only on the board, so the host precomputes it
// (src/gpu/plan.rs, `Tables`). A fold that shares its opponent reach with
// the showdown (both children of one traverser node) is read off the same
// sums, as the CPU's `showdown_batch` does.

#include <solver/core.cuh>
#include <solver/double_float.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/utility>

/// Dynamic shared memory of the terminal kernels (layouts at each kernel).
extern __shared__ alignas(16) unsigned char terminal_shared[];

namespace solver {
namespace {

/// Threads per block (a compile-time constant: loops stride by it, which
/// measured a few percent faster than striding by `blockDim.x`). 512-thread
/// showdown blocks, to fill the warp slots that the shared-memory limit of 3
/// blocks per SM leaves empty, measured 10% slower; reading the opponent
/// weights from the (cached) reach row instead of shared memory, for 4
/// blocks per SM, 3% slower.
inline constexpr u32 kThreads = 256;

template <class T>
using Span = cuda::std::span<T>;

/// Writes 0 for every hand (a dead terminal is worth nothing).
__device__ void zero_row(Span<float> out) noexcept {
  for (u32 h = threadIdx.x; h < out.size(); h += kThreads) {
    out[h] = 0.0f;
  }
}

/// Reach of the opponent's copy of hand `h`'s combo, or 0.
template <Accumulator A>
[[nodiscard]] __device__ auto same_reach(Span<const u32> same, Span<const float> reach, u32 h) noexcept -> A {
  const u32 j = same[h];
  return widen<A>(j == kNone ? 0.0f : reach[j]);
}

/// This thread's contiguous chunk of `[0, len)` when the block splits it
/// into `kThreads` nearly equal chunks.
[[nodiscard]] __device__ auto my_chunk(u32 len) noexcept -> cuda::std::pair<u32, u32> {
  const u32 per = (len + kThreads - 1) / kThreads;
  const u32 begin = cuda::std::min(threadIdx.x * per, len);
  return {begin, cuda::std::min(begin + per, len)};
}

/// Threads sharing one card's list in `card_prefix_sums` (a group of
/// consecutive lanes of one warp).
inline constexpr u32 kThreadsPerCard = 4;
static_assert(kCards * kThreadsPerCard <= kThreads, "every card needs its thread group");
static_assert(32 % kThreadsPerCard == 0, "a card's thread group must not straddle warps");

/// `v` of lane `src` of this thread's card group (every lane of the warp
/// must call it).
template <Accumulator A>
[[nodiscard]] __device__ auto group_lane(A v, u32 src) noexcept -> A {
  constexpr unsigned kAll = 0xFFFF'FFFFu;
  if constexpr (cuda::std::same_as<A, double>) {
    return __shfl_sync(kAll, v, static_cast<int>(src), kThreadsPerCard);
  } else {
    return A{__shfl_sync(kAll, v.hi, static_cast<int>(src), kThreadsPerCard),
             __shfl_sync(kAll, v.lo, static_cast<int>(src), kThreadsPerCard)};
  }
}

/// Prefix sums of the weights of each card's opponent hands: for card `c`,
/// whose entries are `card_pos[card_off[c] .. card_off[c + 1]]` (positions
/// into `weights`), `out[card_off[c] + c + e]` is the sum of its first `e`
/// weights, `e = 0 ..= len`. Every thread of the block must call it.
///
/// `kThreadsPerCard` threads share a card: each sums its quarter of the list
/// into local prefixes, the quarter totals are combined across the group in
/// lane order, and each quarter's prefixes are then offset by the totals
/// before it (independent adds). The dependent chain is about a quarter of
/// one sequential sweep, which left most of the block waiting at the barrier.
template <Accumulator A>
__device__ void card_prefix_sums(Span<const u32> card_off,
                                 Span<const u32> card_pos,
                                 Span<const float> weights,
                                 Span<A> out) noexcept {
  const u32 c = threadIdx.x / kThreadsPerCard;
  const u32 part = threadIdx.x % kThreadsPerCard;
  const bool has_card = c < kCards;
  const u32 first_entry = has_card ? card_off[c] : 0u;
  const u32 len = has_card ? card_off[c + 1] - first_entry : 0u;
  const u32 lo = len * part / kThreadsPerCard;
  const u32 hi = len * (part + 1) / kThreadsPerCard;
  const auto segment = has_card ? out.subspan(first_entry + c, len + 1) : out.first(0);

  // Local prefixes of this quarter (offset below).
  A local{};
#pragma unroll 4
  for (u32 e = lo; e < hi; ++e) {
    segment[e] = local;
    local = add(local, weights[card_pos[first_entry + e]]);
  }
  // The totals of the quarters before this one, summed in lane order.
  A carry{};
  for (u32 q = 0; q + 1 < kThreadsPerCard; ++q) {
    const A total = group_lane(local, q);
    if (q < part) {
      carry = add(carry, total);
    }
  }
  if (part > 0) {
    for (u32 e = lo; e < hi; ++e) {
      segment[e] = add(carry, segment[e]);
    }
  }
  if (has_card && part + 1 == kThreadsPerCard) {
    segment[len] = add(carry, local);
  }
}

/// Compatible opponent reach of a hand for a fold: `eval_fold` computes
/// `total - (m1 + m2) + same`, the batched evaluators `total - m1 - m2 + same`.
template <Accumulator A>
[[nodiscard]] __device__ auto fold_compatible(bool one_at_a_time, A total, A m1, A m2, A same) noexcept -> A {
  const A blocked = one_at_a_time ? sub(total, add(m1, m2)) : sub(sub(total, m1), m2);
  return add(blocked, same);
}

/// The views a fold launch works on.
struct FoldArgs {
  Span<const u32> list;
  Span<const double> payoff;
  Span<const u32> slots;
  Span<const u32> scalar;
  Span<const u32> alive;
  Span<const float> reach;
  Span<float> values;
  Span<const u32> rows;
  Span<const u32> my_cards;
  Span<const u32> same;
  Span<const u32> card_off;
  Span<const u32> card_hands;
  u32 hands;
  u32 opp_hands;
};

/// Fold terminals with accumulator `A`. Dynamic shared memory: the
/// opponent's reach (`opp_hands` floats).
template <Accumulator A>
__device__ void fold_values_impl(const FoldArgs& args) {
  using BlockReduce = cub::BlockReduce<A, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ A minus_storage[kCards];
  __shared__ A total;
  const auto minus = cuda::std::span<A, kCards>{minus_storage};
  const auto weights = Span<float>{reinterpret_cast<float*>(terminal_shared), args.opp_hands};
  const u32 hands = args.hands;
  const u32 opp_hands = args.opp_hands;

  for (u32 k = blockIdx.x; k < args.list.size(); k += gridDim.x) {
    const u32 node = args.list[k];
    const u32 slot = args.slots[node];
    const auto out = row(args.values, args.rows[node], hands);
    if (args.alive[slot] == 0u) {
      zero_row(out);
      continue;  // uniform: every thread of the block skips together
    }
    const auto reach = row(args.reach, slot, opp_hands);

    // Copy the reach to shared memory; the total from per-thread chunks.
    for (u32 j = threadIdx.x; j < opp_hands; j += kThreads) {
      weights[j] = reach[j];
    }
    __syncthreads();
    const auto [begin, end] = my_chunk(opp_hands);
    A partial{};
    for (u32 j = begin; j < end; ++j) {
      partial = add(partial, weights[j]);
    }
    const A block_total = BlockReduce{reduce_storage}.Reduce(partial, Plus{});
    if (threadIdx.x == 0) {
      total = block_total;
    }
    // minus[c]: one thread per card, summing that card's list sequentially.
    for (u32 c = threadIdx.x; c < kCards; c += kThreads) {
      A m{};
      const u32 last = args.card_off[c + 1];
#pragma unroll 8
      for (u32 e = args.card_off[c]; e < last; ++e) {
        m = add(m, weights[args.card_hands[e]]);
      }
      minus[c] = m;
    }
    __syncthreads();

    const A pay = narrow_from<A>(args.payoff[k]);
    const bool one_at_a_time = args.scalar[node] != 0u;
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      const auto cards = unpack_cards(args.my_cards[h]);
      const A compatible = fold_compatible(one_at_a_time, total, minus[cards.first], minus[cards.second],
                                           same_reach<A>(args.same, reach, h));
      out[h] = to_f32(mul(pay, compatible));
    }
    __syncthreads();  // shared memory is reused by the next terminal
  }
}

/// The views a showdown launch works on.
struct ShowdownArgs {
  Span<const u32> list;
  Span<const u32> table_of;
  Span<const double> win;
  Span<const double> lose;
  Span<const double> tie;
  Span<const u32> paired_fold;
  Span<const double> paired_payoff;
  Span<const u32> slots;
  Span<const u32> scalar;
  Span<const u32> alive;
  Span<const float> reach;
  Span<float> values;
  Span<const u32> rows;
  Span<const u32> my_cards;
  Span<const u32> same;
  Span<const u32> opp_off;
  Span<const u32> opp_len;
  Span<const u32> opp_order;
  Span<const u32> card_base;
  Span<const u32> card_off;
  Span<const u32> card_pos;
  Span<const u32> ranges;
  Span<const u32> ranks;
  u32 hands;
  u32 opp_hands;
  u32 max_opp;
};

/// Showdown terminals with accumulator `A`. Dynamic shared memory, sized by
/// the host for `max_opp` opponent hands (8 bytes per accumulator for both
/// precisions): prefix[max_opp + 1] and card_prefix[2 * max_opp + 52], then
/// the gathered weights[max_opp] (f32).
template <Accumulator A>
__device__ void showdown_values_impl(const ShowdownArgs& args) {
  using BlockScan = cub::BlockScan<A, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const u32 max_opp = args.max_opp;
  A* const accumulators = reinterpret_cast<A*>(terminal_shared);
  const auto prefix_all = Span<A>{accumulators, max_opp + 1};
  const auto card_prefix_all = Span<A>{accumulators + max_opp + 1, 2 * max_opp + kCards};
  const auto weights_all = Span<float>{reinterpret_cast<float*>(accumulators + 3 * max_opp + 1 + kCards), max_opp};
  const u32 hands = args.hands;

  for (u32 k = blockIdx.x; k < args.list.size(); k += gridDim.x) {
    const u32 node = args.list[k];
    const u32 slot = args.slots[node];
    const u32 fold_node = args.paired_fold[k];
    const auto out = row(args.values, args.rows[node], hands);
    if (args.alive[slot] == 0u) {
      zero_row(out);
      if (fold_node != kNone) {
        zero_row(row(args.values, args.rows[fold_node], hands));
      }
      continue;  // uniform: every thread of the block skips together
    }
    const auto reach = row(args.reach, slot, args.opp_hands);
    const u32 table = args.table_of[k];
    const u32 n_opp = args.opp_len[table];
    const auto order = args.opp_order.subspan(args.opp_off[table], n_opp);
    const auto card_off = args.card_off.subspan(table * (kCards + 1), kCards + 1);
    const auto card_pos = args.card_pos.subspan(args.card_base[table], 2 * n_opp);
    const auto prefix = prefix_all.first(n_opp + 1);
    const auto weights = weights_all.first(n_opp);

    // Opponent reach in strength order, weakest first.
    for (u32 j = threadIdx.x; j < n_opp; j += kThreads) {
      weights[j] = reach[order[j]];
    }
    __syncthreads();

    // prefix[j] = reach of the j weakest opponent hands: each thread sums
    // its chunk, one block scan offsets the chunks, then the chunk's
    // prefixes are written sequentially.
    const auto [begin, end] = my_chunk(n_opp);
    A chunk_total{};
    for (u32 j = begin; j < end; ++j) {
      chunk_total = add(chunk_total, weights[j]);
    }
    A running{};
    A all{};
    BlockScan{scan_storage}.ExclusiveScan(chunk_total, running, A{}, Plus{}, all);
    for (u32 j = begin; j < end; ++j) {
      prefix[j] = running;
      running = add(running, weights[j]);
    }
    if (threadIdx.x == 0) {
      prefix[n_opp] = all;
    }

    // Card c's opponent hands, in strength order, are card_pos[card_off[c]
    // .. card_off[c+1]]; their prefix sums go to card_prefix[card_off[c] + c
    // ..= card_off[c+1] + c].
    card_prefix_sums(card_off, card_pos, Span<const float>{weights}, card_prefix_all);
    __syncthreads();

    // The prefix sums of card `c`'s list.
    const auto card_prefix = [&](u32 c) {
      return card_prefix_all.subspan(card_off[c] + c, card_off[c + 1] - card_off[c] + 1);
    };
    const auto table_ranges = row(args.ranges, table, hands);
    const auto table_ranks = row(args.ranks, table, hands);
    const bool one_at_a_time = args.scalar[node] != 0u;
    const A win = narrow_from<A>(args.win[k]);
    const A lose = narrow_from<A>(args.lose[k]);
    const bool ties_pay = args.tie[k] != 0.0;
    const A tie = narrow_from<A>(args.tie[k]);
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      const u32 packed = table_ranges[h];
      if (packed == kNone) {
        out[h] = 0.0f;  // the hand conflicts with this board
        continue;
      }
      const auto range = unpack_range(packed);
      const auto ranks = unpack_ranks(table_ranks[h]);
      const auto cards = unpack_cards(args.my_cards[h]);
      const auto first = card_prefix(cards.first);
      const auto second = card_prefix(cards.second);
      const A first_all = first[first.size() - 1];
      const A second_all = second[second.size() - 1];

      // Reach of the weaker / stronger opponent hands, and of those among
      // them holding each of the hand's two cards.
      const A weaker_all = prefix[range.lo];
      const A weaker_first = first[ranks.first_lo];
      const A weaker_second = second[ranks.second_lo];
      const A stronger_all = sub(prefix[n_opp], prefix[range.hi]);
      const A stronger_first = sub(first_all, first[ranks.first_hi]);
      const A stronger_second = sub(second_all, second[ranks.second_hi]);
      float value = 0.0f;
      if (one_at_a_time) {
        // `eval_showdown`: out = (win * (sum - (ma + mb))) as f32, then
        // out += (lose * (sum - (ma + mb))) as f32.
        const A weaker = sub(weaker_all, add(weaker_first, weaker_second));
        const A stronger = sub(stronger_all, add(stronger_first, stronger_second));
        value = add_rn(to_f32(mul(win, weaker)), to_f32(mul(lose, stronger)));
      } else {
        // `showdown_batch`: r = win * (sum - ma - mb); r += lose * (sum - ma
        // - mb); out = r as f32.
        const A weaker = sub(sub(weaker_all, weaker_first), weaker_second);
        const A stronger = sub(sub(stronger_all, stronger_first), stronger_second);
        value = to_f32(add(mul(win, weaker), mul(lose, stronger)));
      }

      if (ties_pay) {
        // CPU `add_ties`: eq = eq_sum - (ma + mb) + same; out += (tie * eq) as f32.
        const A tied = sub(prefix[range.hi], prefix[range.lo]);
        const A first_tied = sub(first[ranks.first_hi], first[ranks.first_lo]);
        const A second_tied = sub(second[ranks.second_hi], second[ranks.second_lo]);
        const A eq = add(sub(tied, add(first_tied, second_tied)), same_reach<A>(args.same, reach, h));
        value = add_rn(value, to_f32(mul(tie, eq)));
      }
      out[h] = value;
    }

    // The paired fold, for every hand (the CPU's `showdown_batch`): payoff *
    // (sum - minus[a] - minus[b] + same) over the same reach. A separate
    // pass: computing it inside the loop above measured 30% slower.
    if (fold_node != kNone) {
      const auto fold_out = row(args.values, args.rows[fold_node], hands);
      const A pay = narrow_from<A>(args.paired_payoff[k]);
      const bool fold_one_at_a_time = args.scalar[fold_node] != 0u;
      for (u32 h = threadIdx.x; h < hands; h += kThreads) {
        const auto cards = unpack_cards(args.my_cards[h]);
        const auto first = card_prefix(cards.first);
        const auto second = card_prefix(cards.second);
        const A compatible = fold_compatible(fold_one_at_a_time, prefix[n_opp], first[first.size() - 1],
                                             second[second.size() - 1], same_reach<A>(args.same, reach, h));
        fold_out[h] = to_f32(mul(pay, compatible));
      }
    }
    __syncthreads();  // shared memory is reused by the next terminal
  }
}

}  // namespace
}  // namespace solver

using solver::u32;

// The C entry points: each kernel once per precision (`*_exact`: f64
// accumulators, `*_fast`: double-single), so each is compiled for its own
// register budget. One entry point picking the precision at run time held
// the fast path to the exact path's registers: 9% slower showdowns on the
// turn. The macros only stamp out the two entry points' identical
// parameter lists.

/// Fold terminals `[segment_start, segment_start + segment_count)` of the
/// list: `value[h] = payoff * compatible_reach(h)`. Dynamic shared memory:
/// `opp_hands` floats.
#define SOLVER_FOLD_ENTRY(NAME, ACC)                                                  \
  extern "C" __global__ void __launch_bounds__(solver::kThreads) NAME(                \
      const u32* list_ptr,                                                            \
      u32 list_len,                                                                   \
      const double* payoff_ptr,                                                       \
      u32 payoff_len,                                                                 \
      const u32* slots_ptr,                                                           \
      u32 slots_len,                                                                  \
      const u32* scalar_ptr,                                                          \
      u32 scalar_len,                                                                 \
      const u32* alive_ptr,                                                           \
      u32 alive_len,                                                                  \
      const float* reach_ptr,                                                         \
      u32 reach_len,                                                                  \
      const u32* rows_ptr,                                                            \
      u32 rows_len,                                                                   \
      float* values_ptr,                                                              \
      u32 values_len,                                                                 \
      const u32* my_cards_ptr,                                                        \
      u32 my_cards_len,                                                               \
      const u32* same_ptr,                                                            \
      u32 same_len,                                                                   \
      const u32* card_off_ptr,                                                        \
      u32 card_off_len,                                                               \
      const u32* card_hands_ptr,                                                      \
      u32 card_hands_len,                                                             \
      u32 segment_start,                                                              \
      u32 segment_count,                                                              \
      u32 hands,                                                                      \
      u32 opp_hands) {                                                                \
    using namespace solver;                                                           \
    const FoldArgs args{                                                              \
        .list = view(list_ptr, list_len).subspan(segment_start, segment_count),       \
        .payoff = view(payoff_ptr, payoff_len).subspan(segment_start, segment_count), \
        .slots = view(slots_ptr, slots_len),                                          \
        .scalar = view(scalar_ptr, scalar_len),                                       \
        .alive = view(alive_ptr, alive_len),                                          \
        .reach = view(reach_ptr, reach_len),                                          \
        .values = view(values_ptr, values_len),                                       \
        .rows = view(rows_ptr, rows_len),                                             \
        .my_cards = view(my_cards_ptr, my_cards_len),                                 \
        .same = view(same_ptr, same_len),                                             \
        .card_off = view(card_off_ptr, card_off_len),                                 \
        .card_hands = view(card_hands_ptr, card_hands_len),                           \
        .hands = hands,                                                               \
        .opp_hands = opp_hands,                                                       \
    };                                                                                \
    fold_values_impl<ACC>(args);                                                      \
  }

SOLVER_FOLD_ENTRY(fold_values_exact, double)
SOLVER_FOLD_ENTRY(fold_values_fast, DoubleFloat)
#undef SOLVER_FOLD_ENTRY

/// Showdown terminals `[segment_start, segment_start + segment_count)` of
/// the list: `value[h] = win * weaker(h) + lose * stronger(h)`, plus
/// `tie * tied(h)` when ties pay something (mid-street roots, rake). A
/// paired fold (`paired_fold[k]`, payoff `paired_payoff[k]`) shares the
/// showdown's opponent reach and is evaluated from the same sums.
#define SOLVER_SHOWDOWN_ENTRY(NAME, ACC)                                                                   \
  extern "C" __global__ void __launch_bounds__(solver::kThreads) NAME(                                     \
      const u32* list_ptr,                                                                                 \
      u32 list_len,                                                                                        \
      const u32* table_of_ptr,                                                                             \
      u32 table_of_len,                                                                                    \
      const double* win_ptr,                                                                               \
      u32 win_len,                                                                                         \
      const double* lose_ptr,                                                                              \
      u32 lose_len,                                                                                        \
      const double* tie_ptr,                                                                               \
      u32 tie_len,                                                                                         \
      const u32* paired_fold_ptr,                                                                          \
      u32 paired_fold_len,                                                                                 \
      const double* paired_payoff_ptr,                                                                     \
      u32 paired_payoff_len,                                                                               \
      const u32* slots_ptr,                                                                                \
      u32 slots_len,                                                                                       \
      const u32* scalar_ptr,                                                                               \
      u32 scalar_len,                                                                                      \
      const u32* alive_ptr,                                                                                \
      u32 alive_len,                                                                                       \
      const float* reach_ptr,                                                                              \
      u32 reach_len,                                                                                       \
      const u32* rows_ptr,                                                                                 \
      u32 rows_len,                                                                                        \
      float* values_ptr,                                                                                   \
      u32 values_len,                                                                                      \
      const u32* my_cards_ptr,                                                                             \
      u32 my_cards_len,                                                                                    \
      const u32* same_ptr,                                                                                 \
      u32 same_len,                                                                                        \
      const u32* opp_off_ptr,                                                                              \
      u32 opp_off_len,                                                                                     \
      const u32* opp_len_ptr,                                                                              \
      u32 opp_len_len,                                                                                     \
      const u32* opp_order_ptr,                                                                            \
      u32 opp_order_len,                                                                                   \
      const u32* card_base_ptr,                                                                            \
      u32 card_base_len,                                                                                   \
      const u32* card_off_ptr,                                                                             \
      u32 card_off_len,                                                                                    \
      const u32* card_pos_ptr,                                                                             \
      u32 card_pos_len,                                                                                    \
      const u32* ranges_ptr,                                                                               \
      u32 ranges_len,                                                                                      \
      const u32* ranks_ptr,                                                                                \
      u32 ranks_len,                                                                                       \
      u32 segment_start,                                                                                   \
      u32 segment_count,                                                                                   \
      u32 hands,                                                                                           \
      u32 opp_hands,                                                                                       \
      u32 max_opp) {                                                                                       \
    using namespace solver;                                                                                \
    const ShowdownArgs args{                                                                               \
        .list = view(list_ptr, list_len).subspan(segment_start, segment_count),                            \
        .table_of = view(table_of_ptr, table_of_len).subspan(segment_start, segment_count),                \
        .win = view(win_ptr, win_len).subspan(segment_start, segment_count),                               \
        .lose = view(lose_ptr, lose_len).subspan(segment_start, segment_count),                            \
        .tie = view(tie_ptr, tie_len).subspan(segment_start, segment_count),                               \
        .paired_fold = view(paired_fold_ptr, paired_fold_len).subspan(segment_start, segment_count),       \
        .paired_payoff = view(paired_payoff_ptr, paired_payoff_len).subspan(segment_start, segment_count), \
        .slots = view(slots_ptr, slots_len),                                                               \
        .scalar = view(scalar_ptr, scalar_len),                                                            \
        .alive = view(alive_ptr, alive_len),                                                               \
        .reach = view(reach_ptr, reach_len),                                                               \
        .values = view(values_ptr, values_len),                                                            \
        .rows = view(rows_ptr, rows_len),                                                                  \
        .my_cards = view(my_cards_ptr, my_cards_len),                                                      \
        .same = view(same_ptr, same_len),                                                                  \
        .opp_off = view(opp_off_ptr, opp_off_len),                                                         \
        .opp_len = view(opp_len_ptr, opp_len_len),                                                         \
        .opp_order = view(opp_order_ptr, opp_order_len),                                                   \
        .card_base = view(card_base_ptr, card_base_len),                                                   \
        .card_off = view(card_off_ptr, card_off_len),                                                      \
        .card_pos = view(card_pos_ptr, card_pos_len),                                                      \
        .ranges = view(ranges_ptr, ranges_len),                                                            \
        .ranks = view(ranks_ptr, ranks_len),                                                               \
        .hands = hands,                                                                                    \
        .opp_hands = opp_hands,                                                                            \
        .max_opp = max_opp,                                                                                \
    };                                                                                                     \
    showdown_values_impl<ACC>(args);                                                                       \
  }

SOLVER_SHOWDOWN_ENTRY(showdown_values_exact, double)
SOLVER_SHOWDOWN_ENTRY(showdown_values_fast, DoubleFloat)
#undef SOLVER_SHOWDOWN_ENTRY
