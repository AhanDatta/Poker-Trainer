// Building blocks of the terminal kernels (one thread block per terminal),
// shared by the heads-up kernels (terminal.cu) and the three-player ones
// (multi_terminal.cu): row helpers, chunked sums, and the per-board prefix
// sums a showdown reads its weaker / tied / stronger opponent mass from.
//
// Sums and prefix sums run sequentially per thread over contiguous chunks
// (or a quarter of one card's list), with only the chunk totals combined by
// a block-wide scan or reduction (or warp shuffles). With f64 accumulators
// that is the CPU's arithmetic in another summation order: the sums of f32
// reach are exact in f64 unless their magnitudes span more than about 29
// binary orders, so the order rarely matters (it hasn't in any test).
#pragma once

#include <solver/core.cuh>
#include <solver/double_float.cuh>

#include <cub/block/block_scan.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/utility>

namespace solver {

/// Threads per block of the terminal kernels (a compile-time constant:
/// loops stride by it, which measured a few percent faster than striding
/// by `blockDim.x`). 512-thread showdown blocks, to fill the warp slots
/// that the shared-memory limit of 3 blocks per SM leaves empty, measured
/// 10% slower; reading the opponent weights from the (cached) reach row
/// instead of shared memory, for 4 blocks per SM, 3% slower.
inline constexpr u32 kThreads = 256;

/// Writes 0 for every hand (a dead terminal is worth nothing).
__device__ inline void zero_row(Span<float> out) noexcept {
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
[[nodiscard]] __device__ inline auto my_chunk(u32 len) noexcept -> cuda::std::pair<u32, u32> {
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

/// One traverser's showdown lookup against one opponent, for every board
/// (the host's `Tables`, src/gpu/tables.rs).
struct ShowdownTables {
  Span<const u32> opp_off;
  Span<const u32> opp_len;
  Span<const u32> opp_order;
  Span<const u32> card_base;
  Span<const u32> card_off;
  Span<const u32> card_pos;
  Span<const u32> ranges;
  Span<const u32> ranks;
};

/// The dynamic shared memory of a showdown block, sized by the host for
/// `max_opp` opponent hands (8 bytes per accumulator for both precisions;
/// `plan::showdown_shared_bytes`): prefix[max_opp + 1] and
/// card_prefix[2 * max_opp + 52], then the gathered weights[max_opp] (f32).
template <Accumulator A>
struct ShowdownShared {
  Span<A> prefix;
  Span<A> card_prefix;
  Span<float> weights;

  [[nodiscard]] __device__ static auto at(unsigned char* shared, u32 max_opp) noexcept -> ShowdownShared {
    A* const accumulators = reinterpret_cast<A*>(shared);
    return {Span<A>{accumulators, max_opp + 1}, Span<A>{accumulators + max_opp + 1, 2 * max_opp + kCards},
            Span<float>{reinterpret_cast<float*>(accumulators + 3 * max_opp + 1 + kCards), max_opp}};
  }
};

/// One board's prefix sums of the opponent's reach, in shared memory: the
/// reach of the `j` weakest opponent hands, and per card the same over the
/// opponent hands holding it.
template <Accumulator A>
struct StrengthSums {
  /// `prefix[j]`, `j = 0 ..= n_opp`.
  Span<A> prefix;
  /// All cards' prefix sums (see `card_prefix_sums`).
  Span<A> card_prefix_all;
  /// This board's per-card offsets.
  Span<const u32> card_off;
  /// The opponent's hands on this board.
  u32 n_opp;

  /// The prefix sums of card `c`'s list.
  [[nodiscard]] __device__ auto card(u32 c) const noexcept -> Span<A> {
    return card_prefix_all.subspan(card_off[c] + c, card_off[c + 1] - card_off[c] + 1);
  }
};

/// Builds board `table`'s `StrengthSums` of `reach` in `shared`. Every
/// thread of the block must call it; it ends with a barrier, and the
/// caller must synchronize again before the shared memory is reused.
template <Accumulator A>
__device__ auto strength_sums(const ShowdownTables& tables,
                              u32 table,
                              Span<const float> reach,
                              const ShowdownShared<A>& shared,
                              typename cub::BlockScan<A, kThreads>::TempStorage& scan_storage) noexcept
    -> StrengthSums<A> {
  using BlockScan = cub::BlockScan<A, kThreads>;
  const u32 n_opp = tables.opp_len[table];
  const auto order = tables.opp_order.subspan(tables.opp_off[table], n_opp);
  const auto card_off = tables.card_off.subspan(table * (kCards + 1), kCards + 1);
  const auto card_pos = tables.card_pos.subspan(tables.card_base[table], 2 * n_opp);
  const auto prefix = shared.prefix.first(n_opp + 1);
  const auto weights = shared.weights.first(n_opp);

  // Opponent reach in strength order, weakest first.
  for (u32 j = threadIdx.x; j < n_opp; j += kThreads) {
    weights[j] = reach[order[j]];
  }
  __syncthreads();

  // prefix[j] = reach of the j weakest opponent hands: each thread sums its
  // chunk, one block scan offsets the chunks, then the chunk's prefixes are
  // written sequentially.
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
  card_prefix_sums(card_off, card_pos, Span<const float>{weights}, shared.card_prefix);
  __syncthreads();
  return {prefix, shared.card_prefix, card_off, n_opp};
}

}  // namespace solver
