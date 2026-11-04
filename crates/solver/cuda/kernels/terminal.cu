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
// Sums are accumulated in f64 like the CPU's (the order differs: tree
// reductions and scans instead of one sequential sweep); the f64 work is a
// tiny share of the solve, so the GPU's slow f64 rate doesn't matter here.
// Final combinations use the CPU's operation order without fused
// multiply-adds.
//
// Showdowns replace the CPU's sequential strength sweep by prefix sums over
// the board's strength-sorted opponent hands (plus one prefix sum per card
// over the opponent hands holding it). Where each of the traverser's hands
// sits in those lists depends only on the board, so the host precomputes it
// (src/gpu/plan.rs, `Tables`).

#include <solver/core.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

namespace solver {
namespace {

inline constexpr u32 kThreads = 256;
inline constexpr u32 kWarpSize = 32;
inline constexpr u32 kWarps = kThreads / kWarpSize;

[[nodiscard]] __device__ auto warp_id() noexcept -> u32 {
  return threadIdx.x / kWarpSize;
}

[[nodiscard]] __device__ auto lane_id() noexcept -> u32 {
  return threadIdx.x % kWarpSize;
}

/// Writes 0 for every hand (a dead terminal is worth nothing).
__device__ void zero_row(cuda::std::span<float> out) noexcept {
  for (u32 h = threadIdx.x; h < out.size(); h += kThreads) {
    out[h] = 0.0f;
  }
}

/// Reach of the opponent's copy of hand `h`'s combo, or 0.
[[nodiscard]] __device__ auto same_reach(cuda::std::span<const u32> same,
                                         cuda::std::span<const float> reach,
                                         u32 h) noexcept -> double {
  const u32 j = same[h];
  return j == kNone ? 0.0 : static_cast<double>(reach[j]);
}

}  // namespace
}  // namespace solver

using solver::u32;

/// Fold terminals: `value[h] = payoff * compatible_reach(h)`.
extern "C" __global__ void __launch_bounds__(solver::kThreads) fold_values(const u32* list_ptr,
                                                                           u32 list_len,
                                                                           const double* payoff_ptr,
                                                                           u32 payoff_len,
                                                                           const u32* slots_ptr,
                                                                           u32 slots_len,
                                                                           const u32* scalar_ptr,
                                                                           u32 scalar_len,
                                                                           const u32* alive_ptr,
                                                                           u32 alive_len,
                                                                           const float* reach_ptr,
                                                                           u32 reach_len,
                                                                           float* values_ptr,
                                                                           u32 values_len,
                                                                           const u32* my_cards_ptr,
                                                                           u32 my_cards_len,
                                                                           const u32* same_ptr,
                                                                           u32 same_len,
                                                                           const u32* card_off_ptr,
                                                                           u32 card_off_len,
                                                                           const u32* card_hands_ptr,
                                                                           u32 card_hands_len,
                                                                           u32 hands,
                                                                           u32 opp_hands) {
  using namespace solver;
  using BlockReduce = cub::BlockReduce<double, kThreads>;
  using WarpReduce = cub::WarpReduce<double>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ typename WarpReduce::TempStorage warp_storage[kWarps];
  __shared__ double minus_storage[kCards];
  __shared__ double total;

  const auto list = view(list_ptr, list_len);
  const auto payoff = view(payoff_ptr, payoff_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto scalar = view(scalar_ptr, scalar_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto reach_all = view(reach_ptr, reach_len);
  const auto values = view(values_ptr, values_len);
  const auto my_cards = view(my_cards_ptr, my_cards_len);
  const auto same = view(same_ptr, same_len);
  const auto card_off = view(card_off_ptr, card_off_len);
  const auto card_hands = view(card_hands_ptr, card_hands_len);
  const auto minus = cuda::std::span<double, kCards>{minus_storage};

  for (u32 k = blockIdx.x; k < list.size(); k += gridDim.x) {
    const u32 node = list[k];
    const u32 slot = slots[node];
    const auto out = row(values, node, hands);
    if (alive[slot] == 0u) {
      zero_row(out);
      continue;  // uniform: every thread of the block skips together
    }
    const auto reach = row(reach_all, slot, opp_hands);

    double partial = 0.0;
    for (u32 j = threadIdx.x; j < opp_hands; j += kThreads) {
      partial += reach[j];
    }
    const double block_total = BlockReduce{reduce_storage}.Sum(partial);
    if (threadIdx.x == 0) {
      total = block_total;
    }
    // minus[c]: one warp per card.
    for (u32 c = warp_id(); c < kCards; c += kWarps) {
      double m = 0.0;
      for (u32 e = card_off[c] + lane_id(); e < card_off[c + 1]; e += kWarpSize) {
        m += reach[card_hands[e]];
      }
      const double card_total = WarpReduce{warp_storage[warp_id()]}.Sum(m);
      if (lane_id() == 0) {
        minus[c] = card_total;
      }
    }
    __syncthreads();

    const double pay = payoff[k];
    const bool one_at_a_time = scalar[node] != 0u;
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      const auto cards = unpack_cards(my_cards[h]);
      const double m1 = minus[cards.first];
      const double m2 = minus[cards.second];
      // `eval_fold`: total - (m1 + m2) + same; batched: total - m1 - m2 + same.
      const double blocked = one_at_a_time ? sub_rn(total, add_rn(m1, m2)) : sub_rn(sub_rn(total, m1), m2);
      out[h] = to_f32(mul_rn(pay, add_rn(blocked, same_reach(same, reach, h))));
    }
    __syncthreads();  // shared memory is reused by the next terminal
  }
}

/// Showdown terminals: `value[h] = win * weaker(h) + lose * stronger(h)`,
/// plus `tie * tied(h)` when ties pay something (mid-street roots, rake).
extern "C" __global__ void __launch_bounds__(solver::kThreads) showdown_values(const u32* list_ptr,
                                                                               u32 list_len,
                                                                               const u32* table_of_ptr,
                                                                               u32 table_of_len,
                                                                               const double* win_ptr,
                                                                               u32 win_len,
                                                                               const double* lose_ptr,
                                                                               u32 lose_len,
                                                                               const double* tie_ptr,
                                                                               u32 tie_len,
                                                                               const u32* slots_ptr,
                                                                               u32 slots_len,
                                                                               const u32* scalar_ptr,
                                                                               u32 scalar_len,
                                                                               const u32* alive_ptr,
                                                                               u32 alive_len,
                                                                               const float* reach_ptr,
                                                                               u32 reach_len,
                                                                               float* values_ptr,
                                                                               u32 values_len,
                                                                               const u32* my_cards_ptr,
                                                                               u32 my_cards_len,
                                                                               const u32* same_ptr,
                                                                               u32 same_len,
                                                                               const u32* opp_off_ptr,
                                                                               u32 opp_off_len,
                                                                               const u32* opp_len_ptr,
                                                                               u32 opp_len_len,
                                                                               const u32* opp_order_ptr,
                                                                               u32 opp_order_len,
                                                                               const u32* card_base_ptr,
                                                                               u32 card_base_len,
                                                                               const u32* card_off_ptr,
                                                                               u32 card_off_len,
                                                                               const u32* card_pos_ptr,
                                                                               u32 card_pos_len,
                                                                               const u32* ranges_ptr,
                                                                               u32 ranges_len,
                                                                               const u32* ranks_ptr,
                                                                               u32 ranks_len,
                                                                               u32 hands,
                                                                               u32 opp_hands,
                                                                               u32 max_opp) {
  using namespace solver;
  using BlockScan = cub::BlockScan<double, kThreads>;
  using WarpScan = cub::WarpScan<double>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ typename WarpScan::TempStorage warp_storage[kWarps];
  // Dynamic shared memory, sized by the host for `max_opp` opponent hands:
  // prefix[max_opp + 1] and card_prefix[2 * max_opp + 52] (f64), then the
  // gathered weights[max_opp] (f32).
  extern __shared__ double dynamic_shared[];
  const auto prefix_all = cuda::std::span<double>{dynamic_shared, max_opp + 1};
  const auto card_prefix_all = cuda::std::span<double>{dynamic_shared + max_opp + 1, 2 * max_opp + kCards};
  const auto weights_all =
      cuda::std::span<float>{reinterpret_cast<float*>(dynamic_shared + 3 * max_opp + 1 + kCards), max_opp};

  const auto list = view(list_ptr, list_len);
  const auto table_of = view(table_of_ptr, table_of_len);
  const auto win = view(win_ptr, win_len);
  const auto lose = view(lose_ptr, lose_len);
  const auto tie = view(tie_ptr, tie_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto scalar = view(scalar_ptr, scalar_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto reach_all = view(reach_ptr, reach_len);
  const auto values = view(values_ptr, values_len);
  const auto my_cards = view(my_cards_ptr, my_cards_len);
  const auto same = view(same_ptr, same_len);
  const auto opp_off = view(opp_off_ptr, opp_off_len);
  const auto opp_len = view(opp_len_ptr, opp_len_len);
  const auto opp_order_all = view(opp_order_ptr, opp_order_len);
  const auto card_base = view(card_base_ptr, card_base_len);
  const auto card_off_all = view(card_off_ptr, card_off_len);
  const auto card_pos_all = view(card_pos_ptr, card_pos_len);
  const auto ranges = view(ranges_ptr, ranges_len);
  const auto ranks_all = view(ranks_ptr, ranks_len);

  for (u32 k = blockIdx.x; k < list.size(); k += gridDim.x) {
    const u32 node = list[k];
    const u32 slot = slots[node];
    const auto out = row(values, node, hands);
    if (alive[slot] == 0u) {
      zero_row(out);
      continue;  // uniform: every thread of the block skips together
    }
    const auto reach = row(reach_all, slot, opp_hands);
    const u32 table = table_of[k];
    const u32 n_opp = opp_len[table];
    const auto order = opp_order_all.subspan(opp_off[table], n_opp);
    const auto card_off = card_off_all.subspan(table * (kCards + 1), kCards + 1);
    const auto card_pos = card_pos_all.subspan(card_base[table], 2 * n_opp);
    const auto prefix = prefix_all.first(n_opp + 1);
    const auto weights = weights_all.first(n_opp);

    // prefix[j] = reach of the j weakest opponent hands.
    double running = 0.0;
    for (u32 base = 0; base < n_opp; base += kThreads) {
      const u32 j = base + threadIdx.x;
      const float w = j < n_opp ? reach[order[j]] : 0.0f;
      if (j < n_opp) {
        weights[j] = w;
      }
      double before = 0.0;
      double tile_total = 0.0;
      BlockScan{scan_storage}.ExclusiveSum(static_cast<double>(w), before, tile_total);
      if (j < n_opp) {
        prefix[j] = running + before;
      }
      running += tile_total;
      __syncthreads();  // scan storage is reused by the next tile
    }
    if (threadIdx.x == 0) {
      prefix[n_opp] = running;
    }

    // Card c's opponent hands, in strength order, occupy positions
    // card_off[c]..card_off[c+1] of card_pos; their prefix sums go to
    // card_prefix[card_off[c] + c ..= card_off[c+1] + c]. One warp per card.
    for (u32 c = warp_id(); c < kCards; c += kWarps) {
      const u32 begin = card_off[c];
      const u32 len = card_off[c + 1] - begin;
      const auto segment = card_prefix_all.subspan(begin + c, len + 1);
      double carry = 0.0;
      for (u32 base = 0; base < len; base += kWarpSize) {
        const u32 e = base + lane_id();
        const double w = e < len ? static_cast<double>(weights[card_pos[begin + e]]) : 0.0;
        double before = 0.0;
        double chunk_total = 0.0;
        WarpScan{warp_storage[warp_id()]}.ExclusiveSum(w, before, chunk_total);
        if (e < len) {
          segment[e] = carry + before;
        }
        carry += chunk_total;
      }
      if (lane_id() == 0) {
        segment[len] = carry;
      }
    }
    __syncthreads();

    const auto table_ranges = row(ranges, table, hands);
    const auto table_ranks = row(ranks_all, table, hands);
    const bool one_at_a_time = scalar[node] != 0u;
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      const u32 packed = table_ranges[h];
      if (packed == kNone) {
        out[h] = 0.0f;  // the hand conflicts with this board
        continue;
      }
      const auto range = unpack_range(packed);
      const auto ranks = unpack_ranks(table_ranks[h]);
      const auto cards = unpack_cards(my_cards[h]);
      const auto first = card_prefix_all.subspan(card_off[cards.first] + cards.first,
                                                 card_off[cards.first + 1] - card_off[cards.first] + 1);
      const auto second = card_prefix_all.subspan(card_off[cards.second] + cards.second,
                                                  card_off[cards.second + 1] - card_off[cards.second] + 1);
      const u32 first_len = static_cast<u32>(first.size()) - 1;
      const u32 second_len = static_cast<u32>(second.size()) - 1;

      // Reach of the weaker / stronger opponent hands, and of those among
      // them holding each of the hand's two cards.
      const double weaker_all = prefix[range.lo];
      const double weaker_first = first[ranks.first_lo];
      const double weaker_second = second[ranks.second_lo];
      const double stronger_all = sub_rn(prefix[n_opp], prefix[range.hi]);
      const double stronger_first = sub_rn(first[first_len], first[ranks.first_hi]);
      const double stronger_second = sub_rn(second[second_len], second[ranks.second_hi]);
      float value = 0.0f;
      if (one_at_a_time) {
        // `eval_showdown`: out = (win * (sum - (ma + mb))) as f32, then
        // out += (lose * (sum - (ma + mb))) as f32.
        const double weaker = sub_rn(weaker_all, add_rn(weaker_first, weaker_second));
        const double stronger = sub_rn(stronger_all, add_rn(stronger_first, stronger_second));
        value = add_rn(to_f32(mul_rn(win[k], weaker)), to_f32(mul_rn(lose[k], stronger)));
      } else {
        // `showdown_batch`: r = win * (sum - ma - mb); r += lose * (sum - ma
        // - mb); out = r as f32.
        const double weaker = sub_rn(sub_rn(weaker_all, weaker_first), weaker_second);
        const double stronger = sub_rn(sub_rn(stronger_all, stronger_first), stronger_second);
        value = to_f32(add_rn(mul_rn(win[k], weaker), mul_rn(lose[k], stronger)));
      }

      if (tie[k] != 0.0) {
        // CPU `add_ties`: eq = eq_sum - (ma + mb) + same; out += (tie * eq) as f32.
        const double tied = sub_rn(prefix[range.hi], prefix[range.lo]);
        const double first_tied = sub_rn(first[ranks.first_hi], first[ranks.first_lo]);
        const double second_tied = sub_rn(second[ranks.second_hi], second[ranks.second_lo]);
        const double eq = add_rn(sub_rn(tied, add_rn(first_tied, second_tied)), same_reach(same, reach, h));
        value = add_rn(value, to_f32(mul_rn(tie[k], eq)));
      }
      out[h] = value;
    }
    __syncthreads();  // shared memory is reused by the next terminal
  }
}
