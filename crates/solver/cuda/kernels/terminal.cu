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
// Sums and prefix sums: see terminal.cuh (shared with multi_terminal.cu).
//
// Showdowns replace the CPU's sequential strength sweep by prefix sums over
// the board's strength-sorted opponent hands, plus one prefix sum per card
// over the opponent hands holding it. Where each of the traverser's hands
// sits in those lists depends only on the board, so the host precomputes it
// (src/gpu/plan.rs, `Tables`). A fold that shares its opponent reach with
// the showdown (both children of one traverser node) is read off the same
// sums, as the CPU's `showdown_batch` does.

#include <solver/terminal.cuh>

#include <cub/block/block_reduce.cuh>

/// Dynamic shared memory of the terminal kernels (layouts at each kernel).
extern __shared__ alignas(16) unsigned char terminal_shared[];

namespace solver {
namespace {

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
  ShowdownTables tables;
  u32 hands;
  u32 opp_hands;
  u32 max_opp;
};

/// Showdown terminals with accumulator `A`. Dynamic shared memory: see
/// `ShowdownShared`.
template <Accumulator A>
__device__ void showdown_values_impl(const ShowdownArgs& args) {
  __shared__ typename cub::BlockScan<A, kThreads>::TempStorage scan_storage;
  const auto shared = ShowdownShared<A>::at(terminal_shared, args.max_opp);
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
    const StrengthSums<A> sums = strength_sums(args.tables, table, reach, shared, scan_storage);
    const auto& prefix = sums.prefix;
    const u32 n_opp = sums.n_opp;
    const auto card_prefix = [&](u32 c) { return sums.card(c); };
    const auto table_ranges = row(args.tables.ranges, table, hands);
    const auto table_ranks = row(args.tables.ranks, table, hands);
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
        .tables = {view(opp_off_ptr, opp_off_len), view(opp_len_ptr, opp_len_len),                        \
                   view(opp_order_ptr, opp_order_len), view(card_base_ptr, card_base_len),                 \
                   view(card_off_ptr, card_off_len), view(card_pos_ptr, card_pos_len),                     \
                   view(ranges_ptr, ranges_len), view(ranks_ptr, ranks_len)},                              \
        .hands = hands,                                                                                    \
        .opp_hands = opp_hands,                                                                            \
        .max_opp = max_opp,                                                                                \
    };                                                                                                     \
    showdown_values_impl<ACC>(args);                                                                       \
  }

SOLVER_SHOWDOWN_ENTRY(showdown_values_exact, double)
SOLVER_SHOWDOWN_ENTRY(showdown_values_fast, DoubleFloat)
#undef SOLVER_SHOWDOWN_ENTRY
