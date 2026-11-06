// Three-player leaf values with independent opponents, one thread block
// per leaf.
//
// CPU twins: `multi::cfr::pay_all` and `multi::cfr::independent_showdown`
// with `exact_removal` off (the CPU then evaluates every leaf one at a
// time; it batches river subtrees only with exact removal). For each
// traverser hand `h` and opponent `o` the CPU needs
// * the compatible mass `c_o(h) = max(total - minus[c1] - minus[c2] +
//   same, 0)` (`Hands::compat`), and
// * at showdowns, the beaten mass `b_o(h) = max(weaker + equal / 2, 0)`
//   (`Table::beat`; `c_o` when `o` folded, 0 for hands conflicting with
//   the board),
// and combines them as `pay * c_q * c_r` (fixed payoffs) or `win * p +
// lose * (c_q * c_r - p)` with `p = b_q * b_r` (showdowns), all in f64,
// rounded to f32 at the end. The sums are built per opponent with the
// heads-up machinery (terminal.cuh): per-card lists for fixed payoffs, the
// board's strength-order prefix sums for showdowns. A showdown handles the
// first opponent, keeps its two values per hand in registers, then reuses
// the shared memory for the second.
//
// f64 only for now (the CPU's arithmetic); a double-single fast path comes
// after these are checked against the CPU, as heads-up.

#include <solver/terminal.cuh>
#include <solver/traversal.cuh>

#include <cub/block/block_reduce.cuh>

extern __shared__ alignas(16) unsigned char terminal_shared[];

namespace solver {
namespace {

/// Most hands a player can have (all combos); bounds the per-thread hand
/// count of the showdown kernel (the plan checks it).
inline constexpr u32 kMaxHands = 1326;
inline constexpr u32 kHandsPerThread = (kMaxHands + kThreads - 1) / kThreads;

/// `max(x, 0)` as Rust's `f64::max` gives it (no -0.0 or NaN reaches it).
[[nodiscard]] __device__ inline auto clamp_positive(double x) noexcept -> double { return x > 0.0 ? x : 0.0; }

/// `Hands::compat`: `max(total - m1 - m2 + same, 0)`, in that order.
[[nodiscard]] __device__ inline auto compatible(double total, double m1, double m2, double same) noexcept -> double {
  return clamp_positive(add_rn(sub_rn(sub_rn(total, m1), m2), same));
}

/// One opponent as a leaf kernel reads it.
struct Opponent {
  /// The opponent's seat (bit of the plan's `folded` flags).
  u32 seat;
  u32 hands;
  Span<const u32> slots;
  Span<const u32> alive;
  Span<const float> reach;
  /// Per traverser hand: the opponent's identical combo, or `kNone`.
  Span<const u32> same;
  /// The opponent's hands holding each card (53 offsets).
  Span<const u32> card_off;
  Span<const u32> card_hands;
  ShowdownTables tables;
};

/// Both opponents' liveness.
[[nodiscard]] __device__ auto live(const Opponent& first, const Opponent& second, u32 node) noexcept -> bool {
  return TwoOpponents{first.slots, first.alive, second.slots, second.alive}(node);
}

/// The views a fixed-payoff launch works on.
struct FixedArgs {
  Span<const u32> list;
  Span<const double> payoff;
  Span<const u32> rows;
  Span<float> values;
  Span<const u32> my_cards;
  u32 hands;
  Opponent opp[2];
};

/// Fixed-payoff leaves (CPU: `pay_all`, `(pay * c_q * c_r) as f32`).
/// Dynamic shared memory: one opponent's reach (the larger hand count of
/// the two, floats).
__device__ void fixed_values_impl(const FixedArgs& args) {
  using BlockReduce = cub::BlockReduce<double, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ double minus_storage[2][kCards];
  __shared__ double totals[2];
  const u32 hands = args.hands;
  for (u32 k = blockIdx.x; k < args.list.size(); k += gridDim.x) {
    const u32 node = args.list[k];
    const auto out = row(args.values, args.rows[node], hands);
    if (!live(args.opp[0], args.opp[1], node)) {
      zero_row(out);
      continue;  // uniform: every thread of the block skips together
    }
    for (u32 o = 0; o < 2; ++o) {
      const Opponent& opp = args.opp[o];
      const auto weights = Span<float>{reinterpret_cast<float*>(terminal_shared), opp.hands};
      const auto reach = row(opp.reach, opp.slots[node], opp.hands);
      for (u32 j = threadIdx.x; j < opp.hands; j += kThreads) {
        weights[j] = reach[j];
      }
      __syncthreads();
      const auto [begin, end] = my_chunk(opp.hands);
      double partial = 0.0;
      for (u32 j = begin; j < end; ++j) {
        partial = add_rn(partial, static_cast<double>(weights[j]));
      }
      const double total = BlockReduce{reduce_storage}.Reduce(partial, Plus{});
      if (threadIdx.x == 0) {
        totals[o] = total;
      }
      // minus[c]: one thread per card, summing that card's list in hand order.
      for (u32 c = threadIdx.x; c < kCards; c += kThreads) {
        double m = 0.0;
        const u32 last = opp.card_off[c + 1];
#pragma unroll 8
        for (u32 e = opp.card_off[c]; e < last; ++e) {
          m = add_rn(m, static_cast<double>(weights[opp.card_hands[e]]));
        }
        minus_storage[o][c] = m;
      }
      __syncthreads();  // the weights and the reduction storage are reused
    }
    const double pay = args.payoff[k];
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      const auto cards = unpack_cards(args.my_cards[h]);
      double mass[2];
#pragma unroll
      for (u32 o = 0; o < 2; ++o) {
        const Opponent& opp = args.opp[o];
        const auto reach = row(opp.reach, opp.slots[node], opp.hands);
        mass[o] = compatible(totals[o], minus_storage[o][cards.first], minus_storage[o][cards.second],
                             same_reach<double>(opp.same, reach, h));
      }
      out[h] = to_f32(mul_rn(mul_rn(pay, mass[0]), mass[1]));
    }
    __syncthreads();  // shared memory is reused by the next leaf
  }
}

/// The views a showdown launch works on.
struct ShowdownArgs {
  Span<const u32> list;
  Span<const u32> table_of;
  Span<const double> win;
  Span<const double> lose;
  Span<const u32> folded;
  Span<const u32> rows;
  Span<float> values;
  Span<const u32> my_cards;
  u32 hands;
  u32 max_opp;
  Opponent opp[2];
};

/// Independent showdowns (CPU: `independent_showdown`). Dynamic shared
/// memory: `ShowdownShared<double>` for `max_opp` (the longer opponent list
/// of either opponent).
__device__ void showdown_values_impl(const ShowdownArgs& args) {
  __shared__ typename cub::BlockScan<double, kThreads>::TempStorage scan_storage;
  const auto shared = ShowdownShared<double>::at(terminal_shared, args.max_opp);
  const u32 hands = args.hands;
  for (u32 k = blockIdx.x; k < args.list.size(); k += gridDim.x) {
    const u32 node = args.list[k];
    const auto out = row(args.values, args.rows[node], hands);
    if (!live(args.opp[0], args.opp[1], node)) {
      zero_row(out);
      continue;  // uniform: every thread of the block skips together
    }
    const u32 table = args.table_of[k];
    const u32 folded = args.folded[node];
    const double win = args.win[k];
    const double lose = args.lose[k];
    // The first opponent's compatible and beaten mass of this thread's hands.
    double first_compat[kHandsPerThread];
    double first_beat[kHandsPerThread];
#pragma unroll
    for (u32 o = 0; o < 2; ++o) {
      const Opponent& opp = args.opp[o];
      const auto reach = row(opp.reach, opp.slots[node], opp.hands);
      const StrengthSums<double> sums = strength_sums(opp.tables, table, reach, shared, scan_storage);
      const bool opp_folded = ((folded >> opp.seat) & 1u) != 0u;
      const auto ranges = row(opp.tables.ranges, table, hands);
      const auto ranks = row(opp.tables.ranks, table, hands);
      const auto& prefix = sums.prefix;
#pragma unroll
      for (u32 i = 0; i < kHandsPerThread; ++i) {
        const u32 h = threadIdx.x + i * kThreads;
        if (h >= hands) {
          continue;
        }
        const auto cards = unpack_cards(args.my_cards[h]);
        const auto first = sums.card(cards.first);
        const auto second = sums.card(cards.second);
        const double same = same_reach<double>(opp.same, reach, h);
        const double compat = compatible(prefix[sums.n_opp], first[first.size() - 1], second[second.size() - 1], same);
        double beat = compat;  // a folded opponent's hand beats nothing
        if (!opp_folded) {
          const u32 packed = ranges[h];
          if (packed == kNone) {
            beat = 0.0;  // the hand conflicts with this board
          } else {
            // `Table::beat`: weaker = sum - minus[a] - minus[b]; equal = eq
            // - eq_minus[a] - eq_minus[b] + same; max(weaker + 0.5 * equal, 0).
            const auto range = unpack_range(packed);
            const auto r = unpack_ranks(ranks[h]);
            const double weaker = sub_rn(sub_rn(prefix[range.lo], first[r.first_lo]), second[r.second_lo]);
            const double tied = sub_rn(prefix[range.hi], prefix[range.lo]);
            const double first_tied = sub_rn(first[r.first_hi], first[r.first_lo]);
            const double second_tied = sub_rn(second[r.second_hi], second[r.second_lo]);
            const double equal = add_rn(sub_rn(sub_rn(tied, first_tied), second_tied), same);
            beat = clamp_positive(add_rn(weaker, mul_rn(0.5, equal)));
          }
        }
        if (o == 0) {
          first_compat[i] = compat;
          first_beat[i] = beat;
        } else {
          // total = c_q * c_r; p = b_q * b_r; win * p + lose * (total - p).
          const double total = mul_rn(first_compat[i], compat);
          const double pwin = mul_rn(first_beat[i], beat);
          out[h] = to_f32(add_rn(mul_rn(win, pwin), mul_rn(lose, sub_rn(total, pwin))));
        }
      }
      __syncthreads();  // shared memory is reused by the next opponent or leaf
    }
  }
}

}  // namespace
}  // namespace solver

using solver::u32;

// Both leaf kernels take the two opponents as the same run of argument
// pairs; the macro only stamps out that list and the struct it fills.
#define SOLVER_OPPONENT_PARAMS(P)                                                                     \
  u32 P##seat, u32 P##hands, const u32 *P##slots_ptr, u32 P##slots_len, const u32 *P##alive_ptr,      \
      u32 P##alive_len, const float *P##reach_ptr, u32 P##reach_len, const u32 *P##same_ptr,          \
      u32 P##same_len, const u32 *P##card_off_ptr, u32 P##card_off_len, const u32 *P##card_hands_ptr, \
      u32 P##card_hands_len, const u32 *P##opp_off_ptr, u32 P##opp_off_len, const u32 *P##opp_len_ptr, \
      u32 P##opp_len_len, const u32 *P##opp_order_ptr, u32 P##opp_order_len, const u32 *P##card_base_ptr, \
      u32 P##card_base_len, const u32 *P##table_card_off_ptr, u32 P##table_card_off_len,             \
      const u32 *P##card_pos_ptr, u32 P##card_pos_len, const u32 *P##ranges_ptr, u32 P##ranges_len,   \
      const u32 *P##ranks_ptr, u32 P##ranks_len

#define SOLVER_OPPONENT(P)                                                                             \
  solver::Opponent {                                                                                   \
    P##seat, P##hands, view(P##slots_ptr, P##slots_len), view(P##alive_ptr, P##alive_len),             \
        view(P##reach_ptr, P##reach_len), view(P##same_ptr, P##same_len),                              \
        view(P##card_off_ptr, P##card_off_len), view(P##card_hands_ptr, P##card_hands_len),            \
        ShowdownTables {                                                                               \
      view(P##opp_off_ptr, P##opp_off_len), view(P##opp_len_ptr, P##opp_len_len),                     \
          view(P##opp_order_ptr, P##opp_order_len), view(P##card_base_ptr, P##card_base_len),          \
          view(P##table_card_off_ptr, P##table_card_off_len), view(P##card_pos_ptr, P##card_pos_len),  \
          view(P##ranges_ptr, P##ranges_len), view(P##ranks_ptr, P##ranks_len)                        \
    }                                                                                                  \
  }

/// Fixed-payoff leaves: `value[h] = payoff * c_q(h) * c_r(h)`. Dynamic
/// shared memory: the larger opponent hand count, floats.
extern "C" __global__ void __launch_bounds__(solver::kThreads)
    multi_fixed_values(const u32* list_ptr,
                       u32 list_len,
                       const double* payoff_ptr,
                       u32 payoff_len,
                       const u32* rows_ptr,
                       u32 rows_len,
                       float* values_ptr,
                       u32 values_len,
                       const u32* my_cards_ptr,
                       u32 my_cards_len,
                       u32 hands,
                       SOLVER_OPPONENT_PARAMS(q_),
                       SOLVER_OPPONENT_PARAMS(r_)) {
  using namespace solver;
  const FixedArgs args{
      .list = view(list_ptr, list_len),
      .payoff = view(payoff_ptr, payoff_len),
      .rows = view(rows_ptr, rows_len),
      .values = view(values_ptr, values_len),
      .my_cards = view(my_cards_ptr, my_cards_len),
      .hands = hands,
      .opp = {SOLVER_OPPONENT(q_), SOLVER_OPPONENT(r_)},
  };
  fixed_values_impl(args);
}

/// Independent showdowns: `value[h] = win * p + lose * (c_q * c_r - p)` with
/// `p = b_q * b_r`. Dynamic shared memory: `ShowdownShared<double>` for
/// `max_opp`.
extern "C" __global__ void __launch_bounds__(solver::kThreads)
    multi_showdown_values(const u32* list_ptr,
                          u32 list_len,
                          const u32* table_of_ptr,
                          u32 table_of_len,
                          const double* win_ptr,
                          u32 win_len,
                          const double* lose_ptr,
                          u32 lose_len,
                          const u32* folded_ptr,
                          u32 folded_len,
                          const u32* rows_ptr,
                          u32 rows_len,
                          float* values_ptr,
                          u32 values_len,
                          const u32* my_cards_ptr,
                          u32 my_cards_len,
                          u32 hands,
                          u32 max_opp,
                          SOLVER_OPPONENT_PARAMS(q_),
                          SOLVER_OPPONENT_PARAMS(r_)) {
  using namespace solver;
  const ShowdownArgs args{
      .list = view(list_ptr, list_len),
      .table_of = view(table_of_ptr, table_of_len),
      .win = view(win_ptr, win_len),
      .lose = view(lose_ptr, lose_len),
      .folded = view(folded_ptr, folded_len),
      .rows = view(rows_ptr, rows_len),
      .values = view(values_ptr, values_len),
      .my_cards = view(my_cards_ptr, my_cards_len),
      .hands = hands,
      .max_opp = max_opp,
      .opp = {SOLVER_OPPONENT(q_), SOLVER_OPPONENT(r_)},
  };
  showdown_values_impl(args);
}

#undef SOLVER_OPPONENT
#undef SOLVER_OPPONENT_PARAMS
