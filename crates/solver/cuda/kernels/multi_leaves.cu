// Three-player leaf values, one leaf per lane (lanes.cuh), warps
// grid-striding over the plan's groups: showdown warps share a board and
// folded players, fixed-payoff lanes each take one pair of reach slots and
// every leaf sharing it.
//
// CPU twins, per terminal model:
// * independent opponents (independent.cuh): `multi::cfr::pay_all` and
//   `independent_showdown`;
// * exact removal (exact.cuh): `multi::exact::fold_values` and `showdown`
//   (also `showdown_lanes`, whose lanes compute the same); `river::subtree`
//   rounds the pair mass to f32 before the payoff multiplies it, which the
//   plan's per-leaf rounding flag mirrors.
// Each kernel exists per model and per precision: `_exact` (f64,
// bit-identical to the CPU) and `_fast` (double-single).
//
// Launch: blocks of `kBlock` threads; thread t of the grid is lane t % 32
// of warp t / 32 and owns column t of the scratch arrays (`Lane`), so the
// host sizes the scratch for the whole grid.

#include <solver/exact.cuh>
#include <solver/independent.cuh>
#include <solver/traversal.cuh>

namespace solver {
namespace {

using lanes::HandList;
using lanes::kLanes;
using lanes::Lane;
using lanes::Lists;

/// The terminal models.
enum class Model { kIndependent, kExact };

/// One opponent's reach state as the leaf kernels read it.
struct OpponentLeaf {
  u32 hands;
  Span<const u32> slots;
  Span<const u32> alive;
  Span<const float> reach;
};

/// Column, lane, warp and warp count of this thread.
struct Grid {
  u32 column;
  u32 lane;
  u32 warp;
  u32 warps;
  u32 stride;
};

[[nodiscard]] __device__ inline auto grid() noexcept -> Grid {
  const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  const u32 stride = gridDim.x * blockDim.x;
  return Grid{t, t % kLanes, t / kLanes, stride / kLanes, stride};
}

/// The scratch arrays (every lane's columns).
template <Accumulator A>
struct Scratch {
  Span<A> state;
  Span<float> reach;
  Span<A> stash;

  [[nodiscard]] __device__ auto lane(const Grid& g) const noexcept -> Lane<A> {
    return Lane<A>{state, reach, stash, g.stride, g.column};
  }
};

/// The lane's reach rows: each opponent's hand-order list, then (at
/// showdowns) each opponent's board list.
struct ReachRows {
  u32 by_hand[2];
  u32 board[2];

  [[nodiscard]] __device__ static auto of(const OpponentLeaf (&opp)[2]) noexcept -> ReachRows {
    const u32 nq = opp[0].hands;
    const u32 n = nq + opp[1].hands;
    return ReachRows{{0u, nq}, {n, n + nq}};
  }
};

/// Opponent k's reach at slot `slot`.
[[nodiscard]] __device__ inline auto reach_of(const OpponentLeaf& opp, u32 slot) noexcept -> Span<const float> {
  return row(opp.reach, slot, opp.hands);
}

__device__ inline void zero(Span<float> out) noexcept {
  for (auto& v : out) {
    v = 0.0f;
  }
}

template <Accumulator A>
struct ShowdownArgs {
  Span<const u32> groups;
  Span<const u32> lanes;
  Span<const u32> showdowns;
  Span<const double> win;
  Span<const double> lose;
  Span<const double> split2;
  Span<const double> split3;
  Span<const u32> rows;
  Span<float> values;
  u32 hands;
  u32 seats[3];
  /// The hand-order sweep (the lists' last order).
  u32 hand_order;
  Lists lists;
  OpponentLeaf opp[2];
  Scratch<A> scratch;
};

template <Model M, Accumulator A>
__device__ void showdowns_impl(const ShowdownArgs<A>& args) {
  const Grid g = grid();
  const Lane<A> lane = args.scratch.lane(g);
  const ReachRows rows = ReachRows::of(args.opp);
  const u32 num_groups = static_cast<u32>(args.groups.size() / 4u);
  const auto [p, q, r] = args.seats;
  const TwoOpponents live{args.opp[0].slots, args.opp[0].alive, args.opp[1].slots, args.opp[1].alive};
  for (u32 w = g.warp; w < num_groups; w += g.warps) {
    const u32 order = args.groups[4u * w];
    const u32 folded = args.groups[4u * w + 1u];
    const u32 start = args.groups[4u * w + 2u];
    const u32 count = args.groups[4u * w + 3u];
    if (g.lane >= count) {
      continue;
    }
    const u32 i = args.lanes[start + g.lane];
    const u32 node = args.showdowns[i];
    const auto out = row(args.values, args.rows[node], args.hands);
    zero(out);  // dead nodes, and (exact removal) hands on this board's cards
    if (!live(node)) {
      continue;
    }
    const Span<const float> src[2] = {reach_of(args.opp[0], args.opp[0].slots[node]),
                                      reach_of(args.opp[1], args.opp[1].slots[node])};
    const bool in[2] = {((folded >> q) & 1u) == 0u, ((folded >> r) & 1u) == 0u};
    const HandList lp{args.lists, order, p};
    const HandList lo[2] = {HandList{args.lists, order, q, rows.board[0]}, HandList{args.lists, order, r, rows.board[1]}};
    lane.load(lo[0], src[0]);
    lane.load(lo[1], src[1]);
    if constexpr (M == Model::kExact) {
      const exact::Payoffs<A> pay{narrow_from<A>(args.win[i]), narrow_from<A>(args.lose[i]),
                                  narrow_from<A>(args.split2[i]), narrow_from<A>(args.split3[i])};
      exact::showdown(lane, lp, lo[0], lo[1], in[0], in[1], pay, out);
    } else {
      const HandList my_hands{args.lists, args.hand_order, p};
      const HandList by_hand[2] = {HandList{args.lists, args.hand_order, q, rows.by_hand[0]},
                                   HandList{args.lists, args.hand_order, r, rows.by_hand[1]}};
      lane.load(by_hand[0], src[0]);
      lane.load(by_hand[1], src[1]);
      independent::showdown(lane, lp, lo, my_hands, by_hand, in, narrow_from<A>(args.win[i]),
                            narrow_from<A>(args.lose[i]), out);
    }
  }
}

template <Accumulator A>
struct FixedArgs {
  Span<const u32> job_slots[2];
  Span<const u32> job_off;
  Span<const u32> job_leaves;
  Span<const u32> fixed;
  Span<const double> payoff;
  Span<const u32> scalar;
  Span<const u32> rows;
  Span<float> values;
  u32 hands;
  u32 seats[3];
  u32 hand_order;
  Lists lists;
  OpponentLeaf opp[2];
  Scratch<A> scratch;
};

/// Writes `pay * mass` for each leaf of a job (`mass` being the pair mass,
/// or `c_q`'s product with `c_r`): the CPU's `(pay * mass) as f32`, or for
/// batched river leaves `(pay * (mass as f32) as f64) as f32`.
template <Model M, Accumulator A>
__device__ void write_leaves(const FixedArgs<A>& args, Span<const u32> leaves, u32 h, A first, A second) noexcept {
  for (const u32 leaf : leaves) {
    const double pay = args.payoff[leaf];
    float v = 0.0f;
    if (args.scalar[leaf] != 0u) {
      // Exact removal: `pay * pairs`; independent: `pay * c_q * c_r`.
      v = M == Model::kExact ? to_f32(mul(narrow_from<A>(pay), first))
                             : to_f32(mul(mul(narrow_from<A>(pay), first), second));
    } else {
      const float mass = M == Model::kExact ? to_f32(first) : to_f32(mul(first, second));
      v = to_f32(mul_rn(pay, static_cast<double>(mass)));
    }
    row(args.values, args.rows[args.fixed[leaf]], args.hands)[h] = v;
  }
}

template <Model M, Accumulator A>
__device__ void fixed_impl(const FixedArgs<A>& args) {
  const Grid g = grid();
  const Lane<A> lane = args.scratch.lane(g);
  const ReachRows rows = ReachRows::of(args.opp);
  const u32 jobs = static_cast<u32>(args.job_slots[0].size());
  const auto [p, q, r] = args.seats;
  for (u32 w = g.warp; w * kLanes < jobs; w += g.warps) {
    const u32 job = w * kLanes + g.lane;
    if (job >= jobs) {
      continue;
    }
    const u32 sq = args.job_slots[0][job];
    const u32 sr = args.job_slots[1][job];
    const auto leaves = args.job_leaves.subspan(args.job_off[job], args.job_off[job + 1u] - args.job_off[job]);
    if (args.opp[0].alive[sq] == 0u || args.opp[1].alive[sr] == 0u) {
      for (const u32 leaf : leaves) {
        zero(row(args.values, args.rows[args.fixed[leaf]], args.hands));
      }
      continue;
    }
    const HandList my_hands{args.lists, args.hand_order, p};
    const HandList by_hand[2] = {HandList{args.lists, args.hand_order, q, rows.by_hand[0]},
                                 HandList{args.lists, args.hand_order, r, rows.by_hand[1]}};
    lane.load(by_hand[0], reach_of(args.opp[0], sq));
    lane.load(by_hand[1], reach_of(args.opp[1], sr));
    if constexpr (M == Model::kExact) {
      exact::pair_masses(lane, my_hands, by_hand[0], by_hand[1],
                         [&](u32 h, A pairs) { write_leaves<M>(args, leaves, h, pairs, A{}); });
    } else {
      independent::compat_masses(lane, my_hands, by_hand,
                                 [&](u32 h, A cq, A cr) { write_leaves<M>(args, leaves, h, cq, cr); });
    }
  }
}

}  // namespace
}  // namespace solver

using solver::u32;
using u16 = cuda::std::uint16_t;

// Arguments both kernels take in the same order: the lists, both
// opponents, the scratch.
#define SOLVER_LEAF_LIST_PARAMS                                                                         \
  const u32 *off_ptr, u32 off_len, const u32 *strength_ptr, u32 strength_len, const u32 *hand_ptr,      \
      u32 hand_len, const u32 *cards_ptr, u32 cards_len, const u32 *card_off_ptr, u32 card_off_len,     \
      const u32 *holding_ptr, u32 holding_len, const u16 *combo_pos_ptr, u32 combo_pos_len
#define SOLVER_LEAF_LISTS                                                                               \
  solver::lanes::Lists {                                                                                \
    view(off_ptr, off_len), view(strength_ptr, strength_len), view(hand_ptr, hand_len),                 \
        view(cards_ptr, cards_len), view(card_off_ptr, card_off_len), view(holding_ptr, holding_len), \
        view(combo_pos_ptr, combo_pos_len)                                                              \
  }
#define SOLVER_LEAF_OPPONENT_PARAMS(P)                                                                  \
  u32 P##hands, const u32 *P##slots_ptr, u32 P##slots_len, const u32 *P##alive_ptr, u32 P##alive_len,   \
      const float *P##reach_ptr, u32 P##reach_len
#define SOLVER_LEAF_OPPONENT(P)                                                                         \
  solver::OpponentLeaf {                                                                                \
    P##hands, view(P##slots_ptr, P##slots_len), view(P##alive_ptr, P##alive_len),                       \
        view(P##reach_ptr, P##reach_len)                                                                \
  }
#define SOLVER_LEAF_SCRATCH_PARAMS(A) \
  A *state_ptr, u32 state_len, float *reach_scratch_ptr, u32 reach_scratch_len, A *stash_ptr, u32 stash_len
#define SOLVER_LEAF_SCRATCH(A)                                                                          \
  solver::Scratch<A> {                                                                                  \
    view(state_ptr, state_len), view(reach_scratch_ptr, reach_scratch_len), view(stash_ptr, stash_len) \
  }

/// Showdowns. Grid: any number of `kBlock`-thread blocks, the scratch sized
/// for all of their threads.
#define SOLVER_SHOWDOWN_ENTRY(NAME, M, A)                                                               \
  extern "C" __global__ void __launch_bounds__(solver::lanes::kBlock)                                  \
      NAME(const u32* groups_ptr, u32 groups_len, const u32* lanes_ptr, u32 lanes_len,                  \
           const u32* showdowns_ptr, u32 showdowns_len, const double* win_ptr, u32 win_len,             \
           const double* lose_ptr, u32 lose_len, const double* split2_ptr, u32 split2_len,              \
           const double* split3_ptr, u32 split3_len, const u32* rows_ptr, u32 rows_len, float* values_ptr, \
           u32 values_len, u32 hands, u32 p, u32 q, u32 r, u32 hand_order, SOLVER_LEAF_LIST_PARAMS,      \
           SOLVER_LEAF_OPPONENT_PARAMS(q_), SOLVER_LEAF_OPPONENT_PARAMS(r_), SOLVER_LEAF_SCRATCH_PARAMS(A)) { \
    using namespace solver;                                                                             \
    const ShowdownArgs<A> args{                                                                         \
        .groups = view(groups_ptr, groups_len),                                                         \
        .lanes = view(lanes_ptr, lanes_len),                                                            \
        .showdowns = view(showdowns_ptr, showdowns_len),                                                \
        .win = view(win_ptr, win_len),                                                                  \
        .lose = view(lose_ptr, lose_len),                                                               \
        .split2 = view(split2_ptr, split2_len),                                                         \
        .split3 = view(split3_ptr, split3_len),                                                         \
        .rows = view(rows_ptr, rows_len),                                                               \
        .values = view(values_ptr, values_len),                                                         \
        .hands = hands,                                                                                 \
        .seats = {p, q, r},                                                                             \
        .hand_order = hand_order,                                                                       \
        .lists = SOLVER_LEAF_LISTS,                                                                     \
        .opp = {SOLVER_LEAF_OPPONENT(q_), SOLVER_LEAF_OPPONENT(r_)},                                    \
        .scratch = SOLVER_LEAF_SCRATCH(A),                                                              \
    };                                                                                                  \
    showdowns_impl<M>(args);                                                                            \
  }

/// Fixed payoffs. Grid as for the showdowns.
#define SOLVER_FIXED_ENTRY(NAME, M, A)                                                                  \
  extern "C" __global__ void __launch_bounds__(solver::lanes::kBlock)                                  \
      NAME(const u32* job_q_ptr, u32 job_q_len, const u32* job_r_ptr, u32 job_r_len, const u32* job_off_ptr, \
           u32 job_off_len, const u32* job_leaves_ptr, u32 job_leaves_len, const u32* fixed_ptr,          \
           u32 fixed_len, const double* payoff_ptr, u32 payoff_len, const u32* scalar_ptr, u32 scalar_len, \
           const u32* rows_ptr, u32 rows_len, float* values_ptr, u32 values_len, u32 hands, u32 p, u32 q,  \
           u32 r, u32 hand_order, SOLVER_LEAF_LIST_PARAMS, SOLVER_LEAF_OPPONENT_PARAMS(q_),              \
           SOLVER_LEAF_OPPONENT_PARAMS(r_), SOLVER_LEAF_SCRATCH_PARAMS(A)) {                             \
    using namespace solver;                                                                             \
    const FixedArgs<A> args{                                                                            \
        .job_slots = {view(job_q_ptr, job_q_len), view(job_r_ptr, job_r_len)},                          \
        .job_off = view(job_off_ptr, job_off_len),                                                      \
        .job_leaves = view(job_leaves_ptr, job_leaves_len),                                             \
        .fixed = view(fixed_ptr, fixed_len),                                                            \
        .payoff = view(payoff_ptr, payoff_len),                                                         \
        .scalar = view(scalar_ptr, scalar_len),                                                         \
        .rows = view(rows_ptr, rows_len),                                                               \
        .values = view(values_ptr, values_len),                                                         \
        .hands = hands,                                                                                 \
        .seats = {p, q, r},                                                                             \
        .hand_order = hand_order,                                                                       \
        .lists = SOLVER_LEAF_LISTS,                                                                     \
        .opp = {SOLVER_LEAF_OPPONENT(q_), SOLVER_LEAF_OPPONENT(r_)},                                    \
        .scratch = SOLVER_LEAF_SCRATCH(A),                                                              \
    };                                                                                                  \
    fixed_impl<M>(args);                                                                                \
  }

using solver::DoubleFloat;
using solver::Model;

SOLVER_SHOWDOWN_ENTRY(multi_showdowns_independent_exact, Model::kIndependent, double)
SOLVER_SHOWDOWN_ENTRY(multi_showdowns_independent_fast, Model::kIndependent, DoubleFloat)
SOLVER_SHOWDOWN_ENTRY(multi_showdowns_exact_exact, Model::kExact, double)
SOLVER_SHOWDOWN_ENTRY(multi_showdowns_exact_fast, Model::kExact, DoubleFloat)
SOLVER_FIXED_ENTRY(multi_fixed_independent_exact, Model::kIndependent, double)
SOLVER_FIXED_ENTRY(multi_fixed_independent_fast, Model::kIndependent, DoubleFloat)
SOLVER_FIXED_ENTRY(multi_fixed_exact_exact, Model::kExact, double)
SOLVER_FIXED_ENTRY(multi_fixed_exact_fast, Model::kExact, DoubleFloat)

#undef SOLVER_FIXED_ENTRY
#undef SOLVER_SHOWDOWN_ENTRY
#undef SOLVER_LEAF_SCRATCH
#undef SOLVER_LEAF_SCRATCH_PARAMS
#undef SOLVER_LEAF_OPPONENT
#undef SOLVER_LEAF_OPPONENT_PARAMS
#undef SOLVER_LEAF_LISTS
#undef SOLVER_LEAF_LIST_PARAMS
