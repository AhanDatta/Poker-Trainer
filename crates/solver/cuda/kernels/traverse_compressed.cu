// The two traversal steps that update storage, for 16-bit ("compressed")
// storage: `opponent_split_compressed` (top-down, opponent's strategy sums)
// and `own_update_compressed` (bottom-up, traverser's regrets). The other
// steps don't touch storage and are shared with f32 storage (traverse.cu).
//
// CPU twins: `Storage::update_strat` / `Storage::update_regrets` around the
// same per-hand math as traverse.cu. Re-encoding a node's block needs that
// block's new largest absolute value, so one thread block owns one node:
// pass 1 computes the new values (and writes the node's outputs) while
// tracking the maximum, a block-wide reduction yields the new scale, and
// pass 2 recomputes the same values and encodes them. Recomputing instead
// of buffering keeps registers low; the inputs pass 2 reads are the ones
// pass 1 just read (still cached), and every value is recomputed by the
// same thread with the same operations, so both passes agree exactly.

#include <solver/compressed.cuh>
#include <solver/core.cuh>
#include <solver/strategy.cuh>

#include <cub/block/block_reduce.cuh>

namespace solver {
namespace {

inline constexpr u32 kThreads = 256;

/// Most actions per node (the host's `plan::MAX_ACTIONS`): alive children
/// are tracked as one bit each in a u32.
inline constexpr u32 kMaxActions = 32;
static_assert(kMaxActions == sizeof(u32) * 8, "one bit per action in the alive mask");

struct MaxOp {
  [[nodiscard]] __device__ auto operator()(float a, float b) const noexcept -> float { return fmaxf(a, b); }
};

struct OrOp {
  [[nodiscard]] __device__ auto operator()(u32 a, u32 b) const noexcept -> u32 { return a | b; }
};

using BlockMax = cub::BlockReduce<float, kThreads>;
using BlockOr = cub::BlockReduce<u32, kThreads>;

/// The largest of every thread's `local` (CPU: `max_abs` over the block),
/// for every thread of the block.
[[nodiscard]] __device__ auto block_max(float local,
                                        typename BlockMax::TempStorage& storage,
                                        float& broadcast) noexcept -> float {
  const float m = BlockMax{storage}.Reduce(local, MaxOp{});
  if (threadIdx.x == 0) {
    broadcast = m;
  }
  __syncthreads();
  return broadcast;
}

}  // namespace
}  // namespace solver

using solver::u16;
using solver::u32;

/// Top-down step at the opponent's decision nodes of one level, with 16-bit
/// storage. One block per node. The current strategy comes straight from
/// the i16 regrets (the scale cancels out in regret matching, as on the
/// CPU); the u16 strategy sums are decoded, updated and re-encoded.
extern "C" __global__ void __launch_bounds__(solver::kThreads)
    opponent_split_compressed(const u32* level_ptr,
                              u32 level_len,
                              u32 level_start,
                              u32 level_count,
                              const u32* actions_ptr,
                              u32 actions_len,
                              const u32* offsets_ptr,
                              u32 offsets_len,
                              const u32* scale_slots_ptr,
                              u32 scale_slots_len,
                              const u32* slots_ptr,
                              u32 slots_len,
                              const u32* child_slots_ptr,
                              u32 child_slots_len,
                              const solver::i16* regrets_ptr,
                              u32 regrets_len,
                              u16* sums_ptr,
                              u32 sums_len,
                              float* strategy_scale_ptr,
                              u32 strategy_scale_len,
                              float* reach_ptr,
                              u32 reach_len,
                              u32* alive_ptr,
                              u32 alive_len,
                              u32 opp_hands,
                              u32 mode_raw,
                              float strategy_discount,
                              u32 reach_weighted) {
  using namespace solver;
  __shared__ typename BlockMax::TempStorage max_storage;
  __shared__ typename BlockOr::TempStorage or_storage;
  __shared__ float node_max;

  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto offsets = view(offsets_ptr, offsets_len);
  const auto scale_slots = view(scale_slots_ptr, scale_slots_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto child_slots = view(child_slots_ptr, child_slots_len);
  const auto regrets_all = view(regrets_ptr, regrets_len);
  const auto sums_all = view(sums_ptr, sums_len);
  const auto strategy_scale = view(strategy_scale_ptr, strategy_scale_len);
  const auto reach = view(reach_ptr, reach_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto mode = static_cast<Mode>(mode_raw);
  const bool weighted = reach_weighted != 0u;

  for (u32 k = blockIdx.x; k < level_count; k += gridDim.x) {
    const u32 node = level[k];
    const u32 slot = slots[node];
    if (alive[slot] == 0u) {
      continue;  // dead: no reach for the children, no strategy-sum update
    }
    const u32 num_actions = actions[node];
    const u32 first_child_slot = child_slots[node];
    const u32 scale_slot = scale_slots[node];
    const Block<const i16> regrets = block_at(regrets_all, offsets[node], num_actions, opp_hands);
    const Block<u16> sums_q = block_at(sums_all, offsets[node], num_actions, opp_hands);
    const Scaled<u16> sums{sums_q, strategy_step(strategy_scale[scale_slot])};

    // The probability of action `a` for hand `h`, and in CFR mode the new
    // strategy sum (CPU: x * g + y, or x * g + y * w).
    const auto cfr_inv = [&](u32 h) { return recip(positive_sum(regrets, h)); };
    const auto updated_sum = [&](u32 a, u32 h, float p, float parent) {
      const float kept = mul_rn(sums(a, h), strategy_discount);
      return add_rn(kept, weighted ? mul_rn(p, parent) : p);
    };

    // Pass 1: children's reach, which children are alive, the new sums' maximum.
    u32 alive_bits = 0;
    float local_max = 0.0f;
    for (u32 h = threadIdx.x; h < opp_hands; h += kThreads) {
      const float parent = row(reach, slot, opp_hands)[h];
      const float inv = mode == Mode::Cfr ? cfr_inv(h) : recip(strategy_total(sums, h));
      for (const u32 a : indices(num_actions)) {
        const float p = mode == Mode::Cfr ? matched(regrets, a, h, inv) : normalized(sums, a, h, inv);
        if (mode == Mode::Cfr) {
          local_max = fmaxf(local_max, fabsf(updated_sum(a, h, p, parent)));
        }
        const float child = mul_rn(parent, p);
        row(reach, first_child_slot + a, opp_hands)[h] = child;
        if (child != 0.0f) {
          alive_bits |= 1u << a;
        }
      }
    }
    const u32 any_alive = BlockOr{or_storage}.Reduce(alive_bits, OrOp{});
    if (threadIdx.x == 0) {
      for (const u32 a : indices(num_actions)) {
        if ((any_alive >> a & 1u) != 0u) {
          mark_alive(alive[first_child_slot + a]);
        }
      }
    }

    if (mode == Mode::Cfr) {
      // Pass 2: the same sums, encoded with the block's new scale.
      const float m = block_max(local_max, max_storage, node_max);
      const float inv_q = m > 0.0f ? __fdiv_rn(kStrategyMax, m) : 0.0f;
      for (u32 h = threadIdx.x; h < opp_hands; h += kThreads) {
        const float parent = row(reach, slot, opp_hands)[h];
        const float inv = cfr_inv(h);
        for (const u32 a : indices(num_actions)) {
          const float p = matched(regrets, a, h, inv);
          const float updated = updated_sum(a, h, p, parent);
          sums_q(a, h) = m > 0.0f ? encode_strategy(updated, inv_q) : u16{0};
        }
      }
      __syncthreads();  // every thread decoded with the old scale
      if (threadIdx.x == 0) {
        strategy_scale[scale_slot] = m;
      }
    }
    __syncthreads();  // shared storage is reused by the next node
  }
}

/// Bottom-up step at the traverser's decision nodes of one level, with
/// 16-bit storage. One block per node. In CFR mode the i16 regrets are
/// decoded, updated (CPU: `cfr::own_node_update`) and re-encoded.
extern "C" __global__ void __launch_bounds__(solver::kThreads)
    own_update_compressed(const u32* level_ptr,
                          u32 level_len,
                          u32 level_start,
                          u32 level_count,
                          const u32* actions_ptr,
                          u32 actions_len,
                          const u32* offsets_ptr,
                          u32 offsets_len,
                          const u32* scale_slots_ptr,
                          u32 scale_slots_len,
                          const u32* first_child_ptr,
                          u32 first_child_len,
                          const u32* slots_ptr,
                          u32 slots_len,
                          const u32* alive_ptr,
                          u32 alive_len,
                          solver::i16* regrets_ptr,
                          u32 regrets_len,
                          float* regret_scale_ptr,
                          u32 regret_scale_len,
                          const u16* sums_ptr,
                          u32 sums_len,
                          const float* strategy_scale_ptr,
                          u32 strategy_scale_len,
                          const u32* rows_ptr,
                          u32 rows_len,
                          float* values_ptr,
                          u32 values_len,
                          u32 hands,
                          u32 mode_raw,
                          float positive_discount,
                          float negative_discount) {
  using namespace solver;
  __shared__ typename BlockMax::TempStorage max_storage;
  __shared__ float node_max;

  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto offsets = view(offsets_ptr, offsets_len);
  const auto scale_slots = view(scale_slots_ptr, scale_slots_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto regrets_all = view(regrets_ptr, regrets_len);
  const auto regret_scale = view(regret_scale_ptr, regret_scale_len);
  const auto sums_all = view(sums_ptr, sums_len);
  const auto strategy_scale = view(strategy_scale_ptr, strategy_scale_len);
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);
  const auto mode = static_cast<Mode>(mode_raw);

  for (u32 k = blockIdx.x; k < level_count; k += gridDim.x) {
    const u32 node = level[k];
    const auto out = row(values, rows[node], hands);
    if (alive[slots[node]] == 0u) {
      for (u32 h = threadIdx.x; h < hands; h += kThreads) {
        out[h] = 0.0f;
      }
      continue;  // uniform: the whole block skips the node
    }
    const u32 num_actions = actions[node];
    const u32 child0 = first_child[node];
    const u32 scale_slot = scale_slots[node];
    const auto cfv = [&](u32 a, u32 h) -> float { return row(values, rows[child0 + a], hands)[h]; };

    if (mode == Mode::BestResponse) {
      for (u32 h = threadIdx.x; h < hands; h += kThreads) {
        float best = cfv(0, h);
        for (u32 a = 1; a < num_actions; ++a) {
          best = fmaxf(best, cfv(a, h));
        }
        out[h] = best;
      }
      continue;  // no shared memory used
    }
    if (mode == Mode::Average) {
      const Block<const u16> sums_q = block_at(sums_all, offsets[node], num_actions, hands);
      const Scaled<u16> sums{sums_q, strategy_step(strategy_scale[scale_slot])};
      for (u32 h = threadIdx.x; h < hands; h += kThreads) {
        const float inv = recip(strategy_total(sums, h));
        float value = 0.0f;
        for (const u32 a : indices(num_actions)) {
          value = add_rn(value, mul_rn(normalized(sums, a, h, inv), cfv(a, h)));
        }
        out[h] = value;
      }
      continue;
    }

    // CFR: the value under regret matching of the decoded regrets, then the
    // update r * discount + cfv - value (traverse.cu `own_update`).
    const Block<i16> regrets_q = block_at(regrets_all, offsets[node], num_actions, hands);
    const Scaled<i16> regrets{regrets_q, regret_step(regret_scale[scale_slot])};
    const auto updated_regret = [&](u32 a, u32 h, float value) {
      const float r = regrets(a, h);
      const float discount = r > 0.0f ? positive_discount : negative_discount;
      return sub_rn(add_rn(mul_rn(r, discount), cfv(a, h)), value);
    };

    // Pass 1: node values and the new regrets' maximum.
    float local_max = 0.0f;
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      float positive = 0.0f;
      float weighted = 0.0f;
      for (const u32 a : indices(num_actions)) {
        const float r = positive_part(regrets(a, h));
        positive = add_rn(positive, r);
        weighted = add_rn(weighted, mul_rn(r, cfv(a, h)));
      }
      const float inv = recip(positive);
      float value = mul_rn(weighted, inv);
      if (inv == 0.0f) {
        float sum = -0.0f;  // Rust's float `Sum` starts at -0.0
        for (const u32 a : indices(num_actions)) {
          sum = add_rn(sum, cfv(a, h));
        }
        value = mul_rn(sum, 1.0f / static_cast<float>(num_actions));
      }
      out[h] = value;
      for (const u32 a : indices(num_actions)) {
        local_max = fmaxf(local_max, fabsf(updated_regret(a, h, value)));
      }
    }

    // Pass 2: the same regrets (from the value just written by this
    // thread), encoded with the block's new scale.
    const float m = block_max(local_max, max_storage, node_max);
    const float inv_q = m > 0.0f ? __fdiv_rn(kRegretMax, m) : 0.0f;
    for (u32 h = threadIdx.x; h < hands; h += kThreads) {
      const float value = out[h];
      for (const u32 a : indices(num_actions)) {
        const float updated = updated_regret(a, h, value);
        regrets_q(a, h) = m > 0.0f ? encode_regret(updated, inv_q) : i16{0};
      }
    }
    __syncthreads();  // every thread decoded with the old scale
    if (threadIdx.x == 0) {
      regret_scale[scale_slot] = m;
    }
    __syncthreads();  // shared storage is reused by the next node
  }
}
