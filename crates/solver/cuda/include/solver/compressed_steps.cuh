// The two traversal steps that update storage, for 16-bit ("compressed")
// storage, one thread block per node: the opponent split (top-down,
// strategy sums) and the traverser's own update (bottom-up, regrets).
// Shared by the heads-up kernels (traverse_compressed.cu) and the
// three-player ones (multi_traverse_compressed.cu); the other steps don't
// touch storage and are the f32 path's.
//
// CPU twins: `Storage::update_strat` / `Storage::update_regrets` around the
// same per-hand math as traversal.cuh. Re-encoding a node's block needs that
// block's new largest absolute value: pass 1 computes the new values (and
// writes the node's outputs) while tracking the maximum, a block-wide
// reduction yields the new scale, and pass 2 recomputes the same values and
// encodes them. Recomputing instead of buffering keeps registers low; the
// inputs pass 2 reads are the ones pass 1 just read (still cached), and
// every value is recomputed by the same thread with the same operations, so
// both passes agree exactly.
#pragma once

#include <solver/compressed.cuh>
#include <solver/core.cuh>
#include <solver/strategy.cuh>
#include <solver/traversal.cuh>

#include <cub/block/block_reduce.cuh>

namespace solver {

/// Threads per block of the 16-bit steps.
inline constexpr u32 kCompressedThreads = 256;

/// Most actions per node (the host's `MAX_ACTIONS`): alive children are
/// tracked as one bit each in a u32.
inline constexpr u32 kMaxActions = 32;
static_assert(kMaxActions == sizeof(u32) * 8, "one bit per action in the alive mask");

struct MaxOp {
  [[nodiscard]] __device__ auto operator()(float a, float b) const noexcept -> float { return fmaxf(a, b); }
};

struct OrOp {
  [[nodiscard]] __device__ auto operator()(u32 a, u32 b) const noexcept -> u32 { return a | b; }
};

using BlockMax = cub::BlockReduce<float, kCompressedThreads>;
using BlockOr = cub::BlockReduce<u32, kCompressedThreads>;

/// The kernels' shared memory for one node.
struct CompressedShared {
  typename BlockMax::TempStorage max_storage;
  typename BlockOr::TempStorage or_storage;
  float node_max;
};

/// The largest of every thread's `local` (CPU: `max_abs` over the block),
/// for every thread of the block.
[[nodiscard]] __device__ inline auto block_max(float local, CompressedShared& sh) noexcept -> float {
  const float m = BlockMax{sh.max_storage}.Reduce(local, MaxOp{});
  if (threadIdx.x == 0) {
    sh.node_max = m;
  }
  __syncthreads();
  return sh.node_max;
}

/// Top-down step at one live opponent node: the current strategy comes
/// straight from the i16 regrets (the scale cancels out in regret matching,
/// as on the CPU); with `update` (CFR mode) the u16 strategy sums are
/// decoded, updated and re-encoded. Every thread of the block must call it.
struct CompressedSplit {
  Span<const u32> actions;
  Span<const u32> offsets;
  Span<const u32> scale_slots;
  Span<const u32> slots;
  Span<const u32> child_slots;
  Span<const i16> regrets;
  Span<u16> sums;
  Span<float> strategy_scale;
  Span<float> reach;
  Span<u32> alive;
  u32 opp_hands;
  Mode mode;
  float strategy_discount;
  bool weighted;

  __device__ void operator()(u32 node, bool update, CompressedShared& sh) const noexcept {
    const u32 slot = slots[node];
    const u32 num_actions = actions[node];
    const u32 first_child_slot = child_slots[node];
    const u32 scale_slot = scale_slots[node];
    const Block<const i16> node_regrets = block_at(regrets, offsets[node], num_actions, opp_hands);
    const Block<u16> sums_q = block_at(sums, offsets[node], num_actions, opp_hands);
    const Scaled<u16> node_sums{sums_q, strategy_step(strategy_scale[scale_slot])};

    // The probability of action `a` for hand `h`, and with `update` the new
    // strategy sum (CPU: x * g + y, or x * g + y * w).
    const auto cfr_inv = [&](u32 h) { return recip(positive_sum(node_regrets, h)); };
    const auto updated_sum = [&](u32 a, u32 h, float p, float parent) {
      const float kept = mul_rn(node_sums(a, h), strategy_discount);
      return add_rn(kept, weighted ? mul_rn(p, parent) : p);
    };

    // Pass 1: children's reach, which children are alive, the new sums' maximum.
    u32 alive_bits = 0;
    float local_max = 0.0f;
    for (u32 h = threadIdx.x; h < opp_hands; h += kCompressedThreads) {
      const float parent = row(reach, slot, opp_hands)[h];
      const float inv = mode == Mode::Cfr ? cfr_inv(h) : recip(strategy_total(node_sums, h));
      for (const u32 a : indices(num_actions)) {
        const float p =
            mode == Mode::Cfr ? matched(node_regrets, a, h, inv) : normalized(node_sums, a, h, inv);
        if (update) {
          local_max = fmaxf(local_max, fabsf(updated_sum(a, h, p, parent)));
        }
        const float child = mul_rn(parent, p);
        row(reach, first_child_slot + a, opp_hands)[h] = child;
        if (child != 0.0f) {
          alive_bits |= 1u << a;
        }
      }
    }
    const u32 any_alive = BlockOr{sh.or_storage}.Reduce(alive_bits, OrOp{});
    if (threadIdx.x == 0) {
      for (const u32 a : indices(num_actions)) {
        if ((any_alive >> a & 1u) != 0u) {
          mark_alive(alive[first_child_slot + a]);
        }
      }
    }

    if (update) {
      // Pass 2: the same sums, encoded with the block's new scale.
      const float m = block_max(local_max, sh);
      const float inv_q = m > 0.0f ? __fdiv_rn(kStrategyMax, m) : 0.0f;
      for (u32 h = threadIdx.x; h < opp_hands; h += kCompressedThreads) {
        const float parent = row(reach, slot, opp_hands)[h];
        const float inv = cfr_inv(h);
        for (const u32 a : indices(num_actions)) {
          const float p = matched(node_regrets, a, h, inv);
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
};

/// Bottom-up step at one live node of the traverser: its values into
/// `out`, and in CFR mode the i16 regrets decoded, updated (CPU:
/// `cfr::own_node_update`) and re-encoded. Every thread of the block must
/// call it.
template <ChildValue C>
struct CompressedOwnUpdate {
  Span<const u32> actions;
  Span<const u32> offsets;
  Span<const u32> scale_slots;
  Span<const u32> first_child;
  Span<i16> regrets;
  Span<float> regret_scale;
  Span<const u16> sums;
  Span<const float> strategy_scale;
  C child_value;
  u32 hands;
  Mode mode;
  float positive_discount;
  float negative_discount;

  __device__ void operator()(u32 node, Span<float> out, CompressedShared& sh) const noexcept {
    const u32 num_actions = actions[node];
    const u32 child0 = first_child[node];
    const u32 scale_slot = scale_slots[node];
    const auto cfv = [&](u32 a, u32 h) -> float { return child_value(child0 + a, h); };

    if (mode == Mode::BestResponse) {
      for (u32 h = threadIdx.x; h < hands; h += kCompressedThreads) {
        float best = cfv(0, h);
        for (u32 a = 1; a < num_actions; ++a) {
          best = fmaxf(best, cfv(a, h));
        }
        out[h] = best;
      }
      return;  // no shared memory used
    }
    if (mode == Mode::Average) {
      const Block<const u16> sums_q = block_at(sums, offsets[node], num_actions, hands);
      const Scaled<u16> node_sums{sums_q, strategy_step(strategy_scale[scale_slot])};
      for (u32 h = threadIdx.x; h < hands; h += kCompressedThreads) {
        const float inv = recip(strategy_total(node_sums, h));
        float value = 0.0f;
        for (const u32 a : indices(num_actions)) {
          value = add_rn(value, mul_rn(normalized(node_sums, a, h, inv), cfv(a, h)));
        }
        out[h] = value;
      }
      return;
    }

    // CFR: the value under regret matching of the decoded regrets, then the
    // update r * discount + cfv - value (traversal.cuh `OwnUpdate`).
    const Block<i16> regrets_q = block_at(regrets, offsets[node], num_actions, hands);
    const Scaled<i16> node_regrets{regrets_q, regret_step(regret_scale[scale_slot])};
    const auto updated_regret = [&](u32 a, u32 h, float value) {
      const float r = node_regrets(a, h);
      const float discount = r > 0.0f ? positive_discount : negative_discount;
      return sub_rn(add_rn(mul_rn(r, discount), cfv(a, h)), value);
    };

    // Pass 1: node values and the new regrets' maximum.
    float local_max = 0.0f;
    for (u32 h = threadIdx.x; h < hands; h += kCompressedThreads) {
      float positive = 0.0f;
      float weighted = 0.0f;
      for (const u32 a : indices(num_actions)) {
        const float r = positive_part(node_regrets(a, h));
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
    const float m = block_max(local_max, sh);
    const float inv_q = m > 0.0f ? __fdiv_rn(kRegretMax, m) : 0.0f;
    for (u32 h = threadIdx.x; h < hands; h += kCompressedThreads) {
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
};

}  // namespace solver
