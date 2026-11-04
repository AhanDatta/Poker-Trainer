// Level-synchronous traversal of the game tree for one traverser.
//
// CPU twin: `river::run` (and `cfr::own_node_update` for the regret
// update). The host plan (src/gpu/plan.rs) lists the decision nodes of
// every tree level; one launch handles one level:
//
// 1. `begin_traversal`: the root's opponent reach and the "alive" flags.
// 2. Top-down, `opponent_split` per level: the opponent's strategy splits
//    its reach among the children (in CFR mode its strategy sums are
//    updated on the way). A node whose opponent reach is all zero is dead:
//    worth 0 and never updated, exactly as on the CPU.
// 3. Terminals (terminal.cu).
// 4. Bottom-up, `own_update` and `opponent_sum` per level: node values,
//    and in CFR mode the traverser's regret updates.
//
// Reach lives in "slots" of `opp_hands` floats; the traverser's own nodes
// pass their slot to their children unchanged, so only opponent nodes write
// reach. Values live in one row of `hands` floats per tree node, and every
// node's row is written exactly once per traversal.
//
// Threads: x covers hands, y walks the level's nodes (grid-stride), so a
// block always works on a single node and branches on node state are
// uniform across the block.

#include <solver/core.cuh>
#include <solver/strategy.cuh>

namespace solver {
namespace {

/// Marks a reach slot alive. Several blocks may mark the same slot.
__device__ void mark_alive(u32& flag) noexcept {
  cuda::atomic_ref<u32, cuda::thread_scope_device>{flag}.store(1u, cuda::std::memory_order_relaxed);
}

[[nodiscard]] __device__ auto hand_index() noexcept -> u32 {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

}  // namespace
}  // namespace solver

using solver::u32;

/// Slot 0 gets the opponent's root weights and is the only live slot.
extern "C" __global__ void begin_traversal(u32* alive_ptr,
                                           u32 alive_len,
                                           float* reach_ptr,
                                           u32 reach_len,
                                           const float* weights_ptr,
                                           u32 weights_len) {
  const auto alive = solver::view(alive_ptr, alive_len);
  const auto reach = solver::view(reach_ptr, reach_len);
  const auto weights = solver::view(weights_ptr, weights_len);
  const u32 i = solver::hand_index();
  if (i < alive_len) {
    alive[i] = i == 0 ? 1u : 0u;
  }
  if (i < weights_len) {
    reach[i] = weights[i];
  }
}

/// Top-down step at the opponent's decision nodes of one level.
extern "C" __global__ void opponent_split(const u32* level_ptr,
                                          u32 level_len,
                                          u32 level_start,
                                          u32 level_count,
                                          const u32* actions_ptr,
                                          u32 actions_len,
                                          const u32* offsets_ptr,
                                          u32 offsets_len,
                                          const u32* slots_ptr,
                                          u32 slots_len,
                                          const u32* child_slots_ptr,
                                          u32 child_slots_len,
                                          const float* regrets_ptr,
                                          u32 regrets_len,
                                          float* sums_ptr,
                                          u32 sums_len,
                                          float* reach_ptr,
                                          u32 reach_len,
                                          u32* alive_ptr,
                                          u32 alive_len,
                                          u32 opp_hands,
                                          u32 mode_raw,
                                          float strategy_discount,
                                          u32 reach_weighted) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto offsets = view(offsets_ptr, offsets_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto child_slots = view(child_slots_ptr, child_slots_len);
  const auto regrets_all = view(regrets_ptr, regrets_len);
  const auto sums_all = view(sums_ptr, sums_len);
  const auto reach = view(reach_ptr, reach_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto mode = static_cast<Mode>(mode_raw);

  const u32 h = hand_index();
  const bool valid = h < opp_hands;
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    const u32 slot = slots[node];
    if (alive[slot] == 0u) {
      continue;  // dead: no reach for the children, no strategy-sum update
    }
    const u32 num_actions = actions[node];
    const u32 first_child_slot = child_slots[node];
    const Block<const float> regrets = block_at(regrets_all, offsets[node], num_actions, opp_hands);
    const Block<float> sums = block_at(sums_all, offsets[node], num_actions, opp_hands);
    const float parent = valid ? row(reach, slot, opp_hands)[h] : 0.0f;
    float inv = 0.0f;
    if (valid) {
      inv = mode == Mode::Cfr ? recip(positive_sum(regrets, h)) : recip(strategy_total(sums, h));
    }
    for (const u32 a : indices(num_actions)) {
      float child = 0.0f;
      if (valid) {
        float p = 0.0f;
        if (mode == Mode::Cfr) {
          // Current strategy; the average accumulates it (CPU: x * g + y,
          // or x * g + y * w with reach weighting).
          p = matched(regrets, a, h, inv);
          const float kept = mul_rn(sums(a, h), strategy_discount);
          sums(a, h) = add_rn(kept, reach_weighted != 0u ? mul_rn(p, parent) : p);
        } else {
          p = normalized(sums, a, h, inv);
        }
        child = mul_rn(parent, p);
        row(reach, first_child_slot + a, opp_hands)[h] = child;
      }
      // Every thread of the block reaches this (the node and its action
      // count are the same for the whole block).
      if (__syncthreads_or(child != 0.0f) != 0 && threadIdx.x == 0) {
        mark_alive(alive[first_child_slot + a]);
      }
    }
  }
}

/// Bottom-up step at the traverser's decision nodes of one level: the node
/// value from its children's values, and in CFR mode the regret update
/// (CPU: `cfr::own_node_update`).
extern "C" __global__ void own_update(const u32* level_ptr,
                                      u32 level_len,
                                      u32 level_start,
                                      u32 level_count,
                                      const u32* actions_ptr,
                                      u32 actions_len,
                                      const u32* offsets_ptr,
                                      u32 offsets_len,
                                      const u32* first_child_ptr,
                                      u32 first_child_len,
                                      const u32* slots_ptr,
                                      u32 slots_len,
                                      const u32* alive_ptr,
                                      u32 alive_len,
                                      float* regrets_ptr,
                                      u32 regrets_len,
                                      const float* sums_ptr,
                                      u32 sums_len,
                                      float* values_ptr,
                                      u32 values_len,
                                      u32 hands,
                                      u32 mode_raw,
                                      float positive_discount,
                                      float negative_discount) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto offsets = view(offsets_ptr, offsets_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto regrets_all = view(regrets_ptr, regrets_len);
  const auto sums_all = view(sums_ptr, sums_len);
  const auto values = view(values_ptr, values_len);
  const auto mode = static_cast<Mode>(mode_raw);

  const u32 h = hand_index();
  if (h >= hands) {
    return;  // no block-wide synchronization in this kernel
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    float& out = row(values, node, hands)[h];
    if (alive[slots[node]] == 0u) {
      out = 0.0f;
      continue;
    }
    const u32 num_actions = actions[node];
    const u32 child0 = first_child[node];
    const auto cfv = [&](u32 a) -> float { return row(values, child0 + a, hands)[h]; };
    switch (mode) {
      case Mode::BestResponse: {
        float best = cfv(0);
        for (u32 a = 1; a < num_actions; ++a) {
          best = fmaxf(best, cfv(a));
        }
        out = best;
        break;
      }
      case Mode::Average: {
        const Block<const float> sums = block_at(sums_all, offsets[node], num_actions, hands);
        const float inv = recip(strategy_total(sums, h));
        float value = 0.0f;
        for (const u32 a : indices(num_actions)) {
          value = add_rn(value, mul_rn(normalized(sums, a, h, inv), cfv(a)));
        }
        out = value;
        break;
      }
      case Mode::Cfr: {
        const Block<float> regrets = block_at(regrets_all, offsets[node], num_actions, hands);
        // Value under regret matching without materializing the strategy:
        // sum(max(r, 0) * cfv) / sum(max(r, 0)), or the mean over actions
        // when no regret is positive.
        float positive = 0.0f;
        float weighted = 0.0f;
        for (const u32 a : indices(num_actions)) {
          const float r = positive_part(regrets(a, h));
          positive = add_rn(positive, r);
          weighted = add_rn(weighted, mul_rn(r, cfv(a)));
        }
        const float inv = recip(positive);
        float value = mul_rn(weighted, inv);
        if (inv == 0.0f) {
          float sum = -0.0f;  // Rust's float `Sum` starts at -0.0
          for (const u32 a : indices(num_actions)) {
            sum = add_rn(sum, cfv(a));
          }
          value = mul_rn(sum, 1.0f / static_cast<float>(num_actions));
        }
        for (const u32 a : indices(num_actions)) {
          const float r = regrets(a, h);
          const float discount = r > 0.0f ? positive_discount : negative_discount;
          regrets(a, h) = sub_rn(add_rn(mul_rn(r, discount), cfv(a)), value);
        }
        out = value;
        break;
      }
    }
  }
}

/// Bottom-up step at the opponent's decision nodes of one level: the sum of
/// the children's values (the children's reach already carries the
/// opponent's strategy).
extern "C" __global__ void opponent_sum(const u32* level_ptr,
                                        u32 level_len,
                                        u32 level_start,
                                        u32 level_count,
                                        const u32* actions_ptr,
                                        u32 actions_len,
                                        const u32* first_child_ptr,
                                        u32 first_child_len,
                                        const u32* slots_ptr,
                                        u32 slots_len,
                                        const u32* alive_ptr,
                                        u32 alive_len,
                                        float* values_ptr,
                                        u32 values_len,
                                        u32 hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto values = view(values_ptr, values_len);

  const u32 h = hand_index();
  if (h >= hands) {
    return;
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    float& out = row(values, node, hands)[h];
    if (alive[slots[node]] == 0u) {
      out = 0.0f;
      continue;
    }
    const u32 child0 = first_child[node];
    float sum = 0.0f;
    for (const u32 a : indices(actions[node])) {
      sum = add_rn(sum, row(values, child0 + a, hands)[h]);
    }
    out = sum;
  }
}
