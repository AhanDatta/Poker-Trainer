// The two three-player traversal steps that update storage, for 16-bit
// ("compressed") storage, one block per node over the shared steps of
// compressed_steps.cuh: `multi_opponent_split_compressed` (top-down, the
// acting opponent's strategy sums, where `strategy_update` marks the
// node) and `multi_own_update_compressed` (bottom-up, the traverser's
// regrets). The other steps don't touch storage and are the f32 path's
// (multi_traverse.cu). A node is alive when both opponents' slots are.

#include <solver/compressed_steps.cuh>

using solver::u16;
using solver::u32;

/// Top-down step at one opponent's decision nodes of one level, with 16-bit
/// storage (`multi_opponent_split`'s arguments, storage as i16 regrets and
/// u16 sums with their per-node scales).
extern "C" __global__ void __launch_bounds__(solver::kCompressedThreads)
    multi_opponent_split_compressed(const u32* level_ptr,
                                    u32 level_len,
                                    u32 level_start,
                                    u32 level_count,
                                    const u32* actions_ptr,
                                    u32 actions_len,
                                    const u32* offsets_ptr,
                                    u32 offsets_len,
                                    const u32* scale_slots_ptr,
                                    u32 scale_slots_len,
                                    const u32* strategy_update_ptr,
                                    u32 strategy_update_len,
                                    const u32* slots_ptr,
                                    u32 slots_len,
                                    const u32* child_slots_ptr,
                                    u32 child_slots_len,
                                    float* reach_ptr,
                                    u32 reach_len,
                                    u32* alive_ptr,
                                    u32 alive_len,
                                    const u32* other_slots_ptr,
                                    u32 other_slots_len,
                                    const u32* other_alive_ptr,
                                    u32 other_alive_len,
                                    const solver::i16* regrets_ptr,
                                    u32 regrets_len,
                                    u16* sums_ptr,
                                    u32 sums_len,
                                    float* strategy_scale_ptr,
                                    u32 strategy_scale_len,
                                    u32 opp_hands,
                                    u32 mode_raw,
                                    float strategy_discount,
                                    u32 reach_weighted) {
  using namespace solver;
  __shared__ CompressedShared sh;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto strategy_update = view(strategy_update_ptr, strategy_update_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const TwoOpponents live{slots, alive, view(other_slots_ptr, other_slots_len), view(other_alive_ptr, other_alive_len)};
  const auto mode = static_cast<Mode>(mode_raw);
  const CompressedSplit split{
      .actions = view(actions_ptr, actions_len),
      .offsets = view(offsets_ptr, offsets_len),
      .scale_slots = view(scale_slots_ptr, scale_slots_len),
      .slots = slots,
      .child_slots = view(child_slots_ptr, child_slots_len),
      .regrets = view(regrets_ptr, regrets_len),
      .sums = view(sums_ptr, sums_len),
      .strategy_scale = view(strategy_scale_ptr, strategy_scale_len),
      .reach = view(reach_ptr, reach_len),
      .alive = alive,
      .opp_hands = opp_hands,
      .mode = mode,
      .strategy_discount = strategy_discount,
      .weighted = reach_weighted != 0u,
  };
  for (u32 k = blockIdx.x; k < level_count; k += gridDim.x) {
    const u32 node = level[k];
    if (live(node)) {  // dead: no reach for the children, no strategy-sum update
      split(node, mode == Mode::Cfr && strategy_update[node] != 0u, sh);
    }
  }
}

/// Bottom-up step at the traverser's decision nodes of one level, with
/// 16-bit storage (`multi_own_update`'s arguments, storage as i16 regrets
/// and u16 sums with their per-node scales).
extern "C" __global__ void __launch_bounds__(solver::kCompressedThreads)
    multi_own_update_compressed(const u32* level_ptr,
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
                                const u32* first_slots_ptr,
                                u32 first_slots_len,
                                const u32* first_alive_ptr,
                                u32 first_alive_len,
                                const u32* second_slots_ptr,
                                u32 second_slots_len,
                                const u32* second_alive_ptr,
                                u32 second_alive_len,
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
  __shared__ CompressedShared sh;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const TwoOpponents live{view(first_slots_ptr, first_slots_len), view(first_alive_ptr, first_alive_len),
                          view(second_slots_ptr, second_slots_len), view(second_alive_ptr, second_alive_len)};
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);
  const CompressedOwnUpdate<ChildRows> update{
      .actions = view(actions_ptr, actions_len),
      .offsets = view(offsets_ptr, offsets_len),
      .scale_slots = view(scale_slots_ptr, scale_slots_len),
      .first_child = view(first_child_ptr, first_child_len),
      .regrets = view(regrets_ptr, regrets_len),
      .regret_scale = view(regret_scale_ptr, regret_scale_len),
      .sums = view(sums_ptr, sums_len),
      .strategy_scale = view(strategy_scale_ptr, strategy_scale_len),
      .child_value = ChildRows{rows, values, hands},
      .hands = hands,
      .mode = static_cast<Mode>(mode_raw),
      .positive_discount = positive_discount,
      .negative_discount = negative_discount,
  };
  for (u32 k = blockIdx.x; k < level_count; k += gridDim.x) {
    const u32 node = level[k];
    const auto out = row(values, rows[node], hands);
    if (!live(node)) {
      for (u32 h = threadIdx.x; h < hands; h += kCompressedThreads) {
        out[h] = 0.0f;
      }
      continue;  // uniform: the whole block skips the node
    }
    update(node, out, sh);
  }
}
