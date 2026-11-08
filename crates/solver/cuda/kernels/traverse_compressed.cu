// The two heads-up traversal steps that update storage, for 16-bit
// ("compressed") storage: `opponent_split_compressed` (top-down, opponent's
// strategy sums) and `own_update_compressed` (bottom-up, traverser's
// regrets), one block per node, over the shared steps of
// compressed_steps.cuh. The other steps don't touch storage and are shared
// with f32 storage (traverse.cu).
#include <solver/compressed_steps.cuh>

using solver::u16;
using solver::u32;

/// Top-down step at the opponent's decision nodes of one level, with 16-bit
/// storage. One block per node. The current strategy comes straight from
/// the i16 regrets (the scale cancels out in regret matching, as on the
/// CPU); the u16 strategy sums are decoded, updated and re-encoded.
extern "C" __global__ void __launch_bounds__(solver::kCompressedThreads)
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
  __shared__ CompressedShared sh;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
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
    if (alive[slots[node]] == 0u) {
      continue;  // dead: no reach for the children, no strategy-sum update
    }
    split(node, mode == Mode::Cfr, sh);
  }
}

/// Bottom-up step at the traverser's decision nodes of one level, with
/// 16-bit storage. One block per node. In CFR mode the i16 regrets are
/// decoded, updated (CPU: `cfr::own_node_update`) and re-encoded.
extern "C" __global__ void __launch_bounds__(solver::kCompressedThreads)
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
                          const u32* fused_ptr,
                          u32 fused_len,
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
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);
  const CompressedOwnUpdate<ChildValues> update{
      .actions = actions,
      .offsets = view(offsets_ptr, offsets_len),
      .scale_slots = view(scale_slots_ptr, scale_slots_len),
      .first_child = first_child,
      .regrets = view(regrets_ptr, regrets_len),
      .regret_scale = view(regret_scale_ptr, regret_scale_len),
      .sums = view(sums_ptr, sums_len),
      .strategy_scale = view(strategy_scale_ptr, strategy_scale_len),
      .child_value = ChildValues{actions, first_child, view(fused_ptr, fused_len), rows, values, hands},
      .hands = hands,
      .mode = static_cast<Mode>(mode_raw),
      .positive_discount = positive_discount,
      .negative_discount = negative_discount,
  };
  for (u32 k = blockIdx.x; k < level_count; k += gridDim.x) {
    const u32 node = level[k];
    const auto out = row(values, rows[node], hands);
    if (alive[slots[node]] == 0u) {
      for (u32 h = threadIdx.x; h < hands; h += kCompressedThreads) {
        out[h] = 0.0f;
      }
      continue;  // uniform: the whole block skips the node
    }
    update(node, out, sh);
  }
}
