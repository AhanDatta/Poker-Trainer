// Level-synchronous three-player traversal for one traverser.
//
// CPU twin: `multi::cfr::traverse` (opponent_node, chance, own_node). The
// host plan (src/gpu/multi/plan.rs) lists every tree level's expanded nodes
// by kind; one launch handles one kind on one level, with the per-node
// steps of traversal.cuh (shared with the heads-up kernels):
//
// 1. Begin: each opponent's root reach in its slot 0 and its alive flags
//    (`begin_traversal` of traverse.cu, once per opponent).
// 2. Top-down, per level: `multi_opponent_split` for each opponent's nodes
//    (only the acting opponent's reach splits; the other passes through
//    its slot) and `multi_chance_split` once per opponent (the chance
//    factor on the first opponent only, as the CPU applies it).
// 3. Leaves (multi_terminal.cu).
// 4. Bottom-up, per level: `multi_own_update`, `multi_opponent_sum` and
//    `multi_chance_sum`.
//
// Each opponent has its own reach slots and alive flags; a node is alive
// when both of its slots are (`TwoOpponents`). A player's average strategy
// is accumulated in the traversal of the next player still in the hand
// (`strategy_update` flags), not in every traversal that sees the node.

#include <solver/traversal.cuh>

using solver::u32;

namespace solver {
namespace {

/// The two opponents' slots and alive flags from kernel argument pairs.
[[nodiscard]] __device__ auto two_opponents(const u32* first_slots_ptr,
                                            u32 first_slots_len,
                                            const u32* first_alive_ptr,
                                            u32 first_alive_len,
                                            const u32* second_slots_ptr,
                                            u32 second_slots_len,
                                            const u32* second_alive_ptr,
                                            u32 second_alive_len) noexcept -> TwoOpponents {
  return {view(first_slots_ptr, first_slots_len), view(first_alive_ptr, first_alive_len),
          view(second_slots_ptr, second_slots_len), view(second_alive_ptr, second_alive_len)};
}

}  // namespace
}  // namespace solver

/// Top-down step at one opponent's decision nodes of one level: its
/// strategy splits its reach (`slots`, `child_slots`, `reach`, `alive`); the
/// other opponent's slots and flags only decide liveness. In CFR mode the
/// node's strategy sums accumulate where `strategy_update[node]` is set.
extern "C" __global__ void multi_opponent_split(const u32* level_ptr,
                                                u32 level_len,
                                                u32 level_start,
                                                u32 level_count,
                                                const u32* actions_ptr,
                                                u32 actions_len,
                                                const u32* offsets_ptr,
                                                u32 offsets_len,
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
                                                const float* regrets_ptr,
                                                u32 regrets_len,
                                                float* sums_ptr,
                                                u32 sums_len,
                                                u32 opp_hands,
                                                u32 mode_raw,
                                                float strategy_discount,
                                                u32 reach_weighted) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto strategy_update = view(strategy_update_ptr, strategy_update_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto live = two_opponents(slots_ptr, slots_len, alive_ptr, alive_len, other_slots_ptr, other_slots_len,
                                  other_alive_ptr, other_alive_len);
  const OpponentSplit<float> split{
      .actions = view(actions_ptr, actions_len),
      .offsets = view(offsets_ptr, offsets_len),
      .regrets = view(regrets_ptr, regrets_len),
      .sums = view(sums_ptr, sums_len),
      .opp = {slots, view(child_slots_ptr, child_slots_len), view(reach_ptr, reach_len), alive, opp_hands},
      .mode = static_cast<Mode>(mode_raw),
      .strategy_discount = strategy_discount,
      .reach_weighted = reach_weighted != 0u,
  };
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    if (live(node)) {  // dead: no reach for the children, no strategy-sum update
      split(node, strategy_update[node] != 0u);
    }
  }
}

/// Top-down step at the chance nodes of one level for one opponent: child
/// `a` gets the opponent's reach (times the chance factor with
/// `apply_factor`, the first opponent's split), 0 for its hands holding
/// the dealt card.
extern "C" __global__ void multi_chance_split(const u32* level_ptr,
                                              u32 level_len,
                                              u32 level_start,
                                              u32 level_count,
                                              const u32* actions_ptr,
                                              u32 actions_len,
                                              const u32* first_child_ptr,
                                              u32 first_child_len,
                                              const u32* cards_ptr,
                                              u32 cards_len,
                                              const float* factor_ptr,
                                              u32 factor_len,
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
                                              const u32* opp_cards_ptr,
                                              u32 opp_cards_len,
                                              u32 opp_hands,
                                              u32 apply_factor) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto live = two_opponents(slots_ptr, slots_len, alive_ptr, alive_len, other_slots_ptr, other_slots_len,
                                  other_alive_ptr, other_alive_len);
  const ChanceSplit split{
      .actions = view(actions_ptr, actions_len),
      .first_child = view(first_child_ptr, first_child_len),
      .cards = view(cards_ptr, cards_len),
      .factors = view(factor_ptr, factor_len),
      .opp_cards = view(opp_cards_ptr, opp_cards_len),
      .opp = {slots, view(child_slots_ptr, child_slots_len), view(reach_ptr, reach_len), alive, opp_hands},
  };
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    if (live(node)) {  // dead: no reach for the children
      split(node, apply_factor != 0u);
    }
  }
}

/// Bottom-up step at the traverser's decision nodes of one level (CPU:
/// `multi::cfr::own_node_combine`).
extern "C" __global__ void multi_own_update(const u32* level_ptr,
                                            u32 level_len,
                                            u32 level_start,
                                            u32 level_count,
                                            const u32* actions_ptr,
                                            u32 actions_len,
                                            const u32* offsets_ptr,
                                            u32 offsets_len,
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
                                            float* regrets_ptr,
                                            u32 regrets_len,
                                            const float* sums_ptr,
                                            u32 sums_len,
                                            const u32* rows_ptr,
                                            u32 rows_len,
                                            float* values_ptr,
                                            u32 values_len,
                                            u32 hands,
                                            u32 mode_raw,
                                            float positive_discount,
                                            float negative_discount) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto live = two_opponents(first_slots_ptr, first_slots_len, first_alive_ptr, first_alive_len,
                                  second_slots_ptr, second_slots_len, second_alive_ptr, second_alive_len);
  const ChildRows children{view(rows_ptr, rows_len), view(values_ptr, values_len), hands};
  const OwnUpdate<ChildRows> update{
      .actions = view(actions_ptr, actions_len),
      .offsets = view(offsets_ptr, offsets_len),
      .first_child = view(first_child_ptr, first_child_len),
      .regrets = view(regrets_ptr, regrets_len),
      .sums = view(sums_ptr, sums_len),
      .child_value = children,
      .hands = hands,
      .mode = static_cast<Mode>(mode_raw),
      .positive_discount = positive_discount,
      .negative_discount = negative_discount,
  };
  const u32 h = hand_index();
  if (h >= hands) {
    return;  // no block-wide synchronization in this kernel
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    row(children.values, children.rows[node], hands)[h] = live(node) ? update(node, h) : 0.0f;
  }
}

/// Bottom-up step at an opponent's decision nodes of one level: the sum of
/// the children's values (CPU: `multi::cfr::opponent_node`, which skips
/// children nobody reaches; their rows hold 0 here, which adds nothing).
extern "C" __global__ void multi_opponent_sum(const u32* level_ptr,
                                              u32 level_len,
                                              u32 level_start,
                                              u32 level_count,
                                              const u32* actions_ptr,
                                              u32 actions_len,
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
                                              const u32* rows_ptr,
                                              u32 rows_len,
                                              float* values_ptr,
                                              u32 values_len,
                                              u32 hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto live = two_opponents(first_slots_ptr, first_slots_len, first_alive_ptr, first_alive_len,
                                  second_slots_ptr, second_slots_len, second_alive_ptr, second_alive_len);
  const ChildRows children{view(rows_ptr, rows_len), view(values_ptr, values_len), hands};
  const u32 h = hand_index();
  if (h >= hands) {
    return;
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    row(children.values, children.rows[node], hands)[h] =
        live(node) ? opponent_value(actions, first_child, children, node, h) : 0.0f;
  }
}

/// Bottom-up step at the chance nodes of one level (CPU: `multi::cfr::
/// chance`): the f64 sum over dealt cards of the children's values, the
/// traverser's hands holding the card counting 0.
extern "C" __global__ void multi_chance_sum(const u32* level_ptr,
                                            u32 level_len,
                                            u32 level_start,
                                            u32 level_count,
                                            const u32* actions_ptr,
                                            u32 actions_len,
                                            const u32* first_child_ptr,
                                            u32 first_child_len,
                                            const u32* cards_ptr,
                                            u32 cards_len,
                                            const u32* my_cards_ptr,
                                            u32 my_cards_len,
                                            const u32* first_slots_ptr,
                                            u32 first_slots_len,
                                            const u32* first_alive_ptr,
                                            u32 first_alive_len,
                                            const u32* second_slots_ptr,
                                            u32 second_slots_len,
                                            const u32* second_alive_ptr,
                                            u32 second_alive_len,
                                            const u32* rows_ptr,
                                            u32 rows_len,
                                            float* values_ptr,
                                            u32 values_len,
                                            u32 hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto live = two_opponents(first_slots_ptr, first_slots_len, first_alive_ptr, first_alive_len,
                                  second_slots_ptr, second_slots_len, second_alive_ptr, second_alive_len);
  const ChanceSum sum{
      .actions = view(actions_ptr, actions_len),
      .first_child = view(first_child_ptr, first_child_len),
      .cards = view(cards_ptr, cards_len),
      .perm = Span<const u32>{},
      .my_cards = view(my_cards_ptr, my_cards_len),
      .children = {view(rows_ptr, rows_len), view(values_ptr, values_len), hands},
  };
  const u32 h = hand_index();
  if (h >= hands) {
    return;  // no block-wide synchronization in this kernel
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    float& out = row(sum.children.values, sum.children.rows[node], hands)[h];
    out = live(node) ? sum(node, h, Span<const u32>{}, Span<const u32>{}) : 0.0f;
  }
}
