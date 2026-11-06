// Level-synchronous heads-up traversal for one traverser.
//
// CPU twins: `river::run` and `cfr::chance` (and `cfr::own_node_update` for
// the regret update). The host plan (src/gpu/plan.rs) lists the
// non-terminal nodes of every tree level; one launch handles one kind of
// node on one level (the steps themselves are in traversal.cuh, shared
// with the three-player kernels):
//
// 1. `begin_traversal`: the root's opponent reach and the "alive" flags.
// 2. Top-down, per level: `opponent_split` and `chance_split`. A node whose
//    opponent reach is all zero is dead: worth 0 and never updated, exactly
//    as on the CPU.
// 3. Terminals (terminal.cu).
// 4. Bottom-up, per level: `own_update`, `opponent_sum` and `chance_sum`.
//    An opponent node right below one of the traverser's nodes is "fused":
//    its row is never written, and `own_update` sums its children's rows
//    itself (`ChildValues`), saving the round trip through memory.
//
// The traverser's own nodes pass their reach slot to their children
// unchanged, so only opponent and chance nodes write reach. Every node's
// value row is written exactly once per pass over its segment (a tree too
// big for memory runs in segments that reuse rows and reach slots; see
// src/gpu/plan/schedule.rs).

#include <solver/traversal.cuh>

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

/// Clears the alive flags of slots `[start, start + count)` (a group's
/// reach slots, before the group runs).
extern "C" __global__ void reset_alive(u32* alive_ptr, u32 alive_len, u32 start, u32 count) {
  const auto alive = solver::view(alive_ptr, alive_len).subspan(start, count);
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    alive[i] = 0u;
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
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const OneOpponent live{slots, alive};
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
      split(node, true);
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
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);
  const OneOpponent live{view(slots_ptr, slots_len), view(alive_ptr, alive_len)};
  const OwnUpdate<ChildValues> update{
      .actions = actions,
      .offsets = view(offsets_ptr, offsets_len),
      .first_child = first_child,
      .regrets = view(regrets_ptr, regrets_len),
      .sums = view(sums_ptr, sums_len),
      .child_value = {actions, first_child, view(fused_ptr, fused_len), rows, values, hands},
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
    row(values, rows[node], hands)[h] = live(node) ? update(node, h) : 0.0f;
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
                                        const u32* rows_ptr,
                                        u32 rows_len,
                                        float* values_ptr,
                                        u32 values_len,
                                        u32 hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const OneOpponent live{view(slots_ptr, slots_len), view(alive_ptr, alive_len)};
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

/// Top-down step at the chance nodes of one level (CPU: `cfr::chance`):
/// child `a` (dealt card `card[child]`) gets the opponent's reach times the
/// chance factor, and 0 for opponent hands holding that card.
extern "C" __global__ void chance_split(const u32* level_ptr,
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
                                        const u32* opp_cards_ptr,
                                        u32 opp_cards_len,
                                        float* reach_ptr,
                                        u32 reach_len,
                                        u32* alive_ptr,
                                        u32 alive_len,
                                        u32 opp_hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const OneOpponent live{slots, alive};
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
      split(node, true);
    }
  }
}

/// Bottom-up step at the chance nodes of one level (CPU: `cfr::chance`):
/// the sum over dealt cards of the children's values, with the traverser's
/// hands holding the card counting 0, plus each suit-isomorphic card's
/// value read from its canonical child through the suit-swap permutation.
/// Accumulated in f64 in the CPU's order.
extern "C" __global__ void chance_sum(const u32* level_ptr,
                                      u32 level_len,
                                      u32 level_start,
                                      u32 level_count,
                                      const u32* actions_ptr,
                                      u32 actions_len,
                                      const u32* first_child_ptr,
                                      u32 first_child_len,
                                      const u32* cards_ptr,
                                      u32 cards_len,
                                      const u32* iso_start_ptr,
                                      u32 iso_start_len,
                                      const u32* iso_count_ptr,
                                      u32 iso_count_len,
                                      const u32* iso_child_ptr,
                                      u32 iso_child_len,
                                      const u32* iso_pair_ptr,
                                      u32 iso_pair_len,
                                      const u32* perm_ptr,
                                      u32 perm_len,
                                      const u32* my_cards_ptr,
                                      u32 my_cards_len,
                                      const u32* slots_ptr,
                                      u32 slots_len,
                                      const u32* alive_ptr,
                                      u32 alive_len,
                                      const u32* rows_ptr,
                                      u32 rows_len,
                                      float* values_ptr,
                                      u32 values_len,
                                      u32 hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto iso_start = view(iso_start_ptr, iso_start_len);
  const auto iso_count = view(iso_count_ptr, iso_count_len);
  const auto iso_child = view(iso_child_ptr, iso_child_len);
  const auto iso_pair = view(iso_pair_ptr, iso_pair_len);
  const OneOpponent live{view(slots_ptr, slots_len), view(alive_ptr, alive_len)};
  const ChanceSum sum{
      .actions = view(actions_ptr, actions_len),
      .first_child = view(first_child_ptr, first_child_len),
      .cards = view(cards_ptr, cards_len),
      .perm = view(perm_ptr, perm_len),
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
    if (!live(node)) {
      out = 0.0f;
      continue;
    }
    const auto isomorphic = iso_child.subspan(iso_start[node], iso_count[node]);
    const auto pairs = iso_pair.subspan(iso_start[node], iso_count[node]);
    out = sum(node, h, isomorphic, pairs);
  }
}
