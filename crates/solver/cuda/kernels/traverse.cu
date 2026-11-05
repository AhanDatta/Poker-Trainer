// Level-synchronous traversal of the game tree for one traverser.
//
// CPU twins: `river::run` and `cfr::chance` (and `cfr::own_node_update` for
// the regret update). The host plan (src/gpu/plan.rs) lists the
// non-terminal nodes of every tree level; one launch handles one kind of
// node on one level:
//
// 1. `begin_traversal`: the root's opponent reach and the "alive" flags.
// 2. Top-down, per level: `opponent_split` (the opponent's strategy splits
//    its reach among the children; in CFR mode its strategy sums are
//    updated on the way) and `chance_split` (each dealt card's child gets
//    the reach times the chance factor, minus opponent hands holding the
//    card). A node whose opponent reach is all zero is dead: worth 0 and
//    never updated, exactly as on the CPU.
// 3. Terminals (terminal.cu).
// 4. Bottom-up, per level: `own_update` (node values and, in CFR mode, the
//    traverser's regret updates), `opponent_sum` and `chance_sum`. An
//    opponent node right below one of the traverser's nodes is "fused":
//    its row is never written, and `own_update` sums its children's rows
//    itself (`ChildValues`), saving the round trip through memory.
//
// Reach lives in "slots" of `opp_hands` floats; the traverser's own nodes
// pass their slot to their children unchanged, so only opponent and chance
// nodes write reach. Values live in rows of `hands` floats, node `i` in
// row `rows[i]`, and every node's row is written exactly once per pass over
// its segment (a tree too big for memory runs in segments that reuse rows
// and reach slots; see src/gpu/plan/schedule.rs).
//
// Threads: x covers hands, y walks the level's nodes (grid-stride), so a
// block always works on a single node and branches on node state are
// uniform across the block.

#include <solver/core.cuh>
#include <solver/strategy.cuh>

namespace solver {
namespace {

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
  const auto offsets = view(offsets_ptr, offsets_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto regrets_all = view(regrets_ptr, regrets_len);
  const auto sums_all = view(sums_ptr, sums_len);
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);
  const auto mode = static_cast<Mode>(mode_raw);
  const ChildValues child_value{actions, first_child, view(fused_ptr, fused_len), rows, values, hands};

  const u32 h = hand_index();
  if (h >= hands) {
    return;  // no block-wide synchronization in this kernel
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    float& out = row(values, rows[node], hands)[h];
    if (alive[slots[node]] == 0u) {
      out = 0.0f;
      continue;
    }
    const u32 num_actions = actions[node];
    const u32 child0 = first_child[node];
    const auto cfv = [&](u32 a) -> float { return child_value(child0 + a, h); };
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
                                        const u32* rows_ptr,
                                        u32 rows_len,
                                        float* values_ptr,
                                        u32 values_len,
                                        u32 hands) {
  using namespace solver;
  const auto level = view(level_ptr, level_len).subspan(level_start, level_count);
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);

  const u32 h = hand_index();
  if (h >= hands) {
    return;
  }
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    float& out = row(values, rows[node], hands)[h];
    if (alive[slots[node]] == 0u) {
      out = 0.0f;
      continue;
    }
    const u32 child0 = first_child[node];
    float sum = 0.0f;
    for (const u32 a : indices(actions[node])) {
      sum = add_rn(sum, row(values, rows[child0 + a], hands)[h]);
    }
    out = sum;
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
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto cards = view(cards_ptr, cards_len);
  const auto factors = view(factor_ptr, factor_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto child_slots = view(child_slots_ptr, child_slots_len);
  const auto opp_cards = view(opp_cards_ptr, opp_cards_len);
  const auto reach = view(reach_ptr, reach_len);
  const auto alive = view(alive_ptr, alive_len);

  const u32 h = hand_index();
  const bool valid = h < opp_hands;
  const CardPair hand = valid ? unpack_cards(opp_cards[h]) : CardPair{kNone, kNone};
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    const u32 slot = slots[node];
    if (alive[slot] == 0u) {
      continue;  // dead: no reach for the children
    }
    const u32 child0 = first_child[node];
    const u32 first_child_slot = child_slots[node];
    const float factor = factors[node];
    const float parent = valid ? row(reach, slot, opp_hands)[h] : 0.0f;
    for (const u32 a : indices(actions[node])) {
      float child = 0.0f;
      if (valid) {
        child = holds(hand, cards[child0 + a]) ? 0.0f : mul_rn(parent, factor);
        row(reach, first_child_slot + a, opp_hands)[h] = child;
      }
      // Every thread of the block reaches this (same node, same children).
      if (__syncthreads_or(child != 0.0f) != 0 && threadIdx.x == 0) {
        mark_alive(alive[first_child_slot + a]);
      }
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
  const auto actions = view(actions_ptr, actions_len);
  const auto first_child = view(first_child_ptr, first_child_len);
  const auto cards = view(cards_ptr, cards_len);
  const auto iso_start = view(iso_start_ptr, iso_start_len);
  const auto iso_count = view(iso_count_ptr, iso_count_len);
  const auto iso_child = view(iso_child_ptr, iso_child_len);
  const auto iso_pair = view(iso_pair_ptr, iso_pair_len);
  const auto perm = view(perm_ptr, perm_len);
  const auto my_cards = view(my_cards_ptr, my_cards_len);
  const auto slots = view(slots_ptr, slots_len);
  const auto alive = view(alive_ptr, alive_len);
  const auto rows = view(rows_ptr, rows_len);
  const auto values = view(values_ptr, values_len);

  const u32 h = hand_index();
  if (h >= hands) {
    return;  // no block-wide synchronization in this kernel
  }
  // Value of `child` for hand `j`: 0 if `j` holds the child's card.
  const auto child_value = [&](u32 child, u32 j) -> double {
    if (holds(unpack_cards(my_cards[j]), cards[child])) {
      return 0.0;
    }
    return static_cast<double>(row(values, rows[child], hands)[j]);
  };
  for (u32 k = blockIdx.y; k < level_count; k += gridDim.y) {
    const u32 node = level[k];
    float& out = row(values, rows[node], hands)[h];
    if (alive[slots[node]] == 0u) {
      out = 0.0f;
      continue;
    }
    const u32 child0 = first_child[node];
    double sum = 0.0;
    for (const u32 a : indices(actions[node])) {
      sum = add_rn(sum, child_value(child0 + a, h));
    }
    const auto isomorphic = iso_child.subspan(iso_start[node], iso_count[node]);
    const auto pairs = iso_pair.subspan(iso_start[node], iso_count[node]);
    for (const u32 e : indices(static_cast<u32>(isomorphic.size()))) {
      const u32 j = row(perm, pairs[e], hands)[h];
      sum = add_rn(sum, child_value(child0 + isomorphic[e], j));
    }
    out = to_f32(sum);
  }
}
