// Per-node steps of a level-synchronous traversal, shared by the heads-up
// kernels (traverse.cu) and the three-player kernels (multi_traverse.cu).
//
// A kernel launch handles one kind of node on one tree level: x covers
// hands, y walks the level's nodes (grid-stride), so a block always works
// on a single node and branches on node state are uniform across the block.
// Reach lives in "slots" (one row of an opponent's hands each), values in
// rows of the traverser's hands; the host plans lay both out and validate
// every index (src/gpu/plan.rs, src/gpu/multi/plan.rs).
//
// Each step mirrors its CPU twin operation for operation (`*_rn`
// arithmetic, the CPU's order), so results are bit-identical:
// * opponent split: `multi::cfr::opponent_node` / `river::run`'s top-down
//   pass: the acting opponent's strategy splits its reach among the
//   children, and in CFR mode its strategy sums accumulate;
// * chance split: `cfr::chance`: each dealt card's child gets the reach
//   (times the chance factor where the CPU applies it), 0 for hands
//   holding the card;
// * own update: `cfr::own_node_update`: node values from the children's,
//   and the traverser's regret update in CFR mode;
// * opponent sum and chance sum: the children's values added up.
//
// Which nodes are alive (reachable by every opponent) is a property of the
// traversal kind: one reach slot per node heads-up (`OneOpponent`), one per
// opponent with three players (`TwoOpponents`).
#pragma once

#include <solver/core.cuh>
#include <solver/strategy.cuh>

namespace solver {

/// This thread's hand (x covers hands).
[[nodiscard]] __device__ inline auto hand_index() noexcept -> u32 { return blockIdx.x * blockDim.x + threadIdx.x; }

/// Heads-up liveness: a node is alive when its opponent slot is.
struct OneOpponent {
  Span<const u32> slots;
  Span<const u32> alive;

  [[nodiscard]] __device__ auto operator()(u32 node) const noexcept -> bool { return alive[slots[node]] != 0u; }
};

/// Three-player liveness: a node is alive when both opponents' slots are
/// (CPU: a node either opponent can't reach is worth 0 and isn't updated).
struct TwoOpponents {
  Span<const u32> first_slots;
  Span<const u32> first_alive;
  Span<const u32> second_slots;
  Span<const u32> second_alive;

  [[nodiscard]] __device__ auto operator()(u32 node) const noexcept -> bool {
    return first_alive[first_slots[node]] != 0u && second_alive[second_slots[node]] != 0u;
  }
};

/// Whether a node is alive (`OneOpponent`, `TwoOpponents`).
template <class L>
concept Liveness = requires(const L l, u32 node) {
  { l(node) } -> cuda::std::convertible_to<bool>;
};

/// The acting opponent's view of the reach at its nodes: where its reach
/// is, where its children's go, and their alive flags.
struct OpponentReach {
  Span<const u32> slots;
  Span<const u32> child_slots;
  Span<float> reach;
  Span<u32> alive;
  u32 hands;
};

/// Marks child slot `slot` alive if any thread of the block holds a nonzero
/// reach for it. Every thread of the block must call it (the node, and so
/// the slot, is the same for the whole block).
__device__ inline void mark_if_reached(Span<u32> alive, u32 slot, float child) noexcept {
  if (__syncthreads_or(child != 0.0f) != 0 && threadIdx.x == 0) {
    mark_alive(alive[slot]);
  }
}

/// Top-down step at one opponent node: its strategy (regret matching in CFR
/// mode, the normalized average otherwise) splits the opponent's reach
/// among the children. With `accumulate` (CFR mode only), the current
/// strategy is added to the strategy sums (CPU: `x * g + y`, or `x * g + y
/// * w` with reach weighting). Every thread of the block must call it.
template <RegretValue R>
struct OpponentSplit {
  Span<const u32> actions;
  Span<const u32> offsets;
  Span<const R> regrets;
  Span<float> sums;
  OpponentReach opp;
  Mode mode;
  float strategy_discount;
  bool reach_weighted;

  __device__ void operator()(u32 node, bool accumulate) const noexcept {
    const u32 h = hand_index();
    const bool valid = h < opp.hands;
    const u32 num_actions = actions[node];
    const u32 first_child_slot = opp.child_slots[node];
    const Block<const R> node_regrets = block_at(regrets, offsets[node], num_actions, opp.hands);
    const Block<float> node_sums = block_at(sums, offsets[node], num_actions, opp.hands);
    const float parent = valid ? row(opp.reach, opp.slots[node], opp.hands)[h] : 0.0f;
    float inv = 0.0f;
    if (valid) {
      inv = mode == Mode::Cfr ? recip(positive_sum(node_regrets, h)) : recip(strategy_total(node_sums, h));
    }
    for (const u32 a : indices(num_actions)) {
      float child = 0.0f;
      if (valid) {
        float p = 0.0f;
        if (mode == Mode::Cfr) {
          p = matched(node_regrets, a, h, inv);
          if (accumulate) {
            const float kept = mul_rn(node_sums(a, h), strategy_discount);
            node_sums(a, h) = add_rn(kept, reach_weighted ? mul_rn(p, parent) : p);
          }
        } else {
          p = normalized(node_sums, a, h, inv);
        }
        child = mul_rn(parent, p);
        row(opp.reach, first_child_slot + a, opp.hands)[h] = child;
      }
      mark_if_reached(opp.alive, first_child_slot + a, child);
    }
  }
};

/// Top-down step at one chance node for one opponent: child `a` (dealt card
/// `cards[child]`) gets the reach (times the chance factor with
/// `apply_factor`), 0 for the opponent's hands holding that card. Every
/// thread of the block must call it.
struct ChanceSplit {
  Span<const u32> actions;
  Span<const u32> first_child;
  Span<const u32> cards;
  Span<const float> factors;
  Span<const u32> opp_cards;
  OpponentReach opp;

  __device__ void operator()(u32 node, bool apply_factor) const noexcept {
    const u32 h = hand_index();
    const bool valid = h < opp.hands;
    const CardPair hand = valid ? unpack_cards(opp_cards[h]) : CardPair{kNone, kNone};
    const u32 child0 = first_child[node];
    const u32 first_child_slot = opp.child_slots[node];
    const float parent = valid ? row(opp.reach, opp.slots[node], opp.hands)[h] : 0.0f;
    const float reach = apply_factor ? mul_rn(parent, factors[node]) : parent;
    for (const u32 a : indices(actions[node])) {
      float child = 0.0f;
      if (valid) {
        child = holds(hand, cards[child0 + a]) ? 0.0f : reach;
        row(opp.reach, first_child_slot + a, opp.hands)[h] = child;
      }
      mark_if_reached(opp.alive, first_child_slot + a, child);
    }
  }
};

/// A child's value row (no fused children: the three-player plan has none).
struct ChildRows {
  Span<const u32> rows;
  Span<float> values;
  u32 hands;

  [[nodiscard]] __device__ auto operator()(u32 child, u32 h) const noexcept -> float {
    return row(values, rows[child], hands)[h];
  }
};

/// How the bottom-up steps read a child's value for hand `h` (`ChildRows`,
/// or `ChildValues` with fused opponent nodes).
template <class C>
concept ChildValue = requires(const C c, u32 child, u32 h) {
  { c(child, h) } -> cuda::std::convertible_to<float>;
};

/// Bottom-up step at one of the traverser's nodes for hand `h`: the node's
/// value from its children's, and in CFR mode the regret update (CPU:
/// `cfr::own_node_update`).
template <ChildValue C>
struct OwnUpdate {
  Span<const u32> actions;
  Span<const u32> offsets;
  Span<const u32> first_child;
  Span<float> regrets;
  Span<const float> sums;
  C child_value;
  u32 hands;
  Mode mode;
  float positive_discount;
  float negative_discount;

  [[nodiscard]] __device__ auto operator()(u32 node, u32 h) const noexcept -> float {
    const u32 num_actions = actions[node];
    const u32 child0 = first_child[node];
    const auto cfv = [&](u32 a) -> float { return child_value(child0 + a, h); };
    switch (mode) {
      case Mode::BestResponse: {
        float best = cfv(0);
        for (u32 a = 1; a < num_actions; ++a) {
          best = fmaxf(best, cfv(a));
        }
        return best;
      }
      case Mode::Average: {
        const Block<const float> node_sums = block_at(sums, offsets[node], num_actions, hands);
        const float inv = recip(strategy_total(node_sums, h));
        float value = 0.0f;
        for (const u32 a : indices(num_actions)) {
          value = add_rn(value, mul_rn(normalized(node_sums, a, h, inv), cfv(a)));
        }
        return value;
      }
      case Mode::Cfr:
        break;
    }
    const Block<float> node_regrets = block_at(regrets, offsets[node], num_actions, hands);
    // Value under regret matching without materializing the strategy:
    // sum(max(r, 0) * cfv) / sum(max(r, 0)), or the mean over actions when
    // no regret is positive.
    float positive = 0.0f;
    float weighted = 0.0f;
    for (const u32 a : indices(num_actions)) {
      const float r = positive_part(node_regrets(a, h));
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
      const float r = node_regrets(a, h);
      const float discount = r > 0.0f ? positive_discount : negative_discount;
      node_regrets(a, h) = sub_rn(add_rn(mul_rn(r, discount), cfv(a)), value);
    }
    return value;
  }
};

/// Bottom-up step at one opponent node for hand `h`: the sum of the
/// children's values in action order (their reach already carries the
/// opponent's strategy).
[[nodiscard]] __device__ inline auto opponent_value(Span<const u32> actions,
                                                    Span<const u32> first_child,
                                                    const ChildRows& children,
                                                    u32 node,
                                                    u32 h) noexcept -> float {
  const u32 child0 = first_child[node];
  float sum = 0.0f;
  for (const u32 a : indices(actions[node])) {
    sum = add_rn(sum, children(child0 + a, h));
  }
  return sum;
}

/// Bottom-up step at one chance node for hand `h` (CPU: `cfr::chance`): the
/// f64 sum over dealt cards of the children's values, the traverser's
/// hands holding the card counting 0, then each suit-isomorphic card's
/// value read from its canonical child through the suit-swap permutation
/// (`isomorphic`, `pairs`: empty without isomorphism).
struct ChanceSum {
  Span<const u32> actions;
  Span<const u32> first_child;
  Span<const u32> cards;
  Span<const u32> perm;
  Span<const u32> my_cards;
  ChildRows children;

  [[nodiscard]] __device__ auto operator()(u32 node, u32 h, Span<const u32> isomorphic, Span<const u32> pairs) const
      noexcept -> float {
    // Value of `child` for hand `j`: 0 if `j` holds the child's card.
    const auto child_value = [&](u32 child, u32 j) -> double {
      if (holds(unpack_cards(my_cards[j]), cards[child])) {
        return 0.0;
      }
      return static_cast<double>(children(child, j));
    };
    const u32 child0 = first_child[node];
    double sum = 0.0;
    for (const u32 a : indices(actions[node])) {
      sum = add_rn(sum, child_value(child0 + a, h));
    }
    for (const u32 e : indices(static_cast<u32>(isomorphic.size()))) {
      const u32 j = row(perm, pairs[e], children.hands)[h];
      sum = add_rn(sum, child_value(child0 + isomorphic[e], j));
    }
    return to_f32(sum);
  }
};

}  // namespace solver
