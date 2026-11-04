// Per-hand strategy computations shared by the kernels. Each mirrors a CPU
// function operation for operation (same order, no fused multiply-adds), so
// GPU and CPU agree bit for bit on identical inputs.
#pragma once

#include <solver/core.cuh>

namespace solver {

/// Sum over actions of `max(regret, 0)` for hand `h`, in action order
/// (CPU: `cfr::positive_sums`).
template <RegretValue R>
[[nodiscard]] __device__ auto positive_sum(Block<const R> regrets, u32 h) noexcept -> float {
  float sum = 0.0f;
  for (const u32 a : indices(regrets.extent(0))) {
    sum = add_rn(sum, positive_part(regrets(a, h)));
  }
  return sum;
}

/// Regret-matching probability of action `a` for hand `h`, given
/// `inv = recip(positive_sum(regrets, h))`: proportional to the positive
/// regret, uniform when nothing is positive (CPU: `cfr::regret_match_into`).
template <RegretValue R>
[[nodiscard]] __device__ auto matched(Block<const R> regrets, u32 a, u32 h, float inv) noexcept -> float {
  if (inv == 0.0f) {
    return 1.0f / static_cast<float>(regrets.extent(0));
  }
  return mul_rn(positive_part(regrets(a, h)), inv);
}

/// Sum over actions of the strategy sums of hand `h`, starting from the
/// first action's (CPU: `cfr::normalize_in_place`).
[[nodiscard]] __device__ inline auto strategy_total(Block<const float> sums, u32 h) noexcept -> float {
  float total = sums(0, h);
  for (u32 a = 1; a < sums.extent(0); ++a) {
    total = add_rn(total, sums(a, h));
  }
  return total;
}

/// Average-strategy probability of action `a` for hand `h`, given
/// `inv = recip(strategy_total(sums, h))`; uniform when the sums are zero
/// (CPU: `cfr::normalize_in_place`).
[[nodiscard]] __device__ inline auto normalized(Block<const float> sums, u32 a, u32 h, float inv) noexcept -> float {
  if (inv == 0.0f) {
    return 1.0f / static_cast<float>(sums.extent(0));
  }
  return mul_rn(sums(a, h), inv);
}

}  // namespace solver
