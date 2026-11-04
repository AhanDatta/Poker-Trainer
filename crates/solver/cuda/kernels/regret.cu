// Regret matching: the current strategy of a decision node from its
// cumulative regrets.
//
// CPU twin: `cfr::regret_match_into`. Same IEEE operations in the same order
// (positive parts summed in action order, one guarded reciprocal, one
// multiply per action), so results are identical to the CPU's.

#include <solver/core.cuh>

#include <cuda/std/ranges>

namespace solver {
namespace {

/// Strategy of hand `h`: proportional to its positive regrets, uniform when
/// none is positive (or their sum is too small to invert).
template <RegretValue R>
__device__ void regret_match_hand(Block<const R> regrets, Block<float> strategy, u32 h) noexcept {
  const u32 actions = regrets.extent(0);
  float sum = 0.0f;
  for (const u32 a : cuda::std::views::iota(u32{0}, actions)) {
    sum += positive_part(regrets(a, h));
  }
  const float inv = recip(sum);
  if (inv == 0.0f) {
    const float uniform = 1.0f / static_cast<float>(actions);
    for (const u32 a : cuda::std::views::iota(u32{0}, actions)) {
      strategy(a, h) = uniform;
    }
    return;
  }
  for (const u32 a : cuda::std::views::iota(u32{0}, actions)) {
    strategy(a, h) = positive_part(regrets(a, h)) * inv;
  }
}

/// One thread per hand. Host guarantees: `actions >= 1`, both buffers hold
/// `actions * hands` elements, and that product fits in `u32`.
template <RegretValue R>
__device__ void regret_match(const R* regrets_ptr, float* strategy_ptr, u32 actions, u32 hands) noexcept {
  const u32 h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hands) {
    return;
  }
  regret_match_hand(block(regrets_ptr, actions, hands), block(strategy_ptr, actions, hands), h);
}

}  // namespace
}  // namespace solver

extern "C" __global__ void regret_match_f32(const float* regrets,
                                            float* strategy,
                                            solver::u32 actions,
                                            solver::u32 hands) {
  solver::regret_match(regrets, strategy, actions, hands);
}

extern "C" __global__ void regret_match_i16(const solver::i16* regrets,
                                            float* strategy,
                                            solver::u32 actions,
                                            solver::u32 hands) {
  solver::regret_match(regrets, strategy, actions, hands);
}
