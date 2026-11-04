// Regret matching: the current strategy of a decision node from its
// cumulative regrets.
//
// CPU twin: `cfr::regret_match_into`. Same IEEE operations in the same order
// (positive parts summed in action order, one guarded reciprocal, one
// multiply per action), so results are identical to the CPU's.

#include <solver/core.cuh>
#include <solver/strategy.cuh>

namespace solver {
namespace {

/// One thread per hand. Host guarantees: `actions >= 1`, both buffers hold
/// `actions * hands` elements, and that product fits in `u32`.
template <RegretValue R>
__device__ void regret_match(const R* regrets_ptr, float* strategy_ptr, u32 actions, u32 hands) noexcept {
  const u32 h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hands) {
    return;
  }
  const auto regrets = block(regrets_ptr, actions, hands);
  const auto strategy = block(strategy_ptr, actions, hands);
  const float inv = recip(positive_sum(regrets, h));
  for (const u32 a : indices(actions)) {
    strategy(a, h) = matched(regrets, a, h, inv);
  }
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
