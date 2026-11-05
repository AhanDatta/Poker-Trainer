// 16-bit ("compressed") storage, as the CPU's `storage::Storage::Compressed`
// lays it out: regrets as i16 and strategy sums as u16, each node's block
// with one f32 scale (the block's largest absolute value). A stored q stands
// for `q * (scale / MAX)`.
//
// Decoding and encoding mirror `Storage::update_regrets` /
// `Storage::update_strat` / `Storage::load_strat` operation for operation,
// so the exact path stays bit-identical to the CPU with compressed storage.
#pragma once

#include <solver/core.cuh>

namespace solver {

using u16 = cuda::std::uint16_t;

/// `i16::MAX` and `u16::MAX` as f32 (the CPU's `i16::MAX as f32`).
inline constexpr float kRegretMax = 32767.0f;
inline constexpr float kStrategyMax = 65535.0f;

/// A decoded view of a 16-bit block: `(a, h)` reads `q(a, h) * step`, where
/// `step = scale / MAX` (CPU: `x as f32 * scale`, the scale divided once).
template <class Q>
  requires cuda::std::same_as<Q, i16> || cuda::std::same_as<Q, u16>
struct Scaled {
  Block<const Q> q;
  float step;

  [[nodiscard]] __device__ auto extent(u32 r) const noexcept -> u32 { return q.extent(r); }
  [[nodiscard]] __device__ auto operator()(u32 a, u32 h) const noexcept -> float {
    return mul_rn(static_cast<float>(q(a, h)), step);
  }
};

/// The decoding step of a regret block with scale `scale`.
[[nodiscard]] __device__ inline auto regret_step(float scale) noexcept -> float {
  return __fdiv_rn(scale, kRegretMax);
}

/// The decoding step of a strategy-sum block with scale `scale`.
[[nodiscard]] __device__ inline auto strategy_step(float scale) noexcept -> float {
  return __fdiv_rn(scale, kStrategyMax);
}

/// Rust's saturating `f32 as i16` / `as u16`: truncation toward zero, NaN to 0.
[[nodiscard]] __device__ inline auto saturate_i16(float t) noexcept -> i16 {
  if (!(t == t)) {
    return 0;
  }
  return static_cast<i16>(__float2int_rz(fminf(fmaxf(t, -32768.0f), 32767.0f)));
}

[[nodiscard]] __device__ inline auto saturate_u16(float t) noexcept -> u16 {
  if (!(t == t)) {
    return 0;
  }
  return static_cast<u16>(__float2uint_rz(fminf(fmaxf(t, 0.0f), 65535.0f)));
}

/// A new regret as i16 given `inv = MAX / block_max` (CPU: `y = v * inv`,
/// then `(y + 0.5f32.copysign(y)) as i16`: round half away from zero).
[[nodiscard]] __device__ inline auto encode_regret(float v, float inv) noexcept -> i16 {
  const float y = mul_rn(v, inv);
  return saturate_i16(add_rn(y, copysignf(0.5f, y)));
}

/// A new strategy sum as u16 (CPU: `(v.max(0.0) * inv + 0.5) as u16`).
[[nodiscard]] __device__ inline auto encode_strategy(float v, float inv) noexcept -> u16 {
  return saturate_u16(add_rn(mul_rn(fmaxf(v, 0.0f), inv), 0.5f));
}

}  // namespace solver
