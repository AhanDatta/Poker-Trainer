// Shared vocabulary for the solver's kernels: typed views over device
// buffers and the numeric helpers that must match the CPU solver exactly.
// See ../../README.md for the rules these follow.
#pragma once

#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/span>

namespace solver {

using u32 = cuda::std::uint32_t;
using i16 = cuda::std::int16_t;

/// Non-owning view of a device buffer that arrived as a `(pointer, length)`
/// kernel argument. Accesses are bounds-checked when the kernels are built
/// with `CCCL_ENABLE_DEVICE_ASSERTIONS` (debug and test builds).
template <class T>
[[nodiscard]] __device__ constexpr auto view(T* data, u32 len) noexcept -> cuda::std::span<T> {
  return cuda::std::span<T>{data, len};
}

/// The storage block of one decision node, as laid out by the CPU solver:
/// action-major, element `(a, h)` at `a * hands + h` (`layout_right`).
template <class T>
using Block = cuda::std::mdspan<T, cuda::std::dextents<u32, 2>>;

template <class T>
[[nodiscard]] __device__ constexpr auto block(T* data, u32 actions, u32 hands) noexcept -> Block<T> {
  return Block<T>{data, actions, hands};
}

/// Element types a regret block can be stored in: `f32`, or `i16` in
/// compressed storage (the per-node scale cancels out in regret matching).
template <class T>
concept RegretValue = cuda::std::same_as<T, float> || cuda::std::same_as<T, i16>;

/// `max(x, 0)` as a float.
template <RegretValue T>
[[nodiscard]] __host__ __device__ constexpr auto positive_part(T x) noexcept -> float {
  const auto f = static_cast<float>(x);
  return f > 0.0f ? f : 0.0f;
}

/// `1 / d` for a positive `d` whose reciprocal is finite; `0` ("nothing
/// positive") otherwise, including subnormal `d` whose reciprocal overflows.
/// Mirrors `cfr::recip` on the CPU: a subnormal positive-regret sum once
/// turned strategies into infinities and NaN.
[[nodiscard]] __host__ __device__ constexpr auto recip(float d) noexcept -> float {
  const float inv = 1.0f / d;
  // For d > 0 the reciprocal is positive, so "finite" is "<= max"; NaN fails
  // every comparison and lands on 0 as well.
  return (d > 0.0f && inv <= cuda::std::numeric_limits<float>::max()) ? inv : 0.0f;
}

static_assert(recip(2.0f) == 0.5f);
static_assert(recip(0.0f) == 0.0f);
static_assert(recip(-0.0f) == 0.0f);
static_assert(recip(-3.0f) == 0.0f);
static_assert(positive_part(-1.5f) == 0.0f);
static_assert(positive_part(static_cast<i16>(-7)) == 0.0f);
static_assert(positive_part(static_cast<i16>(12)) == 12.0f);

}  // namespace solver
