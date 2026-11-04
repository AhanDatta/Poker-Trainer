// Double-single ("double-float") arithmetic: a value held as the unevaluated
// sum of two f32, `hi + lo` with |lo| <= ulp(hi) / 2, giving about 44-48
// bits of precision on the GPU's fast f32 units (this GPU runs f64 at 1/64
// of its f32 rate). The exponent range is f32's: below about 1e-31 the low
// half underflows into f32's subnormals, so precision there is an absolute
// 2^-149 (1.4e-45) per operation rather than relative. That is far below
// anything that changes an f32 result the kernels produce (every reach is
// itself an f32). Above f32's maximum (3.4e38) values overflow; the host plan
// bounds payoffs and weights so that every product stays far below it.
//
// The algorithms are the classic error-free transformations (Dekker,
// Knuth; as in Bailey's QD library and Thall's "Extended-precision
// floating-point numbers for GPU computation"). They rely on every
// operation being rounded exactly as written, so each one is an explicit
// `_rn` intrinsic: the compiler can neither reassociate them nor fuse a
// multiply and an add (which would silently break `two_sum`).
//
// Also the accumulator vocabulary of the terminal kernels: `Accumulator`
// is `double` (exact path: bit-identical to the CPU) or `DoubleFloat`
// (fast path), with the same free functions for both.
#pragma once

#include <solver/core.cuh>

#include <cuda/std/type_traits>

namespace solver {

/// `hi + lo`, normalized so that `hi` is `hi + lo` rounded to f32. Trivial
/// (no member initializers) so it can live in `__shared__` memory;
/// `DoubleFloat{}` is zero.
struct DoubleFloat {
  float hi;
  float lo;
};

static_assert(cuda::std::is_trivial_v<DoubleFloat>);

static_assert(sizeof(DoubleFloat) == sizeof(double), "same shared-memory layout as the exact path");

namespace detail {

/// `a + b` exactly, as the rounded sum and its error (Knuth; no
/// precondition on the magnitudes).
[[nodiscard]] __device__ inline auto two_sum(float a, float b) noexcept -> DoubleFloat {
  const float s = __fadd_rn(a, b);
  const float v = __fsub_rn(s, a);
  const float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v));
  return {s, e};
}

/// `a + b` exactly when `|a| >= |b|` (or `a` is 0) (Dekker).
[[nodiscard]] __device__ inline auto quick_two_sum(float a, float b) noexcept -> DoubleFloat {
  const float s = __fadd_rn(a, b);
  const float e = __fsub_rn(b, __fsub_rn(s, a));
  return {s, e};
}

/// `a * b` exactly, as the rounded product and its error (one FMA).
[[nodiscard]] __device__ inline auto two_prod(float a, float b) noexcept -> DoubleFloat {
  const float p = __fmul_rn(a, b);
  const float e = __fmaf_rn(a, b, -p);
  return {p, e};
}

}  // namespace detail

/// `a + b` (the accurate double-single sum: both halves' errors kept).
[[nodiscard]] __device__ inline auto operator+(DoubleFloat a, DoubleFloat b) noexcept -> DoubleFloat {
  DoubleFloat s = detail::two_sum(a.hi, b.hi);
  const DoubleFloat t = detail::two_sum(a.lo, b.lo);
  s.lo = __fadd_rn(s.lo, t.hi);
  s = detail::quick_two_sum(s.hi, s.lo);
  s.lo = __fadd_rn(s.lo, t.lo);
  return detail::quick_two_sum(s.hi, s.lo);
}

/// `a + b` for an f32 `b`.
[[nodiscard]] __device__ inline auto operator+(DoubleFloat a, float b) noexcept -> DoubleFloat {
  DoubleFloat s = detail::two_sum(a.hi, b);
  s.lo = __fadd_rn(s.lo, a.lo);
  return detail::quick_two_sum(s.hi, s.lo);
}

[[nodiscard]] __device__ inline auto operator-(DoubleFloat a) noexcept -> DoubleFloat {
  return {-a.hi, -a.lo};
}

[[nodiscard]] __device__ inline auto operator-(DoubleFloat a, DoubleFloat b) noexcept -> DoubleFloat {
  return a + (-b);
}

/// `a * b`.
[[nodiscard]] __device__ inline auto operator*(DoubleFloat a, DoubleFloat b) noexcept -> DoubleFloat {
  DoubleFloat p = detail::two_prod(a.hi, b.hi);
  p.lo = __fmaf_rn(a.hi, b.lo, p.lo);
  p.lo = __fmaf_rn(a.lo, b.hi, p.lo);
  return detail::quick_two_sum(p.hi, p.lo);
}

// ---- Accumulators: the exact (f64) and fast (double-single) paths ----------

template <class T>
concept Accumulator = cuda::std::same_as<T, double> || cuda::std::same_as<T, DoubleFloat>;

/// An f32 value as an accumulator (exact for both).
template <Accumulator A>
[[nodiscard]] __device__ inline auto widen(float x) noexcept -> A {
  if constexpr (cuda::std::same_as<A, double>) {
    return static_cast<double>(x);
  } else {
    return DoubleFloat{x, 0.0f};
  }
}

/// An f64 value (a payoff) as an accumulator: exact for `double`, the
/// nearest double-single otherwise.
template <Accumulator A>
[[nodiscard]] __device__ inline auto narrow_from(double x) noexcept -> A {
  if constexpr (cuda::std::same_as<A, double>) {
    return x;
  } else {
    const float hi = __double2float_rn(x);
    return DoubleFloat{hi, __double2float_rn(x - static_cast<double>(hi))};
  }
}

[[nodiscard]] __device__ inline auto add(DoubleFloat a, DoubleFloat b) noexcept -> DoubleFloat { return a + b; }
[[nodiscard]] __device__ inline auto sub(DoubleFloat a, DoubleFloat b) noexcept -> DoubleFloat { return a - b; }
[[nodiscard]] __device__ inline auto mul(DoubleFloat a, DoubleFloat b) noexcept -> DoubleFloat { return a * b; }
[[nodiscard]] __device__ inline auto add(double a, double b) noexcept -> double { return add_rn(a, b); }
[[nodiscard]] __device__ inline auto sub(double a, double b) noexcept -> double { return sub_rn(a, b); }
[[nodiscard]] __device__ inline auto mul(double a, double b) noexcept -> double { return mul_rn(a, b); }

/// Accumulate an f32 value.
[[nodiscard]] __device__ inline auto add(DoubleFloat a, float b) noexcept -> DoubleFloat { return a + b; }
[[nodiscard]] __device__ inline auto add(double a, float b) noexcept -> double {
  return add_rn(a, static_cast<double>(b));
}

/// Rounded to f32 (`hi + lo` rounded once: the f32 nearest the represented
/// value).
[[nodiscard]] __device__ inline auto to_f32(DoubleFloat a) noexcept -> float { return __fadd_rn(a.hi, a.lo); }

/// Whether an accumulator is exactly zero.
[[nodiscard]] __device__ inline auto is_zero(DoubleFloat a) noexcept -> bool { return a.hi == 0.0f && a.lo == 0.0f; }
[[nodiscard]] __device__ inline auto is_zero(double a) noexcept -> bool { return a == 0.0; }

/// Sum functor for CUB scans and reductions.
struct Plus {
  template <Accumulator A>
  [[nodiscard]] __device__ auto operator()(A a, A b) const noexcept -> A {
    return add(a, b);
  }
};

}  // namespace solver
