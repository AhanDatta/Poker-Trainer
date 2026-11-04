// Shared vocabulary for the solver's kernels: typed views over device
// buffers, exact arithmetic, and the encodings shared with the Rust host
// code (src/gpu/plan.rs). See ../../README.md for the rules these follow.
#pragma once

#include <cuda/atomic>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/ranges>
#include <cuda/std/span>

namespace solver {

using u32 = cuda::std::uint32_t;
using i16 = cuda::std::int16_t;

/// "No index" marker in index arrays (the host's `plan::NONE`).
inline constexpr u32 kNone = 0xFFFF'FFFFu;

/// Number of cards in the deck.
inline constexpr u32 kCards = 52;

/// What a traversal computes (the host's `cfr::Mode`; discriminants shared
/// with `plan::MODE_*`).
enum class Mode : u32 {
  /// One DCFR update for the traverser and the opponent's strategy sums.
  Cfr = 0,
  /// The traverser best-responds to the opponent's average strategy.
  BestResponse = 1,
  /// Both players play their average strategies.
  Average = 2,
};

// ---- Views -----------------------------------------------------------------

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

/// The `actions x hands` block starting at `offset` inside `buffer`; the
/// subspan is bounds-checked against the whole buffer, the block's accesses
/// against the block.
template <class T>
[[nodiscard]] __device__ constexpr auto block_at(cuda::std::span<T> buffer, u32 offset, u32 actions, u32 hands) noexcept
    -> Block<T> {
  return block(buffer.subspan(offset, actions * hands).data(), actions, hands);
}

/// Row `row` (of `width` elements) of a row-major buffer.
template <class T>
[[nodiscard]] __device__ constexpr auto row(cuda::std::span<T> buffer, u32 row, u32 width) noexcept
    -> cuda::std::span<T> {
  return buffer.subspan(row * width, width);
}

/// `0, 1, ..., n - 1` as a range.
[[nodiscard]] __host__ __device__ constexpr auto indices(u32 n) noexcept {
  return cuda::std::views::iota(u32{0}, n);
}

// ---- Exact arithmetic --------------------------------------------------------
//
// The CPU twin rounds every multiply and add separately. nvcc would fuse
// `a * b + c` into one FMA (a single rounding) and change results, so code
// that must match the CPU bit for bit uses these instead of `*`, `+`, `-`.

[[nodiscard]] __device__ inline auto mul_rn(float a, float b) noexcept -> float { return __fmul_rn(a, b); }
[[nodiscard]] __device__ inline auto add_rn(float a, float b) noexcept -> float { return __fadd_rn(a, b); }
[[nodiscard]] __device__ inline auto sub_rn(float a, float b) noexcept -> float { return __fsub_rn(a, b); }
[[nodiscard]] __device__ inline auto mul_rn(double a, double b) noexcept -> double { return __dmul_rn(a, b); }
[[nodiscard]] __device__ inline auto add_rn(double a, double b) noexcept -> double { return __dadd_rn(a, b); }
[[nodiscard]] __device__ inline auto sub_rn(double a, double b) noexcept -> double { return __dsub_rn(a, b); }

/// `f64 -> f32`, round to nearest (Rust's `as f32`).
[[nodiscard]] __device__ inline auto to_f32(double x) noexcept -> float { return __double2float_rn(x); }

// ---- Regret matching helpers -------------------------------------------------

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

// ---- Encodings shared with the host ------------------------------------------

/// The two cards of a hand, packed by the host as `c1 | c2 << 8`.
struct CardPair {
  u32 first;
  u32 second;
};

[[nodiscard]] __host__ __device__ constexpr auto unpack_cards(u32 packed) noexcept -> CardPair {
  return CardPair{packed & 0xFFu, (packed >> 8) & 0xFFu};
}

static_assert(unpack_cards(3u | (51u << 8)).first == 3u);
static_assert(unpack_cards(3u | (51u << 8)).second == 51u);

/// Where one of the traverser's hands falls in a board's strength-sorted
/// list of opponent hands (the host packs `lo | hi << 16`): opponent
/// positions `[0, lo)` are weaker, `[lo, hi)` tie, `[hi, len)` are stronger.
struct StrengthRange {
  u32 lo;
  u32 hi;
};

[[nodiscard]] __host__ __device__ constexpr auto unpack_range(u32 packed) noexcept -> StrengthRange {
  return StrengthRange{packed & 0xFFFFu, packed >> 16};
}

/// For each of the hand's two cards, how many opponent hands holding that
/// card are weaker (`lo`) and how many are weaker or tied (`hi`); packed by
/// the host as `first.lo | first.hi << 8 | second.lo << 16 | second.hi << 24`.
struct CardRanks {
  u32 first_lo;
  u32 first_hi;
  u32 second_lo;
  u32 second_hi;
};

[[nodiscard]] __host__ __device__ constexpr auto unpack_ranks(u32 packed) noexcept -> CardRanks {
  return CardRanks{packed & 0xFFu, (packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, packed >> 24};
}

static_assert(unpack_range(7u | (900u << 16)).lo == 7u);
static_assert(unpack_range(7u | (900u << 16)).hi == 900u);
static_assert(unpack_ranks(1u | (2u << 8) | (3u << 16) | (4u << 24)).second_hi == 4u);

}  // namespace solver
