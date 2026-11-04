// Test kernels for the double-single arithmetic (double_float.cuh): the host
// compares their results with f64 (src/gpu/tests.rs). Not used by solves.

#include <solver/core.cuh>
#include <solver/double_float.cuh>

using solver::u32;

namespace solver {
namespace {

/// The represented value `hi + lo`, exactly, as f64.
[[nodiscard]] __device__ auto to_f64(DoubleFloat a) noexcept -> double {
  return static_cast<double>(a.hi) + static_cast<double>(a.lo);
}

/// Operation codes shared with the host (`DoubleFloatOp`).
enum class Op : u32 {
  Add = 0,
  Sub = 1,
  Mul = 2,
  AddF32 = 3,
  RoundToF32 = 4,
};

}  // namespace
}  // namespace solver

/// `out[i] = op(x[i], y[i])` with both operands converted to double-single
/// (`narrow_from`) and the result read back exactly; `AddF32` adds `y[i]`
/// rounded to f32, `RoundToF32` returns `to_f32(x[i])`.
extern "C" __global__ void double_float_check(const double* x_ptr,
                                              const double* y_ptr,
                                              double* out_ptr,
                                              u32 len,
                                              u32 op_raw) {
  using namespace solver;
  const auto x = view(x_ptr, len);
  const auto y = view(y_ptr, len);
  const auto out = view(out_ptr, len);
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len) {
    return;
  }
  const DoubleFloat a = narrow_from<DoubleFloat>(x[i]);
  const DoubleFloat b = narrow_from<DoubleFloat>(y[i]);
  switch (static_cast<Op>(op_raw)) {
    case Op::Add:
      out[i] = to_f64(a + b);
      break;
    case Op::Sub:
      out[i] = to_f64(a - b);
      break;
    case Op::Mul:
      out[i] = to_f64(a * b);
      break;
    case Op::AddF32:
      out[i] = to_f64(a + __double2float_rn(y[i]));
      break;
    case Op::RoundToF32:
      out[i] = static_cast<double>(to_f32(a));
      break;
  }
}

/// One thread sums `values` sequentially into a double-single (as the
/// terminal kernels accumulate a chunk) and writes the result exactly.
extern "C" __global__ void double_float_sum(const float* values_ptr, u32 len, double* out_ptr) {
  using namespace solver;
  const auto values = view(values_ptr, len);
  const auto out = view(out_ptr, 1);
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  DoubleFloat sum{};
  for (const u32 i : indices(len)) {
    sum = add(sum, values[i]);
  }
  out[0] = to_f64(sum);
}
