// Self-test of the safety net: a kernel that deliberately reads past the end
// of its buffer. In checked builds (CCCL_ENABLE_DEVICE_ASSERTIONS) the span
// access must trap and reach Rust as an error, never as a crash or a silent
// out-of-bounds read. Only `tests/gpu_fault.rs` launches it, in a process of
// its own, because a trap leaves the CUDA context unusable.

#include <solver/core.cuh>

extern "C" __global__ void selftest_out_of_bounds(const float* data, solver::u32 len, float* out_ptr) {
  const auto in = solver::view(data, len);
  const auto out = solver::view(out_ptr, 1);
  // Index `len` is one past the end.
  out[0] = in[len];
}
