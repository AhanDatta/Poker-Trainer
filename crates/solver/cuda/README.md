# GPU kernels (CUDA C++)

Kernels for the optional `gpu` feature of the `solver` crate. The default build
never touches this directory; `cargo build -p solver --features gpu` compiles
every `kernels/*.cu` with `nvcc` (see `../build.rs`) into a fat binary that is
embedded in the Rust library and loaded through the CUDA driver at run time.

Requirements for `--features gpu`: CUDA Toolkit 13.x (`nvcc`; found through
`NVCC`, then `CUDA_PATH`, then `PATH`) and, on Windows, Visual Studio 2022's
MSVC as the host compiler. A built binary needs only the NVIDIA driver.

## Ground rules

1. **No panics, no faults.** The CPU solver is the reference; the GPU path must
   either produce the same answer (within documented tolerances) or return an
   error that the caller turns into a CPU fallback. Never a crash, a hang, or
   a silently wrong number.
2. **Modern, typed C++.** Kernels are C++20 and use the CUDA standard library
   (`cuda::std`, a.k.a. libcu++) and CUB instead of raw pointers, macros and
   hand-written reductions.

## How rule 1 is enforced

| Layer | Mechanism |
|---|---|
| Rust API (`src/gpu`) | Every fallible call returns `Result<_, GpuError>`. The module denies `unwrap`, `expect`, `panic!`, `unreachable!` and slice indexing (clippy lints). Before calling into cudarc it checks that the driver library exists and is new enough, because cudarc itself panics in those two cases. |
| Inputs to kernels | Validated on the host before launch: lengths match, sizes fit in `u32`, products don't overflow, every index stored in a schedule is in range. Kernels may assume validated invariants; the validators have their own unit and property tests. |
| Kernel memory access | Only through `cuda::std::span` / `cuda::std::mdspan` built from `(pointer, length)` pairs at the top of the kernel. Debug and test builds define `CCCL_ENABLE_DEVICE_ASSERTIONS`, which bounds-checks every `span[]` / `mdspan()` access; a violation traps the kernel and surfaces as a `GpuError`. Release builds drop the checks (`SOLVER_GPU_CHECKED=1` keeps them). |
| Whole-program checking | The GPU tests are run under NVIDIA's `compute-sanitizer` (memcheck, racecheck, initcheck, synccheck), which catches out-of-bounds and misaligned accesses, shared-memory races, reads of uninitialized device memory, and invalid `__syncthreads` use. |
| Results | Every kernel has a CPU twin (the existing solver code) and is tested against it: hand-picked edge cases (zero/subnormal/huge values, empty ranges, a single action, sizes that straddle block boundaries) plus `proptest` over random inputs. Where the GPU does the same IEEE operations in the same order (no FMA contraction) the test demands bit-for-bit equality; elsewhere it uses the documented tolerances of the existing CPU-vs-CPU tests, never looser ones. |
| Build | `nvcc -Werror all-warnings`; IEEE division and square root (no `--use_fast_math`), denormals preserved (the `recip()` guard depends on seeing subnormals). |

## How rule 2 is applied (and where it can't be)

Device code has no heap, no exceptions and no RTTI, so some host-side C++
idioms don't carry over. What we use instead:

| Host idiom | In kernels |
|---|---|
| Smart pointers / RAII ownership | Device memory is **owned by Rust** (`cudarc::CudaSlice` frees on drop). Kernels never own memory; they receive non-owning `cuda::std::span` / `mdspan` views — the modern C++ way to express borrowing. |
| `std::span`, `std::mdspan` | `cuda::std::span`, `cuda::std::mdspan` (an action-major block is `mdspan<T, dextents<uint32_t, 2>>`, layout `[action][hand]`). |
| Ranges | `cuda::std::views::iota` for index loops; `cuda::std::ranges` algorithms where they exist on device. |
| Algebraic types | `enum class` for closed sets (player, node kind); `cuda::std::optional` / `cuda::std::expected` for values that may be absent; strong index types so a hand index can't be passed where a node index is expected. `cuda::std::variant` is available, but per-thread dispatch on a variant makes warps diverge, so node kinds are separated on the host into one kernel launch per kind — the type distinction lives in the schedule's structure. |
| Concepts, `constexpr` | Kernel templates are constrained with concepts (e.g. the regret type must be `f32` or `i16`); pure math is `constexpr` and shared with host-side static tests. |
| Exceptions / error codes | None on device. Bad inputs are rejected on the host before launch; assertion failures trap; driver errors become `GpuError`. |
| Reductions / scans | CUB (`cub::BlockScan`, `cub::BlockReduce`, warp primitives) instead of hand-rolled shared-memory loops. |

Kernel entry points are `extern "C"` (stable names for the Rust loader) and take
only `(pointer, length)` pairs and scalars, never C++ class types, so the
Rust-side argument layout can't silently disagree with the C++ one. The first
lines of each kernel wrap those pairs into typed views.

## Precision

The GPU's f64 rate is 1/64 of f32, so kernels accumulate in f32. Where the CPU
accumulates in f64 (chance-node sums, showdown sweeps) the kernel must use
compensated (Kahan/Neumaier) or pairwise summation, and the CPU-parity tests
decide whether that is enough.

## Running the GPU checks

```bash
cargo test -p solver --features gpu gpu::
```

```bash
cargo test -p solver --features gpu --no-run --message-format=short
```

Then run the printed test executable under the sanitizer, once per tool:

```bash
compute-sanitizer --tool memcheck --leak-check full <test-exe> gpu::
```

GPU tests skip (with a message) when no NVIDIA driver or device is present.
