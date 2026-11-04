# GPU kernels (CUDA C++)

Kernels for the optional `gpu` feature of the `solver` crate. The default build
never touches this directory; `cargo build -p solver --features gpu` compiles
every `kernels/*.cu` with `nvcc` (see `../build.rs`) into a fat binary that is
embedded in the Rust library and loaded through the CUDA driver at run time.

Requirements for `--features gpu`: CUDA Toolkit 13.x (`nvcc`; found through
`NVCC`, then `CUDA_PATH`, then `PATH`) and, on Windows, Visual Studio 2022's
MSVC as the host compiler. A built binary needs only the NVIDIA driver.

## Status

Experimental. `Solver::use_gpu()` (with `--features gpu`) moves a solve to the
GPU: river and turn roots, and flop roots small enough for `f32` storage
(sampled runouts). **Compressed storage** (16-bit, which exact flop trees use)
is not supported yet, and a tree that doesn't fit in free GPU memory (keeping a
0.5 GB margin) is declined; both stay on the CPU.

| File | Contents | CPU twin |
|---|---|---|
| `include/solver/core.cuh` | views, exact arithmetic, encodings shared with `src/gpu/plan.rs` | |
| `include/solver/strategy.cuh` | per-hand regret matching and normalization | `cfr::regret_match_into`, `cfr::normalize_in_place` |
| `kernels/traverse.cu` | level-by-level traversal: opponent and chance splits (top-down), own-node regret update, opponent and chance sums (bottom-up) | `river::run`, `cfr::chance`, `cfr::own_node_update` |
| `kernels/terminal.cu` | fold and showdown values (card removal by inclusion–exclusion, prefix sums), with the rounding of the CPU evaluator that handles each terminal | `eval::fold_batch` / `eval_fold`, `eval::showdown_batch` / `eval_showdown` |
| `kernels/regret.cu` | standalone regret matching | `cfr::regret_match_into` |
| `kernels/selftest.cu` | deliberate out-of-bounds read for `tests/gpu_fault.rs` | |

The host side (`src/gpu/`): `plan.rs` flattens the tree into arrays, a level
schedule per traverser and per-board showdown lookups, and validates every
index the kernels will use; `engine.rs` uploads it once and launches one
kernel per tree level; `Solver::solve` drives the same loop (iterations,
exploitability checks, cancellation) for both backends.

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
| Algebraic types | `enum class` for closed sets (`Mode`); packed host encodings decoded into named structs (`CardPair`, `StrengthRange`, `CardRanks`) instead of bit-twiddling at use sites; `cuda::std::optional` / `cuda::std::expected` where a value may be absent. `cuda::std::variant` is available, but per-thread dispatch on a variant makes warps diverge, so node kinds are separated on the host into one kernel launch per kind — the type distinction lives in the schedule's structure. Indices (node, slot, hand) are plain `u32` checked by `Plan::validate`; strong index types are not used yet. |
| Concepts, `constexpr` | Kernel templates are constrained with concepts (e.g. the regret type must be `f32` or `i16`); pure math is `constexpr` and shared with host-side static tests. |
| Exceptions / error codes | None on device. Bad inputs are rejected on the host before launch; assertion failures trap; driver errors become `GpuError`. |
| Reductions / scans | CUB (`cub::BlockScan`, `cub::BlockReduce`, warp primitives) instead of hand-rolled shared-memory loops. |

Kernel entry points are `extern "C"` (stable names for the Rust loader) and take
only `(pointer, length)` pairs and scalars, never C++ class types, so the
Rust-side argument layout can't silently disagree with the C++ one. The first
lines of each kernel wrap those pairs into typed views.

## Precision

Kernels mirror the CPU's arithmetic operation for operation:

- **f32 work** (regret matching, regret and strategy-sum updates, node
  values) uses `mul_rn` / `add_rn` / `sub_rn` (`core.cuh`), which nvcc can't
  fuse into FMAs, in the CPU's order. Same inputs give bit-identical results.
- **Two CPU evaluators.** The CPU batches the terminals of river subtrees
  (`river::run`) and evaluates the others one at a time (folds before the
  river, showdowns right after an all-in runout, or everything with batching
  off); the two group and round their sums differently. The plan marks each
  terminal with the CPU evaluator that handles it (`Plan::scalar`), and the
  kernels use that one's operation order. Mirroring only the batched one left
  the turn a rounding step off, which regret matching amplified to visible
  regret differences.
- **Terminal sums** run in **f64**, as on the CPU. The GPU's f64 rate is 1/64
  of its f32 rate, and the profile shows it matters (the first version
  assumed the f64 work would be negligible; it wasn't): the terminal kernels are
  written to do about one f64 add per summed value (see Performance). Only the
  summation order differs from the CPU (chunked sums and a block scan instead
  of one sweep), which could move a result by one f32 rounding step; in every
  test so far it hasn't.

Measured: GPU and CPU solves agree **bit for bit** (every strategy, EV and
regret identical after 5 iterations) on the river, turn, suit-isomorphic turn,
all-in turn and sampled flop spots, on 24 random river and 10 random turn
spots, and with river batching off. The tests still allow the CPU-vs-CPU tolerances of
`batched_river_matches_recursion` (5e-5 in strategy, 0.25 chips in EV after 5
iterations), because a different summation order may occasionally round
differently. Seeding a kernel bug makes several of these tests fail: wrong
regret discount, dropped tie term, one card's removal skipped, dealt card not
blocking opponent or own hands, isomorphic cards skipped or unpermuted, chance
factor dropped, wrong evaluator rounding.

## Performance

RTX 4060 Laptop GPU (8 GB) and i7-13700H (6P + 8E cores), plugged in,
2026-11-04. `examples/bench` in release mode with its default spots, stopping at
0.3% of the pot. Each cell gives the median and the range of the runs. `main`
and this branch ran interleaved on the CPU. All builds reach the same
iterations, exploitability and root strategy.

| Solve | Iterations | CPU, `main` | CPU, this branch | GPU, this branch |
|---|---|---|---|---|
| river `Ks7d2c9h4s` (5 runs) | 80 | 9.9 ms (9.6–10.9) | 10.6 ms (9.8–11.8) | 13.5 ms (13.2–14.5) + 0.15–0.3 s setup |
| river, fixed 3000 / 1000 iterations | | 0.132 ms/iter | 0.135 ms/iter | 0.155–0.167 ms/iter (CPU 0.129–0.138 in the same 1000-iteration runs) |
| turn `Ks7d2c9h` (3 runs) | 168 | 3.09 s (3.00–3.29), 18.4 ms/iter | 3.08 s (2.97–3.13), 18.3 ms/iter | **1.20 s** (1.19–1.37), 7.2 ms/iter, + 0.35–0.37 s setup |
| flop-fast `Ks7d2c`, 8 × 6 sampled (2 runs) | 116 | 4.67 s, 40.2 ms/iter | 4.60 s, 39.7 ms/iter | **1.58 s**, 13.6 ms/iter, + 0.55–0.65 s setup |
| flop-exact `Ks7s2d` (2 runs) | 184 | 152.7–155.9 s, 830–847 ms/iter | 151.2–155.2 s, 822–843 ms/iter | not yet (16-bit storage) |

Peak memory (this branch's bench; `main`'s can't measure it on Windows): river
0.01 GB, turn 0.15 GB, flop-fast 0.26 GB, flop-exact 2.98 GB (16-bit storage).
The GPU river process peaks at 0.16 GB, mostly the CUDA runtime.

The GPU rows for the turn and the flop come from a later session that ran the
CPU, the first GPU version (before the terminal optimization below) and the
current one interleaved. In that session the CPU took 2.79–3.22 s on the turn
(median 3.19 s) and 4.27–4.35 s on the sampled flop; the first GPU version
1.94–2.01 s and 2.45–2.46 s. So the GPU is now **about 2.7× faster than the CPU
on the turn and the sampled flop** (1.6× faster than the first GPU version),
with the same iterations, exploitability and root strategy; counting the
one-time setup (CUDA context, plan, upload), the turn takes 1.7–1.8 s against
2.9–3.3 s. The river stays slightly slower than the CPU (11.9 ms against about
10 ms): its 14-node tree can't fill the GPU and launch latency sets the time.

### Where the GPU time goes

Nsight Systems, 60 turn iterations (122 traversals), kernel time per traversal:

| Kernel | First version | Now | Change |
|---|---|---|---|
| `showdown_values` (now also the paired folds) | 2.40 ms | 1.55 ms | f64 work cut to about one add per summed value |
| `fold_values` | 1.58 ms | 0.41 ms | half the folds paired with showdowns; per-card sums sequential |
| `opponent_split` | 0.59 ms | 0.59 ms | |
| `own_update` | 0.38 ms | 0.38 ms | |
| `opponent_sum` | 0.22 ms | 0.22 ms | |
| `chance_split` + `chance_sum` | 0.06 ms | 0.06 ms | |
| total per iteration | 10.5 ms | 6.4 ms | −39% |

What changed in the terminal kernels:

- **f64 operation count.** The GPU does f64 at 2 operations per cycle per SM
  (128 for f32). The first version summed each card's opponent hands with
  warp shuffle reductions and scanned with warp-wide prefix scans: about 5
  f64 adds per summed value. Now each thread sums a contiguous chunk (or one
  card's list) sequentially and only the 256 chunk totals go through a
  block-wide scan or reduction.
- **Folds paired with showdowns.** A fold and a showdown that are both
  children of one of the traverser's nodes share the opponent reach; the
  showdown kernel now evaluates the fold from its own sums (the CPU's
  `showdown_batch` grouping), which removes about half of the fold blocks.

Tried and rejected (measured slower or no better): 512-thread showdown blocks
(10% slower), computing the paired fold inside the showdown's per-hand loop
(30% slower), caching the board's card offsets in shared memory (no gain),
loop strides from `blockDim.x` instead of a compile-time constant (a few
percent slower).

What remains, and why it isn't simple to cut:

- The traversal kernels (`opponent_split`, `own_update`, `opponent_sum`, 1.2
  ms per traversal) already move about 70% of the GPU's 256 GB/s; the turn
  tree's 16 million regret/strategy cells are read and written every
  traversal. 16-bit storage (phase 3) would halve that traffic.
- The terminal kernels are still bound by f64 arithmetic. A timing probe with
  every f64 replaced by f32 (wrong results, same structure) ran showdowns in
  0.47 ms instead of 1.60 ms and folds in 0.09 ms instead of 0.41 ms per
  traversal. Emulating f64 with pairs of f32 ("double-single" arithmetic,
  about 44 bits of precision on the fast f32 units) would recover much of
  that, at the cost of rare one-rounding-step differences from the CPU (no
  longer bit for bit; within the test tolerances).
- Hardware performance counters need administrator rights on Windows
  (`ERR_NVGPUCTRPERM`); with them enabled, Nsight Compute could say exactly
  which stalls remain.

## Running the GPU checks

```bash
cargo test -p solver --features gpu gpu::
```

`gpu::plan` tests need no device: they check the host-side plan, including a
Rust reference of the showdown/fold formula built from the plan's lookup tables
against `eval_showdown` / `eval_fold`. `gpu::engine` tests compare GPU
traversals and solves with the CPU solver.

```bash
cargo test -p solver --features gpu --no-run --message-format=short
```

Then run the printed test executable under the sanitizer, once per tool:

```bash
compute-sanitizer --tool memcheck --leak-check full <test-exe> gpu::
```

GPU tests skip (with a message) when no NVIDIA driver or device is present.
