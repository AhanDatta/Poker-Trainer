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
- **Terminal sums** run in **f64**, as on the CPU, even though the GPU's f64
  rate is 1/64 of its f32 rate: a showdown does a few thousand f64 adds, a tiny
  share of an iteration's work, so compensated f32 summation isn't needed. Only
  the summation order differs (tree reductions and prefix scans instead of one
  sweep), which can move a result by at most one f32 rounding step.

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
| turn `Ks7d2c9h` (3 runs) | 168 | 3.09 s (3.00–3.29), 18.4 ms/iter | 3.08 s (2.97–3.13), 18.3 ms/iter | 1.95 s (1.94–1.95), 11.6 ms/iter, + 0.4–0.46 s setup |
| flop-fast `Ks7d2c`, 8 × 6 sampled (2 runs) | 116 | 4.67 s, 40.2 ms/iter | 4.60 s, 39.7 ms/iter | 2.47 s (2.45–2.49), 21.3 ms/iter, + 0.5–0.6 s setup |
| flop-exact `Ks7s2d` (2 runs) | 184 | 152.7–155.9 s, 830–847 ms/iter | 151.2–155.2 s, 822–843 ms/iter | not yet (16-bit storage) |

Peak memory (this branch's bench; `main`'s can't measure it on Windows): river
0.01 GB, turn 0.15 GB, flop-fast 0.26 GB, flop-exact 2.98 GB (16-bit storage).
The GPU river process peaks at 0.16 GB, mostly the CUDA runtime.

The GPU rows for the turn and the flop come from a second session (same
binaries, CPU and GPU interleaved), in which the CPU ran 10–15% slower than in
the first (turn 3.42–3.76 s, flop-fast 4.85–5.29 s), probably from background
load; the GPU times varied by under 2%. Against either CPU session the GPU is
**1.6–1.8× faster on the turn and 1.9–2.1× on the sampled flop**, with the same
iterations, exploitability and root strategy. The setup (CUDA context, plan,
upload) is paid once per solve.

The GPU loses on the river, as expected. The tree has 14 decision nodes, so
an iteration is about 60 small kernel launches, and launch latency, not
arithmetic, sets the time. Creating the CUDA context costs another 0.15–0.3 s
once per process.

Where the GPU time goes (Nsight Systems, 60 turn iterations, 122 traversals,
10.5 ms of kernel time per iteration of 11.6 ms wall):

| Kernel | Share | Per traversal |
|---|---|---|
| `showdown_values` | 46% | 2.4 ms (about 4,300 showdowns) |
| `fold_values` | 30% | 1.6 ms (about 3,600 folds) |
| `opponent_split` | 11% | 0.6 ms |
| `own_update` | 7% | 0.4 ms |
| `opponent_sum` | 4% | 0.2 ms |
| `chance_split` + `chance_sum` | 1% | 0.06 ms |

The terminals dominate. Each terminal block recomputes reach totals and per-card
sums that terminals sharing a reach slot could share, and the f64 scans are
sync-heavy; that is the first target for optimization.

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
