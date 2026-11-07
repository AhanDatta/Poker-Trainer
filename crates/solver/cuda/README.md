# GPU kernels (CUDA C++)

Kernels for the optional `gpu` feature of the `solver` crate. The default build
never touches this directory; `cargo build -p solver --features gpu` compiles
every `kernels/*.cu` with `nvcc` (see `../build.rs`) into a fat binary that is
embedded in the Rust library and loaded through the CUDA driver at run time.

Requirements for `--features gpu`: CUDA Toolkit 13.x (`nvcc`; found through
`NVCC`, then `CUDA_PATH`, then `PATH`) and, on Windows, Visual Studio 2022's
MSVC as the host compiler. A built binary needs only the NVIDIA driver.

## Status

Experimental. `Solver::use_gpu(precision)` (with `--features gpu`) moves a
solve to the GPU: river, turn and flop roots, with `f32` or 16-bit
(compressed) storage, bit-identical to the CPU with `GpuPrecision::Exact`.
`MultiSolver::use_gpu(precision)` does the same for three-player solves,
with either terminal model (independent opponents or exact card removal
between them), so far with f32 storage and whole trees (see "Three-player
solves").

GPU memory holds the regret/strategy storage, the plan, and scratch buffers:
opponent reach per "slot" and one row of values per node. When the whole
tree's scratch doesn't fit in free memory (keeping a 0.5 GB margin), the tree
is **split**: the subtrees under the root street's chance nodes (turn
subtrees on a flop, river subtrees on a turn) become *chunks*, and groups of
chunks run one after another in one shared scratch region (see
`src/gpu/plan/schedule.rs`). Splitting changes no result bit; it only costs
more, smaller launches. A tree whose storage plus a single chunk doesn't fit
is declined (`GpuError::OutOfMemory`) and stays on the CPU.

| File | Contents | CPU twin |
|---|---|---|
| `include/solver/core.cuh` | views, exact arithmetic, encodings shared with `src/gpu/plan.rs` | |
| `include/solver/strategy.cuh` | per-hand regret matching and normalization | `cfr::regret_match_into`, `cfr::normalize_in_place` |
| `include/solver/double_float.cuh` | double-single arithmetic and the `Accumulator` vocabulary (f64 or double-single) of the terminal kernels | |
| `include/solver/compressed.cuh` | 16-bit storage: decoding views, encoding with the CPU's rounding and saturation | `storage::Storage::Compressed` |
| `include/solver/traversal.cuh` | the per-node traversal steps (opponent and chance splits, own update, opponent and chance sums), parameterized by liveness: one opponent slot per node heads-up, two with three players | `river::run`, `cfr::chance`, `cfr::own_node_update`, `multi::cfr::traverse` |
| `include/solver/terminal.cuh` | heads-up terminal building blocks: chunked sums, per-card prefix sums, a board's strength-order prefix sums (`strength_sums`) | |
| `include/solver/lanes.cuh` | three-player leaves one per GPU lane: the sweep lists (`HandList`), a lane's columns of the scratch (`Lane`) | |
| `include/solver/independent.cuh` | a lane's independent-opponent leaf: hand-order compatible mass, strength-order beaten mass | `Hands::compat`, `Table::beat`, `multi::cfr::independent_showdown`, `pay_all` |
| `include/solver/exact.cuh` | a lane's exact-removal leaf: the two-set inclusion–exclusion sweep (`Set`) | `multi::exact` (`Pairs`, `showdown`, `fold_values`) |
| `kernels/traverse.cu` | heads-up level-by-level traversal: the steps of `traversal.cuh` over one level's nodes | `river::run`, `cfr::chance`, `cfr::own_node_update` |
| `kernels/traverse_compressed.cu` | the two storage-touching traversal steps for 16-bit storage, one block per node (a block-wide max sets each node's new scale) | `Storage::update_regrets`, `Storage::update_strat` |
| `kernels/terminal.cu` | fold and showdown values (card removal by inclusion–exclusion, prefix sums), with the rounding of the CPU evaluator that handles each terminal | `eval::fold_batch` / `eval_fold`, `eval::showdown_batch` / `eval_showdown` |
| `kernels/multi_traverse.cu` | three-player level-by-level traversal: each opponent's splits, two-slot chance splits, own update, sums | `multi::cfr::traverse` |
| `kernels/multi_leaves.cu` | three-player leaves, per terminal model and precision: showdown warps (one board, same folded players), fixed-payoff lanes (one pair of reach slots each) | `multi::cfr::pay_all`, `independent_showdown`, `multi::exact::showdown`, `river::subtree`'s leaves |
| `kernels/regret.cu` | standalone regret matching | `cfr::regret_match_into` |
| `kernels/selftest.cu` | deliberate out-of-bounds read for `tests/gpu_fault.rs` | |
| `kernels/double_float_check.cu` | double-single operations for the tests against host f64 | |

The host side (`src/gpu/`):

- `plan.rs` flattens a heads-up tree into arrays, `plan/schedule.rs` splits
  it into segments and lays out a level schedule per traverser, and both
  validate every index the kernels will use; `engine.rs` uploads the plan
  once and launches one kernel per tree level (and segment); `Solver::solve`
  drives the same loop (iterations, exploitability checks, cancellation) for
  both backends.
- `multi/plan.rs` (+ `multi/plan/validate.rs`, and `multi/plan/leaves.rs`
  for the leaf kernels' lists and groups) and `multi/engine.rs` (+
  `multi/engine/leaves.rs`) are the same for three-player trees;
  `MultiSolver::solve` runs either backend through one loop.
- `tables.rs`: the heads-up showdown lookups. Shared by both: `shared.rs`
  (encodings, limits, validation helpers), `device.rs` (buffers with their
  kernel-visible lengths, launch grids, traversal modes, storage upload and
  download, memory checks).

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

`Solver::use_gpu` takes a `GpuPrecision`:

- **`Exact`**: bit-identical to the CPU solver in every test so far. Kept as
  the reference and the backup.
- **`Fast`** (the default): terminal sums in double-single arithmetic. About
  1.4× faster than `Exact` on the turn and the sampled flop, and not bit
  identical to the CPU (see below).

Both run the same kernels; only the terminal kernels' accumulator type
differs (`Accumulator` in `double_float.cuh`), chosen per launch.

How the exact path matches the CPU bit for bit:

- **f32 work** (regret matching, regret and strategy-sum updates, node
  values) uses `mul_rn` / `add_rn` / `sub_rn` (`core.cuh`), which nvcc can't
  fuse into FMAs, in the CPU's order. Same inputs give bit-identical results.
  This part is shared by both precisions.
- **Two CPU evaluators.** The CPU batches the terminals of river subtrees
  (`river::run`) and evaluates the others one at a time (folds before the
  river, showdowns right after an all-in runout, or everything with batching
  off); the two group and round their sums differently. The plan marks each
  terminal with the CPU evaluator that handles it (`Plan::scalar`), and the
  kernels use that one's operation order. Mirroring only the batched one left
  the turn a rounding step off, which regret matching amplified to visible
  regret differences.
- **Terminal sums in f64**, as on the CPU. Only the summation order differs
  (chunked sums and a block scan instead of one sweep), which could move a
  result by one f32 rounding step; in every test so far it hasn't.

Measured on the exact path: GPU and CPU solves agree **bit for bit** (every
strategy, EV and regret identical after 5 iterations) on the river, turn,
suit-isomorphic turn, all-in turn and sampled flop spots, on 24 random river
and 10 random turn spots, and with river batching off: over 46,000
comparisons, none different. `exact_solves_match_cpu_bit_for_bit` keeps
this checked on every test spot, with `f32` and 16-bit storage (raw bits,
scales included), and splitting a tree into groups changes no bit either
(`split_solves_match_whole_solves_*`).

How the fast path is checked, in layers (a bug has to get past each):

1. **The arithmetic** (`double_float.cuh`, the classic error-free
   transformations TwoSum / TwoProd) against host f64 on the GPU: edge cases
   (exact cancellation, a tiny addend next to 1, subnormals, large values),
   random operands spread over 90 binary orders of magnitude, sums of up to
   10,000 values. The accuracy bound is 2^-44 relative plus a few 2^-149
   absolute: f32's exponent range means values below about 1e-31 keep less
   relative precision than f64 would give them, which can't change an f32
   result; above f32's range (3.4e38) values overflow, and the plan bounds
   payoffs and weights so every product stays below 1e22.
2. **Terminal values** of the fast path against the exact path on identical
   reach: at least 99.99% must be bit-identical and none may differ by more
   than 2^-22 of its row's largest value. Measured: 99.997% or more identical
   on the turn and flop spots (all of them on the river spots), worst 7e-8.
3. **Short solves** (5 iterations) against the CPU with the CPU-vs-CPU
   standard of `batched_river_matches_recursion` (5e-5 in strategy, 0.25
   chips in EV). Measured differences: about 1e-7.
4. **Converged solves** against the CPU: both reach the target, iterations
   within 20%, range-level strategies within 0.03 (the standard of
   `compressed_storage_is_close`).

Why per-hand strategies of longer solves aren't compared: CFR amplifies any
rounding difference near an indifference point (a regret that is zero up to
rounding decides between a uniform and a pure strategy). The CPU shows the
same effect against itself: its batched and recursive river paths, which
differ only by rounding, end up 0.18 apart in a hand's strategy and 28 chips
in an EV on these spots after 60 iterations, while converging equally well.

Seeded bugs are caught. In the shared kernels (by the exact path's
bit-for-bit tests): wrong regret discount, dropped tie term, one card's
removal skipped, dealt card not blocking opponent or own hands, isomorphic
cards skipped or unpermuted, chance factor dropped, wrong evaluator rounding,
paired fold not written or with the wrong payoff, chunk and card-list sums
off by one. In the double-single arithmetic (each by 5-6 tests across layers
1-3): TwoSum or quick-TwoSum error term dropped, the f32 add ignoring the low
half, products without cross terms, conversion without the low half, a sign
error in negation.

## Performance

RTX 4060 Laptop GPU (8 GB) and i7-13700H (6P + 8E cores), plugged in,
2026-11-04. `examples/bench` in release mode, stopping at 0.3% of the pot.

Exact flops (16-bit storage; the GPU splits these trees into groups of turn
subtrees to fit its 8 GB), one run each, interleaved:

| Exact flop solve | Iterations | CPU | GPU, `Exact` | GPU, `Fast` |
|---|---|---|---|---|
| `flop-exact-lean Ks7s2d` (the trainer's default tree: all-in raises on turn and river) | 217 (fast: 218) | 127.8 s, 589 ms/iter | 48.6 s, 224 ms/iter | **32.2 s**, 148 ms/iter |
| `flop-exact Ks7s2d` (two-tone, 3x raises) | 184 (fast: 187) | 146.8 s, 798 ms/iter | 56.2 s, 306 ms/iter | **39.0–39.1 s**, 209 ms/iter |
| `flop-exact Ks7d2c` (rainbow, 3x raises) | 174 (fast: 172) | 218.3 s, 1255 ms/iter | 87.4 s, 502 ms/iter | **60.4 s**, 351 ms/iter |

**The fast GPU path solves exact flops 3.6-4.0× faster than the CPU**, the
exact path 2.5-2.6×. Setup per solve (CUDA context, plan, upload) adds 1.2-2.4
s. The exact path took the same iterations to the same root strategy as the
CPU (it is bit-identical); the fast path's root strategy was within 0.2
percentage points.

Smaller trees (f32 storage, whole trees), ranges over 2 runs in the same
session:

| Solve | Iterations | CPU | GPU, `Fast` |
|---|---|---|---|
| turn `Ks7d2c9h` | 168 (fast: 169) | 2.70–2.74 s, 16 ms/iter | **0.865–0.867 s**, 5.1 ms/iter |
| flop-fast `Ks7d2c`, 8 × 6 sampled | 116 | 4.39–4.42 s, 38 ms/iter | **1.15–1.17 s**, 10.0 ms/iter |

That is 3.1× on the turn and 3.8× on the sampled flop (setup: 0.3-0.4 s).
Earlier in the day, before 16-bit storage and splitting existed, the same
GPU runs took 0.87–0.89 s and 1.15–1.16 s, so neither added cost to trees
that don't need it. The CPU measured faster in this session than in that
earlier one (turn 3.03–3.31 s, flop-fast 4.60–5.28 s then); each table
compares runs from one session. Earlier measurements also had the exact path
1.4× slower than fast on these trees, and the river (a 14-node tree that
can't fill the GPU) at 11.7-12.3 ms on the GPU against 9.9-10.3 ms on the
CPU, which is why `Backend::Auto` keeps river roots on the CPU.

The CPU on this branch measured the same as on `main` (interleaved, same
iterations and results: turn 3.08 vs 3.09 s, flop-fast 4.60 vs 4.67 s, river
3000 iterations 0.135 vs 0.132 ms/iter, all within run-to-run noise). Peak
host memory: turn 0.15 GB, flop-fast 0.26 GB, exact flops 2.1-4.8 GB; a GPU
process adds 0.13-0.23 GB (CUDA runtime).

### Where the GPU time goes

Nsight Systems, 60 turn iterations (122 traversals), kernel time per traversal:

| Kernel | First version | Exact now | Fast | Change |
|---|---|---|---|---|
| `showdown_values` (now also the paired folds) | 2.40 ms | 1.61 ms | 0.88 ms | f64 work cut to about one add per summed value; double-single |
| `fold_values` | 1.58 ms | 0.42 ms | 0.14 ms | half the folds paired with showdowns; per-card sums sequential |
| `opponent_split` | 0.59 ms | 0.59 ms | 0.59 ms | |
| `own_update` | 0.38 ms | 0.38 ms | 0.38 ms | |
| `opponent_sum` | 0.22 ms | 0.22 ms | 0.22 ms | |
| `chance_split` + `chance_sum` | 0.06 ms | 0.06 ms | 0.06 ms | |
| kernel time per iteration | 10.5 ms | 6.6 ms | 4.6 ms | |

Nsight Compute on one showdown launch: on the exact path the f64 pipe is 78%
busy and limits the kernel (0.63 instructions per cycle); on the fast path the
busiest pipe is the integer/logic ALU at 30%, 1.93 instructions per cycle, and
the launch takes 0.29 ms instead of 0.46 ms. Folds were never f64-bound to the
same degree (36%); they wait on memory gathers.

What changed in the terminal kernels:

- **f64 operation count.** The GPU does f64 at 2 operations per cycle per SM
  (128 for f32). The first version summed each card's opponent hands with
  warp shuffle reductions and scanned with warp-wide prefix scans: about 5
  f64 adds per summed value. Now each thread sums a contiguous chunk (or one
  card's list) sequentially and only the 256 chunk totals go through a
  block-wide scan or reduction.
- **Folds paired with showdowns.** A fold and a showdown that are both
  children of one of the traverser's nodes share the opponent reach; the
  showdown kernel evaluates the fold from its own sums (the CPU's
  `showdown_batch` grouping), which removes about half of the fold blocks.
- **Double-single accumulators** (fast path): the same sums on the f32 units.

Tried and rejected (measured slower or no better): 512-thread showdown blocks
(10% slower), computing the paired fold inside the showdown's per-hand loop
(30% slower), caching the board's card offsets in shared memory (no gain),
loop strides from `blockDim.x` instead of a compile-time constant (a few
percent slower).

### Exact flops

Nsight Systems, exact flop `Ks7s2d` (3x raises), 30 iterations: the tree
(1.0 million nodes, 16-bit storage) ran as its top plus 2 groups of turn
subtrees. Kernel time per traversal, fast path, before and after the
optimization round of 2026-11-05:

| Kernel | Before | After | Bound by |
|---|---|---|---|
| `showdown_values` | 38.4 ms | 34.8 ms | latency: 50% occupancy (shared memory, 3 blocks per SM), barrier waits |
| `opponent_split_compressed` | 18.5 ms | 18.6 ms | DRAM, 74% of peak |
| `own_update_compressed` | 14.6 ms | 20.3 ms | DRAM, 77% of peak; now also sums fused children |
| `opponent_sum` | 11.6 ms | 2.6 ms | DRAM (child value rows) |
| `fold_values` | 5.1 ms | 5.1 ms | |
| `chance_split` + `chance_sum` | 2.9 ms | 2.9 ms | |
| total | 91 ms | 84 ms | |

Kept:

- **Per-card prefix sums with 4 threads per card** (quarter sums combined
  with warp shuffles) instead of one sequential sweep per card, which left
  most of the block waiting at the barrier: showdowns 5% faster.
- **One entry point per precision** (`*_exact`, `*_fast`): one kernel for
  both held the fast path to the exact path's registers. Fast showdowns
  8-10% faster, 4% less kernel time on the turn and the exact flop.
- **Opponent nodes fused into their parent's update**: an opponent node
  right below one of the traverser's nodes no longer writes a value row only
  for the parent to read it back; the parent sums the node's children
  itself, in `opponent_sum`'s order (bit-identical). 4% less kernel time on
  the exact flop, 2% on the turn. (The parent re-sums a fused child each
  time it reads it, which is why `own_update_compressed` grew.)

Effect on whole solves, measured by alternating the old and new builds on
the same spot (a laptop GPU's clocks drop as it heats, so back-to-back runs
of one build vary by up to 10%): exact flop `Ks7s2d` 224-225 -> 212-213
ms per iteration (-5%), turn 0.95-0.96 -> 0.88-0.90 s (-7%). Root
strategies and iteration counts unchanged; the exact path still matches
the CPU bit for bit.

Tried and rejected, with measurements:

- Opponent weights read from the (cached) reach row instead of shared
  memory, for 4 showdown blocks per SM instead of 3: 3% slower.
- Staging the 16-bit storage kernels' new values in shared memory instead
  of recomputing them in the encoding pass: no faster for opponent splits
  and 6% slower own updates. Counters show why: pass 2's re-reads already
  hit the L2 cache, so DRAM traffic was close to the minimum.
- Caching each hand's first 8 child values (to read fused children once):
  the indexed array lands in local memory; own updates 40% slower.
- CUDA graphs against launch overhead: measured gaps between kernels are
  ~6% of a turn iteration and under 1% of an exact flop's; deferred.

What remains:

- Showdowns are 40% of exact-flop time and latency-bound at 50% occupancy;
  the two cheap fixes tried didn't help, so a gain needs a different scheme
  (e.g. several showdowns per block sharing their board's tables).
- The 16-bit storage kernels are close to the DRAM minimum for their data.

## Three-player solves

`MultiSolver::use_gpu(precision)` moves a 3-way solve to the GPU, with
independent opponents or exact card removal between them (`exact_removal`,
with or without river batching), f32 storage and whole trees; 16-bit
storage is declined with an error and the solve stays on the CPU. A GPU
failure mid-solve restarts the solve on the CPU, as heads-up.
`GpuPrecision::Exact` is bit-identical to the CPU; `Fast` sums the leaves
in double-single.

How a traversal for player `p` maps to kernels (plan: `src/gpu/multi/plan.rs`):

- Two reach-slot spaces, one per opponent (`opponents(p)`). An opponent's
  node splits only that opponent's reach (`multi_opponent_split`); a chance
  node splits both (`multi_chance_split`, one launch per opponent, the
  chance factor on the first one only, as the CPU applies it). A node is
  alive when both of its slots are (`TwoOpponents`).
- A player's average strategy accumulates only in the traversal of the next
  player still in the hand (the plan's `strategy_update` flags).
- Nodes after `p`'s fold are fixed-payoff leaves, even when they have
  children; nothing below them is scheduled.
- The per-node steps are the heads-up kernels' (`traversal.cuh`).

### Leaves: one per lane

The CPU evaluates a leaf with sweeps over the opponents' hands, in hand
order (the compatible mass, `Hands::compat`; exact removal's pair sums for
fixed payoffs, `exact::fold_values`) or in a board's strength order
(`Table::beat`; `exact::showdown`), each traverser hand reading the
running sums at its turn. The order of a sweep depends only on the board
and the folded players, so `multi_leaves.cu` runs one leaf per GPU lane:
the plan groups showdowns into warps of up to 32 on one board with the same
folded players, and fixed-payoff leaves into lanes by their pair of reach
slots (leaves with the same reach share the sweep, as `river::subtree`
groups them), and the 32 lanes of a warp run the same sweep in lockstep,
each on its own reach, with the CPU's arithmetic in the CPU's order. So
the f64 path is bit-identical to the CPU, rounding noise included: a value
that is 0 in exact arithmetic (a hand blocking every remaining combo of an
opponent) comes out as the CPU's noise, bit for bit.

The independent model used one thread block per leaf with tree reductions
in step 2.2; a new random-storage seed showed that this was not
bit-identical after all (6 of 559,296 storage values on the sampled flop,
from leaf values ~1e-16 of their row's scale whose noise depended on the
summation order), so both models now share the lanes.

Exact removal (`exact.cuh`) mirrors `multi::exact`'s two sets (`Pairs`):
the inclusion–exclusion `pairs(h) = tQ'·tR' − Σ_c mQ'[c]·mR'[c] +
Σ_k xQ'(k)·xR'(k)` from per-card sums updated as hands enter. The plan's
`SweepLists` give every seat's list in every order with per-card entry
lists (`by_card`) and a combo-to-position map (`x`), so "is this combo in
the set" is a position test (a set holds a contiguous range of each list).
Per-lane state (two sets of 5 × 52 per-card sums, f64 or double-single)
lives in global scratch, field-major so a warp's accesses to one field are
coalesced; each lane copies its reach in list order.

Checked against the CPU (`src/gpu/multi/engine/tests.rs` and
`tests/leaves.rs`), on every 3-way test spot (rivers, turns, a sampled
flop, all-in runouts, a player folded at the root) under each terminal
model (independent; exact removal with river batching on and off):

- Exact precision, bit for bit: every leaf value of every traversal from
  random storage, root values in every mode, CFR iterations,
  5-iteration solves with their progress reports, a solve to 1% of the pot
  with each model, a GPU solve continuing a CPU one, and 32 random spots
  (property test, either model). No value differed.
- Fast precision: every leaf value within one f32 rounding of the CPU's,
  `|gpu − cpu| <= 2 eps32 · size · c_q · c_r + 1e-12 · size · T_q · T_r`
  per hand (`size` the payoff, `c` an opponent's compatible mass, `T` its
  total reach: the second term bounds the f64 noise of sums that cancel
  terms of that size, which the CPU itself leaves on values that are 0 in
  exact arithmetic), and at least 99.9% identical (measured: all but a few
  hundred of millions). Solves: after 5 iterations exploitability within
  3% and the range-level root strategy within 0.01 of the CPU's (CFR
  amplifies rounding near indifference: a regret that is 0 on the CPU and
  ±1e-19 on the GPU flips an indifferent hand between uniform and pure
  play; with independent opponents nothing differed at all); a solve to 1%
  of the pot in as many iterations (±20%) with the same range strategy.
- A brute force over hand triples (both models, both precisions, small
  spots): within two f32 roundings plus the noise floor.
- The plans' lists against the CPU's `Table::lists`, and Rust models of the
  kernels' sweeps, built only from the plan data, against the CPU
  evaluators bit for bit (`src/gpu/multi/plan/tests/leaves.rs`, no device
  needed); 26 corruptions of the leaf data rejected by `validate`.

Planted bugs: 29, each caught (25 of them by 5 to 12 tests; the
compatible mass not clamped at 0, independent model, by one): the chance
factor, strategy updates, liveness and convergence of step 2.2, and in the
leaves a missing identical combo, folded opponents beating everything or
not blocking, board-blocked hands, ties counted fully or paid two-way,
tied mass of one card, sums missing a card, the Y and cross sums of the
exact model, a set's lower bound, the eq set, the payoffs, batched-river
rounding, overlapping reach rows, every lane on one showdown, stale dead
leaves, the wrong hand order, fast kernels at the exact precision.
memcheck, initcheck and synccheck on all 3-way engine tests and racecheck
on the traversal and leaf tests: no errors.

### Benchmarks

Chart ranges (CO opens, BTN and BB call) on `Js9s4d`, 8 sampled turns and
4 sampled rivers, f32 storage on both sides, 100 iterations with a
convergence check every 20 (`multibench`: `F32=1` vs `GPU=1`, `GPU=1
FAST=1`), runs alternated over two rounds (they agreed within 1%), on the
machine above:

| Terminal model | CPU (20 threads) | GPU exact | GPU fast |
|---|---|---|---|
| exact removal (river batching on) | 711 ms/iteration | 770 (0.92×) | 400 (1.78×) |
| independent | 291 ms/iteration | 140 (2.1×) | 108 (2.7×) |

The exact precision reaches the CPU's convergence bit for bit (0.256% of
the pot after 100 iterations with exact removal, 0.245% independent);
fast: 0.259% and 0.244%. GPU setup: 0.55–0.77 s.

Where the time goes (Nsight, exact removal): the showdown kernel is ~83%,
the fixed payoffs ~15%, the whole traversal ~2%. The leaf kernels are
latency-bound, not compute-bound: each lane carries about 10 KB of state,
reach and stash in global scratch, so a resident grid's working set is far
past L1 and L2 (memory throughput 45 GB/s, achieved occupancy 42%; fewer
lanes are slower: 8192 lanes 1.3×, 3072 2.3×). Fast is only 1.9× faster
than exact for the same reason. The f64 work alone bounds the exact
precision with exact removal at about 150 ms/iteration on this GPU (1/64
f64 rate), whatever the layout. The independent model was faster with the
step 2.2 block kernels (82 ms/iteration), which weren't strictly
bit-identical.

### Faster leaves (step 2.4)

Since 2.4, `GpuPrecision::Fast` may sum in any order (decided
2026-11-06). Each idea was measured on the chart-range flop with exact
removal, 40 iterations, runs alternated (ms/iteration):

| Idea | Fast | Exact | Kept |
|---|---|---|---|
| step 2.3 kernels (baseline) | 392 | 767 | |
| fast sweep (`exact_fast.cuh`): one merged per-card array `G = xqr + xrq − 2 same_card` and one scalar `same − s` instead of five arrays and two scalars, ties with both opponents summed directly over the tied hands instead of a second set, one stash row per hand (`partial − lose · le`): a third of the per-lane state | 368 (vs 389–410 without it, same packing) | — | yes |
| packing live leaves into full warps (in-warp ballots over chunks of 128, then a compaction pass per segment) | ±0 | +1.5% | no |

What limits these kernels (Nsight, Fast): only ~16 of a warp's 32 lanes
are busy, whatever the packing, because the tree has 1,176 boards (all-in
runouts sample their own) and so 3,514 showdown segments of ~18 showdowns
each, and lanes can only share a warp when they share a board's sweep;
and a lane's state still doesn't fit on chip, so the kernel waits on L2
(26–51% achieved occupancy, memory throughput 16–45 GB/s, far from either
peak). Dead leaves cost little (their lanes simply wait), which is why
packing them didn't pay. Estimates for the other layouts (a leaf per warp
with lanes owning cards and the state in registers, or several lanes per
leaf) come out no faster: ~180k warp instructions per leaf. The step 2.4
target for exact removal (well under 10 s, ~150 ms/iteration) is out of
reach of these designs; independent removal meets it (Fast ~104
ms/iteration, about 7 s for the solve).

Planted bugs in the fast sweep: 8, all caught but one (the pair mass not
clamped at 0, which only moves values that are 0 in exact arithmetic,
inside the tolerance's noise floor).

## Running the GPU checks

```bash
cargo test -p solver --features gpu gpu::
```

`gpu::plan` and `gpu::multi::plan` tests need no device: they check the
host-side plans, including Rust references of the terminal formulas (and
the three-player leaf sweeps) built from the plans' data against the CPU
evaluators. `gpu::engine` and
`gpu::multi::engine` tests compare GPU traversals and solves with the CPU
solvers.

```bash
cargo test -p solver --features gpu --no-run --message-format=short
```

Then run the printed test executable under the sanitizer, once per tool:

```bash
compute-sanitizer --tool memcheck --leak-check full <test-exe> gpu::
```

GPU tests skip (with a message) when no NVIDIA driver or device is present.
