# Roadmap

Work agreed on 2026-11-05, in order. Each step lands with unit and property
tests, benchmarks against the previous state, and an updated README section.
The ground rules of `crates/solver/cuda/README.md` (bulletproof kernels,
modern safe C++, no CUDA dependency in the default build) apply throughout.

Status: `[ ]` to do, `[~]` in progress, `[x]` done.

## 0. Show which solver produced each review `[x]`

- Record the solver behind every review variant as a value stored with the
  review (and so in saved sessions): CPU, GPU fast, GPU exact, CPU 3-way, and
  later the NN solver. A GPU solve that fell back to the CPU says so, with
  the reason.
- Show it on the review screen.

## 1. Optimize the heads-up GPU solver `[ ]`

Single-solve speed (reviews). No changes made only for NN training data yet
(batched many-spot solving waits until the NN design is known).

Measured starting point (exact flop `Ks7s2d`, 3x raises, fast path, kernel
time per traversal, 91 ms total): showdowns 38 ms (latency-bound, 50%
occupancy), 16-bit storage kernels 33 ms (DRAM-bound, 74-77%), opponent sums
12 ms (DRAM-bound), folds 5 ms, chance nodes 3 ms.

1. **One-pass 16-bit storage kernels.** Keep pass 1's new values in shared
   memory for nodes that fit, instead of recomputing them in pass 2. Expected
   ~10% of an exact flop. Must stay bit-identical; the two-pass path stays
   for nodes too big for shared memory, and tests force both.
2. **Separate exact and fast entry points** for the terminal kernels, so the
   fast path isn't held to the exact path's register count. Expected 5-15% of
   showdown time.
3. **Fuse opponent sums into the parent's update** (an opponent node's value
   row is written only for its parent to read it back). Expected 5-10%.
   Changes the traversal structure: needs its own validation and tests.
4. **Launch overhead** (CUDA graphs or fewer, merged launches) for turn and
   river trees, which are launch-bound.
5. Re-profile after each step; record kept and rejected ideas with numbers
   in `crates/solver/cuda/README.md`.

Done when the ideas above are measured and kept or rejected, and the exact
path is still bit-identical to the CPU.

## 2. Three-way solves on the GPU `[ ]`

Port `solver::multi` (three players, sampled runouts, exact or independent
card removal between the two opponents) with the heads-up port's workflow:
plan and validation first, then kernels level by level, then terminals,
each checked bit for bit against the CPU solver (exact path) before any fast
path. 4+-way pots stay out of scope (rare, approximate, hard to validate).

Expected shape (to confirm against the code when starting):
- Plan: three traversers; reach for the two non-traversing players per slot.
- Terminals: showdowns and folds summed over pairs of opponent hands with
  inclusion-exclusion over shared cards (exact removal), or factored
  per-opponent sweeps (independent). Compute-heavy, which suits the GPU.
- Memory: reuse 16-bit storage and tree splitting.
- Backend setting: 3-way solves follow `backend` like heads-up ones.

Target: the chart-range 3-way flop (J♠9♠4♦, 8 × 4 sampled, exact removal)
from 47 s on the CPU to well under 10 s.

## 3. Preflop heuristics (no full preflop solving) `[ ]`

Fill the gaps in the preflop charts (heads-up 3-bet and deeper lines,
9-max, other stack depths and rake) with a model rather than a full preflop
solve:

- Solve only the preflop betting tree; value each pot that reaches the flop
  as equity × an **equity-realization factor** by hand class, position
  (in/out of position), and line (raiser/caller, pot type), minus rake.
- Calibrate the realization factors with the postflop solver: solve a sample
  of flops per preflop line and measure how much equity each hand class
  realizes.
- Check the model against the built-in 6-max solver charts before trusting
  it where no charts exist.
- Then: preflop decision grading in the trainer.

## Later

- NN solver (architecture undecided). Then: batched many-spot GPU solving for
  training data.
