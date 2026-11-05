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

## 1. Optimize the heads-up GPU solver `[x]`

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
   river trees. Measured: gaps between kernels are ~6% of a turn
   iteration and under 1% of an exact flop's. CUDA graphs would need the
   per-iteration discounts moved into device memory; deferred as low value.
5. Re-profile after each step; record kept and rejected ideas with numbers
   in `crates/solver/cuda/README.md`.

Done when the ideas above are measured and kept or rejected, and the exact
path is still bit-identical to the CPU.

Outcome (2026-11-05): 2 and 3 kept, plus 4-thread per-card prefix sums in
showdowns; 1 rejected (pass 2 already hits L2), 4 deferred. Whole solves
5% faster on exact flops, 7% on the turn. Details and numbers in
`crates/solver/cuda/README.md` ("Exact flops").

## 2. Three-way solves on the GPU `[~]`

Port `solver::multi` (three players, sampled runouts, exact or independent
card removal between the two opponents) with the heads-up port's workflow:
plan and validation first, then kernels level by level, then terminals,
each checked against the CPU solver before any fast path. 4+-way pots stay
out of scope (rare, approximate, hard to validate).

What the CPU solver does that the port must mirror (`src/multi/`):
- A traversal for player `p` carries **two** opponent reach vectors. At an
  opponent's node only that opponent's reach splits (the other passes
  through); at a chance node both are copied with blocked hands zeroed (the
  chance factor applied to one of them only, since values are bilinear).
- A player's average strategy is accumulated in the traversal of the **next
  player still in the hand** (`strategy_updater`), not in "the opponent's"
  traversal.
- Once `p` has folded, its subtree is a fixed payoff times the opponent pair
  mass (no traversal below); a node either opponent can't reach is dead.
- Terminals: `pay_all` (fixed payoff × mass of compatible opponent pairs),
  showdowns with **exact removal** (inclusion-exclusion over shared cards,
  `exact.rs`: per-card sums plus per-card hand lists, terminals on one board
  batched 4 lanes at a time) or **independent** opponents
  (`P(win) = P(beat q) · P(beat r)` from per-opponent compat/beat sums).
- River subtrees evaluated breadth-first with batched terminals
  (`river.rs`), as heads-up; fixed-payoff leaves grouped by reach pair.

GPU shape:
- Plan: per traverser, two reach slots per node (one per opponent: only the
  acting opponent's or a chance node's children get fresh slots), the
  strategy-updater mark per opponent node, folded-subtree cut-offs.
- Kernels: opponent split/sum and own update generalize directly; chance
  split writes two slots; terminals are the new work, with exact removal
  the hard part (pair sums per hand: per-card statistics, then a gather per
  hand, as in the heads-up showdown tables).
- Memory: 16-bit storage and tree splitting carry over.
- Backend: 3-way solves follow `backend` like heads-up ones.

Decided (2026-11-05):
1. **Exactness:** bit-identical to the CPU for the traversal kernels and
   independent-removal terminals; exact-removal terminals within f32
   rounding of the CPU, plus a brute-force reference over hand triples.
2. **Order:** independent removal first (proves the 3-player traversal),
   then exact removal.
3. Runout sampling defaults stay as they are until there are GPU numbers.

Steps:
1. 3-way plan (host): flattened multi tree, per-traverser schedule with two
   reach slots per node, strategy-updater marks, folded cut-offs, terminal
   lists; validation and mutation tests. `[ ]`
2. Traversal kernels (opponent split for either opponent, two-slot chance
   split, own update, sums) with independent-removal terminals; bit for bit
   against the CPU with `exact_removal` off. `[ ]`
3. Exact-removal terminals; against the CPU and a brute force. `[ ]`
4. River batching parity, 16-bit storage, splitting, backend wiring,
   benchmarks. `[ ]`

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
