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

Status (2026-11-09): complete except the exact-removal speed target, which
is not met (below); it stays `[~]` for that alone.

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

Decided (2026-11-06, for step 4):
4. **Fast may reorder:** 3-way `GpuPrecision::Fast` may give up bit-identity
   and the CPU's summation order for speed (block-parallel or shuffle
   reductions, 2.2-style block kernels), for both card-removal models. It
   stays validated within f32 rounding of the CPU (the 2.3 bound plus its
   f64 noise floor for values that are 0 in exact arithmetic) and by short
   and converged solves against the CPU with agreeing exploitability, as
   heads-up fast mode. `GpuPrecision::Exact` keeps the one-leaf-per-lane
   CPU-order kernels and stays bit-identical as the reference; it gets
   faster only where bit-identity allows (e.g. per-lane state in shared
   memory).

Steps:
1. 3-way plan (host): flattened multi tree, per-traverser schedule with two
   reach slots per node, strategy-updater marks, folded cut-offs, terminal
   lists; validation and mutation tests. `[x]`
   Done (`src/gpu/multi_plan.rs`): one reach-slot space per opponent, an
   opponent-node list per opponent seat, fixed-payoff leaves (wins and every
   node after the traverser's fold) and showdowns per traverser, the CPU
   evaluator's rounding per leaf, and per-opponent showdown lookup tables
   (the heads-up layout) for independent removal. Checked against a walk of
   the CPU traversal on the `multi` test spots and random 3-way spots; about
   100 corruptions rejected; seeded builder bugs caught. Left for later
   steps: exact-removal lookup data (3), and splitting, fused opponent
   nodes and fixed leaves grouped by reach pair (4).
2. Traversal kernels (opponent split for either opponent, two-slot chance
   split, own update, sums) with independent-removal terminals; bit for bit
   against the CPU with `exact_removal` off. `[x]`
   Done (`src/gpu/multi/engine.rs`, `kernels/multi_*.cu`):
   `MultiSolver::use_gpu(GpuPrecision::Exact)` (f32 storage, whole trees,
   f64 terminal sums), bit-identical to the CPU in every test; 18 planted
   bugs caught; sanitizer clean. The per-node steps and the showdown prefix
   sums are now shared with the heads-up kernels (`traversal.cuh`,
   `terminal.cuh`), and the host code is split into shared modules
   (`device.rs`, `shared.rs`, `tables.rs`). Benchmarks in
   `crates/solver/cuda/README.md` ("Three-player solves"). Left for later:
   a double-single fast path (done in 3), `backend` wiring.
3. Exact-removal terminals; against the CPU and a brute force. `[x]`
   Done (`kernels/multi_leaves.cu`, `include/solver/{lanes,exact,independent}.cuh`,
   `src/gpu/multi/{plan,engine}/leaves.rs`): every 3-way leaf runs on one
   GPU lane, a warp's 32 lanes sweeping the same board in lockstep with the
   CPU's arithmetic in its order. Decided on the way (2026-11-06): one
   kernel per accumulator, `GpuPrecision::Exact` (f64) and `Fast`
   (double-single), Fast checked at one f32 rounding per hand
   (`2 eps32 · |pay| · c_q · c_r`, plus an f64 noise floor); river
   batching accepted (its leaves' rounding mirrored); and the independent
   leaves ported to the same lanes after a new seed showed the 2.2 block
   kernels weren't strictly bit-identical (sum-order noise on values that
   are 0 in exact arithmetic). Exact: bit-identical in every test, every
   leaf included; Fast: within the bound; a brute force over hand triples
   agrees; 29 planted bugs caught; sanitizer clean. Chart-range flop with
   exact removal: CPU 711 ms/iteration, GPU exact 770, fast 400
   (independent: 291 / 140 / 108; the 2.2 block kernels took 82). The leaf
   kernels are latency-bound on per-lane scratch (README, "Three-player
   solves"); speeding them up is step 4's first item.
4. Faster leaves (per-lane state in shared memory or a block per leaf with
   in-order scans, keeping bit-identity), 16-bit storage, splitting,
   backend wiring, benchmarks. `[x]`
   So far (2026-11-07): the fast exact-removal sweep (`exact_fast.cuh`,
   392 to 368 ms/iteration; warp packing measured and rejected, README
   "Faster leaves"); 16-bit storage (`compressed_steps.cuh`, shared with
   heads-up; bit-identical); backend wiring: `MultiConfig::backend` and
   the trainer's `[solver] backend` move 3-way solves to the GPU (default
   precision, CPU fallback with the reason), and reviews and study tag
   them (CPU / GPU fast / GPU exact). Done (2026-11-08): splitting trees
   that don't fit, the heads-up segmentation made generic over reach-slot
   spaces (`gpu/segments.rs`), two per traverser with three players, leaf
   groups per segment; bit-identical to whole-tree solves, whole trees no
   slower. The exact-removal speed target is not met (below).

Target: the chart-range 3-way flop (J♠9♠4♦, 8 × 4 sampled, exact removal)
from 47 s on the CPU to well under 10 s. Status (2026-11-07): met for
independent removal (Fast ~104 ms/iteration, about 7 s); not met for exact
removal (Fast ~368 ms/iteration, about 24 s): the one-leaf-per-lane leaves
are limited by the tree's 1,176 boards (warps ~half full) and by per-lane
state that doesn't fit on chip, and the alternatives estimated no better
(`crates/solver/cuda/README.md`, "Faster leaves").

Tried and rejected (2026-11-09):
- Share runout boards (`MultiConfig::share_runouts`, kept in the code as an
  experimental option, off by default, not in the trainer): every chance
  node deals from one shared sample, betting lines included (they alone
  reach all 1,176 boards, so sharing only the runouts couldn't help). Boards
  1,176 to 32 on the chart-range flop. Rejected for accuracy: against
  every-card solves of a small flop (12 seeds, both removal models) root
  values ended about 2.5 times further off (8.7% vs 3.3% of the pot,
  independent removal), since every line then sees the same few turns; on
  2 turn boards (6 seeds each, 3 runs) it was mixed. Speed was never
  benchmarked.
  Numbers: `crates/solver/cuda/README.md`, "Shared runout boards".

## 3. Preflop heuristics (no full preflop solving) `[~]`

Fill the gaps in the preflop charts (9-max, heads-up 3-bet and deeper
lines, other stack depths and rake) with a model rather than a full preflop
solve: solve only the preflop betting tree, and value each pot that sees a
flop as `factor x equity x pot` per player, where the **equity-realization
factor** depends on the hand class, the player's position after the flop,
its role (last raiser or caller) and the pot type.

Decided (2026-11-09):
1. **First target:** 9-max 100bb charts (today 9-max only has
   rule-of-thumb ranges). Heads-up deeper lines and other stacks/rake after.
2. **Delivery:** an offline `poker-trainer solve-preflop` command writes
   chart files in the existing text format; the first set ships built in.
   Every new command is documented in the README for people, not only in
   code.
3. **Charts first, grading later:** preflop decision grading in reviews is
   the round after the model is validated.

Design (`crates/preflop`):
- **Tree:** built with the trainer's own engine, bet menu and spot keys, so
  each decision is a chart spot. Rules keep it small: no open limps (or the
  small blind only), all-in from the second raise on, and a cap on cold
  callers per raise (the big blind and players who already acted can always
  call). 6-max with one cold caller: 10k decisions, covering all 506
  built-in chart spots; 9-max with one: 191k decisions (68M cells, about
  0.5 GB); with two it is 1.2M decisions, too big.
- **Solver:** external-sampling Monte Carlo CFR on real deals (hole cards
  for every seat and a board), so card removal between any number of
  players is exact; 169 hand classes per decision; the average strategy is
  the count of sampled actions. About 2.4M deals/s on 4 threads at 6-max.
- **Checks:** heads-up push-or-fold reaches the known equilibrium; leaf
  payoffs equal the engine's settlement; a convergence measure (what a
  player gains by changing one decision, counted only where an action is
  clearly better on held-out deals) falls from about 1 bb/100 after 4M
  deals to 0 after 400M at 6-max.
- **Charts:** hands that practically never reach a spot use the trainer's
  fallback rules, as imported charts do. A comparison weighs spots by how
  often the reference reaches them.

Steps:
1. Tree, solver, charts, comparison, tests (16 planted bugs caught). `[x]`
   With one flat factor per position and nothing per hand (1.05 in
   position, 0.8 out of position), the 6-max open-raise frequencies
   already land within 2 points of the solver charts from UTG to the
   button (17.5/17.6, 27.9/27.8, 40.1/41.5); the small blind and the
   blind-versus-blind defense are off, which is what measured factors are
   for. With factor 1 everywhere the big blind defends 90% and nobody
   opens enough: the factor is what makes the model work.
2. Calibrate the factors with the postflop solver: per context, solve a
   sample of flops with the ranges of the 6-max solver charts and measure
   what each hand class wins relative to its equity share. `[ ]`
3. Validate on 6-max against the built-in solver charts (open, defend and
   3-bet frequencies per spot, and the per-hand distance). `[ ]`
4. `solve-preflop` command, 9-max 100bb charts built in, README. `[ ]`
5. Then: heads-up deeper lines, other stacks and rake; preflop decision
   grading in the trainer. `[ ]`

## Later

- NN solver (architecture undecided). Then: batched many-spot GPU solving for
  training data.
