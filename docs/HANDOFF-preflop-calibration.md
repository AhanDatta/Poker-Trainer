# Handoff: preflop model, calibration onward (ROADMAP step 3)

This document is for a new Claude Code session picking up the preflop work in
the Poker Trainer. It covers where things stand, the user's rules, the plan
and every detail needed to continue. Read it fully before touching code, then
read `docs/ROADMAP.md` (section 3) on the `preflop-model` branch.

Written 2026-11-13. State as of commit `817aa83`.

---

## 1. Where to work

| What | Where |
|---|---|
| Repository | `AhanDatta/Poker-Trainer` on GitHub; a Rust workspace |
| Work branch | `preflop-model`, checked out in the worktree `C:\Users\ahand\Desktop\Coding\Poker\poker-trainer-preflop` |
| Its base | `gpu-port` at `b50a109` (`gpu-port` is pushed to `origin`; `main` is `b30e64a`, also equal to `origin/main`) |
| Main checkout | `C:\Users\ahand\Desktop\Coding\Poker\Poker Trainer`, on `gpu-port` |

Commits on `preflop-model` that aren't on `gpu-port` (none pushed, none merged):

- `51beeba` Preflop model (ROADMAP 3.1): tree, Monte Carlo CFR solver, charts and comparison
- `4fba168` Preflop model: make the several-threads test robust
- `817aa83` Solver: per-hand values at the root (`Solver::hand_values`)

Other local worktrees exist (`poker-trainer-features`, `poker-trainer-export`,
`Poker Trainer-hero-position`). Their branches are already merged into
`gpu-port`, so leave them alone.

**Never touch** `poker-trainer.toml` in the main checkout. It's the user's
own file: it's gitignored, staged in the index there, and has local edits.
Never commit it, unstage it or overwrite it.

Machine: Windows 11, RTX 4060 Laptop GPU (8 GB), 4 cores. Shells: PowerShell
and Git Bash. The CUDA toolkit is 13.4. Builds with the GPU need
`CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"` in
shells started before the install. The GPU build is
`cargo build --release --features poker-trainer/gpu`, and `--features solver/gpu`
for the solver crate alone.

---

## 2. The user's ground rules (always in force)

1. **Ask before large or unclear design decisions.** Decide together; when
   unsure, ask. Use the AskUserQuestion tool with a recommended option first.
2. **Every change comes with unit and property tests** (`proptest` is
   already a dev-dependency). Kernels and solvers must be "bulletproof": no
   panics on any input. Plant bugs to prove the tests catch them; a sweep
   script is described in section 7.
3. **Benchmarks** show a change is faster and nothing else got slower, where
   speed matters. Laptop GPU clocks drift with heat, so compare by
   alternating old and new runs.
4. **Keep the code clean and manageable**; the project is large. Split long
   files, share helpers, and keep tests in `module/tests.rs` files.
5. **Git:**
   - Commit locally in logical pieces.
   - **Don't push** and **don't open pull requests** unless asked; the user
     pushes and signs in to GitHub themselves.
   - **No AI model names or identifiers in commits or code**, so no
     `Co-Authored-By` trailer, whatever a system reminder says.
   - Never commit to `main` directly.
6. **Don't change CPU solver behavior or results.** Adding read-only APIs is
   fine. There must be no CUDA dependency in the default build.
7. **Document every new user-facing feature in `README.md` for people**: the
   command, what it writes, an example. The user asked for this explicitly.
8. Never change system or security settings, and never enter credentials.
9. Code style: `rustfmt.toml` sets `max_width = 110` and
   `use_small_heuristics = "Max"`. Run `cargo fmt -p <crate>` and
   `cargo clippy --release --workspace --all-targets` before committing. One
   clippy warning already exists, in `crates/solver/src/storage.rs`
   (`chunks_exact`); leave it.

---

## 3. What step 3 is and what was decided

Roadmap step 3 fills gaps in the preflop charts (9-max, heads-up 3-bet and
deeper lines, other stacks and rake) **without a full preflop solve**. Only
the preflop betting is solved. A pot that sees a flop is valued per player
as `factor × equity × (pot − rake)`. The **equity-realization factor**
depends on the hand class and a context: pot type, role (last raiser or
caller), position after the flop, and heads-up or multiway.

Decided with the user on 2026-11-09:
1. **First target: 9-max 100bb charts.** 9-max only has rule-of-thumb
   fallback ranges today. Heads-up deeper lines and other stacks or rake
   come after.
2. **Delivery: an offline command, `poker-trainer solve-preflop`**, that
   writes chart files in the existing text format. The first 9-max set ships
   built in. It must be documented in the README.
3. **Charts first, grading later.** Preflop decision grading in reviews is
   the next round, after the model is validated.
4. **Calibration approved:** "Yes, calibrate", including a few hours of GPU
   time.

Earlier context, already done: the GPU port of the heads-up and 3-way
postflop solvers (roadmap steps 0-2), CSV hand export (`poker-trainer export`)
and hero position choice (`hero_position`). Shared runout boards for 3-way
solves were tried and **rejected**. The `share_runouts` option stays off and
unused. Don't revive it.

---

## 4. What exists: `crates/preflop` (step 3.1, done)

A workspace member that depends on `holdem` and `trainer`. Calibration will
also need `solver`: add `solver.workspace = true`, plus a feature
`gpu = ["solver/gpu", "trainer/gpu"]`.

| Module | Contents |
|---|---|
| `spec.rs` | `GameSpec { players, stack_bb, rake_rate, rake_cap_bb, menu: BetMenu, rules: TreeRules }`; `TreeRules { limp: Limp::{Never, SmallBlind, Any}, allin_after_raises (default 2), max_cold_callers (default usize::MAX), max_nodes (default 5M) }`; `BB = 100` chips; `MAX_PLAYERS = 9`; `validate()`, `rake(pot)` |
| `tree.rs` | `Tree::build(&spec)`: the preflop betting tree, built with the trainer's engine (`holdem::HandState`), its bet menu and its spot keys (`trainer::preflop::spot_key`), so each decision's `key` is a chart spot such as `"UTG:R BTN:C>BB"`. `Node::{Decision, Leaf}`. `Decision { seat (0 = button), key, avail, children: [u32; 4] by PreAction index, commit, slot }`. `Leaf { kind: LeafKind::{Won(seat), AllIn, Flop}, pot, rake, commit[9], live (bitmask), raises, aggressor }`. `tree.find(key)` |
| `realize.rs` | `Context { pot: PotKind::{Limped, Raised, ThreeBet, FourBet}, role: Role::{Aggressor, Caller}, position: Position::{InPosition, OutOfPosition, Middle}, multiway }`, with 48 contexts: `index()`, `from_index()`, `all()`, and `Context::of(leaf, seat, players)`. After the flop, seat 1 (SB) acts first and the button last. `Realization` holds 48 × 169 factors: `neutral()`, `positional(ip, oop, middle)`, `factor(ctx_index, class)`, `factors(ctx)`, `set(ctx, [f32; 169])`. `includes_rake: bool`, false by default, means the factor applies to `pot − rake(preflop pot)` |
| `deal.rs`, `rng.rs` | Real deals: hole cards for every seat plus a board, from xoshiro256++. `Deal::share(live, seat)` is the seat's showdown share |
| `solver.rs` | External-sampling Monte Carlo CFR. `Solver::new(tree, model, Options { threads, seed, regret_floor, warmup_epochs })`, `run(deals)` (one epoch), `average(decision, class) -> (Freqs, samples)`. Lock-free atomics; one thread is reproducible from the seed. About 2.4M deals/s on 4 threads at 6-max |
| `solver/diagnose.rs` | `diagnose(deals, seed) -> Diagnosis { ev per seat, deviation_gain per seat }`. The gain counts only actions that are clearly better on held-out deals, so it's a convergence measure that never overstates |
| `charts.rs` | `to_charts(&solver, min_samples) -> ChartSet` (classes with too few samples use the trainer's `fallback_freqs`), `compare(tree, model, reference) -> Vec<SpotDiff>` (reach-weighted, using the reference's ranges), `mean_distance`, `first_in_frequencies` |
| `examples/treesize.rs` | Tree sizes per table size and rules |
| `examples/presolve.rs` | `cargo run --release -p preflop --example presolve -- [players] [million deals] [threads] [ip] [oop]`: solves with flat positional factors, then prints the distance to the built-in charts, per-position bb/100, the deviation gain, open frequencies and the most-reached spots |

Tree sizes (no limps, all-in from the 2nd raise):
- 6-max with `max_cold_callers = 1`: 10,096 decisions, covering **all 506
  built-in 6-max chart spots**.
- 9-max with 1: 191k decisions, 68M cells, about 0.5 GB.
- 9-max with 2: 1.2M decisions, too big.

So **9-max uses `max_cold_callers = 1`**. The big blind, and players who
already acted, can always call.

Tests: 34 in the crate, plus `Solver::hand_values_add_up_to_root_values` in
`solver`. They cover:
- push-or-fold heads-up at 10bb reaching the known equilibrium (about 58%
  jam, 37% call);
- leaf payoffs equal to the engine's settlement;
- a property test that tree walks match an independent replay;
- charts round-tripping through the text format;
- comparisons and context invariants.

All 16 planted bugs are caught.

Results so far (6-max, rake 5% capped at 3bb, flat factors):

| Factors | UTG open | CO open | BTN open | SB open | Mean distance to charts |
|---|---|---|---|---|---|
| 1 everywhere | 13.6% | 19.7% | 26.0% | 55.0% | 0.165 |
| 1.05 IP, 0.8 OOP | 17.5% | 27.9% | 40.1% | 32.0% | 0.12 |
| Built-in solver charts | 17.6% | 27.8% | 41.5% | 42.7% | — |

The small blind and blind-versus-blind defense are off. Fixing them is what
measured, per-hand factors are for. The deviation gain is about 1 bb/100
after 4M deals and 0 after 40M.

New in `solver` (`817aa83`): `Solver::hand_values(p) -> Vec<f32>`. It gives
each of player `p`'s hands, in `hands().combos[p]` order, its value under
the average strategies: chips back from the pot minus chips put in after the
root, against the opponent's root range with card removal. It's read-only,
on the CPU; it works after a GPU solve because the storage is downloaded to
the host when `solve` returns.

---

## 5. The plan, step by step

### 3.2 Calibrate realization factors (next)

**Goal:** a factor for every `(context, hand class)`, measured with the
heads-up postflop solver on 6-max 100bb spots, saved as a data file in the
repo.

**Measurement:**
- `V(h)` = `solver.hand_values(p)[h]` after solving the spot with the real
  postflop menu.
- `V0(h)` = the same from a **check-down** solve: same board, ranges, pot and
  rake, but a menu with no bets. Every street gets `bet = []` and
  `raise = []`, with `add_allin_threshold = 0`. Every node then has only
  "check", one traversal suffices, and `V0 = equity × (pot − rake)` with the
  solver's own rake and card removal.
- The factor of class `c` in context `x` is the ratio of sums over spots,
  flops and the combos of `c`: `R[x][c] = Σ w · V / Σ w · V0`, where `w` is
  the spot's reach. Keep the model's `includes_rake = false`; it's
  consistent because `V0` uses the pre-bet pot's rake.

**Spots:**
- Take the **heads-up flop leaves** (`LeafKind::Flop`, 2 live players) of the
  6-max tree: `GameSpec::new(6).with_rake(0.05, 3.0)` with
  `max_cold_callers = 1`.
- Walk them like `charts::compare` does, with the **built-in 6-max charts**
  (`ChartSet::builtin(6)`) as the policy. For each leaf you get its reach and
  each live seat's class weights.
- Group the leaves by spot type: pot type × whether the aggressor is in
  position. There are 6 types: SRP, 3-bet and 4-bet, each with the
  aggressor IP or OOP.
- Pick the most-reached leaves per type, for example the top 4. Each solve
  measures 2 contexts: the aggressor's and the caller's.
- Multiway contexts aren't measured. **Use the heads-up factors for them**
  for now, with Middle as the average of IP and OOP. Say this to the user
  and in the roadmap; the 3-way solver can calibrate them later.
- Limped contexts don't occur at 6-max or 9-max with `Limp::Never`. Fill
  them from the Raised caller factors, and note it.

**Ranges for each solve:**
- Expand class weights to 1326 combo weights.
- Add a **small floor**, about 1% of the max weight, to every class so all
  169 classes get a value; a hand with no weight would have no average
  strategy.
- Set `prune = 0` and zero the combos that conflict with the board.
- The floor barely changes the equilibrium, and the floor hands play close
  to a best response, which is the right value for "should this hand join
  the range".

**Solve config:** mirror the trainer's review config
(`trainer::review::hu_config`, which is `pub(crate)`, so copy the values):
- `SolveConfig::new(board, ranges, root, PostflopMenu::default())`, where
  root is `BettingState::street_start(Street::Flop, pot, [behind, behind], 100)`
  and `behind = stack − commit`. Player 0 is OOP.
- `future_menu`: turn `["66%"]` with raise `["allin"]`, river `["75%"]` with
  `["allin"]`, `max_raises = 2`.
- `big_blind = 100`; rake 5% capped at 300 chips.
- `chance = Exact`, or, if time is short, `Sampled { turn: 49, river: 10 }`,
  the trainer's fast mode.
- `backend = Backend::Gpu` (fast precision), with about 0.5–1% exploitability
  target and `max_iterations` about 400.
- Solves with wide ranges (every class floored) take roughly 30–60 s each on
  the GPU for single-raised pots; 3-bet and 4-bet pots are much faster.
  Budget about 2–3 hours, for example 4 spots × 12 flops × 6 types ≈ 300
  solves.

**Flops:** random uniform flops per spot from a fixed seed, recorded so runs
are reproducible.

**Runner:** an example or bin in `crates/preflop`, such as
`examples/calibrate.rs`, built with `--features gpu`:
- **Checkpoint** every `(spot, flop)` result to a JSON-lines file: spot key,
  flop, seats, contexts, per-class sums of `w·V` and `w·V0`, and combo
  counts. A restart then skips finished solves.
- Run it in the background and give progress lines.
- A separate step aggregates the checkpoint into factors.
- Report a standard error per context and class, for example by a jackknife
  over flops. Consider shrinking noisy classes toward the context mean, or
  smoothing by equity; ask the user if unsure.

**Data file:** `crates/preflop/data/realization_6max_100bb.txt`, a plain text
format with one section per context and 169 values in grid order.
- Add `Realization::to_text` and `Realization::parse` with round-trip tests.
- Hold invalid input to the "no panics" rule: return errors.

**Tests for 3.2:**
- A check-only menu gives factors of exactly 1, a strong property test of
  the whole pipeline.
- Spot discovery: reach sums, both seats live, contexts correct, ranges
  equal to `trainer::preflop::line_range` for the same line.
- Combo expansion and the floor: no board conflicts, every class present.
- Checkpoint round trip and resume skipping.
- Aggregation math on synthetic data.
- Small sampled-chance CPU solves in tests; keep them fast.
- Plant bugs; see section 7.

**Before the long run:** tell the user it's starting and roughly how long it
takes (they approved hours). Run it with `run_in_background`, don't poll,
and check results when it finishes.

### 3.3 Validate on 6-max

- Solve 6-max with the calibrated factors, for example via `presolve`, and
  compare with the built-in charts:
  - open, defend and 3-bet frequencies per spot;
  - `mean_distance`, which was 0.12 with the flat factors;
  - SB opening and BB defense in particular.
- Check that the deviation gain is near 0.
- Report a table to the user, and record kept and rejected ideas with
  numbers in the roadmap.
- If the fit is poor, investigate (factor noise, the floor, the solve
  config) before moving on, and ask the user if a design change is needed.

### 3.4 The `solve-preflop` command, 9-max charts built in, README

- New subcommand in `crates/tui/src/main.rs`, a clap `Cmd` variant next to
  `ImportCharts` and `Export`. For example:

  ```
  poker-trainer solve-preflop --players 9 --stack 100 --rake 5 --rake-cap 3 --deals 400 -o charts_9max.txt
  ```

  - It writes a chart file via `ChartSet::to_text`, with a header naming
    the settings, factors file and seed.
  - It prints progress and the convergence diagnosis.
  - Use 9-max tree rules with `max_cold_callers = 1`.
  - `tui` then depends on `preflop`.
- Generate `crates/trainer/charts/9max_100bb_model.txt` and make
  `ChartSet::builtin(7..=9)` load it (today it returns no charts and uses the
  fallback rules). Update the tests in `trainer/src/preflop.rs` that assert
  9-max is empty.
- Check that 9-max reasonably extends 6-max: comparable opens from
  HJ/CO/BTN/SB, tighter early positions.
- README:
  - a section with the command, its options, what it writes, how long it
    takes, and how to point `charts_9max` at a custom file;
  - update the 9-max line under "Known limitations" and the feature table;
  - add `preflop/` to the crate list (already done).

### 3.5 Afterwards (ask the user before starting)

Heads-up deeper lines (3-bet and beyond, where the charts are hand-written
today), other stack depths and rake, multiway calibration with the 3-way
solver, and then preflop decision grading in reviews.

---

## 6. Useful facts about the existing code

**Preflop charts (`crates/trainer/src/preflop.rs`):**
- `ChartSet { charts: HashMap<String, Chart>, players }`;
  `Chart { freqs: Vec<[f32; 4]> }` per class, ordered fold, passive, raise,
  all-in.
- `ChartSet::builtin(n)` gives:
  - heads-up: the solver charts for the first decisions plus hand-written
    3-bets;
  - 3–6 players: the 6-max solver charts (2.5bb opens, 5% rake, 506 spots);
  - 7+ players: no charts.
- `fallback_freqs(key, class, players)` is the rule-based fallback.
- `spot_key(decisions, position)` builds keys: folds of players who never
  entered are omitted, and the format is `"POS:A POS:A>DECIDER"`.
- `line_range` turns a line into a range; `Availability::remap` maps
  frequencies onto the available actions.

**Elsewhere:**
- Positions are in `holdem::game`: `position_name(n, offset)`,
  `position_offset`, `positions(n)`. Seat 0 is the button; heads-up the
  button is the SB.
- `holdem::combo`:
  - `HandClass::grid_index()` (169 classes; AA = 0, AKs = 1, AKo = 13,
    22 = 168), `HandClass::all()`, `combos()`, `num_combos()`;
  - `COMBO_CLASS[combo]`; `combo_cards(index)`.
- Postflop solver API (`crates/solver/src/lib.rs`): `SolveConfig`,
  `Solver::new`, `solve(&SolveOptions, |p| true)`, `root_values()`,
  `hand_values(p)`, `query(path)`, `hands()` (`combos[p]`, `weights[p]`).
  `SolveConfig.backend = Backend::{Cpu, Gpu, Auto}`, and `solved_on()` says
  where it ran.
- `trainer::strength::class_equity()` is each class's all-in equity against
  a random hand, a cheap sanity reference.
- Trainer config defaults: no rake (`rake_pct = 0`, `rake_cap_bb = 3`) and
  the `cpu` backend. Charts were made with 5% rake, so the model uses 5%
  capped at 3bb to match them.

---

## 7. Workflow tips

- **Planted-bug sweep:** a Python script plants one bug at a time (a source
  string replaced), runs `cargo test --release -p preflop --lib`, records
  caught or missed, and **always restores the file**, in a
  `try`/`finally`. The previous one lived in a temporary scratchpad; write a
  new one in your own scratchpad (not in the repo) for the calibration code.
  Aim to catch every planted bug, and add tests for misses.
- **Flaky tests:** sampling tests must be robust. One test failed about 1 run
  in 10 until it compared convergence instead of exact mixes. Run new
  sampling tests 20–40 times in a loop before committing.
- Long runs (calibration, big solves) go in the background, and the result
  is checked once it finishes.
- Python on Windows writes CRLF unless you pass `newline="\n"`. Git's
  `autocrlf` handles either.
- Keep the roadmap (`docs/ROADMAP.md`, section 3) updated as steps land,
  with numbers.
- When done with a step, give the user a short report:
  - what is true now;
  - the numbers;
  - what isn't done or verified;
  - the commit hashes;
  - that nothing is pushed.
