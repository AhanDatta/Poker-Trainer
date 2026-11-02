# Poker Trainer

A No-Limit Hold'em trainer for the terminal, written in Rust. Think chess
puzzles, but for NLHE. You're dealt into a hand at a heads-up, 6-max or 9-max
table, play it from preflop to the river against AI opponents with
recognizable styles, and afterwards check every postflop decision you made
against a solver.

```
┌ Your decisions ───────────────┐┌ Flop — GTO ranges ──────────────────────────────────────────┐
│▶ Flop  Bet 1bb (50%)  ?? Blunder││Board 2♠ 4♥ J♦   Pot 2bb                                     │
│  Turn  Check           ✓ Best  ││You Q♦ 6♠ chose Bet 1bb (50%)                                │
│  River Call 2bb        ✓ Good  ││?? Blunder  −0.49bb (24.6% of pot) vs. Check                 │
│                               ││Solver, your hand:                                           │
│                               ││ Check             ██████████ 100.0%  EV   1.04bb ★ best      │
│                               ││ Bet 1bb (50%)     ░░░░░░░░░░   0.0%  EV   0.55bb ◀ you       │
│                               ││ Bet 1.5bb (75%)   ░░░░░░░░░░   0.0%  EV   0.63bb             │
│                               ││Whole range:  Check 34% · Bet 50% 50% · Bet 75% 16%          │
│                               ││Your range at this spot (13×13 grid, colored by action) ...  │
```

## Status

| Feature | State |
|---|---|
| Heads-up, 6-max and 9-max tables, 100bb default | ✅ |
| Full hands from preflop to river vs. AI | ✅ |
| AI styles per seat: Reg, Nit, TAG, LAG, Calling Station, Maniac, or random | ✅ |
| Heads-up pots: exact solver review of every hero decision (EV per action, mix, grade) | ✅ (at any table size) |
| 3-way pots: approximate 3-player CFR review | ✅ approximate |
| 4+-way pots | ❌ not analyzed (analysis resumes once ≤3 players remain) |
| Flop solving: exact (default) or fast | ✅ |
| Villain range in review: GTO charts / villain's actual style range / both | ✅ |
| Rake (no flop, no drop, capped) | ✅ |
| Solver-based preflop charts (heads-up, 6-max), built in | ✅ |
| Preflop grading | ❌ not yet (charts drive the AIs and the starting ranges) |

## Quick start

Requires Rust 1.88 or newer ([rustup](https://rustup.rs); `rustup update`
if you already have it). Everything else is pure Rust: no system libraries,
and it builds on Linux, macOS and Windows.

```bash
git clone https://github.com/AhanDatta/Poker-Trainer
cd Poker-Trainer
cargo run --release
```

Always use `--release`: the solver is far slower in a debug build. To put a
`poker-trainer` command on your `PATH` (in `~/.cargo/bin`) instead:

```bash
cargo install --path crates/tui
```

Optional: build for your exact CPU. Solves were ~10% faster this way on the
development machine. The binary then only runs on CPUs with the same
features, which is fine for a local build.

```bash
RUSTFLAGS="-C target-cpu=native" cargo install --path crates/tui
```

Use a terminal with Unicode and colors (any modern one; on Windows,
Windows Terminal). The review shows its two panels side by side at 110+
columns and stacked otherwise.

Other commands:

```bash
cargo run --release -- init-config            # writes poker-trainer.toml with every setting
cargo run --release -- --config my.toml       # use a specific config
cargo run --release -- --seed 42              # reproducible hands
cargo run --release -- selfplay --hands 3     # headless: scripted hero + printed reviews (speed check)
cargo run --release -- import-charts DIR --format lines --players 6 -o charts.txt   # convert your own ranges
cargo test --workspace                        # unit, property and end-to-end tests
cargo run --release -p solver --example bench -- flop-exact-lean Ks7s2d   # solver benchmarks
cargo run --release -p solver --example compare -- Ks7s2d                 # fast vs exact accuracy
```

### Keys

| Where | Keys |
|---|---|
| Setup | ↑/↓ select · ←/→ change (table size, styles, position, stacks, rake, solver) · Enter play · W save settings · Q quit |
| Table | **F** fold · **C**/**X** check/call · **1–9** bet/raise sizes · **A** largest raise · **N** next hand · **R** solver review · **H** history · **S** settings |
| Review | ↑/↓ choose decision · **Tab** switch GTO / villain's-actual-range solve · **N** next hand · Esc back |

After you fold, the rest of the hand plays out instantly.

### Grades

Each grade is the EV you lost compared with the best action for your exact
hand, as a percentage of the pot at the decision:

| ✓ Best | ✓ Good | ?! Inaccuracy | ? Mistake | ?? Blunder |
|---|---|---|---|---|
| ≤ 1% | ≤ 3% | ≤ 6% | ≤ 12% | > 12% |

EV is measured from the decision onwards, so folding is always worth 0. When
the solver mixes between actions, those actions have (nearly) the same EV, and
either choice grades as Best.

## How it works

```
crates/
  holdem/   cards, 7-card evaluator, ranges, bet menu, N-player engine (side pots, rake)
  solver/   heads-up postflop DCFR solver + approximate 3-player solver (solver::multi)
  trainer/  preflop charts + importers, AI styles, hand controller, review pipeline, config
  tui/      the `poker-trainer` binary (ratatui)
```

### One bet menu for the table and the solver

`holdem::menu` turns a betting situation into the list of allowed actions.
Both the live game and the solvers' tree builders call the same function, so
every action played at the table, by you or the AI, exists in the solved tree,
and grades never depend on mapping an off-tree bet size.

### Heads-up solver

Our own implementation of Discounted CFR (α = 1.5, β = 0, γ = 3, alternating
updates), informed by reading
[b-inary/postflop-solver](https://github.com/b-inary/postflop-solver); no code
from it was copied. Like postflop-solver, the average strategy adds each
iteration's strategy without weighting it by the player's own reach, and
restarts at iterations 4, 16, 64 and 256. With reach-weighted averages, rarely
reached hands' contributions rounded away in 16-bit storage, so big flop
solves stalled between 1% and 0.5% of the pot.

River solutions were cross-checked against postflop-solver on the same games
(strategies within ~1.5 points, per-hand EVs within ~0.5% of the pot).

* **Range vs. range.** Each decision node stores regrets and strategy sums
  per (action, hand). One traversal handles every hand at once by passing the
  opponent's reach vector down the tree.
* **Flat arena with optional 16-bit compressed storage.** Regrets are stored
  as `i16` and strategy sums as `u16`, each with one scale per node. This
  halves memory and memory traffic. It switches on automatically for trees
  over 1 GB.
* **Suit isomorphism.** On two-tone and monotone boards, mirror-image
  turn/river cards are solved once. The hero-hand floor is applied to the
  hand's whole suit orbit so ranges stay symmetric.
* **O(n) showdowns.** Strength-sorted sweeps with exact card removal by
  inclusion–exclusion. River subtrees are evaluated breadth-first, with their
  showdowns and folds swept in batches.
* **Exactness.** All-in runouts are always enumerated. Mid-street roots
  (needed for off-equilibrium re-solves) and rake are supported.

### Approximate 3-player solver (`solver::multi`)

Exact multiway solving is far beyond a desktop machine, so 3-way pots use
DCFR on a three-player game with deliberate approximations:

* The two opponents are treated as independent. Card removal against each
  opponent is exact, but their hands aren't checked against each other. This
  lets showdowns factor into per-opponent sweeps (P(win) = P(beat A) × P(beat B)).
  Ties count half.
* Sampled turn/river cards, and no raises on later streets.
* No equilibrium guarantee: CFR with 3 players has none, though in practice
  it's sensible, as in Pluribus. Convergence is reported as the average
  best-response gain.

A regression test checks that, with the third player already folded, it
reproduces the exact heads-up solver.

### Review pipeline (runs on a background thread while you play)

1. **Flop dealt.** Ranges of everyone in the pot are built from the preflop
   line: charts for the GTO variant, each AI style's own preflop frequencies
   for the "villain's actual range" variant. Your actual hand always gets a
   small weight so the solver has a strategy for it.
2. **Each street.** Depending on how many players are still in, the street
   gets the exact heads-up solver (2 players), the approximate 3-way solver
   (3), or a note (4+). Your decisions are graded once the street's betting
   ends. Ranges are narrowed by the solver's strategy along the actual line
   and carried to the next street, which is re-solved as soon as it's dealt.
3. **Off-equilibrium lines (heads-up).** If someone takes a line the solver
   almost never takes, the node is re-solved from that point with floored
   ranges, so the advice stays meaningful against non-GTO opponents.

### AI opponents

* **Preflop:** chart frequencies for the position and line, adjusted by the
  style. *Looseness* widens or narrows the continuing range along a
  playability order; *aggression* turns calls into raises, or raises into
  calls (limping).
* **Postflop:** Monte-Carlo equity against reads of every opponent's range,
  narrowed after each of their actions. Style thresholds then decide
  value-betting, bluffing (more with draws), calling down, raising,
  slow-playing and sizing.

### Preflop charts

Chart files are plain text sections keyed by the position-labelled preflop
line and the deciding position, e.g. `[UTG:R BTN:C>BB]`, each with `allin:`,
`raise:` and `call:` ranges.

* **Built-in solver charts** (`crates/trainer/charts/`), converted from the
  ranges of [bupticybee/TexasSolverGPU](https://github.com/bupticybee/TexasSolverGPU)
  and redistributed with its owner's
  [permission](https://github.com/bupticybee/TexasSolverGPU/issues/11):
  * **6-max 100bb** (2.5bb opens, 5% rake): 506 spots. Four spots deep in
    5-way all-in lines were left out because the data has aces folding
    there (untrained, practically unreachable nodes).
  * **Heads-up 100bb**: the export only contains lines that reach a flop, so
    only the spots it covers completely are used (SB open, BB vs. limp, BB
    vs. open). 3-bets and beyond use hand-written approximations
    (`hu_100bb.txt`), since there the export's missing all-in lines would
    turn into folds.
  * Hands that never reach a spot under the solver's strategy (e.g. aces
    after flatting) use position-aware rules there, so an AI that
    slow-played them doesn't fold them later.
* **9-max** uses position-aware fallback rules (open percentages by
  position, defense vs. opens, 3-/4-bets, all-ins), which also cover any
  spot no chart has.
* **Your own charts:** `import-charts` converts range collections in either
  supported layout (per-decision line files, or leaf range trees). Point
  `charts_hu` / `charts_6max` / `charts_9max` at the result; its spots
  replace the built-in ones.

## Performance

Measured on the 4-core / 2.1 GHz cloud VM used for development (no
hardware performance counters; your machine is likely faster). The solver
scales almost linearly with cores (3.8× on 4 threads).

| Heads-up single-raised pot (~560 × 970 combos) | Memory | Time |
|---|---|---|
| River, full menu | tiny | ~0.03 s |
| Turn, full menu, exact rivers | ~0.1 GB | ~6–8 s |
| Flop **exact**, two-tone board (isomorphism) | ~1.9 GB | ~1.7 s/iteration, ~4.5 min to 0.5% pot |
| Flop **exact**, rainbow board | ~3.1 GB | ~8–12 min to 0.5% pot |
| Flop **fast** (49 turns × 10 rivers) | ~0.4–0.7 GB | ~1.5–4 min |
| Heads-up **limped** pot, flop exact (~880 × 860 combos) | ~2.7 GB | ~11–12 min |
| 6-max single-raised pot, flop exact (~290 × 220 combos) | <1 GB | a few minutes |

Exact flop trees use one bet size on the turn and river, with all-in as the
only raise there. The flop itself always uses the full menu. Before this
round's optimizations, exact flop solves needed ~9 GB. Trees larger than
`max_memory_gb` (default 5) fall back to fast mode.

### Is fast mode trustworthy?

The `compare` example solves the same flop exactly and in fast mode. It then
measures, at several decision nodes, how much EV a hand loses by following
fast mode's best action, judged by the exact solve:

| Fast-mode sampling | Avg EV lost | Worst hand | Hands with a different best action |
|---|---|---|---|
| 8 turns × 6 rivers (old default) | 1.2–1.5% pot | 25–60% pot | 20–34% |
| 24 turns × 12 rivers | 0.2–0.9% pot | 13–31% pot | 5–26% |
| **49 turns × 10 rivers (current fast mode)** | **0.0–0.2% pot** | 0.4–24% pot | 0–8% |

Sampling turn cards is the main error source, so fast mode deals every turn
and samples only rivers. Exact mode is the default.

## Testing

* **Unit and end-to-end tests:** whole hands at 2/6/9-max tables, including
  background solver reviews.
* **Property-based tests (`proptest`)** check invariants on random inputs and
  shrink failures to a minimal case:
  * evaluator: best-5-of-7, suit-relabeling and monotonicity invariants;
  * range notation round-trips;
  * engine (400 random 2–9-player hands with random stacks and rake): chip
    conservation, legality of every menu action, side-pot and rake
    accounting;
  * solver evaluators against O(n²) brute force;
  * batched vs. recursive and isomorphic vs. full trees give the same
    results;
  * random river solutions are well-formed;
  * AI style adjustments always produce valid frequencies.
* **Debug-build sentinels.** The engine re-checks its invariants after every
  action. The solver checks strategies, reach and values for validity at
  every node. They're free in release builds.

Profiling: `cargo build --profile profiling` (release + debug symbols), then
e.g. `perf record -g target/profiling/examples/bench turn`.

## Configuration

`poker-trainer init-config` writes every option with its default. The most
useful ones:

```toml
players = 6                  # 2 (heads-up), 6, 9
opponent = "random"          # Reg | Nit | TAG | LAG | Calling Station | Maniac | random
random_each_hand = true      # with "random": new style per seat each hand (else per session)
show_opponent_style = false  # reveal styles during the hand (always shown in the review)
hero_position = "alternate"  # alternate (rotate) | random | sb | bb
stack_bb = 100.0
rake_pct = 5.0               # 0 = no rake
rake_cap_bb = 3.0
charts_6max = "my_6max.txt"  # optional: your own charts (also charts_hu, charts_9max)

[solver]
flop_mode = "exact"          # exact | fast
villain_range = "gto"        # gto | actual | both
target_exploitability_pct = 0.5
max_memory_gb = 5.0
threads = 0                  # 0 = all cores
multiway_turn_samples = 8    # 3-way pots
multiway_river_samples = 4

[menu.postflop.flop]
bet = ["33%", "75%"]
raise = ["3x"]

[menu.preflop]
open_bb = 2.5
reraise_mult = [3.5, 2.3]    # 3-bet, 4-bet; then all-in only
```

Custom opponent styles go under `[[custom_profiles]]`, using the same fields as
the presets in `crates/trainer/src/profiles.rs`.

## Known limitations / next steps

* **Reviews take a while on small machines.** Solves run one at a time,
  oldest hand first, while you keep playing; the review screen shows each
  solve's progress and elapsed time. Wide ranges (limped heads-up pots) are
  the slowest.

* **4+-way pots** aren't analyzed.
* **3-way review is approximate:** it ignores card removal between the two
  opponents and uses simplified later streets.
* **Preflop grading** isn't implemented.
* **9-max charts** use fallback rules; no public solver set was found.
* **Exact flop solves still take minutes** on a 4-core machine. The next
  steps would be SIMD-friendlier showdown sweeps and a predictive
  (PDCFR+-style) update.
* **Exploitative review:** node-lock the villain to its AI style's postflop
  tendencies to show the max-EV play against that specific opponent.
* The preflop menu has one raise size per level, while imported charts may
  use slightly different sizes (e.g. 2.6bb opens). Their frequencies are
  used as-is.
