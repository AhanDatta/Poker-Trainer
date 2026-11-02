# Poker Trainer

A No-Limit Hold'em trainer for the terminal, written in Rust. Think chess
puzzles, but for NLHE. You're dealt into a hand, play it from preflop to the
river against an AI opponent with a recognizable style, and afterwards check
every postflop decision you made against a solver.

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
| Heads-up (SB/button vs BB), 100bb default | ✅ |
| Full hands from preflop to river vs. AI | ✅ |
| AI styles: Reg, Nit, TAG, LAG, Calling Station, Maniac, or random | ✅ |
| Postflop solver review of every hero decision (EV per action, mix, grade) | ✅ |
| Villain range in review: GTO charts / villain's actual style range / both | ✅ |
| Flop solving: fast (sampled runouts) or exact | ✅ |
| Preflop grading | ❌ not yet (charts drive the AIs and the starting ranges) |
| 3–9 players | ❌ planned. The engine already handles N players and side pots; the AI, charts and review are heads-up only |

## Quick start

Requires Rust 1.88 or newer (`rustup update` if needed).

```bash
cargo run --release
```

Always use `--release`: the solver is far slower in a debug build.

Other commands:

```bash
cargo run --release -- init-config            # writes poker-trainer.toml with every setting
cargo run --release -- --config my.toml       # use a specific config
cargo run --release -- --seed 42              # reproducible hands
cargo run --release -- selfplay --hands 3     # headless: scripted hero + printed reviews (speed check)
cargo test --workspace                        # unit + end-to-end tests
cargo run --release -p solver --example bench -- turn   # solver benchmarks (river|turn|flop-fast|flop-fast-simple|flop-exact)
```

### Keys

| Where | Keys |
|---|---|
| Setup | ↑/↓ select · ←/→ change · Enter play · W save settings · Q quit |
| Table | **F** fold · **C**/**X** check/call · **1–9** bet/raise sizes · **A** largest raise · **N** next hand · **R** solver review · **H** history · **S** settings |
| Review | ↑/↓ choose decision · **Tab** switch GTO / villain's-actual-range solve · **N** next hand · Esc back |

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
  holdem/   cards, 7-card evaluator, 1326 combos / 169 classes, ranges, bet menu, N-player game engine
  solver/   heads-up postflop Discounted-CFR solver
  trainer/  preflop charts, AI styles, hand controller, background review pipeline, config
  tui/      the `poker-trainer` binary (ratatui)
```

### One bet menu for the table and the solver

`holdem::menu` turns a betting situation into the list of allowed actions.
Both the live game and the solver's tree builder call the same function, so
every action played at the table, by you or the AI, exists in the solved tree,
and grades never depend on mapping an off-tree bet size. Sizes are
configurable per street, including pot-fraction bets, ×-multiplier raises, the
all-in threshold, a raise cap and merging of near-identical sizes.

### Solver

Our own implementation. The approach follows the published DCFR algorithm and
was informed by reading
[b-inary/postflop-solver](https://github.com/b-inary/postflop-solver); no code
from it was copied.

* **Range vs. range DCFR** (α = 1.5, β = 0, γ = 2, alternating updates).
  Each node stores regrets and strategy sums per (action, hand). One recursive
  pass passes the opponent's reach vector down and returns the traverser's
  counterfactual values for every hand at once.
* **O(n) terminal evaluation.** Showdowns sweep both players' hands sorted by
  strength. Card removal is exact, by inclusion–exclusion over the 52 cards.
* **Chance nodes.** Hands containing a dealt card are zeroed, and reach is
  scaled by 1/45 (turn) or 1/44 (river). Fast mode uses a stratified sample of
  turn/river cards (suits in proportion, ranks spread) with a matching scale
  factor. All-in runouts are always enumerated exactly.
* **Mid-street roots.** These handle unequal commitments and ties, which the
  off-equilibrium re-solves need.
* **Queries** return the average strategy and the EV of every action for every
  hand at any node, plus range narrowing along a path.
* Tests cover a polarized river toy game (it converges to the textbook 50%
  bluff / 50% call equilibrium with bluff EV = 0), an exact 2-outer all-in
  equity, ties, convergence on a real turn spot, and fast mode.

### Review pipeline (runs on a background thread while you play)

1. **Flop dealt.** Both ranges are built from the preflop line: the GTO
   variant uses the charts; the "villain's actual range" variant uses the AI
   style's own preflop frequencies. Your actual hand always gets a small
   weight so the solver has a strategy for it, then the flop solve starts.
2. **A street's betting ends.** Each of your decisions on that street is
   graded. Both ranges are narrowed by the solver's strategy along the actual
   line, and the next street is solved as soon as it's dealt (re-solving with
   the full bet menu).
3. **Off-equilibrium lines.** If you or the villain take a line the solver
   almost never takes, the node is re-solved from that point with floored
   ranges, so the advice stays meaningful against a non-GTO opponent.

### AI opponents

* **Preflop:** chart frequencies adjusted by the style. *Looseness* widens or
  narrows the continuing range along a preflop strength order, and
  *aggression* turns calls into raises or raises into calls (limping).
* **Postflop:** Monte-Carlo equity against a read of your range (your chart
  range, narrowed after each of your postflop actions with hand-strength and
  draw heuristics), then style thresholds for value-betting, bluffing (more
  with draws), calling down, raising, slow-playing and bet sizing.

### Preflop charts

`crates/trainer/charts/hu_100bb.txt` holds **hand-written approximations** of
heads-up 100bb strategies. They are not solver output. The format is plain
text sections keyed by the preflop line (`[-]`, `[R]`, `[RR]`, `[C]`, …) with
`allin:` / `raise:` / `call:` ranges. Point `charts_file` in the config at your
own file to use charts you trust.

## Performance

Measured on the 4-core / 2.1 GHz cloud container used for development. Your
machine is likely faster; use `selfplay` or the `bench` example to check.

| Solve (heads-up single-raised pot, ~550 × 920 combos) | Time to 0.3–0.5% pot | Memory |
|---|---|---|
| River, full menu | ~0.03 s | tiny |
| Turn, full menu, exact rivers | ~6–8 s | ~0.1 GB |
| Flop, **fast**: 8 turns × 6 rivers, 1 size on later streets | ~10–13 s | ~0.2 GB |
| Flop, **exact**: every runout, 1 size on later streets | ~5–7 min | **~9 GB** |

Ranges after real preflop lines are often narrower (3-bet pots, narrowed
weights, pruning), which makes solves faster. Exact flop solves bigger than
`max_memory_gb` fall back to fast mode automatically, and the review says so.

## Configuration

`poker-trainer init-config` writes every option with its default. The most
useful ones:

```toml
opponent = "random"          # Reg | Nit | TAG | LAG | Calling Station | Maniac | random
random_each_hand = true
show_opponent_style = false  # reveal the style during the hand (always shown in the review)
hero_position = "alternate"  # alternate | random | sb | bb
stack_bb = 100.0
ai_delay_ms = 450

[solver]
flop_mode = "fast"           # fast | exact
flop_turn_samples = 8
flop_river_samples = 6
villain_range = "gto"        # gto | actual | both
target_exploitability_pct = 0.5
max_iterations = 400
max_memory_gb = 6.0
threads = 0                  # 0 = all cores

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

* **More players (3–9).** Needs per-position charts, a multiway-aware AI, and
  a policy for multiway postflop review (heads-up solvers can't grade
  multiway pots).
* **Preflop grading** against charts, or against a preflop solve.
* **Solver speed:** suit isomorphism, 16-bit compressed storage, and SIMD in
  the showdown sweeps.
* **Exploitative review.** Node-lock the villain to its AI style's postflop
  tendencies and show the max-EV play against that specific opponent.
* Grade thresholds are fixed constants (`trainer::review::Verdict`).
