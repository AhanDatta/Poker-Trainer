# Poker Trainer

A No-Limit Hold'em trainer for the terminal, written in Rust. Think chess
puzzles, but for NLHE. You're dealt into a hand at a heads-up, 6-max or 9-max
table, play it from preflop to the river against AI opponents with
recognizable styles, and afterwards check every postflop decision you made
against a solver. A separate **solver study** mode lets you browse the
preflop charts and the solver's strategy in any spot, GTO Wizard style.

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
| 8 AI styles, tuned to typical tracker stats (Reg, TAG, LAG, Nit, Rock, Fish, Calling Station, Maniac) | ✅ |
| Game types (Tough, Typical, Soft, Wild, Custom counts) with session lineups | ✅ |
| Heads-up pots: exact solver review of every hero decision (EV per action, mix, grade) | ✅ (at any table size) |
| 3-way pots: 3-player CFR review with exact card removal (sampled runouts) | ✅ approximate |
| 4+-way pots | ❌ not analyzed (analysis resumes once ≤3 players remain) |
| Flop solving: exact (default) or fast | ✅ |
| Villain range in review: GTO charts / villain's actual style range / both | ✅ |
| Rake (no flop, no drop, capped) | ✅ |
| Solver-based preflop charts (heads-up, 6-max), built in | ✅ |
| Solver study mode: browse charts and solutions, any line, both ranges, typed spots | ✅ (heads-up exact, 3-way approximate) |
| Layouts that fit zoomed terminals (~100×30) and grow on big windows; mouse clicks | ✅ |
| Session frequency score: how your action frequencies match the solver's mixes, spot by spot | ✅ (heads-up pots) |
| Random number (0–99) at each decision, for playing mixed strategies | ✅ |
| Saved sessions: continue, start new, delete | ✅ |
| CSV export of saved hands and graded decisions, bb/100 by format | ✅ |
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

Optional, experimental: heads-up and 3-way solves on an NVIDIA GPU. This needs the CUDA
Toolkit 13.x to build (and on Windows, Visual Studio's C++ compiler); the
built program needs only the NVIDIA driver. Then pick **Solve on: Auto** (or
GPU) in the setup screen, or set `backend` in the config. Without a usable
GPU, solves run on the CPU as usual. Details and numbers:
[crates/solver/cuda/README.md](crates/solver/cuda/README.md).

```bash
cargo install --path crates/tui --features gpu
```

Use a terminal with Unicode and 256 colors (any modern one; on Windows,
Windows Terminal).

**Text too small?** A terminal program can't set its own font size, so zoom
the terminal itself (usually Ctrl/Cmd + and Ctrl/Cmd −). Every screen adapts
down to about 100×30 characters; on bigger windows the hand grids, cards and
panels grow to use the space. You can click actions, grid cells and review
decisions with the mouse (hold Shift to select text instead).

Other commands:

```bash
cargo run --release -- study                  # open the solver study mode directly
cargo run --release -- init-config            # writes poker-trainer.toml with every setting
cargo run --release -- --config my.toml       # use a specific config
cargo run --release -- --seed 42              # reproducible hands
cargo run --release -- selfplay --hands 3     # headless: scripted hero + printed reviews (speed check)
cargo run --release -- import-charts DIR --format lines --players 6 -o charts.txt   # convert your own ranges
cargo run --release -- export -o DIR          # saved hands as CSV + bb/100 by format (see below)
cargo test --workspace                        # unit, property and end-to-end tests
cargo run --release -p solver --example bench -- flop-exact-lean Ks7s2d   # solver benchmarks
cargo run --release -p solver --example compare -- Ks7s2d                 # fast vs exact accuracy
```

### Keys

| Where | Keys |
|---|---|
| Setup | ↑/↓ select · ←/→ change (mode, table size, styles, position, stacks, rake, solver) · Enter play/study (on "Bet sizes": open the size editor) · **O** sessions · **G** frequency score · W save settings · Q Q quit |
| Table | **F** fold · **C**/**X** check/call · **1–9** bet/raise sizes · **A** largest raise · **N** next hand · **R** solver review · **H** history · **G** frequency score · **S** settings |
| Review | ↑/↓ choose decision · **Tab** switch GTO / villain's-actual-range solve · **N** next hand · **G** frequency score · Esc back |
| Sessions | ↑/↓ select · Enter continue (or start a new session) · **D** delete (then **Y**) · Esc back |
| Study | **F**/**C**/**1–9** take an action · arrows (or click) pick a hand · **Tab** strategy / each player's range · **Backspace** back · **G**/Home start · **N** new random board · **E** type a spot · **P** back to preflop · Esc menu |

After you fold, the rest of the hand plays out instantly. Quitting takes Q
twice within two seconds (any other key cancels), so a stray key press
doesn't end a session; Ctrl+C quits at once. Either way the session is
saved (see below).

### Your position

**Your position** on the setup screen (`hero_position` in the config) picks
where you sit:

* **Rotate** (the default): the button moves every hand, as at a real table
  (heads-up: you alternate SB and BB).
* **Random**: a random position every hand.
* **Only …**: only the positions you tick in the rows below it, e.g. only BTN,
  or CO and BTN. Hands follow the normal rotation but skip the positions you
  didn't tick (CO, BTN, CO, BTN, …); with one position you play it every
  hand. The other players keep their seats around you; new players still sit
  down now and then.

Positions use the table's names. 6-max: UTG, HJ, CO, BTN, SB, BB. 9-max: UTG,
UTG+1, UTG+2, LJ, HJ, CO, BTN, SB, BB. Heads-up: SB (the button) and BB, and
`btn` also means the SB. In the config, names are case-insensitive:

```toml
hero_position = "btn"            # always the button
hero_position = ["co", "btn"]    # cutoff and button only
```

A name the table size doesn't have (say `"lj"` at 6-max) is an error. When you
switch table size on the setup screen, those positions are unticked instead;
if none are left, you go back to Rotate. The history list and the reviews show
your position in every hand.

### Saved sessions

Every finished hand is saved, with its review, as soon as the review is done.
Quitting saves the rest; a review still running is cut short. The next time
you start the trainer it offers to continue a saved session (stats, history,
reviews and the frequency score carry on), start a new one, or delete old
ones. The **Session** row on the setup screen (or **O**) opens the same list.

Sessions live in `~/.local/share/poker-trainer/sessions` (macOS:
`~/Library/Application Support/poker-trainer/sessions`; Windows:
`%APPDATA%\poker-trainer\sessions`). Each is a directory with `meta.json`
(settings, lineup, summary) and `hands.jsonl` (one hand per line, about 20 KB
per reviewed hand). Set `sessions_dir` to keep them elsewhere,
`save_sessions = false` to turn saving off, or pass `--no-save` for one run.
A continued session uses its own saved settings.

### Exporting hands (CSV)

`poker-trainer export` writes every saved session (oldest first) as two CSV
files in the current directory, or in `-o DIR`. Any earlier export there is
replaced. `--session ID` exports just one session (its directory name). It then
prints bb/100 by format:

```text
Format             Hands    Net (bb)    bb/100
HU 100bb            1200      +54.30      +4.5
6-max 100bb          800      -21.60      -2.7
All                 2000      +32.70      +1.6
```

* `poker-trainer-hands.csv` has one row per hand:
  * session and session start (UTC; hands have no time of their own), hand
    number;
  * players, format (`HU`, `6-max`, …), stack in bb, big blind, your position,
    your cards, the board, and each opponent's position and style;
  * net in chips and in bb, and whether you reached showdown;
  * the review state (`done`, `cancelled`, `error`, or `none` when there was
    nothing postflop to grade), graded decisions, their total EV loss in bb,
    and the worst verdict.

  bb/100 for any slice is `100 × sum(net_bb) / rows`.
* `poker-trainer-decisions.csv` has one row per graded decision:
  * street, board, the street's earlier actions, your cards, pot and to-call
    in bb;
  * your action and the best one, EV loss (bb and % pot), verdict,
    exploitability, which solver ran, and the solver's notes.

  When a review solved both the GTO and the villain's-actual ranges, both rows
  are there. `counted = true` marks the one the in-app stats use. Join to the
  hands file on `session` + `hand`.

Net results come from settling each saved hand, just as the in-app stats do,
so the totals match the session list. The table size and stacks are read from
each hand, so a session where you changed the table size is split correctly.
Fields are RFC 4180 CSV, UTF-8.

### Frequency score

The EV grade can't reward mixing. When the solver mixes, its actions have
nearly equal EV, so always taking the most common one grades as Best every
time. More generally, no per-decision score can reward mixing: its expected
value is linear in your frequencies, so a pure strategy always does at least
as well. The frequency score (**G**) therefore looks at the whole session.
It covers your decisions in heads-up pots, from the GTO-range solve.

* **Calibration, per spot type.** A spot type is street × facing nothing / a
  bet / a raise × in or out of position. Each action category is the
  solver's own option: fold, check, call, each bet or raise size of the
  menu, and all-in.
  * The **expected share** of a category is the sum of the solver's
    probabilities for the hands you actually held.
  * The **observed share** is how often you chose it.
  * A 5% bluff taken once in 20 such spots matches exactly; always taking
    the main option of a 70/30 spot shows up as 30 points of deviation.
  * The deviation (total-variation distance) is compared with what a player
    sampling exactly from the solver's mixes would show on the same
    decisions. This comes from a Monte Carlo of 1000 sessions, which gives
    the noise level and a p-value per spot and overall.
  * The **score** is 100 × (1 − deviation beyond luck). It is shown with the
    ± that luck alone could move it. Spots with p < 0.05 are listed as
    leaks, biggest first, with the category furthest off (e.g. "Flop IP no
    bet: Bet 75% 12% vs solver 40% ▼").
* **Aggression and mixing.** Order each spot's actions from passive to
  aggressive. Place your choice at the midpoint of its interval on the
  solver's cumulative mix: u = F(a−) + p(a)/2.
  * For a perfect mixer E[u] = ½ exactly, so mean(u) − ½ measures
    **aggression**, with an exact z-score.
  * The spread of u around ½ relative to its exact expectation measures
    **mixing**. Below 1, you take the solver's main option too often (too
    pure). Above 1, you take its rare options too often. This catches what
    the totals can't: always checking the hands the solver checks 70% and
    always betting those it bets 70% matches the totals but never mixes.
* **Weighting.** Decisions are weighted by the pot in big blinds, so leaks in
  big pots count more. The setup row *Frequency score weighting* switches to
  equal weights.

The EV grades stay alongside, for actions the solver doesn't take at all.

**Random number.** At each of your decisions the table shows *RNG nn* (0–99,
independent of the deal). Use it to play mixes: with a 30% bet, bet on 0–29.
Low numbers go to the most aggressive option. The review shows which action
the solver's mix for your hand gives that number. Turn it off with the setup
row *Random number* (`show_rng`).

### Bet sizes

The setup screen's **Bet sizes** row opens an editor for the sizes used at
the table, by the AI and in every solve: the preflop open, iso-raise and
3-bet/4-bet sizes, bet and raise sizes per street (`33%, 75%`, `150%` for
an overbet, `3x`, `allin`), the cap on bets per street, when all-in is
added, and the simplified turn/river sizes inside flop solves. It previews
the options in a sample pot and shows the solve cost against the default
sizes (a turn tree's decision count and an exact flop's memory), since every
extra size makes trees bigger. W on the setup screen saves the sizes to the
config (`[menu]` and `[solver.flop_future_menu]`).

### Solver study mode

Set **Mode** to *Solver study* on the setup screen and press Enter (or run
`poker-trainer study`). It uses the table size, stacks, rake and flop-solving
settings from the same screen.

1. **Preflop:** click through a line in the charts, seat by seat. The grid
   shows the acting seat's chart mix per hand; a cell's colored width is the
   share of that hand still in the seat's range, so you see the range narrow
   as the line goes on. **Tab** cycles through each player's range.
2. **Flop:** once two or three players see the flop, type it (`Ks7d2c`) or
   press Enter for a random one. The flop is solved with the line's ranges,
   pot and stacks (your Fast/Exact setting applies; Fast is recommended).
3. **Browse:** the grid shows the acting player's strategy (colored by
   action mix; the second row of each cell is the hand's EV in bb on taller
   windows). The panel lists every action with its frequency over the whole
   range and for the selected hand, the selected hand's EV per action, and
   each of its combos. Take any action, the solver's or not, to go down the
   tree. **Tab** shows each player's range at this point.
4. **Turn and river:** when a street's betting ends, type the next card (or
   Enter for a random one). That street is then solved from the ranges the
   previous street narrowed to, with the full bet menu, like the hand
   reviews do. On the development machine (4 cores) a fast flop took ~30 s,
   a turn ~2–8 s, a river well under a second, and a 3-way flop ~1 min.
5. **Rare lines:** if you take a line the solver almost never takes (under
   1.5% of a range gets there), the spot is re-solved from that point with
   that range widened, so the strategy shown still means something. A note
   says so.

Press **E** to type a spot instead: 2 or 3 players, their ranges in the
usual notation (`22+,A2s+,KTo+`, weights like `AKs:0.5`), pot, stacks and a
board of 3 to 5 cards (empty = random flop). The editor starts from the
current spot, with a live grid preview of the range you're editing. **N**
deals a new random board for the same ranges.

Going back and forth is cheap: finished turn and river solves are kept for
lines you leave, and solves you abandon before they finish are cancelled.
Study solves go ahead of hand reviews in the solver queue, and opening study
mode cancels unfinished hand reviews.

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
  preflop/  preflop model: solves the preflop betting tree, pots that see a flop valued by equity realization
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

Alternatives that were measured and didn't help at the 0.5%-of-pot target
(turn and sampled-flop benchmarks, exploitability every 50 iterations):
predictive CFR+ and predictive DCFR (about 2× worse at every checkpoint,
and they need 50% more memory), other α/β/γ values and clipping negative
regrets (equal or worse), and reach-weighted averages without restarts
(~10% better on turns, worse on flops until ~250 iterations).

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

### 3-player solver (`solver::multi`)

Exact multiway solving of a whole flop is beyond a desktop machine, so 3-way
pots use DCFR on a three-player game with some approximations:

* **Sampled runouts and simplified later streets.** All-in runouts are
  sampled like the rest of the tree (they held 72% of a sampled flop tree's
  showdowns when every card was dealt). A 3-way turn with every river card
  is small (~0.2 GB), but a 3-way flop with every runout needs well over
  14 GB, so flops stay sampled; study mode re-solves each street when it's
  dealt.
* **No equilibrium guarantee:** CFR with 3 players has none, though in
  practice it's sensible, as in Pluribus. Convergence is reported as the
  average best-response gain.

**Card removal between the two opponents** is exact by default ("3-way
pots" on the setup screen, `multiway_exact_removal` in the config).
Terminal values sum over pairs of opponent hands that share no card, using
an inclusion–exclusion identity over shared cards. Per-card sums and sparse
per-card hand lists are kept as hands enter the strength sweep, so a query
costs about as much as the ranges are dense, and showdowns on the same board
share sweeps (4 at a time). With exact removal and no rake the game is
exactly zero-sum (a test checks this), and terminal values match a brute
force over all hand triples. *Approximate* treats the opponents' hands as
independent: removal against each opponent is exact and showdowns factor
into per-opponent sweeps (P(win) = P(beat A) × P(beat B)).

How much the independence approximation costs, judged in the exact game
(average best-response gain, % of the pot; 3-way turns with every river
card, measured with `cargo run --release -p trainer --example multibench`):

| Spot | Independent strategy | Exact strategy | Exact removal's time |
|---|---|---|---|
| CO open, BTN + BB call (chart ranges), J♠9♠4♦2♣ | 0.40 | 0.13 | ×2.0 |
| Three wide ranges (~600 combos), J♠9♠4♦2♣ | 0.32 | 0.15 | ×3.1 |
| Narrow overlapping ranges (AA–QQ/AK vs AK/AQ/KK–JJ vs AQ/AJ/TT–88), A♥7♣2♦5♠ | **3.40** | 0.04 | ×2.2 |

(150–300 iterations; on the narrow spot the first player's root strategy
moves from 25% check / 67% small bet / 8% big bet to 32% / 65% / 3%.)

With wide or lopsided ranges the opponents rarely collide and the
approximation is close; when narrow ranges compete for the same cards
(squeeze and 3-bet multiway pots) it's off by several % of the pot and the
strategy visibly changes.

Design (as in the heads-up solver): a flat node arena, regrets and strategy
sums in the shared storage (16-bit for big trees), pooled buffers, and a
memory check before allocating. A traversal skips the subtrees after its
player folded (its payoff there is fixed). Each player's average strategy is
reach-weighted and accumulated at its nodes during the next active player's
traversal; DCFR uses α 1.5, β 0, γ 3 without average restarts (restarts
left 3-player averages stuck or bouncing).

| Chart-range 3-way flop (J♠9♠4♦, 8 turns × 4 rivers) to 1.5% of the pot | Before the rebuild | Now |
|---|---|---|
| Independent opponents | ~57 s | 19 s |
| Exact card removal | ~190 s | 47 s |

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
  playability order (aggressive styles open the extra hands, passive ones
  limp them); *aggression* turns calls into raises in unopened and limped
  pots, *3-bet tendency* does the same when facing a raise.
* **Postflop:** Monte-Carlo equity against reads of every opponent's range,
  narrowed after each of their actions. Style thresholds then decide
  value-betting, calling down, raising, slow-playing and sizing. The last
  aggressor c-bets and barrels at its style's frequencies; bluffs and
  c-bets shrink in multiway pots, and strength thresholds use a heads-up
  equivalent of multiway equity.

The styles were tuned with all-AI simulations
(`cargo run --release -p trainer --example population`) to match typical
6-max tracker stats for each player type. Measured over 20,000 hands each:

| Style | VPIP | PFR | 3-bet | C-bet | Plays like |
|---|---|---|---|---|---|
| Reg | 20–23 | 15–19 | 10 | 67 | solid winning regular, close to the solver charts |
| TAG | 18–20 | 14–16 | 9 | 72 | a bit tighter than a reg |
| LAG | 31 | 24 | 13 | 74 | loose-aggressive pro, not reckless |
| Nit | 13 | 10 | 3–4 | 55 | very tight, bets its hands, folds to pressure |
| Rock | 16 | 8 | 4 | 36 | weak-tight recreational, limps and gives up |
| Fish | 39–40 | 9–10 | 4 | 45 | loose-passive recreational |
| Calling Station | 53 | 5–7 | 3–4 | 34 | calls almost anything (aggression factor 0.4) |
| Maniac | 54 | 43 | 25 | 89 | raises and bluffs constantly |

A test (`crates/trainer/tests/styles.rs`) keeps every style within the
ranges of its player type.

**Game types.** At 6-max and 9-max you choose the kind of game: *Tough*
(regs, TAGs, some LAGs and nits), *Typical* (regs plus fish, rocks and the
odd maniac), *Soft* (mostly fish, stations and weak-tight players), *Wild*
(maniacs and LAGs), or a *Custom* table with your own count of each style.
You can also make every opponent the same style. Players keep their seats
for the session; now and then one leaves and a new one, drawn from the game
type, sits down (the hand log says so). In simulated typical games, fish and
stations fill about half of the multiway-pot seats, as at real tables; in
tough games they're regs and LAGs.

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

**With an NVIDIA GPU** (experimental `gpu` build, `backend = "auto"`), on a
laptop RTX 4060 against its i7-13700H (14 cores), solving to 0.3% of the pot:

| Heads-up spot | CPU | GPU |
|---|---|---|
| Turn, full menu | 2.7 s | 0.87 s |
| Flop **exact**, two-tone (the default exact tree) | 128 s | 32 s |
| Flop exact, two-tone, 3x raises on turn and river | 147 s | 39 s |
| Flop exact, rainbow, 3x raises on turn and river | 218 s | 60 s |

The GPU's default precision is within a fraction of a percentage point of
the CPU's strategies; an exact mode reproduces the CPU bit for bit at about
2.6× the CPU's speed. Details: [crates/solver/cuda/README.md](crates/solver/cuda/README.md).

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

### Scoring and saved sessions

`cargo run --release -p trainer --example scorebench` builds a 2,000-hand
session with three graded decisions per hand (6,000 decisions, grids
included).

| Operation | Time |
|---|---|
| Frequency score, 500 / 2,000 / 6,000 decisions (after each settled hand) | 3 ms / 11 ms / 15 ms |
| Saving one hand (append + meta) | 0.45 ms average |
| Loading the 2,000-hand session (43 MB) + resuming it | 0.26 s + 0.03 s |
| `stats()` per frame / `refresh()` per tick | 23 ns / 2 ns |

The score's Monte Carlo is capped at about two million draws, with at least
200 simulated sessions. Spots with a pure mix are counted once instead of
being drawn. Drawing the history, score and sessions screens takes under
0.5 ms per frame with 2,000 hands. Solver speed is unchanged: the bet-menu
change behind the categories was benchmarked against the previous commit on
the turn and flop benchmarks.

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
  * AI style adjustments always produce valid frequencies;
  * every bet-menu action gets exactly one frequency category;
  * the frequency score is invariant to scaling the weights, its shares and
    p-values stay in range, and the aggression z-score is standard normal
    for a perfect mixer.
* **Scoring and sessions:**
  * a perfect mixer scores ~100 with no leaks;
  * always-the-main-option, over-bluffing and balanced-but-pure play are
    each caught by the right statistic;
  * saved sessions round-trip, with stats, score, reviews, rolls and grids
    to 1/10000;
  * a session can be continued, and quitting mid-review is handled.
* **CSV export:**
  * arbitrary fields (commas, quotes, line breaks) round-trip through the CSV
    writer;
  * the exported nets, per-format totals and bb/100 equal the session's own
    stats, for random tables, table-size changes and hero lines;
  * the counted decisions and their EV loss equal the stats of a reviewed
    session;
  * hands saved before reviews recorded their solver still export.
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
opponent = "random"          # random (drawn by game type) | Reg | TAG | LAG | Nit | Rock | Fish | Calling Station | Maniac
game = "typical"             # tough | typical | soft | wild | custom
custom_table = { Reg = 2, TAG = 1, Fish = 1, Maniac = 1 }   # with game = "custom"
seat_change_pct = 2.0        # chance per opponent and hand that a new player sits down (0 = never, 100 = every hand)
show_opponent_style = false  # reveal styles during the hand (always shown in the review)
hero_position = "alternate"  # alternate (rotate) | random | a position ("btn") | a list (["co", "btn"]); see "Your position"
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
backend = "cpu"              # cpu | auto | gpu: where heads-up and 3-way solves run (GPU needs a `--features gpu` build)
multiway_turn_samples = 8    # 3-way pots
multiway_river_samples = 4

[scoring]
pot_weighted = true          # frequency score: weight decisions by the pot
simulations = 1000           # Monte Carlo samples (capped on long sessions)

show_rng = true              # random number at each decision
save_sessions = true         # sessions_dir = "..." to save elsewhere

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
* **3-way solves are approximate:** sampled runouts (including all-in
  runouts) and simplified later streets, and no equilibrium guarantee.
* **Preflop grading** isn't implemented, and the frequency score covers only
  postflop decisions in heads-up pots (3-way solves are too noisy to score
  frequencies against).
* **Frequency score blind spots:** a spot type pools all your hands, so
  swapping hands (bluffing one the solver checks while checking one it
  bluffs) keeps the totals right. The mixing measure and the EV grades
  catch most of this, but not all. An estimate of exploitability in bb/100
  (a best response against your measured tendencies) would be the next
  step.
* **Study mode** uses the sizes from the bet-size editor for every spot (no
  per-spot sizes). Its 3-way solves have the same limits as the 3-way
  review.
* **9-max charts** use fallback rules; no public solver set was found.
* **Exact flop solves still take minutes** on a 4-core machine. Wider SIMD,
  single-pass sweeps and predictive CFR variants were tried and didn't help
  (see the solver sections); the remaining levers are smaller trees (e.g.
  fewer bet sizes) and the experimental GPU backend (about 30-60 s per exact
  flop on a laptop RTX 4060; see Performance).
* **Exploitative review:** node-lock the villain to its AI style's postflop
  tendencies to show the max-EV play against that specific opponent.
* The preflop menu has one raise size per level, while imported charts may
  use slightly different sizes (e.g. 2.6bb opens). Their frequencies are
  used as-is.
