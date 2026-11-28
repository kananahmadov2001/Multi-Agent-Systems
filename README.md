# Multi-Agent-Systems

CSE 5106 Final Project: Centralized Conflict Resolution for Independently Planning Agents.

A central agent takes plans it did not write and repairs the collisions between them. We
compare a fixed Conflict-Based Search baseline with two LLM arbitration strategies:

1. **CBS** — a fixed best-first search. No API calls.
2. **LLM, no tree** — the model picks which agent to constrain, with no search structure.
3. **LLM-guided CBS** — the same tree, with the model choosing which open node to expand.

A **no-LLM control** (approach 3 with the model replaced by "always take the cheapest node")
is what every claim about LLM guidance is measured against.

Around that sits a four-stage pipeline: select `k` agents from their advertised capabilities,
decompose the task into an ordered route per agent, plan each route independently on the grid
or in PDDL with Fast Downward, then detect and arbitrate the collisions and verify the joint
plan. A failed split backtracks to selection and tries the next covering team.

## Setup

Python 3.10+. Fast Downward 24.06.1 unpacked **next to** this repo (`../fast-downward-24.06.1`);
build instructions are further down.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ARBITER_MODEL=claude-haiku-4-5-20251001
export ARBITER_MAX_CALLS=4000
# optional: export ARBITER_MAX_TOKENS=4000       (default 4000; the first live runs used 1500,
#                                                 which cut some replies off)
```

## Tests

| Command | What it checks |
|---|---|
| `python3 -m mas.verify` | The independent plan checker: 1 good plan accepted, 7 bad ones rejected (collision, swap, pocket swap, wrong start, wrong goal, illegal jump) |
| `python3 -m mas.arbitration.cbs_baseline` | Approach 1: CBS, best-first and the depth-first ablation |
| `python3 -m tests.test_fd [FD_DIR]` | Approach 1 with Fast Downward as the low-level planner. **Needs Fast Downward** (default `../fast-downward-24.06.1`) |
| `python3 -m tests.test_llm` | Approach 2 with fake LLMs |
| `python3 -m tests.test_hybrid` | Approach 3 with fake LLMs |
| `python3 -m mas.instances.instances` | The hard test cases and their backtrack counts |
| `python3 -m mas.selection.agent_pool` | Capability advertisement and tag matching |
| `python3 -m mas.selection.selection` | The three selection policies and their ranked alternatives |
| `python3 -m mas.decomposition.decompose` | Greedy, exhaustive and LLM decomposition |
| `python3 -m mas.pipeline` | The pipeline loop and the backtrack-to-selection edge |
| `python3 -m mas.instances.pipeline_instances` | The capability-overlap domain |
| `python3 -m mas.instances.colored_box` | The colored-cell visibility domain, where each agent sees a different grid |
| `python3 -m checks.check_fd_route` | The generated PDDL against `plan_route`, **without needing Fast Downward**. Add `--adversarial` or `--pipeline` |
| `python3 -m experiments.control` | Approach 3's search with no LLM, on all 60 benchmark instances |
| `python3 -m mas.arbitration.ordering` | Cross-agent task order: detection, bounded repair, the release constraint, and `verify_order`, which checks order without sharing code with the detector that drives the repair |
| `python3 -m mas.planning.fd_planner` | The PDDL encoding of vertex and edge bans, for single goals and routes, checked on the generated text. No Fast Downward needed |
| `python3 -m mas.selection.llm_select` | The selection prompt and the team a reply is parsed into |
| `python3 -m checks.check_authored` | Rebuilds the authored-PDDL cases from their seed and counts how many replies are exactly the compiled problem. Free, no Fast Downward |
| `python3 -m checks.check_guidance` | Approach 3's node choices against the no-LLM control: how often the model disagreed, and whether any outcome changed. Free, no Fast Downward |
| `python3 -m checks.check_refusals` | Re-reads the live composite transcript and checks every refused decomposition against the team's own advertisements |
| `python3 -m checks.check_edge` | Standard edge constraints against our stronger vertex rule |
| `python3 -m checks.check_edge_oracle` | Both conflict rules against brute force, so "stronger" can be priced. A case that runs out of the node budget (`--cap`, default 2,000) is counted as capped, not as a miss; with 3 agents one case does, and `--cap 20000` judges it too |
| `python3 -m mas.instances.hard_cbs` | Regenerates the hard set by how many nodes best-first CBS itself spends |
| `python3 -m experiments.bench` | All three approaches on the hard cases with fake LLMs (harness only) |

One fuzz test, free, that needs neither Fast Downward nor a key:

```bash
python3 -m checks.fuzz_ordering --n 60 --sites 1 --oracle
```

Every instance the order repair gives up on is put to a brute-force joint oracle, which keeps
"incomplete" and "wrong" distinguishable: 55 of 56 feasible single-pair instances solved, 0
unsound, 1 suboptimal by one step. `--sites 2` is the harder case, a total order over four
parts, where the same repair solves only about a third of the feasible instances.

```bash
python3 -m checks.probe_e2e --fd "$FD" --fake     # the whole pipeline, stage by stage
python3 -m checks.check_search --fd "$FD"         # does the search configuration change what we get back?
```

## Benchmarks

`$FD` is the Fast Downward directory, e.g. `export FD="$HOME/Desktop/CSE 5106/fast-downward-24.06.1"`.

### Arbitration, fixed team (`bench_fd.py`)

60 instances in three sets, plus `hcb`. Approach 1 only, and free, unless you pass `--live`
or `--fakes`.

```bash
python3 -m experiments.bench_fd --set sweep --fd "$FD" --out /tmp/check.csv        # free, Approach 1
python3 -m experiments.bench_fd --set hard --fd "$FD" --live --out results/new/hard.csv --log results/new/hard.jsonl
python3 -m experiments.analyze results/fd/hard_fd_haiku.csv results/fd/hard_fd_haiku.jsonl   # free: read a finished run
```

`bench.py` takes `--conflicts vertex|standard` too, and there it applies to **all three**
approaches, so the comparison stays like-for-like. Under `bench_fd.py` the same flag reaches
all three as well.

Options: `--set easy|hard|hcb|random|sweep|all`, `--conflicts vertex|standard`, `--start N`,
`--limit N`, `--only NAME[,NAME]`, `--repeat R`, `--search CONFIG`, `--a2-calls N`,
`--a3-calls N`, `--fakes`, `--dfs`, `--live`, `--log FILE`, `--fd DIR`,
`--fd-mode run|exec`, `--fd-container NAME`, `--fd-workdir DIR`, `--out FILE`, `--force`.

**Pass `--out`.** Without it the results go to `results/fd/fd_results.csv`, which is
committed; a run that would replace a file already there stops and says so, and `--force`
is how you mean it. That guard exists because a `--limit 3` smoke run once replaced the
full 25-instance set in silence.

`--set hcb` is the hard set regenerated against best-first CBS itself: an instance is kept when
CBS spends at least 20 expansions on it, rather than when two depth-first rule orders disagree,
which is what filtered the original hard set. It is opt-in; `--set all` does not include it.

`--conflicts standard` swaps our vertex rule for proper edge constraints, which keeps the tree
complete. It reaches all three approaches, so an edge run is a like-for-like comparison and not
only a study of Approach 1. On the grid hard set there is a live Haiku run as well,
`results/haiku/hard_edge_haiku.csv`: Approach 3 solves the same 21 of 25 as the no-LLM control,
with the same plan lengths, and Approach 2 solves 14 of 25. The Fast Downward edge results
(`results/fd/*edge*`) use the scripted policies only.

`--fakes` adds the scripted policies to Approaches 2 and 3 at no cost. Use it as a regression
test, not as a result: the policies are deterministic, so any difference between two runs is the
code rather than the model.

### The full pipeline (`pipeline_bench.py`)

```bash
# free: deterministic, no API calls
python3 -m experiments.pipeline_bench --fd "$FD" --out results/new/p.csv
python3 -m experiments.pipeline_bench --fd "$FD" --k 3 --arbiter hybrid --out results/new/control.csv   # the no-LLM control

# API calls
python3 -m experiments.pipeline_bench --fd "$FD" --k 3 --arbiter hybrid --live --out results/new/hybrid.csv
python3 -m experiments.pipeline_bench --fd "$FD" --k 3 --arbiter llm --live --out results/new/llm.csv
```

Options: `--domain capability|colored`, `--n N`, `--seed N`, `--strategies LIST`, `--k N`,
`--arbiter cbs|hybrid|llm`, `--selector tag|llm`, `--decomposer greedy|llm`,
`--ordering off|check|repair`, `--ordering-repairs N`, `--authored`,
`--authored-attempts N`, `--authored-lenient`, `--live`, `--calls N`, `--max-teams N`,
`--agents N`, `--tags N`, `--scarce`, `--size N`, `--walls F`, `--horizon N`,
`--search CONFIG`, `--fd DIR`, `--log FILE`, `--out FILE`, `--force`, `--quiet`.

`--ordering check` adds cross-agent task order to verification and independently replays it
with `verify_order`; `--ordering repair` also tries to fix violations by delaying the later
part. Both are free and deterministic, and both work on either low-level planner and with any
arbiter. Ordering sits outside the arbiter, so the LLM never sees an ordering conflict even
under `--live`.

Ordering is opt-in (`off` by default). `Task.from_text` reads capability tags, not temporal
language: a temporal requirement must be supplied explicitly as `Task(..., order=...)`.
Selection and decomposition are also deterministic by default. `--selector llm` chooses only
the first team; later fallback teams are deterministic. Team switching is bounded by
`--max-teams` (3 by default), with one decomposition attempted per team. `--calls` is a cap
per arbitration solve, not a whole-instance cap across team switches or ordering rounds.

`--authored` is a partial version of the proposal's stage 3: each agent writes only its first,
unconstrained problem file. Every constrained replan uses the compiled problem. It needs
`--fd`, since Fast Downward solves whatever the agent produces. Without `--live` a scripted
author stands in, which costs nothing and establishes the ceiling for a given configuration.
`--authored-attempts N` re-asks up to N times with the fault fed back, the repair loop of
Smirnov et al.; the default of 1 is one shot. `--authored-lenient` falls back to the compiled
problem when authoring fails, which measures authoring quality without letting a bad problem
decide the run.

`--horizon N` matters for `--authored`: the time-expanded encoding grows with it, and at the
default 5x5 horizon 20 a two-site route problem is about 51k output tokens, beyond what a
model will emit reliably. 4x4 with horizon 12 is about 18k and is where the authored runs
below were done.

Without `--live`, an LLM arbiter runs on a scripted fake that always takes the first option —
which is exactly the no-LLM control, at no cost.

**Pass `--out` whenever you change `--k` or `--agents`.** Without `--out` the filename is
built from the flags, and those two are not among them, so every run of a sweep claims the
same name. The second run of that sweep now stops rather than replace the first (`--force`
overrides), but a named file is still what you want.

### Agent-authored PDDL (`authored_pddl.py`)

A standalone measurement of the proposal's authoring step: the agent writes one problem file
instead of the harness generating it. Needs `--live` and a key; without it the checker runs on
a canned reply. Results are printed, so redirect them.

```bash
python3 -m mas.planning.authored_pddl --live --cases 20 > results/authored/single.txt
python3 -m mas.planning.authored_pddl --live --cases 12 --routes > results/authored/routes.txt
python3 -m mas.planning.authored_pddl --live --cases 20 --dump /tmp/replies.txt   # read a fault, don't guess it
```

Options: `--cases N`, `--routes`, `--size N`, `--horizon N`, `--seed N`, `--dump FILE`, and
`--fd DIR` to also solve each authored problem for real rather than only checking its shape.
A 5x5 single-goal problem at horizon 20 needs roughly 15k output tokens; a two-site route at
that size needs about 51k. Raise `ARBITER_MAX_TOKENS` before a routes run or the replies are
cut off mid-problem.

This measures one problem at a time. To run the same thing inside the pipeline, where two
agents must both succeed before arbitration can start, use `pipeline_bench.py --authored`:

```bash
python3 -m experiments.pipeline_bench --authored --fd "$FD" --n 6 --strategies cheapest \
  --size 4 --horizon 12 --walls 0.15 --out /tmp/ceiling.csv

export ARBITER_MAX_TOKENS=32000
python3 -m experiments.pipeline_bench --authored --live --authored-attempts 2 --fd "$FD" --n 6 \
  --strategies cheapest --size 4 --horizon 12 --walls 0.15 --out /tmp/authored.csv
```

Run the first one first. It uses the scripted author, costs nothing, and tells you what the
configuration scores with a perfect stage 3; without it the live number cannot be read.

Authored problems are cached on (grid, start, route, horizon), as `make_fd_planner` caches its
own, because an agent that reappears in a later team would otherwise be paid for twice. In the
authored all-stages run below (`pipeline_composite_authored_4x4.csv`) that suppressed 52 of 67
calls.

### Do the stages compose?

Every stage above was measured on its own, one substitution at a time, which is what makes the
comparisons readable. That is not the same as having run the pipeline.

Without `--live`, every stage that would call a model uses a scripted stand-in instead (the
`Oracle` in `pipeline_bench.py` for selection and decomposition, a cheapest-option fake for
the arbiter), so the whole thing can be run at once for nothing. This is what produced
`results/pipeline/pipeline_composite.csv` (13/20), on the grid planner:

```bash
python3 -m experiments.pipeline_bench --selector llm --decomposer llm --arbiter hybrid \
  --ordering repair --out /tmp/composite.csv
```

This is an integration test, not an experiment: the stand-ins are not a model and say nothing
about judgment.

A smaller variant also has the agents author their own PDDL, on Fast Downward at 4x4 and
horizon 12, where a problem still fits in one reply. It is
`results/pipeline/pipeline_composite_authored_4x4.csv` (3/6):

```bash
python3 -m experiments.pipeline_bench --selector llm --decomposer llm --arbiter hybrid \
  --ordering repair --authored --fd "$FD" --n 6 --strategies cheapest \
  --size 4 --horizon 12 --walls 0.15 --out /tmp/composite_authored.csv
```

The same run without `--authored` is identical row for row, so with a perfect author the
authored planner is indistinguishable from the compiled one. Three of this project's bugs
lived in the seam between two features, which is why these runs exist.

The 20-instance configuration with Haiku at all three stages is
`results/pipeline/pipeline_composite_live.csv`, with every prompt and reply in the matching
`.jsonl`:

```bash
python3 -m experiments.pipeline_bench --selector llm --decomposer llm --arbiter hybrid \
  --ordering repair --live --out results/pipeline/pipeline_composite_live.csv \
  --log results/pipeline/pipeline_composite_live.jsonl
```

It solves 7 of 20 against the stand-ins' 13 — 8 instances lost at decomposition, 4 at
selection, 1 at arbitration — and decomposition is the whole of the gap.

Across the run the model refused 19 decompositions, and **all 19 were of teams that covered
the task**, which `check_refusals.py` establishes without a model by running the same
`advertise()` that stage 1 used to pick the team:

```bash
python3 -m checks.check_refusals            # 19 refusals, 19 of covering teams
python3 -m checks.check_refusals --reasons  # every team, advert and stated reason
```

Sorting the stated reasons by the fault they show (keyword matching over one sentence each,
so a rough split): 10 invented a constraint the prompt does not impose, mostly that an agent
"cannot reach (3,3) in time"; 5 named the agent able to do a part and then called that part
unassignable; 2 read the vocabulary literally, declining because "walk is not explicitly
listed as a capability"; 1 denied a capability the prompt advertised in the same breath
(co2: "neither description includes flying capability" about an agent advertised as "flies
over walls and inspects rooms"); and 1 matched none of the patterns.

The first group is the largest and the most interesting one for the paper: stage 2 is asked
only to assign parts to agents that advertise them, and whether anyone can get there in time
is what stages 3 and 4 are for. The model answered a harder question than it was given and
refused on the answer.

Pass `--log` for any live run -- a refusal you cannot read is not evidence. Every row records
the model it used in a `model` column, `scripted` when a stand-in answered rather than a
model.

Create an output folder before writing to it (`mkdir -p results/new`). Ctrl+C is safe:
finished instances are kept.

Current benchmark paths count a plan as solved only after `verify.py` — an independent replay
written without reference to the search code — accepts it. The no-LLM control uses the same
check. Colored runs additionally check each agent's palette, and both ordering modes use the
independent `verify_order` replay. This describes the current code; older artifacts without
transcripts or current metadata remain limited evidence.

## Layout

Folders follow the four pipeline stages, so a section of the paper maps to a directory.

```
mas/                 the system
  selection/         stage 1: agent_pool, selection (3 heuristics), llm_select
  decomposition/     stage 2: decompose (greedy, exhaustive, llm)
  planning/          stage 3: fd_planner (compiled PDDL), authored_pddl (the agent writes it)
  arbitration/       stage 4: cbs_baseline (Approach 1), llm_arbitrator (2), hybrid (3), ordering
  instances/         the domains and their generators
  pipeline.py        joins the stages, including the backtrack to selection
  verify.py          the independent checker, stage 5
experiments/         bench, bench_fd, pipeline_bench, control, analyze -- these write results/
checks/              check_authored, check_edge, check_edge_oracle, check_fd_route,
                     check_guidance, check_refusals, check_search, fuzz_ordering,
                     probe_e2e
tests/               test_fd, test_hybrid, test_llm
domains/             the hand-written PDDL from the midterm
scripts/             the standalone conflict detector built for it
results/             every committed run
```

Run anything as a module from the repo root, never as a path:

```bash
python3 -m mas.arbitration.ordering
python3 -m experiments.bench --set hard --out /tmp/x.csv
```

Two things the layout does not say on its own. `cbs_baseline.py` is in `arbitration/`
because it is Approach 1, but it also holds `Grid` and the space-time grid planner that
nearly every other module imports; it is the one file worth reading first. And
`checks/check_fd_route.py` is imported by `mas/planning/authored_pddl.py`, so `checks/`
is not purely a leaf.

### One caveat on the live results

The committed Haiku runs were made before the prompts were reworded to handle more than one
constraint shape. Approach 2 used to read `forbid a1 from being at (0, 2) at time 2`; it now
reads `require that a1 not be at (0, 2) at t=2`, and the instruction line, the constraint list
and the decision log changed with it. Approach 3's node descriptions changed the same way.

The two wordings say the same thing and no committed number moved, because every run since has
used the scripted policies, which read only the `OPTION` lines. But a model is not guaranteed to
answer a reworded prompt identically. Any new live run therefore belongs in its own column, not
appended to a table of the old ones, and a live rerun of an old configuration is a fresh
measurement rather than a reproduction.

### Scope relative to the proposal and midterm

The code implements the capability pipeline and cell/swap/ordering conflicts, but not shared
consumable tools or general non-spatial resource conflicts. The colored domain is a
colored-cell visibility abstraction, not the proposal's box-moving domain. Agent-authored PDDL
is partial as described above, and team switching is bounded. Bulk live results use Haiku;
Sonnet remains a two-instance spot check, not a full run. These are scope limits to state in
the paper and presentation, not completed features.

## Results folder

| Folder | What it holds |
|---|---|
| `results/fd/` | The main arbitration runs, Fast Downward as the low-level planner: `*_fd_haiku` (the headline runs on all three sets), `random_fd_rep3` (3 repeats), `hard_fd_sat` (satisficing search), `*_fd_equal20` and `*_fd_equal50` (both LLM approaches at the same call cap), `*_fd_edge` (`--conflicts standard`, Approach 1 only), and `hcb_fd_cbs` / `hcb_fd_equal50` / `hcb_fd_edge` (the CBS-regenerated hard set), and `hard_fd_edge_all` (all three approaches under edge constraints, scripted policies). `fd_results`, `fd_random` and `fd_sweep` are the first Fast Downward runs, Approach 1 only; their numbers are reproduced exactly by the CBS rows of the matching `*_fd_haiku` file, so prefer those. |
| `results/pipeline/` | Pipeline runs: `pipeline_results` (grid planner), `pipeline_fd` (PDDL), `pipeline_fd_hybrid` and `pipeline_fd_hybrid_fake` (LLM arbitration and its control), `pipeline_fd_llmselect` and `pipeline_fd_llmdecomp` (an LLM in place of the selection and decomposition heuristics), `pipeline_colored*`, the `k1`/`k2`/`k3` team-size sweep, and `pipeline_k3_fd_*` (all four arbiters on the conflict-dense configuration). Newer pipeline CSVs carry a `detail` column saying why the last attempt ended; older files predate that field. No committed configuration has a measured node-cap failure. |
| `results/pipeline/pipeline_composite*` | The all-stages configuration, 20 instances at one strategy. `_composite` uses the scripted stand-ins (13/20); `_composite_live` puts Haiku at selection, decomposition and arbitration (7/20), with every prompt and reply in `_composite_live.jsonl`. `_composite_authored_4x4` is the separate 6-instance variant that also authors its PDDL, at 4x4 and horizon 12 where a problem still fits in one reply (3/6); it predates the `--log` option, so it has no transcript and no model column. |
| `results/ordering/` | `pipeline_order_check` and `pipeline_order_repair` (grid planner), `pipeline_order_check_fd` and `pipeline_order_repair_fd` (Fast Downward). Both planners give 21/60 under checking and 39 against 42 under repair. Read against `results/pipeline/pipeline_results.csv` and `pipeline_fd.csv` (44/60 each), the same configuration with ordering off. |
| `results/authored/` | `single`, `single_summary`, `routes` are `.txt`: one problem at a time, standalone. `pipeline_authored_4x4*.csv` are the in-pipeline runs at 4x4 horizon 12 -- `_compiled` is the scripted-author ceiling, the other two are one shot and one repair round. |
| `results/haiku/` | Earlier arbitration runs on our own grid planner, kept so the PDDL numbers can be compared against them, and `hard_edge_haiku`: the live Haiku run of all three approaches on the hard set under standard edge constraints, to be read against `results/control/hard_edge_all.csv`. |
| `results/haiku_cap150/` | Re-run of six of the nine Approach-3 losses at the 50-call cap: hard3, hard15, hard18, r2, r13 and a2-2, with a 150-call cap. |
| `results/sonnet/` | `live_check`: Sonnet 5.5 spot check on 2 cases. |
| `results/control/` | `control.txt`, the output of `control.py`, and `hard_edge_all.csv`: all three approaches on the hard set under standard edge constraints with the scripted policies, to be read against the vertex numbers in `results/haiku/`. |
| `results/first_run_depth_first/` | **First run only, kept for the record.** `batch2`, `var1` and `results_live`, made with the earlier depth-first version of Approach 1 and a 1,500-token reply limit that cut some replies off. Do not mix with the tables above. |

Each run has a `.csv` (one row per instance and approach). New live runs should also have a
`.jsonl` with every prompt and reply; some older live artifacts predate `--log` and do not.
Do not mix models in one table.

---

## Setup: Fast Downward (macOS, build from source)

Fast Downward is an external dependency and is **not** part of this repo.
Install it locally, as a sibling directory next to this repo (not inside it):

```
CSE 5106/
├── Multi-Agent-Systems/     <- this repo
└── fast-downward-24.06.1/   <- installed here, next to the repo
```

### 1. Install prerequisites (macOS)

You need Xcode command line tools, Homebrew, and cmake.

    xcode-select --install

If you don't already have Homebrew:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Follow the on-screen instructions it prints at the end (usually two lines
adding Homebrew to your PATH), then restart your terminal or run:

    source ~/.zshrc

Then install cmake:

    brew install cmake

Confirm it worked:

    cmake --version

### 2. Download and extract Fast Downward

Run this from the parent folder that contains this repo (i.e. one level up):

    curl -O https://www.fast-downward.org/latest/files/release24.06/fast-downward-24.06.1.tar.gz
    tar -xzf fast-downward-24.06.1.tar.gz
    cd fast-downward-24.06.1

(Check https://www.fast-downward.org/latest/releases/ in case a newer
version exists by the time you're reading this — adjust the version
number above if so.)

### 3. Build

    python3 build.py

This takes a few minutes. A CPLEX warning during the cmake step is
expected and can be ignored — CPLEX is an optional commercial solver
we don't use.

You should see it end with:

    Built configuration release successfully.

### 4. Verify with the built-in help

    ./fast-downward.py --help

This confirms the driver script runs, but does **not** confirm the
planner itself was built — it's just the Python wrapper.

### 5. Verify with a real test problem - solvable

**domain.pddl**

```lisp
(define (domain test)
  (:requirements :strips)
  (:predicates (at-a) (at-b))
  (:action move
    :parameters ()
    :precondition (at-a)
    :effect (and (at-b) (not (at-a)))))
```

**problem.pddl**

```lisp
(define (problem test-problem)
  (:domain test)
  (:init (at-a))
  (:goal (at-b)))
```

Run it:

```
./fast-downward.py domain.pddl problem.pddl --search "astar(lmcut())"
```

Expected: `Solution found. Plan length: 1 step(s).`

### 6. Verify with a real test problem - Conflict

Copy our test domain and problem files (in this repo, under `/domains/sanity-check/`)
into your `fast-downward-24.06.1` folder, or reference them by path:

**domain.pddl**

```lisp
(define (domain conflict-test)
  (:requirements :strips :typing)
  (:types agent cell)
  (:predicates
    (at ?a - agent ?c - cell)
    (free ?c - cell))
  (:action move
    :parameters (?a - agent ?from - cell ?to - cell)
    :precondition (and (at ?a ?from) (free ?to))
    :effect (and (at ?a ?to) (not (at ?a ?from))
                 (free ?from) (not (free ?to)))))
```

**problem.pddl**

```lisp
(define (problem two-agents)
  (:domain conflict-test)
  (:objects
    a1 a2 - agent
    start1 start2 shared - cell)
  (:init
    (at a1 start1)
    (at a2 start2)
    (free shared))
  (:goal (and (at a1 shared) (at a2 shared))))
```

This problem is deliberately unsolvable — both agents want the same
exclusive cell, and only one can hold it. Run it:

```
./fast-downward.py domain.pddl problem.pddl --search "astar(lmcut())"
```

**Expected result: no plan found.** That failure is correct — it's
your basic conflict, confirmed mechanically. If you instead get an
error (not a "no solution" result, but a crash or a parse error),
something's wrong with the install, not with the problem.

To see the planner solve the same conflict rather than report it, give agent 2 a
different target cell. That is the next section.

### 7. Verify with a real test problem - solving the conflict

**fixed-problem.pddl**

```lisp
(define (problem two-agents-fixed)
  (:domain conflict-test)
  (:objects
    a1 a2 - agent
    start1 start2 shared other - cell)
  (:init
    (at a1 start1)
    (at a2 start2)
    (free shared)
    (free other))
  (:goal (and (at a1 shared) (at a2 other))))
```

Then run it:

```
./fast-downward.py domain.pddl fixed-problem.pddl --search "astar(lmcut())"
```

Expected: `Solution found. Plan length: 2 step(s).`

## Setup: Fast Downward (Windows, Docker)

On Windows we run the official Fast Downward Docker image instead of
building from source. Nothing is compiled, and no `fast-downward-24.06.1/`
folder is created.

### 1. Install and start Docker Desktop

Install Docker Desktop (WSL 2 backend) if you don't already have it:

    https://www.docker.com/products/docker-desktop/

Start it and wait until the engine is up — the whale icon in the taskbar
stops animating. Confirm:

    docker --version
    docker info

If `docker info` fails with `open //./pipe/dockerDesktopLinuxEngine`,
the engine isn't up yet; give it another minute and retry.

### 2. Pull the image

    docker pull aibasel/downward

If the pull dies mid-download with an EOF error from
`production.cloudfront.docker.com` (seen on some networks), pull through
a mirror and retag:

    docker pull docker.1ms.run/aibasel/downward
    docker tag docker.1ms.run/aibasel/downward:latest aibasel/downward:latest

### 3. Verify with the test problems

The PDDL files are the same ones shown in the macOS sections above and
live under `domains/sanity-check/` in this repo. Only the command changes.
Run these from the repo root in PowerShell.

Solvable:

    docker run --rm -v "${PWD}\domains\sanity-check\trivial:/shared" aibasel/downward --build release /shared/domain.pddl /shared/problem.pddl --search "astar(lmcut())"

Expected: `Solution found. Plan length: 1 step(s).`

Conflict — deliberately unsolvable:

    docker run --rm -v "${PWD}\domains\sanity-check\conflict:/shared" aibasel/downward --build release /shared/domain.pddl /shared/problem.pddl --search "astar(lmcut())"

Expected: `Task is provably unsolvable.` The container exits with code
11 — that is the planner reporting "no plan", not a crash or a broken
install.

Conflict resolved:

    docker run --rm -v "${PWD}\domains\sanity-check\conflict:/shared" aibasel/downward --build release /shared/domain.pddl /shared/fixed-problem.pddl --search "astar(lmcut())"

Expected: `Solution found. Plan length: 2 step(s).`

The `${PWD}` mount syntax is PowerShell. In cmd.exe, replace `${PWD}`
with the absolute path, e.g. `c:\CSE5106\Multi-Agent-Systems`.

### Notes

- Don't commit anything from `fast-downward-24.06.1/` to this repo —
  it's a large external dependency, and the compiled binary is
  platform-specific. (On Windows this folder doesn't exist; the Docker
  image replaces it.)
- Do commit our own `.pddl` files (like the ones above) to this repo,
  under `/domains/` — those are our work, not part of the Fast Downward
  install.
- Every script that runs Fast Downward takes its location as `--fd DIR`
  (set `$FD` once and pass `--fd "$FD"`); only `tests.test_fd` falls back
  to `../fast-downward-24.06.1/`, relative to where you run it, and takes
  the directory as its first argument instead. Pass the path rather than
  committing a path change. On Windows, `experiments.bench_fd --fd-mode exec` drives
  the `aibasel/downward` container itself; `--fd-mode run` executes the local
  Fast Downward Python wrapper. For the
  hand-written PDDL, mount the folder at `/shared` as shown above.

Bulk runs use Claude Haiku 4.5 (`ARBITER_MODEL=claude-haiku-4-5-20251001`), which is about
nine times cheaper per call in measured spend. Sonnet 5.5 was used for a two-instance spot
check only. Do not mix models within one results table.
