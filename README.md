# Multi-Agent-Systems

CSE 5106 final project: centralized conflict resolution for independently planning agents.

A central agent takes plans it did not write and repairs the collisions between them. We
compare three ways of doing that, all over the same Conflict-Based Search tree:

1. **CBS** — a fixed best-first search. No API calls.
2. **LLM, no tree** — the model picks which agent to constrain, with no search structure.
3. **LLM-guided CBS** — the same tree, with the model choosing which open node to expand.

A **no-LLM control** (approach 3 with the model replaced by "always take the cheapest node")
is what every claim about LLM guidance is measured against.

Around that sits a four-stage pipeline: select `k` agents from their advertised capabilities,
decompose the task into an ordered route per agent, plan each route independently in PDDL with
Fast Downward, then detect and arbitrate the collisions and verify the joint plan. A failed
split backtracks to selection and tries the next covering team.

## Setup

Python 3.10+. Fast Downward 24.06.1 unpacked **next to** this repo (`../fast-downward-24.06.1`);
build instructions are further down.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic
```

Never commit a key. Set it only in your terminal (each new terminal needs it again):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ARBITER_MODEL=claude-haiku-4-5-20251001   # bulk runs; don't mix models in one table
export ARBITER_MAX_CALLS=4000                    # hard stop on API calls per run
# optional: export ARBITER_MAX_TOKENS=4000       (default 4000; the first live runs used 1500,
#                                                 which cut some replies off)
```

## Tests (free, no key)

| Command | What it checks |
|---|---|
| `python3 verify.py` | The independent plan checker (1 good plan, 7 bad plans rejected), and `verify_order`, which checks task order without sharing code with the detector that drives the repair |
| `python3 cbs_baseline.py` | Approach 1: CBS, best-first and the depth-first ablation |
| `python3 test_fd.py` | Approach 1 with Fast Downward as the low-level planner |
| `python3 test_llm.py` | Approach 2 with fake LLMs |
| `python3 test_hybrid.py` | Approach 3 with fake LLMs |
| `python3 instances.py` | The hard test cases and their backtrack counts |
| `python3 agent_pool.py` | Capability advertisement and tag matching |
| `python3 selection.py` | The three selection policies and their ranked alternatives |
| `python3 decompose.py` | Greedy, exhaustive and LLM decomposition |
| `python3 pipeline.py` | The pipeline loop and the backtrack-to-selection edge |
| `python3 pipeline_instances.py` | The capability-overlap domain |
| `python3 colored_box.py` | The colored-box domain, where each agent sees a different grid |
| `python3 check_fd_route.py` | The generated PDDL against `plan_route`, **without needing Fast Downward**. Add `--adversarial` or `--pipeline` |
| `python3 control.py` | Approach 3's search with no LLM, on all 60 benchmark instances |
| `python3 ordering.py` | Cross-agent task order: detection, bounded repair, and the release constraint |
| `python3 check_edge.py` | Standard edge constraints against our stronger vertex rule |
| `python3 check_edge_oracle.py` | Both conflict rules against brute force, so "stronger" can be priced |
| `python3 hard_cbs.py` | Regenerates the hard set by how many nodes best-first CBS itself spends |
| `python3 bench.py` | All three approaches on the hard cases with fake LLMs (harness only) |

One fuzz test, free, that needs neither Fast Downward nor a key:

```bash
python3 fuzz_ordering.py --n 60 --sites 1 --oracle
```

Every instance the order repair gives up on is put to a brute-force joint oracle, which keeps
"incomplete" and "wrong" distinguishable: 55 of 56 feasible single-pair instances solved, 0
unsound, 1 suboptimal by one step. `--sites 2` is the harder case, a total order over four
parts, where the same repair solves only about a third of the feasible instances.

Two diagnostics that need Fast Downward but no key:

```bash
python3 probe_e2e.py --fd "$FD" --fake     # the whole pipeline, stage by stage
python3 check_search.py --fd "$FD"         # does the search configuration change what we get back?
```

## Benchmarks

`$FD` is the Fast Downward directory, e.g. `export FD="$HOME/Desktop/CSE 5106/fast-downward-24.06.1"`.

### Arbitration, fixed team (`bench_fd.py`)

60 instances in three sets, plus `hcb`. Approach 1 only, and free, unless you pass `--live`
or `--fakes`.

```bash
python3 bench_fd.py --set sweep --fd "$FD" --out /tmp/check.csv        # free, Approach 1
python3 bench_fd.py --set hard --fd "$FD" --live --out results/new/hard.csv --log results/new/hard.jsonl
python3 analyze.py results/fd/hard_fd_haiku.csv results/fd/hard_fd_haiku.jsonl   # free: read a finished run
```

`bench.py` takes `--conflicts vertex|standard` too, and there it applies to **all three**
approaches, so the comparison stays like-for-like. Under `bench_fd.py` the same flag reaches
all three as well.

Options: `--set easy|hard|hcb|random|sweep|all`, `--conflicts vertex|standard`, `--start N`,
`--limit N`, `--only NAME[,NAME]`, `--repeat R`, `--search CONFIG`, `--a2-calls N`,
`--a3-calls N`, `--fakes`, `--fd-mode run|exec`.
**Always pass `--out`** — the default overwrites a committed file.

`--set hcb` is the hard set regenerated against best-first CBS itself: an instance is kept when
CBS spends at least 20 expansions on it, rather than when two depth-first rule orders disagree,
which is what filtered the original hard set. It is opt-in; `--set all` does not include it.

`--conflicts standard` swaps our vertex rule for proper edge constraints, which keeps the tree
complete. It reaches all three approaches, so an edge run is a like-for-like comparison and not
only a study of Approach 1. Live LLM runs under edge constraints were not affordable, so the
committed edge results use the scripted policies.

`--fakes` adds the scripted policies to Approaches 2 and 3 at no cost. Use it as a regression
test, not as a result: the policies are deterministic, so any difference between two runs is the
code rather than the model.

### The full pipeline (`pipeline_bench.py`)

```bash
# free: deterministic, no API calls
python3 pipeline_bench.py --fd "$FD" --out results/new/p.csv
python3 pipeline_bench.py --fd "$FD" --k 3 --arbiter hybrid --out results/new/control.csv   # the no-LLM control

# costs money
python3 pipeline_bench.py --fd "$FD" --k 3 --arbiter hybrid --live --out results/new/hybrid.csv
python3 pipeline_bench.py --fd "$FD" --k 3 --arbiter llm --live --out results/new/llm.csv
```

Options: `--domain capability|colored`, `--k N`, `--arbiter cbs|hybrid|llm`,
`--selector tag|llm`, `--decomposer greedy|llm`, `--ordering off|check|repair`,
`--ordering-repairs N`, `--authored`, `--authored-attempts N`, `--authored-lenient`,
`--live`, `--calls N`, `--max-teams N`, `--agents N`, `--size N`, `--walls F`,
`--horizon N`, `--search CONFIG`, `--fd DIR`, `--out FILE`.

`--ordering check` adds cross-agent task order to verification; `--ordering repair` also tries
to fix violations by delaying the later part. Both are free and deterministic, and both work
on either low-level planner and with any arbiter. Ordering sits outside the arbiter, so the
LLM never sees an ordering conflict even under `--live`.

`--authored` replaces the compiled problem file with one the agent writes itself: stage 3 as
the proposal describes it. It needs `--fd`, since Fast Downward solves whatever the agent
produces. Without `--live` a scripted author stands in, which costs nothing and establishes
the ceiling for a given configuration. `--authored-attempts N` re-asks up to N times with the
fault fed back, the repair loop of Smirnov et al.; the default of 1 is one shot.
`--authored-lenient` falls back to the compiled problem when authoring fails, which measures
authoring quality without letting a bad problem decide the run.

`--horizon N` matters for `--authored`: the time-expanded encoding grows with it, and at the
default 5x5 horizon 20 a two-site route problem is about 51k output tokens, beyond what a
model will emit reliably. 4x4 with horizon 12 is about 18k and is where the authored runs
below were done.

Without `--live`, an LLM arbiter runs on a scripted fake that always takes the first option —
which is exactly the no-LLM control, at no cost. **Pass `--out` whenever you change `--k` or
`--agents`**: the default filename does not encode them, so a sweep will overwrite itself.

### Agent-authored PDDL (`authored_pddl.py`)

The proposal's version of the low-level step: the agent writes its own problem file instead of
the harness generating it. Needs `--live` and a key; without it the checker runs on a canned
reply. Results are printed, so redirect them.

```bash
python3 authored_pddl.py --live --cases 20 > results/authored/single.txt
python3 authored_pddl.py --live --cases 12 --routes > results/authored/routes.txt
python3 authored_pddl.py --live --cases 20 --dump /tmp/replies.txt   # read a fault, don't guess it
```

Options: `--cases N`, `--routes`, `--size N`, `--horizon N`, `--seed N`, `--dump FILE`, and
`--fd DIR` to also solve each authored problem for real rather than only checking its shape.
A 5x5 grid at horizon 20 needs roughly 15k output tokens, so raise `ARBITER_MAX_TOKENS` before
a routes run or the replies are cut off mid-problem.

This measures one problem at a time. To run the same thing inside the pipeline, where two
agents must both succeed before arbitration can start, use `pipeline_bench.py --authored`:

```bash
python3 pipeline_bench.py --authored --fd "$FD" --n 6 --strategies cheapest \
  --size 4 --horizon 12 --walls 0.15 --out /tmp/ceiling.csv

export ARBITER_MAX_TOKENS=32000
python3 pipeline_bench.py --authored --live --authored-attempts 2 --fd "$FD" --n 6 \
  --strategies cheapest --size 4 --horizon 12 --walls 0.15 --out /tmp/authored.csv
```

Run the first one first. It uses the scripted author, costs nothing, and tells you what the
configuration scores with a perfect stage 3; without it the live number cannot be read.

Authored problems are cached on (grid, start, route, horizon), as `make_fd_planner` caches its
own, because an agent that reappears in a later team would otherwise be paid for twice. In the
composite run below that suppressed 52 of 67 calls.

### Do the stages compose?

Every stage above was measured on its own, one substitution at a time, which is what makes the
comparisons readable. It is not the same as having run the pipeline. `--selector llm` and
`--decomposer llm` fall back to scripted stand-ins when `--live` is absent, so the whole thing
can be run at once for nothing:

```bash
python3 pipeline_bench.py --selector llm --decomposer llm --arbiter hybrid \
  --ordering repair --authored --fd "$FD" --n 6 --strategies cheapest \
  --size 4 --horizon 12 --walls 0.15 --out /tmp/composite.csv
```

This is an integration test, not an experiment: the stand-ins are not a model and say nothing
about judgement. What it establishes is that the five stages run together and that the authored
planner is indistinguishable from the compiled one when the author is perfect -- the same run
with and without `--authored` is identical row for row. Three of this project's bugs lived in
the seam between two features, which is why it exists.

Create an output folder before writing to it (`mkdir -p results/new`). Ctrl+C is safe:
finished instances are kept.

A run counts as solved only if `verify.py` — an independent replay of every plan, written
without reference to the search code — accepts it.

## Which file is which

| Area | Files |
|---|---|
| Approach 1 (CBS, best-first; `solve` is the depth-first ablation) | `cbs_baseline.py`, `fd_planner.py`, `test_fd.py` |
| Approach 2 (LLM, no tree) | `llm_arbitrator.py`, `test_llm.py` |
| Approach 3 (LLM-guided CBS) | `hybrid.py`, `test_hybrid.py` |
| Pipeline stages 1–2 | `agent_pool.py`, `selection.py`, `llm_select.py`, `decompose.py` |
| Pipeline loop and domains | `pipeline.py`, `pipeline_instances.py`, `colored_box.py` |
| Ordering conflicts | `ordering.py`, `verify.py` (`verify_order`), `fuzz_ordering.py` |
| Evaluation | `instances.py`, `verify.py`, `bench.py`, `bench_fd.py`, `pipeline_bench.py`, `analyze.py`, `control.py` |
| Diagnostics | `probe_e2e.py`, `check_fd_route.py`, `check_search.py`, `check_edge.py`, `check_edge_oracle.py`, `fuzz_ordering.py` |
| Instance generation | `instances.py`, `hard_cbs.py` (the `hcb` set), `pipeline_instances.py` |
| Agent-authored PDDL | `authored_pddl.py` |

## Results folder

| Folder | What it holds |
|---|---|
| `results/fd/` | The main arbitration runs, Fast Downward as the low-level planner: `*_fd_haiku` (the headline runs on all three sets), `random_fd_rep3` (3 repeats), `hard_fd_sat` (satisficing search), `*_fd_equal20` and `*_fd_equal50` (both LLM approaches at the same call cap), `*_fd_edge` (`--conflicts standard`, Approach 1 only), and `hcb_fd_cbs` / `hcb_fd_equal50` / `hcb_fd_edge` (the CBS-regenerated hard set), and `hard_fd_edge_all` (all three approaches under edge constraints, scripted policies). |
| `results/pipeline/` | Pipeline runs: `pipeline_results` (grid planner), `pipeline_fd` (PDDL), `pipeline_fd_hybrid` and `pipeline_fd_hybrid_fake` (LLM arbitration and its control), `pipeline_fd_llmselect` and `pipeline_fd_llmdecomp` (an LLM in place of the selection and decomposition heuristics), `pipeline_colored*`, the `k1`/`k2`/`k3` team-size sweep, and `pipeline_k3_fd_*` (all four arbiters on the conflict-dense configuration). Every pipeline CSV carries a `detail` column saying why the last attempt ended, so a capped arbitration failure is distinguishable from an exhausted one; measured, no committed configuration fails on the node cap. |
| `results/ordering/` | `pipeline_order_check` and `pipeline_order_repair` (grid planner) and `pipeline_order_repair_fd` (Fast Downward). Read against `results/pipeline/pipeline_results.csv` (44/60), the same configuration with ordering off. |
| `results/authored/` | `single`, `single_summary`, `routes` are `.txt`: one problem at a time, standalone. `pipeline_authored_4x4*.csv` are the in-pipeline runs at 4x4 horizon 12 -- `_compiled` is the scripted-author ceiling, the other two are one shot and one repair round. |
| `results/haiku/` | Earlier arbitration runs on our own grid planner, kept so the PDDL numbers can be compared against them. |
| `results/haiku_cap150/` | Re-run of the cases Approach 3 lost at the 50-call cap, with a 150-call cap. |
| `results/sonnet/` | `live_check`: Sonnet 5.5 spot check on 2 cases. |
| `results/control/` | `control.txt`, the output of `control.py`, and `hard_edge_all.csv`: all three approaches on the hard set under standard edge constraints with the scripted policies, to be read against the vertex numbers in `results/haiku/`. |
| `results/first_run_depth_first/` | **First run only, kept for the record.** Made with the earlier depth-first version of Approach 1 and a 1,500-token reply limit that cut some replies off. Do not mix with the tables above. |

Each run has a `.csv` (one row per instance and approach) and, for live runs, a `.jsonl` with
every prompt and reply. Do not mix models in one table.

## Known differences from the proposal

- **PDDL is compiled by default, and authoring was measured end to end rather than skipped.**
  The proposal says each agent *writes* its PDDL under a vocabulary the central agent fixes. By
  default the problem file is generated by the harness from a fixed two-action domain and each
  agent solves its own. Both halves of the proposal's version were tested. One problem at a
  time (`authored_pddl.py`), the model stays inside the fixed vocabulary on 18–19 of 20
  single-goal cases but writes an exactly correct problem on only 4–5 of 20, and 0 of 12
  ordered-route cases. Inside the pipeline (`--authored`), at the benchmark size of 5x5 and
  horizon 20 the architecture does not run at all: a two-site route problem is about 51k output
  tokens and the attempt timed out mid-generation. At 4x4 and horizon 12, where a route problem
  is about 18k, the scripted-author ceiling is 5 of 6 and the model scores 0 of 6 one shot and
  2 of 6 with one repair round; per problem it goes from 17% to 45% once the fault is fed back.
  Of 20 live calls, 13 produced something that did not parse and none produced a problem Fast
  Downward could not solve, so the failure is in writing the file rather than in writing an
  impossible one. An LLM selector and decomposer are implemented and measured as well
  (`--selector llm`, `--decomposer llm`).
- **A verified plan is not a good plan.** One authored problem passed every check we have -- it
  parsed, Fast Downward solved it, and the plan replayed legally on the real grid -- and took 15
  steps where the optimum is 7. The problem was well formed and described the wrong world, and
  nothing downstream can tell. `verify.py` rejects illegal plans, not bad ones, so Approach 1's
  optimality is only ever relative to the problems the agents hand it. This is the strongest
  argument in our results for keeping a compiler in stage 3.
- **Ordering conflicts are implemented; shared resources are not.** Conflict detection covers
  contested cells and cross-agent task order (`ordering.py`): a requirement that one part be
  served before another, which each agent's plan can satisfy alone and the pair can violate
  together. Repair delays the later part and is deliberately incomplete — it never hurries the
  earlier one — and reports that a give-up is not a proof of infeasibility. Shared consumable
  resources, the "tool" of the proposal's conflict-detection paragraph, are still not modelled:
  every conflict we detect is spatial or temporal.
- **Planner independence survives detection but not repair.** Our grid planner and Fast Downward
  agree exactly on all 60 pipeline rows with ordering off (44/60), and exactly again with order
  checking on (21/60, the same eight failing instances). Under `--ordering repair` they diverge,
  39 against 42. Nothing is wrong with either: the repair heuristic is path-dependent, so two
  planners that always agree on plan *cost* can return different cost-equal *paths*, which give
  different service times, which give different releases, and the delays compound. Cost
  equivalence does not imply outcome equivalence once order is what is being measured.
- **The default swap constraint is stronger than standard CBS, and the standard rule is
  available.** By default a swap forbids each agent's destination cell rather than the edge, so
  the tree is incomplete and Approach 1's optimality is relative to our constraint scheme.
  `--conflicts standard` uses proper edge constraints; `check_edge_oracle.py` confirms those
  optimal against brute force on 803 cases. The gap is small but real: the vertex rule is one
  step off the optimum on 21 of 65 brute-forced cases, about 1.3% total slack across 57
  benchmark instances, and pocket-3's true optimum is 7, not the 8 the midterm reported. Edge
  constraints solve 57 of 60 where the vertex rule solves 58, because a complete tree is a
  bigger tree.
- **Approach 3 stops at the first conflict-free plan it generates**, so its plans are not
  guaranteed cheapest.
- **The midterm's solve-count gap between Approaches 2 and 3 was budget, not judgement.** It
  compared Approach 2 at 20 calls with Approach 3 at 50. At matched caps the gap is one instance
  either way: 48 vs 49 at 20 calls, 51 vs 52 at 50, inside Approach 2's own run-to-run variance.
  Plan quality still separates them — on the hard set at 50 calls, 20.5 steps against 14.2 — so
  the claim that survives is about cost per conflict, not about solving more.
- **Which LLM approach wins depends on the instance set.** On the `hcb` set, regenerated against
  best-first CBS itself, the ordering reverses: Approach 2 solves 17 of 20 and Approach 3 solves
  15, because Approach 3 pays one call per node expansion while Approach 2 pays one per conflict,
  and these instances need many expansions. Any claim that one arbiter beats the other should
  name the set it was measured on.
- **The satisficing claim is retracted.** The midterm said optimality is conditional on the
  low-level planner, holding for `astar(lmcut())` but not for a satisficing configuration. It is
  not: `check_search.py` finds identical plans, not merely identical costs, on all 38 of 40
  random cases both configurations solved. The time-expanded encoding fixes plan cost by the goal
  layer reached rather than the route taken, so a satisficing search has nothing to trade.

---

## Setup: Fast Downward (macOS, build from source)

Fast Downward is an external dependency and is **not** part of this repo.
Install it locally, as a sibling directory next to this repo (not inside it):

CSE 5106/
├── Multi-Agent-Systems/     <- this repo
└── fast-downward-24.06.1/   <- installed here, next to the repo

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

(define (domain test)
  (:requirements :strips)
  (:predicates (at-a) (at-b))
  (:action move
    :parameters ()
    :precondition (at-a)
    :effect (and (at-b) (not (at-a)))))

**problem.pddl**

(define (problem test-problem)
  (:domain test)
  (:init (at-a))
  (:goal (at-b)))

### 6. Verify with a real test problem - Conflict

Copy our test domain and problem files (in this repo, under `/domains/sanity-check/`)
into your `fast-downward-24.06.1` folder, or reference them by path:

**domain.pddl**

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

**problem.pddl**

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

This problem is deliberately unsolvable — both agents want the same
exclusive cell, and only one can hold it. Run it:

    ./fast-downward.py domain.pddl problem.pddl --search "astar(lmcut())"

**Expected result: no plan found.** That failure is correct — it's
your basic conflict, confirmed mechanically. If you instead get an
error (not a "no solution" result, but a crash or a parse error),
something's wrong with the install, not with the problem.

If you want to confirm the planner also works on a solvable case, once you're a fixed version of the above so that agent 2 targets a different cell than `shared`.

### 6. Verify with a real test problem - Solving the Conflict

**fixed-problem.pddl**

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

Then run it:

    ./fast-downward.py domain.pddl fixed-problem.pddl --search "astar(lmcut())"

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
- Scripts in this repo assume Fast Downward lives at `../fast-downward-24.06.1/`
  relative to the repo root. If your setup differs, adjust paths locally
  rather than committing a path change. On Windows, invoke the planner
  through the `aibasel/downward` container with the PDDL folder mounted
  at `/shared`, as shown in the Windows section above.

Bulk runs use Claude Haiku 4.5 (`ARBITER_MODEL=claude-haiku-4-5-20251001`), which is about
nine times cheaper per call in measured spend. Sonnet 5.5 was used for a two-instance spot
check only. Do not mix models within one results table.