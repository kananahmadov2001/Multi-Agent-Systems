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
| `python3 verify.py` | The independent plan checker (1 good plan, 7 bad plans rejected) |
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
| `python3 bench.py` | All three approaches on the hard cases with fake LLMs (harness only) |

Two diagnostics that need Fast Downward but no key:

```bash
python3 probe_e2e.py --fd "$FD" --fake     # the whole pipeline, stage by stage
python3 check_search.py --fd "$FD"         # does the search configuration change what we get back?
```

## Benchmarks

`$FD` is the Fast Downward directory, e.g. `export FD="$HOME/Desktop/CSE 5106/fast-downward-24.06.1"`.

### Arbitration, fixed team (`bench_fd.py`)

60 instances in three sets. Approach 1 only, and free, unless you pass `--live` or `--fakes`.

```bash
python3 bench_fd.py --set sweep --fd "$FD" --out /tmp/check.csv        # free, Approach 1
python3 bench_fd.py --set hard --fd "$FD" --live --out results/new/hard.csv --log results/new/hard.jsonl
python3 analyze.py results/fd/hard_fd_haiku.csv results/fd/hard_fd_haiku.jsonl   # free: read a finished run
```

Options: `--set easy|hard|random|sweep|all`, `--start N`, `--limit N`, `--only NAME[,NAME]`,
`--repeat R`, `--search CONFIG`, `--a2-calls N`, `--a3-calls N`, `--fd-mode run|exec`.
**Always pass `--out`** — the default overwrites a committed file.

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
`--selector tag|llm`, `--decomposer greedy|llm`, `--live`, `--calls N`, `--max-teams N`,
`--agents N`, `--size N`, `--walls F`, `--search CONFIG`, `--fd DIR`, `--out FILE`.

Without `--live`, an LLM arbiter runs on a scripted fake that always takes the first option —
which is exactly the no-LLM control, at no cost. **Pass `--out` whenever you change `--k` or
`--agents`**: the default filename does not encode them, so a sweep will overwrite itself.

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
| Evaluation | `instances.py`, `verify.py`, `bench.py`, `bench_fd.py`, `pipeline_bench.py`, `analyze.py`, `control.py` |
| Diagnostics | `probe_e2e.py`, `check_fd_route.py`, `check_search.py` |

## Results folder

| Folder | What it holds |
|---|---|
| `results/fd/` | The main arbitration runs, Fast Downward as the low-level planner: `hard_fd_haiku`, `random_fd_haiku`, `sweep_fd_haiku`, `random_fd_rep3` (3 repeats), `hard_fd_sat` (satisficing search). |
| `results/pipeline/` | Pipeline runs. `pipeline_results` (grid planner), `pipeline_fd` (PDDL), `pipeline_fd_hybrid` and `pipeline_fd_hybrid_fake` (LLM arbitration and its control), `pipeline_colored`, the `k1`/`k2`/`k3` team-size sweep, and `pipeline_k3_fd_*` (all four arbiters on the conflict-dense configuration). |
| `results/haiku/` | Earlier arbitration runs on our own grid planner, kept so the PDDL numbers can be compared against them. |
| `results/haiku_cap150/` | Re-run of the cases Approach 3 lost at the 50-call cap, with a 150-call cap. |
| `results/sonnet/` | `live_check`: Sonnet 5.5 spot check on 2 cases. |
| `results/control/` | `control.txt`: output of `control.py`. |
| `results/first_run_depth_first/` | **First run only, kept for the record.** Made with the earlier depth-first version of Approach 1 and a 1,500-token reply limit that cut some replies off. Do not mix with the tables above. |

Each run has a `.csv` (one row per instance and approach) and, for live runs, a `.jsonl` with
every prompt and reply. Do not mix models in one table.

## Known differences from the proposal

- **PDDL is compiled, not authored.** The proposal says each agent *writes* its PDDL under a
  vocabulary the central agent fixes. Each agent *solves* its own PDDL, but the problem file is
  generated by the harness from a fixed two-action domain. An LLM selector and decomposer are
  implemented (`--selector llm`, `--decomposer llm`); an LLM that writes PDDL is not.
- **Conflicts are contested cells only.** We did not grow conflict detection to resource and
  ordering conflicts.
- **Our swap constraint is stronger than standard CBS.** A swap forbids each agent's destination
  cell rather than the edge, so the tree is incomplete and Approach 1's optimality is relative to
  our constraint scheme, not to the instance.
- **Approach 3 stops at the first conflict-free plan it generates**, so its plans are not
  guaranteed cheapest.
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