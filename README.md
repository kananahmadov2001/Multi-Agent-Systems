# Multi-Agent-Systems

CSE 5106 final project: centralized conflict resolution for independently planning agents.
A central agent resolves collisions between agents' plans in three ways, all on the same
Conflict-Based-Search tree: (1) a fixed best-first search (CBS), (2) an LLM that picks who to
constrain, with no tree, and (3) the same tree with the LLM choosing which node to expand next.

## Project code: how to run Approaches 1-3

Needs Python 3.10+. Approach 1 with the Fast Downward planner also needs Fast Downward 24.06.1 unpacked **next to** this repo (`../fast-downward-24.06.1`).

### Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic
```
Never commit a key. Set it only in your terminal (each new terminal needs it again):
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ARBITER_MAX_CALLS=300      # hard stop on API calls per run (about $0.01 per call)
# optional: export ARBITER_MODEL=claude-haiku-4-5-20251001   (cheaper; don't mix models in one table)
# optional: export ARBITER_MAX_TOKENS=4000   (default 4000; the first live runs used 1500, which cut some replies off)
```

### Tests (free, no key)
| Command | What it checks |
|---|---|
| `python3 cbs_baseline.py` | Approach 1: Conflict-Based Search, depth-first and best-first (toy cases) |
| `python3 test_fd.py` | Approach 1 with Fast Downward as the low-level planner |
| `python3 test_llm.py` | Approach 2 with fake LLMs |
| `python3 test_hybrid.py` | Approach 3 with fake LLMs |
| `python3 verify.py` | The independent plan checker (1 good plan, 7 bad plans rejected) |
| `python3 instances.py` | The hard test cases and their backtrack counts |
| `python3 bench.py` | All three approaches on the 25 hard cases, fake LLMs (checks the harness only) |

### Real model (costs money)
```bash
python3 test_llm.py --live            # Approach 2, real model, 2 toy cases
python3 test_hybrid.py --live         # Approach 3, real model, 2 toy cases
python3 bench.py --live --limit 3     # first 3 benchmark cases (about $1)
python3 bench.py --live --start 5 --limit 8 --out results/new/batch.csv --log results/new/batch.jsonl   # next batch, with transcripts
python3 bench.py --live --only pocket-5-mid --repeat 3 --out results/new/var.csv --log results/new/var.jsonl   # variance
python3 analyze.py results/haiku/hard_haiku.csv results/haiku/hard_haiku.jsonl   # free: read a finished run
python3 control.py                    # free: Approach 3's search with no LLM (always the cheapest node)
```
`bench.py` options: `--set easy|hard|random|sweep|all` (random = held-out unfiltered cases, sweep = 2/3/4 agents), `--start N`, `--limit N`, `--only NAME[,NAME]`, `--repeat R`, `--out FILE`, `--log FILE`, `--dfs` (also run the depth-first ablation).
Use a different `--out` for each batch (the folder must exist: `mkdir -p results/new`). Ctrl+C is safe: finished instances are kept.
A run counts as solved only if `verify.py` (an independent replay of all plans) accepts it.

### Which file is which
- Approach 1 (Conflict-Based Search, best-first; `solve` is the depth-first ablation): `cbs_baseline.py`, `fd_planner.py`, `test_fd.py`
- Approach 2 (LLM, no tree, sees its own earlier decisions): `llm_arbitrator.py`, `test_llm.py`
- Approach 3 (LLM-guided CBS: the LLM picks which open node to expand next, so it also decides when to backtrack): `hybrid.py`, `test_hybrid.py`
- Evaluation: `instances.py`, `verify.py`, `bench.py`, `analyze.py`, `control.py`

### Results folder
| Folder | What it holds |
|---|---|
| `results/haiku/` | Main runs, model `claude-haiku-4-5-20251001`: `hard_haiku` (25 hard cases), `random_haiku` (20 unfiltered cases), `sweep_haiku` (2/3/4 agents), `var_haiku` (10 hard cases x 3 repeats). Caps: 20 calls (Approach 2), 50 calls (Approach 3). |
| `results/haiku_cap150/` | Re-run of the cases Approach 3 lost at the 50-call cap, with a 150-call cap (Haiku). |
| `results/sonnet/` | `live_check`: Sonnet 5.5 spot check on 2 cases. A larger Sonnet comparison is planned. |
| `results/control/` | `control.txt`: output of `control.py`, Approach 3's search with no LLM. |
| `results/first_run_depth_first/` | **First run only, kept for the record.** Made with the earlier depth-first version of Approach 1 and a 1,500-token reply limit that cut some replies off. Do not mix with the tables above. |

Each run has a `.csv` (one row per instance and approach) and a `.jsonl` (every prompt and reply). Do not mix models in one table.

### Known differences from the proposal
- Agent selection, decomposition, agent-written PDDL with a fixed vocabulary, and the capability-overlap / colored-box domains are not built yet (planned for the final paper).
- The live runs use our own grid planner as the low-level planner; Fast Downward is checked against it (`test_fd.py`) but not yet used in the live runs.
- Approach 3 stops at the first conflict-free plan it generates, so its plans are not guaranteed cheapest (Approach 1 is optimal).

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

Using Claude Sonnet 5.5 for initial tests, can see if we want to compare other models if costs permit, otherwise Sonnet 5.5 is a good benchmark