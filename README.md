# Multi-Agent-Systems

## Setup: Fast Downward

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

### 5. Verify with a real test problem

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

### Notes

- Don't commit anything from `fast-downward-24.06.1/` to this repo —
  it's a large external dependency, and the compiled binary is
  platform-specific.
- Do commit our own `.pddl` files (like the ones above) to this repo,
  under `/domains/` — those are our work, not part of the Fast Downward
  install.
- Scripts in this repo assume Fast Downward lives at `../fast-downward-24.06.1/`
  relative to the repo root. If your setup differs, adjust paths locally
  rather than committing a path change.