"""Can an agent write its own PDDL problem, given a vocabulary fixed in advance?

Stage 3 of our pipeline says each agent writes and solves its own PDDL under a
vocabulary the central agent fixes, with no communication. The solving half is
what the rest of the repo does. This measures the writing half.

The central agent fixes: the domain (two actions, one type, four predicates),
the object naming convention c<row>_<col>_<t>, and the horizon. The agent is
told its own start, its own route and the walls, and must emit the problem
file: the objects, the edges that encode legal moves, and the goal copies. It
is told nothing about the other agents, which is what makes the plans
independent.

Two things get measured, and they pull in opposite directions:

  correctness  does the authored problem admit exactly the plans the compiled
               one does? Checked by searching the generated PDDL as a graph,
               so no planner is needed; --fd additionally solves it for real
               and replays the plan on the grid.
  size         the compiled problem's object and fact counts, as the grid and
               horizon grow. This is why the pipeline compiles rather than
               asks: authoring is testable at toy scale and runs out of output
               budget well before the 5x5, horizon-20 instances we benchmark.

    python3 authored_pddl.py                      # free, scaffolding + size table
    python3 authored_pddl.py --live --cases 8     # the real measurement
    python3 authored_pddl.py --live --fd "$FD"    # also solve and replay each one
"""
import argparse
import os
import random
import re
import subprocess
import tempfile

from cbs_baseline import Grid, plan_single, plan_route
from fd_planner import DOMAIN, as_route, make_problem, make_route_problem
from check_fd_route import shortest_in_problem, EDGE, GOAL, START

PROMPT = """You are one agent in a multi-agent system. You plan alone: you know nothing
about the other agents, and you must not reason about them.

A central agent has fixed the vocabulary. Do not invent predicates, types or naming.

DOMAIN (fixed, do not restate it):
{domain}
OBJECT NAMING (fixed): one node per free cell per time step, named c<row>_<col>_<t>
for t from 0 to {horizon}. A cell that is a wall has no node at any time. There are
no other objects.

EDGES (fixed meaning): (edge c<r>_<k>_<t> c<r2>_<k2>_<t+1>) means you may move from
cell (r,k) at time t to cell (r2,k2) at time t+1. A move is to one of the four
orthogonal neighbours or to the same cell, which is how waiting is expressed. Both
cells must be free. There is no edge out of time {horizon}.

GOAL (fixed meaning): (goal-node X) marks a node where you may finish. The problem's
goal is always exactly (:goal (done)).

YOUR TASK
Grid, {h} rows of {w} characters, '.' is free and '#' is a wall, row 0 first:
{rows}
Your start cell: {start}
{goalspec}
Horizon: {horizon}

Write the complete PDDL problem file. Output only the problem file, starting with
"(define (problem" and ending with the matching ")". No commentary, no code fences."""

SINGLE = "Your goal cell: {goal}. You may finish at your goal cell at any time from {first_ok} to {horizon}."
ROUTE = ("Your route, which you must visit in this order: {route}. Because a node records only "
         "a cell and a time, you cannot express the order with these objects alone unless you "
         "add the number of sites served to the name: use c<row>_<col>_<served>_<t>, where "
         "<served> counts the route sites already entered, and entering route site number "
         "<served> is what increases it. You may finish only on the last site with all "
         "{n} served.")


def fence_strip(text):
    """The model is told not to fence the output; some replies fence it anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text.strip())
    start = text.find("(define")
    return text[start:] if start > 0 else text


def flatten(problem):
    """One space between tokens. PDDL does not care where the newlines are and
    neither should these checks: a file laid out over several lines is valid,
    and an earlier version of this script failed every reply for formatting."""
    return re.sub(r"\s+", " ", problem).strip()


OBJECTS = re.compile(r"\(:objects\s+(.*?)\s*-\s*node\s*\)", re.S)
DONE = re.compile(r"\(:goal\s*\(done\)\s*\)")


def structural(problem):
    """Cheap sanity checks before the expensive one. Returns a list of faults."""
    if problem is None or "(define" not in problem:
        return ["not a problem file"]
    flat = flatten(problem)
    faults = []
    if not re.search(r"\(:domain\s+timed-grid\s*\)", flat):
        faults.append("wrong or missing :domain")
    if not DONE.search(flat):
        faults.append("goal is not (:goal (done))")
    if not START.search(flat):
        faults.append("no (at ...) in :init")
    if not EDGE.findall(flat):
        faults.append("no edges")
    if not GOAL.findall(flat):
        faults.append("no goal-node")
    m = OBJECTS.search(flat)
    if not m:
        faults.append("no (:objects ... - node)")
    else:
        declared = set(m.group(1).split())
        refs = set()
        for a, b in EDGE.findall(flat):
            refs |= {a, b}
        refs |= set(GOAL.findall(flat))
        s = START.search(flat)
        if s:
            refs.add(s.group(1))
        undeclared = refs - declared
        if undeclared:
            faults.append(f"{len(undeclared)} undeclared objects, e.g. {sorted(undeclared)[:2]}")
    return faults


def legal_on_grid(path, grid, start, route):
    """Replay the plan on the real grid: the check the PDDL cannot do for itself."""
    if not path or path[0] != start:
        return "does not start at the start cell"
    for a, b in zip(path, path[1:]):
        if not grid.free(b) or b not in grid.neighbors(a):
            return f"illegal move {a} -> {b}"
    served = 0
    for cell in path:
        if served < len(route) and cell == route[served]:
            served += 1
    if served < len(route):
        return f"visited only {served} of {len(route)} sites in order"
    return None


def solve_with_fd(fd_dir, problem, search="astar(lmcut())"):
    """Run Fast Downward on an authored problem. Returns the cell path, or None."""
    fd = os.path.join(fd_dir, "fast-downward.py")
    with tempfile.TemporaryDirectory() as d:
        dom, prob, out = (os.path.join(d, f) for f in ("d.pddl", "p.pddl", "plan"))
        open(dom, "w").write(DOMAIN)
        open(prob, "w").write(problem)
        res = subprocess.run([fd, "--plan-file", out, "--sas-file", os.path.join(d, "o.sas"),
                              dom, prob, "--search", search],
                             capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(out):
            path = []
            for line in open(out):
                if line.startswith("(move"):
                    to = line.split()[2].rstrip(")")
                    r, k = to[1:].split("_")[:2]
                    path.append((int(r), int(k)))
            return path
        return None if res.returncode in (10, 11, 12) else "ERROR"


def build_case(rng, size, horizon, want_route):
    rows = ["".join("#" if rng.random() < 0.15 else "." for _ in range(size))
            for _ in range(size)]
    grid = Grid(rows)
    free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
    if len(free) < 4:
        return None
    start = rng.choice(free)
    route = rng.sample(free, 2) if want_route else [rng.choice(free)]
    ref = (plan_route if len(route) > 1 else plan_single)(
        grid, start, route if len(route) > 1 else route[0], frozenset(), horizon)
    # Skip cases the agent could satisfy by standing still: a zero-cost problem
    # tests nothing about whether it understood the edges.
    return (grid, start, tuple(route), horizon) if ref and len(ref) - 1 >= 2 else None


def prompt_for(grid, start, route, horizon):
    rows = "\n".join("".join("#" if not grid.free((r, c)) else "." for c in range(grid.w))
                     for r in range(grid.h))
    if len(route) == 1:
        goalspec = SINGLE.format(goal=route[0], first_ok=0, horizon=horizon)
    else:
        goalspec = ROUTE.format(route=", ".join(map(str, route)), n=len(route))
    return PROMPT.format(domain=DOMAIN, horizon=horizon, h=grid.h, w=grid.w,
                         rows=rows, start=start, goalspec=goalspec)


def reference(grid, start, route, horizon):
    return (make_problem(grid, start, route[0], frozenset(), horizon) if len(route) == 1
            else make_route_problem(grid, start, route, frozenset(), horizon))


def size_table():
    """How big the problem an agent would have to write actually is.

    Two sites, not one, is the case that settles whether authoring is possible:
    the layered encoding copies every cell once per site served, so a route
    problem is several times a single-goal one. The rightmost column is the
    number that matters, because it is what the agent has to emit in one reply
    and models cap their output in tokens, not characters.
    """
    from fd_planner import make_route_problem
    print("\nWhat the compiled problem costs, which is why the pipeline compiles:")
    print(f"  {'grid':>6} {'horizon':>8} {'goal':>8} {'objects':>9} {'edges':>8} "
          f"{'chars':>9} {'~tokens':>9}")
    for size, horizon in ((3, 5), (3, 8), (4, 8), (4, 12), (5, 12), (5, 20)):
        g = Grid(["." * size] * size)
        far = (size - 1, size - 1)
        cases = [("single", make_problem(g, (0, 0), far, frozenset(), horizon)),
                 ("2 sites", make_route_problem(g, (0, 0), ((0, size - 1), far),
                                                frozenset(), horizon))]
        for label, p in cases:
            if not p:
                continue
            objs = len(re.search(r"\(:objects (.*?) - node\)", p, re.S).group(1).split())
            print(f"  {str(size)+'x'+str(size):>6} {horizon:>8} {label:>8} {objs:>9} "
                  f"{len(EDGE.findall(p)):>8} {len(p):>9} {len(p)//4:>9,}")
    print("  Our benchmark instances are 5x5 at horizon 20. A two-site route problem there")
    print("  is past what a model will emit in one reply, which is why --authored runs")
    print("  were done at 4x4 horizon 12 instead.")


REPAIR = """Your previous attempt was rejected. Fix it and return the corrected problem file.

WHAT WAS WRONG
{fault}

WHAT YOU WROTE
{reply}

THE ORIGINAL TASK, UNCHANGED
{prompt}

Output only the corrected problem file, starting with "(define (problem" and ending
with the matching ")". No commentary, no code fences."""


def make_authored_planner(fd_dir, llm, base, search="astar(lmcut())", stats=None,
                          strict=True, attempts=1):
    """A planner that asks the agent to write its own PDDL, as the proposal says.

    Same signature as every other planner, so run_pipeline cannot tell the
    difference: (grid, start, goal, constraints, horizon).

    The authoring prompt has no way to express an arbitration constraint -- it
    describes a grid, a start and a route, and nothing else -- so a call that
    arrives carrying constraints falls through to `base`, the compiled encoding.
    That is the honest division of the question. Stage 3 asks whether an agent
    can state its own problem; restating it under forty accumulated constraints
    is a different and much harder question, and one the measured authoring
    accuracy says is not worth the API spend.

    strict=True is the proposal's reading: an agent that cannot write a usable
    problem has no plan, and the arbiter treats that as it treats any other
    agent with no plan. strict=False falls back to the compiled encoding and
    only records what authoring would have done, which measures quality without
    letting a bad problem decide the run.

    attempts > 1 re-asks with the fault fed back, the repair loop Smirnov et al.
    describe. One shot is the harsher reading and the cheaper one; saying which
    was used matters, because "cannot write it" and "cannot write it even when
    told what broke" are different claims.
    """
    seen = {} if stats is None else stats.setdefault("authored", {})
    for key in ("calls", "ok", "malformed", "unsolvable", "illegal",
                "fallback_constrained", "ok_first", "ok_repair", "problems"):
        seen.setdefault(key, 0)

    def plan(grid, start, goal, constraints, horizon):
        if constraints:
            seen["fallback_constrained"] += 1
            return base(grid, start, goal, constraints, horizon)

        route = as_route(goal)
        first = prompt_for(grid, start, route, horizon)
        seen["problems"] += 1
        reply = fault = None

        for attempt in range(1, attempts + 1):
            seen["calls"] += 1
            ask = first if attempt == 1 else REPAIR.format(fault=fault, reply=reply,
                                                           prompt=first)
            problem = fence_strip(llm(ask))

            faults = structural(problem)
            if faults:
                seen["malformed"] += 1
                reply, fault = problem, "; ".join(faults)
                continue

            path = solve_with_fd(fd_dir, problem, search)
            if path is None or path == "ERROR":
                seen["unsolvable"] += 1
                reply = problem
                fault = ("Fast Downward found no plan for this problem, so some fact "
                         "it needs is missing or wrong.")
                continue

            # solve_with_fd returns the destination of each move, so the start cell
            # is not in it; main() prepends it for the same reason. Every planner in
            # this repo returns a path that begins at the start, and the arbiter
            # costs a plan as len(path) - 1, so it has to be prepended here too.
            path = [start] + path
            why = legal_on_grid(path, grid, start, route)
            if why:
                seen["illegal"] += 1
                reply = problem
                fault = (f"the problem is well formed and solvable, but the plan it "
                         f"admits is wrong on the real grid: {why}")
                continue

            seen["ok"] += 1
            seen["ok_first" if attempt == 1 else "ok_repair"] += 1
            return path

        return None if strict else base(grid, start, goal, constraints, horizon)

    return _Authored(plan, base, seen)


class _Authored:
    """Callable planner that still reports `calls` the way make_fd_planner does.

    The bench reads planner.calls for its fd_calls column, so a planner that is
    only a function breaks it. Fast Downward runs here in two places -- once per
    authored problem, and once per constrained call that falls through to the
    compiled planner -- and both belong in that count.
    """

    def __init__(self, plan, base, seen):
        self._plan, self._base, self._seen = plan, base, seen

    def __call__(self, *args):
        return self._plan(*args)

    @property
    def calls(self):
        return self._seen["calls"] + getattr(self._base, "calls", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="ask the real model")
    ap.add_argument("--fd", default=None, help="also solve each authored problem for real")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--routes", action="store_true", help="two-site routes instead of one goal")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--dump", default=None,
                    help="write every reply here, so a fault can be read rather than guessed")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cases = []
    while len(cases) < a.cases:
        made = build_case(rng, a.size, a.horizon, a.routes)
        if made:
            cases.append(made)

    if a.live:
        from llm_arbitrator import AnthropicLLM
        llm = AnthropicLLM()
        author = llm
        label = llm.model
    else:
        # The scaffolding check: hand back the compiled problem. Everything below
        # must pass, or a live failure cannot be blamed on the model.
        author = None
        label = "compiled reference (scaffolding check, no API calls)"

    print(f"{len(cases)} cases, {a.size}x{a.size} grids, horizon {a.horizon}, "
          f"{'two-site routes' if a.routes else 'single goals'}")
    print(f"author: {label}\n")

    tally = {"malformed": 0, "no plan": 0, "wrong cost": 0, "illegal": 0, "correct": 0}
    for i, (grid, start, route, horizon) in enumerate(cases, 1):
        ref = reference(grid, start, route, horizon)
        want = shortest_in_problem(ref)
        if author is None:
            got = ref
        else:
            got = fence_strip(author(prompt_for(grid, start, route, horizon)))

        if a.dump:
            with open(a.dump, 'a') as fh:
                fh.write(f"\n{'='*60}\ncase {i}  start {start} route {route}\n{'='*60}\n{got}\n")
        faults = structural(got)
        if faults:
            tally["malformed"] += 1
            print(f"  {i:>2}. malformed: {'; '.join(faults[:2])}")
            continue
        cost = shortest_in_problem(flatten(got))
        if cost is None:
            tally["no plan"] += 1
            print(f"  {i:>2}. admits no plan (reference cost {want})")
            continue
        if cost != want:
            tally["wrong cost"] += 1
            print(f"  {i:>2}. cost {cost}, reference {want}")
            continue
        if a.fd:
            path = solve_with_fd(a.fd, got)
            if path in (None, "ERROR"):
                tally["no plan"] += 1
                print(f"  {i:>2}. Fast Downward could not use it")
                continue
            why = legal_on_grid([start] + path, grid, start, route)
            if why:
                tally["illegal"] += 1
                print(f"  {i:>2}. plan is not legal on the grid: {why}")
                continue
        tally["correct"] += 1
        print(f"  {i:>2}. correct (cost {cost})")

    print(f"\n  correct {tally['correct']}/{len(cases)}   " +
          "   ".join(f"{k} {v}" for k, v in tally.items() if k != "correct" and v))
    if author is None and tally["correct"] != len(cases):
        print("\nSCAFFOLDING BROKEN: the compiled problem failed its own checks.")
    size_table()


if __name__ == "__main__":
    main()