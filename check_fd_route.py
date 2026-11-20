"""Check fd_planner's PDDL against cbs_baseline.plan_route, without Fast Downward.

cbs_baseline.plan_route searches (cell, served, time) directly and is the
authority on what a route costs. fd_planner has to express the same thing as a
PDDL problem. This script reads the problem fd_planner generates, searches it
as a graph, and compares the shortest plan it admits against plan_route.

That isolates the encoding, which is the part most likely to be wrong, from
Fast Downward, which already drives 360 verified benchmark rows. A pass here
means any later disagreement is the planner or the harness, not the PDDL.

    python3 check_fd_route.py                  # random cases, routes and single goals
    python3 check_fd_route.py --pipeline       # the real pipeline's multi-site goals
    python3 check_fd_route.py --adversarial    # constraints piled on the waypoints
"""
import argparse
import random
import re
from collections import deque

from cbs_baseline import Grid, plan_route
from fd_planner import as_route, make_problem, make_route_problem

# [^)\s]+ not \S+: the last fact in (:init ...) is followed by that block's own
# closing paren, so a greedy match swallows it.
EDGE = re.compile(r"\(edge ([^)\s]+) ([^)\s]+)\)")
GOAL = re.compile(r"\(goal-node ([^)\s]+)\)")
START = re.compile(r"\(at ([^)\s]+)\)")


def shortest_in_problem(problem):
    """Fewest `move` actions the problem admits, or None if it admits no plan.

    Fast Downward with astar(lmcut()) returns an optimal plan, and every move
    costs one, so this is what it would report.
    """
    if problem is None:
        return None
    adjacency = {}
    for a, b in EDGE.findall(problem):
        adjacency.setdefault(a, []).append(b)
    goals = set(GOAL.findall(problem))
    start = START.search(problem).group(1)
    if not goals:
        return None
    seen, q = {start}, deque([(start, 0)])
    while q:
        node, d = q.popleft()
        if node in goals:
            return d
        for n in adjacency.get(node, ()):
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return None


def encode(grid, start, goal, constraints, horizon):
    """What fd_planner would hand Fast Downward, for either goal shape."""
    route = as_route(goal)
    if (start, 0) in constraints:
        return None
    if len(route) == 1:
        return make_problem(grid, start, route[0], constraints, horizon)
    return make_route_problem(grid, start, route, constraints, horizon)


def compare(grid, start, route, constraints, horizon):
    """('agree'|'longer'|'shorter'|'missed'|'phantom', pddl_len, route_len)."""
    want = plan_route(grid, start, list(route), constraints, horizon)
    want = None if want is None else len(want) - 1
    got = shortest_in_problem(encode(grid, start, route, constraints, horizon))
    if want is None and got is None:
        return "agree", None, None
    if want is not None and got is None:
        return "missed", got, want          # PDDL admits no plan but one exists
    if want is None and got is not None:
        return "phantom", got, want         # PDDL admits a plan that is not legal
    if got == want:
        return "agree", got, want
    return ("longer" if got > want else "shorter"), got, want


def random_case(rng, size, horizon, adversarial):
    rows = ["".join("#" if rng.random() < (0.22 if adversarial else 0.18) else "."
                    for _ in range(size)) for _ in range(size)]
    grid = Grid(rows)
    free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
    if len(free) < 5:
        return None
    start = rng.choice(free)
    route = [rng.choice(free) for _ in range(rng.randint(2, 5) if adversarial
                                             else rng.randint(1, 3))]
    constraints = set()
    if adversarial:
        # Block the waypoints themselves at many times: this is what defeats
        # solving a route leg by leg, so it is what the encoding must survive.
        for w in route:
            for t in range(rng.randint(0, 12)):
                if rng.random() < 0.55:
                    constraints.add((w, t))
    for _ in range(rng.randint(0, 14 if adversarial else 8)):
        constraints.add((rng.choice(free), rng.randint(0, horizon)))
    return grid, start, tuple(route), frozenset(constraints), horizon


def pipeline_routes(horizon):
    """The goals the real pipeline produces, multi-site ones included."""
    from decompose import greedy_decompose
    from pipeline_instances import capability_overlap
    from selection import select
    out = []
    for inst in capability_overlap(20, seed=7):
        for strategy in ("cheapest", "greedy", "coverage"):
            chosen = select(inst.pool, inst.task, 2, strategy)
            goals, _ = greedy_decompose(inst.task, chosen.team, inst.sites)
            if not goals:
                continue
            members = {a.name: a for a in chosen.team}
            for name, route in goals.items():
                out.append((inst.grid, members[name].start, tuple(route),
                            frozenset(), horizon))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--adversarial", action="store_true")
    a = ap.parse_args()

    if a.pipeline:
        cases = pipeline_routes(a.horizon)
        print(f"{len(cases)} real pipeline goals")
    else:
        rng = random.Random(a.seed)
        cases = []
        while len(cases) < a.cases:
            made = random_case(rng, a.size, a.horizon, a.adversarial)
            if made:
                cases.append(made)
        kind = "adversarial" if a.adversarial else "random"
        print(f"{len(cases)} {kind} cases, grids up to {a.size}x{a.size}")

    tally = {}
    examples = []
    singles = sum(1 for c in cases if len(c[2]) == 1)
    for grid, start, route, constraints, horizon in cases:
        verdict, got, want = compare(grid, start, route, constraints, horizon)
        tally[verdict] = tally.get(verdict, 0) + 1
        if verdict != "agree" and len(examples) < 4:
            examples.append((verdict, len(route), got, want))

    print(f"  single-cell goals: {singles}   routes: {len(cases) - singles}")
    for k in ("agree", "longer", "shorter", "missed", "phantom"):
        if tally.get(k):
            print(f"  {k:8s} {tally[k]}")
    for verdict, n, got, want in examples:
        print(f"    {verdict}: waypoints={n} pddl={got} plan_route={want}")
    print()
    if set(tally) <= {"agree"}:
        print("PASS: the PDDL admits exactly plan_route's optimal cost on every case.")
    else:
        print("FAIL: the encoding and plan_route disagree. Details above.")
        print("  missed  = PDDL finds no plan where one exists (would surface as a")
        print("            phantom decomposition or arbitration failure)")
        print("  phantom = PDDL admits a plan plan_route rejects (unsound)")


if __name__ == "__main__":
    main()