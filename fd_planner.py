"""
Low-level planner backed by Fast Downward. Drop-in replacement for
cbs_baseline.plan_single, including its ordered-waypoint goals.

Trick for "forbid (cell, t)": PDDL has no clock, so every cell is copied once
per time step (object c<row>_<col>_<t>). A move goes from (cell, t) to
(neighbor, t+1). A forbidden (cell, t) simply has no copy, so no move can
enter it. An edge ban (from, to, t) omits that one edge fact instead. Waiting
is a move from (cell, t) to (cell, t+1).

A goal may be one cell or an ordered list of cells to visit in order, the
second form being what cbs_baseline.plan_route accepts. The two are encoded
differently on purpose:

  * One cell: the encoding above, unchanged. This is the only path a
    single-goal benchmark ever takes, so those results are unaffected.
  * A route: cells are copied once per (waypoint served, time) instead, giving
    objects c<row>_<col>_<served>_<time>. Entering route[served] is what moves
    a plan into the next layer, which mirrors plan_route's (cell, served, time)
    search and so is complete and optimal over the whole route. Solving a route
    leg by leg and taking the earliest arrival at each waypoint is neither: it
    can miss a plan that exists when waiting on a waypoint is forbidden.

Objects grow as free cells x (len(route) + 1) x (horizon + 1), so a route costs
roughly len(route) + 1 times a single goal. Routes are a small minority of
goals in our domains, which is what makes that affordable.
"""
import os
import subprocess
import tempfile

from cbs_baseline import Grid, _is_waypoints, _last_vertex_block   # shared goal and ban rules

DOMAIN = """(define (domain timed-grid)
  (:requirements :strips :typing)
  (:types node)
  (:predicates (at ?n - node) (edge ?a ?b - node) (goal-node ?n - node) (done))
  (:action move
    :parameters (?a ?b - node)
    :precondition (and (at ?a) (edge ?a ?b))
    :effect (and (not (at ?a)) (at ?b)))
  (:action finish
    :parameters (?n - node)
    :precondition (and (at ?n) (goal-node ?n))
    :effect (done)))
"""


def _name(cell, t):
    return f"c{cell[0]}_{cell[1]}_{t}"


def _rname(cell, served, t):
    return f"c{cell[0]}_{cell[1]}_{served}_{t}"


def as_route(goal, constraints=frozenset()):
    """A goal as a tuple of cells, consecutive repeats dropped.

    Matches plan_route, which serves waypoints in order and so gains nothing
    from a repeat of the cell it is already standing on.

    A release constraint names a waypoint by its index, so dropping a repeat
    would renumber the waypoints underneath it. When any release is present
    the route is left exactly as given, which is what plan_route does too;
    the two must agree on what index 2 means or the planners diverge.
    """
    route = tuple(goal) if _is_waypoints(goal) else (goal,)
    if not releases_of(constraints):
        route = tuple(c for i, c in enumerate(route) if i == 0 or c != route[i - 1])
    if not route:
        raise ValueError("a goal must be one cell or a non-empty list of cells")
    return route


def releases_of(constraints):
    """{waypoint index: earliest service time} from ("release", index, time)."""
    return {e[1]: e[2] for e in constraints
            if len(e) == 3 and e[0] == "release"}


def _problem(objects, start, facts):
    return (f"(define (problem p) (:domain timed-grid)\n"
            f"  (:objects {' '.join(objects)} - node)\n"
            f"  (:init (at {start})\n    " + "\n    ".join(facts) +
            f")\n  (:goal (done)))\n")


def make_problem(grid, start, goal, constraints, horizon):
    """One cell as the goal. The agent parks there, so it may only finish once
    that cell is forbidden no longer, the rule cbs_baseline.plan_single uses."""
    last_block = max(_last_vertex_block(constraints, goal),
                     releases_of(constraints).get(0, 0) - 1)
    if last_block >= horizon:
        return None
    nodes, edges = [], []
    for t in range(horizon + 1):
        for r in range(grid.h):
            for k in range(grid.w):
                if grid.free((r, k)) and ((r, k), t) not in constraints:
                    nodes.append(_name((r, k), t))
    for t in range(horizon):
        for r in range(grid.h):
            for k in range(grid.w):
                a = (r, k)
                if not grid.free(a) or (a, t) in constraints:
                    continue
                for n in grid.neighbors(a):
                    if (n, t + 1) not in constraints and (a, n, t) not in constraints:
                        edges.append(f"(edge {_name(a, t)} {_name(n, t + 1)})")
    goals = [f"(goal-node {_name(goal, t)})"
             for t in range(last_block + 1, horizon + 1)
             if (goal, t) not in constraints]
    return _problem(nodes, _name(start, 0), edges + goals)


def make_route_problem(grid, start, route, constraints, horizon):
    """An ordered route as the goal, as one problem rather than one per leg.

    A cell copy carries how many waypoints have been served, and entering
    route[served] advances that count, so a plan cannot serve them out of
    order or skip one. The agent parks on the last waypoint, which is why the
    goal copies start after the last time that cell is forbidden.
    """
    last = len(route)
    releases = releases_of(constraints)
    if releases and any(x == y for x, y in zip(route, route[1:])):
        return None              # same guard plan_route uses: no distinct arrival
    last_block = max(_last_vertex_block(constraints, route[-1]),
                     releases.get(last - 1, 0) - 1)
    if last_block >= horizon:
        return None
    cells = [(r, k) for r in range(grid.h) for k in range(grid.w) if grid.free((r, k))]
    nodes = [_rname(c, s, t)
             for t in range(horizon + 1) for s in range(last + 1) for c in cells
             if (c, t) not in constraints]
    edges = []
    for t in range(horizon):
        for s in range(last + 1):
            for a in cells:
                if (a, t) in constraints:
                    continue
                for n in grid.neighbors(a):
                    if (n, t + 1) in constraints or (a, n, t) in constraints:
                        continue
                    nxt = s + 1 if s < last and n == route[s] else s
                    if nxt != s and t + 1 < releases.get(s, 0):
                        nxt = s          # too early to serve: pass through only
                    edges.append(f"(edge {_rname(a, s, t)} {_rname(n, nxt, t + 1)})")
    goals = [f"(goal-node {_rname(route[-1], last, t)})"
             for t in range(last_block + 1, horizon + 1)
             if (route[-1], t) not in constraints]
    served = 1 if start == route[0] and releases.get(0, 0) <= 0 else 0
    return _problem(nodes, _rname(start, served, 0), edges + goals)


def make_fd_planner(fd_dir, search='astar(lmcut())'):
    """fd_dir: folder containing fast-downward.py."""
    fd = os.path.join(fd_dir, "fast-downward.py")
    cache = {}

    def plan(grid, start, goal, constraints, horizon):
        if (start, 0) in constraints:
            return None
        route = as_route(goal, constraints)
        key = (start, route, frozenset(constraints), horizon)
        if key in cache:
            return cache[key]
        plan.calls += 1
        problem = (make_problem(grid, start, route[0], constraints, horizon)
                   if len(route) == 1
                   else make_route_problem(grid, start, route, constraints, horizon))
        path = None
        if problem is not None:
            with tempfile.TemporaryDirectory() as d:
                dom, prob, out = (os.path.join(d, f) for f in ("d.pddl", "p.pddl", "plan"))
                open(dom, "w").write(DOMAIN)
                open(prob, "w").write(problem)
                res = subprocess.run(
                    [fd, "--plan-file", out, "--sas-file", os.path.join(d, "o.sas"),
                     dom, prob, "--search", search],
                    capture_output=True, text=True)
                if res.returncode == 0 and os.path.exists(out):
                    path = [start]
                    for line in open(out):
                        if line.startswith("(move"):
                            to = line.split()[2].rstrip(")")   # c<r>_<k>[_<served>]_<t>
                            r, k = to[1:].split("_")[:2]
                            path.append((int(r), int(k)))
                elif res.returncode not in (10, 11, 12):   # 10-12 = no plan exists
                    raise RuntimeError(f"Fast Downward failed ({res.returncode}):\n"
                                       + res.stdout[-800:] + res.stderr[-800:])
        cache[key] = path
        return path

    plan.calls = 0
    return plan


def main():
    """Check the encoding without invoking Fast Downward."""
    grid = Grid(["..", ".."])
    # A vertex ban leaves no copy of that cell at that time, so it cannot be entered.
    text = make_problem(grid, (0, 0), (0, 1), frozenset({((0, 1), 2)}), 3)
    assert "c0_1_2" not in text
    assert "(goal-node c0_1_3)" in text
    # An edge ban removes only that direction at that time.
    text = make_problem(grid, (0, 0), (0, 1), frozenset({((0, 0), (0, 1), 0)}), 3)
    assert "(edge c0_0_0 c0_1_1)" not in text
    assert "(edge c0_1_0 c0_0_1)" in text
    # An edge ban must not be read as a vertex ban on its first cell.
    assert "(goal-node c0_1_0)" in text
    # A route honours both kinds of ban too.
    text = make_route_problem(grid, (0, 0), ((0, 1), (1, 1)),
                              frozenset({((0, 0), (0, 1), 0)}), 4)
    assert "(edge c0_0_0_0 c0_1_1_1)" not in text
    assert "(goal-node c1_1_2_4)" in text
    print("all checks passed")


if __name__ == "__main__":
    main()
