"""
Low-level planner backed by Fast Downward. Drop-in replacement for
cbs_baseline.plan_single.

Trick for "forbid (cell, t)": PDDL has no clock, so every cell is copied once
per time step (object c<row>_<col>_<t>). A move goes from (cell, t) to
(neighbor, t+1). A forbidden (cell, t) simply has no copy, so no move can
enter it. Waiting is a move from (cell, t) to (cell, t+1).
"""
import os
import subprocess
import tempfile

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


def make_problem(grid, start, goal, constraints, horizon):
    last_block = max((t for (c, t) in constraints if c == goal), default=-1)
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
                    if (n, t + 1) not in constraints:
                        edges.append(f"(edge {_name(a, t)} {_name(n, t + 1)})")
    goals = [f"(goal-node {_name(goal, t)})"
             for t in range(last_block + 1, horizon + 1)
             if (goal, t) not in constraints]
    return (f"(define (problem p) (:domain timed-grid)\n"
            f"  (:objects {' '.join(nodes)} - node)\n"
            f"  (:init (at {_name(start, 0)})\n    " + "\n    ".join(edges + goals) +
            f")\n  (:goal (done)))\n")


def make_fd_planner(fd_dir, search='astar(lmcut())'):
    """fd_dir: folder containing fast-downward.py."""
    fd = os.path.join(fd_dir, "fast-downward.py")
    cache = {}

    def plan(grid, start, goal, constraints, horizon):
        if (start, 0) in constraints:
            return None
        key = (start, goal, frozenset(constraints), horizon)
        if key in cache:
            return cache[key]
        plan.calls += 1
        problem = make_problem(grid, start, goal, constraints, horizon)
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
                            to = line.split()[2].rstrip(")")        # c<r>_<k>_<t>
                            r, k, _ = to[1:].split("_")
                            path.append((int(r), int(k)))
                elif res.returncode not in (10, 11, 12):   # 10-12 = no plan exists
                    raise RuntimeError(f"Fast Downward failed ({res.returncode}):\n"
                                       + res.stdout[-800:] + res.stderr[-800:])
        cache[key] = path
        return path

    plan.calls = 0
    return plan