"""Does Fast Downward's search configuration change what our encoding returns?

Two separate questions, and the midterm conflated them:

  1. Does a satisficing configuration return LONGER plans? Our time-expanded
     encoding makes this unlikely by construction: every plan that reaches a
     goal copy at time t costs exactly t moves, because a move is the only way
     to advance time. Length is fixed by the layer reached, not by the route
     through it.
  2. Does it return a DIFFERENT plan of the same length? This is the one that
     matters for us. Our benchmark result is that Approach 2 is fragile to
     which equally optimal path the planner returns, while a constraint tree
     absorbs that variation. If a configuration change perturbs the path, it
     should perturb CBS node counts downstream; if it never does, identical
     node counts across configurations are expected rather than suspicious.

This asks the planner directly, with no CBS in the way, so neither answer can
hide behind the search above it.

    python3 -m checks.check_search --fd /path/to/downward
    python3 -m checks.check_search --fd /path/to/downward --alt "lazy_greedy([cea()])"
"""
import argparse
import random
import time

from mas.arbitration.cbs_baseline import Grid, plan_single
from mas.planning.fd_planner import make_fd_planner


def case(rng, size, horizon):
    rows = ["".join("#" if rng.random() < 0.2 else "." for _ in range(size))
            for _ in range(size)]
    grid = Grid(rows)
    free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
    if len(free) < 3:
        return None
    start, goal = rng.choice(free), rng.choice(free)
    constraints = frozenset((rng.choice(free), rng.randint(0, horizon))
                            for _ in range(rng.randint(0, 6)))
    return grid, start, goal, constraints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", required=True, help="directory holding fast-downward.py")
    ap.add_argument("--alt", default="lazy_greedy([ff()])",
                    help="the satisficing configuration to compare against")
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cases = []
    while len(cases) < a.cases:
        made = case(rng, a.size, a.horizon)
        if made:
            cases.append(made)

    print(f"{len(cases)} cases, grids {a.size}x{a.size}, horizon {a.horizon}")
    print(f"  optimal:     astar(lmcut())")
    print(f"  satisficing: {a.alt}\n")

    same_len = longer = shorter = diff_path = both_none = disagree = 0
    t_opt = t_alt = 0.0
    examples = []
    for grid, start, goal, constraints in cases:
        # A fresh planner per case per configuration: make_fd_planner caches on
        # (start, goal, constraints, horizon) without the grid, and these cases
        # have different walls.
        got = {}
        for name, cfg in (("opt", "astar(lmcut())"), ("alt", a.alt)):
            plan = make_fd_planner(a.fd, search=cfg)
            t0 = time.time()
            got[name] = plan(grid, start, goal, constraints, a.horizon)
            dt = time.time() - t0
            if name == "opt":
                t_opt += dt
            else:
                t_alt += dt

        o, s = got["opt"], got["alt"]
        if o is None and s is None:
            both_none += 1
            continue
        if (o is None) != (s is None):
            disagree += 1
            continue
        lo, ls = len(o) - 1, len(s) - 1
        if ls > lo:
            longer += 1
            if len(examples) < 4:
                examples.append(("longer", lo, ls))
        elif ls < lo:
            shorter += 1              # would mean astar was not optimal: a bug
        else:
            same_len += 1
            if o != s:
                diff_path += 1
                if len(examples) < 4:
                    examples.append(("same length, different path", lo, ls))

    solved = same_len + longer + shorter
    print(f"  both found no plan:        {both_none}")
    print(f"  disagree on solvability:   {disagree}")
    print(f"  same length:               {same_len}")
    print(f"     of which different path:{diff_path}")
    print(f"  satisficing longer:        {longer}")
    print(f"  satisficing shorter:       {shorter}  (nonzero would mean astar is not optimal)")
    print(f"\n  time   astar(lmcut()) {t_opt:6.1f}s   {a.alt} {t_alt:6.1f}s"
          f"   ratio {t_alt / t_opt:.2f}" if t_opt else "")
    for kind, lo, ls in examples:
        print(f"    {kind}: optimal {lo}, satisficing {ls}")

    print()
    if solved == 0:
        print("INCONCLUSIVE: nothing was solved. Check the --fd path.")
    elif longer == 0 and diff_path == 0:
        print("The configuration changed NOTHING: same length and the same plan,")
        print("every time. Identical CBS node counts across configurations are then")
        print("expected, not evidence that the flag failed to take effect. The time")
        print("ratio above is the independent check that it did.")
    elif longer == 0:
        print("Same length every time, but a different plan on some cases. Our")
        print("encoding fixes cost by the time layer reached, so optimality survives")
        print("a satisficing search here; what changes is which equally optimal plan")
        print("comes back, which is the variation Approach 2 is fragile to.")
    else:
        print("The satisficing configuration returned longer plans, so Approach 1's")
        print("optimality really is conditional on the low-level planner. Report the")
        print("counts above.")


if __name__ == "__main__":
    main()
