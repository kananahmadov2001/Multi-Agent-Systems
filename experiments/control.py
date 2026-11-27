"""Free control for Approach 3 (no API calls).

Approach 3 stops as soon as it GENERATES a conflict-free plan; best-first CBS (Approach 1) stops when that
plan is the cheapest open node, optimal within its constraint scheme, horizon and node cap. So Approach 3 can use fewer
nodes without the LLM helping at all. This script runs Approach 3's exact search with a dumb "always take
the cheapest open node" policy, so its node counts can be compared with the LLM's. Every
returned plan is independently replayed with verify before it counts as solved.

    python3 -m experiments.control > control.txt
"""
import re
from mas.arbitration.cbs_baseline import solve_best_first
from mas.arbitration.hybrid import solve_hybrid
from mas.instances.instances import gadgets, random_hard, random_plain, sweep, HORIZON
from mas.verify import verify

cheapest = lambda prompt: '{"expand": "%s"}' % re.findall(r"OPTION (\w+):", prompt)[0]   # first option = cheapest
SETS = {"hard": gadgets() + random_hard(20, 11), "random": random_plain(20, 23), "sweep": sweep()}


def outcome(grid, agents, paths, stats):
    """A nonempty plan is not a solution until the independent replay accepts it."""
    if paths is None:
        return f"{stats['nodes']} FAIL", False
    ok, _ = verify(grid, agents, paths)
    return (f"{stats['nodes']}", True) if ok else (f"{stats['nodes']} INVALID", False)


def main():
    print(f"{'set':<7}{'instance':<18}{'CBS-bf':>12}{'cheapest-policy, cap 50':>26}{'cap 150':>14}")
    tot = {}
    for sname, inst in SETS.items():
        for name, grid, agents in inst:
            p1, s1 = solve_best_first(grid, agents, horizon=HORIZON, max_nodes=6000)
            c1, solved = outcome(grid, agents, p1, s1)
            tot[(sname, "cbs")] = tot.get((sname, "cbs"), 0) + solved
            row = [c1]
            for cap in (50, 150):
                p, s = solve_hybrid(grid, agents, cheapest, horizon=HORIZON,
                                    max_nodes=500, max_calls=cap)
                cell, solved = outcome(grid, agents, p, s)
                row.append(cell)
                tot[(sname, cap)] = tot.get((sname, cap), 0) + solved
            print(f"{sname:<7}{name:<18}{row[0]:>12}{row[1]:>26}{row[2]:>14}")
    print("\nsolved:", {k: v for k, v in tot.items()})


if __name__ == "__main__":
    main()
