"""Free control for Approach 3 (no API calls).

Approach 3 stops as soon as it GENERATES a conflict-free plan; best-first CBS (Approach 1) stops when that
plan is the cheapest open node, which is needed to guarantee the shortest plan. So Approach 3 can use fewer
nodes without the LLM helping at all. This script runs Approach 3's exact search with a dumb "always take
the cheapest open node" policy, so its node counts can be compared with the LLM's.

    python3 control.py > control.txt
"""
import re
from cbs_baseline import solve_best_first
from hybrid import solve_hybrid
from instances import gadgets, random_hard, random_plain, sweep, HORIZON

cheapest = lambda prompt: '{"expand": "%s"}' % re.findall(r"OPTION (\w+):", prompt)[0]   # first option = cheapest
SETS = {"hard": gadgets() + random_hard(20, 11), "random": random_plain(20, 23), "sweep": sweep()}

print(f"{'set':<7}{'instance':<18}{'CBS-bf':>12}{'cheapest-policy, cap 50':>26}{'cap 150':>14}")
tot = {}
for sname, inst in SETS.items():
    for name, grid, agents in inst:
        p1, s1 = solve_best_first(grid, agents, horizon=HORIZON, max_nodes=6000)
        c1 = f"{s1['nodes']}" + ("" if p1 else " FAIL")
        row = [c1]
        for cap in (50, 150):
            p, s = solve_hybrid(grid, agents, cheapest, horizon=HORIZON, max_nodes=500, max_calls=cap)
            row.append(f"{s['nodes']}" + ("" if p else " FAIL"))
            tot[(sname, cap)] = tot.get((sname, cap), 0) + (p is not None)
        tot[(sname, "cbs")] = tot.get((sname, "cbs"), 0) + (p1 is not None)
        print(f"{sname:<7}{name:<18}{row[0]:>12}{row[1]:>26}{row[2]:>14}")
print("\nsolved:", {k: v for k, v in tot.items()})