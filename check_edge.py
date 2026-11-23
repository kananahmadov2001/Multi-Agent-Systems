"""Does the stronger swap rule change what best-first CBS returns?

A swap is resolved two ways. Default ("vertex"): each agent is forbidden the
cell it would enter, which also blocks that cell by any other route. Standard
CBS ("standard"): each agent is forbidden that edge at that time. The default is
stronger, so it can prune optimal solutions and is the reason the paper needs an
optimality caveat. This measures how often the difference is visible.

    python3 check_edge.py
"""
from cbs_baseline import find_conflict, plan_single, solve_best_first
from hard_cbs import hard_by_search
from instances import gadgets, random_hard, random_plain
from verify import verify

HORIZON = 20
CAP = 400


def run(grid, agents, conflicts):
    sol, st = solve_best_first(grid, agents, horizon=HORIZON, max_nodes=CAP,
                               conflicts=conflicts)
    if sol is None:
        return None, st
    ok, problems = verify(grid, agents, sol)
    assert ok, f"{conflicts} solution rejected by verify: {problems[0]}"
    return sum(len(p) - 1 for p in sol.values()), st


def compare(name, grid, agents, rows):
    vertex, sv = run(grid, agents, "vertex")
    standard, ss = run(grid, agents, "standard")
    if vertex == standard and (vertex is None or sv["nodes"] == ss["nodes"]):
        return
    rows.append((name, "vertex", vertex, sv["nodes"], sv["hit_limit"],
                 "standard", standard, ss["nodes"], ss["hit_limit"]))
    # The stronger rule must never be cheaper: it explores a subset of the
    # standard tree's solutions, so a smaller cost here would be a bug.
    if vertex is not None and standard is not None:
        assert vertex >= standard, f"{name}: vertex mode found a cheaper plan"


def main():
    rows = []
    sets = [("gadget", gadgets()),
            ("old hard", random_hard(20, 11)),
            ("new hard", hard_by_search(20, seed=41, min_nodes=20)),
            ("random", random_plain(20, 23))]
    for label, inst in sets:
        for name, grid, agents in inst:
            compare(f"{label}/{name}", grid, agents, rows)

    if not rows:
        print("no difference on any instance")
        return
    print(f"{'instance':<22}{'vertex':>8}{'v-nodes':>9}{'std':>8}{'s-nodes':>9}  status")
    for name, _, v, vn, vl, _, s, sn, sl in rows:
        vtxt = "fail" if v is None else str(v)
        stxt = "fail" if s is None else str(s)
        if v is None and s is not None:
            status = "default misses a solution"
        elif v is not None and s is not None and v > s:
            status = f"default is {v - s} more expensive"
        elif v is None and s is None:
            status = "both fail"
        elif sl and not vl:
            status = "default cheaper but standard hits cap"
        else:
            status = "same cost, different tree"
        print(f"{name:<22}{vtxt:>8}{vn:>9}{stxt:>8}{sn:>9}  {status}")

    print(f"\n{len(rows)} instance(s) differ; the rest are identical")


if __name__ == "__main__":
    main()
