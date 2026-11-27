"""Regenerate the hard set against best-first CBS instead of the depth-first tree.

instances.random_hard keeps an instance when the two depth-first rules differ in
backtracks. That is the property the midterm needed, but every benchmark we run
now is best-first CBS, so the filter selects for a search we no longer use and
carries its bias. Here an instance is kept only when best-first CBS actually
spends expansions on it, and the cost of conflict repair is reported directly.

Standard library only.

    python3 -m mas.instances.hard_cbs          # self-check, then the regenerated set
"""
import random

from mas.arbitration.cbs_baseline import Grid, find_conflict, plan_single, solve_best_first

HORIZON = 20
CAP = 400


def _paths(grid, agents):
    return {a: plan_single(grid, s, g, frozenset(), HORIZON)
            for a, (s, g) in agents.items()}


def lower_bound(paths):
    """Cost when every agent takes its own shortest path and nothing waits."""
    return sum(len(p) - 1 for p in paths.values())


def effort(grid, agents, cap=CAP, conflicts="vertex"):
    """Best-first CBS stats, or None when the instance is not usable.

    None means one of: some agent cannot reach its goal even alone, the initial
    plans do not conflict, or the search does not finish within `cap` nodes.
    """
    paths = _paths(grid, agents)
    if any(p is None for p in paths.values()):
        return None
    if find_conflict(paths, mode=conflicts) is None:
        return None
    sol, st = solve_best_first(grid, agents, horizon=HORIZON,
                               max_nodes=cap, conflicts=conflicts)
    if sol is None:
        return None
    return {"nodes": st["nodes"], "replans": st["replans"],
            "backtracks": st["backtracks"],
            "cost": sum(len(p) - 1 for p in sol.values()),
            "lb": lower_bound(paths)}


def hard_by_search(n, seed, size=5, walls=0.25, n_agents=3, min_nodes=20, cap=CAP):
    """Random instances that best-first CBS finds hard, by node count.

    The selection criterion is our own search's effort, so the set is biased
    toward instances CBS struggles with rather than toward rule-order
    asymmetry. Instances past the cap are rejected rather than kept: a set where
    every method fails compares nothing. Everything else (size, wall density,
    agent count) matches instances.random_hard so the two sets are comparable.
    """
    rng, out, tries = random.Random(seed), [], 0
    while len(out) < n and tries < 20000:
        tries += 1
        grid = Grid(["".join("#" if rng.random() < walls else "." for _ in range(size))
                     for _ in range(size)])
        free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
        if len(free) < 2 * n_agents:
            continue
        pick = rng.sample(free, 2 * n_agents)
        agents = {f"a{i + 1}": (pick[2 * i], pick[2 * i + 1]) for i in range(n_agents)}
        st = effort(grid, agents, cap=cap)
        if st is None or st["nodes"] < min_nodes:
            continue
        out.append((f"hcb{len(out) + 1}", grid, agents))
    return out


def report(name, grid, agents):
    st = effort(grid, agents)
    if st is None:
        return f"{name:<8} unusable"
    return (f"{name:<8} nodes {st['nodes']:>5}  replans {st['replans']:>5}  "
            f"cost {st['cost']:>4}  lower bound {st['lb']:>4}  "
            f"waiting {st['cost'] - st['lb']:>3}")


def main():
    first = hard_by_search(20, seed=41, min_nodes=20)
    second = hard_by_search(20, seed=41, min_nodes=20)
    assert [n for n, _, _ in first] == [n for n, _, _ in second]
    for (_, g1, a1), (_, g2, a2) in zip(first, second):
        assert g1.rows == g2.rows and a1 == a2, "the generator is not deterministic"

    assert len(first) == 20, f"only {len(first)} instances cleared min_nodes"
    for name, grid, agents in first:
        st = effort(grid, agents)
        assert st is not None, f"{name} should be solvable under the cap"
        assert st["nodes"] >= 20, f"{name} did not clear the node threshold"
        assert st["cost"] >= st["lb"], "cost below the independent-path bound"
    print("all checks passed\n")

    print("regenerated hard set (seed 41, min_nodes 20)")
    for name, grid, agents in first:
        print("  " + report(name, grid, agents))

    from mas.instances.instances import gadgets, random_hard
    print("\nwhat the old filter produced, measured the same way")
    for name, grid, agents in gadgets() + random_hard(20, 11):
        print("  " + report(name, grid, agents))

    new_nodes = [effort(g, a)["nodes"] for _, g, a in first]
    old = [effort(g, a) for _, g, a in random_hard(20, 11)]
    old_nodes = [s["nodes"] if s else 0 for s in old]
    print(f"\nmedian nodes: new {sorted(new_nodes)[len(new_nodes) // 2]}"
          f"  old {sorted(old_nodes)[len(old_nodes) // 2]}")
    print(f"old set reachable through the node cap: {sum(1 for s in old if s)}"
          f"/{len(old)}")


if __name__ == "__main__":
    main()
