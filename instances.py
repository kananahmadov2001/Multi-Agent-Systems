"""
Harder test instances: ones where picking the wrong agent first actually costs something.

An instance is kept only if it is SOLVABLE (the fixed-rule tree solves it with one rule or the
other, within the node cap) and the choice MATTERS (the two fixed rules differ a lot in backtracks).
"""
import random
from cbs_baseline import Grid, plan_single, solve, solve_best_first, find_conflict

HORIZON = 20
CAP = 400


def _paths(grid, agents):
    return {a: plan_single(grid, s, g, frozenset(), HORIZON) for a, (s, g) in agents.items()}


def measure(grid, agents):
    """Backtracks for the two fixed rules, or None if neither solves it within the cap."""
    out = []
    for rule in (sorted, lambda x: sorted(x, reverse=True)):
        paths, st = solve(grid, agents, rule=rule, horizon=HORIZON, max_nodes=CAP)
        out.append(st["backtracks"] if paths is not None else None)
    return out if any(v is not None for v in out) else None


def gadgets():
    """Hand-built cases. Pocket corridors of different sizes, plus a 3-agent bottleneck."""
    return [
        ("pocket-3", Grid(["...", "#.#"]), {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}),
        ("pocket-5", Grid([".....", "##.##"]), {"a1": ((0, 0), (0, 4)), "a2": ((0, 4), (0, 0))}),
        ("pocket-5-mid", Grid([".....", "#.#.#"]), {"a1": ((0, 0), (0, 4)), "a2": ((0, 4), (0, 0))}),
        ("three-in-corridor", Grid([".....", "#.#.#"]),
         {"a1": ((0, 0), (0, 4)), "a2": ((0, 4), (0, 0)), "a3": ((0, 2), (0, 3))}),
        ("plus-crossing-3", Grid(["...", "...", "..."]),
         {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2)), "a3": ((2, 1), (0, 1))}),
    ]


def random_hard(n, seed, size=5, walls=0.25, n_agents=3, min_bt=4):
    rng, out, tries = random.Random(seed), [], 0
    while len(out) < n and tries < 20000:
        tries += 1
        grid = Grid(["".join("#" if rng.random() < walls else "." for _ in range(size)) for _ in range(size)])
        free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
        if len(free) < 2 * n_agents:
            continue
        pick = rng.sample(free, 2 * n_agents)
        agents = {f"a{i + 1}": (pick[2 * i], pick[2 * i + 1]) for i in range(n_agents)}
        paths = _paths(grid, agents)
        if any(p is None for p in paths.values()) or find_conflict(paths) is None:
            continue
        m = measure(grid, agents)
        if m is None:
            continue
        hi = max(v for v in m if v is not None)
        lo = min(v for v in m if v is not None) if None not in m else hi
        if hi >= min_bt and (None in m or lo * 2 <= hi):   # choice matters
            out.append((f"hard{len(out) + 1}", grid, agents))
    return out


def random_plain(n, seed, size=5, walls=0.25, n_agents=3, prefix="r", cap=1500):
    """Unfiltered random cases: only requires that a collision exists and that best-first CBS can solve it
    (so no API calls are wasted on impossible cases). No test of whether the choice of agent matters."""
    rng, out, tries = random.Random(seed), [], 0
    while len(out) < n and tries < 20000:
        tries += 1
        grid = Grid(["".join("#" if rng.random() < walls else "." for _ in range(size)) for _ in range(size)])
        free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
        if len(free) < 2 * n_agents:
            continue
        pick = rng.sample(free, 2 * n_agents)
        agents = {f"a{i + 1}": (pick[2 * i], pick[2 * i + 1]) for i in range(n_agents)}
        paths = _paths(grid, agents)
        if any(p is None for p in paths.values()) or find_conflict(paths) is None:
            continue
        sol, _ = solve_best_first(grid, agents, horizon=HORIZON, max_nodes=cap)
        if sol is None:
            continue
        out.append((f"{prefix}{len(out) + 1}", grid, agents))
    return out


def sweep(per_group=5, seed=31):
    """Difficulty sweep: the same kind of random case with 2, 3 and 4 agents (names a2-1, a3-1, a4-1, ...)."""
    out = []
    for k in (2, 3, 4):
        out += random_plain(per_group, seed + k, n_agents=k, prefix=f"a{k}-")
    return out


if __name__ == "__main__":
    print("gadgets:")
    for name, g, ag in gadgets():
        print(f"  {name:<20} backtracks (alpha, reversed) = {measure(g, ag)}")
    print(f"\nunfiltered random (held out, seed 23): {len(random_plain(20, 23))} cases; sweep: {[n for n, _, _ in sweep()]}")
    hard = random_hard(20, 11)
    print(f"\nrandom hard: kept {len(hard)}")
    for name, g, ag in hard[:8]:
        print(f"  {name:<8} {len(ag)} agents, backtracks = {measure(g, ag)}")