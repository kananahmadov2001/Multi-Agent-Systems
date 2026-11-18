"""
Independent check of a finished solution. Written separately from the search code on purpose:
it does not call find_conflict, so a bug there cannot hide a bad plan.

It replays all agents' paths together, one time step at a time (an agent that has finished
stays on its goal cell), and confirms:
  1. every agent starts at its start cell and ends on its goal cell
  2. every step is legal: a wait, or one move up/down/left/right onto a free cell
  3. no two agents are on the same cell at the same time
  4. no two agents swap cells in one step
"""


def verify(grid, agents, paths):
    """Returns (ok, problems). `paths` is {agent: [cell at t=0, t=1, ...]}."""
    problems = []
    if paths is None:
        return False, ["no paths"]
    for a, (start, goal) in agents.items():
        p = paths.get(a)
        if not p:
            problems.append(f"{a}: no path")
            continue
        if p[0] != start:
            problems.append(f"{a}: starts at {p[0]}, not {start}")
        if p[-1] != goal:
            problems.append(f"{a}: ends at {p[-1]}, not its goal {goal}")
        for t in range(1, len(p)):
            (r0, c0), (r1, c1) = p[t - 1], p[t]
            if not grid.free(p[t]):
                problems.append(f"{a}: enters a wall or leaves the grid at t={t}: {p[t]}")
            if abs(r0 - r1) + abs(c0 - c1) > 1:
                problems.append(f"{a}: illegal jump {p[t - 1]} -> {p[t]} at t={t}")

    if problems:
        return False, problems

    def at(a, t):
        p = paths[a]
        return p[min(t, len(p) - 1)]

    names = sorted(agents)
    horizon = max(len(paths[a]) for a in names)
    for t in range(horizon + 1):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if at(a, t) == at(b, t):
                    problems.append(f"{a} and {b} both on {at(a, t)} at t={t}")
                if at(a, t) == at(b, t + 1) and at(a, t + 1) == at(b, t) and at(a, t) != at(a, t + 1):
                    problems.append(f"{a} and {b} swap {at(a, t)} <-> {at(b, t)} between t={t} and t={t + 1}")
    return (not problems), problems


if __name__ == "__main__":
    from cbs_baseline import Grid, solve

    grid = Grid(["...", "...", "..."])
    ag = {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}

    good, _ = solve(grid, ag)
    print("real solution passes:", verify(grid, ag, good))

    tests = {
        "both cross the centre at t=1": {"a1": [(0, 1), (1, 1), (2, 1)], "a2": [(1, 0), (1, 1), (1, 2)]},
        "wrong goal": {"a1": [(0, 1), (1, 1)], "a2": [(1, 0), (1, 0), (1, 1), (1, 2)]},
        "illegal jump": {"a1": [(0, 1), (2, 1)], "a2": [(1, 0), (1, 0), (1, 1), (1, 2)]},
        "wrong start": {"a1": [(0, 0), (1, 1), (2, 1)], "a2": [(1, 0), (1, 0), (1, 1), (1, 2)]},
    }
    for name, paths in tests.items():
        ok, why = verify(grid, ag, paths)
        print(f"bad plan '{name}': ok={ok}  {why[:1]}")

    g2 = Grid(["..."])
    ag2 = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}
    ok, why = verify(g2, ag2, {"a1": [(0, 0), (0, 1), (0, 2)], "a2": [(0, 2), (0, 1), (0, 0)]})
    print(f"head-on swap: ok={ok}  {why[:1]}")
    g3 = Grid(["...", "#.#"])
    ok, why = verify(g3, ag2, {"a1": [(0, 0), (0, 1), (0, 2)], "a2": [(0, 2), (0, 1), (0, 0)]})
    print(f"pocket, both walk straight through: ok={ok}  {why[:1]}")
    g4 = Grid(["..."])
    ag4 = {"a1": ((0, 0), (0, 1)), "a2": ((0, 1), (0, 0))}
    ok, why = verify(g4, ag4, {"a1": [(0, 0), (0, 1)], "a2": [(0, 1), (0, 0)]})
    print(f"pure swap, no shared cell: ok={ok}  {why[:1]}")