"""
Independent check of a finished solution. Written separately from the search code on purpose:
it does not call find_conflict, so a bug there cannot hide a bad plan.

It replays all agents' paths together, one time step at a time (an agent that has finished
stays on its goal cell), and confirms:
  1. every agent starts at its start cell, reaches its subgoals in order, and ends on the last one
  2. every step is legal: a wait, or one move up/down/left/right onto a free cell
  3. no two agents are on the same cell at the same time
  4. no two agents swap cells in one step
"""


def _waypoints(goal):
    """One goal cell, or the ordered list of cells a compound subgoal needs."""
    if isinstance(goal, (list, tuple)) and goal and isinstance(goal[0], (list, tuple)):
        return list(goal)
    return [goal]


def verify_order(agents, paths, task, parts):
    """Independent check that parts are served in the task's order.

    Service times are replayed here rather than taken from ordering.py, for
    the same reason as the rest of this file: the function that drives the
    repair loop must not also be the one that approves its result. "Before"
    is strict, so serving two parts in one step is a violation.
    """
    when, problems = {}, []
    for a in sorted(agents):
        route, labels = _waypoints(agents[a][1]), list(parts.get(a, []))
        if not labels:
            continue
        if len(route) != len(labels):
            problems.append(f"order: {a} has {len(route)} waypoints for {len(labels)} parts")
            continue
        path, k = paths.get(a), 0
        if not path:
            problems.append(f"order: {a} has no path")
            continue
        for t, cell in enumerate(path):
            if k < len(route) and cell == route[k]:
                if labels[k] in when:
                    problems.append(f"order: part {labels[k]!r} is served more than once")
                when[labels[k]] = t
                k += 1
        if k < len(route):
            problems.append(f"order: {a} serves only {k} of {len(route)} parts")
    for x, y in zip(task.order, task.order[1:]):
        if x not in when or y not in when:
            problems.append(f"order: no service time for {x!r} or {y!r}")
        elif when[x] >= when[y]:
            problems.append(f"order: {x} at t={when[x]} is not before {y} at t={when[y]}")
    return (not problems), problems


def verify(grid, agents, paths):
    """Returns (ok, problems). `paths` is {agent: [cell at t=0, t=1, ...]}.

    A subgoal is one cell or an ordered list of cells: a compound subgoal is
    served when the path reaches each cell in that order, and the agent then
    parks on the last one.
    """
    problems = []
    if paths is None:
        return False, ["no paths"]
    for a, (start, goal) in agents.items():
        goals = _waypoints(goal)
        p = paths.get(a)
        if not p:
            problems.append(f"{a}: no path")
            continue
        if p[0] != start:
            problems.append(f"{a}: starts at {p[0]}, not {start}")
        served = 0
        for cell in p:
            if served < len(goals) and cell == goals[served]:
                served += 1
        if served < len(goals):
            problems.append(f"{a}: reaches only {served} of {len(goals)} subgoals {goals}")
        elif p[-1] != goals[-1]:
            problems.append(f"{a}: ends at {p[-1]}, not its goal {goals[-1]}")
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
    from mas.arbitration.cbs_baseline import Grid, solve

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
