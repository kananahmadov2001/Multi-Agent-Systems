"""Cross-agent task-order checking and bounded heuristic repair.

Service times follow waypoint occurrences, not just cell visits. Repair delays the
later part and is neither complete nor optimality-preserving. Standard library only.

    python3 ordering.py       # self-check, no API calls
"""
from cbs_baseline import plan_single, solve_best_first
from verify import verify, verify_order


def _waypoints(goal):
    if isinstance(goal, (list, tuple)) and goal and isinstance(goal[0], (list, tuple)):
        return list(goal)
    return [goal]


def service_times(paths, agents, parts):
    """Map each part to (agent, waypoint occurrence, service time), or report why not."""
    times, problems = {}, []
    for agent in sorted(agents):
        route = _waypoints(agents[agent][1])
        assigned = parts.get(agent, [])
        if not assigned:
            continue
        if len(route) != len(assigned):
            problems.append(f"ordering: {agent} has {len(route)} waypoints for {len(assigned)} parts")
            continue
        if any(a == b for a, b in zip(route, route[1:])):
            problems.append(f"ordering: {agent} has consecutive parts on one cell; service is unsupported")
            continue
        path = paths.get(agent)
        if not path:
            problems.append(f"ordering: {agent} has no path")
            continue
        served = 0
        for t, cell in enumerate(path):
            if served < len(route) and cell == route[served]:
                part = assigned[served]
                if part in times:
                    problems.append(f"ordering: part {part!r} is assigned more than once")
                else:
                    times[part] = (agent, served, t)
                served += 1
        if served < len(route):
            problems.append(f"ordering: {agent} serves only {served} of {len(route)} parts")
    return times, problems


def ordering_conflicts(task, paths, agents, parts):
    """Return (structured order violations, reasons the order cannot be judged)."""
    times, problems = service_times(paths, agents, parts)
    for part in task.order:
        if part not in times:
            problems.append(f"ordering: no service time for part {part!r}")
    for part in times:
        if part not in task.order:
            problems.append(f"ordering: unknown part {part!r}")
    if problems:
        return [], problems
    conflicts = []
    for before, after in zip(task.order, task.order[1:]):
        earlier, later = times[before], times[after]
        if earlier[2] >= later[2]:
            conflicts.append({"before": before, "after": after,
                              "before_agent": earlier[0], "before_time": earlier[2],
                              "after_agent": later[0], "after_index": later[1],
                              "after_time": later[2]})
    return conflicts, []


def describe_conflict(conflict):
    return (f"ordering: {conflict['before']} is served at t={conflict['before_time']}, "
            f"but {conflict['after']} is served at t={conflict['after_time']}; "
            "strict before < after is required")


def _goal_key(goal):
    return tuple(goal) if isinstance(goal, list) else goal


def _merge_stats(total, current):
    for key, value in current.items():
        if key == "hit_limit":
            total[key] = total.get(key, False) or value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def solve_ordered(grid, agents, task, parts, solver=solve_best_first,
                  planner=plan_single, checker=verify, horizon=20, max_nodes=2000,
                  max_repairs=8):
    """Solve, then delay later waypoints until strict task order holds.

    Each repair fixes releases observed in the previous solution, so exhaustion or
    a failed replan is not evidence that no ordered solution exists.
    """
    if max_repairs < 0:
        raise ValueError("max_repairs must be nonnegative")
    release = {agent: {} for agent in agents}
    # Releases are routed back to an agent by its start and route, which is all
    # the planner callback is given. Two agents with the same pair would be
    # indistinguishable there and would silently inherit each other's releases,
    # so refuse the instance instead of misattributing them.
    route_keys = {}
    for agent, (start, goal) in sorted(agents.items()):
        key = (start, _goal_key(goal))
        if key in route_keys:
            return None, {"ordering_conflicts": 0, "ordering_repairs": 0,
                          "ordering_rounds": 0, "hit_limit": False,
                          "ordering_error": (f"{agent} and {route_keys[key]} share a start "
                                             "and route; releases cannot be told apart")}
        route_keys[key] = agent

    def released_planner(g, start, goal, constraints, h):
        agent = route_keys[(start, _goal_key(goal))]
        extra = {("release", index, time)
                 for index, time in release[agent].items()}
        return planner(g, start, goal, constraints | extra, h)

    total = {"ordering_conflicts": 0, "ordering_repairs": 0, "ordering_rounds": 0,
             "hit_limit": False}
    while True:
        paths, current = solver(grid, agents, horizon=horizon, max_nodes=max_nodes,
                                planner=released_planner)
        _merge_stats(total, current)
        total["ordering_rounds"] += 1
        if paths is None:
            if total["ordering_repairs"]:
                total["ordering_error"] = ("ordering repair could not replan; "
                                           "this is not a proof of infeasibility")
            return None, total
        ok, problems = checker(grid, agents, paths)
        if not ok:
            total["ordering_error"] = f"spatial verification failed: {problems[0]}"
            return None, total
        conflicts, problems = ordering_conflicts(task, paths, agents, parts)
        if problems:
            total["ordering_error"] = problems[0]
            return None, total
        total["ordering_conflicts"] += len(conflicts)
        if not conflicts:
            # the detector says the order holds; now ask verify.py, which does
            # not share a line of code with it
            ok, problems = verify_order(agents, paths, task, parts)
            if not ok:
                total["ordering_error"] = f"independent order check disagrees: {problems[0]}"
                return None, total
            return paths, total
        if total["ordering_repairs"] + len(conflicts) > max_repairs:
            total["ordering_error"] = (f"ordering repair exhausted after "
                                       f"{total['ordering_repairs']} repair(s); "
                                       "this is not a proof of infeasibility")
            return None, total
        for conflict in conflicts:
            agent = conflict["after_agent"]
            index = conflict["after_index"]
            earliest = conflict["before_time"] + 1
            release[agent][index] = max(release[agent].get(index, 0), earliest)
            total["ordering_repairs"] += 1


def main():
    from agent_pool import Task
    from cbs_baseline import Grid

    grid = Grid(["...", "...", "..."])
    agents = {"a1": ((0, 0), [(2, 2)]), "a2": ((0, 2), [(0, 1)])}
    parts = {"a1": ["carry"], "a2": ["inspect"]}
    task = Task("carry then inspect", {"carry", "inspect"}, ("carry", "inspect"))
    paths, _ = solve_best_first(grid, agents, horizon=10, max_nodes=100)
    conflicts, problems = ordering_conflicts(task, paths, agents, parts)
    assert not problems and len(conflicts) == 1, (conflicts, problems)
    assert conflicts[0]["before_time"] == 4 and conflicts[0]["after_time"] == 1

    repaired, stats = solve_ordered(grid, agents, task, parts, horizon=10,
                                    max_nodes=100, max_repairs=4)
    assert repaired is not None, stats
    conflicts, problems = ordering_conflicts(task, repaired, agents, parts)
    assert not problems and not conflicts, (conflicts, problems)
    times, _ = service_times(repaired, agents, parts)
    assert times["carry"][2] == 4 and times["inspect"][2] == 5, times
    assert stats["ordering_repairs"] == 1 and stats["ordering_rounds"] == 2, stats
    again, again_stats = solve_ordered(grid, agents, task, parts, horizon=10,
                                       max_nodes=100, max_repairs=4)
    assert again == repaired and again_stats == stats, "repair is not deterministic"

    exhausted, why = solve_ordered(grid, agents, task, parts, horizon=10,
                                   max_nodes=100, max_repairs=0)
    assert exhausted is None and "exhausted" in why["ordering_error"], why

    # A release delays service at the start cell without forbidding the cell.
    parked = plan_single(Grid(["..."]), (0, 0), [(0, 0)],
                         frozenset({("release", 0, 2)}), 3)
    assert parked == [(0, 0), (0, 0), (0, 0)], parked

    # Passing a future waypoint early does not serve it; repeated cells use occurrences.
    repeated_agents = {"a1": ((0, 0), [(1, 1), (0, 0), (1, 1)])}
    repeated_parts = {"a1": ["p1", "p2", "p3"]}
    repeated_path = {"a1": [(0, 0), (1, 1), (0, 0), (1, 1)]}
    times, problems = service_times(repeated_path, repeated_agents, repeated_parts)
    assert not problems and [times[f"p{i}"][2] for i in (1, 2, 3)] == [1, 2, 3], times

    # Different agents may have different parts on the same cell; equality is a violation.
    shared_agents = {"a1": ((0, 0), [(1, 1)]), "a2": ((2, 1), [(1, 1)])}
    shared_parts = {"a1": ["carry"], "a2": ["inspect"]}
    same_time = {"a1": [(0, 0), (1, 1)], "a2": [(2, 1), (1, 1)]}
    conflicts, problems = ordering_conflicts(task, same_time, shared_agents, shared_parts)
    assert not problems and len(conflicts) == 1, (conflicts, problems)
    later = {"a1": [(0, 0), (1, 1)], "a2": [(2, 1), (2, 1), (1, 1)]}
    conflicts, problems = ordering_conflicts(task, later, shared_agents, shared_parts)
    assert not problems and not conflicts, (conflicts, problems)

    bad_agents = {"a1": ((0, 0), [(0, 0), (0, 0)])}
    _, problems = service_times({"a1": [(0, 0)]}, bad_agents, {"a1": ["p1", "p2"]})
    assert problems and "unsupported" in problems[0], problems

    print("all checks passed")


if __name__ == "__main__":
    main()