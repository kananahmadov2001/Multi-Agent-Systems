"""
Approach 1: deterministic conflict-resolution search (Conflict-Based Search).

Two ways to walk the same constraint tree:
  solve(...)             depth-first with backtracking (the proposal's strategy table). Returns the
                         first conflict-free plan it reaches, so plans are not necessarily shortest.
  solve_best_first(...)  best-first: always expand the open node with the lowest total path cost,
                         as in Sharon et al. 2015 (the proposal's Related Work). Optimal plans.

High level (this is the part we are testing):
  1. Every agent plans alone.
  2. Central agent finds the first conflict between plans.
  3. A FIXED RULE picks which agent to constrain first.
  4. That agent replans under the new constraint.
  5. If it cannot, flip to the other agent (backtrack).
  6. If neither works, give up on this node (and backtrack further up).

Low level: a small space-time BFS on a grid. Later this gets swapped for
Fast Downward; the high level does not change.
"""
from collections import deque
import itertools


class Grid:
    def __init__(self, rows):
        self.rows = rows
        self.h, self.w = len(rows), len(rows[0])

    def free(self, c):
        r, k = c
        return 0 <= r < self.h and 0 <= k < self.w and self.rows[r][k] != "#"

    def neighbors(self, c):
        """Waiting in place counts as a move."""
        r, k = c
        for dr, dk in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, k + dk)
            if self.free(n):
                yield n


def _is_waypoints(goal):
    """A goal is one cell (r, k) or an ordered list of cells [c1, c2, ...]."""
    if isinstance(goal, list):
        return True
    return isinstance(goal, tuple) and bool(goal) and isinstance(goal[0], (list, tuple))


def _last_vertex_block(constraints, cell):
    """Latest time `cell` is forbidden outright.

    Constraints may be vertex bans, edge bans, or internal waypoint releases;
    only a vertex ban stops an agent from standing somewhere.
    """
    return max((e[1] for e in constraints if len(e) == 2 and e[0] == cell),
               default=-1)


def describe(x):
    """A constraint in words, whatever shape it is.

    Four shapes share one tree now -- a vertex ban, an edge ban, a deadline and
    a release -- so a caller that used to unpack a (cell, time) pair asks here
    instead. The LLM prompts are the reason this exists: under --conflicts
    standard a conflict is a 3-tuple, and a formatter that assumes a pair
    silently writes nonsense into the prompt.
    """
    if isinstance(x, (set, frozenset)):
        return f"stay off {sorted({e[0] for e in x})[0]} until after t={max(e[1] for e in x)}"
    if len(x) == 3 and x[0] == "by":
        return f"serve {x[1]} no later than t={x[2]}"
    if len(x) == 3 and x[0] == "release":
        return f"not serve waypoint {x[1]} before t={x[2]}"
    if len(x) == 3:
        return f"not move {x[0]} -> {x[1]} at t={x[2]}"
    return f"not be at {x[0]} at t={x[1]}"


def sort_key(x):
    """Order constraints of any shape by time, then by how they read."""
    return (x[2] if len(x) == 3 and not isinstance(x, (set, frozenset)) else x[1], describe(x))


def plan_single(grid, start, goal, constraints, horizon):
    """Shortest-time path honoring vertex and edge bans, or None.

    `goal` is one cell, or a list of cells to visit in order; the second form is
    what lets one agent serve two parts of a task (see plan_route).
    """
    if _is_waypoints(goal):
        return plan_route(grid, start, goal, constraints, horizon)
    if (start, 0) in constraints:
        return None
    # The agent must be able to sit on its goal, so it may only finish after
    # the last time the goal cell is forbidden.
    last_block = _last_vertex_block(constraints, goal)
    q = deque([(start, 0)])
    parent = {(start, 0): None}
    while q:
        cell, t = q.popleft()
        if cell == goal and t > last_block:
            path, node = [], (cell, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1]
        if t >= horizon:
            continue
        for n in grid.neighbors(cell):
            if ((n, t + 1) in constraints or (cell, n, t) in constraints
                    or (n, t + 1) in parent):
                continue
            parent[(n, t + 1)] = (cell, t)
            q.append((n, t + 1))
    return None


def plan_route(grid, start, goals, constraints, horizon):
    """Shortest-time path that serves `goals` in order, ending on the last one.

    The search state is (cell, served, time): `served` is how many waypoints have
    been reached, and reaching one is instantaneous, which matches the conflict
    model where a cell is occupied at a time step. After the last waypoint the
    agent parks there, so it may only finish after that cell is last forbidden,
    the same rule plan_single uses for a single goal. An internal constraint
    ("release", waypoint_index, time) delays service but not earlier transit.
    """
    raw_goals = tuple(goals)
    releases = {e[1]: e[2] for e in constraints
                if len(e) == 3 and e[0] == "release"}
    if releases and any(a == b for a, b in zip(raw_goals, raw_goals[1:])):
        return None  # consecutive services on one cell have no distinct arrival
    goals = (raw_goals if releases else
             tuple(g for i, g in enumerate(raw_goals) if i == 0 or g != raw_goals[i - 1]))
    if not goals:
        raise ValueError("plan_route needs at least one goal")
    if (start, 0) in constraints:
        return None
    last_block = _last_vertex_block(constraints, goals[-1])

    served = 1 if start == goals[0] and releases.get(0, 0) <= 0 else 0
    q = deque([(start, served, 0)])
    parent = {(start, served, 0): None}
    while q:
        cell, done, t = q.popleft()
        if done == len(goals) and cell == goals[-1] and t > last_block:
            path, node = [], (cell, done, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1]
        if t >= horizon:
            continue
        for n in grid.neighbors(cell):
            if (n, t + 1) in constraints or (cell, n, t) in constraints:
                continue
            nxt = (done + 1 if done < len(goals) and n == goals[done]
                   and t + 1 >= releases.get(done, 0) else done)
            key = (n, nxt, t + 1)
            if key in parent:
                continue
            parent[key] = (cell, done, t)
            q.append(key)
    return None





def pos_at(path, t):
    return path[min(t, len(path) - 1)]  # agents wait at their goal


def find_conflict(paths, mode="vertex"):
    """Earliest conflict. Returns {agent: constraint} or None.

    mode="vertex"   a swap is resolved by forbidding each agent the cell it
                    would enter, (cell, time). This is stronger than standard
                    CBS: it also stops the agent reaching that cell by another
                    route, so the tree is incomplete and an optimal solution can
                    be pruned.
    mode="standard" a swap adds an edge ban (from, to, time) instead, which is
                    the scheme of Sharon et al. 2015 and keeps the tree complete.
    """
    agents = sorted(paths)
    horizon = max(len(p) for p in paths.values())
    for t in range(horizon):
        for a, b in itertools.combinations(agents, 2):
            if pos_at(paths[a], t) == pos_at(paths[b], t):  # same cell
                cell = pos_at(paths[a], t)
                return {a: (cell, t), b: (cell, t)}
            if (pos_at(paths[a], t) == pos_at(paths[b], t + 1)
                    and pos_at(paths[a], t + 1) == pos_at(paths[b], t)):  # swap
                if mode == "standard":
                    return {a: (pos_at(paths[a], t), pos_at(paths[a], t + 1), t),
                            b: (pos_at(paths[b], t), pos_at(paths[b], t + 1), t)}
                return {a: (pos_at(paths[a], t + 1), t + 1),
                        b: (pos_at(paths[b], t + 1), t + 1)}
    return None


def solve(grid, agents, rule=sorted, horizon=30, max_nodes=2000,
          planner=plan_single, conflicts="vertex"):
    """
    agents: {name: (start, goal)}
    planner: low-level single-agent planner (plan_single, or the Fast Downward one)
    rule:   function taking [agent, agent] -> same agents in the order to try
            constraining. THIS is the 'fixed rule' (alphabetical by default).
    conflicts: "vertex" (our stronger swap rule) or "standard" (edge constraints).
    """
    stats = {"nodes": 0, "replans": 0, "backtracks": 0, "hit_limit": False}

    paths = {}
    for a, (s, g) in agents.items():
        stats["replans"] += 1
        p = planner(grid, s, g, frozenset(), horizon)
        if p is None:
            return None, stats
        paths[a] = p

    def dfs(constraints, paths):
        stats["nodes"] += 1
        if stats["nodes"] > max_nodes:
            stats["hit_limit"] = True
            return None
        conflict = find_conflict(paths, mode=conflicts)
        if conflict is None:
            return paths
        for agent in rule(list(conflict)):
            new_cons = dict(constraints)
            new_cons[agent] = constraints[agent] | {conflict[agent]}
            s, g = agents[agent]
            stats["replans"] += 1
            new_path = planner(grid, s, g, new_cons[agent], horizon)
            if new_path is None:           # this agent cannot avoid it: flip
                stats["backtracks"] += 1
                continue
            new_paths = dict(paths)
            new_paths[agent] = new_path
            result = dfs(new_cons, new_paths)
            if result is not None:
                return result
            if stats["hit_limit"]:
                return None
            stats["backtracks"] += 1       # whole subtree failed: try other agent
        return None

    root_cons = {a: frozenset() for a in agents}
    return dfs(root_cons, paths), stats


def solve_best_first(grid, agents, rule=sorted, horizon=30, max_nodes=2000,
                     planner=plan_single, conflicts="vertex"):
    """
    Conflict-Based Search as in Sharon et al. 2015: a priority queue of tree nodes ordered by the
    total cost (sum of path lengths). Each expansion picks the first conflict and creates BOTH
    children (one per agent, each forbidden from the conflict cell at that time) and replans that
    agent. A child whose agent cannot replan is dropped. The first node popped with no conflict is
    a solution with minimal total cost (for the given horizon).
    `rule` only breaks ties between equal-cost nodes (the order the two children are created in).
    stats: nodes = nodes expanded, backtracks = children dropped because the agent could not replan.
    """
    import heapq
    stats = {"nodes": 0, "replans": 0, "backtracks": 0, "hit_limit": False}

    paths = {}
    for a, (s, g) in agents.items():
        stats["replans"] += 1
        p = planner(grid, s, g, frozenset(), horizon)
        if p is None:
            return None, stats
        paths[a] = p

    cost = lambda ps: sum(len(p) - 1 for p in ps.values())
    tick = itertools.count()
    root_cons = {a: frozenset() for a in agents}
    heap = [(cost(paths), next(tick), root_cons, paths)]
    while heap:
        _, _, constraints, paths = heapq.heappop(heap)
        stats["nodes"] += 1
        if stats["nodes"] > max_nodes:
            stats["hit_limit"] = True
            return None, stats
        conflict = find_conflict(paths, mode=conflicts)
        if conflict is None:
            return paths, stats
        for agent in rule(list(conflict)):
            new_cons = dict(constraints)
            new_cons[agent] = constraints[agent] | {conflict[agent]}
            s, g = agents[agent]
            stats["replans"] += 1
            new_path = planner(grid, s, g, new_cons[agent], horizon)
            if new_path is None:
                stats["backtracks"] += 1
                continue
            new_paths = dict(paths)
            new_paths[agent] = new_path
            heapq.heappush(heap, (cost(new_paths), next(tick), new_cons, new_paths))
    return None, stats


def show(title, result, stats):
    print(f"\n=== {title} ===")
    if result is None:
        why = "search limit hit" if stats["hit_limit"] else "no solution"
        print(f"FAILED ({why})")
    else:
        for a in sorted(result):
            print(f"  {a}: {result[a]}")
    print(f"  stats: {stats}")


if __name__ == "__main__":
    # Test 1: two agents cross at the centre of an open 3x3 grid.
    grid = Grid(["...", "...", "..."])
    agents = {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}
    show("crossing paths (solvable)", *solve(grid, agents))

    # Test 2: same, but the rule constrains a2 first instead of a1.
    show("crossing paths, reversed rule",
         *solve(grid, agents, rule=lambda ags: sorted(ags, reverse=True)))

    # Test 3: head-on in a 1-wide corridor: provably unsolvable.
    grid = Grid(["..."])
    agents = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}
    show("head-on in corridor (unsolvable)", *solve(grid, agents, horizon=8))

    # Test 4: corridor with a side pocket: one agent must duck in and let the other pass.
    grid = Grid(["...", "#.#"])
    agents = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}
    show("corridor with a pocket (solvable)", *solve(grid, agents, horizon=10))

    # Same cases with best-first CBS (lowest total cost first).
    show("pocket, best-first CBS", *solve_best_first(grid, agents, horizon=10))
    grid = Grid(["...", "...", "..."])
    agents = {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}
    show("crossing paths, best-first CBS", *solve_best_first(grid, agents))