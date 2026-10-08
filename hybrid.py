"""
Approach 3: hybrid. The tree from Approach 1 (real backtracking, depth-first),
but the LLM replaces the fixed rule: at each node it picks which agent to
constrain FIRST. The other agent is still tried if that whole branch fails.
So the tree guarantees the search is complete; the LLM only decides the order.
"""
from cbs_baseline import find_conflict, plan_single
from llm_arbitrator import make_prompt, parse_choice

HYBRID_NOTE = ("This is a search. If your choice leads to a dead end, the other agent will be "
               "tried next. Pick the agent you think is more likely to lead to a full solution.")


def solve_hybrid(grid, agents, llm, horizon=30, max_nodes=2000, max_calls=200,
                 planner=plan_single):
    """Returns (paths or None, stats). Same shape as cbs_baseline.solve."""
    stats = {"nodes": 0, "replans": 0, "backtracks": 0, "llm_calls": 0,
             "invalid_answers": 0, "hit_limit": False}

    paths = {}
    for a, (s, g) in agents.items():
        stats["replans"] += 1
        paths[a] = planner(grid, s, g, frozenset(), horizon)
        if paths[a] is None:
            return None, stats

    def order(conflict, paths):
        options = sorted(conflict)
        stats["llm_calls"] += 1
        text = llm(make_prompt(grid, agents, paths, conflict, [], extra=HYBRID_NOTE))
        first = parse_choice(text, options)
        if first is None:                      # bad answer: fall back to alphabetical
            stats["invalid_answers"] += 1
            first = options[0]
        return [first] + [a for a in options if a != first]

    def dfs(constraints, paths):
        stats["nodes"] += 1
        if stats["nodes"] > max_nodes or stats["llm_calls"] >= max_calls:
            stats["hit_limit"] = True
            return None
        conflict = find_conflict(paths)
        if conflict is None:
            return paths
        for agent in order(conflict, paths):
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
            result = dfs(new_cons, new_paths)
            if result is not None:
                return result
            if stats["hit_limit"]:
                return None
            stats["backtracks"] += 1
        return None

    return dfs({a: frozenset() for a in agents}, paths), stats