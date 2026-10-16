"""
Approach 3: LLM-guided search on the same Conflict-Based-Search tree as Approach 1.

Approach 1 (solve_best_first) always expands the open tree node with the lowest total cost.
Here the LLM takes over that choice: it is shown the most promising open nodes and picks the one to
expand next.
  * choosing a child of the node just expanded      = "continue" down the branch
  * choosing an older open node                      = "backtrack"
  * choosing between the two children of a collision = "which agent to constrain"
The tree itself is unchanged (both children of every expansion are created, a child whose agent
cannot replan is dropped), so the LLM only steers; it cannot make an invalid plan. The search stops at
the first conflict-free plan it generates, so plans are not guaranteed to be the cheapest.

solve_hybrid_dfs is the earlier version (depth-first tree, LLM only orders the two children). It is
kept as an ablation.
"""
import itertools

from cbs_baseline import find_conflict, plan_single
from llm_arbitrator import make_prompt, parse_choice

HYBRID_NOTE = ("This is a search. If your choice leads to a dead end, the other agent will be "
               "tried next. Pick the agent you think is more likely to lead to a full solution.")


def _cons_text(cons):
    parts = [f"{a} not at {c} at t={t}" for a in sorted(cons) for c, t in sorted(cons[a], key=lambda x: (x[1], x[0]))]
    return "; ".join(parts) or "none"


def make_node_prompt(grid, agents, cands, history):
    lines = ["Several agents move on a grid, one step per time unit (a step can be a wait).",
             "Each agent planned alone and then plans were repaired by a search tree. Every tree node is a set of",
             "constraints (an agent may not be at a cell at a time) plus each agent's current plan under them.",
             "Each open node below still has one collision. Expanding a node creates two children: one forbids the",
             "first agent from the collision cell at that time, the other forbids the second agent. A child whose",
             "agent has no path is dropped. The search ends at the first node with no collision.",
             "",
             "Grid (row 0 is the top, '#' is a wall):", "\n".join("".join(r) for r in grid.rows), "",
             "Agents (start -> goal):"]
    for a in sorted(agents):
        lines.append(f"  {a}: {agents[a][0]} -> {agents[a][1]}")
    lines += ["", "Open nodes you may expand next (cheapest first). Total steps = moves and waits of all agents."]
    for n in cands:
        c = n["conflict"]
        lines.append(f"OPTION {n['id']}: depth {n['depth']}, total steps {n['cost']}")
        lines.append(f"    constraints: {_cons_text(n['cons'])}")
        lines.append("    next collision, children would be: " +
                     " | ".join(f"forbid {a} at {c[a][0]} t={c[a][1]}" for a in sorted(c)))
        for a in sorted(n["paths"]):
            lines.append(f"    plan {a}: {n['paths'][a]}")
    if history:
        lines += ["", "Nodes expanded so far (most recent last):"] + [f"  {h}" for h in history[-8:]]
    lines += ["",
              "Cheaper nodes are not always better: a node with more constraints may be closer to a collision-free plan.",
              "Expanding a child of the node you just expanded continues down that branch; expanding an older node backtracks.",
              'Reply with ONLY this JSON, no other text: {"expand": "<option id>", "reason": "<one sentence>"}']
    return "\n".join(lines)


def solve_hybrid(grid, agents, llm, horizon=30, max_nodes=500, max_calls=50,
                 planner=plan_single, top_k=4):
    """LLM-guided CBS. Returns (paths or None, stats), same shape as cbs_baseline.solve_best_first.
    An invalid reply is retried once; if it is still invalid the cheapest node is used and stats["fallbacks"]
    counts it, so a result with fallbacks > 0 is partly plain best-first and must be reported as such."""
    stats = {"nodes": 0, "replans": 0, "backtracks": 0, "llm_calls": 0,
             "invalid_answers": 0, "fallbacks": 0, "hit_limit": False}

    paths = {}
    for a, (s, g) in agents.items():
        stats["replans"] += 1
        paths[a] = planner(grid, s, g, frozenset(), horizon)
        if paths[a] is None:
            return None, stats
    conflict = find_conflict(paths)
    if conflict is None:
        return paths, stats

    counter = itertools.count()
    cost = lambda ps: sum(len(p) - 1 for p in ps.values())

    def node(cons, ps, depth, conf):
        k = next(counter)
        return {"k": k, "id": f"N{k}", "cons": cons, "paths": ps, "cost": cost(ps), "depth": depth, "conflict": conf}

    open_ = [node({a: frozenset() for a in agents}, paths, 0, conflict)]
    history = []
    while open_:
        open_.sort(key=lambda n: (n["cost"], n["k"]))
        cands = open_[:top_k]
        if len(cands) == 1:
            chosen = cands[0]                           # nothing to decide
        else:
            prompt = make_node_prompt(grid, agents, cands, history)
            pick = None
            for attempt in range(2):                    # one retry on an invalid answer
                if stats["llm_calls"] >= max_calls:
                    stats["hit_limit"] = True
                    return None, stats
                stats["llm_calls"] += 1
                text = llm(prompt if attempt == 0 else
                           prompt + "\n\nYour last reply was not valid. Reply with ONLY the JSON, using one of the option ids.")
                pick = parse_choice(text, [c["id"] for c in cands], key="expand")
                if pick is not None:
                    break
                stats["invalid_answers"] += 1
            if pick is None:                            # still invalid: cheapest node, and we count it
                stats["fallbacks"] += 1
                chosen = cands[0]
            else:
                chosen = next(c for c in cands if c["id"] == pick)
        open_.remove(chosen)
        stats["nodes"] += 1
        if stats["nodes"] > max_nodes:
            stats["hit_limit"] = True
            return None, stats

        made = []
        for agent in sorted(chosen["conflict"]):
            cons = dict(chosen["cons"])
            cons[agent] = chosen["cons"][agent] | {chosen["conflict"][agent]}
            s, g = agents[agent]
            stats["replans"] += 1
            new_path = planner(grid, s, g, cons[agent], horizon)
            if new_path is None:
                stats["backtracks"] += 1                # dropped child
                continue
            ps = dict(chosen["paths"])
            ps[agent] = new_path
            conf = find_conflict(ps)
            if conf is None:
                return ps, stats                        # a conflict-free plan: done
            child = node(cons, ps, chosen["depth"] + 1, conf)
            open_.append(child)
            made.append(f"{child['id']} ({child['cost']} steps, forbid {agent})")
        history.append(f"expanded {chosen['id']} (depth {chosen['depth']}, {chosen['cost']} steps) -> "
                       + (", ".join(made) if made else "no children"))
    return None, stats


# ---------------------------------------------------------------------------------------------
# Ablation: the earlier depth-first version (LLM only orders the two children at each node).
# ---------------------------------------------------------------------------------------------

def solve_hybrid_dfs(grid, agents, llm, horizon=30, max_nodes=2000, max_calls=200,
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