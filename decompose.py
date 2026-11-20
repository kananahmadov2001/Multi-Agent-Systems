
"""Stage 2 of the pipeline: the LLM, as the central agent, splits the task over the selected team.

Input is the task, the team, and `sites` (part name -> the grid cell where that part is done).
Output is `({agent: [waypoint, ...]}, None)` or `(None, reason)`. A member given several parts
gets several waypoints, served in the order given; a member given none waits on its start cell.
The goals feed cbs_baseline.plan_route unchanged.

The LLM reads the task in words and each member's OWN natural-language description. It never
sees capability tags. It decides (1) whether this team can do the task at all, (2) which member
does each part, (3) the order of each member's route. If it says the team cannot, the pipeline
picks another team.

Hardcoded: the list of parts and their required order (the task spec), the cells, the agents'
start cells, and the validator `check_routes`, which judges every reply against the true
capabilities and rejects it with a specific message that the LLM sees on its retry. No reply
reaches the planner unless it passes, and there is no fallback: if no valid reply arrives, the
decomposition fails. Standard library only.
"""
import json


def world_preflight(task, team, sites):
    """Reason the split cannot even be attempted (missing site or start cell), or None."""
    if task.requires - set(sites):
        return f"no site for {sorted(task.requires - set(sites))}"
    if any(a.start is None for a in team):
        return "a selected agent has no start cell"
    return None


def goals_from_routes(team, sites, routes):
    """{agent: [part, ...]} -> {agent: [cell, ...]}; a member with no part waits on its start."""
    return {a.name: ([sites[p] for p in routes.get(a.name, [])] or [a.start]) for a in team}


def make_decompose_prompt(task, team, sites, grid=None, notes=()):
    """The task in words, the part names and cells, and each member's OWN description (no tags)."""
    lines = ["You are the central agent of a team of agents that move on a grid, one step per time unit.",
             "You chose this team from the agents' own descriptions. Now split the task over it.",
             "",
             "Split the task into one subgoal per team member. A subgoal is an ordered list of task parts.",
             "Give every part to exactly one member whose description says it can do that part. A member",
             "may take several parts (it visits their cells in the order you list them) or none (it then",
             "waits where it starts). Keep the parts in the task order for any one member.",
             "If nobody on this team can do some part, say so. Do not give a part to a member whose",
             "description does not support it.",
             "", f"Task: {task.description}"]
    if grid is not None:
        lines += ["", "Grid (row 0 is the top, '#' is a wall):", "\n".join("".join(r) for r in grid.rows)]
    lines += ["", "Parts of the task, in the order they belong:"]
    for tag in task.order:
        lines.append(f"  PART {tag}: done at cell {sites[tag]}")
    lines += ["", "Team (each agent's own description of itself):"]
    for a in sorted(team, key=lambda a: a.name):
        lines.append(f"  {a.name}: starts at {a.start}; \"{a.advert}\"")
    if notes:
        lines += ["", "Problems with your previous answer:"] + [f"  - {n}" for n in notes]
    lines += ["", "Reply with ONLY one of these JSON objects, no other text:",
              '  {"feasible": true, "routes": {"<agent>": ["<part>", ...], ...}, "reason": "<one sentence>"}',
              '  {"feasible": false, "missing": ["<part>", ...], "reason": "<one sentence>"}']
    return "\n".join(lines)


def parse_decomposition(text):
    """The LAST JSON object in `text` that has a boolean "feasible", else None."""
    decoder, found = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("feasible"), bool):
            found = obj
    return found


def check_routes(task, team, routes):
    """Problems with an LLM's {agent: [part, ...]} reply, judged against the TRUE capabilities
    and the task order. An empty list means valid."""
    if not isinstance(routes, dict):
        return ["\"routes\" is not an object that maps each agent to a list of parts"]
    members = {a.name: a for a in team}
    problems, given = [], {}
    for who in sorted(routes):
        route = routes[who]
        if who not in members:
            problems.append(f"{who!r} is not on the team")
            continue
        if not isinstance(route, list) or not all(isinstance(p, str) for p in route):
            problems.append(f"the route of {who} is not a list of part names")
            continue
        last = -1
        for part in route:
            if part not in task.order:
                problems.append(f"{part!r} is not a part of this task")
                continue
            if part in given:
                problems.append(f"{part!r} is given twice (to {given[part]} and to {who})")
                continue
            given[part] = who
            if part not in members[who].capabilities:
                problems.append(f"{who} cannot do {part!r}")
            index = task.order.index(part)
            if index < last:
                problems.append(f"{who} lists {part!r} out of task order {list(task.order)}")
            last = max(last, index)
    for part in task.order:
        if part not in given:
            problems.append(f"part {part!r} is not given to anyone")
    return problems


def llm_decompose(task, team, sites, llm, grid=None, max_calls=3, stats=None):
    """The LLM decomposes the task over the team. Returns (goals, None) or (None, reason).

    `llm` is a callable prompt -> text (llm_arbitrator.AnthropicLLM). A reply that says the team
    cannot do the task ends the call at once and is trusted: the pipeline then picks another team.
    Any other reply is validated by check_routes and retried with the problems listed. `stats`
    accumulates llm_calls / invalid_answers / declared_infeasible across calls.
    """
    stats = stats if stats is not None else {}
    for key in ("llm_calls", "invalid_answers", "declared_infeasible"):
        stats.setdefault(key, 0)
    why = world_preflight(task, team, sites)
    if why:
        return None, why
    notes, last = [], "no reply"
    for _ in range(max_calls):
        stats["llm_calls"] += 1
        reply = parse_decomposition(llm(make_decompose_prompt(task, team, sites, grid, notes)))
        if reply is None:
            problems = ["the reply was not one of the two required JSON objects"]
        elif reply["feasible"] is False:
            stats["declared_infeasible"] += 1
            detail = reply.get("reason") or f"missing {reply.get('missing')}"
            return None, f"llm declared the team cannot do the task: {detail}"
        else:
            problems = check_routes(task, team, reply.get("routes"))
            if not problems:
                return goals_from_routes(team, sites, reply["routes"]), None
        stats["invalid_answers"] += 1
        notes, last = problems, problems[0]
    return None, f"no valid decomposition after {max_calls} call(s): {last}"
