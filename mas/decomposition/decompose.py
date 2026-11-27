"""Stage 2 of the pipeline: split the task over the selected team.

Input is the task, the team, and `sites` (part name -> the grid cell where that part is
done). Output is `({agent: [waypoint, ...]}, None)` or `(None, reason)`. A member given
several parts gets several waypoints, served in the order given; a member given none waits
on its start cell. The goals feed cbs_baseline.plan_route unchanged.

Three decomposers share one signature, so run_pipeline can swap them (`decomposer=`):

  greedy_decompose      free. Each part, in task order, goes to the capable member that adds
                        the least travel. Reads the capability TAGS (hardcoded ground truth).
  exhaustive_decompose  free. Tries every valid assignment, keeps the least total travel.
                        Also reads the tags. The reference the LLM is compared against.
  llm_decompose         the central agent's LLM decomposes from the agents' NATURAL-LANGUAGE
                        descriptions only. It never sees the tags. It may also answer that the
                        team cannot do the task, which makes the pipeline pick another team.

What the LLM decides in llm_decompose: whether this team can do the task at all, which member
does each part, and the order of each member's route. What stays hardcoded: the list of parts
and their required order (the task spec), the cells, the agents' start cells, and the
validator below, which checks every reply against the true capabilities and rejects it with a
specific message. No reply can reach the planner unless it passes that check, and there is no
silent fallback: if no valid reply arrives, the decomposition fails.

Standard library only.

    python3 -m mas.decomposition.decompose       # self-check, free (scripted fake LLMs)
"""
import json
from itertools import product


# ---------- helpers ----------

def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def route_cost(start, route):
    """Manhattan travel from `start` through `route` in order."""
    total, here = 0, start
    for cell in route:
        total += _distance(here, cell)
        here = cell
    return total


def travel_cost(goals, team):
    """Total travel of a decomposition: the number to compare decomposers on."""
    starts = {a.name: a.start for a in team}
    return sum(route_cost(starts[n], route) for n, route in goals.items())


def world_preflight(task, team, sites):
    """Reason the split cannot even be attempted (missing site or start cell), or None."""
    if task.requires - set(sites):
        return f"no site for {sorted(task.requires - set(sites))}"
    if any(a.start is None for a in team):
        return "a selected agent has no start cell"
    return None


def preflight(task, team, sites):
    """world_preflight plus the tag check the two free decomposers rely on."""
    why = world_preflight(task, team, sites)
    if why:
        return why
    for tag in task.order:
        if not any(tag in a.capabilities for a in team):
            return f"no selected agent can do {tag!r}"
    return None


class GoalMap(dict):
    """Waypoint lists that remain a dict, plus the parts that produced them."""

    def __init__(self, goals, parts):
        super().__init__(goals)
        self.parts = parts


def goals_from_routes(team, sites, routes):
    """Keep both {agent: [cell, ...]} and the part name behind each waypoint."""
    goals = {a.name: ([sites[p] for p in routes.get(a.name, [])] or [a.start]) for a in team}
    parts = {a.name: list(routes.get(a.name, [])) for a in team}
    return GoalMap(goals, parts)


# ---------- Decomposers 1 and 2: free, tag-based ----------

def greedy_decompose(task, team, sites):
    """task + team -> ({agent: [cell, ...]}, None), or (None, reason)."""
    why = preflight(task, team, sites)
    if why:
        return None, why
    where = {a.name: a.start for a in team}
    routes = {a.name: [] for a in team}
    for tag in task.order:
        options = [a for a in team if tag in a.capabilities]
        best = min(options, key=lambda a: (_distance(where[a.name], sites[tag]), a.name))
        routes[best.name].append(tag)
        where[best.name] = sites[tag]
    return goals_from_routes(team, sites, routes), None


decompose = greedy_decompose        # the name earlier code imported


def exhaustive_decompose(task, team, sites, max_assignments=50000):
    """The assignment with the least total travel (ties: first in name order)."""
    why = preflight(task, team, sites)
    if why:
        return None, why
    members = sorted(team, key=lambda a: a.name)
    options = [[a.name for a in members if tag in a.capabilities] for tag in task.order]
    count = 1
    for o in options:
        count *= len(o)
    if count > max_assignments:
        return None, f"{count} assignments exceeds the cap of {max_assignments}"
    best, best_cost = None, None
    for combo in product(*options):
        routes = {a.name: [] for a in team}
        for tag, who in zip(task.order, combo):
            routes[who].append(tag)
        goals = goals_from_routes(team, sites, routes)
        cost = travel_cost(goals, team)
        if best_cost is None or cost < best_cost:
            best, best_cost = goals, cost
    return best, None


# ---------- Decomposer 3: the LLM, from natural-language descriptions ----------

def make_decompose_prompt(task, team, sites, grid=None, notes=()):
    """Shows the task in words, the part names and cells, and each member's OWN description.
    The capability tags are not shown."""
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

    `llm` is a callable prompt -> text (llm_arbitrator.AnthropicLLM works). A reply that says
    the team cannot do the task ends the call at once and is trusted: the pipeline then picks
    another team. Any other reply is validated by check_routes and retried with the problems
    listed. `stats` accumulates llm_calls / invalid_answers / declared_infeasible across calls.
    Hand it to run_pipeline with functools.partial(llm_decompose, llm=llm, grid=grid).
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


# ---------- self-check ----------

class Scripted:
    """Offline stand-in: returns the given replies in order, remembers the prompts."""
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


def main():
    from mas.selection.agent_pool import Agent, Task

    # greedy reproduces the rule pipeline.py used inline
    two = [Agent("c1", "carries crates and inspects rooms", start=(0, 0)),
           Agent("c2", "moves around", start=(0, 4))]
    t2 = Task.from_text("carry the crate and inspect it")
    goals, why = greedy_decompose(t2, two, {"carry": (2, 2), "inspect": (2, 3)})
    assert why is None and goals == {"c1": [(2, 2), (2, 3)], "c2": [(0, 4)]}, goals
    assert goals.parts == {"c1": ["carry", "inspect"], "c2": []}, goals.parts

    # greedy is not always optimal: a1 is nearer the carry site but is the only inspector
    team = [Agent("a1", "carries crates and inspects rooms", start=(0, 0)),
            Agent("a2", "carries crates", start=(3, 4))]
    task = Task("carry then inspect", {"carry", "inspect"}, order=["carry", "inspect"])
    sites = {"carry": (3, 0), "inspect": (0, 3)}
    g, _ = greedy_decompose(task, team, sites)
    e, _ = exhaustive_decompose(task, team, sites)
    assert travel_cost(g, team) == 9 and travel_cost(e, team) == 7
    assert e == {"a1": [(0, 3)], "a2": [(3, 0)]}, e

    # LLM: a correct reply is accepted in one call, and the prompt never shows the tags
    stats = {}
    llm = Scripted('{"feasible": true, "routes": {"a2": ["carry"], "a1": ["inspect"]}, "reason": "a1 inspects"}')
    goals, why = llm_decompose(task, team, sites, llm, stats=stats)
    assert goals == e and why is None, (goals, why)
    assert stats == {"llm_calls": 1, "invalid_answers": 0, "declared_infeasible": 0}, stats
    assert "['carry'" not in llm.prompts[0] and "capabilit" not in llm.prompts[0]   # NL only, no tags

    # LLM: wrong capability, duplicate part and bad order are each named, then a valid retry
    stats = {}
    llm = Scripted('thinking {"feasible": true, "routes": {"a2": ["carry", "inspect"]}, "reason": "x"}',
                   '{"feasible": true, "routes": {"a1": ["inspect", "carry"]}, "reason": "x"}',
                   '{"feasible": true, "routes": {"a1": ["carry", "inspect"]}, "reason": "ok"}')
    goals, why = llm_decompose(task, team, sites, llm, stats=stats)
    assert goals == g and stats["llm_calls"] == 3 and stats["invalid_answers"] == 2, (goals, stats)
    assert "a2 cannot do 'inspect'" in llm.prompts[1], llm.prompts[1]
    assert "out of task order" in llm.prompts[2], llm.prompts[2]

    # LLM: "this team cannot do it" is trusted and ends the call
    stats = {}
    llm = Scripted('{"feasible": false, "missing": ["inspect"], "reason": "nobody inspects"}')
    goals, why = llm_decompose(task, [team[1]], sites, llm, stats=stats)
    assert goals is None and "nobody inspects" in why and stats["declared_infeasible"] == 1 and len(llm.prompts) == 1

    # LLM: never valid -> a failure, not a silent fallback
    stats = {}
    goals, why = llm_decompose(task, team, sites, Scripted("no idea"), max_calls=2, stats=stats)
    assert goals is None and "2 call(s)" in why and stats["invalid_answers"] == 2

    # a missing site or start cell costs no LLM call
    llm = Scripted("unused")
    assert llm_decompose(task, team, {"carry": (0, 0)}, llm)[1] == "no site for ['inspect']" and not llm.prompts
    assert greedy_decompose(task, [Agent("z", "carries crates and inspects rooms")], sites)[1] \
        == "a selected agent has no start cell"

    print("\n----- a decomposition prompt -----\n" + make_decompose_prompt(task, team, sites))
    print("\nall checks passed")


if __name__ == "__main__":
    main()
