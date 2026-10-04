"""
Approach 2: LLM arbitrator, no tree.

Same job as Approach 1 (pick which agent to constrain at a conflict), but an
LLM makes the choice instead of a fixed rule. There is no stack and no undo:
constraints only pile up. The only "backtracking" is in context: if the chosen
agent cannot replan, the LLM is told and asked again for the same conflict.
Giving up happens when both agents fail at one conflict or the call budget runs out.

Everything else is identical to Approach 1 (same conflict finder, same
constraint, same low-level planner), so the only difference between the two
approaches is who makes the choice.
"""
import json
import os
import re

from cbs_baseline import find_conflict, plan_single


# ---------- LLM backends: each is a callable prompt -> text, with a .log ----------

class AnthropicLLM:
    """Real model. Needs `pip install anthropic` and ANTHROPIC_API_KEY set."""

    def __init__(self, model=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ARBITER_MODEL", "claude-sonnet-5-5")
        self.log = []

    def __call__(self, prompt):
        r = self.client.messages.create(
            model=self.model, max_tokens=300, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        text = r.content[0].text
        self.log.append({"prompt": prompt, "response": text})
        return text


class FakeLLM:
    """Offline stand-in for tests. `chooser(options, prompt) -> agent name`."""

    def __init__(self, chooser):
        self.chooser, self.log = chooser, []

    def __call__(self, prompt):
        options = re.findall(r"OPTION (\w+):", prompt)
        text = json.dumps({"constrain": self.chooser(options, prompt), "reason": "fake"})
        self.log.append({"prompt": prompt, "response": text})
        return text


# ---------- prompt and parsing ----------

def _picture(grid, paths):
    return "\n".join("".join(row) for row in grid.rows)


def make_prompt(grid, agents, paths, conflict, notes):
    lines = ["Several agents move on a grid, one step per time unit (a step can be a wait).",
             "Each agent planned alone. Two of the plans collide. Pick ONE agent to be",
             "forbidden from the colliding cell at that time. That agent will replan around it.",
             "",
             "Grid (row 0 is the top, '#' is a wall):", _picture(grid, paths), "",
             "Agents (start -> goal) and their current plans, as the cell at t=0,1,2,...:"]
    for a in sorted(agents):
        s, g = agents[a]
        lines.append(f"  {a}: {s} -> {g}, plan {paths[a]}")
    lines += ["", "The collision. You can only constrain one of these agents:"]
    for a in sorted(conflict):
        cell, t = conflict[a]
        lines.append(f"  OPTION {a}: forbid {a} from being at {cell} at time {t}")
    if notes:
        lines += ["", "What already happened at this collision:"] + [f"  - {n}" for n in notes]
    lines += ["", 'Answer with JSON only, like {"constrain": "<agent name>", "reason": "<one sentence>"}']
    return "\n".join(lines)


def parse_choice(text, options):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        choice = json.loads(m.group(0)).get("constrain")
    except (ValueError, AttributeError):
        return None
    return choice if choice in options else None


# ---------- the arbitrator ----------

def solve_llm(grid, agents, llm, horizon=30, max_calls=20, planner=plan_single):
    """Returns (paths or None, stats), same shape as cbs_baseline.solve."""
    stats = {"llm_calls": 0, "replans": 0, "failed_replans": 0,
             "invalid_answers": 0, "hit_limit": False}
    constraints = {a: frozenset() for a in agents}
    paths = {}
    for a, (s, g) in agents.items():
        stats["replans"] += 1
        paths[a] = planner(grid, s, g, frozenset(), horizon)
        if paths[a] is None:
            return None, stats

    while True:
        conflict = find_conflict(paths)
        if conflict is None:
            return paths, stats
        failed, notes, resolved = set(), [], False
        while not resolved:
            options = [a for a in sorted(conflict) if a not in failed]
            if not options:                      # both agents failed here: give up
                return None, stats
            if stats["llm_calls"] >= max_calls:
                stats["hit_limit"] = True
                return None, stats
            stats["llm_calls"] += 1
            text = llm(make_prompt(grid, agents, paths,
                                   {a: conflict[a] for a in options}, notes))
            agent = parse_choice(text, options)
            if agent is None:
                stats["invalid_answers"] += 1
                notes.append(f"Your last answer was not valid. Choose one of: {options}.")
                continue
            cons = constraints[agent] | {conflict[agent]}
            s, g = agents[agent]
            stats["replans"] += 1
            new_path = planner(grid, s, g, cons, horizon)
            if new_path is None:
                stats["failed_replans"] += 1
                failed.add(agent)
                notes.append(f"{agent} has no path that avoids {conflict[agent][0]} "
                             f"at time {conflict[agent][1]}. Choose the other agent.")
                continue
            constraints[agent] = cons
            paths[agent] = new_path
            resolved = True