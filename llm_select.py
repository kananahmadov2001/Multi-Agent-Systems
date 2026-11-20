"""Stage 1: the LLM, as the central agent, picks the team from natural-language descriptions.

It sees only each agent's advert, never capability tags, costs or start cells; a team that
failed is never proposed twice. Standard library only.

    python3 llm_select.py
"""
import json
from math import comb


def make_selection_prompt(pool, task, k, history, notes=()):
    lines = [f"You are the central agent. Choose a team of exactly {k} agents from the pool to carry out a task.",
             "You know only what each agent says about itself.",
             "", f"Task: {task.description}", "", "Pool (name: the agent's own description):"]
    for a in sorted(pool, key=lambda a: a.name):
        lines.append(f"  {a.name}: \"{a.advert}\"")
    if history:
        lines += ["", "Teams you already tried that did not work. Do not pick any of them again:"]
        for h in history:
            lines.append(f"  [{', '.join(h.names)}]: {h.detail or h.kind}")
    if notes:
        lines += ["", "Problems with your previous answer:"] + [f"  - {n}" for n in notes]
    lines += ["", f"Pick agents whose descriptions together cover every part of the task. The team must have exactly {k} distinct agents from the pool.",
              'Reply with ONLY this JSON, no other text: {"team": ["<agent name>", ...], "reason": "<one sentence>"}']
    return "\n".join(lines)


def parse_team(text):
    """The LAST JSON object in `text` that has a list under "team", else None."""
    decoder, found = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("team"), list):
            found = obj["team"]
    return found


def check_team(pool, k, team, tried):
    """Problems with a proposed team (names only, no capability judgement). Empty is valid."""
    if not all(isinstance(n, str) for n in team):
        return ["the team must be a list of agent names"]
    names = {a.name for a in pool}
    problems = [f"{n!r} is not in the pool" for n in team if n not in names]
    if len(set(team)) != len(team):
        problems.append("the team has a repeated agent")
    if len(team) != k:
        problems.append(f"the team has {len(team)} agents, not exactly {k}")
    if not problems and tuple(sorted(team)) in tried:
        problems.append(f"[{', '.join(sorted(team))}] was already tried")
    return problems


def propose_team(pool, task, k, llm, history=(), stats=None, max_calls=3):
    """Returns (team of Agent objects, None) or (None, reason).

    `history` holds the failed attempts so far (anything with .names, .kind, .detail); they are
    shown to the LLM and excluded. `stats` accumulates llm_calls / invalid_answers.
    """
    stats = stats if stats is not None else {}
    for key in ("llm_calls", "invalid_answers"):
        stats.setdefault(key, 0)
    if k > len(pool):
        return None, f"k={k} is larger than the pool of {len(pool)}"
    tried = {tuple(sorted(h.names)) for h in history}
    if len([t for t in tried if len(t) == k]) >= comb(len(pool), k):
        return None, f"every team of {k} has already been tried"
    notes, last = [], "no reply"
    for _ in range(max_calls):
        stats["llm_calls"] += 1
        team = parse_team(llm(make_selection_prompt(pool, task, k, history, notes)))
        problems = ["the reply was not the required JSON with a \"team\" list"] if team is None \
            else check_team(pool, k, team, tried)
        if not problems:
            by_name = {a.name: a for a in pool}
            return tuple(by_name[n] for n in team), None
        stats["invalid_answers"] += 1
        notes, last = problems, problems[0]
    return None, f"no valid team after {max_calls} call(s): {last}"


def main():
    from agent_pool import Agent, AgentPool, Task

    class Scripted:
        """Returns its replies in order, repeating the last one."""

        def __init__(self, *texts):
            self.texts, self.prompts = list(texts), []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            return self.texts[min(len(self.prompts), len(self.texts)) - 1]

    class Tried:
        names, kind, detail = ["a1", "a2"], "arbitration", "no conflict-free plan"

    pool = AgentPool([Agent("a1", "carries crates", cost=1.0),
                      Agent("a2", "inspects rooms", cost=1.2),
                      Agent("a3", "carries crates and inspects rooms", cost=2.0)])
    task = Task.from_text("carry the crate and inspect it")

    team, why = propose_team(pool, task, 2, Scripted('{"team": ["a1", "a2"], "reason": "x"}'))
    assert [a.name for a in team] == ["a1", "a2"] and why is None

    stats, llm = {}, Scripted('{"team": ["a1"], "reason": "x"}',
                              '{"team": ["a1", "a3"], "reason": "x"}')
    team, why = propose_team(pool, task, 2, llm, stats=stats)
    assert [a.name for a in team] == ["a1", "a3"] and stats["invalid_answers"] == 1
    assert "exactly 2" in llm.prompts[1] and stats["llm_calls"] == 2

    team, why = propose_team(pool, task, 2, Scripted('{"team": ["a1", "a2"]}'), history=[Tried()])
    assert team is None and "already tried" in why, (team, why)
    print("all checks passed")


if __name__ == "__main__":
    main()
