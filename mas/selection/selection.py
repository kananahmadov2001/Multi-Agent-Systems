"""Selection: pick exactly k agents whose adverts cover the task.

cheapest / greedy / coverage on the same AgentPool interface, each returning
ranked alternatives for team-level backtracking. Standard library only.

    python3 selection.py       # self-check: prints what each policy picks
"""
from mas.selection.agent_pool import Agent, AgentPool, Task


class Selection:
    """One policy's answer: the team, its cost, what it misses, and the fallbacks."""

    def __init__(self, strategy, team, cost, missing, alternatives=()):
        self.strategy = strategy
        self.team = tuple(team)
        self.cost = cost
        self.missing = frozenset(missing)
        self.alternatives = tuple(alternatives)

    @property
    def ok(self):
        return not self.missing

    @property
    def names(self):
        return [a.name for a in self.team]

    def __repr__(self):
        return (f"Selection({self.strategy}, {self.names}, cost={self.cost}, "
                f"missing={sorted(self.missing)}, alternatives={len(self.alternatives)})")


def _score_per_cost(agent, remaining):
    gain = len(agent.capabilities & remaining)
    return (gain / agent.cost, gain, -agent.cost)


def _score_coverage(agent, remaining):
    return (len(agent.capabilities & remaining), -agent.cost)


def _greedy_team(pool, task, k, score):
    """Add the best-scoring agent until k slots are used or nothing helps.

    Leftover slots are filled with the cheapest agents, because the proposal
    fixes the team size at exactly k. Ties go to the alphabetically first name
    (the pool is walked in name order), so a run is reproducible.
    """
    remaining = set(task.requires)
    chosen = []
    while len(chosen) < k and remaining:
        options = [a for a in sorted(pool, key=lambda x: x.name)
                   if a not in chosen and a.capabilities & remaining]
        if not options:
            break
        best = max(options, key=lambda a: score(a, remaining))
        chosen.append(best)
        remaining -= best.capabilities
    for a in sorted(pool, key=lambda x: (x.cost, x.name)):
        if len(chosen) >= k:
            break
        if a not in chosen:
            chosen.append(a)
    return tuple(chosen)


def _fallbacks(pool, task, k, chosen, limit):
    """The next-cheapest covering teams, minus the one already chosen."""
    taken = tuple(sorted(a.name for a in chosen))
    ranked = pool.teams(task, k, limit + 1)
    return tuple(t for t in ranked if tuple(sorted(a.name for a in t)) != taken)[:limit]


def _cheapest(pool, task, k, limit):
    ranked = pool.teams(task, k, limit)
    return (ranked[0], tuple(ranked[1:])) if ranked else ((), ())


def _greedy(pool, task, k, limit):
    team = _greedy_team(pool, task, k, _score_per_cost)
    return team, _fallbacks(pool, task, k, team, limit)


def _coverage(pool, task, k, limit):
    team = _greedy_team(pool, task, k, _score_coverage)
    return team, _fallbacks(pool, task, k, team, limit)


POLICIES = {"cheapest": _cheapest, "greedy": _greedy, "coverage": _coverage}


def select(pool, task, k, strategy="cheapest", limit=5):
    """Apply one policy. Returns a Selection even when no covering team exists."""
    if k < 1:
        raise ValueError("team size k must be at least 1")
    if strategy not in POLICIES:
        raise ValueError(f"unknown strategy {strategy!r}; have {sorted(POLICIES)}")
    team, alternatives = POLICIES[strategy](pool, task, k, limit)
    return Selection(strategy, team, pool.team_cost(team), pool.missing(team, task), alternatives)


def main():
    # Same task, two pools: the second is built so that greedy dead-ends.
    simple = AgentPool([
        Agent("a1", "walks around the grid and carries small boxes", cost=1.0),
        Agent("a2", "flies over walls", cost=2.0),
        Agent("a3", "moves on the ground and inspects rooms", cost=1.5),
        Agent("a4", "carries heavy crates between rooms", cost=1.2),
    ])
    tricky = AgentPool([
        Agent("g1", "moves and carries", cost=1.0),
        Agent("g2", "moves", cost=0.4),
        Agent("g3", "carries and inspects", cost=1.5),
        Agent("g4", "inspects", cost=0.6),
    ])
    task = Task.from_text("move the crate, carry it to the bay and inspect it")

    for label, pool, k in (("simple, k=2", simple, 2), ("tricky, k=2", tricky, 2)):
        print(f"\n{label}  task requires {sorted(task.requires)}")
        for strategy in POLICIES:
            s = select(pool, task, k, strategy)
            flag = "ok" if s.ok else f"MISSING {sorted(s.missing)}"
            print(f"  {strategy:<9} {str(s.names):<22} cost={s.cost:<4} {flag:<18} "
                  f"fallbacks={len(s.alternatives)}")

    best = select(tricky, task, 2)
    greedy = select(tricky, task, 2, "greedy")
    assert best.names == ["g1", "g4"] and best.cost == 1.6
    assert [[a.name for a in t] for t in best.alternatives] == [["g2", "g3"], ["g1", "g3"]]
    assert not greedy.ok and greedy.missing == {"carry"}, greedy
    assert len(greedy.team) == 2, "the team still has exactly k members"
    assert select(tricky, task, 1, "cheapest").team == (), "no single agent covers it"
    assert select(simple, task, 2, "coverage").ok
    assert select(simple, task, 3).alternatives, "k=3 has cheaper fallbacks"
    print("\nall checks passed")


if __name__ == "__main__":
    main()
