"""The pipeline: select a team, decompose, plan, verify, switch teams on failure.

Selection runs before decomposition, so every candidate team gets its own split.
A team that cannot be reconciled is dropped and the next covering team is tried,
which is the missing edge from Section 6. Standard library only.

    python3 pipeline.py       # self-check: one solved run and each failure kind
"""
from agent_pool import Agent, AgentPool, Task
from selection import select
from cbs_baseline import Grid, plan_single, solve_best_first
from verify import verify

HORIZON = 20
MAX_NODES = 2000

# What an attempt can end as. selection and decomposition happen before any
# planning, so the published failure breakdown is selection vs arbitration.
KINDS = ("solved", "selection", "decomposition", "arbitration", "verification")


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def decompose(task, team, sites):
    """task + team -> ({agent: [cell, ...]}, None), or (None, reason).

    `sites` maps a capability tag to the cell where that part of the task is
    done. Tags are assigned in task order to the capable member that adds the
    least travel, so one member can take several tags (it then gets several
    waypoints) and a member with none waits on its start.
    """
    if task.requires - set(sites):
        return None, f"no site for {sorted(task.requires - set(sites))}"
    by_name = {a.name: a for a in team}
    if any(a.start is None for a in team):
        return None, "a selected agent has no start cell"

    routes = {name: [] for name in by_name}
    where = {name: by_name[name].start for name in by_name}
    for tag in task.order:
        options = [a for a in team if tag in a.capabilities]
        if not options:
            return None, f"no selected agent can do {tag!r}"
        best = min(options, key=lambda a: (_distance(where[a.name], sites[tag]), a.name))
        routes[best.name].append(sites[tag])
        where[best.name] = sites[tag]
    return {name: (route or [by_name[name].start]) for name, route in routes.items()}, None


class Attempt:
    """One team tried, how it ended, and what it produced."""

    def __init__(self, names, kind, detail, goals=None, paths=None, stats=None):
        assert kind in KINDS, kind
        self.names = list(names)
        self.kind = kind
        self.detail = detail
        self.goals = goals
        self.paths = paths
        self.stats = stats

    @property
    def ok(self):
        return self.kind == "solved"

    def __repr__(self):
        tail = f" ({self.detail})" if self.detail else ""
        return f"Attempt({self.names}, {self.kind}{tail})"


class PipelineResult:
    """Every attempt made, plus the winning one if there is one."""

    def __init__(self, strategy, attempts):
        self.strategy = strategy
        self.attempts = list(attempts)
        self.winner = next((a for a in self.attempts if a.ok), None)

    @property
    def ok(self):
        return self.winner is not None

    @property
    def names(self):
        return self.winner.names if self.winner else []

    @property
    def goals(self):
        return self.winner.goals if self.winner else None

    @property
    def paths(self):
        return self.winner.paths if self.winner else None

    @property
    def stats(self):
        return self.winner.stats if self.winner else None

    @property
    def failure(self):
        """None when solved, else the kind of the last attempt."""
        return None if self.ok else self.attempts[-1].kind

    @property
    def breakdown(self):
        out = {}
        for a in self.attempts:
            out[a.kind] = out.get(a.kind, 0) + 1
        return out

    def __repr__(self):
        return f"PipelineResult({self.strategy}, ok={self.ok}, names={self.names})"


def run_pipeline(grid, pool, task, k, sites, strategy="cheapest", horizon=HORIZON,
                 max_nodes=MAX_NODES, max_teams=3, planner=plan_single,
                 solver=solve_best_first, checker=verify):
    """Select, decompose, plan, verify; on failure try the next covering team.

    `max_teams` bounds the switch edge because each extra team is another full
    CBS run. A team that misses a required tag is never planned for: that is a
    selection failure by construction, which is what the breakdown reports.
    """
    if max_teams < 1:
        raise ValueError("max_teams must be at least 1")
    chosen = select(pool, task, k, strategy)
    candidates = [chosen.team] + list(chosen.alternatives)
    seen, attempts = set(), []
    for team in candidates:
        if len(attempts) >= max_teams:
            break
        names = sorted(a.name for a in team)
        if tuple(names) in seen:
            continue
        seen.add(tuple(names))
        missing = pool.missing(team, task)
        if missing:
            attempts.append(Attempt(names, "selection", f"does not cover {sorted(missing)}"))
            continue
        goals, why = decompose(task, team, sites)
        if goals is None:
            attempts.append(Attempt(names, "decomposition", why))
            continue
        members = {a.name: a for a in team}
        mapf = {n: (members[n].start, g) for n, g in goals.items()}
        paths, stats = solver(grid, mapf, horizon=horizon, max_nodes=max_nodes, planner=planner)
        if paths is None:
            why = "hit the node cap" if stats["hit_limit"] else "no conflict-free plan"
            attempts.append(Attempt(names, "arbitration", why, goals=goals, stats=stats))
            continue
        ok, problems = checker(grid, mapf, paths)
        if not ok:
            attempts.append(Attempt(names, "verification", problems[0],
                                    goals=goals, paths=paths, stats=stats))
            continue
        attempts.append(Attempt(names, "solved", "", goals=goals, paths=paths, stats=stats))
        break
    return PipelineResult(strategy, attempts)


def describe(result):
    """The lines the demo prints: what each attempt was, and how it ended."""
    lines = [f"strategy {result.strategy}: {len(result.attempts)} attempt(s)"]
    for i, a in enumerate(result.attempts, 1):
        line = f"  {i}. {a.names} -> {a.kind}"
        if a.detail:
            line += f": {a.detail}"
        lines.append(line)
        if a.goals:
            lines.append(f"     goals {a.goals}")
        if a.stats:
            line = f"     nodes={a.stats['nodes']} replans={a.stats['replans']}"
            if a.paths:
                line += f" steps={sum(len(p) - 1 for p in a.paths.values())}"
            lines.append(line)
    lines.append(f"  ok={result.ok} names={result.names} failure={result.failure}")
    return "\n".join(lines)


def main():
    # 1. The plain case: cheapest team, both tags on different agents, solves.
    open_grid = Grid([".....", ".....", ".....", ".....", "....."])
    pool = AgentPool([
        Agent("s1", "moves around and carries boxes", cost=1.0, start=(0, 0)),
        Agent("s2", "moves quickly", cost=0.4, start=(0, 4)),
        Agent("s3", "carries crates and inspects rooms", cost=1.5, start=(4, 0)),
        Agent("s4", "inspects rooms", cost=0.6, start=(4, 4)),
    ])
    task = Task.from_text("carry the crate to the bay and inspect it")
    solved = run_pipeline(open_grid, pool, task, 2, {"carry": (2, 2), "inspect": (2, 3)})
    print(describe(solved))
    assert solved.ok and solved.names == ["s1", "s4"], solved.names
    assert solved.failure is None and solved.breakdown == {"solved": 1}
    assert solved.stats["nodes"] >= 1

    # 2. Arbitration failure, then a different team: b1 sits in a sealed pocket,
    # so the cheapest team cannot plan at all and the switch edge is what saves it.
    pocket = Grid(["#.#..", "###..", ".....", ".....", "....."])
    sealed = AgentPool([
        Agent("b1", "carries crates", cost=0.5, start=(0, 1)),
        Agent("b2", "inspects rooms", cost=0.2, start=(4, 4)),
        Agent("g1", "carries crates", cost=1.0, start=(4, 0)),
        Agent("g2", "inspects rooms", cost=1.0, start=(4, 2)),
    ])
    task2 = Task.from_text("carry the crate and inspect it")
    switched = run_pipeline(pocket, sealed, task2, 2,
                            {"carry": (2, 0), "inspect": (2, 4)}, max_teams=2)
    print("\n" + describe(switched))
    assert [a.kind for a in switched.attempts] == ["arbitration", "solved"], switched.attempts
    assert switched.names == ["b2", "g1"], switched.names

    # 3. Selection failure: nothing advertises flying, so nothing is planned.
    no_flyer = run_pipeline(open_grid, pool, Task.from_text("fly to the tower"), 2,
                            {"fly": (0, 0)})
    print("\n" + describe(no_flyer))
    assert not no_flyer.ok and no_flyer.failure == "selection"
    assert no_flyer.breakdown == {"selection": 1}

    # 4. One agent takes both tags: it gets two waypoints, not a fake second goal.
    one_pair = AgentPool([
        Agent("c1", "carries crates and inspects rooms", cost=1.0, start=(0, 0)),
        Agent("c2", "moves around", cost=0.5, start=(0, 4)),
    ])
    both = run_pipeline(open_grid, one_pair, task2, 2,
                        {"carry": (2, 2), "inspect": (2, 3)})
    print("\n" + describe(both))
    assert both.ok, both.attempts
    assert both.goals == {"c1": [(2, 2), (2, 3)], "c2": [(0, 4)]}, both.goals

    # 5. Three tags on two agents, so selection really chooses how many are used.
    tricky = AgentPool([
        Agent("g1", "moves and carries", cost=1.0, start=(0, 0)),
        Agent("g2", "moves", cost=0.4, start=(0, 4)),
        Agent("g3", "carries and inspects", cost=1.5, start=(4, 0)),
        Agent("g4", "inspects", cost=0.6, start=(4, 4)),
    ])
    task3 = Task("move the crate, carry it and inspect it", {"move", "carry", "inspect"},
                 order=["move", "carry", "inspect"])
    three = run_pipeline(open_grid, tricky, task3, 2,
                         {"move": (0, 2), "carry": (2, 2), "inspect": (2, 4)})
    print("\n" + describe(three))
    assert three.ok, three.attempts
    assert three.goals == {"g1": [(0, 2), (2, 2)], "g4": [(2, 4)]}, three.goals

    print("\nall checks passed")


if __name__ == "__main__":
    main()
