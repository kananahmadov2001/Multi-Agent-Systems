"""Colored box: every agent plans on its own view of the grid.

A cell may carry a colour and each agent has a palette; a cell outside the
palette is a wall to that agent. A covering team can therefore still be unable
to carry out its own split, which is a decomposition failure, not a selection
one. Standard library only.

    python3 colored_box.py       # self-check: a blind split, then a team switch
"""
import random

from agent_pool import Agent, AgentPool, Task
from cbs_baseline import Grid, plan_single, solve_best_first
from pipeline import Attempt, PipelineResult, greedy_decompose
from pipeline_instances import PipelineInstance, capability_overlap, reachable
from selection import select
from verify import verify

OPEN = "."
WALL = "#"


class ColoredGrid(Grid):
    """Digits are colours, '.' is open to everyone, '#' is a wall.

    An agent with no palette entry sees only the open cells.
    """

    def __init__(self, rows, palettes):
        super().__init__(rows)
        self.palettes = {name: frozenset(tags) for name, tags in palettes.items()}

    def sees(self, name, cell):
        """Whether `name` may stand on `cell`: walls and unseen colours say no."""
        r, c = cell
        if not (0 <= r < self.h and 0 <= c < self.w):
            return False
        ch = self.rows[r][c]
        return ch == OPEN or ch in self.palettes.get(name, frozenset())

    def view(self, name):
        """What `name` plans on: every cell outside its palette becomes a wall."""
        palette = self.palettes.get(name, frozenset())
        return Grid([
            "".join(ch if ch == OPEN or ch in palette else WALL for ch in row)
            for row in self.rows
        ])


def planner_for(cgrid, team):
    """One low-level planner per agent, looked up by start cell.

    solve_best_first only hands the planner a grid, not a name, so the agent's
    view has to be recovered from its start; these instances keep starts
    distinct, which is also what lets verify treat them as separate actors.
    """
    starts = [a.start for a in team]
    if len(set(starts)) != len(starts):
        raise ValueError("agents on one team must start on different cells")
    views = {a.start: cgrid.view(a.name) for a in team}

    def planner(_grid, start, goal, constraints, horizon):
        return plan_single(views[start], start, goal, constraints, horizon)

    return planner


def blind_spots(cgrid, team, goals):
    """Agents that cannot reach one of their own waypoints through their view."""
    out = []
    for a in team:
        view = cgrid.view(a.name)
        if any(not reachable(view, a.start, cell) for cell in goals[a.name]):
            out.append(a.name)
    return out


def colored_instances(n, seed, see=0.8, **kw):
    """The capability-overlap instances with the site cells painted a colour.

    A site on a colour an agent cannot see never enters that agent's view, so a
    team that covers the task can still be unable to carry out its own split.
    `see` is the chance an agent has a given colour; no agent is ever blinded to
    the cell it starts on.
    """
    rng = random.Random(seed + 1)
    out = []
    for inst in capability_overlap(n, seed, **kw):
        tags = sorted(inst.task.requires)
        colors = [str(i) for i in range(min(3, max(2, len(tags))))]
        paint = {tag: colors[i % len(colors)] for i, tag in enumerate(tags)}
        rows = [list(row) for row in inst.grid.rows]
        for tag, cell in inst.sites.items():
            rows[cell[0]][cell[1]] = paint[tag]
        palettes = {a.name: {c for c in colors if rng.random() < see} for a in inst.pool}
        for a in inst.pool:
            r, c = a.start
            if rows[r][c] in colors:          # you can always see where you stand
                palettes[a.name].add(rows[r][c])
        grid = ColoredGrid(["".join(r) for r in rows], palettes)
        out.append(PipelineInstance(inst.name, grid, inst.pool, inst.task, inst.sites,
                                    sealed=inst.sealed, scarce=inst.scarce))
    return out


def run_colored(cgrid, pool, task, k, sites, strategy="cheapest", max_teams=3,
                horizon=20, max_nodes=2000):
    """The pipeline, with each agent planning on its own coloured view.

    A split that lands a waypoint outside its owner's palette never reaches the
    solver: it is recorded as a decomposition failure and the next covering team
    is tried, which is the switch edge under a per-agent world model.
    """
    chosen = select(pool, task, k, strategy)
    seen, attempts = set(), []
    for team in [chosen.team] + list(chosen.alternatives):
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
        goals, why = greedy_decompose(task, team, sites)
        if goals is None:
            attempts.append(Attempt(names, "decomposition", why))
            continue
        blind = blind_spots(cgrid, team, goals)
        if blind:
            attempts.append(Attempt(names, "decomposition",
                                    f"cannot see its site: {', '.join(sorted(blind))}",
                                    goals=goals))
            continue
        members = {a.name: a for a in team}
        mapf = {n: (members[n].start, g) for n, g in goals.items()}
        paths, stats = solve_best_first(cgrid, mapf, horizon=horizon,
                                        max_nodes=max_nodes, planner=planner_for(cgrid, team))
        if paths is None:
            why = "hit the node cap" if stats["hit_limit"] else "no conflict-free plan"
            attempts.append(Attempt(names, "arbitration", why, goals=goals, stats=stats))
            continue
        ok, problems = verify(cgrid, mapf, paths)
        if not ok:
            attempts.append(Attempt(names, "verification", problems[0],
                                    goals=goals, paths=paths, stats=stats))
            continue
        attempts.append(Attempt(names, "solved", "", goals=goals,
                                paths=paths, stats=stats))
        break
    return PipelineResult(strategy, attempts)


def describe(result):
    """The lines the demo prints, one per attempt."""
    lines = [f"strategy {result.strategy}: {len(result.attempts)} attempt(s)"]
    for i, a in enumerate(result.attempts, 1):
        line = f"  {i}. {a.names} -> {a.kind}"
        if a.detail:
            line += f": {a.detail}"
        lines.append(line)
        if a.paths:
            lines.append(f"     steps={sum(len(p) - 1 for p in a.paths.values())}")
    lines.append(f"  ok={result.ok} names={result.names} failure={result.failure}")
    return "\n".join(lines)


def main():
    # The right column is colour 1, so only an agent whose palette has 1 can
    # stand on it. c1 and c2 are the cheapest covering team but c1 cannot see it.
    cgrid = ColoredGrid(
        ["....1",
         "....1",
         "....1",
         ".....",
         "....."],
        {"c1": "", "c2": "", "g1": "1"},
    )
    c1 = Agent("c1", "carries crates", cost=1.0, start=(0, 0))
    c2 = Agent("c2", "inspects rooms", cost=0.4, start=(0, 3))
    g1 = Agent("g1", "carries crates", cost=1.8, start=(4, 0))
    pool = AgentPool([c1, c2, g1])
    task = Task("carry then inspect", {"carry", "inspect"}, ("carry", "inspect"))
    sites = {"carry": (2, 4), "inspect": (0, 0)}

    print("c1's view (colour 1 is a wall to c1)")
    print("\n".join("  " + row for row in cgrid.view("c1").rows))
    assert cgrid.free((2, 4)), "the cell itself is open, only c1 cannot see it"
    assert not cgrid.view("c1").free((2, 4)), "c1 should not see the coloured cell"
    assert cgrid.view("g1").free((2, 4)), "g1 has the palette for it"
    assert not cgrid.sees("c1", (2, 4)) and cgrid.sees("g1", (2, 4))

    chosen = select(pool, task, 2)
    print(f"\nselection: {chosen.names} cost={chosen.cost} ok={chosen.ok}")
    assert chosen.ok and chosen.names == ["c1", "c2"], chosen
    assert pool.missing(chosen.team, task) == frozenset(), "the team covers the task"

    blocked, why = greedy_decompose(task, chosen.team, sites)
    print("\nsplit for the cheapest team")
    assert why is None, "the split itself is well formed"
    for name in sorted(blocked):
        print(f"  {name}: {blocked[name]}")
    assert blind_spots(cgrid, chosen.team, blocked) == ["c1"]

    result = run_colored(cgrid, pool, task, 2, sites)
    print("\n" + describe(result))
    assert [a.kind for a in result.attempts] == ["decomposition", "solved"], result.attempts
    assert "cannot see" in result.attempts[0].detail
    assert result.names == ["c2", "g1"], result.names
    assert result.stats["nodes"] >= 1
    for a in result.attempts:
        if a.paths:
            for path in a.paths.values():
                assert all(cgrid.free(c) for c in path), "a plan left the real grid"

    # Nowhere to switch to: every covering team is blind somewhere, so the
    # failure stays a decomposition one instead of turning into arbitration.
    d1 = Agent("d1", "carries crates", cost=0.9, start=(4, 0))
    stuck = run_colored(cgrid, AgentPool([c1, c2, d1]), task, 2, sites)
    print("\n" + describe(stuck))
    assert not stuck.ok and stuck.failure == "decomposition", stuck
    assert set(stuck.breakdown) == {"decomposition"}, stuck.breakdown
    assert len(stuck.attempts) == 2, "only two teams cover the task here"
    assert select(AgentPool([c1, c2, d1]), task, 2).ok, "selection still succeeds"

    again = run_colored(cgrid, pool, task, 2, sites)
    assert [a.kind for a in again.attempts] == [a.kind for a in result.attempts]

    # Generated instances: seeded, and nobody is blinded to the cell it starts on.
    first = colored_instances(3, seed=5, n_agents=6, n_tags=3, k=2)
    second = colored_instances(3, seed=5, n_agents=6, n_tags=3, k=2)
    assert first and len(first) == len(second)
    for x, y in zip(first, second):
        assert isinstance(x.grid, ColoredGrid), x
        assert x.grid.rows == y.grid.rows and x.grid.palettes == y.grid.palettes
        for a in x.pool:
            assert x.grid.sees(a.name, a.start), (x.name, a.name)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
