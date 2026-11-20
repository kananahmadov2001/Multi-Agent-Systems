
"""Colored box: every agent plans on its own view of the grid.

A cell may carry a colour and each agent has a palette; a cell outside the palette is a wall to
that agent. A covering team can therefore still be unable to carry out its own split, which is a
decomposition failure, not a selection one. The pipeline is pipeline.run_pipeline: the world model
enters through two hooks, a per-agent planner (`colored_planner`) and a decomposer wrapper
(`sight_aware`) that rejects a split which sends an agent outside its palette.

    python3 colored_box.py       # self-check of the world model (no LLM)
"""
import random

from cbs_baseline import Grid, plan_single
from pipeline import run_pipeline
from pipeline_instances import PipelineInstance, capability_overlap, reachable

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
        return Grid(["".join(ch if ch == OPEN or ch in palette else WALL for ch in row)
                     for row in self.rows])


def colored_planner(cgrid, pool):
    """One low-level planner for the whole pool: each agent plans on its own view, found by its
    start cell (starts must be distinct, which the generated instances guarantee)."""
    starts = [a.start for a in pool]
    if len(set(starts)) != len(starts):
        raise ValueError("agents in one pool must start on different cells")
    views = {a.start: cgrid.view(a.name) for a in pool}

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


def sight_aware(cgrid):
    """run_pipeline `wrap=`: a split that sends an agent outside its palette becomes a
    decomposition failure, and the LLM is told so when it repicks."""
    def wrap(decomposer):
        def decompose(task, team, sites):
            goals, why = decomposer(task, team, sites)
            if goals is None:
                return None, why
            blind = blind_spots(cgrid, team, goals)
            if blind:
                return None, f"cannot see its site: {', '.join(sorted(blind))}"
            return goals, None
        return decompose
    return wrap


def colored_instances(n, seed, see=0.8, **kw):
    """The capability-overlap instances with the site cells painted a colour.

    A site on a colour an agent cannot see never enters that agent's view, so a team that covers
    the task can still be unable to carry out its own split. `see` is the chance an agent has a
    given colour; no agent is ever blinded to the cell it starts on.
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


def run_colored(cgrid, pool, task, k, sites, llm, **kw):
    """run_pipeline on a coloured world. **kw reaches run_pipeline (solver=, max_teams=, ...)."""
    kw.setdefault("planner", colored_planner(cgrid, pool))
    return run_pipeline(cgrid, pool, task, k, sites, llm, wrap=sight_aware(cgrid), **kw)


def main():
    from agent_pool import Agent, AgentPool
    # The right column is colour 1, so only an agent whose palette has 1 can stand on it.
    cgrid = ColoredGrid(["....1", "....1", "....1", ".....", "....."], {"c1": "", "c2": "", "g1": "1"})
    c1 = Agent("c1", "carries crates", cost=1.0, start=(0, 0))
    c2 = Agent("c2", "inspects rooms", cost=0.4, start=(0, 3))
    g1 = Agent("g1", "carries crates", cost=1.8, start=(4, 0))
    assert cgrid.free((2, 4)), "the cell itself is open, only c1 cannot see it"
    assert not cgrid.view("c1").free((2, 4)) and cgrid.view("g1").free((2, 4))
    assert not cgrid.sees("c1", (2, 4)) and cgrid.sees("g1", (2, 4))
    goals = {"c1": [(2, 4)], "c2": [(0, 0)]}
    assert blind_spots(cgrid, [c1, c2], goals) == ["c1"]
    assert blind_spots(cgrid, [g1, c2], {"g1": [(2, 4)], "c2": [(0, 0)]}) == []
    wrapped = sight_aware(cgrid)(lambda task, team, sites: (goals, None))
    assert wrapped(None, [c1, c2], None) == (None, "cannot see its site: c1")
    planner = colored_planner(cgrid, AgentPool([c1, c2, g1]))
    assert planner(cgrid, (0, 0), [(2, 4)], frozenset(), 20) is None, "c1 plans on its own view"
    assert planner(cgrid, (4, 0), [(2, 4)], frozenset(), 20) is not None

    first = colored_instances(3, seed=5, n_agents=6, n_tags=3, k=2)
    second = colored_instances(3, seed=5, n_agents=6, n_tags=3, k=2)
    assert first and len(first) == len(second)
    for x, y in zip(first, second):
        assert x.grid.rows == y.grid.rows and x.grid.palettes == y.grid.palettes
        for a in x.pool:
            assert x.grid.sees(a.name, a.start), (x.name, a.name)
    print("all checks passed")


if __name__ == "__main__":
    main()
