"""Capability-overlap instances: a grid, a pool and a task that selection must solve.

Agents differ in which tags they can serve and what they cost, and one may sit in
a sealed pocket, so which team selection picks decides whether the grid instance
can be planned at all. Standard library only.

    python3 pipeline_instances.py       # self-check: prints a few instances
"""
import random

from agent_pool import Agent, AgentPool, Task
from cbs_baseline import Grid

TAGS = ("move", "carry", "inspect", "fly")
PHRASES = {
    "carry": "carries crates",
    "inspect": "inspects rooms",
    "fly": "flies over walls",
    "move": "walks around the grid",
}
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class PipelineInstance:
    """One pipeline problem: grid, candidates, task, and the cell of each part."""

    def __init__(self, name, grid, pool, task, sites, sealed=(), scarce=False):
        self.name = name
        self.grid = grid
        self.pool = pool
        self.task = task
        self.sites = dict(sites)
        self.sealed = set(sealed)
        self.scarce = scarce

    def __repr__(self):
        return (f"PipelineInstance({self.name!r}, agents={len(self.pool)}, "
                f"order={list(self.task.order)}, sealed={sorted(self.sealed)}, "
                f"scarce={self.scarce})")

    def describe(self):
        lines = [f"{self.name}  sites={self.sites}  sealed={sorted(self.sealed)}  "
                 f"scarce={self.scarce}"]
        lines += ["  " + "".join(row) for row in self.grid.rows]
        for a in self.pool:
            lines.append(f"  {a.name}: {sorted(a.capabilities)} cost={a.cost} start={a.start}")
        return "\n".join(lines)


def cells(rows):
    """Every non-wall cell of a mutable grid."""
    return [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "."]


def seal(rows, cell):
    """Wall off the four neighbours, so an agent standing on `cell` can never leave."""
    r, c = cell
    for dr, dc in NEIGHBOURS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < len(rows) and 0 <= cc < len(rows[rr]):
            rows[rr][cc] = "#"


def reachable(grid, start, goal):
    """Can one agent walk from start to goal? Waiting allowed, walls not."""
    if start == goal:
        return True
    seen, stack = {start}, [start]
    while stack:
        for nxt in grid.neighbors(stack.pop()):
            if nxt == goal:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def capability_sets(rng, tags, n_agents):
    """Everyone serves one tag, some serve a second, so overlap is not avoidable."""
    caps = [{tag} for tag in tags]
    while len(caps) < n_agents:
        caps.append({rng.choice(tags)})
    for c in caps:
        if rng.random() < 0.35:
            c.add(rng.choice(tags))
    rng.shuffle(caps)
    return caps[:n_agents]


def feasible(inst, k):
    """A covering team without a sealed member can reach every site it serves.

    A scarce instance is the exception: its pool is meant to lack a tag, so the
    absence of a covering team is the point, not a rejection reason.
    """
    if not inst.pool.teams(inst.task, k):
        return inst.scarce
    for team in inst.pool.teams(inst.task, k):
        if any(a.name in inst.sealed for a in team):
            continue
        for tag in sorted(inst.task.requires):
            holders = [a for a in team if tag in a.capabilities]
            if not any(reachable(inst.grid, a.start, inst.sites[tag]) for a in holders):
                break
        else:
            return True
    return False


def capability_overlap(n, seed, size=5, walls=0.25, n_agents=4, n_tags=2, k=2,
                       trap=0.35, scarce=0.2, tries=None):
    """`n` feasible instances. Deterministic for a fixed seed.

    `trap` seals a cheap agent into a pocket, so the cheapest team cannot be
    planned and the switch edge is what saves the instance. `scarce` strips one
    tag from the whole pool, so the failure is selection rather than arbitration.
    """
    if n_agents < n_tags:
        raise ValueError("need at least one agent per tag")
    rng = random.Random(seed)
    out, budget = [], (tries if tries is not None else 200 * n)
    scarce_cap = int(round(scarce * n))       # scarce instances pass the filter trivially,
    while len(out) < n and budget > 0:        # so cap them instead of letting them crowd out
        budget -= 1
        rows = [["#" if rng.random() < walls else "." for _ in range(size)] for _ in range(size)]

        pocket = None
        if rng.random() < trap:
            inside = [c for c in cells(rows)
                      if all(0 <= c[0] + dr < size and 0 <= c[1] + dc < size for dr, dc in NEIGHBOURS)]
            if inside:
                pocket = rng.choice(inside)
                seal(rows, pocket)

        free = cells(rows)
        if len(free) < n_agents + n_tags:
            continue
        starts = rng.sample(free, n_agents)
        if pocket is not None:
            starts[0] = pocket                    # the sealed one, made cheap below
            if len(set(starts)) != n_agents:
                continue

        spots = [c for c in free if c not in set(starts)]
        if len(spots) < n_tags:
            continue

        tags = rng.sample(list(TAGS), n_tags)
        caps = capability_sets(rng, tags, n_agents)
        short = rng.random() < scarce and sum(i.scarce for i in out) < scarce_cap
        if short:                                  # nobody advertises one required tag
            absent = rng.choice(tags)
            caps = [c - {absent} for c in caps]
        # A generalist costs more than a specialist, which is what makes
        # "one expensive agent for both parts" and "two cheap specialists"
        # different choices rather than the same team by another route.
        costs = [round(rng.uniform(2.2, 3.0) if len(c) > 1 else rng.uniform(0.8, 1.8), 1)
                 for c in caps]
        if pocket is not None:
            costs[0] = 0.4                        # pulls the cheapest team onto it

        agents = []
        for i in range(n_agents):
            advert = " and ".join(PHRASES[t] for t in sorted(caps[i]))
            agents.append(Agent(f"a{i + 1}", advert, capabilities=caps[i],
                                cost=costs[i], start=starts[i]))

        order = list(tags)
        rng.shuffle(order)
        inst = PipelineInstance(
            f"co{len(out) + 1}",
            Grid(["".join(row) for row in rows]),
            AgentPool(agents),
            Task(" and ".join(PHRASES[t] for t in order), tags, order=order),
            dict(zip(tags, rng.sample(spots, n_tags))),
            sealed={"a1"} if pocket is not None else (),
            scarce=short,
        )
        if feasible(inst, k):
            out.append(inst)
    return out


def main():
    insts = capability_overlap(6, seed=5)
    for inst in insts:
        print(inst.describe())
        print(f"  task: {inst.task}")
        print()
    assert len(insts) == 6, len(insts)
    assert [i.name for i in insts] == [f"co{i + 1}" for i in range(6)]
    for inst in insts:
        assert set(inst.sites) == inst.task.requires, inst.name
        assert all(inst.grid.free(c) for c in inst.sites.values()), inst.name
        assert all(a.start and inst.grid.free(a.start) for a in inst.pool), inst.name
        assert inst.pool.teams(inst.task, 2) or inst.scarce, inst.name
        assert feasible(inst, 2), inst.name

    poor = capability_overlap(3, seed=11, scarce=1.0)
    assert len(poor) == 3 and all(i.scarce and not i.pool.teams(i.task, 2) for i in poor)

    again = capability_overlap(6, seed=5)
    assert [repr(i) for i in insts] == [repr(i) for i in again], "generator is not deterministic"
    assert capability_overlap(3, seed=5, n_agents=2, n_tags=2), "two agents should still give instances"
    print("all checks passed")


if __name__ == "__main__":
    main()
