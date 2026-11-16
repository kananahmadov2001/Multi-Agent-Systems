"""Stage 1 of the pipeline: the agent pool and its capability advertisements.

Selection happens before decomposition, so this is the input to selection only:
Agent (advert, tags, cost), Task (required tags), AgentPool (coverage, size-k
teams). Adverts are read by a keyword table, not an LLM, so selection stays
free and deterministic; replacing that with a model means changing `advertise`
alone.

    python3 agent_pool.py       # self-check: prints the pool and the teams
"""
from itertools import combinations


# Capability tag -> words that advertise it. Matched case-insensitively as whole
# words. Extend this table (or pass your own) rather than parsing in two places.
DEFAULT_KEYWORDS = (
    ("move", ("move", "moves", "walk", "walks", "drive", "drives", "navigate", "roll", "rolls")),
    ("carry", ("carry", "carries", "lift", "lifts", "grab", "grabs", "haul", "hauls", "transport")),
    ("fly", ("fly", "flies", "drone", "drones", "aerial", "hover", "hovers")),
    ("push", ("push", "pushes", "shove", "shoves")),
    ("inspect", ("inspect", "inspects", "scan", "scans", "check", "checks", "survey", "surveys")),
)


def advertise(text, keywords=DEFAULT_KEYWORDS):
    """The capability tags a private description is understood to advertise."""
    words = {w.strip(".,;:!?()[]\"'").lower() for w in text.split()}
    return frozenset(tag for tag, names in keywords if words & set(names))


class Agent:
    """One candidate in the pool.

    `start` is optional here: the pool is the selection input, and a grid
    position only matters once a team exists and the task has been decomposed.
    """

    def __init__(self, name, advert, capabilities=None, cost=1.0, start=None):
        self.name = name
        self.advert = advert
        self.capabilities = frozenset(capabilities) if capabilities is not None else advertise(advert)
        self.cost = cost
        self.start = start

    def __repr__(self):
        return (f"Agent({self.name!r}, caps={sorted(self.capabilities)}, "
                f"cost={self.cost}, start={self.start})")

    def to_mapf_entry(self, goal):
        """`(name, (start, goal))`: the pair cbs_baseline.solve expects.

        Decomposition supplies the goal; the start has to be known by then, so
        this is where a missing one is reported rather than passed on as None.
        """
        if self.start is None:
            raise ValueError(f"agent {self.name!r} has no start cell")
        return self.name, (self.start, goal)


class Task:
    """What the user asked for, as the capability tags it needs."""

    def __init__(self, description, requires):
        self.description = description
        self.requires = frozenset(requires)
  
    @classmethod
    def from_text(cls, description, keywords=DEFAULT_KEYWORDS):
        """Read the requirement the way an advert is read, so the demo stays free."""
        return cls(description, advertise(description, keywords))

    def __repr__(self):
        return f"Task({self.description!r}, requires={sorted(self.requires)})"


class AgentPool:
    """The candidates, plus the two questions selection asks about them."""

    def __init__(self, agents):
        self.agents = list(agents)
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate agent name in pool: {sorted(names)}")

    def __len__(self):
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)

    def __repr__(self):
        return f"AgentPool({[a.name for a in self.agents]})"

    def covering(self, task):
        """Agents whose advert touches the task at all (a cheap prefilter)."""
        return [a for a in self.agents if a.capabilities & task.requires]

    @staticmethod
    def capabilities_of(team):
        out = set()
        for a in team:
            out |= a.capabilities
        return frozenset(out)

    def missing(self, team, task):
        """Required tags the team does not cover. Empty means the team covers it."""
        return task.requires - self.capabilities_of(team)

    def covers(self, team, task):
        return not self.missing(team, task)

    @staticmethod
    def team_cost(team):
        return sum(a.cost for a in team)

    def teams(self, task, k, limit=None):
        """Every size-k team that covers `task`, cheapest first."""
        found = []
        for team in combinations(self.agents, k):
            if self.covers(team, task):
                found.append((self.team_cost(team), tuple(sorted(a.name for a in team)), team))
        found.sort(key=lambda x: (x[0], x[1]))
        out = [team for _, _, team in found]
        return out[:limit] if limit else out


def main():
    pool = AgentPool([
        Agent("a1", "walks around the grid and carries small boxes", cost=1.0, start=(0, 0)),
        Agent("a2", "flies over walls", cost=2.0, start=(0, 2)),
        Agent("a3", "moves on the ground and inspects rooms", cost=1.5, start=(2, 0)),
        Agent("a4", "carries heavy crates between rooms", cost=1.2, start=(2, 2)),
    ])
    task = Task.from_text("carry the crate to the bay and inspect it")

    print("pool")
    for a in pool:
        print(f"  {a.name}: {sorted(a.capabilities)}  cost={a.cost}  advert={a.advert!r}")
    print(f"\ntask: {task}")
    print(f"agents that can help at all: {[a.name for a in pool.covering(task)]}")

    for k in (1, 2, 3):
        teams = pool.teams(task, k)
        print(f"\nteams of size {k}: {len(teams)}")
        for t in teams:
            print(f"  {[a.name for a in t]}  cost={pool.team_cost(t)}")

    # Checks: this is the contract select.py and decompose.py will rely on.
    assert task.requires == {"carry", "inspect"}, task.requires
    assert pool.teams(task, 1) == [], "no single agent has both carry and inspect"
    assert [[a.name for a in t] for t in pool.teams(task, 2)] == [["a1", "a3"], ["a3", "a4"]]
    assert [pool.team_cost(t) for t in pool.teams(task, 2)] == [2.5, 2.7]
    assert [[a.name for a in t] for t in pool.teams(task, 3)] == [["a1", "a3", "a4"], ["a1", "a2", "a3"], ["a2", "a3", "a4"]]
    assert pool.missing((pool.agents[0], pool.agents[1]), task) == {"inspect"}
    assert pool.covers((pool.agents[0], pool.agents[2]), task)
    assert pool.teams(task, 2, limit=1) == [pool.teams(task, 2)[0]]
    assert pool.agents[0].to_mapf_entry((4, 4)) == ("a1", ((0, 0), (4, 4)))
    print("\nall checks passed")


if __name__ == "__main__":
    main()
