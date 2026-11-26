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
from decompose import greedy_decompose
from ordering import describe_conflict, ordering_conflicts, solve_ordered

HORIZON = 20
MAX_NODES = 2000

# What an attempt can end as. selection and decomposition happen before any
# planning, so the published failure breakdown is selection vs arbitration.
KINDS = ("solved", "selection", "decomposition", "arbitration", "verification")


class Attempt:
    """One team tried, how it ended, and what it produced."""

    def __init__(self, names, kind, detail, goals=None, paths=None, stats=None,
                 assignment=None):
        assert kind in KINDS, kind
        self.names = list(names)
        self.kind = kind
        self.detail = detail
        self.goals = goals
        self.paths = paths
        self.stats = stats
        self.assignment = assignment

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
    def detail(self):
        """Why the last attempt ended, or None when solved.

        The kind alone cannot separate "no conflict-free plan exists" from "we
        ran out of node budget", and a failure count that mixes the two reads
        as a capability claim when half of it is a cap.
        """
        return None if self.ok else self.attempts[-1].detail

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
                 solver=solve_best_first, checker=verify, decomposer=greedy_decompose,
                 selector=select, precheck=None, ordering="off", max_ordering_repairs=8):
    """Select, decompose, plan, verify; on failure try the next covering team.

    `max_teams` bounds the switch edge because each extra team is another full
    CBS run. A team that misses a required tag is never planned for: that is a
    selection failure by construction, which is what the breakdown reports.
    `precheck(team, goals)` returns a reason string when the split is impossible
    before the solver runs (the coloured domain uses it for unseen waypoints).
    ordering="check" rejects cross-agent order violations; "repair" additionally
    runs bounded heuristic repair, on either low-level planner and any arbiter.
    """
    if max_teams < 1:
        raise ValueError("max_teams must be at least 1")
    if ordering not in ("off", "check", "repair"):
        raise ValueError(f"unknown ordering mode {ordering!r}")
    if max_ordering_repairs < 0:
        raise ValueError("max_ordering_repairs must be nonnegative")
    chosen = selector(pool, task, k, strategy)
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
        goals, why = decomposer(task, team, sites)
        if goals is None:
            attempts.append(Attempt(names, "decomposition", why))
            continue
        blocked = precheck(team, goals) if precheck else None
        if blocked:
            attempts.append(Attempt(names, "decomposition", blocked, goals=goals))
            continue
        parts = getattr(goals, "parts", None)
        if ordering != "off" and parts is None:
            attempts.append(Attempt(names, "decomposition",
                                    "ordering requires a part-labelled decomposition", goals=goals))
            continue
        if ordering == "off" or parts is None:
            assignment = {a.name: [t for t in task.order if sites[t] in goals[a.name]]
                          for a in team}
        else:
            assignment = {a.name: list(parts.get(a.name, [])) for a in team}
        members = {a.name: a for a in team}
        mapf = {n: (members[n].start, g) for n, g in goals.items()}
        if ordering == "repair":
            # Any planner that honours ("release", index, time) works here, which
            # since fd_planner gained release support means both of ours. A planner
            # that ignored one could not slip a bad plan through either: the repair
            # loop only returns a plan verify_order accepts, so dropped releases
            # surface as "could not repair", never as an unordered answer.
            paths, stats = solve_ordered(grid, mapf, task, assignment, solver=solver,
                                         planner=planner, checker=checker, horizon=horizon,
                                         max_nodes=max_nodes, max_repairs=max_ordering_repairs)
        else:
            paths, stats = solver(grid, mapf, horizon=horizon, max_nodes=max_nodes,
                                  planner=planner)
        if paths is None:
            why = stats.get("ordering_error")
            if not why:
                why = "hit the node cap" if stats["hit_limit"] else "no conflict-free plan"
            attempts.append(Attempt(names, "arbitration", why, goals=goals, stats=stats,
                                    assignment=assignment))
            continue
        ok, problems = checker(grid, mapf, paths)
        if not ok:
            attempts.append(Attempt(names, "verification", problems[0],
                                    goals=goals, paths=paths, stats=stats,
                                    assignment=assignment))
            continue
        if ordering != "off":
            stats = dict(stats)
            conflicts, order_problems = ordering_conflicts(task, paths, mapf, assignment)
            if ordering == "check":
                stats.update(ordering_conflicts=len(conflicts), ordering_repairs=0,
                             ordering_rounds=1)
            if order_problems or conflicts:
                why = (order_problems[0] if order_problems else
                       "ordering repair returned an invalid plan" if ordering == "repair" else
                       describe_conflict(conflicts[0]))
                attempts.append(Attempt(names, "verification", why,
                                        goals=goals, paths=paths, stats=stats,
                                        assignment=assignment))
                continue
        attempts.append(Attempt(names, "solved", "", goals=goals, paths=paths, stats=stats,
                                assignment=assignment))
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
        if a.assignment:
            lines.append(f"     tags {a.assignment}")
        if a.goals:
            lines.append(f"     goals {a.goals}")
        if a.stats:
            line = f"     nodes={a.stats['nodes']} replans={a.stats['replans']}"
            if a.paths:
                line += f" steps={sum(len(p) - 1 for p in a.paths.values())}"
            if "ordering_conflicts" in a.stats:
                line += (f" ordering_conflicts={a.stats['ordering_conflicts']}"
                         f" ordering_repairs={a.stats['ordering_repairs']}")
            lines.append(line)
    lines.append(f"  ok={result.ok} names={result.names} failure={result.failure}")
    return "\n".join(lines)


def main():
    open_grid = Grid([".....", ".....", ".....", ".....", "....."])
    pool = AgentPool([
        Agent("s1", "moves around and carries boxes", cost=1.0, start=(0, 0)),
        Agent("s2", "moves quickly", cost=0.4, start=(0, 4)),
        Agent("s3", "carries crates and inspects rooms", cost=1.5, start=(4, 0)),
        Agent("s4", "inspects rooms", cost=0.6, start=(4, 4)),
    ])
    task = Task.from_text("carry the crate to the bay and inspect it")
    solved = run_pipeline(open_grid, pool, task, 2, {"carry": (2, 2), "inspect": (2, 3)})
    assert solved.ok and solved.names == ["s1", "s4"], solved.names
    assert solved.failure is None and solved.breakdown == {"solved": 1}
    assert solved.stats["nodes"] >= 1

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
    assert [a.kind for a in switched.attempts] == ["arbitration", "solved"], switched.attempts
    assert switched.names == ["b2", "g1"], switched.names

    no_flyer = run_pipeline(open_grid, pool, Task.from_text("fly to the tower"), 2,
                            {"fly": (0, 0)})
    assert not no_flyer.ok and no_flyer.failure == "selection"
    assert no_flyer.breakdown == {"selection": 1}

    one_pair = AgentPool([
        Agent("c1", "carries crates and inspects rooms", cost=1.0, start=(0, 0)),
        Agent("c2", "moves around", cost=0.5, start=(0, 4)),
    ])
    both = run_pipeline(open_grid, one_pair, task2, 2,
                        {"carry": (2, 2), "inspect": (2, 3)})
    assert both.ok, both.attempts
    assert both.goals == {"c1": [(2, 2), (2, 3)], "c2": [(0, 4)]}, both.goals

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
    assert three.ok, three.attempts
    assert three.goals == {"g1": [(0, 2), (2, 2)], "g4": [(2, 4)]}, three.goals

    order_grid = Grid(["...", "...", "..."])
    ordered_pool = AgentPool([
        Agent("o1", "carries crates", start=(0, 0)),
        Agent("o2", "inspects rooms", start=(0, 2)),
    ])
    ordered_task = Task("carry then inspect", {"carry", "inspect"}, ("carry", "inspect"))
    order_sites = {"carry": (2, 2), "inspect": (0, 1)}
    plain = run_pipeline(order_grid, ordered_pool, ordered_task, 2, order_sites)
    assert plain.ok, "default mode must keep accepting the spatially valid plan"
    checked = run_pipeline(order_grid, ordered_pool, ordered_task, 2, order_sites,
                           ordering="check")
    assert not checked.ok and checked.failure == "verification", checked.attempts
    assert checked.attempts[0].stats["ordering_conflicts"] == 1
    fixed = run_pipeline(order_grid, ordered_pool, ordered_task, 2, order_sites,
                         ordering="repair")
    assert fixed.ok and fixed.stats["ordering_repairs"] == 1, fixed.attempts
    assert fixed.stats["ordering_rounds"] == 2
    # repair is no longer tied to one planner: a planner that honours releases
    # works whoever supplies it, so wrapping the default must change nothing
    def passthrough(grid, start, goal, constraints, horizon):
        return plan_single(grid, start, goal, constraints, horizon)

    wrapped = run_pipeline(order_grid, ordered_pool, ordered_task, 2, order_sites,
                           planner=passthrough, ordering="repair")
    assert wrapped.ok and wrapped.stats["ordering_repairs"] == 1, wrapped.attempts
    assert wrapped.stats["ordering_rounds"] == fixed.stats["ordering_rounds"]

    # a planner that quietly drops releases must not yield an unordered plan: the
    # repair loop can only return what verify_order accepts, so it reports failure
    def drops_releases(grid, start, goal, constraints, horizon):
        kept = frozenset(c for c in constraints if not (len(c) == 3 and c[0] == "release"))
        return plan_single(grid, start, goal, kept, horizon)

    dropped = run_pipeline(order_grid, ordered_pool, ordered_task, 2, order_sites,
                           planner=drops_releases, ordering="repair")
    assert not dropped.ok, "a planner that ignores releases must not appear to succeed"

    # a solver that returns nothing is an arbitration failure, not a crash
    none_solver = run_pipeline(order_grid, ordered_pool, ordered_task, 2, order_sites,
                               solver=lambda *a, **kw: (None, {}), ordering="repair")
    assert not none_solver.ok and none_solver.failure == "arbitration"

    print("\nall checks passed")


if __name__ == "__main__":
    main()