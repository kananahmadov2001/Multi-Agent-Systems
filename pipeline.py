
"""The pipeline, with the LLM as the central agent.

  1 select      the LLM picks exactly k agents from their natural-language descriptions
  2 decompose   the LLM splits the task over that team (or says the team cannot do it)
  3 plan        each agent's route is planned (grid planner, or Fast Downward via planner=)
  4 arbitrate   collisions are repaired by Approach 1 (best-first CBS), 2 (LLM, no tree) or
                3 (LLM-guided CBS); see make_solver
  5 verify      an independent replay accepts or rejects the joint plan

If a stage fails, the team is dropped and the LLM is shown which teams failed and why before it
repicks, up to `max_teams` teams. One `llm` callable (llm_arbitrator.AnthropicLLM) serves stages
1-2; the arbiter gets its own llm through make_solver, so the two are costed separately.

Each failed attempt is labelled by cause, using the true capabilities as ground truth:
  selection      the team does not actually cover the task (a wrong pick)
  decomposition  the team covers it but no valid split was produced
  arbitration    the split was fine but no conflict-free plan exists
  verification   the independent checker rejected the plan
The LLM does the detecting (it is the one that says "this team cannot do it"); the ground truth
only labels what it noticed. Attempt.covers records whether the team truly covered the task, so a
team the LLM wrongly rejected is visible in the results.

    python3 pipeline.py       # live demo: one real LLM run (needs ANTHROPIC_API_KEY, costs cents)
"""
from functools import partial

from llm_select import propose_team
from decompose import llm_decompose
from cbs_baseline import plan_single, solve_best_first
from llm_arbitrator import solve_llm
from hybrid import solve_hybrid
from verify import verify

HORIZON = 20
MAX_NODES = 2000

KINDS = ("solved", "selection", "decomposition", "arbitration", "verification")


class Attempt:
    """One team tried, how it ended, and what it produced."""

    def __init__(self, names, kind, detail, goals=None, paths=None, stats=None,
                 assignment=None, covers=None):
        assert kind in KINDS, kind
        self.names = list(names)
        self.kind = kind
        self.detail = detail
        self.goals = goals
        self.paths = paths
        self.stats = stats
        self.assignment = assignment
        self.covers = covers          # did the team truly cover the task (ground truth)?

    @property
    def ok(self):
        return self.kind == "solved"

    def __repr__(self):
        tail = f" ({self.detail})" if self.detail else ""
        return f"Attempt({self.names}, {self.kind}{tail})"


class PipelineResult:
    """Every attempt made, plus the winning one if there is one."""

    def __init__(self, attempts, select_stats=None, decompose_stats=None):
        self.attempts = list(attempts)
        self.select_stats = select_stats or {}
        self.decompose_stats = decompose_stats or {}
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

    @property
    def llm_calls(self):
        """LLM calls by stage, summed over every team tried (failed teams cost calls too)."""
        arbitration = sum(a.stats.get("llm_calls", 0) for a in self.attempts if a.stats)
        return {"selection": self.select_stats.get("llm_calls", 0),
                "decomposition": self.decompose_stats.get("llm_calls", 0),
                "arbitration": arbitration}

    def __repr__(self):
        return f"PipelineResult(ok={self.ok}, names={self.names})"


def make_solver(approach, llm=None, a2_calls=20, a3_calls=50):
    """The arbiter for stage 4, in the shape run_pipeline calls it.

    1 best-first CBS (no LLM)   2 LLM picks the agent to constrain, no tree   3 LLM-guided CBS
    """
    if approach == 1:
        return solve_best_first
    if approach not in (2, 3):
        raise ValueError("approach must be 1, 2 or 3")
    if llm is None:
        raise ValueError(f"approach {approach} needs an llm callable")
    if approach == 2:
        return lambda grid, agents, horizon, max_nodes, planner: solve_llm(
            grid, agents, llm, horizon=horizon, max_calls=a2_calls, planner=planner)
    return lambda grid, agents, horizon, max_nodes, planner: solve_hybrid(
        grid, agents, llm, horizon=horizon, max_nodes=max_nodes, max_calls=a3_calls, planner=planner)


def run_pipeline(grid, pool, task, k, sites, llm, horizon=HORIZON, max_nodes=MAX_NODES,
                 max_teams=3, planner=plan_single, solver=solve_best_first, checker=verify,
                 wrap=None, select_calls=3, decompose_calls=3):
    """LLM selects, LLM decomposes, plan, arbitrate, verify; on failure the LLM repicks.

    llm        callable prompt -> text for stages 1-2 (llm_arbitrator.AnthropicLLM).
    solver     the arbiter: make_solver(1 | 2 | 3, ...).
    wrap       optional function decomposer -> decomposer, e.g. colored_box.sight_aware(cgrid).
    max_teams  bounds the repick edge, since each extra team is another full CBS run.
    """
    if max_teams < 1:
        raise ValueError("max_teams must be at least 1")
    select_stats, decompose_stats = {}, {}
    decomposer = partial(llm_decompose, llm=llm, grid=grid, max_calls=decompose_calls,
                         stats=decompose_stats)
    if wrap is not None:
        decomposer = wrap(decomposer)
    attempts = []
    while len(attempts) < max_teams:
        team, why = propose_team(pool, task, k, llm, attempts, select_stats, max_calls=select_calls)
        if team is None:                       # invalid replies, or every team already tried
            attempts.append(Attempt([], "selection", why, covers=False))
            break
        names = sorted(a.name for a in team)
        covers = not pool.missing(team, task)
        goals, why = decomposer(task, team, sites)
        if goals is None:
            attempts.append(Attempt(names, "decomposition" if covers else "selection", why,
                                    covers=covers))
            continue
        assignment = {a.name: [t for t in task.order if sites[t] in goals[a.name]] for a in team}
        members = {a.name: a for a in team}
        mapf = {n: (members[n].start, g) for n, g in goals.items()}
        paths, stats = solver(grid, mapf, horizon=horizon, max_nodes=max_nodes, planner=planner)
        if paths is None:
            why = "hit the node cap" if stats.get("hit_limit") else "no conflict-free plan"
            attempts.append(Attempt(names, "arbitration", why, goals=goals, stats=stats,
                                    assignment=assignment, covers=covers))
            continue
        ok, problems = checker(grid, mapf, paths)
        if not ok:
            attempts.append(Attempt(names, "verification", problems[0], goals=goals, paths=paths,
                                    stats=stats, assignment=assignment, covers=covers))
            continue
        attempts.append(Attempt(names, "solved", "", goals=goals, paths=paths, stats=stats,
                                assignment=assignment, covers=covers))
        break
    return PipelineResult(attempts, select_stats, decompose_stats)


def describe(result):
    """One block per attempt, then the LLM calls by stage."""
    lines = [f"{len(result.attempts)} attempt(s)"]
    for i, a in enumerate(result.attempts, 1):
        line = f"  {i}. {a.names} -> {a.kind}"
        if a.detail:
            line += f": {a.detail}"
        if a.covers is not None and a.kind != "solved":
            line += f"  [team truly covers task: {a.covers}]"
        lines.append(line)
        if a.assignment:
            lines.append(f"     parts {a.assignment}")
        if a.goals:
            lines.append(f"     goals {a.goals}")
        if a.stats:
            line = f"     nodes={a.stats.get('nodes', '-')} replans={a.stats['replans']}"
            if a.paths:
                line += f" steps={sum(len(p) - 1 for p in a.paths.values())}"
            lines.append(line)
    lines.append(f"  llm calls {result.llm_calls}")
    lines.append(f"  ok={result.ok} names={result.names} failure={result.failure}")
    return "\n".join(lines)


def main():
    """Live demo with the real model: it picks, splits, and (approach 3) arbitrates one instance."""
    from agent_pool import Agent, AgentPool, Task
    from cbs_baseline import Grid
    from llm_arbitrator import AnthropicLLM

    grid = Grid([".....", ".....", ".....", ".....", "....."])
    pool = AgentPool([
        Agent("s1", "moves around and carries boxes", cost=1.0, start=(0, 0)),
        Agent("s2", "moves quickly", cost=0.4, start=(0, 4)),
        Agent("s3", "carries crates and inspects rooms", cost=1.5, start=(4, 0)),
        Agent("s4", "inspects rooms", cost=0.6, start=(4, 4)),
    ])
    task = Task.from_text("carry the crate to the bay and inspect it")
    llm = AnthropicLLM()
    result = run_pipeline(grid, pool, task, 2, {"carry": (2, 2), "inspect": (2, 3)}, llm,
                          solver=make_solver(3, AnthropicLLM()))
    print(describe(result))
    print(f"API usage: {AnthropicLLM.total_calls} calls, {AnthropicLLM.tokens_in} in / "
          f"{AnthropicLLM.tokens_out} out tokens (model {llm.model})")


if __name__ == "__main__":
    main()
