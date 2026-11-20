"""Run the pipeline over capability-overlap instances: one row per strategy.

Same instances, three selection strategies, so the rows answer the question the
midterm asks for: when the pipeline fails, was it selection or arbitration.
`--domain colored` swaps in the coloured-box world, where each agent plans on its
own view and a covering team can still be unable to carry out its own split.
Deterministic for a fixed --seed. Standard library only, no API calls.

    python3 pipeline_bench.py                 # 20 instances, seed 7, all three strategies
    python3 pipeline_bench.py --strategies cheapest --out results/pipeline/x.csv
    python3 pipeline_bench.py --domain colored
"""
import argparse
import csv
import json
import os
import re

from colored_box import colored_instances, run_colored
from decompose import llm_decompose
from hybrid import solve_hybrid
from llm_arbitrator import AnthropicLLM, FakeLLM, solve_llm
from llm_select import propose_team
from pipeline import run_pipeline
from pipeline_instances import capability_overlap
from selection import POLICIES, Selection

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "pipeline", "pipeline_results.csv")
FIELDS = ("instance", "sealed", "scarce", "strategy", "solved", "failure", "teams_tried",
          "team", "team_cost", "steps", "nodes", "fd_calls", "llm_calls", "arbiter")
OUTCOMES = ("solved", "selection", "decomposition", "arbitration", "verification")


class Oracle:
    """Scripted replies for --selector llm and --decomposer llm.

    Selection is told by "Choose a team", decomposition by every other prompt. The
    answers come from the instance's true capabilities, so a run with this LLM checks
    the wiring and says nothing about how well a model chooses.
    """

    def __init__(self, inst, k):
        self.inst, self.k, self.i, self.team = inst, k, 0, []
        self.picks = [[a.name for a in t] for t in inst.pool.teams(inst.task, k, 20)
                      if not inst.pool.missing(t, inst.task)]

    def __call__(self, prompt):
        if "Choose a team" in prompt:
            self.team = self.picks[self.i] if self.i < len(self.picks) else self.team
            self.i += 1
            return json.dumps({"team": self.team, "reason": "scripted oracle"})
        by_name = {a.name: a for a in self.inst.pool}
        # The pipeline may fall back to another covering team, so answer for the team
        # the prompt names rather than for the one this object picked last.
        here = [n for n in re.findall(r"^  (\w+): starts at", prompt, re.M) if n in by_name]
        routes, left = {}, list(self.inst.task.order)
        for name in here or self.team:
            mine = [p for p in left if p in by_name[name].capabilities]
            if mine:
                routes[name] = mine
                left = [p for p in left if p not in mine]
        if left:
            return json.dumps({"feasible": False, "missing": left, "reason": "scripted oracle"})
        return json.dumps({"feasible": True, "routes": routes, "reason": "scripted oracle"})


def llm_selector(llm, stats):
    """Adapt propose_team to run_pipeline's selector hook.

    The model picks the first team; the remaining covering teams stay as fallbacks, so
    the switch edge is unchanged. A failed pick becomes an empty team, which
    run_pipeline records as a selection failure.
    """

    def pick(pool, task, k, strategy):
        team, _ = propose_team(pool, task, k, llm, stats=stats)
        names = [a.name for a in team] if team else []
        chosen = [a for a in pool if a.name in names]
        alts = [[a for a in t] for t in pool.teams(task, k, 5)
                if sorted(a.name for a in t) != sorted(names)]
        return Selection("llm", chosen, round(sum(a.cost for a in chosen), 1),
                         pool.missing(chosen, task), alts)

    return pick


def llm_decomposer(llm, grid, stats):
    """Bind the model and the grid to llm_decompose(task, team, sites)."""
    return lambda task, team, sites: llm_decompose(task, team, sites, llm,
                                                   grid=grid, stats=stats)


def make_solver(arbiter, llm, calls):
    """The pipeline's solver for one of the three arbitration approaches.

    run_pipeline calls solver(grid, agents, horizon=, max_nodes=, planner=), so
    the LLM is bound here rather than passed down. Approach 2 takes no
    max_nodes, having no tree to bound, so that keyword is dropped for it.
    """
    if arbiter == "cbs":
        return None                                  # run_pipeline's own default
    if arbiter == "hybrid":
        return lambda g, ag, **kw: solve_hybrid(g, ag, llm, max_calls=calls, **kw)
    return lambda g, ag, **kw: solve_llm(
        g, ag, llm, max_calls=calls,
        **{k: v for k, v in kw.items() if k != "max_nodes"})


def one(inst, k, strategy, max_teams, domain="capability", fd=None, search=None,
        arbiter="cbs", make_llm=None, calls=50,
        selector="tag", decomposer="greedy", live=False):
    """One instance under one strategy, as a CSV row.

    A fresh Fast Downward planner per row, never one shared across rows:
    make_fd_planner caches on (start, goal, constraints, horizon) and not on the
    grid, and these instances each have their own walls, so a shared planner
    would answer one instance with another's path. Sharing it also hides the
    real cost, which is the mistake bench_fd made across approaches.
    """
    run = run_colored if domain == "colored" else run_pipeline
    extra, planner, stats = {}, None, {}
    if selector == "llm" or decomposer == "llm":
        llm = AnthropicLLM() if live else Oracle(inst, k)
        if selector == "llm":
            extra["selector"] = llm_selector(llm, stats)
        if decomposer == "llm":
            extra["decomposer"] = llm_decomposer(llm, inst.grid, stats)
    if fd:
        from fd_planner import make_fd_planner
        planner = make_fd_planner(fd, search) if search else make_fd_planner(fd)
        extra["planner"] = planner
    if arbiter != "cbs":
        extra["solver"] = make_solver(arbiter, make_llm(), calls)
    result = run(inst.grid, inst.pool, inst.task, k, inst.sites,
                 strategy=strategy, max_teams=max_teams, **extra)
    win = result.winner
    # Every team tried costs calls, not just the one that worked.
    spent = (sum((a.stats or {}).get("llm_calls", 0) for a in result.attempts)
             + stats.get("llm_calls", 0))
    return {
        "instance": inst.name,
        "sealed": ";".join(sorted(inst.sealed)) or "-",
        "scarce": int(inst.scarce),
        "strategy": strategy,
        "solved": int(result.ok),
        "failure": "" if result.ok else result.failure,
        "teams_tried": len(result.attempts),
        "team": ";".join(win.names) if win else "",
        "team_cost": round(sum(a.cost for a in inst.pool if win and a.name in win.names), 1) if win else "",
        "steps": sum(len(p) - 1 for p in win.paths.values()) if win else "",
        # Approach 2 has no tree, so it reports no node count.
        "nodes": win.stats.get("nodes", "") if win else "",
        "fd_calls": planner.calls if planner else "",
        "llm_calls": (spent if (arbiter != "cbs" or selector == "llm" or decomposer == "llm")
                      else ""),
        "arbiter": arbiter,
    }


def summary(rows, strategies):
    """Selection strategy x outcome counts, plus how often the switch edge fired."""
    lines = [f"{'strategy':<11}" + "".join(f"{o:<14}" for o in OUTCOMES)]
    for s in strategies:
        mine = [r for r in rows if r["strategy"] == s]
        cells = [sum(1 for r in mine
                     if ("solved" if r["solved"] else r["failure"]) == o) for o in OUTCOMES]
        lines.append(f"{s:<11}" + "".join(f"{n:<14}" for n in cells))
    lines.append("")
    for s in strategies:
        mine = [r for r in rows if r["strategy"] == s]
        switched = sum(1 for r in mine if r["solved"] and r["teams_tried"] > 1)
        lines.append(f"  {s}: solved {sum(r['solved'] for r in mine)}/{len(mine)}, "
                     f"needed a team switch {switched}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="instances to generate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=2, help="team size, exactly k agents")
    ap.add_argument("--max-teams", type=int, default=3, help="candidate teams tried per instance")
    ap.add_argument("--strategies", default="cheapest,greedy,coverage")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--tags", type=int, default=3,
                    help="more tags than k, so a team needs a generalist and the policies differ")
    ap.add_argument("--walls", type=float, default=0.25)
    ap.add_argument("--scarce", type=float, default=0.2,
                    help="chance that the pool lacks one required tag, so the failure is selection")
    ap.add_argument("--domain", choices=("capability", "colored"), default="capability",
                    help="colored gives each agent its own view of the grid")
    ap.add_argument("--fd", default=None,
                    help="plan with Fast Downward instead of the grid planner: a directory "
                         "containing fast-downward.py. No API calls either way")
    ap.add_argument("--search", default=None,
                    help="Fast Downward search configuration, default astar(lmcut()). "
                         "A satisficing configuration costs optimality, which is the point "
                         "of passing one")
    ap.add_argument("--arbiter", choices=("cbs", "hybrid", "llm"), default="cbs",
                    help="who resolves conflicts: cbs is Approach 1 and free, hybrid is "
                         "Approach 3 (LLM-guided CBS), llm is Approach 2 (no tree)")
    ap.add_argument("--live", action="store_true",
                    help="use the real model. Without it a non-cbs arbiter runs on FakeLLM, "
                         "which costs nothing and checks the wiring first")
    ap.add_argument("--calls", type=int, default=50,
                    help="per-instance call cap for the LLM arbiter")
    ap.add_argument("--selector", choices=("tag", "llm"), default="tag",
                    help="tag: the free capability policies; llm: the model reads the adverts "
                         "and picks the team")
    ap.add_argument("--decomposer", choices=("greedy", "llm"), default="greedy",
                    help="greedy: the free reference split; llm: the model, as the central "
                         "agent, splits the task over the selected team")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    if a.selector == "llm" or a.decomposer == "llm":
        if a.domain == "colored":
            raise SystemExit("--selector/--decomposer llm need the hooks run_colored does "
                             "not have yet")
        strategies = ["llm"]
    unknown = [s for s in strategies if s not in POLICIES and s != "llm"]
    if unknown:
        raise SystemExit(f"unknown strategy {unknown}; pick from {sorted(POLICIES)}")
    if a.fd and a.domain == "colored":
        raise SystemExit("--fd needs a planner hook that run_colored does not have yet; "
                         "it rebuilds the pipeline loop instead of calling run_pipeline")
    if a.arbiter != "cbs" and a.domain == "colored":
        raise SystemExit("--arbiter needs the same hook run_colored does not have yet")
    if (a.live and a.arbiter == "cbs" and a.selector == "tag"
            and a.decomposer == "greedy"):
        raise SystemExit("--live has nothing to do here: nothing in this run calls a model; "
                         "pass --arbiter hybrid/llm, or --selector llm / --decomposer llm")

    model = os.environ.get("ARBITER_MODEL", "claude-sonnet-5-5") if a.live else "fake"
    make_llm = ((lambda: AnthropicLLM()) if a.live
                else (lambda: FakeLLM(lambda options, prompt: options[0])))
    if a.arbiter != "cbs":
        print(f"arbiter {a.arbiter}, model {model}, cap {a.calls} calls per instance"
              + ("" if a.live else "  (no API calls)"))
    if a.selector == "llm" or a.decomposer == "llm":
        print(f"llm selector {a.selector}, decomposer {a.decomposer}, model {model}"
              + ("" if a.live else "  (scripted oracle, no API calls)"))

    if a.out is None:
        parts = ["pipeline"]
        if a.domain == "colored":
            parts.append("colored")
        if a.fd:
            parts.append("fd")
        if a.arbiter != "cbs":
            parts.append(a.arbiter if a.live else f"{a.arbiter}_fake")
        if a.selector == "llm" or a.decomposer == "llm":
            parts.append("llm")
        name = ("_".join(parts) if len(parts) > 1 else "pipeline_results") + ".csv"
        a.out = os.path.join(HERE, "results", "pipeline", name)
    build = colored_instances if a.domain == "colored" else capability_overlap
    insts = build(a.n, a.seed, size=a.size, walls=a.walls,
                  n_agents=a.agents, n_tags=a.tags, k=a.k, scarce=a.scarce)
    if not insts:
        raise SystemExit("no feasible instance was generated; lower --walls or raise --agents")
    if len(insts) < a.n:
        print(f"note: only {len(insts)} of {a.n} requested instances were feasible")

    rows = []
    for inst in insts:
        for strategy in strategies:
            row = one(inst, a.k, strategy, a.max_teams, a.domain, fd=a.fd, search=a.search,
                      arbiter=a.arbiter, make_llm=make_llm, calls=a.calls,
                      selector=a.selector, decomposer=a.decomposer, live=a.live)
            rows.append(row)
            if not a.quiet:
                end = "solved" if row["solved"] else row["failure"]
                print(f"{row['instance']:<6} {strategy:<9} {end:<14} teams={row['teams_tried']} "
                      f"team={row['team'] or '-'} steps={row['steps'] or '-'}")

    # A row must be internally consistent: solved rows have a plan, failed ones do not.
    for r in rows:
        if r["solved"]:
            assert r["team"] and r["steps"] != "", r
            # Not truthiness: Approach 3 stops on generating a conflict-free plan, so
            # plans that never conflicted expand 0 nodes. Approach 2 keeps no tree at all.
            assert r["nodes"] != "" or r["arbiter"] == "llm", r
            assert r["teams_tried"] >= 1, r
        else:
            assert r["failure"] in OUTCOMES[1:], r
            assert r["team"] == "" and r["steps"] == "", r

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(summary(rows, strategies))
    if a.arbiter != "cbs":
        spent = sum(int(r["llm_calls"] or 0) for r in rows)
        print(f"\n  LLM calls: {spent} across {len(rows)} rows"
              f"  (cap {a.calls} per instance, arbiter {a.arbiter}, model {model})")
        if a.live:
            # Haiku 4.5 list price; wrong for another model, so the model is named above.
            pin, pout = 1.00, 5.00
            cost = (AnthropicLLM.tokens_in * pin + AnthropicLLM.tokens_out * pout) / 1e6
            print(f"  API usage: {AnthropicLLM.total_calls} calls, "
                  f"{AnthropicLLM.tokens_in} in / {AnthropicLLM.tokens_out} out tokens, "
                  f"about ${cost:.2f}")
    print(f"\n{len(insts)} instances, {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()