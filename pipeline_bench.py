"""Run the LLM pipeline over capability-overlap instances: one row per instance.

Every stage is the real code path: selection and decomposition go through the LLM
interfaces, planning and arbitration are unchanged. What changes is who answers the
prompts. By default a scripted oracle answers them from the instance's true
capabilities, which costs nothing and checks the wiring; --live sends the prompts to
the real model (needs ANTHROPIC_API_KEY, costs money). `--domain colored` swaps in the
coloured-box world, where each agent plans on its own view.

Deterministic for a fixed --seed. Standard library only, no API calls unless --live.

    python3 pipeline_bench.py                      # 20 instances, seed 7, scripted, $0
    python3 pipeline_bench.py --domain colored
    python3 pipeline_bench.py --live --out results/pipeline/pipeline_llm.csv
"""
import argparse
import csv
import json
import os

from colored_box import colored_instances, run_colored
from pipeline import run_pipeline, make_solver
from pipeline_instances import capability_overlap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "pipeline", "pipeline_llm.csv")
FIELDS = ("instance", "sealed", "scarce", "mode", "solved", "failure", "teams_tried",
          "team", "team_cost", "steps", "nodes", "llm_calls")
OUTCOMES = ("solved", "selection", "decomposition", "arbitration", "verification")


class Oracle:
    """Scripted answers to the selection and decomposition prompts.

    Selection is told by "Choose a team" in the prompt, decomposition by everything else.
    The teams are the covering ones in cost order, so a repick after a failure lands on
    the next candidate, which is what the search tree is supposed to do. The replies come
    from the instance's true capabilities: a run with this LLM checks the pipeline's
    wiring, and says nothing about how well a model chooses.
    """

    def __init__(self, inst, k):
        self.inst, self.k, self.i, self.team = inst, k, 0, None
        rank = inst.pool.teams(inst.task, k, 5)
        self.picks = [[a.name for a in t] for t in rank]
        if not self.picks:                       # nothing covers it; any k names will do
            names = [a.name for a in sorted(inst.pool, key=lambda a: (a.cost, a.name))]
            self.picks = [names[i:i + k] for i in range(len(names) - k + 1)]

    def __call__(self, prompt):
        if "Choose a team" in prompt:
            return self._pick()
        return self._split()

    def _pick(self):
        self.team = self.picks[min(self.i, len(self.picks) - 1)]
        self.i += 1
        return json.dumps({"team": self.team, "reason": "scripted oracle"})

    def _split(self):
        by_name = {a.name: a for a in self.inst.pool}
        routes, remaining = {}, list(self.inst.task.order)
        for name in self.team:
            mine = [p for p in remaining if p in by_name[name].capabilities]
            if mine:
                routes[name] = mine
                remaining = [p for p in remaining if p not in mine]
        if remaining:
            return json.dumps({"feasible": False, "missing": remaining,
                               "reason": "scripted oracle: nobody on the team covers it"})
        return json.dumps({"feasible": True, "routes": routes, "reason": "scripted oracle"})


def one(inst, k, max_teams, domain="capability", live=False):
    """One instance, as a CSV row."""
    run = run_colored if domain == "colored" else run_pipeline
    if live:
        from llm_arbitrator import AnthropicLLM
        llm, mode = AnthropicLLM(), "live"
    else:
        llm, mode = Oracle(inst, k), "scripted"
    result = run(inst.grid, inst.pool, inst.task, k, inst.sites, llm,
                 max_teams=max_teams, solver=make_solver(1))
    win = result.winner
    return {
        "instance": inst.name,
        "sealed": ";".join(sorted(inst.sealed)) or "-",
        "scarce": int(inst.scarce),
        "mode": mode,
        "solved": int(result.ok),
        "failure": "" if result.ok else result.failure,
        "teams_tried": len(result.attempts),
        "team": ";".join(win.names) if win else "",
        "team_cost": round(sum(a.cost for a in inst.pool if win and a.name in win.names), 1)
                     if win else "",
        "steps": sum(len(p) - 1 for p in win.paths.values()) if win else "",
        "nodes": win.stats["nodes"] if win else "",
        "llm_calls": sum(result.llm_calls.values()),
    }


def summary(rows):
    """Outcome counts, and how often a team switch was needed."""
    lines = [f"{'outcome':<14}{'count':>6}"]
    for o in OUTCOMES:
        n = sum(1 for r in rows if ("solved" if r["solved"] else r["failure"]) == o)
        lines.append(f"{o:<14}{n:>6}")
    switched = sum(1 for r in rows if r["solved"] and r["teams_tried"] > 1)
    lines.append(f"\nsolved {sum(r['solved'] for r in rows)}/{len(rows)}, "
                 f"needed a team switch {switched}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="instances to generate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=2, help="team size, exactly k agents")
    ap.add_argument("--max-teams", type=int, default=3, help="candidate teams tried per instance")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--tags", type=int, default=3,
                    help="more tags than k, so a team needs a generalist")
    ap.add_argument("--walls", type=float, default=0.25)
    ap.add_argument("--scarce", type=float, default=0.2,
                    help="chance that the pool lacks one required tag, so the failure is selection")
    ap.add_argument("--domain", choices=("capability", "colored"), default="capability",
                    help="colored gives each agent its own view of the grid")
    ap.add_argument("--live", action="store_true",
                    help="send the prompts to the real model. Without it a scripted oracle "
                         "answers them, which costs nothing")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.out is None:
        name = "pipeline_llm_colored.csv" if a.domain == "colored" else "pipeline_llm.csv"
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
        row = one(inst, a.k, a.max_teams, a.domain, a.live)
        rows.append(row)
        if not a.quiet:
            end = "solved" if row["solved"] else row["failure"]
            print(f"{row['instance']:<6} {end:<14} teams={row['teams_tried']} "
                  f"team={row['team'] or '-'} steps={row['steps'] or '-'}")

    for r in rows:                             # solved rows have a plan, failed ones do not
        if r["solved"]:
            assert r["team"] and r["steps"] and r["nodes"], r
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
    print(summary(rows))
    print(f"\n{len(insts)} instances -> {a.out}")


if __name__ == "__main__":
    main()
