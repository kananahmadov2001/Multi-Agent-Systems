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
import os

from colored_box import colored_instances, run_colored
from pipeline import run_pipeline
from pipeline_instances import capability_overlap
from selection import POLICIES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "pipeline", "pipeline_results.csv")
FIELDS = ("instance", "sealed", "scarce", "strategy", "solved", "failure", "teams_tried",
          "team", "team_cost", "steps", "nodes")
OUTCOMES = ("solved", "selection", "decomposition", "arbitration", "verification")


def one(inst, k, strategy, max_teams, domain="capability"):
    """One instance under one strategy, as a CSV row."""
    run = run_colored if domain == "colored" else run_pipeline
    result = run(inst.grid, inst.pool, inst.task, k, inst.sites,
                 strategy=strategy, max_teams=max_teams)
    win = result.winner
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
        "nodes": win.stats["nodes"] if win else "",
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown strategy {unknown}; pick from {sorted(POLICIES)}")

    if a.out is None:
        name = "pipeline_colored.csv" if a.domain == "colored" else "pipeline_results.csv"
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
            row = one(inst, a.k, strategy, a.max_teams, a.domain)
            rows.append(row)
            if not a.quiet:
                end = "solved" if row["solved"] else row["failure"]
                print(f"{row['instance']:<6} {strategy:<9} {end:<14} teams={row['teams_tried']} "
                      f"team={row['team'] or '-'} steps={row['steps'] or '-'}")

    # A row must be internally consistent: solved rows have a plan, failed ones do not.
    for r in rows:
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
    print(summary(rows, strategies))
    print(f"\n{len(insts)} instances, {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
