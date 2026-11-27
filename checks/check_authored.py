"""How many authored problems are the problem the vocabulary determines?

    python3 -m checks.check_authored
    python3 -m checks.check_authored --detail

Free and offline: it re-reads a committed transcript and compiles the same cases
locally. No API calls, no Fast Downward.

`results/authored/single_summary.txt` scores a reply "correct" when Fast Downward
solves it and the plan costs what the reference plan costs -- 14 of 20 on the
committed run. That is a question about the plan, not about the problem, and the
two come apart: a problem can invent edges the grid does not have, or omit ones
it does, and still happen to admit an optimal plan for the one start and goal it
was asked about. So this asks the stricter question instead, by rebuilding each
case and diffing the authored problem against `make_problem`'s output.

The cases are reproducible because `authored_pddl.build_case` draws from
`random.Random(--seed)`, so the same seed, size and horizon give the same 20
cases the committed run used; the script checks that the regenerated starts and
goals match the transcript's own headers before trusting the comparison.

The gap this measures is the argument for compiling stage 3 rather than
authoring it. A wrong problem that yields a right plan is the worst case for a
pipeline, because nothing downstream has any reason to object: `verify.py`
replays the plan on the real grid and the plan is legal there.
"""
import argparse
import os
import random
import re
import sys

from mas.planning.authored_pddl import build_case
from mas.planning.fd_planner import make_problem

DEFAULT = os.path.join("results", "authored", "single.txt")
HEAD = re.compile(r"(\d+)\s+start \((\d+), (\d+)\) route \(\((\d+), (\d+)\),\)")
CELL = r"c\d+_\d+_\d+"


def parse(path):
    """{case number: (start, goal, problem text)} from a single-goal transcript."""
    out = {}
    for block in re.split(r"={40,}\ncase ", open(path).read())[1:]:
        m = HEAD.match(block)
        if not m:
            continue
        body = re.search(r"\(define \(problem.*", block, re.S)
        out[int(m.group(1))] = ((int(m.group(2)), int(m.group(3))),
                                (int(m.group(4)), int(m.group(5))),
                                body.group(0) if body else "")
    return out


def shape(text):
    """The parts of a problem that decide what it means, order thrown away."""
    head = text.split("(:init")[0]
    return (frozenset(re.findall(rf"\b{CELL}\b", head)),
            frozenset(re.findall(rf"\(edge ({CELL}) ({CELL})\)", text)),
            frozenset(re.findall(rf"\(at ({CELL})\)", text)),
            frozenset(re.findall(rf"\(goal-node ({CELL})\)", text)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transcript", nargs="?", default=DEFAULT)
    ap.add_argument("--seed", type=int, default=5, help="the run's --seed")
    ap.add_argument("--size", type=int, default=3, help="the run's --size")
    ap.add_argument("--horizon", type=int, default=5, help="the run's --horizon")
    ap.add_argument("--detail", action="store_true",
                    help="one line per case: object and edge counts, authored against compiled")
    a = ap.parse_args()
    if not os.path.exists(a.transcript):
        raise SystemExit(f"no transcript at {a.transcript}")

    authored = parse(a.transcript)
    if not authored:
        raise SystemExit(f"{a.transcript} holds no single-goal cases. --routes "
                         f"transcripts are not comparable this way: a route "
                         f"problem has a layer per site served.")

    rng = random.Random(a.seed)
    cases = []
    while len(cases) < len(authored):
        c = build_case(rng, a.size, a.horizon, False)
        if c:
            cases.append(c)

    drift = [i for i in authored
             if (cases[i - 1][1], cases[i - 1][2][0]) != (authored[i][0], authored[i][1])]
    if drift:
        raise SystemExit(
            f"regenerated cases do not match the transcript on {len(drift)} of "
            f"{len(authored)} (first: case {drift[0]}). Pass the --seed, --size "
            f"and --horizon the run used; the comparison is meaningless otherwise.")

    exact = []
    rows = []
    for i in sorted(authored):
        grid, start, route, horizon = cases[i - 1]
        ao, ae, aat, ag = shape(authored[i][2])
        co, ce, cat, cg = shape(make_problem(grid, start, route[0], frozenset(), horizon))
        same = (ao, ae, aat, ag) == (co, ce, cat, cg)
        exact.append(same)
        rows.append((i, len(ao), len(co), len(ae), len(ce), same,
                     "exact" if same else
                     ("more edges than the grid has" if len(ae) > len(ce) else
                      "fewer edges than the grid has" if len(ae) < len(ce) else
                      "same edge count, different edges")))

    if a.detail:
        print(f"{'case':>4} {'objects':>13} {'edges':>13}   verdict")
        for i, ao, co, ae, ce, _, why in rows:
            print(f"{i:>4} {ao:6d}/{co:<6d} {ae:6d}/{ce:<6d}   {why}")
        print()

    n = len(exact)
    print(f"{a.transcript}   ({a.size}x{a.size}, horizon {a.horizon}, seed {a.seed})\n")
    print(f"cases compared                                  {n}")
    print(f"authored problem identical to the compiled one   {sum(exact)}")
    print(f"authored problem differs                         {n - sum(exact)}")
    print(f"\nFor what the differing problems still manage, see "
          f"{os.path.join('results', 'authored', 'single_summary.txt')}:")
    print("a problem can be wrong about the grid and still admit an optimal plan")
    print("for the one start and goal it was handed.")


if __name__ == "__main__":
    main()