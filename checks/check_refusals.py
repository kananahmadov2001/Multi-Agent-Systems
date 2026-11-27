"""Were the decomposer's refusals right?

    python3 -m checks.check_refusals
    python3 -m checks.check_refusals results/pipeline/pipeline_composite_live.jsonl

The live all-stages run loses most of its instances at stage 2: the model is
handed a team and a task and answers {"feasible": false}. That is only a fair
answer if the team really cannot cover the task, and "cover" is a question the
harness can settle without a model -- `advertise()` is the same function stage 1
used to pick the team in the first place.

So this re-reads the committed transcript, pulls the team descriptions and the
task parts straight out of each refused prompt, and checks coverage itself. The
prompt is the whole input the model had, so nothing else is needed to judge it.

It also sorts the stated reasons into the faults they show. That part is keyword
matching over one free-text sentence, so read the printed counts as a rough
split and the sentences under --reasons as the evidence. The coverage number
above them is exact.

Why it matters: a refusal of a covering team is not a hard instance, it is a
wrong answer, and the four faults are different bugs. A model inventing a
reachability constraint is being asked the wrong question -- stage 2 assigns
parts, stages 3 and 4 decide whether anyone can get there in time -- while a
model that denies a capability its own prompt advertises is simply misreading.
"""
import argparse
import json
import os
import re

from mas.selection.agent_pool import advertise

DEFAULT = os.path.join("results", "pipeline", "pipeline_composite_live.jsonl")

TEAM = re.compile(r'(\w+): starts at \([^)]*\); "([^"]*)"')
PART = re.compile(r"PART (\w+):")
FALSE = re.compile(r'"feasible"\s*:\s*false', re.I)

# First match wins, so the order is the claim: an invented constraint is the
# fault even when the sentence also names a capability correctly.
FAULTS = (
    ("invented a constraint",
     r"cannot reach|reach .* in time|in time|blocked by walls|in one step|"
     r"required order|required sequence|after the previous|in sequence"),
    ("read the vocabulary literally", r"not explicitly listed|only walks"),
    ("contradicted the advertisement", r"description|includes"),
    ("contradicted itself", r"unassignable|can (carry|fly|inspect|walk|move)"),
)


def refusals(path):
    """(instance, parts, team, reason) for every refused decomposition call."""
    out = []
    for line in open(path):
        rec = json.loads(line)
        for call in rec.get("calls", []):
            p = call["prompt"]
            if "feasible" not in p or not FALSE.search(call["response"]):
                continue
            try:
                reply = json.loads(call["response"])
            except ValueError:
                m = re.search(r"\{.*\}", call["response"], re.S)
                reply = json.loads(m.group(0)) if m else {}
            out.append((rec["instance"], frozenset(PART.findall(p)),
                        dict(TEAM.findall(p)), reply.get("reason") or ""))
    return out


def fault_of(reason):
    for name, pattern in FAULTS:
        if re.search(pattern, reason, re.I):
            return name
    return "unclassified"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transcript", nargs="?", default=DEFAULT,
                    help=f"a live run's .jsonl (default {DEFAULT})")
    ap.add_argument("--reasons", action="store_true",
                    help="print every refused team, its advertisements and its stated reason")
    a = ap.parse_args()
    path, show = a.transcript, a.reasons
    if not os.path.exists(path):
        raise SystemExit(f"no transcript at {path}. Pass one, or run the live "
                         f"composite with --log first.")

    rs = refusals(path)
    if not rs:
        raise SystemExit(f"{path} holds no refused decomposition calls.")

    counts, wrong = {}, 0
    for inst, parts, team, reason in rs:
        covered = set().union(*(advertise(d) for d in team.values())) if team else set()
        if parts <= covered:
            wrong += 1
        counts[fault_of(reason)] = counts.get(fault_of(reason), 0) + 1

    print(f"{path}\n")
    print(f"refused decomposition calls      {len(rs)}")
    print(f"  of teams that covered the task {wrong}   <- wrong answers, not hard instances")
    print(f"  of teams that did not          {len(rs) - wrong}\n")
    print("the fault each stated reason shows (keyword match, first wins):")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {name}")

    if show:
        print()
        for i, (inst, parts, team, reason) in enumerate(rs, 1):
            covered = set().union(*(advertise(d) for d in team.values())) if team else set()
            mark = "COVERED" if parts <= covered else "short  "
            print(f"{i:3d} {inst:6s} {mark}  needs={sorted(parts)} "
                  f"team advertises={sorted(covered)}")
            for who, desc in sorted(team.items()):
                print(f"      {who}: \"{desc}\"")
            print(f"      -> {reason}\n")

    if wrong == len(rs):
        print(f"\nEvery one of the {len(rs)} refusals was of a team that could cover the task.")


if __name__ == "__main__":
    main()