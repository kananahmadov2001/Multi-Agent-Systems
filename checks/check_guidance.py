"""Did the model's node choices change anything?

    python3 -m checks.check_guidance
    python3 -m checks.check_guidance --detail

Free and offline: it reads committed transcripts and the committed control table.
No API calls, no Fast Downward.

Approach 3 asks the model which of four open CBS nodes to expand. The no-LLM
control answers that question with "the cheapest one", which is Approach 1's rule
applied to the same four options. So the interesting comparison has two halves,
and a claim about LLM guidance needs both:

  1. How often did the model answer differently from the control? If it almost
     always picked the cheapest node, a matching solve count says nothing -- the
     model and the control were the same policy.
  2. Did any instance come out differently?

Taken together these say whether guidance is doing nothing or doing something
that happens to cancel out. On the committed grid runs at a 50-call cap the
answer is the first: the model deviates on about a quarter of its expansions and
not one of the 60 instances changes.

The comparison is only honest if both sides ran the same way. These transcripts
and `results/control/control.txt` are both the grid planner at cap 50, so they
are comparable; the Fast Downward runs are not a control for them and are not
read here. `--detail` prints the per-instance deviation rate.
"""
import argparse
import csv
import os
import re

from mas.arbitration.llm_arbitrator import parse_choice

SETS = ("hard", "random", "sweep")
LIVE = os.path.join("results", "haiku", "{}_haiku")
CONTROL = os.path.join("results", "control", "control.txt")
OPTION = re.compile(r"OPTION (\w+): depth (\d+), total steps (\d+)")
LINE = re.compile(r"^(hard|random|sweep)\s+(\S+)\s+(.*)$")


def control_table(path):
    """{(set, instance): solved} for the cheapest-node policy at cap 50.

    control.txt has three result columns -- best-first CBS, the policy at cap 50,
    and at cap 150 -- each a node count, with FAIL appended when it gave up. The
    cap-50 column is the one that matches the live runs' cap.
    """
    out = {}
    for line in open(path):
        m = LINE.match(line.rstrip())
        if not m:
            continue
        cells = re.findall(r"\d+(?:\s+FAIL)?", m.group(3))
        if len(cells) >= 2:
            out[(m.group(1), m.group(2))] = "FAIL" not in cells[1]
    return out


def choices(path):
    """[(instance, rank, deeper)] for every Approach 3 node choice in a transcript."""
    import json
    out = []
    for line in open(path):
        d = json.loads(line)
        if not d.get("approach", "").startswith("3 LLM-guided"):
            continue
        for c in d["calls"]:
            shown = OPTION.findall(c["prompt"])
            if not shown:
                continue
            ids = [s[0] for s in shown]
            got = parse_choice(c["response"], ids, key="expand")
            if got is None:
                continue
            i = ids.index(got)
            out.append((d["instance"], i + 1, int(shown[i][1]) > int(shown[0][1])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detail", action="store_true",
                    help="per-instance deviation rate")
    a = ap.parse_args()

    for s in SETS:
        for ext in (".csv", ".jsonl"):
            p = LIVE.format(s) + ext
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
    if not os.path.exists(CONTROL):
        raise SystemExit(f"missing {CONTROL}. Run `python3 -m experiments.control "
                         f"> {CONTROL}` to rebuild it (free, no key).")

    # --- half one: how often did it disagree with "take the cheapest"? ---
    picks = []
    for s in SETS:
        picks += [(s,) + c for c in choices(LIVE.format(s) + ".jsonl")]
    if not picks:
        raise SystemExit("no Approach 3 node choices found in the transcripts.")
    off = [p for p in picks if p[2] != 1]

    # --- half two: did any instance change? ---
    ctl = control_table(CONTROL)
    live = {}
    for s in SETS:
        for x in csv.DictReader(open(LIVE.format(s) + ".csv")):
            if x["approach"].startswith("3 "):
                live[(s, x["instance"])] = x["solved"] == "1"
    shared = sorted(set(ctl) & set(live))
    only = (set(ctl) ^ set(live))
    differ = [k for k in shared if ctl[k] != live[k]]

    print("Approach 3, grid planner, 50-call cap: the model against "
          "'always take the cheapest'\n")
    print(f"node choices in the transcripts        {len(picks)}")
    print(f"  the cheapest node shown              {len(picks) - len(off)}"
          f"  ({100 * (len(picks) - len(off)) / len(picks):.0f}%)")
    print(f"  some other node                      {len(off)}"
          f"  ({100 * len(off) / len(picks):.0f}%)")
    print(f"  of those, deeper than the cheapest   {sum(1 for p in off if p[3])}\n")
    print(f"instances comparable on both sides     {len(shared)}")
    if only:
        print(f"  (skipped, present on one side only: {len(only)})")
    print(f"  the control solved                   {sum(ctl[k] for k in shared)}")
    print(f"  the model solved                     {sum(live[k] for k in shared)}")
    print(f"  instances with a different outcome   {len(differ)}"
          + (f"  {differ}" if differ else ""))

    if a.detail:
        print("\nper instance: choices, and how many were not the cheapest")
        seen = {}
        for s, inst, rank, _ in picks:
            k = f"{s}/{inst}"
            seen.setdefault(k, [0, 0])
            seen[k][0] += 1
            seen[k][1] += rank != 1
        for k, (n, d) in sorted(seen.items(), key=lambda kv: -kv[1][1]):
            print(f"  {k:28s} {n:3d} choices, {d:3d} not the cheapest ({100*d/n:3.0f}%)")

    if not differ and off:
        print(f"\nThe model answered differently from the control on {len(off)} of "
              f"{len(picks)} expansions\nand changed none of the "
              f"{len(shared)} outcomes.")


if __name__ == "__main__":
    main()