"""
Read bench results and transcripts. No API calls, free.

    python3 analyze.py batch2.csv batch2.jsonl
    python3 analyze.py var1.csv var1.jsonl

Prints: (1) one row per instance and approach (all repeats combined), (2) plan length on the
instances ALL approaches solved, including a "best of both fixed rules" baseline,
(3) why each invalid answer was invalid, (4) the agents the no-tree approach chose, in order.
"""
import csv, json, re, sys
from collections import Counter
from llm_arbitrator import parse_choice

csv_path, log_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
logs = [json.loads(l) for l in open(log_path)]

short = lambda s: s.split("[")[0].strip().replace("CBS best-first", "CBS-bf").replace("fixed rule ", "fx").replace("LLM no tree", "LLM").replace("hybrid tree", "hyb").replace("LLM-guided CBS", "hyb-CBS").replace("hybrid DFS ablation", "hyb-DFS").replace("DFS ablation ", "DFS-")
insts = list(dict.fromkeys(r["instance"] for r in rows))
apps = list(dict.fromkeys(r["approach"] for r in rows))
fixed = [a for a in apps if a.startswith("1 fixed rule")]   # the two depth-first rules
I = lambda r, k: int(r[k])


def mean(v):
    return sum(v) / len(v)


print("== per instance (all repeats): solved/runs  steps  nodes(avg)  LLM calls(total)  invalid(total) ==")
print(f"{'instance':<14}" + "".join(f"{short(a):<26}" for a in apps))
for i in insts:
    line = f"{i:<14}"
    for a in apps:
        rs = [r for r in rows if r["instance"] == i and r["approach"] == a]
        steps = [I(r, "steps") for r in rs if r["solved"] == "1"]
        st = (f"{min(steps)}" if steps and min(steps) == max(steps) else f"{min(steps)}-{max(steps)}") if steps else "-"
        cell = (f"{sum(I(r, 'solved') for r in rs)}/{len(rs)} st{st} n{mean([I(r, 'nodes') for r in rs]):.0f} "
                f"c{sum(I(r, 'llm_calls') for r in rs)} i{sum(I(r, 'invalid') for r in rs)}")
        line += f"{cell:<26}"
    print(line)

# plan length per instance and approach (average over solved repeats); "best of both fixed rules" = shorter plan
steps_by = {}
for a in apps:
    for i in insts:
        v = [I(r, "steps") for r in rows if r["instance"] == i and r["approach"] == a and r["solved"] == "1"]
        if v:
            steps_by[(a, i)] = mean(v)
best_name = "fx best-of-both"
if len(fixed) >= 2:
    for i in insts:
        v = [steps_by[(a, i)] for a in fixed if (a, i) in steps_by]
        if v:
            steps_by[(best_name, i)] = min(v)
cols = apps + ([best_name] if len(fixed) >= 2 else [])

print("\n== plan length on instances every approach solved (fair comparison) ==")
common = [i for i in insts if all((a, i) in steps_by for a in cols)]
print(f"{len(common)} instances: {common}")
for a in cols:
    v = [steps_by[(a, i)] for i in common]
    if v:
        print(f"  {short(a) if a != best_name else best_name:<18} average steps {mean(v):.1f}")
print("  CBS-bf = best-first CBS: the shortest possible plan. The LLM approaches cannot beat it on plan length."
      + ("\n  best-of-both = run both depth-first rules, keep the shorter plan (old result files only)." if len(fixed) >= 2 else ""))

# by number of agents (difficulty), when the file has more than one group
groups = sorted({int(r["agents"]) for r in rows})
if len(groups) > 1:
    print("\n== solved / runs by number of agents ==")
    print(f"{'approach':<54}" + "".join(f"{str(k) + ' agents':>12}" for k in groups) + "   (avg nodes per run)")
    for a_ in apps:
        cells = []
        for k in groups:
            rs = [r for r in rows if r["approach"] == a_ and int(r["agents"]) == k]
            cells.append(f"{sum(I(r, 'solved') for r in rs)}/{len(rs)} n{mean([I(r, 'nodes') for r in rs]):.0f}" if rs else "-")
        print(f"{a_:<54}" + "".join(f"{c:>12}" for c in cells))

if rows and "fallbacks" in rows[0]:
    print("\n== replies that fell back to the cheapest node / were cut off by the token limit ==")
    for a_ in apps:
        rs = [r for r in rows if r["approach"] == a_]
        if sum(I(r, "llm_calls") for r in rs):
            print(f"  {short(a_):<12} fallbacks {sum(I(r, 'fallbacks') for r in rs):>3}   truncated {sum(I(r, 'truncated') for r in rs):>3}   "
                  f"invalid {sum(I(r, 'invalid') for r in rs):>3}   seconds {sum(float(r['seconds']) for r in rs):.0f}")

print("\n== why replies were invalid ==")
cause = Counter()
examples = []
for d in logs:
    for k, c in enumerate(d["calls"], 1):
        opts = re.findall(r"OPTION (\w+):", c["prompt"])
        key = "expand" if '{"expand"' in c["prompt"] else "constrain"
        if parse_choice(c["response"], opts, key=key) is not None:
            continue
        out_tok = c["tokens"][1] if c.get("tokens") else None
        at_cap = (c.get("stop_reason") == "max_tokens") or (c.get("stop_reason") is None and out_tok is not None and out_tok >= 1500)
        if at_cap and len(c["response"]) == 0:
            why = "empty reply, hit token cap (thinking used the whole budget)"
        elif at_cap:
            why = "reply cut off at token cap"
        else:
            why = "other (not at the cap)"
            examples.append((d["instance"], short(d["approach"]), k, out_tok, c["response"][-200:]))
        cause[(short(d["approach"]), why)] += 1
for (a, why), n in sorted(cause.items()):
    print(f"  {a:<8} {n:>3}  {why}")
print(f"  total invalid: {sum(cause.values())}")
for e in examples[:4]:
    print(f"  example 'other': {e[0]} {e[1]} call {e[2]} out_tokens={e[3]} ends: ...{e[4]!r}")

print("\n== no-tree approach: agents chosen, in order ==")
for d in logs:
    if d["approach"].startswith("2"):
        ch = []
        for c in d["calls"]:
            opts = re.findall(r"OPTION (\w+):", c["prompt"])
            ch.append(parse_choice(c["response"], opts) or "?")
        print(f"  {d['instance']} run{d['run']}: {' '.join(ch)}")

print("\n== LLM-guided CBS: which open node did the LLM choose? (rank 1 = the cheapest shown) ==")
for d in logs:
    if not d["approach"].startswith("3 LLM-guided"):
        continue
    ranks, deeper = [], 0
    for c in d["calls"]:
        shown = re.findall(r"OPTION (\w+): depth (\d+), total steps (\d+)", c["prompt"])
        got = parse_choice(c["response"], [s[0] for s in shown], key="expand")
        if got is None or not shown:
            continue
        ids = [s[0] for s in shown]
        ranks.append(ids.index(got) + 1)
        if int(shown[ids.index(got)][1]) > int(shown[0][1]):
            deeper += 1
    if ranks:
        share = sum(1 for r in ranks if r == 1) / len(ranks)
        print(f"  {d['instance']} run{d['run']}: {len(ranks)} choices, cheapest {share:.0%}, "
              f"chose a deeper node than the cheapest {deeper}x, ranks {ranks}")