"""
Read bench results and transcripts. No API calls, free.

    python3 analyze.py batch2.csv batch2.jsonl
    python3 analyze.py var1.csv var1.jsonl

Prints: (1) one row per instance and approach, (2) plan length on the instances ALL approaches solved
(the fair comparison), (3) every invalid answer from the model (why did parsing fail?),
(4) for the no-tree approach, the sequence of agents the model chose at each call.
"""
import csv, json, re, sys
from llm_arbitrator import parse_choice

csv_path, log_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
logs = [json.loads(l) for l in open(log_path)]

short = lambda s: s.split("[")[0].strip().replace("fixed rule ", "fx").replace("LLM no tree", "LLM").replace("hybrid tree", "hyb")
insts = list(dict.fromkeys(r["instance"] for r in rows))
apps = list(dict.fromkeys(r["approach"] for r in rows))

print("== per instance: solved / steps / nodes / LLM calls / invalid ==")
print(f"{'instance':<14}" + "".join(f"{short(a):<22}" for a in apps))
for i in insts:
    line = f"{i:<14}"
    for a in apps:
        rs = [r for r in rows if r["instance"] == i and r["approach"] == a]
        s = sum(int(r["solved"]) for r in rs)
        steps = [int(r["steps"]) for r in rs if r["solved"] == "1"]
        cell = f"{s}/{len(rs)} st{steps[0] if steps else '-'} n{rs[0]['nodes']} c{rs[0]['llm_calls']} i{rs[0]['invalid']}"
        line += f"{cell:<22}"
    print(line)

print("\n== plan length on instances every approach solved (fair comparison) ==")
common = [i for i in insts if all(any(r["instance"] == i and r["approach"] == a and r["solved"] == "1" for r in rows) for a in apps)]
print(f"{len(common)} instances: {common}")
for a in apps:
    v = [int(r["steps"]) for r in rows if r["approach"] == a and r["instance"] in common and r["solved"] == "1"]
    if v:
        print(f"  {short(a):<10} average steps {sum(v) / len(v):.1f}")

print("\n== invalid answers (the model's reply could not be parsed) ==")
bad = 0
for d in logs:
    for k, c in enumerate(d["calls"], 1):
        opts = re.findall(r"OPTION (\w+):", c["prompt"])
        if parse_choice(c["response"], opts) is None:
            bad += 1
            out_tok = c.get("tokens", ["?", "?"])[1]
            if bad <= 8:
                print(f"- {d['instance']} {short(d['approach'])} call {k}: output tokens {out_tok}, reply length {len(c['response'])}")
                print(f"    ends with: ...{c['response'][-160:]!r}")
print(f"total invalid: {bad}")

print("\n== no-tree approach: agents chosen, in order ==")
for d in logs:
    if d["approach"].startswith("2"):
        ch = []
        for c in d["calls"]:
            opts = re.findall(r"OPTION (\w+):", c["prompt"])
            ch.append(parse_choice(c["response"], opts) or "?")
        print(f"  {d['instance']} run{d['run']}: {' '.join(ch)}")