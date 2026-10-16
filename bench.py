"""
Compare the three approaches on the same instances.

    python3 bench.py                                offline, hard set, fake LLM policies (checks the harness only)
    python3 bench.py --live --limit 5               real model on the first 5 instances
    python3 bench.py --live --start 25 --limit 8    real model on instances 25..32 (a new batch)
    python3 bench.py --live --only pocket-5-mid --log transcripts.jsonl     one case, save every prompt and answer
    python3 bench.py --live --only pocket-5-mid,hard3 --repeat 3            variance: 3 runs of each

Options:
  --set easy|hard|all   which instances (default hard: 5 gadgets + 20 random = 25, numbered from 0)
  --start N             skip the first N instances        --limit N   then take at most N
  --only A,B            keep only instances with these names (applied first)
  --repeat R            run every approach R times per instance (the real model varies between runs)
  --log FILE            live only: save every prompt and answer as one JSON line (for failure analysis)
  --out FILE            results CSV (default results.csv), rewritten after every instance
  --dfs                 also run the depth-first ablation (two fixed rules, and the depth-first hybrid)
A run counts as solved only if verify.py's independent replay accepts it.
'avg steps' = total moves by all agents, averaged over the cases that approach solved.
Live runs stop by themselves at ARBITER_MAX_CALLS (default 300) API calls. Ctrl+C is safe: results so far are kept.
Model: env ARBITER_MODEL (default claude-sonnet-5-5). Do not mix models in one table.

Approaches (all use the same Conflict-Based-Search tree):
  1 CBS best-first        always expands the cheapest open node (Sharon et al. 2015). No LLM, no API cost.
  2 LLM no tree           the LLM picks which agent to constrain at each collision, no tree, with its own
                          earlier decisions shown to it as its only memory.
  3 LLM-guided CBS        the same tree, but the LLM picks which open node to expand next.
"""
import argparse, csv, json, os, random, re, time
from cbs_baseline import Grid, plan_single, solve, solve_best_first, find_conflict
from llm_arbitrator import FakeLLM, AnthropicLLM, solve_llm
from hybrid import solve_hybrid, solve_hybrid_dfs
from instances import gadgets, random_hard, random_plain, sweep
from verify import verify

HORIZON = 20
PRICES = {"haiku": (1.0, 5.0)}           # $ per million tokens (input, output); check the console price page
DEFAULT_PRICE = (2.0, 10.0)              # Sonnet 5.5 (matches the bill from our first live runs)
CALL_COST = {"haiku": 0.004}             # rough $ per call, measured on Sonnet at 0.015


def price_for(model):
    return next((v for k, v in PRICES.items() if k in model), DEFAULT_PRICE)


def call_cost_for(model):
    return next((v for k, v in CALL_COST.items() if k in model), 0.015)


def easy_instances(n=30, seed=7, size=5, walls=0.2):
    rng, out = random.Random(seed), []
    out.append(("crossing", Grid(["...", "...", "..."]), {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}))
    while len(out) < n + 1:
        grid = Grid(["".join("#" if rng.random() < walls else "." for _ in range(size)) for _ in range(size)])
        free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
        if len(free) < 4:
            continue
        p = rng.sample(free, 4)
        agents = {"a1": (p[0], p[1]), "a2": (p[2], p[3])}
        paths = {a: plan_single(grid, s, g, frozenset(), HORIZON) for a, (s, g) in agents.items()}
        if any(v is None for v in paths.values()) or find_conflict(paths) is None:
            continue
        out.append((f"easy{len(out)}", grid, agents))
    return out


def longest_path(options, prompt):
    """Fake policy: constrain the agent with the longer plan. A guess, not a real LLM."""
    lens = {a: len(eval(p)) for a, p in re.findall(r"  (\w+): .*?plan (\[.*\])", prompt)}
    return max(options, key=lambda a: lens.get(a, 0))


def save_csv(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--set", default="hard", choices=["easy", "hard", "random", "sweep", "all"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--dfs", action="store_true")
    ap.add_argument("--a2-calls", type=int, default=20, help="call budget per case for Approach 2 (default 20)")
    ap.add_argument("--a3-calls", type=int, default=50, help="call budget per case for Approach 3 (default 50)")
    a = ap.parse_args()

    inst = []
    if a.set in ("hard", "all"):
        inst += gadgets() + random_hard(20, 11)
    if a.set in ("easy", "all"):
        inst += easy_instances()
    if a.set == "random":                       # held-out, unfiltered: nothing selected for being hard
        inst += random_plain(20, 23)
    if a.set == "sweep":                        # same kind of case with 2, 3 and 4 agents
        inst += sweep()
    if a.only:
        want = set(a.only.split(","))
        missing = want - {i[0] for i in inst}
        if missing:
            raise SystemExit(f"No instance named {sorted(missing)} in the '{a.set}' set. "
                             f"Names: {[i[0] for i in inst]}")
        inst = [i for i in inst if i[0] in want]
    inst = inst[a.start:]
    if a.limit:
        inst = inst[:a.limit]

    # policies: (name, make LLM for Approach 2 (agent choice), make LLM for Approach 3 (node choice))
    if a.live:
        model = os.environ.get("ARBITER_MODEL", "claude-sonnet-5-5")
        policies = [("real LLM", lambda: AnthropicLLM(), lambda: AnthropicLLM())]
        print(f"Live run, model {model}, {len(inst)} instance(s), repeat {a.repeat}: "
              f"{[i[0] for i in inst]}", flush=True)
        n_runs = len(inst) * a.repeat
        worst = a.a2_calls + a.a3_calls
        cc = call_cost_for(model)
        print(f"Max calls per run: {a.a2_calls} (no tree) + {a.a3_calls} (LLM-guided CBS) = {worst}, so at most "
              f"{worst * n_runs} calls (global cap {AnthropicLLM.max_total_calls}). Rough cost: about "
              f"${0.5 * worst * cc * n_runs:.1f} typical, ${worst * cc * n_runs:.1f} worst case.", flush=True)
    else:
        model = "fake"
        policies = [("fake: a1 / cheapest node", lambda: FakeLLM(lambda o, p: "a1" if "a1" in o else o[0]),
                     lambda: FakeLLM(lambda o, p: o[0])),
                    ("fake: longer plan / last of 4", lambda: FakeLLM(longest_path),
                     lambda: FakeLLM(lambda o, p: o[-1]))]
    if a.log and not a.live:
        print("(--log only matters with --live; offline fakes are not saved)")

    # every approach: (label, make LLM or None, function(grid, agents, llm)); a fresh LLM object per run
    approaches = [
        ("1 CBS best-first", None,
         lambda g, ag, llm: solve_best_first(g, ag, horizon=HORIZON, max_nodes=6000)),
    ]
    if a.dfs:                                   # ablation: the earlier depth-first search
        approaches += [
            ("1 DFS ablation (alphabetical)", None,
             lambda g, ag, llm: solve(g, ag, horizon=HORIZON, max_nodes=500)),
            ("1 DFS ablation (reversed)", None,
             lambda g, ag, llm: solve(g, ag, rule=lambda x: sorted(x, reverse=True), horizon=HORIZON, max_nodes=500)),
        ]
    for name, mk2, mk3 in policies:
        approaches.append((f"2 LLM no tree [{name}]", mk2,
                           lambda g, ag, llm: solve_llm(g, ag, llm, horizon=HORIZON, max_calls=a.a2_calls)))
        approaches.append((f"3 LLM-guided CBS [{name}]", mk3,
                           lambda g, ag, llm: solve_hybrid(g, ag, llm, horizon=HORIZON, max_nodes=500, max_calls=a.a3_calls)))
        if a.dfs:
            approaches.append((f"3 hybrid DFS ablation [{name}]", mk2,
                               lambda g, ag, llm: solve_hybrid_dfs(g, ag, llm, horizon=HORIZON, max_nodes=500, max_calls=40)))
    labels = [x[0] for x in approaches]

    rows, stopped = [], None
    logf = open(a.log, "a") if (a.log and a.live) else None
    try:
        for iname, grid, agents in inst:
            for label, mk, fn in approaches:
                for rep in range(a.repeat):
                    llm = mk() if mk else None
                    t0 = time.perf_counter()
                    try:
                        paths, st = fn(grid, agents, llm)
                    finally:
                        if logf and llm is not None and llm.log:     # save even if the call cap fires mid-run
                            logf.write(json.dumps({"instance": iname, "approach": label, "run": rep + 1,
                                                   "model": model, "calls": llm.log}) + "\n")
                            logf.flush()
                    seconds = time.perf_counter() - t0
                    calls_log = llm.log if llm is not None else []
                    tin = sum(c["tokens"][0] for c in calls_log if c.get("tokens"))
                    tout = sum(c["tokens"][1] for c in calls_log if c.get("tokens"))
                    truncated = sum(1 for c in calls_log if c.get("stop_reason") == "max_tokens")
                    claimed = paths is not None
                    if claimed:                      # success is decided by the independent check
                        ok, _why = verify(grid, agents, paths)
                        if not ok:
                            paths = None
                    mismatch = int(claimed and paths is None)
                    steps = sum(len(p) - 1 for p in paths.values()) if paths else 0
                    makespan = max(len(p) - 1 for p in paths.values()) if paths else 0
                    rows.append({"instance": iname, "agents": len(agents), "approach": label, "run": rep + 1,
                                 "solved": int(paths is not None), "failed_verify": mismatch,
                                 "steps": steps, "makespan": makespan,
                                 "nodes": st.get("nodes", 0),
                                 "replans": st["replans"], "backtracks": st.get("backtracks", 0),
                                 "llm_calls": st.get("llm_calls", 0), "invalid": st.get("invalid_answers", 0),
                                 "fallbacks": st.get("fallbacks", 0), "truncated": truncated,
                                 "tokens_in": tin, "tokens_out": tout, "seconds": round(seconds, 2),
                                 "hit_limit": int(bool(st.get("hit_limit"))), "model": model,
                                 "call_cap": f"{a.a2_calls}/{a.a3_calls}"})
            save_csv(rows, a.out)                     # progressive save after every instance
            if a.live:
                print(f"  done {iname}  (API calls so far: {AnthropicLLM.total_calls})", flush=True)
    except RuntimeError as e:                         # the call cap
        stopped = str(e)
    except KeyboardInterrupt:
        stopped = "Interrupted with Ctrl+C. Results for the finished instances were kept."
    finally:
        if logf:
            logf.close()
    save_csv(rows, a.out)

    done = sorted({r["instance"] for r in rows})
    print(f"\n{len(inst)} instances requested; table covers those finished by every approach.")
    full = [i for i in done if sum(r["instance"] == i for r in rows) == len(approaches) * a.repeat]
    print(f"{len(full)} instances complete. Results saved to {a.out}"
          + (f", transcripts in {a.log}" if a.log and a.live else "") + "\n")
    print(f"{'approach':<54}{'solved':>8}{'nodes':>8}{'replans':>9}{'backtr.':>9}{'LLM calls':>11}{'invalid':>9}{'avg steps':>11}")
    for label in labels:
        rs = [r for r in rows if r["approach"] == label and r["instance"] in full]
        t = lambda k: sum(r[k] for r in rs)
        ok = [r for r in rs if r["solved"]]
        avg = f"{sum(r['steps'] for r in ok) / len(ok):.1f}" if ok else "-"
        print(f"{label:<54}{t('solved'):>5}/{len(rs):<3}{t('nodes'):>8}{t('replans'):>9}{t('backtracks'):>9}{t('llm_calls'):>11}{t('invalid'):>9}{avg:>11}")

    print(f"\n{'approach':<54}{'fallbacks':>10}{'truncated':>10}{'tokens in':>11}{'tokens out':>11}{'seconds':>9}")
    for label in labels:
        rs = [r for r in rows if r["approach"] == label and r["instance"] in full]
        t = lambda k: sum(r[k] for r in rs)
        print(f"{label:<54}{t('fallbacks'):>10}{t('truncated'):>10}{t('tokens_in'):>11}{t('tokens_out'):>11}{t('seconds'):>9.1f}")
    print("  fallbacks = replies still invalid after a retry, so the cheapest node was used instead of the LLM's choice (Approach 3).")
    print("  truncated = replies cut off by the token limit.")

    if a.repeat > 1:                                  # variance: how often does each case succeed?
        print(f"\nVariance over {a.repeat} runs (LLM approaches): runs solved / runs, and LLM calls per run")
        for label in labels:
            if label.startswith("1"):
                continue
            print(f"  {label}")
            for i in full:
                rs = [r for r in rows if r["approach"] == label and r["instance"] == i]
                calls = [r["llm_calls"] for r in rs]
                steps = [r["steps"] for r in rs if r["solved"]]
                st_txt = f"steps {min(steps)}-{max(steps)}" if steps else "steps -"
                print(f"    {i:<20}{sum(r['solved'] for r in rs)}/{len(rs)} solved   "
                      f"calls {min(calls)}-{max(calls)}   {st_txt}")

    bad = sum(r["failed_verify"] for r in rows)
    print(f"\nVerification: {bad} run(s) claimed a solution that failed the independent check "
          f"(should be 0).")
    if a.live:
        pin, pout = price_for(model)
        cost = (AnthropicLLM.tokens_in * pin + AnthropicLLM.tokens_out * pout) / 1e6
        print(f"\nAPI usage: {AnthropicLLM.total_calls} calls, {AnthropicLLM.tokens_in} in / "
              f"{AnthropicLLM.tokens_out} out tokens, about ${cost:.2f}  (model {model})")
    if stopped:
        print("\n" + stopped)


if __name__ == "__main__":
    main()