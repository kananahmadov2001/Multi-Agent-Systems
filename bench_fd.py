"""
Section 6, item 1 ("Close the PDDL gap"): run the benchmark with Fast Downward as
the low-level single-agent planner instead of the built-in grid BFS.

This is a NEW file. It does not touch bench.py, cbs_baseline.py, fd_planner.py,
llm_arbitrator.py or hybrid.py. Every approach already accepts a `planner`
argument, so we only inject fd_planner.make_fd_planner into the same search tree.
The high level of the comparison does not change, which is exactly the point:
the numbers should be checkable against the grid run.

By default this runs Approach 1 only (CBS best-first), which is deterministic and
free: no API calls. Approaches 2/3 need LLM calls, so they are opt-in -- --live for
the real model (needs API budget) or --fakes for the $0 fake policies.

    python bench_fd.py --set hard --only pocket-3,hard3      # Approach 1 only, $0
    python bench_fd.py --set hard --limit 5                  # first 5 instances
    python bench_fd.py --set hard --limit 5 --dfs            # add the depth-first ablations
    python bench_fd.py --set hard --limit 5 --live           # also 2/3 (needs API budget)

Requires Fast Downward available. Locally that means Docker Desktop running with
the `aibasel/downward` image tagged; pass --fd <dir> instead if you have a native
build. The first argument of make_fd_planner is passed positionally for that reason.
Each CBS expansion costs roughly one Fast Downward call (~1 s through Docker), so
this is far slower than the grid run: start with --only or --limit. A run counts
as solved only if verify.py accepts it, the same rule as bench.py.
"""
import argparse, json, os, time
from cbs_baseline import solve, solve_best_first
from llm_arbitrator import FakeLLM, AnthropicLLM, solve_llm
from hybrid import solve_hybrid, solve_hybrid_dfs
from instances import gadgets, random_hard, random_plain, sweep
from verify import verify
from fd_planner import make_fd_planner
from bench import HORIZON, easy_instances, longest_path, save_csv, price_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--set", default="hard", choices=["easy", "hard", "random", "sweep", "all"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=os.path.join("results", "fd", "fd_results.csv"))
    ap.add_argument("--dfs", action="store_true")
    ap.add_argument("--fakes", action="store_true",
                    help="also run Approaches 2/3 with the $0 fake policies (off by default: they "
                         "triple the Fast Downward calls without telling us anything about the LLM)")
    ap.add_argument("--a2-calls", type=int, default=20)
    ap.add_argument("--a3-calls", type=int, default=50)
    ap.add_argument("--fd", default="aibasel/downward",
                    help="first argument of fd_planner.make_fd_planner. Either a Fast Downward "
                         "Docker image tag (the local Docker build) or a directory containing "
                         "fast-downward.py (a native build). Passed positionally so this script "
                         "works with either version of fd_planner.py.")
    ap.add_argument("--search", default="astar(lmcut())", help="Fast Downward search configuration")
    a = ap.parse_args()

    inst = []
    if a.set in ("hard", "all"):
        inst += gadgets() + random_hard(20, 11)
    if a.set in ("easy", "all"):
        inst += easy_instances()
    if a.set == "random":
        inst += random_plain(20, 23)
    if a.set == "sweep":
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

    # A fresh planner per instance. fd_planner caches by
    # (start, goal, constraints, horizon) WITHOUT the grid, so one planner shared
    # across two instances with the same start/goal pair would reuse paths computed
    # on the wrong grid. Intra-instance caching is kept: it is what makes a full CBS
    # expansion usually cost about one Fast Downward call rather than one per replan.
    planner = None

    if a.live:
        model = os.environ.get("ARBITER_MODEL", "claude-sonnet-5-5")
        policies = [("real LLM", lambda: AnthropicLLM(), lambda: AnthropicLLM())]
        print(f"Live run, model {model}, {len(inst)} instance(s), repeat {a.repeat}: "
              f"{[i[0] for i in inst]}", flush=True)
    else:
        model = "fake"
        policies = ([("fake: a1 / cheapest node", lambda: FakeLLM(lambda o, p: "a1" if "a1" in o else o[0]),
                      lambda: FakeLLM(lambda o, p: o[0])),
                     ("fake: longer plan / last of 4", lambda: FakeLLM(longest_path),
                      lambda: FakeLLM(lambda o, p: o[-1]))] if a.fakes else [])

    # every approach: (label, make LLM or None, function(grid, agents, llm)); planner is injected below.
    approaches = [
        ("1 CBS best-first [FD]", None,
         lambda g, ag, llm: solve_best_first(g, ag, horizon=HORIZON, max_nodes=6000, planner=planner)),
    ]
    if a.dfs:
        approaches += [
            ("1 DFS ablation (alphabetical) [FD]", None,
             lambda g, ag, llm: solve(g, ag, horizon=HORIZON, max_nodes=500, planner=planner)),
            ("1 DFS ablation (reversed) [FD]", None,
             lambda g, ag, llm: solve(g, ag, rule=lambda x: sorted(x, reverse=True),
                                      horizon=HORIZON, max_nodes=500, planner=planner)),
        ]
    for name, mk2, mk3 in policies:
        approaches.append((f"2 LLM no tree [{name}] [FD]", mk2,
                           lambda g, ag, llm: solve_llm(g, ag, llm, horizon=HORIZON,
                                                        max_calls=a.a2_calls, planner=planner)))
        approaches.append((f"3 LLM-guided CBS [{name}] [FD]", mk3,
                           lambda g, ag, llm: solve_hybrid(g, ag, llm, horizon=HORIZON, max_nodes=500,
                                                           max_calls=a.a3_calls, planner=planner)))
        if a.dfs:
            approaches.append((f"3 hybrid DFS ablation [{name}] [FD]", mk2,
                               lambda g, ag, llm: solve_hybrid_dfs(g, ag, llm, horizon=HORIZON,
                                                                   max_nodes=500, max_calls=40,
                                                                   planner=planner)))
    labels = [x[0] for x in approaches]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows, stopped, total_fd = [], None, 0
    logf = open(a.log, "a") if (a.log and a.live) else None
    try:
        for iname, grid, agents in inst:
            planner = make_fd_planner(a.fd, search=a.search)
            for label, mk, fn in approaches:
                for rep in range(a.repeat):
                    llm = mk() if mk else None
                    fd0 = planner.calls
                    t0 = time.perf_counter()
                    try:
                        paths, st = fn(grid, agents, llm)
                    finally:
                        if logf and llm is not None and llm.log:
                            logf.write(json.dumps({"instance": iname, "approach": label, "run": rep + 1,
                                                   "model": model, "calls": llm.log}) + "\n")
                            logf.flush()
                    seconds = time.perf_counter() - t0
                    fd_calls = planner.calls - fd0
                    total_fd += fd_calls
                    calls_log = llm.log if llm is not None else []
                    tin = sum(c["tokens"][0] for c in calls_log if c.get("tokens"))
                    tout = sum(c["tokens"][1] for c in calls_log if c.get("tokens"))
                    truncated = sum(1 for c in calls_log if c.get("stop_reason") == "max_tokens")
                    claimed = paths is not None
                    if claimed:
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
                                 "fd_calls": fd_calls, "call_cap": f"{a.a2_calls}/{a.a3_calls}"})
                    print(f"  {iname:<20}{label:<40} {'ok' if paths else 'fail':<5} "
                          f"fd={fd_calls:<4} nodes={st.get('nodes', 0):<4} {seconds:5.1f}s", flush=True)
            save_csv(rows, a.out)
    except RuntimeError as e:
        stopped = str(e)
    except KeyboardInterrupt:
        stopped = "Interrupted with Ctrl+C. Results for the finished instances were kept."
    finally:
        if logf:
            logf.close()
    save_csv(rows, a.out)

    done = sorted({r["instance"] for r in rows})
    full = [i for i in done if sum(r["instance"] == i for r in rows) == len(approaches) * a.repeat]
    print(f"\n{len(inst)} instances requested; {len(full)} complete. Results saved to {a.out}\n")
    print(f"{'approach':<54}{'solved':>8}{'nodes':>8}{'replans':>9}{'backtr.':>9}"
          f"{'FD calls':>10}{'LLM calls':>11}{'avg steps':>11}")
    for label in labels:
        rs = [r for r in rows if r["approach"] == label and r["instance"] in full]
        t = lambda k: sum(r[k] for r in rs)
        ok = [r for r in rs if r["solved"]]
        avg = f"{sum(r['steps'] for r in ok) / len(ok):.1f}" if ok else "-"
        print(f"{label:<54}{t('solved'):>5}/{len(rs):<3}{t('nodes'):>8}{t('replans'):>9}"
              f"{t('backtracks'):>9}{t('fd_calls'):>10}{t('llm_calls'):>11}{avg:>11}")
    print(f"  total Fast Downward calls: {total_fd}")

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
