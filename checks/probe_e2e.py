"""Probe: can the pipeline run on Fast Downward with LLM arbitration?

Three stages, run in order, each reporting what actually happened. Nothing is
written to results/ and no file is modified; this only answers "where does the
end-to-end chain break, and is it where we predicted".

  A  unbound solver        expect TypeError: missing 1 required positional argument: 'llm'
  B  bound + grid planner  expect SUCCESS, which proves the binding is correct
  C  bound + FD planner    expect TypeError: unhashable type: 'list'

Stage A and C make no API calls: both fail before the first call. Stage B runs
one instance with the real model, which is a handful of calls.

    export ANTHROPIC_API_KEY=...
    export ARBITER_MODEL=claude-haiku-4-5-20251001
    export ARBITER_MAX_CALLS=40
    python3 -m checks.probe_e2e --fd /path/to/downward

    python3 -m checks.probe_e2e --fd /path/to/downward --fake   # stage B offline
"""
import argparse
import traceback

from mas.decomposition.decompose import greedy_decompose
from mas.planning.fd_planner import make_fd_planner
from mas.arbitration.hybrid import solve_hybrid
from mas.arbitration.llm_arbitrator import AnthropicLLM, FakeLLM
from mas.pipeline import run_pipeline
from mas.instances.pipeline_instances import capability_overlap


def stage(name, what, fn):
    """Run one stage and report the outcome without stopping the probe."""
    print(f"\n--- {name}: {what}")
    try:
        result = fn()
    except Exception:
        last = traceback.format_exc().strip().splitlines()[-1]
        where = [l.strip() for l in traceback.format_exc().splitlines()
                 if l.strip().startswith("File ")][-1]
        print(f"    FAILED   {last}")
        print(f"    at       {where}")
        return None
    print(f"    OK       {result}")
    return result


def describe(res):
    """One line for a PipelineResult."""
    last = res.attempts[-1] if res.attempts else None
    kind = "solved" if (last and last.ok) else (last.kind if last else "no attempt")
    return (f"outcome={kind}  teams_tried={len(res.attempts)}"
            + (f"  steps={sum(len(p) - 1 for p in last.paths.values())}"
               if last and last.paths else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", required=True,
                    help="directory containing fast-downward.py (native build)")
    ap.add_argument("--instance", type=int, default=0, help="which instance to probe")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fake", action="store_true",
                    help="use FakeLLM in stage B instead of the real model")
    a = ap.parse_args()

    inst = capability_overlap(a.instance + 1, seed=a.seed)[a.instance]
    common = dict(grid=inst.grid, pool=inst.pool, task=inst.task, k=2,
                  sites=inst.sites, decomposer=greedy_decompose)

    # What the pipeline will hand the planner, before anything runs.
    chosen = run_pipeline(**common).attempts[0]
    print(f"instance {a.instance} (seed {a.seed}), grid {inst.grid.h}x{inst.grid.w}")
    print(f"goals the planner will receive: {chosen.goals}")

    llm = (FakeLLM(lambda options, prompt: options[0]) if a.fake
           else AnthropicLLM())
    bound = lambda g, m, **kw: solve_hybrid(g, m, llm, **kw)

    stage("A", "unbound solver, grid planner",
          lambda: describe(run_pipeline(**common, solver=solve_hybrid)))

    stage("B", f"bound solver ({'FakeLLM' if a.fake else llm.model}), grid planner",
          lambda: describe(run_pipeline(**common, solver=bound)))

    fd = make_fd_planner(a.fd)
    stage("C", "bound solver, Fast Downward planner",
          lambda: describe(run_pipeline(**common, solver=bound, planner=fd)))

    print("\nIf A failed on the missing 'llm', B succeeded, and C failed on the")
    print("unhashable list, the break is isolated to fd_planner's single-cell goal")
    print("assumption, which is step 3 of the plan.")


if __name__ == "__main__":
    main()
