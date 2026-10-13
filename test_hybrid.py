"""
Approach 3 tests (the hybrid tree: LLM picks which agent to constrain first, tree still backtracks).
    python3 test_hybrid.py          fake LLMs only (free)
    python3 test_hybrid.py --live   real model only (costs a few cents)
"""
import sys
from cbs_baseline import Grid, show
from llm_arbitrator import FakeLLM, AnthropicLLM
from hybrid import solve_hybrid

cross = Grid(["...", "...", "..."])
cross_agents = {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}
pocket = Grid(["...", "#.#"])
pocket_agents = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}
corridor = Grid(["..."])
corridor_agents = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}


def pick(name):
    return lambda options, prompt: name if name in options else options[0]


def run(title, grid, agents, llm, horizon):
    paths, stats = solve_hybrid(grid, agents, llm, horizon=horizon)
    show(title, paths, stats)
    return llm


live = "--live" in sys.argv
if not live:       # fake LLMs: free, checks that the tree + LLM loop works
    for who in ("a1", "a2"):
        run(f"FAKE LLM (always {who}): crossing", cross, cross_agents, FakeLLM(pick(who)), 30)
    for who in ("a1", "a2"):
        run(f"FAKE LLM (always {who}): pocket", pocket, pocket_agents, FakeLLM(pick(who)), 10)
    run("FAKE LLM (always a1): head-on corridor (unsolvable)", corridor, corridor_agents,
        FakeLLM(pick("a1")), 8)
    print("\nNote: unlike Approach 2, the fake LLMs SOLVE the pocket here, because the tree "
          "backtracks and tries the other agent.")

if live:           # real model only
    for title, g, ag, h in [("crossing", cross, cross_agents, 30), ("pocket", pocket, pocket_agents, 10)]:
        llm = AnthropicLLM()
        run(f"REAL LLM ({llm.model}), hybrid: {title}", g, ag, llm, h)
        for i, e in enumerate(llm.log):
            print(f"  call {i + 1}: {e['response'].strip()[-300:]}")
    print(f"\nAPI usage so far: {AnthropicLLM.total_calls} calls, "
          f"{AnthropicLLM.tokens_in} input tokens, {AnthropicLLM.tokens_out} output tokens")