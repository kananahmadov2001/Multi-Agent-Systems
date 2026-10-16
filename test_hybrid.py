"""
Approach 3 tests (LLM-guided CBS: the LLM picks which open tree node to expand next).
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


def first(options, prompt):       # options are listed cheapest first: this is plain best-first search
    return options[0]


def last(options, prompt):        # always the least promising of the shown nodes
    return options[-1]


def run(title, grid, agents, llm, horizon):
    paths, stats = solve_hybrid(grid, agents, llm, horizon=horizon)
    show(title, paths, stats)
    return llm


live = "--live" in sys.argv
if not live:       # fake LLMs: free, checks that the tree + LLM loop works
    for who, f in (("cheapest node", first), ("last of 4 nodes", last)):
        run(f"FAKE LLM ({who}): crossing", cross, cross_agents, FakeLLM(f), 30)
    for who, f in (("cheapest node", first), ("last of 4 nodes", last)):
        run(f"FAKE LLM ({who}): pocket", pocket, pocket_agents, FakeLLM(f), 10)
    run("FAKE LLM (cheapest node): head-on corridor (unsolvable)", corridor, corridor_agents,
        FakeLLM(first), 8)
    print("\nNote: unlike Approach 2, the fake LLMs SOLVE the pocket here, because the tree "
          "keeps both children of every collision and can return to an older node.")
    fake = run("prompt example", pocket, pocket_agents, FakeLLM(first), 10)
    print("\n----- a prompt sent to the LLM (second call) -----\n" + fake.log[1]["prompt"])

if live:           # real model only
    for title, g, ag, h in [("crossing", cross, cross_agents, 30), ("pocket", pocket, pocket_agents, 10)]:
        llm = AnthropicLLM()
        run(f"REAL LLM ({llm.model}), LLM-guided CBS: {title}", g, ag, llm, h)
        for i, e in enumerate(llm.log):
            print(f"  call {i + 1}: {e['response'].strip()[-300:]}")
    print(f"\nAPI usage so far: {AnthropicLLM.total_calls} calls, "
          f"{AnthropicLLM.tokens_in} input tokens, {AnthropicLLM.tokens_out} output tokens")