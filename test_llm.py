"""
Approach 2 tests. Default is OFFLINE with fake LLMs, so you can check the loop works.
    python3 test_llm.py          fake LLMs only (free)
    python3 test_llm.py --live   real model only (needs: pip install anthropic, ANTHROPIC_API_KEY)
"""
import sys
from cbs_baseline import Grid, show
from llm_arbitrator import FakeLLM, AnthropicLLM, solve_llm

def pick(name):
    return lambda options, prompt: name if name in options else options[0]

cross = Grid(["...", "...", "..."])
cross_agents = {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}
pocket = Grid(["...", "#.#"])
pocket_agents = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}
corridor = Grid(["..."])

def run(title, grid, agents, llm, horizon=30):
    paths, stats = solve_llm(grid, agents, llm, horizon=horizon)
    show(title, paths, stats)
    return llm

live = "--live" in sys.argv
if not live:      # fake LLMs: free, checks that the loop works
    run("FAKE LLM (always a1): crossing", cross, cross_agents, FakeLLM(pick("a1")))
    run("FAKE LLM (always a2): crossing", cross, cross_agents, FakeLLM(pick("a2")))
    run("FAKE LLM (always a1): pocket", pocket, pocket_agents, FakeLLM(pick("a1")), horizon=10)
    run("FAKE LLM (always a2): pocket", pocket, pocket_agents, FakeLLM(pick("a2")), horizon=10)
    run("FAKE LLM (always a1): head-on corridor (unsolvable)", corridor,
        {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}, FakeLLM(pick("a1")), horizon=8)

    # one example prompt, so you can see exactly what the LLM is shown
    fake = run("prompt example", cross, cross_agents, FakeLLM(pick("a1")))
    print("\n----- prompt sent to the LLM -----\n" + fake.log[0]["prompt"])

if live:      # real model only (costs a few cents)
    for title, g, ag, h in [("crossing", cross, cross_agents, 30), ("pocket", pocket, pocket_agents, 10)]:
        llm = AnthropicLLM()
        run(f"REAL LLM ({llm.model}): {title}", g, ag, llm, horizon=h)
        for i, e in enumerate(llm.log):
            print(f"  call {i + 1}: {e['response']}")
    print(f"\nAPI usage so far: {AnthropicLLM.total_calls} calls, "
          f"{AnthropicLLM.tokens_in} input tokens, {AnthropicLLM.tokens_out} output tokens")