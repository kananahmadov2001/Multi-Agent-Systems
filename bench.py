"""
Compare the three approaches on the same instances.

    python3 bench.py              offline: fake LLM policies (checks the harness, says nothing about real LLMs)
    python3 bench.py --live       real model for Approaches 2 and 3 (needs ANTHROPIC_API_KEY)

Instances: the three toy cases plus N random small grids (fixed seed, so everyone gets the same ones).
"""
import random, sys
from cbs_baseline import Grid, plan_single, solve, find_conflict
from llm_arbitrator import FakeLLM, AnthropicLLM, solve_llm
from hybrid import solve_hybrid

N_RANDOM, SEED, HORIZON = 30, 7, 20


def toy_instances():
    return [
        ("crossing", Grid(["...", "...", "..."]), {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}),
        ("pocket", Grid(["...", "#.#"]), {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}),
    ]


def random_instances(n, seed, size=5, walls=0.2, n_agents=2):
    rng, out = random.Random(seed), []
    while len(out) < n:
        rows = ["".join("#" if rng.random() < walls else "." for _ in range(size)) for _ in range(size)]
        grid = Grid(rows)
        free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
        if len(free) < 2 * n_agents:
            continue
        pick = rng.sample(free, 2 * n_agents)
        agents = {f"a{i + 1}": (pick[2 * i], pick[2 * i + 1]) for i in range(n_agents)}
        if any(plan_single(grid, s, g, frozenset(), HORIZON) is None for s, g in agents.values()):
            continue                                   # each agent must be solvable alone
        if find_conflict({a: plan_single(grid, s, g, frozenset(), HORIZON)
                          for a, (s, g) in agents.items()}) is None:
            continue                                   # keep only instances with a real conflict
        out.append((f"random{len(out) + 1}", grid, agents))
    return out


def longest_path(options, prompt):
    """Fake policy: constrain the agent with the longer plan. A guess, not a real LLM."""
    import re
    lens = {a: len(eval(p)) for a, p in re.findall(r"  (\w+): .*?plan (\[.*\])", prompt)}
    return max(options, key=lambda a: lens.get(a, 0))


def main():
    live = "--live" in sys.argv
    instances = toy_instances() + random_instances(N_RANDOM, SEED)
    make_llm = (lambda: AnthropicLLM()) if live else None
    policies = ([("real LLM", None)] if live else
                [("fake: always a1", lambda: FakeLLM(lambda o, p: "a1" if "a1" in o else o[0])),
                 ("fake: longer plan", lambda: FakeLLM(longest_path))])

    runs = [("1 fixed rule (alphabetical)", lambda g, ag: solve(g, ag, horizon=HORIZON, max_nodes=500)),
            ("1 fixed rule (reversed)", lambda g, ag: solve(g, ag, rule=lambda x: sorted(x, reverse=True),
                                                              horizon=HORIZON, max_nodes=500))]
    for name, mk in policies:
        mk = mk or make_llm
        runs.append((f"2 LLM no tree [{name}]", lambda g, ag, mk=mk: solve_llm(g, ag, mk(), horizon=HORIZON)))
        runs.append((f"3 hybrid tree [{name}]", lambda g, ag, mk=mk: solve_hybrid(g, ag, mk(), horizon=HORIZON, max_nodes=500)))

    print(f"{len(instances)} instances (2 toy + {N_RANDOM} random 5x5 grids, seed {SEED}). "
          f"Unsolvable cases are left out here; cbs_baseline.py covers the head-on case.\n")
    print(f"{'approach':<42}{'solved':>7}{'nodes':>8}{'replans':>9}{'backtr.':>9}{'LLM calls':>11}")
    for label, fn in runs:
        solved, tot = 0, {"nodes": 0, "replans": 0, "backtracks": 0, "llm_calls": 0}
        for _, grid, agents in instances:
            paths, st = fn(grid, agents)
            solved += paths is not None
            for k in tot:
                tot[k] += st.get(k, 0)
        print(f"{label:<42}{solved:>7}{tot['nodes']:>8}{tot['replans']:>9}{tot['backtracks']:>9}{tot['llm_calls']:>11}")


if __name__ == "__main__":
    main()