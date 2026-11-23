"""Is standard-mode CBS actually optimal? Compare it against brute force.

check_edge.py establishes that the stronger swap rule is never cheaper than the
standard one, which is the right invariant but does not show that the standard
one is right. Standard CBS is meant to be complete and optimal, so on instances
small enough to search exhaustively it must return exactly the brute-force
optimum. This checks that, and reports where the stronger rule deviates.

The brute force is a Dijkstra over joint states under the same conventions the
rest of the repo uses: a step may be a wait, two agents may not share a cell or
swap, and an agent that reaches its goal stops there and keeps occupying it
(which is what cbs_baseline.pos_at does by padding). Cost is the sum of each
agent's arrival time.

    python3 check_edge_oracle.py
    python3 check_edge_oracle.py --agents 3 --cases 300
"""
import argparse
import heapq
import itertools
import random

from cbs_baseline import Grid, solve_best_first
from verify import verify


def brute_force(grid, agents, horizon):
    """Least total arrival time over all collision-free joint plans, or None.

    State: each agent's cell, plus whether it has stopped on its goal. An agent
    that has stopped cannot move again, so a goal cell stays occupied -- the
    same thing pos_at does when it pads a finished path.
    """
    names = sorted(agents)
    starts = tuple(agents[n][0] for n in names)
    goals = tuple(agents[n][1] for n in names)
    # An agent already on its goal at t=0 has stopped there at cost 0.
    done0 = tuple(s == g for s, g in zip(starts, goals))
    start = (starts, done0, 0)
    best = {start: 0}
    pq = [(0, start)]
    while pq:
        cost, (cells, done, t) = heapq.heappop(pq)
        if cost > best.get((cells, done, t), float("inf")):
            continue
        if all(done):
            return cost
        if t >= horizon:
            continue
        # Options per agent: a stopped agent stays; others move or wait, and may
        # stop if that lands them on their goal.
        options = []
        for i, n in enumerate(names):
            if done[i]:
                options.append([(cells[i], True)])
                continue
            moves = []
            for nxt in grid.neighbors(cells[i]):
                moves.append((nxt, False))
                if nxt == goals[i]:
                    moves.append((nxt, True))      # arrive and stop here
            options.append(moves)
        for combo in itertools.product(*options):
            nxt_cells = tuple(c for c, _ in combo)
            nxt_done = tuple(d for _, d in combo)
            if len(set(nxt_cells)) != len(nxt_cells):          # vertex collision
                continue
            if any(nxt_cells[i] == cells[j] and nxt_cells[j] == cells[i]
                   for i in range(len(names)) for j in range(i + 1, len(names))):
                continue                                        # swap
            # Each agent not yet stopped at the END of this step pays for it, and
            # an agent that stops now pays for the step that got it there.
            step = sum(1 for i in range(len(names)) if not done[i])
            key = (nxt_cells, nxt_done, t + 1)
            if cost + step < best.get(key, float("inf")):
                best[key] = cost + step
                heapq.heappush(pq, (cost + step, key))
    return None


def cbs(grid, agents, mode, horizon, cap):
    sol, st = solve_best_first(grid, agents, horizon=horizon, max_nodes=cap,
                               conflicts=mode)
    if sol is None:
        return None, st
    ok, problems = verify(grid, agents, sol)
    assert ok, f"{mode} returned a plan verify rejects: {problems[0]}"
    return sum(len(p) - 1 for p in sol.values()), st


def case(rng, size, n_agents, walls):
    grid = Grid(["".join("#" if rng.random() < walls else "." for _ in range(size))
                 for _ in range(size)])
    free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
    if len(free) < 2 * n_agents:
        return None
    pick = rng.sample(free, 2 * n_agents)
    return grid, {f"a{i+1}": (pick[2*i], pick[2*i+1]) for i in range(n_agents)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--walls", type=float, default=0.12)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cases = []
    while len(cases) < a.cases:
        made = case(rng, a.size, a.agents, a.walls)
        if made:
            cases.append(made)

    tally = {"std optimal": 0, "std SUBOPTIMAL": 0, "std MISSED": 0,
             "vertex optimal": 0, "vertex worse": 0, "vertex missed": 0,
             "no solution": 0}
    bad = []
    for grid, agents in cases:
        opt = brute_force(grid, agents, a.horizon)
        std, _ = cbs(grid, agents, "standard", a.horizon, a.cap)
        vtx, _ = cbs(grid, agents, "vertex", a.horizon, a.cap)
        if opt is None:
            tally["no solution"] += 1
            if std is not None or vtx is not None:
                bad.append(("CBS solved an unsolvable instance", grid, agents, opt, std, vtx))
            continue
        if std is None:
            tally["std MISSED"] += 1
            bad.append(("standard missed a solution", grid, agents, opt, std, vtx))
        elif std > opt:
            tally["std SUBOPTIMAL"] += 1
            bad.append(("standard returned a suboptimal plan", grid, agents, opt, std, vtx))
        elif std < opt:
            bad.append(("standard beat brute force, so one of them is wrong",
                        grid, agents, opt, std, vtx))
        else:
            tally["std optimal"] += 1
        if vtx is None:
            tally["vertex missed"] += 1
        elif vtx > opt:
            tally["vertex worse"] += 1
        else:
            tally["vertex optimal"] += 1

    print(f"{len(cases)} cases, {a.size}x{a.size} grids, {a.agents} agents, horizon {a.horizon}\n")
    for k, v in tally.items():
        if v:
            print(f"  {k:<16} {v}")
    print()
    for why, grid, agents, opt, std, vtx in bad[:5]:
        print(f"  !! {why}: brute force {opt}, standard {std}, vertex {vtx}")
        for row in grid.rows:
            print("       ", row)
        print("       ", agents)
    solvable = len(cases) - tally["no solution"]
    if tally["std SUBOPTIMAL"] or tally["std MISSED"] or any("wrong" in b[0] for b in bad):
        print("FAIL: standard mode is not behaving as complete and optimal CBS.")
    else:
        print(f"PASS: standard mode matched the brute-force optimum on all "
              f"{solvable} solvable cases.")
        print(f"      The stronger vertex rule was worse on {tally['vertex worse']} "
              f"and missed {tally['vertex missed']} of them.")


if __name__ == "__main__":
    main()