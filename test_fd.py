"""Check the Fast Downward low-level planner against the BFS one, then run the tree with it."""
import random, sys, time
from cbs_baseline import Grid, plan_single, solve, show
from fd_planner import make_fd_planner

FD_DIR = sys.argv[1] if len(sys.argv) > 1 else "../fast-downward-24.06.1"
fd = make_fd_planner(FD_DIR)

# 1. Same shortest length as BFS on random constraint sets
random.seed(1)
grid = Grid(["...", "...", "..."])
cells = [(r, c) for r in range(3) for c in range(3)]
bad = 0
for i in range(15):
    cons = frozenset((random.choice(cells), random.randint(0, 5)) for _ in range(random.randint(0, 6)))
    s, g = random.sample(cells, 2)
    a, b = plan_single(grid, s, g, cons, 10), fd(grid, s, g, cons, 10)
    same = (a is None and b is None) or (a and b and len(a) == len(b))
    bad += not same
    if not same: print("MISMATCH", s, g, sorted(cons), a, b)
print(f"low-level check: {15 - bad}/15 match BFS")

# 2. The tree, with Fast Downward as the low-level planner
agents = {"a1": ((0, 1), (2, 1)), "a2": ((1, 0), (1, 2))}
fd.calls = 0
t0 = time.time()
show("crossing paths (Fast Downward)", *solve(grid, agents, planner=fd, horizon=8))
print(f"  FD calls: {fd.calls}, {time.time() - t0:.1f}s")

# 3. Backtracking with Fast Downward: corridor with a side pocket
grid = Grid(["...", "#.#"])
agents = {"a1": ((0, 0), (0, 2)), "a2": ((0, 2), (0, 0))}
fd.calls = 0
t0 = time.time()
show("corridor with a pocket (Fast Downward)", *solve(grid, agents, planner=fd, horizon=10))
print(f"  FD calls: {fd.calls}, {time.time() - t0:.1f}s")