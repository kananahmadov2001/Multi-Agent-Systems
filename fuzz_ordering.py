"""Fuzz Kaidi's solve_ordered: is it sound, and how often does it give up?

Soundness is checked with a replay written here, not with ordering.py's own
service_times, so a bug in the detector cannot pass itself.

Give-ups are then put to a brute-force joint oracle, which says whether an
ordered plan existed. That separates "incomplete, as documented" from "wrong".
"""
import argparse
import heapq
import random
import sys

from agent_pool import Task
from cbs_baseline import Grid
from ordering import solve_ordered
from verify import verify

HORIZON = 16


def make(rng, size=4, walls=0.15, sites=2):
    grid = Grid(["".join("#" if rng.random() < walls else "." for _ in range(size))
                 for _ in range(size)])
    free = [(r, c) for r in range(size) for c in range(size) if grid.free((r, c))]
    need = 2 * (1 + sites)
    if len(free) < need:
        return None
    pick = rng.sample(free, need)
    agents = {"a1": (pick[0], pick[1:1 + sites]),
              "a2": (pick[1 + sites], pick[2 + sites:])}
    parts = {"a1": [f"p{i+1}" for i in range(sites)],
             "a2": [f"q{i+1}" for i in range(sites)]}
    if sites == 1:
        order = rng.choice([("p1", "q1"), ("q1", "p1")])
    else:
        order = rng.choice([("p1", "q1", "p2", "q2"), ("q1", "p1", "q2", "p2"),
                            ("p1", "q1", "q2", "p2"), ("q1", "p1", "p2", "q2")])
    task = Task("ordered", set(order), order)
    return grid, agents, parts, task


def replay(paths, agents, parts):
    """Service times, computed independently of ordering.py."""
    when = {}
    for a in sorted(agents):
        route, labels, k = agents[a][1], parts[a], 0
        for t, cell in enumerate(paths[a]):
            if k < len(route) and cell == route[k]:
                when[labels[k]] = t
                k += 1
        if k < len(route):
            return None
    return when


def oracle(grid, agents, parts, task, horizon=HORIZON):
    """Cheapest ordered, collision-free joint plan, or None. Exponential."""
    names = sorted(agents)
    route = {n: list(agents[n][1]) for n in names}
    label = {n: parts[n] for n in names}
    rank = {p: i for i, p in enumerate(task.order)}
    done = tuple(len(route[n]) for n in names)

    def s0(k):
        n = names[k]
        return 1 if agents[n][0] == route[n][0] else 0

    start = (tuple(agents[n][0] for n in names), tuple(s0(k) for k in range(len(names))),
             (False,) * len(names), ())        # () = labels served so far, in order
    pq, best = [(0, 0, start)], {}
    while pq:
        cost, t, st = heapq.heappop(pq)
        cells, served, stopped, seq = st
        if all(stopped):
            return cost
        if t > horizon:
            continue
        if best.get((st, t), 1 << 30) < cost:
            continue

        opts = []
        for k, n in enumerate(names):
            if stopped[k]:
                opts.append([(cells[k], served[k], True, None)])
                continue
            here = []
            if served[k] == done[k] and cells[k] == route[n][-1]:
                here.append((cells[k], served[k], True, None))
            if t < horizon:
                for nx in list(grid.neighbors(cells[k])) + [cells[k]]:
                    adv, got = served[k], None
                    if adv < done[k] and nx == route[n][adv]:
                        got = label[n][adv]
                        adv += 1
                    here.append((nx, adv, False, got))
            opts.append(here)

        def go(k, acc):
            if k == len(names):
                nc = tuple(x[0] for x in acc)
                ns = tuple(x[1] for x in acc)
                nst = tuple(x[2] for x in acc)
                got = [x[3] for x in acc if x[3]]
                if len(set(nc)) != len(nc):
                    return
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        if nc[i] == cells[j] and nc[j] == cells[i]:
                            return
                if len(got) > 1:
                    return                     # two parts served in one step: a tie
                nseq = seq + tuple(got)
                if any(rank[nseq[i]] >= rank[nseq[i + 1]] for i in range(len(nseq) - 1)):
                    return                     # out of order
                step = sum(0 if x else 1 for x in nst)
                nxt = (nc, ns, nst, nseq)
                c = cost + step
                if best.get((nxt, t + 1), 1 << 30) > c:
                    best[(nxt, t + 1)] = c
                    heapq.heappush(pq, (c, t + 1, nxt))
                return
            for o in opts[k]:
                go(k + 1, acc + [o])

        go(0, [])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repairs", type=int, default=8)
    ap.add_argument("--oracle", action="store_true", help="adjudicate the give-ups")
    ap.add_argument("--sites", type=int, default=2, help="waypoints per agent")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    unsound = solved = gaveup = 0
    recoverable, truly_infeasible, oracle_skipped = 0, 0, 0
    reasons = {}
    cost_gap = []

    made = 0
    while made < a.n:
        got = make(rng, sites=a.sites)
        if got is None:
            continue
        grid, agents, parts, task = got
        made += 1
        paths, stats = solve_ordered(grid, agents, task, parts,
                                     horizon=HORIZON, max_nodes=2000,
                                     max_repairs=a.repairs)
        if paths is not None:
            solved += 1
            ok, probs = verify(grid, agents, paths)
            when = replay(paths, agents, parts)
            bad = probs if not ok else []
            if when is None:
                bad.append("a part is never served")
            else:
                for x, y in zip(task.order, task.order[1:]):
                    if when[x] >= when[y]:
                        bad.append(f"{x} at {when[x]} not before {y} at {when[y]}")
            if bad:
                unsound += 1
                if unsound <= 4:
                    print(f"  UNSOUND: {bad[:2]}  order={task.order}")
            elif a.oracle:
                want = oracle(grid, agents, parts, task)
                mine = sum(len(p) - 1 for p in paths.values())
                if want is not None and mine > want:
                    cost_gap.append(mine - want)
        else:
            gaveup += 1
            why = stats.get("ordering_error", "no plan")
            key = ("exhausted" if "exhausted" in why else
                   "replan" if "replan" in why else
                   "spatial" if "spatial" in why else why[:40])
            reasons[key] = reasons.get(key, 0) + 1
            if a.oracle:
                want = oracle(grid, agents, parts, task)
                if want is None:
                    truly_infeasible += 1
                else:
                    recoverable += 1
            else:
                oracle_skipped += 1

    print(f"\n{made} instances, {a.sites} site(s) per agent, max_repairs={a.repairs}, horizon={HORIZON}\n")
    print(f"  solved                {solved}")
    print(f"  gave up               {gaveup}   {reasons}")
    print(f"  UNSOUND (bad plan)    {unsound}")
    if a.oracle:
        print(f"\n  of the {gaveup} give-ups:")
        print(f"    no ordered plan exists      {truly_infeasible}  (correct to give up)")
        print(f"    an ordered plan DID exist   {recoverable}  (incompleteness, as documented)")
        if cost_gap:
            print(f"\n  suboptimal solves: {len(cost_gap)}, total extra cost {sum(cost_gap)}, "
                  f"worst {max(cost_gap)}")
        else:
            print("\n  suboptimal solves: 0")
    print("\n" + ("SOUND: never returned a plan that breaks the order or collides"
                  if unsound == 0 else "UNSOUND — see above"))


if __name__ == "__main__":
    sys.exit(main())