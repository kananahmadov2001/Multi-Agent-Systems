"""Detect temporal conflicts between independently-solved single-agent plans.

Each agent planned alone, so plans can collide when executed in parallel.
Simulation model: plan step i of every agent executes at time i; an agent
whose plan has ended stays in its final cell forever.

Conflict types:
  vertex  - two agents occupy the same cell at the same time
  edge    - two agents swap cells in the same step (a: x->y, b: y->x)

Usage:
  python detect_conflict.py problem-a1.pddl plan-a1.sol problem-a2.pddl plan-a2.sol [...]

Exit code: 0 = no conflict, 1 = at least one conflict, 2 = usage/parse error.
"""
import re
import sys

AT_INIT = re.compile(r"\(at\s+(\S+)\s+(\S+)\)", re.IGNORECASE)
ACTION = re.compile(r"^\((\S+)((?:\s+\S+)*)\)")


def parse_init_positions(problem_path):
    """Return {agent: cell} from the :init section of a problem file."""
    text = open(problem_path, encoding="utf-8").read()
    init = re.search(r":init(.*)\(:goal", text, re.DOTALL | re.IGNORECASE)
    if not init:
        raise ValueError(f"no :init section in {problem_path}")
    positions = {a: c for a, c in AT_INIT.findall(init.group(1))}
    if not positions:
        raise ValueError(f"no (at agent cell) fact in {problem_path}")
    return positions


def parse_plan(plan_path):
    """Return [(action, [args...]), ...] from a Fast Downward plan file."""
    steps = []
    for line in open(plan_path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = ACTION.match(line)
        if not m:
            raise ValueError(f"unparseable plan line in {plan_path}: {line!r}")
        args = m.group(2).split()
        steps.append((m.group(1).lower(), args))
    return steps


def detect(problem_plan_pairs):
    positions = {}   # agent -> current cell
    plans = {}       # agent -> [(action, args), ...]
    for problem, plan in problem_plan_pairs:
        init = parse_init_positions(problem)
        if len(init) != 1:
            raise ValueError(f"{problem}: expected exactly 1 agent, got {init}")
        agent = next(iter(init))
        positions[agent] = init[agent]
        plans[agent] = parse_plan(plan)

    conflicts = []
    horizon = max((len(p) for p in plans.values()), default=0)
    for t in range(1, horizon + 1):
        moves = {}  # agent -> (from_cell, to_cell)
        for agent, plan in plans.items():
            if t <= len(plan):
                action, args = plan[t - 1]
                if action != "move" or len(args) != 3:
                    raise ValueError(f"{agent} step {t}: expected (move a from to), got ({action} {' '.join(args)})")
                _, frm, to = args
                if positions[agent] != frm:
                    raise ValueError(f"{agent} step {t}: plan says from {frm}, but agent is at {positions[agent]}")
                moves[agent] = (frm, to)
                positions[agent] = to

        # vertex conflicts: same cell at time t
        by_cell = {}
        for agent, cell in positions.items():
            by_cell.setdefault(cell, []).append(agent)
        for cell, agents in by_cell.items():
            if len(agents) > 1:
                conflicts.append({"time": t, "type": "vertex", "cell": cell,
                                  "agents": sorted(agents)})

        # edge conflicts: simultaneous swap
        agents = sorted(moves)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if moves[a] == (moves[b][1], moves[b][0]):
                    conflicts.append({"time": t, "type": "edge",
                                      "cells": list(moves[a]), "agents": [a, b]})
    return conflicts


def main(argv):
    if len(argv) < 4 or len(argv) % 2 != 0:
        print(__doc__.strip())
        return 2
    pairs = list(zip(argv[::2], argv[1::2]))
    try:
        conflicts = detect(pairs)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}")
        return 2
    if not conflicts:
        print("NO CONFLICT")
        return 0
    for c in conflicts:
        if c["type"] == "vertex":
            print(f"CONFLICT t={c['time']} vertex cell={c['cell']} agents={' '.join(c['agents'])}")
        else:
            print(f"CONFLICT t={c['time']} edge cells={' '.join(c['cells'])} agents={' '.join(c['agents'])}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
