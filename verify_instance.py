#!/usr/bin/env python3
"""Verify finite lower-bound set-system instances.

1. feasibility of the metric constraints for a fixed alpha,
2. binary search over alpha,
3. feasibility of nonnegative weights certifying set coverage.

"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import TextIO

import gurobipy as gp
from gurobipy import GRB

from instance_io import Instance, read_instance_csv


# --------------------------------------------------------------------------- #
#  LP feasibility check                                                        #
# --------------------------------------------------------------------------- #


def check_feasibility(
    alpha: float,
    instance: Instance,
    time_limit: float | None = None,
    distance_ub: float = 1e6,
) -> tuple[bool, dict | None, bool]:
    """Return ``(feasible, solution, timed_out)`` for a fixed alpha."""
    k = instance.k
    agent_sets = instance.agent_sets
    n_agents = len(agent_sets)
    n_pts = k + n_agents
    nodes = [f"c{t}" for t in range(k)] + [f"a{j}" for j in range(n_agents)]

    model = gp.Model("alpha_lp")
    model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # Distances between distinct points. The diagonal is implicitly 0.
    d = {}
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            var = model.addVar(
                lb=1.0,
                ub=distance_ub,
                vtype=GRB.CONTINUOUS,
                name=f"d_{nodes[i]}_{nodes[j]}",
            )
            d[i, j] = var
            d[j, i] = var

    # Triangle inequalities.
    for a, b, c in combinations(range(n_pts), 3):
        model.addConstr(d[a, b] <= d[a, c] + d[c, b])
        model.addConstr(d[a, c] <= d[a, b] + d[b, c])
        model.addConstr(d[b, c] <= d[b, a] + d[a, c])

    # Lower-bound constraints for every agent/set membership.
    for agent, sets in enumerate(agent_sets):
        p_agent = k + agent
        for t in sorted(sets):
            if not 0 <= t < k:
                raise ValueError(f"invalid set index {t}; expected 0 <= t < {k}")
            t_prev = (t - 1) % k
            model.addConstr(
                d[p_agent, t_prev] >= alpha * d[p_agent, t] + 1.0,
                name=f"alpha_agent{agent}_set{t}",
            )

    model.optimize()

    timed_out = model.Status == GRB.TIME_LIMIT
    found_feasible = model.Status == GRB.OPTIMAL or (timed_out and model.SolCount > 0)
    if not found_feasible:
        return False, None, timed_out

    return True, extract_solution(instance, nodes, d), timed_out


def extract_solution(instance: Instance, nodes: list[str], d: dict) -> dict:
    k = instance.k
    n_agents = len(instance.agent_sets)
    n_pts = k + n_agents
    sol = {
        "agent_sets": instance.agent_sets,
        "nodes": nodes,
        "k": k,
        "n_agents": n_agents,
        "d_sets": {},
        "d_agent_center": {},
        "d_all": {},
    }

    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            sol["d_all"][(i, j)] = d[i, j].X

    for i in range(k):
        for j in range(i + 1, k):
            sol["d_sets"][(i, j)] = d[i, j].X

    for agent in range(n_agents):
        p_agent = k + agent
        sol["d_agent_center"][agent] = {t: d[p_agent, t].X for t in range(k)}

    return sol


# --------------------------------------------------------------------------- #
#  Binary search                                                               #
# --------------------------------------------------------------------------- #


def binary_search_alpha(
    instance: Instance,
    lo: float = 1.0,
    hi: float = 3.0,
    tol: float = 1e-6,
    max_iter: int = 80,
    time_limit: float = 300,
    distance_ub: float = 1e6,
) -> tuple[float | None, dict | None]:
    """Find the maximum alpha for which the metric LP is feasible."""
    print(f"Binary search for max feasible alpha in [{lo}, {hi}]")
    print(f"  instance = {instance.name}")
    print(f"  k = {instance.k},  {len(instance.agent_sets)} agents")
    print("=" * 65)

    feas_lo, sol_lo, to_lo = check_feasibility(
        lo, instance, time_limit=time_limit, distance_ub=distance_ub
    )
    if not feas_lo:
        msg = "TIMEOUT" if to_lo else "Infeasible"
        print(f"  {msg} already at alpha = {lo:.6f} -- aborting.")
        return None, None

    best_alpha, best_sol = lo, sol_lo

    feas_hi, sol_hi, to_hi = check_feasibility(
        hi, instance, time_limit=time_limit, distance_ub=distance_ub
    )
    if feas_hi:
        print(f"  Feasible even at alpha = {hi:.6f} -- widen the range.")
        return hi, sol_hi
    if to_hi:
        print(f"  Timed out at alpha = {hi:.6f} -- aborting.")
        return None, None

    for it in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        feasible, sol, timed_out = check_feasibility(
            mid, instance, time_limit=time_limit, distance_ub=distance_ub
        )
        if timed_out and not feasible:
            print(f"  iter {it:3d}:  alpha = {mid:.8f}   TIMEOUT -- aborting.")
            return None, None

        tag = "FEASIBLE  " if feasible else "infeasible"
        print(f"  iter {it:3d}:  alpha = {mid:.8f}   {tag}   [{lo:.8f}, {hi:.8f}]")

        if feasible:
            lo = mid
            best_alpha = mid
            best_sol = sol
        else:
            hi = mid

    print("=" * 65)
    print(f"  Max alpha approximately {best_alpha:.8f}")
    return best_alpha, best_sol


# --------------------------------------------------------------------------- #
#  Weights                                                                     #
# --------------------------------------------------------------------------- #


def find_weights(instance: Instance, eps: float = 1e-6) -> list[float] | None:
    """Find weights with sum 1 and set coverage at least 0.5 + eps."""
    k = instance.k
    agent_sets = instance.agent_sets
    n_agents = len(agent_sets)

    model = gp.Model("find_weights")
    model.setParam("OutputFlag", 0)
    model.setParam("FeasibilityTol", 1e-9)

    w = model.addVars(n_agents, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")
    model.addConstr(gp.quicksum(w[j] for j in range(n_agents)) == 1.0, name="sum_to_one")

    for t in range(k):
        covering_agents = [j for j, sets in enumerate(agent_sets) if t in sets]
        if not covering_agents:
            print(f"  Warning: C_{t} is not covered by any agent -- infeasible.")
            return None
        model.addConstr(
            gp.quicksum(w[j] for j in covering_agents) >= 0.5 + eps,
            name=f"coverage_{t}",
        )

    model.optimize()
    if model.Status == GRB.OPTIMAL:
        return [w[j].X for j in range(n_agents)]
    return None


def compute_coverage(instance: Instance, weights: list[float]) -> list[float]:
    return [
        sum(weight for j, weight in enumerate(weights) if t in instance.agent_sets[j])
        for t in range(instance.k)
    ]


# --------------------------------------------------------------------------- #
#  Printing                                                                    #
# --------------------------------------------------------------------------- #


def print_instance(instance: Instance) -> None:
    print(f"\nInstance: {instance.name}")
    print(f"  k = {instance.k},  {len(instance.agent_sets)} agents")
    print("\n  Agent memberships:")
    for j, sets in enumerate(instance.agent_sets):
        print(f"    agent {j:2d}  sets={str(sorted(sets)):>25s}")


def print_solution(sol: dict | None, alpha: float | None = None, include_distances: bool = False) -> None:
    if sol is None:
        print("  No feasible solution to display.")
        return

    if alpha is not None:
        print(f"\n  Solution at alpha = {alpha:.8f}")
    print(f"  k = {sol['k']},  {sol['n_agents']} agents")

    if not include_distances:
        return

    print("\n  Center-to-center distances:")
    for (i, j), val in sorted(sol["d_sets"].items()):
        print(f"    d(c{i}, c{j}) = {val:.4f}")

    print("\n  Agent-to-center distances:")
    for agent in range(sol["n_agents"]):
        dists = "  ".join(
            f"c{t}: {sol['d_agent_center'][agent][t]:9.4f}" for t in range(sol["k"])
        )
        print(f"    a{agent:2d}:  {dists}")


def write_distances_csv(sol: dict | None, path: str | Path, alpha: float | None = None) -> None:
    """Write the feasible distance solution as a full distance matrix."""
    if sol is None:
        raise ValueError("cannot write distances: no feasible solution is available")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes = sol["nodes"]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + nodes)

        for u_idx, u in enumerate(nodes):
            row = [u]
            for v_idx, v in enumerate(nodes):
                if u_idx == v_idx:
                    row.append("0")
                else:
                    i, j = sorted((u_idx, v_idx))
                    row.append(f"{sol['d_all'][(i, j)]:.12g}")
            writer.writerow(row)


def print_weights(instance: Instance, weights: list[float]) -> None:
    print(f"\n  k = {instance.k},  {len(instance.agent_sets)} agents")
    print("\n  Weights:")
    for j, (sets, weight) in enumerate(zip(instance.agent_sets, weights)):
        print(f"    agent {j:2d}  sets={str(sorted(sets)):>25s}   w = {weight:.8f}")

    coverage = compute_coverage(instance, weights)
    print("\n  Set coverage:")
    for t, total in enumerate(coverage):
        flag = "OK" if total > 0.5 else "FAIL"
        print(f"    C_{t}:  {total:.8f}  ({flag})")

    print(f"\n  Sum of weights: {sum(weights):.8f}")
    print(f"  Minimum coverage: {min(coverage):.8f}")


# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #


class Tee:
    def __init__(self, *files: TextIO):
        self.files = files

    def write(self, data: str) -> int:
        for f in self.files:
            f.write(data)
            f.flush()
        return len(data)

    def flush(self) -> None:
        for f in self.files:
            f.flush()


@contextmanager
def optional_log_file(path: str | Path | None):
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    with path.open("w") as f:
        sys.stdout = Tee(old_stdout, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a finite lower-bound set-system instance.")
    parser.add_argument(
        "--instance",
        default="./instances/metric37_wide.csv",
        help="Path to an agent,set membership CSV. Default: ./instances/metric37.csv",
    )
    parser.add_argument("--k", type=int, default=None, help="Number of centers. Inferred if omitted.")
    parser.add_argument("--lo", type=float, default=1.0, help="Lower end of alpha search interval.")
    parser.add_argument("--hi", type=float, default=2.42, help="Upper end of alpha search interval.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Binary-search tolerance.")
    parser.add_argument("--max-iter", type=int, default=80, help="Maximum binary-search iterations.")
    parser.add_argument("--time-limit", type=float, default=300, help="Gurobi time limit per LP solve.")
    parser.add_argument("--distance-ub", type=float, default=1e6, help="Upper bound for distance variables.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Coverage slack: coverage >= 0.5 + eps.")
    parser.add_argument("--skip-alpha-search", action="store_true", help="Skip metric LP binary search.")
    parser.add_argument("--skip-weights", action="store_true", help="Skip weight computation.")
    parser.add_argument("--print-distances", action="store_true", help="Print full distance solution.")
    parser.add_argument(
        "--distances-csv",
        default=None,
        help="Optional path to write feasible distances as CSV.",
    )
    parser.add_argument("--log-file", default=None, help="Optional path for a copy of stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = read_instance_csv(args.instance, k=args.k)

    with optional_log_file(args.log_file):
        print_instance(instance)

        best_alpha = None
        best_sol = None
        if not args.skip_alpha_search:
            best_alpha, best_sol = binary_search_alpha(
                instance,
                lo=args.lo,
                hi=args.hi,
                tol=args.tol,
                max_iter=args.max_iter,
                time_limit=args.time_limit,
                distance_ub=args.distance_ub,
            )
            if best_alpha is None:
                print("\nAlpha search did not produce a feasible certificate.")
            else:
                print_solution(best_sol, alpha=best_alpha, include_distances=args.print_distances)
                if args.distances_csv is not None:
                    write_distances_csv(best_sol, args.distances_csv, alpha=best_alpha)
                    print(f"\n  Wrote distances CSV to {args.distances_csv}")

        if not args.skip_weights:
            print(f"\nFinding weights for k={instance.k}, {len(instance.agent_sets)} agents ...")
            weights = find_weights(instance, eps=args.eps)
            if weights is None:
                print("  Infeasible -- no valid weight assignment exists.")
            else:
                print("  Feasible!")
                print_weights(instance, weights)


if __name__ == "__main__":
    main()
