#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import math
import multiprocessing as mp
import random
import time
from dataclasses import asdict
from typing import List, Sequence

import pandas as pd

from kics.grid import def_interest, def_sat1, def_sat2, def_sat3, grid_exhaustive_search, grid_random_search, grid_cic_objective
from kics.params import KrigingParams
from kics.variogram import bisection_find_h_for_semivariogram


def _import_gurobi():
    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB, LinExpr  # type: ignore
        return gp, GRB, LinExpr
    except Exception as e:
        raise ImportError("This script requires gurobipy. Install Gurobi + gurobipy and ensure your license is set up.") from e


def _find_acceptablevar_given_res(accept: float, max_res: int, res: int, delta: float) -> float:
    return accept / (delta ** (max_res - res))


def solve_sk_grid_gurobi(interest, satellites, per_satellite_samples, resolutions, params: KrigingParams):
    """Grid SK model from the project scripts (multi-satellite, multi-resolution)."""
    gp, GRB, LinExpr = _import_gurobi()

    rows, cols = len(interest), len(interest[0])
    start = time.time()

    # per-resolution neighborhood radii d[lam]
    d = [
        math.floor(
            bisection_find_h_for_semivariogram(
                _find_acceptablevar_given_res(params.accept_variance, max_res=3, res=res, delta=params.delta),
                params,
            )
        )
        for res in resolutions
    ]

    # feasibility / resolution indicator l_param and visibility v
    l_param = [[[[0 for _ in range(len(resolutions))] for _ in range(len(satellites))] for _ in range(cols)] for _ in range(rows)]
    v = [[[0 for _ in range(len(satellites))] for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            for j, sat in enumerate(satellites):
                for lam, lam_res in enumerate(resolutions):
                    if sat[r][c] == lam_res:
                        l_param[r][c][j][lam] = 1
                        v[r][c][j] = 1

    m = gp.Model("SK_grid")
    m.setParam("OutputFlag", 0)

    y = m.addVars(rows, cols, vtype=GRB.BINARY, name="y")
    x = m.addVars(rows, cols, len(satellites), vtype=GRB.BINARY, name="x")
    z = m.addVars(rows, cols, len(resolutions), vtype=GRB.BINARY, name="z")

    # x only if visible
    for r in range(rows):
        for c in range(cols):
            for j in range(len(satellites)):
                m.addConstr(x[r, c, j] <= v[r][c][j])

    # per-sat budgets
    for j in range(len(satellites)):
        m.addConstr(gp.quicksum(x[r, c, j] for r in range(rows) for c in range(cols)) == per_satellite_samples[j])

    # at most one satellite samples each cell
    for r in range(rows):
        for c in range(cols):
            m.addConstr(gp.quicksum(x[r, c, j] for j in range(len(satellites))) <= 1)

    # z[k,l,lam] indicates k,l covered at resolution lam by at least one in-neighborhood sample
    for k in range(rows):
        for l in range(cols):
            for lam in range(len(resolutions)):
                expr = LinExpr()
                for j in range(len(satellites)):
                    for r in range(max(k - d[lam], 0), min(k + 1 + d[lam], rows)):
                        for c in range(max(l - d[lam], 0), min(l + 1 + d[lam], cols)):
                            expr.addTerms(l_param[r][c][j][lam], x[r, c, j])
                m.addConstr(z[k, l, lam] <= expr)

    # y <= sum_lam z
    for r in range(rows):
        for c in range(cols):
            m.addConstr(y[r, c] <= gp.quicksum(z[r, c, lam] for lam in range(len(resolutions))))

    m.setObjective(gp.quicksum(interest[r][c] * y[r, c] for r in range(rows) for c in range(cols)), GRB.MAXIMIZE)
    m.optimize()

    opt_samples = []
    for var in m.getVars():
        if var.X > 0.5 and var.VarName.startswith("x"):
            # VarName looks like x[<r>,<c>,<j>]
            idx = ast.literal_eval(var.VarName.replace("x", ""))
            opt_samples.append([int(idx[0]), int(idx[1]), int(idx[2])])

    end = time.time()
    return opt_samples, float(m.ObjVal) if m.SolCount > 0 else 0.0, end - start


def local_search_grid(interest, satellites, samples, params: KrigingParams, ls_radius: int = 2):
    """Single-sample-at-a-time local search improvement."""
    rows, cols = len(interest), len(interest[0])
    best = [list(s) for s in samples]
    best_obj = grid_cic_objective(interest, best, satellites, params)

    for idx, s in enumerate(list(best)):
        r0, c0, j = s
        for r in range(r0 - ls_radius, r0 + ls_radius + 1):
            for c in range(c0 - ls_radius, c0 + ls_radius + 1):
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                if satellites[j][r][c] <= 0:
                    continue
                if (r, c) in {(x[0], x[1]) for k, x in enumerate(best) if k != idx}:
                    continue
                cand = [list(x) for x in best]
                cand[idx] = [r, c, j]
                obj = grid_cic_objective(interest, cand, satellites, params)
                if obj > best_obj:
                    best_obj = obj
                    best = cand
    return best, best_obj


def run_one(grid_size: int, rep: int, params: KrigingParams):
    satellites = [def_sat1(grid_size), def_sat2(grid_size), def_sat3(grid_size)]
    interest = def_interest(grid_size, seed=1000 * grid_size + rep)
    budgets = [1, 1, 1]

    out = {
        "grid_size": grid_size,
        "rep": rep,
    }

    # exact only for very small g
    if grid_size <= 6:
        ss = grid_exhaustive_search(interest, satellites, budgets, params)
        out.update({"SS_obj": ss.objective, "SS_time": ss.wall_time_sec, "SS_samples": ss.samples})

    rs1 = grid_random_search(interest, satellites, budgets, params, time_limit_sec=1 * grid_size, seed=rep)
    rs3 = grid_random_search(interest, satellites, budgets, params, time_limit_sec=3 * grid_size, seed=rep)
    out.update({
        "RS1_obj": rs1.objective,
        "RS1_time": rs1.wall_time_sec,
        "RS3_obj": rs3.objective,
        "RS3_time": rs3.wall_time_sec,
    })

    sk_samples, sk_sur_obj, sk_time = solve_sk_grid_gurobi(interest, satellites, budgets, [1, 2, 3], params)
    sk_true_obj = grid_cic_objective(interest, sk_samples, satellites, params)

    ls_samples, ls_obj = local_search_grid(interest, satellites, sk_samples, params)

    out.update({
        "SK_sur_obj": sk_sur_obj,
        "SK_true_obj": sk_true_obj,
        "SK_time": sk_time,
        "SK_samples": sk_samples,
        "SK_LS_obj": ls_obj,
        "SK_LS_samples": ls_samples,
    })

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-sizes", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--processes", type=int, default=0, help="0 -> no multiprocessing")
    ap.add_argument("--out", type=str, default="grid_results.csv")

    ap.add_argument("--range", dest="range_", type=float, default=4.0)
    ap.add_argument("--sill", type=float, default=2.0)
    ap.add_argument("--nugget", type=float, default=0.0)
    ap.add_argument("--accept", dest="accept_variance", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=1.1)

    args = ap.parse_args()
    params = KrigingParams(range_=args.range_, sill=args.sill, nugget=args.nugget, accept_variance=args.accept_variance, delta=args.delta)

    jobs = [(g, r, params) for g in args.grid_sizes for r in range(args.reps)]

    if args.processes and args.processes > 0:
        with mp.Pool(processes=args.processes) as pool:
            results = pool.starmap(run_one, jobs)
    else:
        results = [run_one(*j) for j in jobs]

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
