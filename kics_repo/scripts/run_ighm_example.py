#!/usr/bin/env python
from __future__ import annotations

import argparse

from kics.ighm import load_ighm_json
from kics.kriging import cic_objective
from kics.params import KrigingParams
from kics.variogram import bisection_find_h_for_semivariogram
from kics.algorithms.sk_gurobi import solve_sk_mclp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to IGHM json")
    ap.add_argument("--k", type=int, default=5, help="Number of samples")
    ap.add_argument("--radius", type=float, default=None, help="Coverage radius for SK")
    ap.add_argument("--auto-radius", action="store_true", help="Compute radius by inverting gamma(h) <= accept")

    ap.add_argument("--range", dest="range_", type=float, default=3.0)
    ap.add_argument("--sill", type=float, default=2.0)
    ap.add_argument("--nugget", type=float, default=0.0)
    ap.add_argument("--accept", dest="accept_variance", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=1.1)

    ap.add_argument("--penalty-mode", choices=["zoom", "suitability"], default="zoom")
    ap.add_argument("--no-range-cutoff", action="store_true", help="Use all samples (no range cutoff) in variance eval")

    args = ap.parse_args()

    params = KrigingParams(range_=args.range_, sill=args.sill, nugget=args.nugget, accept_variance=args.accept_variance, delta=args.delta)
    ighm = load_ighm_json(args.data)

    if args.auto_radius:
        radius = bisection_find_h_for_semivariogram(params.accept_variance, params)
    else:
        if args.radius is None:
            raise SystemExit("Provide --radius or use --auto-radius")
        radius = float(args.radius)

    sk = solve_sk_mclp(ighm, k=args.k, radius=radius)
    obj_true, _ = cic_objective(
        ighm,
        sample_cells=sk.samples,
        params=params,
        penalty_mode=args.penalty_mode,
        use_range_cutoff=not args.no_range_cutoff,
    )

    print("=== SK (Gurobi) ===")
    print(f"status: {sk.solver_status}")
    print(f"radius: {radius}")
    print(f"k: {args.k}")
    print(f"SK surrogate objective: {sk.objective:.6f}")
    print(f"True CIC objective (kriging eval): {obj_true:.6f}")
    print(f"samples: {sk.samples}")


if __name__ == "__main__":
    main()
