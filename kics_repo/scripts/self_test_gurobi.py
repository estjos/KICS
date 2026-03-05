#!/usr/bin/env python
from __future__ import annotations

import random

from kics.ighm import load_ighm_json, IGHM
from kics.params import KrigingParams
from kics.algorithms.sk_gurobi import solve_sk_mclp
from kics.kriging import cic_objective


def main() -> None:
    # small-ish subset so this runs quickly
    ighm = load_ighm_json("data/ighm_1.json")
    ids = ighm.cell_ids
    subset = ids[:40]
    sub = IGHM(df=ighm.df[subset])

    params = KrigingParams(range_=3.0, sill=2.0, nugget=0.0, accept_variance=2.0, delta=1.1)

    k = 5
    radius = 3.0

    res = solve_sk_mclp(sub, k=k, radius=radius, time_limit_sec=30)
    obj, _ = cic_objective(sub, res.samples, params)

    assert len(res.samples) == k, "SK did not return k samples"
    assert obj >= 0, "Objective should be nonnegative"

    print("Gurobi SK sanity check passed.")
    print(f"status={res.solver_status} surrogate_obj={res.objective:.3f} true_obj={obj:.3f}")


if __name__ == "__main__":
    main()
