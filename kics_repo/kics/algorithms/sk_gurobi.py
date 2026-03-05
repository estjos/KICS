from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ighm import IGHM


@dataclass(frozen=True)
class SKResult:
    samples: List[str]
    objective: float
    wall_time_sec: float
    solver_status: str


def _import_gurobi():
    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB  # type: ignore
        return gp, GRB
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "gurobipy is required to solve SK. Install Gurobi + gurobipy and ensure your license is set up."
        ) from e


def neighborhoods_within_radius(ighm: IGHM, radius: float) -> List[List[int]]:
    """Return neighborhood indices for each demand point i.

    Output: hoods[i] = list of facility indices j that can cover i.

    We index cells as 0..n-1 in the order `ighm.cell_ids`.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    ids = ighm.cell_ids
    coords = np.array([ighm.coords(cid) for cid in ids], dtype=float)

    n = len(ids)
    hoods: List[List[int]] = [[] for _ in range(n)]

    # O(n^2) neighborhood build (fine for moderate case-study sizes)
    for i in range(n):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        hoods[i] = [int(j) for j in np.where(dist <= radius)[0]]
    return hoods


def solve_sk_mclp(
    ighm: IGHM,
    k: int,
    radius: float,
    time_limit_sec: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> SKResult:
    """Solve the SK integer program (MCLP form) with Gurobi.

    Model (classic maximal coverage):
      - choose k facilities (samples)
      - maximize covered demand weight (cell gains)

    Parameters
    ----------
    radius
        Coverage radius used in the surrogate (derived from variogram/range and the CIC threshold).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    gp, GRB = _import_gurobi()

    ids = ighm.cell_ids
    n = len(ids)
    gains = [ighm.gain(cid) for cid in ids]
    hoods = neighborhoods_within_radius(ighm, radius)

    start = time.time()
    m = gp.Model("SK_MCLP")

    if not verbose:
        m.setParam("OutputFlag", 0)
    if time_limit_sec is not None:
        m.setParam("TimeLimit", float(time_limit_sec))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    y = m.addVars(n, vtype=GRB.BINARY, name="y")  # facility opened at cell j
    x = m.addVars(n, vtype=GRB.BINARY, name="x")  # demand i covered?

    m.addConstr(gp.quicksum(y[j] for j in range(n)) == k, name="budget")

    for i in range(n):
        m.addConstr(x[i] <= gp.quicksum(y[j] for j in hoods[i]), name=f"cover[{i}]")

    m.setObjective(gp.quicksum(gains[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    m.optimize()

    status = int(m.Status)
    status_name = {
        2: "OPTIMAL",
        9: "TIME_LIMIT",
        3: "INFEASIBLE",
        5: "UNBOUNDED",
    }.get(status, f"STATUS_{status}")

    chosen_idx = [j for j in range(n) if y[j].X >= 0.5]
    chosen_cells = [ids[j] for j in chosen_idx]

    end = time.time()
    return SKResult(samples=chosen_cells, objective=float(m.ObjVal) if m.SolCount > 0 else 0.0, wall_time_sec=end - start, solver_status=status_name)
