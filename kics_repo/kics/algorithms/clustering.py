from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..ighm import IGHM
from ..params import KrigingParams
from ..variogram import bisection_find_h_for_semivariogram, spherical_semivariogram
from .exhaustive import exhaustive_search
from .sk_gurobi import solve_sk_mclp


@dataclass(frozen=True)
class ClusterResult:
    cluster_id: int
    k: int
    samples: List[str]
    objective: float
    wall_time_sec: float


def _import_sklearn_dbscan():
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        return DBSCAN
    except Exception as e:  # pragma: no cover
        raise ImportError("scikit-learn is required for clustering experiments (DBSCAN).") from e


def cluster_cells_dbscan(ighm: IGHM, eps: float = 0.1, min_samples: int = 1) -> List[List[str]]:
    """Cluster IGHM cells using DBSCAN on (x,y) coordinates.

    Returns a list of clusters, each a list of cell ids.
    """
    DBSCAN = _import_sklearn_dbscan()

    ids = ighm.cell_ids
    coords = np.array([ighm.coords(cid) for cid in ids], dtype=float)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(np.radians(coords))
    labels = db.labels_
    clusters: List[List[str]] = []
    for lab in sorted(set(labels)):
        clusters.append([cid for cid, l in zip(ids, labels) if l == lab])
    return clusters


def allocate_samples_proportional(cluster_gains: Sequence[float], total_samples: int) -> List[int]:
    """Simple deterministic allocation: round proportional weights, then fix sum."""
    if total_samples < 0:
        raise ValueError("total_samples must be nonnegative")
    if len(cluster_gains) == 0:
        return []

    gains = np.array(cluster_gains, dtype=float)
    if gains.sum() <= 0:
        # all zero -> uniform
        base = [total_samples // len(cluster_gains)] * len(cluster_gains)
        for i in range(total_samples % len(cluster_gains)):
            base[i] += 1
        return base

    raw = gains / gains.sum() * total_samples
    alloc = np.floor(raw).astype(int)
    # distribute remainder to largest fractional parts
    rem = total_samples - int(alloc.sum())
    frac = raw - alloc
    for idx in np.argsort(-frac)[:rem]:
        alloc[int(idx)] += 1
    return [int(x) for x in alloc]


def solve_clustered(
    ighm: IGHM,
    clusters: Sequence[Sequence[str]],
    total_samples: int,
    params: KrigingParams,
    method: str = "SK",  # 'SK' or 'SS'
    radius: Optional[float] = None,
    time_limit_per_cluster_sec: Optional[float] = None,
    seed: int = 0,
) -> Tuple[List[ClusterResult], List[str]]:
    """Solve per-cluster subproblems and return combined samples.

    This is a helper for the case-study style pipeline. It is intentionally simple:
    it allocates samples proportionally to cluster gain, then solves each cluster
    with either:
      - SK (Gurobi MCLP) or
      - SS (exact enumeration)
    """
    # compute cluster gains
    cluster_gains = []
    for cl in clusters:
        cluster_gains.append(sum(ighm.gain(cid) for cid in cl))

    alloc = allocate_samples_proportional(cluster_gains, total_samples)

    if radius is None:
        # radius for a *single* representative resolution: invert gamma(h) <= accept
        radius = bisection_find_h_for_semivariogram(params.accept_variance, params)

    results: List[ClusterResult] = []
    all_samples: List[str] = []

    for cluster_id, (cl, k) in enumerate(zip(clusters, alloc)):
        if k <= 0:
            continue

        # Build a sub-IGHM view
        df_sub = ighm.df[list(cl)]
        sub = IGHM(df=df_sub)

        start = time.time()
        if method.upper() == "SK":
            sk = solve_sk_mclp(sub, k=k, radius=radius, time_limit_sec=time_limit_per_cluster_sec)
            samples = sk.samples
            obj = sk.objective
        elif method.upper() == "SS":
            ss = exhaustive_search(sub, k=k, params=params, seed=seed + cluster_id, time_limit_sec=time_limit_per_cluster_sec)
            samples = ss.samples
            obj = ss.objective
        else:
            raise ValueError("method must be 'SK' or 'SS'")

        end = time.time()
        results.append(ClusterResult(cluster_id=cluster_id, k=k, samples=samples, objective=float(obj), wall_time_sec=end - start))
        all_samples.extend(samples)

    return results, all_samples
