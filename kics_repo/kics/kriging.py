from __future__ import annotations

import math
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np

from .ighm import IGHM
from .params import KrigingParams
from .variogram import covariance_from_semivariogram, spherical_semivariogram


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


PenaltyMode = Literal["zoom", "suitability"]


def _sample_penalty_ighm(ighm: IGHM, sample_cell: str, params: KrigingParams, mode: PenaltyMode) -> float:
    """Resolution penalty factor for a sampled cell.

    mode='zoom'
        Uses zoom integers encoded in suitability keys. The chosen zoom is the
        key with max suitability weight; lambda_star is max zoom available.

    mode='suitability'
        Uses the max suitability *value* (in [0,1]) and computes
        delta^(1 - max_suit_value).
    """
    if mode == "zoom":
        zooms = ighm.zoom_levels(sample_cell)
        if not zooms:
            return 1.0
        lambda_star = max(zooms)
        best_key = ighm.best_suitability_key(sample_cell)
        try:
            lambda_q = int(best_key.split("_Z")[-1])
        except ValueError:
            lambda_q = lambda_star
        return params.delta ** (lambda_star - lambda_q)

    max_suit_value = max(ighm.suitability(sample_cell).values())
    return params.delta ** (1.0 - float(max_suit_value))


def kriging_weights_and_lagrange(
    ighm: IGHM,
    target_cell: str,
    sample_cells: Sequence[str],
    params: KrigingParams,
    sill_cov: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Compute ordinary kriging weights for `target_cell` from `sample_cells`.

    Returns
    -------
    weights : ndarray shape (n_samples,)
    lagrange_multiplier : float

    Notes
    -----
    This follows the structure used in the original project code:
      cov(h) = sill_cov - gamma(h)
    and solves the standard ordinary-kriging linear system.
    """
    n = len(sample_cells)
    if n == 0:
        raise ValueError("sample_cells must be non-empty")

    # Build (n+1)x(n+1) matrix
    M = np.zeros((n + 1, n + 1), dtype=float)
    b = np.zeros((n + 1,), dtype=float)

    # covariance among samples
    for i, si in enumerate(sample_cells):
        for j, sj in enumerate(sample_cells):
            h = euclidean(ighm.coords(si), ighm.coords(sj))
            M[i, j] = covariance_from_semivariogram(h, params, sill_cov=sill_cov)
        M[i, n] = 1.0
        h_it = euclidean(ighm.coords(si), ighm.coords(target_cell))
        b[i] = covariance_from_semivariogram(h_it, params, sill_cov=sill_cov)

    # unbiasedness row/col
    M[n, :n] = 1.0
    M[n, n] = 0.0
    b[n] = 1.0

    sol = np.linalg.solve(M, b)
    weights = sol[:n]
    lag = float(sol[n])
    return weights, lag


def kriging_prediction_variance(
    ighm: IGHM,
    target_cell: str,
    sample_cells: Sequence[str],
    params: KrigingParams,
    penalty_mode: PenaltyMode = "zoom",
    use_range_cutoff: bool = True,
) -> float:
    """Compute kriging-based prediction variance for one cell.

    The implementation matches the project code used to generate paper results:
    - Only samples within `params.range_` are used when `use_range_cutoff=True`.
    - Variance is computed as sum_i w_i * gamma(h_{i,target}) * penalty(sample_i).

    If no samples are within range, a large variance is returned.
    """
    params.validate()

    if use_range_cutoff:
        in_range = [s for s in sample_cells if euclidean(ighm.coords(s), ighm.coords(target_cell)) <= params.range_]
    else:
        in_range = list(sample_cells)

    if len(in_range) == 0:
        return 1e6

    weights, _ = kriging_weights_and_lagrange(ighm, target_cell, in_range, params)
    var = 0.0
    for w, s in zip(weights, in_range):
        h = euclidean(ighm.coords(s), ighm.coords(target_cell))
        var += float(w) * spherical_semivariogram(h, params) * _sample_penalty_ighm(ighm, s, params, mode=penalty_mode)
    return float(var)


def kriging_variance_surface(
    ighm: IGHM,
    sample_cells: Sequence[str],
    params: KrigingParams,
    penalty_mode: PenaltyMode = "zoom",
    use_range_cutoff: bool = True,
) -> List[float]:
    """Return kriging variance values for all cells in `ighm` in cell-id order."""
    return [
        kriging_prediction_variance(
            ighm,
            target_cell=cid,
            sample_cells=sample_cells,
            params=params,
            penalty_mode=penalty_mode,
            use_range_cutoff=use_range_cutoff,
        )
        for cid in ighm.cell_ids
    ]


def cic_objective(
    ighm: IGHM,
    sample_cells: Sequence[str],
    params: KrigingParams,
    penalty_mode: PenaltyMode = "zoom",
    use_range_cutoff: bool = True,
) -> Tuple[float, List[int]]:
    """Compute CIC objective (sum of gains of confidently covered cells).

    Returns objective value and a 0/1 coverage indicator list aligned with `ighm.cell_ids`.
    """
    variances = kriging_variance_surface(
        ighm,
        sample_cells=sample_cells,
        params=params,
        penalty_mode=penalty_mode,
        use_range_cutoff=use_range_cutoff,
    )

    covered = [1 if v <= params.accept_variance else 0 for v in variances]
    obj = 0.0
    for cid, c in zip(ighm.cell_ids, covered):
        if c:
            obj += ighm.gain(cid)
    return float(obj), covered
