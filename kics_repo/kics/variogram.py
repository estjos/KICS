from __future__ import annotations

import math

from .params import KrigingParams


def spherical_semivariogram(h: float, params: KrigingParams) -> float:
    """Spherical semivariogram gamma(h) used throughout the paper."""
    if h < 0:
        raise ValueError("h must be nonnegative")

    r = params.range_
    if math.isclose(h, 0.0):
        return params.nugget
    if 0.0 < h <= r:
        return params.nugget + params.sill * (1.5 * (h / r) - 0.5 * ((h ** 3) / (r ** 3)))
    return params.nugget + params.sill


def bisection_find_h_for_semivariogram(target_gamma: float, params: KrigingParams, tol: float = 1e-6) -> float:
    """Invert the spherical semivariogram on [0, range_].

    Returns the largest h in [0, range_] such that gamma(h) <= target_gamma.
    """
    params.validate()
    if target_gamma < params.nugget:
        return 0.0
    if target_gamma >= params.nugget + params.sill:
        return float(params.range_)

    lo, hi = 0.0, float(params.range_)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if spherical_semivariogram(mid, params) <= target_gamma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
