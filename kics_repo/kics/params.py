from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KrigingParams:
    """Parameters controlling the semivariogram and CIC thresholding.

    Notes
    -----
    The spherical semivariogram is implemented as:
        gamma(h) = nugget,                           if h = 0
        gamma(h) = nugget + sill*( 1.5*(h/r) - 0.5*(h^3/r^3) ),  if 0 < h <= r
        gamma(h) = nugget + sill,                    if h > r

    Here `sill` is the *structured* component (often (s - nugget) in geostatistics).

    accept_variance
        The CIC threshold (epsilon_0 in the paper): a location is "confidently covered"
        if its kriging prediction variance is <= accept_variance.

    delta
        Resolution penalty (>1). Used as delta^(lambda_star - lambda).
    """

    range_: float
    sill: float
    nugget: float
    accept_variance: float
    delta: float = 1.1

    def validate(self) -> None:
        if self.range_ <= 0:
            raise ValueError("range_ must be positive")
        if self.sill < 0:
            raise ValueError("sill must be nonnegative")
        if self.nugget < 0:
            raise ValueError("nugget must be nonnegative")
        if self.accept_variance <= 0:
            raise ValueError("accept_variance must be positive")
        if self.delta <= 1:
            raise ValueError("delta must be > 1")
