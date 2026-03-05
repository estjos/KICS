import math

from kics.params import KrigingParams
from kics.variogram import spherical_semivariogram, bisection_find_h_for_semivariogram


def test_spherical_semivariogram_basic():
    p = KrigingParams(range_=3.0, sill=2.0, nugget=0.0, accept_variance=2.0)
    assert spherical_semivariogram(0.0, p) == 0.0
    assert spherical_semivariogram(100.0, p) == 2.0

    # monotone-ish on [0, range]
    vals = [spherical_semivariogram(h, p) for h in [0.0, 0.5, 1.0, 2.0, 3.0]]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_bisection_inverse():
    p = KrigingParams(range_=4.0, sill=2.0, nugget=0.0, accept_variance=2.0)
    target = 1.0
    h = bisection_find_h_for_semivariogram(target, p)
    # gamma(h) should be close to target
    assert abs(spherical_semivariogram(h, p) - target) < 1e-3
