"""Kriging-informed coverage sampling (KICS).

This package accompanies the paper:
  "Kriging-informed coverage sampling: An Integer Program Approach to Optimizing Spatial Sampling"

It provides:
- Kriging-variance evaluation (used to score candidate sampling plans)
- Exact enumeration baselines for small instances
- The SK (Kriging-informed coverage sampling) integer-program approximation solved with Gurobi
- Heuristics used in the computational experiments (time-limited random search, local search, clustering)

The code is written to be *reproducible*, with explicit random seeds and fully specified inputs.
"""

from .params import KrigingParams
from .ighm import load_ighm_json

__all__ = ["KrigingParams", "load_ighm_json"]
