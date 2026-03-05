from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..ighm import IGHM
from ..kriging import cic_objective
from ..params import KrigingParams


@dataclass(frozen=True)
class SearchResult:
    samples: List[str]
    objective: float
    wall_time_sec: float


def exhaustive_search(
    ighm: IGHM,
    k: int,
    params: KrigingParams,
    seed: int = 0,
    time_limit_sec: Optional[float] = None,
) -> SearchResult:
    """Exact enumeration over all k-subsets (only for tiny instances)."""
    if k <= 0:
        raise ValueError("k must be positive")

    rng = random.Random(seed)
    cells = list(ighm.cell_ids)

    # Generate all combinations and shuffle for early good solutions
    combs = list(itertools.combinations(cells, k))
    rng.shuffle(combs)

    best_obj = float("-inf")
    best = None

    start = time.time()
    for comb in combs:
        if time_limit_sec is not None and (time.time() - start) >= time_limit_sec:
            break

        obj, _ = cic_objective(ighm, comb, params)
        if obj > best_obj:
            best_obj = obj
            best = list(comb)

    end = time.time()
    return SearchResult(samples=best or [], objective=float(best_obj if best is not None else 0.0), wall_time_sec=end - start)


def random_search(
    ighm: IGHM,
    k: int,
    params: KrigingParams,
    seed: int = 0,
    time_limit_sec: float = 5.0,
) -> SearchResult:
    """Time-limited random search baseline."""
    if k <= 0:
        raise ValueError("k must be positive")

    rng = random.Random(seed)
    cells = list(ighm.cell_ids)

    best_obj = float("-inf")
    best = None

    start = time.time()
    while (time.time() - start) < time_limit_sec:
        comb = rng.sample(cells, k)
        obj, _ = cic_objective(ighm, comb, params)
        if obj > best_obj:
            best_obj = obj
            best = list(comb)

    end = time.time()
    return SearchResult(samples=best or [], objective=float(best_obj if best is not None else 0.0), wall_time_sec=end - start)
