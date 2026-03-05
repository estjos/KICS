from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .params import KrigingParams
from .variogram import spherical_semivariogram


Grid = List[List[int]]  # interest grid or satellite-resolution grid
Sample = List[int]      # [row, col, sat_idx]


def def_sat1(grid: int) -> Grid:
    sat = [[0 for _ in range(grid)] for _ in range(grid)]
    for i in range(math.floor(0.7 * grid)):
        for j in range(math.floor(0.7 * grid) - i):
            sat[i][-j - 1] = 1
    for i in range(math.floor(0.5 * grid)):
        for j in range(math.floor(0.5 * grid) - i):
            sat[i][-j - 1] = 2
    for i in range(math.floor(0.3 * grid)):
        for j in range(math.floor(0.3 * grid) - i):
            sat[i][-j - 1] = 3
    return sat


def def_sat2(grid: int) -> Grid:
    sat = [[0 for _ in range(grid)] for _ in range(grid)]
    for i in range(math.floor(grid * 0.5), grid):
        for j in range(math.floor(grid * 0.5)):
            sat[i][j] = 1
    for i in range(math.floor(grid * 0.6), grid):
        for j in range(math.floor(grid * 0.4)):
            sat[i][j] = 2
    for i in range(math.floor(grid * 0.8), grid):
        for j in range(math.floor(grid * 0.2)):
            sat[i][j] = 3
    return sat


def def_sat3(grid: int) -> Grid:
    sat = [[0 for _ in range(grid)] for _ in range(grid)]
    for i in range(math.floor(grid * 0.3), math.floor(grid * 0.7)):
        for j in range(math.floor(grid * 0.3), math.floor(grid * 0.7)):
            sat[i][j] = 1
    for i in range(math.floor(grid * 0.36), math.floor(grid * 0.64)):
        for j in range(math.floor(grid * 0.36), math.floor(grid * 0.64)):
            sat[i][j] = 2
    for i in range(math.floor(grid * 0.42), math.floor(grid * 0.58)):
        for j in range(math.floor(grid * 0.42), math.floor(grid * 0.58)):
            sat[i][j] = 3
    return sat


def def_interest(grid: int, seed: int = 0) -> Grid:
    rng = random.Random(seed)
    interest = [[0 for _ in range(grid)] for _ in range(grid)]
    n_rect = rng.randint(math.floor(grid * 0.5), grid)
    for _ in range(n_rect):
        k = rng.randint(math.floor(grid * 0.3), math.floor(grid * 0.5))
        l = rng.randint(math.floor(grid * 0.3), math.floor(grid * 0.5))
        x = rng.randint(0, grid - 1)
        y = rng.randint(0, grid - 1)
        level = rng.randint(1, 3)
        for i in range(x, x + k):
            for j in range(y, y + l):
                if i < grid and j < grid:
                    interest[i][j] = level
    return interest


def euclidean_grid(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _grid_variance_at_cell(
    target: Tuple[int, int],
    samples: Sequence[Sample],
    satellites: Sequence[Grid],
    params: KrigingParams,
    max_res: int = 3,
) -> float:
    in_range = [s for s in samples if euclidean_grid((s[0], s[1]), target) <= params.range_]
    if not in_range:
        return 1e6

    pts = [(s[0], s[1]) for s in in_range]
    sat_idx = [s[2] for s in in_range]

    if len(pts) == 1:
        h = euclidean_grid(pts[0], target)
        res = satellites[sat_idx[0]][pts[0][0]][pts[0][1]]
        return spherical_semivariogram(h, params) * (params.delta ** (max_res - res))

    n = len(pts)
    M = np.zeros((n + 1, n + 1), dtype=float)
    b = np.zeros((n + 1,), dtype=float)

    for i, pi in enumerate(pts):
        for j, pj in enumerate(pts):
            M[i, j] = 1.0 - spherical_semivariogram(euclidean_grid(pi, pj), params)
        M[i, n] = 1.0
        b[i] = 1.0 - spherical_semivariogram(euclidean_grid(pi, target), params)

    M[n, :n] = 1.0
    M[n, n] = 0.0
    b[n] = 1.0

    sol = np.linalg.solve(M, b)
    weights = sol[:n]

    var = 0.0
    for w, pi, si in zip(weights, pts, sat_idx):
        h = euclidean_grid(pi, target)
        res = satellites[si][pi[0]][pi[1]]
        var += float(w) * spherical_semivariogram(h, params) * (params.delta ** (max_res - res))
    return float(var)


def grid_cic_objective(
    interest: Grid,
    samples: Sequence[Sample],
    satellites: Sequence[Grid],
    params: KrigingParams,
) -> int:
    rows, cols = len(interest), len(interest[0])
    total = 0
    for r in range(rows):
        for c in range(cols):
            v = _grid_variance_at_cell((r, c), samples, satellites, params)
            if v <= params.accept_variance:
                total += int(interest[r][c])
    return total


@dataclass(frozen=True)
class GridResult:
    samples: List[Sample]
    objective: int
    wall_time_sec: float


def grid_exhaustive_search(
    interest: Grid,
    satellites: Sequence[Grid],
    per_satellite_samples: Sequence[int],
    params: KrigingParams,
) -> GridResult:
    """Exact enumeration for the grid (only small instances)."""
    rows, cols = len(interest), len(interest[0])

    start = time.time()

    sat_cells: List[List[Sample]] = []
    sat_combs: List[List[Tuple[Sample, ...]]] = []
    for sat_idx, sat in enumerate(satellites):
        cells = []
        for r in range(rows):
            for c in range(cols):
                if sat[r][c] > 0:
                    cells.append([r, c, sat_idx])
        sat_cells.append(cells)
        sat_combs.append(list(combinations(cells, per_satellite_samples[sat_idx])))

    best_obj = -1
    best_samples: List[Sample] = []

    for combo in product(*sat_combs):
        flat: List[Sample] = [list(s) for tup in combo for s in tup]
        # enforce no duplicates across satellites
        coords = [(s[0], s[1]) for s in flat]
        if len(coords) != len(set(coords)):
            continue
        obj = grid_cic_objective(interest, flat, satellites, params)
        if obj > best_obj:
            best_obj = obj
            best_samples = flat

    end = time.time()
    return GridResult(samples=best_samples, objective=int(best_obj), wall_time_sec=end - start)


def grid_random_search(
    interest: Grid,
    satellites: Sequence[Grid],
    per_satellite_samples: Sequence[int],
    params: KrigingParams,
    time_limit_sec: float,
    seed: int = 0,
) -> GridResult:
    """Time-limited random feasible sampling plans."""
    rows, cols = len(interest), len(interest[0])
    rng = random.Random(seed)

    start = time.time()

    sat_cells: List[List[Sample]] = []
    for sat_idx, sat in enumerate(satellites):
        cells = []
        for r in range(rows):
            for c in range(cols):
                if sat[r][c] > 0:
                    cells.append([r, c, sat_idx])
        sat_cells.append(cells)

    best_obj = -1
    best_samples: List[Sample] = []

    while (time.time() - start) < time_limit_sec:
        plan: List[Sample] = []
        used = set()
        feasible = True
        for sat_idx, k in enumerate(per_satellite_samples):
            # try a few times to avoid duplicates
            tries = 0
            while tries < 50:
                cand = rng.choice(sat_cells[sat_idx])
                rc = (cand[0], cand[1])
                if rc not in used:
                    used.add(rc)
                    plan.append(list(cand))
                    break
                tries += 1
            else:
                feasible = False
                break
        if not feasible:
            continue

        obj = grid_cic_objective(interest, plan, satellites, params)
        if obj > best_obj:
            best_obj = obj
            best_samples = plan

    end = time.time()
    return GridResult(samples=best_samples, objective=int(best_obj), wall_time_sec=end - start)
