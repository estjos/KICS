# Kriging-informed coverage sampling — reproducible code

This repository is the reproducible research artifact for the paper:

**Kriging-informed coverage sampling: An Integer Program Approach to Optimizing Spatial Sampling**

It contains:
- the **SK** surrogate (a maximal-coverage-style IP) solved with Gurobi
- the **CIC** (“confident information coverage”) objective evaluation using Kriging
- baselines (small-instance exact enumeration + random search)
- scripts to reproduce the computational experiments and the IGHM example

---

## quick start

> Run these from the **repo root** (the folder that contains `kics/`, `scripts/`, etc.)

### 1) Create & activate the environment
```bash
conda create -n kics python=3.11 -y
conda activate kics
python -m pip install -r requirements.txt
```

### 2) Install the repo so `import kics` works
```bash
python -m pip install -e .
```

### 3) Install Gurobi’s Python package (`gurobipy`) and verify
```bash
python -m pip install -r requirements-gurobi.txt
python -c "import gurobipy as gp; print('gurobi version:', gp.gurobi.version())"
```

If you see a license/version mismatch like:
`Version number is 13.0, license is for version 12.0`
then you installed the wrong major version of `gurobipy`. Fix by pinning to your license major version (see Troubleshooting below).

### 4) Run the included IGHM example
```bash
python scripts/run_ighm_example.py --data data/ighm_1.json --k 5 --radius 3.0
```

---

## Repository layout

- `kics/` — the core library (IGHM loader, variogram, kriging proxy, SK solver, grid experiments)
- `scripts/` — runnable entrypoints used for experiments
- `data/ighm_1.json` — a small IGHM example instance
- `tests/` — unit tests (Gurobi tests are skipped if `gurobipy` is unavailable)

---

## Input format: IGHM JSON

The IGHM JSON format is the same one used in the project codebase. It is a JSON object keyed by **cell id**, where each cell stores:

- `x`, `y` : coordinates
- `gain` : information gain weight for that cell
- `suitability` : a dict mapping sensor/zoom keys (e.g., `SA_Z3`) to suitability values in `[0,1]`

Example (one cell):
```json
{
  "1": {
    "x": -79.10,
    "y": -26.19,
    "gain": 0.54,
    "suitability": {"SA_Z1": 0.339, "SA_Z3": 0.19, "...": 0.01}
  }
}
```

The loader is `kics.ighm.load_ighm_json`, implemented via `pandas.read_json`.

---

## Scripts

### 1) `scripts/run_ighm_example.py` (IGHM + SK + CIC evaluation)

This script:
1. loads an IGHM instance (`--data`)
2. solves the SK surrogate IP (MCLP form) with Gurobi using a coverage radius (`--radius` or `--auto-radius`)
3. evaluates the chosen samples under the CIC objective (kriging variance thresholded by `--accept`)

#### Common runs

**A. Run SK with a fixed coverage radius**
```bash
python scripts/run_ighm_example.py --data data/ighm_1.json --k 5 --radius 3.0
```

**B. Auto-compute a radius from the variogram + CIC threshold**
This finds an `h` such that `gamma(h) <= accept` under the spherical semivariogram.
```bash
python scripts/run_ighm_example.py --data data/ighm_1.json --k 5 --auto-radius --no-range-cutoff --penalty-mode suitability --delta 1.1
```
This should give the following results:
=== SK (Gurobi) ===
status: OPTIMAL
radius: 3.0
k: 5
SK surrogate objective: 44.446035
True CIC objective (kriging eval): 16.837096
samples: [13, 328, 360, 401, 507]

#### Arguments (and what they mean)

Core:
- `--data PATH` : path to an IGHM JSON file (quote the path if it contains spaces)
- `--k INT` : number of samples to select
- `--radius FLOAT` : coverage radius used in the SK surrogate (neighborhood definition)
- `--auto-radius` : compute `radius` by solving `gamma(h) <= accept` for `h`

Variogram / CIC thresholding (KrigingParams):
- `--range FLOAT` : variogram range parameter `r`
- `--sill FLOAT` : structured component of the sill (total sill is `nugget + sill`)
- `--nugget FLOAT` : nugget
- `--accept FLOAT` : CIC threshold (`epsilon_0`): cell is “covered” if variance <= accept
- `--delta FLOAT` : penalty base (>1), used by the resolution penalty

CIC evaluation mode:
- `--penalty-mode zoom|suitability`
  - `zoom` (default): uses the **zoom integer** parsed from keys like `SA_Z3`; applies `delta^(lambda_star - lambda_q)`
  - `suitability`: legacy behavior; uses the **max suitability value** in `[0,1]`; applies `delta^(1 - max_suit_value)`
- `--no-range-cutoff`
  - if provided: **all** samples are used in the kriging proxy evaluation (no distance cutoff)
  - otherwise: only samples within `--range` are used; if none are in range the code returns a large variance sentinel

---

### 2) `scripts/run_grid_experiments.py` (synthetic grid computational experiments)

This script runs the synthetic grid experiments and writes a CSV.

**Important:** `--grid-sizes` expects **space-separated integers**, not a Python list.
```bash
python scripts/run_grid_experiments.py --grid-sizes 5 10 15 20 --reps 10 --out grid_results.csv
```

Key args:
- `--grid-sizes INT [INT ...]` : grid sizes `g` (a g×g grid)
- `--reps INT` : number of replicates per grid size
- `--processes INT` : multiprocessing pool size (0 disables multiprocessing)
- `--out PATH` : output CSV path

The CSV includes (when applicable): SS (exact), random search baselines, SK surrogate objective, SK “true” CIC objective, and a local-search improvement.

---

### 3) `scripts/self_test_gurobi.py`

Use this when you want to confirm:
- `gurobipy` imports
- a model can be created/optimized
- the license is recognized

```bash
python scripts/self_test_gurobi.py
```

---

## CIC evaluation modes (why you might see different CIC scores)

The SK surrogate objective (the MIP objective) can match across implementations while the *true* CIC score differs if:

1) **Range cutoff differs**  
   - cutoff ON: only samples within `range` contribute
   - cutoff OFF: all samples contribute (variogram saturates, but samples remain in the system)

2) **Penalty definition differs**  
   - `zoom`: uses the zoom integer from keys like `*_Z3`
   - `suitability`: uses max suitability value in `[0,1]` (legacy)

3) **Multiple optimal SK solutions exist**  
   Different sample sets can yield the same SK objective but different CIC (“true”) objective.

---

## Testing

Run unit tests:
```bash
pytest -q
```

Notes:
- Gurobi-dependent tests are skipped if `gurobipy` cannot be imported.
- If you want to run only non-Gurobi tests:
```bash
pytest -q -k "not gurobi"
```

---

## Troubleshooting

### “No module named `kics`”
You didn’t install the repo as a package. From the repo root:
```bash
python -m pip install -e .
```

### zsh: `no matches found: [3,3,3]`
zsh treats `[...]` as a filename pattern. For `--grid-sizes`, pass space-separated ints:
```bash
python scripts/run_grid_experiments.py --grid-sizes 3 3 3
```

### GurobiError: “Version number is 13.0, license is for version 12.0”
Your `gurobipy` major version does not match your license major version.

Fix (example for a v12 license):
```bash
python -m pip uninstall -y gurobipy
python -m pip install "gurobipy>=12,<13"
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```



