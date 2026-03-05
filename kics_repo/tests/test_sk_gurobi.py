import pytest

from kics.ighm import load_ighm_json, IGHM
from kics.algorithms.sk_gurobi import solve_sk_mclp


def test_sk_gurobi_runs_on_small_instance():
    gp = pytest.importorskip("gurobipy")

    ighm = load_ighm_json("data/ighm_1.json")
    subset = ighm.cell_ids[:30]
    sub = IGHM(df=ighm.df[subset])

    res = solve_sk_mclp(sub, k=4, radius=3.0, time_limit_sec=10)
    assert len(res.samples) == 4
    assert res.objective >= 0
