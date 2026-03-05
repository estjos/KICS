from kics.ighm import load_ighm_json, IGHM
from kics.kriging import cic_objective
from kics.params import KrigingParams


def test_cic_objective_runs_on_subset():
    ighm = load_ighm_json("data/ighm_1.json")
    subset = ighm.cell_ids[:25]
    sub = IGHM(df=ighm.df[subset])

    params = KrigingParams(range_=3.0, sill=2.0, nugget=0.0, accept_variance=2.0, delta=1.1)

    obj0, cov0 = cic_objective(sub, [], params)
    assert obj0 == 0.0
    assert all(c == 0 for c in cov0)

    # one sample
    sample = [subset[0]]
    obj1, cov1 = cic_objective(sub, sample, params)
    assert obj1 >= 0.0
    assert len(cov1) == len(subset)
