from kics.ighm import load_ighm_json


def test_load_ighm_and_zoom_parsing():
    ighm = load_ighm_json("data/ighm_1.json")
    assert len(ighm.cell_ids) > 0

    cid = ighm.cell_ids[0]
    x, y = ighm.coords(cid)
    assert isinstance(x, float)
    assert isinstance(y, float)

    s = ighm.suitability(cid)
    assert isinstance(s, dict)
    assert len(s) > 0

    z = ighm.zoom_levels(cid)
    # many files encode zooms like Z1, Z3, Z9
    assert all(isinstance(v, int) for v in z)
