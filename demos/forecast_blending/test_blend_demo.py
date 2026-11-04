import numpy as np
from blend_demo import make_sales, add_features, run, TREE_FEATS


def test_features_do_not_leak_todays_target():
    """오늘 매출을 바꿔도 오늘 행의 피처는 그대로여야 한다(shift(1) 누수 차단)."""
    raw = make_sales()
    base = add_features(raw)
    i = 300
    day = base.loc[i, "date"]
    bumped = raw.copy()
    bumped.loc[bumped["date"] == day, "y"] += 10_000
    after = add_features(bumped)
    row_b, row_a = base[base["date"] == day], after[after["date"] == day]
    assert np.allclose(row_b[TREE_FEATS].to_numpy(), row_a[TREE_FEATS].to_numpy())


def test_blend_weight_is_convex_and_not_worse_on_validation():
    r = run()
    assert 0.0 <= r["weight_tree"] <= 1.0
    assert r["val_mape_blend"] <= min(r["val_mape_tree"], r["val_mape_seasonal"]) + 1e-9


def test_choice_uses_validation_mape():
    r = run()
    assert r["chosen_tree"] == min(r["candidates_val_mape"], key=r["candidates_val_mape"].get)
