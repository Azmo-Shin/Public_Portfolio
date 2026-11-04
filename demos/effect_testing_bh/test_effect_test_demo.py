import numpy as np
from scipy.stats import false_discovery_control
from effect_test_demo import benjamini_hochberg, run, summary, TRUE_ITEMS


def test_bh_matches_scipy():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50) ** 2
    assert np.allclose(benjamini_hochberg(p), false_discovery_control(p, method="bh"))


def test_bh_is_monotone_in_p():
    p = np.array([0.04, 0.001, 0.03, 0.2, 0.01])
    adj = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12) and np.all(adj >= p)


def test_bh_controls_false_positives_better_than_naive_on_average():
    naive_fp, bh_fp, bh_found = 0, 0, 0
    for seed in range(20):
        s = summary(run(seed=seed))
        naive_fp += s["naive"]["false_pos"]; bh_fp += s["bh"]["false_pos"]; bh_found += s["bh"]["found"]
    assert bh_fp < naive_fp                      # 보정이 가짜 '유의'를 줄인다
    assert bh_found / (20 * len(TRUE_ITEMS)) > 0.5   # 진짜 효과는 대부분 남는다
