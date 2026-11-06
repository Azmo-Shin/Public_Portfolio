"""가상 데이터로 보는 다중비교 보정: 항목 30개를 각각 검정하면 우연히 '유의'가 섞인다.

실제 업무(04 효과 검증)의 구조만 옮긴 예제다. 매장 120곳, 마케팅 점검 항목 30개 중
**진짜 효과가 있는 항목 6개**를 심어 두고, 보정 없는 p<0.05와 BH-FDR(adj<0.1)을 비교한다.
    python effect_test_demo.py
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

N_STORES, N_ITEMS, TRUE_ITEMS, SEED = 120, 30, (0, 3, 7, 12, 18, 25), 7


def benjamini_hochberg(p):
    """BH 보정 p값(단조 보장). scipy.stats.false_discovery_control 과 같은 결과."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def make_data(seed=SEED, n_stores=N_STORES, n_items=N_ITEMS, true_items=TRUE_ITEMS, effect_range=(0.6, 1.0), noise=1.0):
    """매장 × 항목 점수(0/1: 고/저 점수군)와 매출 변화율. 진짜 항목만 매출에 영향."""
    rng = np.random.default_rng(seed)
    high = rng.integers(0, 2, size=(n_stores, n_items))
    effect = np.zeros(n_items)
    effect[list(true_items)] = rng.uniform(*effect_range, size=len(true_items))
    sales_change = high @ effect + rng.normal(0, noise, size=n_stores)
    items = [f"item_{i:03d}" for i in range(n_items)]
    return pd.DataFrame(high, columns=items), pd.Series(sales_change, name="sales_change")


def run(seed=SEED, alpha=0.05, q=0.10, **data_kw):
    scores, y = make_data(seed, **data_kw)
    true_items = set(data_kw.get("true_items", TRUE_ITEMS))
    rows = []
    for col in scores:
        hi, lo = y[scores[col] == 1], y[scores[col] == 0]
        p = mannwhitneyu(hi, lo, alternative="two-sided").pvalue      # 비모수: 분포 가정 없음
        rows.append((col, p))
    res = pd.DataFrame(rows, columns=["item", "p"])
    res["p_adj_bh"] = benjamini_hochberg(res["p"])
    res["truly_effective"] = [int(c.split("_")[1]) in true_items for c in res["item"]]
    res["naive_sig"] = res["p"] < alpha
    res["bh_sig"] = res["p_adj_bh"] < q
    return res.sort_values("p").reset_index(drop=True)


def summary(res):
    def fp(col): return int((res[col] & ~res["truly_effective"]).sum())
    def tp(col): return int((res[col] & res["truly_effective"]).sum())
    return {"naive": {"found": tp("naive_sig"), "false_pos": fp("naive_sig")},
            "bh": {"found": tp("bh_sig"), "false_pos": fp("bh_sig")}}


if __name__ == "__main__":
    res = run()
    print(res.head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    s = summary(res)
    print(f"\n진짜 효과 항목 {len(TRUE_ITEMS)}개 중")
    print(f"  보정 없음 p<0.05 : 찾음 {s['naive']['found']} · 가짜 {s['naive']['false_pos']}")
    print(f"  BH-FDR adj<0.10 : 찾음 {s['bh']['found']} · 가짜 {s['bh']['false_pos']}")
    runs = [summary(run(seed=k)) for k in range(20)]           # 한 번의 우연에 기대지 않도록 20번 평균
    avg = lambda a, b: sum(r[a][b] for r in runs) / len(runs)
    print(f"\n가상 데이터 20벌 평균 — 보정 없음: 찾음 {avg('naive','found'):.1f} · 가짜 {avg('naive','false_pos'):.1f}"
          f" | BH-FDR: 찾음 {avg('bh','found'):.1f} · 가짜 {avg('bh','false_pos'):.1f}")
