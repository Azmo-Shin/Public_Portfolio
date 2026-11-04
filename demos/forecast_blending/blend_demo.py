"""가상 일매출로 보는 예측 블렌딩: 트리 후보를 '선택 지표'로 고르고 계절성 모델과 볼록가중으로 섞는다.

실제 업무(01 매출 예측)의 구조만 옮긴 예제다. Prophet 대신 같은 역할(추세·연/주 계절성·프로모션 회귀자)을
하는 선형 계절 모델을 써서 의존성을 줄였다.
핵심 규칙: (1) 롤링·시차 피처는 shift(1) 뒤에 계산해 타깃 누수 차단 (2) 튜닝 지표(MSE)와 선택 지표(MAPE) 분리
(3) 블렌딩 가중치는 검증 구간에서만 정하고 테스트 구간은 마지막에 한 번만 본다.
    python blend_demo.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

SEED = 3


def make_sales(days=730, seed=SEED):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2024-01-01", periods=days, freq="D")
    t = np.arange(days)
    promo = (rng.uniform(size=days) < 0.12).astype(int)
    y = (1000 + 0.4 * t                                     # 추세
         + 120 * np.sin(2 * np.pi * t / 365.25)              # 연 계절성
         + np.where(d.day_of_week >= 4, 180, 0)                # 금·토·일
         + 150 * promo                                       # 프로모션
         + rng.normal(0, 60, size=days))
    return pd.DataFrame({"date": d, "y": y, "promo": promo})


def add_features(df):
    out = df.copy()
    out["dow"], out["month"] = out["date"].dt.day_of_week, out["date"].dt.month
    past = out["y"].shift(1)                                 # 오늘 값은 절대 쓰지 않는다
    out["lag1"], out["lag7"] = past, out["y"].shift(7)
    out["ma7"], out["ma14"] = past.rolling(7).mean(), past.rolling(14).mean()
    same_dow = out.groupby("dow")["y"].transform(lambda s: s.shift(1).rolling(5).mean())
    out["same_dow_mean5"] = same_dow
    return out.dropna().reset_index(drop=True)


TREE_FEATS = ["dow", "month", "promo", "lag1", "lag7", "ma7", "ma14", "same_dow_mean5"]


def seasonal_design(df):
    t = (df["date"] - pd.Timestamp("2024-01-01")).dt.days.to_numpy()
    cols = {"t": t, "promo": df["promo"].to_numpy()}
    for k in (1, 2):
        cols[f"s{k}"], cols[f"c{k}"] = np.sin(2 * np.pi * k * t / 365.25), np.cos(2 * np.pi * k * t / 365.25)
    X = pd.DataFrame(cols)
    dow = pd.get_dummies(pd.Categorical(df["dow"], categories=range(7)), prefix="dow", dtype=float)
    return pd.concat([X, dow.reset_index(drop=True)], axis=1)


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def run(seed=SEED):
    df = add_features(make_sales(seed=seed))
    n = len(df)
    tr, va, te = df.iloc[: n - 120], df.iloc[n - 120: n - 60], df.iloc[n - 60:]
    candidates = {"RandomForest": RandomForestRegressor(n_estimators=200, random_state=0),
                  "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=0),
                  "GradBoost": GradientBoostingRegressor(random_state=0)}      # 기본 손실 = MSE
    va_mape = {}
    for name, m in candidates.items():
        m.fit(tr[TREE_FEATS], tr["y"])
        va_mape[name] = mape(va["y"], m.predict(va[TREE_FEATS]))           # 선택은 MAPE로
    best = min(va_mape, key=va_mape.get)
    tree = candidates[best]
    seas = Ridge(alpha=1.0).fit(seasonal_design(tr), tr["y"])
    p_tree_va, p_seas_va = tree.predict(va[TREE_FEATS]), seas.predict(seasonal_design(va))
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    w = min(grid, key=lambda w: mape(va["y"], w * p_tree_va + (1 - w) * p_seas_va))
    p_te = w * tree.predict(te[TREE_FEATS]) + (1 - w) * seas.predict(seasonal_design(te))
    return {"candidates_val_mape": va_mape, "chosen_tree": best, "weight_tree": float(w),
            "val_mape_blend": mape(va["y"], w * p_tree_va + (1 - w) * p_seas_va),
            "val_mape_tree": mape(va["y"], p_tree_va), "val_mape_seasonal": mape(va["y"], p_seas_va),
            "test_mape_blend": mape(te["y"], p_te)}


if __name__ == "__main__":
    r = run()
    for k, v in r["candidates_val_mape"].items():
        print(f"후보 {k:12s} 검증 MAPE {v:5.2f}%")
    print(f"선택: {r['chosen_tree']}  · 블렌딩 가중치(트리) {r['weight_tree']:.2f}")
    print(f"검증 MAPE  트리 {r['val_mape_tree']:.2f}% · 계절 {r['val_mape_seasonal']:.2f}% · 블렌딩 {r['val_mape_blend']:.2f}%")
    print(f"테스트 MAPE 블렌딩 {r['test_mape_blend']:.2f}%  (가상 데이터 값 — 실제 업무 성능과 무관)")
