"""신홍조 · 방법론 데모 — 포트폴리오의 판단을 직접 만져 보는 화면.

탭 1·2는 가상 데이터(회사 데이터 아님), 탭 3은 서울시 공공데이터.
  streamlit run demos/app/streamlit_app.py
"""
import json, os, sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "effect_testing_bh"))
from effect_test_demo import run as bh_run, summary as bh_summary  # noqa: E402

PAL = dict(bg="#1d2021", panel="#282828", fg="#ebdbb2", gray="#928374", red="#fb4934", green="#b8bb26",
           yellow="#fabd2f", blue="#83a598", aqua="#8ec07c", orange="#fe8019", purple="#d3869b")
REPO = "https://github.com/Azmo-Shin/Public_Portfolio"
st.set_page_config(page_title="신홍조 · 방법론 데모", page_icon="📊", layout="wide")


def style(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor=PAL["bg"], plot_bgcolor=PAL["panel"], font_color=PAL["fg"],
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor="#3c3836"); fig.update_yaxes(gridcolor="#3c3836")
    return fig


st.title("방법론 데모")
st.markdown(f"포트폴리오에서 설명한 **판단**을 직접 바꿔 보는 화면입니다. "
            f"탭 1·2는 **가상 데이터**, 탭 3은 **서울시 공공데이터**입니다(회사 데이터 없음). · [포트폴리오]({REPO}) · [코드]({REPO}/tree/main/demos/app)")
tab1, tab2, tab3 = st.tabs(["① 여러 번 검정하면 우연이 섞인다 (BH-FDR)", "② 요일을 무시한 이상탐지", "③ 서울 생활인구 핫스팟 (LISA)"])

# ------------------------------------------------------------------ ① BH-FDR
with tab1:
    st.markdown("마케팅 점검 항목을 하나씩 검정하면, 효과가 없는 항목도 **우연히** p<0.05가 나옵니다. "
                "진짜 효과가 있는 항목을 몇 개 심어 둔 가상 매장 데이터에서, 보정 없이 고를 때와 **BH-FDR**로 보정할 때를 비교합니다. "
                f"([04 폴더]({REPO}/tree/main/04_effect_testing_fdr))")
    c = st.columns(4)
    n_items = c[0].slider("검정 항목 수", 10, 80, 30, 5)
    n_true = c[1].slider("진짜 효과가 있는 항목", 0, 15, 6)
    n_stores = c[2].slider("매장 수", 40, 400, 120, 20)
    eff = c[3].slider("효과 크기(최대)", 0.2, 1.5, 1.0, 0.1)
    seed = st.number_input("가상 데이터 시드", 0, 999, 7)
    rng = np.random.default_rng(seed + 1000)
    true_items = tuple(sorted(rng.choice(n_items, size=min(n_true, n_items), replace=False).tolist()))
    kw = dict(n_stores=n_stores, n_items=n_items, true_items=true_items, effect_range=(eff * 0.6, eff))
    res = bh_run(seed=seed, **kw)
    s = bh_summary(res)
    m = st.columns(4)
    m[0].metric("보정 없음 · 찾은 진짜 효과", f"{s['naive']['found']} / {len(true_items)}")
    m[1].metric("보정 없음 · 가짜 '유의'", s["naive"]["false_pos"])
    m[2].metric("BH-FDR · 찾은 진짜 효과", f"{s['bh']['found']} / {len(true_items)}")
    m[3].metric("BH-FDR · 가짜 '유의'", s["bh"]["false_pos"])
    r = res.reset_index(drop=True).copy()
    r["rank"] = np.arange(1, len(r) + 1)
    r["종류"] = np.where(r["truly_effective"], "진짜 효과", "효과 없음")
    fig = px.scatter(r, x="rank", y="p", color="종류", log_y=True, hover_data=["item", "p_adj_bh"],
                     color_discrete_map={"진짜 효과": PAL["green"], "효과 없음": PAL["gray"]},
                     labels={"rank": "p값 순위 (작은 것부터)", "p": "p값 (로그)"})
    fig.add_hline(y=0.05, line_dash="dot", line_color=PAL["red"], annotation_text="p = 0.05 (보정 없음)")
    fig.add_trace(go.Scatter(x=r["rank"], y=r["rank"] / len(r) * 0.10, mode="lines", name="BH 기준선 k/m·q (q=0.10)",
                             line=dict(color=PAL["orange"], dash="dash")))
    st.plotly_chart(style(fig), use_container_width=True)
    if st.button("시드 20개로 평균 내기 (한 번의 우연에 기대지 않기)"):
        runs = [bh_summary(bh_run(seed=k, **kw)) for k in range(20)]
        a = lambda x, y: np.mean([q[x][y] for q in runs])
        st.write(f"20번 평균 — 보정 없음: 찾음 **{a('naive','found'):.1f}** · 가짜 **{a('naive','false_pos'):.1f}**  |  "
                 f"BH-FDR: 찾음 **{a('bh','found'):.1f}** · 가짜 **{a('bh','false_pos'):.1f}**")
    st.caption("검정: Mann-Whitney U(고/저 점수군 매출 변화). 보정: Benjamini-Hochberg(adj < 0.10). 모든 값은 가상 데이터.")

# ------------------------------------------------------------------ ② 이상탐지
with tab2:
    st.markdown("외식 매출은 요일 효과가 큽니다. **전체 기간을 한 기준으로** 밴드를 그리면 요일 차이 때문에 밴드가 넓어져 평일 급증·주말 급감 같은 진짜 이상을 놓치고, "
                "밴드를 좁히면 이번엔 주말이 통째로 '이상'이 됩니다. **같은 요일 과거 값**과 비교하면 진짜 이상을 잡을 수 있고, 남는 오탐은 밴드 폭(kσ)으로 조절합니다. "
                f"([03 폴더]({REPO}/tree/main/03_store_clustering_anomaly))")
    c = st.columns(4)
    lift = c[0].slider("주말 매출 증가(%)", 0, 100, 45, 5)
    noise = c[1].slider("평소 흔들림(%)", 2, 20, 6)
    shock = c[2].slider("심은 이상의 크기(%)", 10, 80, 35, 5)
    ksig = c[3].slider("밴드 폭 (±kσ)", 1.0, 3.0, 2.0, 0.1)
    seed2 = st.number_input("가상 데이터 시드 ", 0, 999, 3)
    rg = np.random.default_rng(seed2)
    days = pd.date_range("2025-01-01", periods=168, freq="D")
    wd = days.day_of_week
    base = 100 * (1 + np.where(wd >= 5, lift / 100, 0)) * (1 + 0.0006 * np.arange(len(days)))
    y = base * (1 + rg.normal(0, noise / 100, len(days)))
    cand = np.arange(60, len(days))
    weekday_idx = rg.choice(cand[wd[cand] < 5], 3, replace=False)
    weekend_idx = rg.choice(cand[wd[cand] >= 5], 3, replace=False)
    y[weekday_idx] *= 1 + shock / 100            # 평일 급증
    y[weekend_idx] *= 1 - shock / 100            # 주말 급감
    truth = np.zeros(len(days), bool); truth[np.r_[weekday_idx, weekend_idx]] = True
    s_all = pd.Series(y, index=days)
    g_mu = s_all.shift(1).rolling(56, min_periods=28).mean(); g_sd = s_all.shift(1).rolling(56, min_periods=28).std()
    by_dow = pd.DataFrame({"y": y, "dow": wd}, index=days)
    d_mu = by_dow.groupby("dow")["y"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
    d_sd = by_dow.groupby("dow")["y"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).std())
    valid = (np.arange(len(days)) >= 60)
    flag_g = valid & (np.abs(y - g_mu) > ksig * g_sd).to_numpy()
    flag_d = valid & (np.abs(y - d_mu) > ksig * d_sd).to_numpy()

    def score(f):
        return int((f & truth).sum()), int((~f & truth).sum()), int((f & ~truth).sum())
    rows = []
    for name, f in [("전체 기간 밴드", flag_g), ("요일별 밴드", flag_d)]:
        hit, miss, fp = score(f)
        rows.append({"방법": name, "잡은 진짜 이상": f"{hit} / {truth.sum()}", "놓친 이상": miss, "오탐(정상인데 이상)": fp})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    method = st.radio("그래프에 표시할 방법", ["요일별 밴드", "전체 기간 밴드"], horizontal=True)
    mu, sd, flag = (d_mu, d_sd, flag_d) if method == "요일별 밴드" else (g_mu, g_sd, flag_g)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=mu + ksig * sd, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=mu - ksig * sd, fill="tonexty", fillcolor="rgba(142,192,124,0.18)",
                             line=dict(width=0), name=f"{method} ±{ksig}σ"))
    fig.add_trace(go.Scatter(x=days, y=y, mode="lines", line=dict(color=PAL["fg"], width=1.4), name="일매출(가상)"))
    fig.add_trace(go.Scatter(x=days[truth], y=y[truth], mode="markers", marker=dict(size=13, color="rgba(0,0,0,0)",
                             line=dict(color=PAL["yellow"], width=2)), name="심어 둔 진짜 이상"))
    fig.add_trace(go.Scatter(x=days[flag], y=y[flag], mode="markers", marker=dict(size=7, color=PAL["red"]),
                             name="탐지됨"))
    fig.add_vrect(x0=days[0], x1=days[59], fillcolor="#3c3836", opacity=0.35, line_width=0,
                  annotation_text="기준을 쌓는 기간", annotation_position="top left")
    st.plotly_chart(style(fig, 440), use_container_width=True)
    st.caption("요일별 밴드: 같은 요일 과거 12주(n≥3)의 평균±kσ. 전체 기간 밴드: 과거 56일 전체의 평균±kσ. 둘 다 당일 값은 쓰지 않음(shift 1). ±kσ 밴드는 정상인 날도 일부를 이상으로 잡는다(정규분포면 ±1.5σ 밖이 약 13%) — 실제 업무에서는 편차 크기로 심각도를 나눠 사람이 대조했다. 모든 값은 가상 데이터.")

# ------------------------------------------------------------------ ③ LISA
@st.cache_data
def load_seoul():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "data")) if f.startswith("seoul_dong_pop_"))
    pop = pd.read_csv(os.path.join(HERE, "data", files[-1]), dtype={"dong_code": str})
    with open(os.path.join(HERE, "data", "seoul_dong.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    return pop, gj, files[-1].split("_")[-1][:6]


def knn_weights(lon, lat, k):
    xy = np.c_[lon * np.cos(np.radians(37.55)) * 111.0, lat * 111.0]
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    return idx[:, 1:]                                   # 자기 자신 제외, 행표준화는 평균으로


def morans(x, nb, perms=999, seed=0):
    z = (x - x.mean()) / x.std()
    lag = z[nb].mean(axis=1)
    I = float((z * lag).sum() / (z * z).sum())
    rng = np.random.default_rng(seed)
    sims = np.array([(z * rng.permutation(z)[nb].mean(axis=1)).sum() / (z * z).sum() for _ in range(perms)])
    p_global = (1 + (np.abs(sims) >= abs(I)).sum()) / (perms + 1)
    n, k = nb.shape
    li = z * lag
    p_local = np.ones(n)
    others = np.array([np.r_[0:i, i + 1:n] for i in range(n)])           # 조건부 순열: i 자신은 고정
    cnt = np.zeros(n)
    for _ in range(199):
        pick = others[np.arange(n)[:, None], rng.integers(0, n - 1, size=(n, k))]
        sim = z * z[pick].mean(axis=1)
        cnt += (np.abs(sim) >= np.abs(li))
    p_local = (cnt + 1) / 200
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    return I, p_global, quad, p_local


with tab3:
    pop, gj, month = load_seoul()
    st.markdown("서울 행정동의 **생활인구 밀도**가 공간적으로 뭉쳐 있는지(Global Moran's I), 어디가 핫스팟·콜드스팟인지(Local Moran's I · LISA) 봅니다. "
                "시간대를 바꾸면 낮의 업무지구와 밤의 주거지가 뒤바뀌는 것이 보입니다. "
                f"([02 폴더]({REPO}/tree/main/02_spatial_esda) · [GIS 저장소](https://github.com/Azmo-Shin/seoul-living-population-gis))")
    c = st.columns(4)
    day_type = c[0].radio("요일", ["주중", "주말"], horizontal=True)
    hour = c[1].slider("시간대", 0, 23, 14)
    k = c[2].slider("이웃 수 (KNN)", 4, 10, 6)
    view = c[3].radio("지도", ["LISA 핫스팟", "생활인구 밀도"], horizontal=True)
    d = pop[(pop["day_type"] == day_type) & (pop["hour"] == hour)].reset_index(drop=True)
    d["density"] = d["pop"] / d["area_km2"]
    x = np.log1p(d["density"].to_numpy())
    nb = knn_weights(d["lon"].to_numpy(), d["lat"].to_numpy(), k)
    I, pg, quad, pl = morans(x, nb, seed=hour)
    d["군집"] = np.where(pl < 0.05, quad, "유의하지 않음")
    m = st.columns(4)
    m[0].metric("Global Moran's I", f"{I:.3f}")
    m[1].metric("순열 p (999회)", f"{pg:.3f}")
    m[2].metric("핫스팟 HH 동", int((d["군집"] == "HH").sum()))
    m[3].metric("콜드스팟 LL 동", int((d["군집"] == "LL").sum()))
    colors = {"HH": PAL["red"], "LL": PAL["blue"], "HL": PAL["orange"], "LH": PAL["aqua"], "유의하지 않음": "#3c3836"}
    if view == "LISA 핫스팟":
        fig = px.choropleth_map(d, geojson=gj, locations="dong_code", color="군집", color_discrete_map=colors,
                                hover_name="dong_name", hover_data={"density": ":,.0f", "dong_code": False},
                                category_orders={"군집": ["HH", "LL", "HL", "LH", "유의하지 않음"]},
                                map_style="carto-darkmatter", center=dict(lat=37.565, lon=126.98), zoom=9.6, opacity=0.8)
    else:
        fig = px.choropleth_map(d, geojson=gj, locations="dong_code", color=np.log10(d["density"]),
                                color_continuous_scale=["#282828", PAL["blue"], PAL["yellow"], PAL["orange"], PAL["red"]],
                                hover_name="dong_name", hover_data={"density": ":,.0f", "dong_code": False},
                                labels={"color": "log10 밀도(명/㎢)"}, map_style="carto-darkmatter",
                                center=dict(lat=37.565, lon=126.98), zoom=9.6, opacity=0.8)
    st.plotly_chart(style(fig, 560), use_container_width=True)
    top = d[d["군집"] == "HH"].nlargest(8, "density")[["dong_name", "density"]]
    if len(top):
        st.write("핫스팟(HH) 밀도 상위: " + " · ".join(f"{n}" for n in top["dong_name"]))
    st.caption(f"값: 행정동 × {day_type} × {hour}시 평균 생활인구 ÷ 면적, 로그 변환. 가중치: 대표점 KNN(k={k}), 행표준화. "
               f"LISA: 조건부 순열 199회, p<0.05만 색칠. "
               f"출처: 서울특별시 열린데이터광장 — 행정동 단위 서울 생활인구(내국인) {month[:4]}-{month[4:]}, 서울시 상권분석서비스(영역-행정동). 공공누리 1유형(출처표시).")
