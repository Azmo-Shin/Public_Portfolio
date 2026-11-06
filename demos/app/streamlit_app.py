"""신홍조 · 판단 데모 — 포트폴리오의 판단을 직접 바꿔 보는 화면.

① 다중검정 보정(가상 데이터) ② 요일 조건부 이상탐지(가상 데이터) ③ 서울 생활인구 핫스팟(서울시 공공데이터).
회사 데이터는 쓰지 않는다.  streamlit run demos/app/streamlit_app.py   (특정 데모: ?demo=1|2|3)
"""
import json, os, sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "effect_testing_bh"))
from effect_test_demo import run as bh_run, summary as bh_summary  # noqa: E402

REPO = "https://github.com/Azmo-Shin/Public_Portfolio"
st.set_page_config(page_title="신홍조 · 판단 데모", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- 디자인 토큰 (dataviz 검증 통과 색)
C = dict(bg="#1d2021", surface="#282828", border="#3c3836", grid="#32302f",
         text="#ebdbb2", text2="#bdae93", muted="#a89984", accent="#fabd2f",
         true="#4a90c2", false="#fb4934", rest="#7c6f64", mixed="#b16286")
SEQ = ["#7a5230", "#a8581a", "#d65d0e", "#fe8019", "#fabd2f"]        # 밀도: 주황 한 계열

st.markdown(f"""<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
html, body, .stMarkdown, button, input, label, p {{font-family:'Pretendard Variable',Pretendard,'Noto Sans KR',sans-serif;}}
.block-container {{padding-top:1.4rem; max-width:1200px;}}
.eyebrow {{font-family:ui-monospace,'JetBrains Mono',Menlo,monospace; font-size:12px; color:{C['muted']}; letter-spacing:.02em;}}
.hero {{font-size:2.0rem; font-weight:750; color:{C['text']}; margin:.1rem 0 .2rem;}}
.lede {{font-size:1.02rem; color:{C['text2']}; margin-bottom:.5rem;}}
.chip {{display:inline-block; font-size:12px; padding:2px 10px; margin:0 6px 6px 0; border:1px solid {C['border']};
        border-radius:999px; color:{C['text2']}; background:{C['surface']};}}
.chip a {{color:{C['accent']}; text-decoration:none;}}
.headline {{font-size:1.4rem; font-weight:700; line-height:1.45; color:{C['text']}; margin:.15rem 0 .25rem;}}
.headline b {{color:#fbf1c7; border-bottom:2px solid {C['accent']};}}
.sub {{color:{C['muted']}; font-size:.92rem; margin-bottom:.6rem;}}
[data-testid="stMetric"] {{background:{C['surface']}; border-color:{C['border']} !important;}}
[data-testid="stMetricValue"] {{font-weight:650;}}
.hero, .lede, .headline, .sub, .stMarkdown p, .stCaption {{word-break:keep-all; overflow-wrap:break-word;}}
@media (max-width: 640px) {{
  .hero {{font-size:1.55rem;}} .headline {{font-size:1.15rem;}}
  div[data-testid="stElementContainer"]:has(> div[data-testid="stMetric"]) {{width:calc(50% - 6px) !important; flex:0 0 calc(50% - 6px) !important; min-width:0 !important;}}
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMetric"]) {{flex-direction:row !important; flex-wrap:wrap !important; gap:12px !important;}}
  [data-testid="stMetricValue"] {{font-size:1.5rem;}} [data-testid="stMetricLabel"] p {{font-size:.8rem;}}
}}
</style>""", unsafe_allow_html=True)

pio.templates["gruv"] = go.layout.Template(layout=dict(
    font=dict(family="Pretendard Variable, Pretendard, 'Noto Sans KR', sans-serif", size=13, color=C["text2"]),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(gridcolor=C["grid"], linecolor=C["border"], zeroline=False, ticks="", automargin=True),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["border"], zeroline=False, ticks="", automargin=True),
    legend=dict(orientation="h", x=0, y=-0.08, yanchor="top", title_text="", font_size=12),
    hoverlabel=dict(bgcolor=C["bg"], bordercolor=C["border"], font_color=C["text"])))
pio.templates.default = "gruv"


def show(fig, h=380, **cfg):
    fig.update_layout(height=h)
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False, **cfg})


def write_head(box, eyebrow, headline, sub=None):
    with box:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="headline">{headline}</div>'
                    + (f'<div class="sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def cards(box, items):
    with box:
        with st.container(horizontal=True):
            for label, value, delta, color, help_ in items:
                st.metric(label, value, delta=delta, delta_color=color, help=help_, border=True, width=230)


# ---------------------------------------------------------------- 첫 화면
st.markdown('<div class="eyebrow">~/portfolio/demos $ ./run</div>'
            '<div class="hero">신홍조 · 데이터 사이언티스트</div>'
            '<div class="lede">"유의하다"는 결과를 믿기 전에, <b>우연·요일·공간</b> 때문에 생긴 착시인지 먼저 확인합니다. 세 가지 판단을 직접 바꿔 보세요.</div>'
            f'<span class="chip">가상 데이터 ①②</span><span class="chip">서울시 공공데이터 ③</span><span class="chip">회사 데이터 없음</span>'
            f'<span class="chip"><a href="{REPO}">포트폴리오 ↗</a></span><span class="chip"><a href="{REPO}/tree/main/demos/app">코드 ↗</a></span>',
            unsafe_allow_html=True)

DEMOS = {"1": "① 다중검정 보정", "2": "② 요일 이상탐지", "3": "③ 서울 핫스팟"}
qp = st.query_params.get("demo", "1")
choice = st.segmented_control("데모 고르기", list(DEMOS.values()), default=DEMOS.get(qp, DEMOS["1"]),
                              label_visibility="collapsed") or DEMOS["1"]
key = [k for k, v in DEMOS.items() if v == choice][0]
if st.query_params.get("demo") != key:
    st.query_params["demo"] = key
st.write("")


# ================================================================ ① 다중검정 보정
@st.cache_data
def bh_avg(n_items, n_true, n_stores, eff, runs=20):
    nh = nf = bh = bf = 0
    for k in range(runs):
        ti = tuple(sorted(np.random.default_rng(k + 1000).choice(n_items, size=min(n_true, n_items), replace=False).tolist()))
        s = bh_summary(bh_run(seed=k, n_stores=n_stores, n_items=n_items, true_items=ti, effect_range=(eff * .6, eff)))
        nh += s["naive"]["found"]; nf += s["naive"]["false_pos"]; bh += s["bh"]["found"]; bf += s["bh"]["false_pos"]
    rate = lambda f, h: f / (f + h) if f + h else 0.0
    return dict(nh=nh / runs, nf=nf / runs, bh=bh / runs, bf=bf / runs, nfdr=rate(nf, nh), bfdr=rate(bf, bh))


LAB = {"hit": "찾은 진짜", "fp": "가짜 경보", "miss": "놓친 진짜", "rest": "효과 없음(걸러냄)"}
MK = {"hit": dict(symbol="square", color=C["true"]), "fp": dict(symbol="square", color=C["false"]),
      "miss": dict(symbol="square-open", color=C["true"], line=dict(width=2, color=C["true"])),
      "rest": dict(symbol="square", color=C["rest"], opacity=.45)}


def waffle(r, cols=10):
    t = r["truly_effective"].to_numpy()
    titles = []
    for col in ["naive_sig", "bh_sig"]:
        s = r[col].to_numpy()
        titles.append(f"찾음 {(s & t).sum()}/{t.sum()} · 가짜 {(s & ~t).sum()}")
    fig = make_subplots(1, 2, horizontal_spacing=.08,
                        subplot_titles=[f"보정 없음 (p<0.05) — {titles[0]}", f"BH-FDR (q=0.10) — {titles[1]}"])
    for j, col in enumerate(["naive_sig", "bh_sig"], 1):
        s = r[col].to_numpy()
        o = np.select([s & t, s & ~t, ~s & t], ["hit", "fp", "miss"], "rest")
        idx = np.argsort(pd.Categorical(o, list(LAB)).codes, kind="stable"); o = o[idx]
        i = np.arange(len(o)); x, y = i % cols, -(i // cols)
        for k in LAB:
            m = o == k
            fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="markers", name=LAB[k], legendgroup=k, showlegend=j == 1,
                                     marker=dict(size=14, **MK[k]),
                                     customdata=np.c_[r["item"].to_numpy()[idx][m], r["p"].to_numpy()[idx][m]],
                                     hovertemplate="%{customdata[0]} · p=%{customdata[1]:.4f}<extra>" + LAB[k] + "</extra>"), 1, j)
    fig.update_xaxes(visible=False, range=[-.7, cols - .3]); fig.update_yaxes(visible=False)
    fig.update_annotations(font=dict(size=13, color=C["text"]))
    return fig, 110 + 24 * int(np.ceil(len(r) / cols))


def demo_bh():
    head, kpi = st.container(), st.container()
    st.session_state.setdefault("seed1", 7)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.container(border=True):
            n_items = st.slider("검정한 항목 수", 10, 100, 60, 5, help="마케팅 점검 항목처럼, 하나씩 따로 검정하는 항목의 개수")
            n_true = st.slider("진짜 효과가 있는 항목", 0, 20, 8)
            with st.expander("조건 더 바꾸기"):
                n_stores = st.slider("매장 수", 40, 400, 120, 20)
                eff = st.slider("효과 크기(최대)", 0.2, 1.5, 1.0, 0.1)
            if st.button("다른 가상 데이터", icon=":material/casino:", width="stretch"):
                st.session_state.seed1 += 1
            st.caption(f"가상 데이터 시드 {st.session_state.seed1}")
    seed = st.session_state.seed1
    a = bh_avg(n_items, n_true, n_stores, eff)
    write_head(head, f"① 다중검정 보정 · 가상 데이터 · <a href='{REPO}/tree/main/04_effect_testing_fdr'>04 폴더</a>",
               f"보정 없이 고르면 '효과 있음' 중 <b>{a['nfdr']:.0%}가 가짜</b>입니다. BH-FDR은 <b>{a['bfdr']:.0%}</b>로 줄이고, "
               f"진짜는 {a['nh']:.1f}개 → {a['bh']:.1f}개로 조금 덜 찾습니다.",
               "가상 데이터 20벌의 평균입니다(한 번의 우연에 기대지 않기 위해). 아래 그림은 그중 한 벌.")
    cards(kpi, [("가짜 비율 · BH-FDR", f"{a['bfdr']:.0%}", f"{(a['bfdr'] - a['nfdr']) * 100:+.0f}%p vs 보정 없음", "inverse",
                 "'효과 있음'으로 고른 것 중 실제로는 효과가 없는 비율 (20벌 합산)"),
                ("가짜 경보 · 평균", f"{a['bf']:.1f}개", f"{a['bf'] - a['nf']:+.1f} vs 보정 없음", "inverse", None),
                ("찾은 진짜 · 평균", f"{a['bh']:.1f} / {n_true}", f"{a['bh'] - a['nh']:+.1f} vs 보정 없음", "normal",
                 "보정의 대가: 진짜 효과도 조금 덜 찾는다")])
    ti = tuple(sorted(np.random.default_rng(seed + 1000).choice(n_items, size=min(n_true, n_items), replace=False).tolist()))
    r = bh_run(seed=seed, n_stores=n_stores, n_items=n_items, true_items=ti, effect_range=(eff * .6, eff)).reset_index(drop=True)
    with right:
        with st.container(border=True):
            fig, h = waffle(r)
            show(fig, h)
            st.caption("한 칸 = 검정한 항목 하나. 파랑 채움 = 찾은 진짜 · 빨강 = 가짜 경보 · 파랑 테두리 = 놓친 진짜 · 회색 = 효과 없음(제대로 걸러냄)")
        with st.expander("통계 그림으로 보기 (정렬한 p값과 BH 기준선)"):
            r["rank"] = np.arange(1, len(r) + 1)
            fig = go.Figure()
            for lab, m, col in [("진짜 효과", r["truly_effective"], C["true"]), ("효과 없음", ~r["truly_effective"], C["rest"])]:
                fig.add_trace(go.Scatter(x=r["rank"][m], y=r["p"][m], mode="markers", name=lab, marker=dict(size=8, color=col)))
            fig.add_trace(go.Scatter(x=r["rank"], y=r["rank"] / len(r) * .10, mode="lines", showlegend=False,
                                     line=dict(color=C["accent"], width=2)))
            fig.add_hline(y=.05, line_dash="dot", line_color=C["muted"])
            fig.add_annotation(x=len(r), y=np.log10(.05), text="p = 0.05", showarrow=False, xanchor="right", yanchor="bottom",
                               font=dict(color=C["muted"], size=12))
            fig.add_annotation(x=len(r), y=np.log10(.10), text="BH 기준선 k/m·q", showarrow=False, xanchor="right", yanchor="top",
                               font=dict(color=C["accent"], size=12))
            fig.update_yaxes(type="log", title="p값 (로그)", exponentformat="power")
            fig.update_xaxes(title="p값 순위 (작은 것부터)")
            show(fig, 360)
    with st.expander("방법과 가정"):
        st.markdown("- 가상 매장의 항목별 점수를 고/저 두 집단으로 나누고, 매출 변화율을 **Mann-Whitney U**로 비교합니다(분포 가정 없음).\n"
                    "- 보정 없음은 p<0.05를, BH-FDR은 Benjamini-Hochberg 조정 p<0.10을 '효과 있음'으로 봅니다.\n"
                    "- 실제 업무에서는 담당자 배정이 무작위가 아니어서 인과가 아니라 **연관성**으로만 해석했습니다.\n"
                    "- 모든 값은 가상 데이터이며 실제 업무 성능과 무관합니다.")


# ================================================================ ② 요일 이상탐지
def anomaly(seed, lift, noise, shock, ksig):
    rg = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=168, freq="D")
    wd = days.day_of_week
    base = 100 * (1 + np.where(wd >= 5, lift / 100, 0)) * (1 + 0.0006 * np.arange(len(days)))
    y = base * (1 + rg.normal(0, noise / 100, len(days)))
    cand = np.arange(60, len(days))
    up = rg.choice(cand[wd[cand] < 5], 3, replace=False)
    down = rg.choice(cand[wd[cand] >= 5], 3, replace=False)
    y[up] *= 1 + shock / 100; y[down] *= 1 - shock / 100
    truth = np.zeros(len(days), bool); truth[np.r_[up, down]] = True
    s = pd.Series(y, index=days)
    g_mu, g_sd = s.shift(1).rolling(56, min_periods=28).mean(), s.shift(1).rolling(56, min_periods=28).std()
    by = pd.DataFrame({"y": y, "dow": wd}, index=days).groupby("dow")["y"]
    d_mu = by.transform(lambda v: v.shift(1).rolling(12, min_periods=3).mean())
    d_sd = by.transform(lambda v: v.shift(1).rolling(12, min_periods=3).std())
    valid = np.arange(len(days)) >= 60
    fg = valid & (np.abs(y - g_mu) > ksig * g_sd).to_numpy()
    fd = valid & (np.abs(y - d_mu) > ksig * d_sd).to_numpy()
    return days, y, truth, (g_mu, g_sd, fg), (d_mu, d_sd, fd)


@st.cache_data
def anomaly_avg(lift, noise, shock, ksig, runs=20):
    acc = np.zeros(4)
    for k in range(runs):
        _, _, t, (_, _, fg), (_, _, fd) = anomaly(k, lift, noise, shock, ksig)
        acc += [(fg & t).sum(), (fg & ~t).sum(), (fd & t).sum(), (fd & ~t).sum()]
    return acc / runs


def demo_anomaly():
    head, kpi = st.container(), st.container()
    st.session_state.setdefault("seed2", 3)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.container(border=True):
            lift = st.slider("주말 매출 증가(%)", 0, 100, 45, 5)
            ksig = st.slider("정상 범위 폭 (±kσ)", 1.0, 3.5, 2.5, 0.1, help="넓히면 오탐이 줄고, 좁히면 더 많이 잡지만 오탐이 는다")
            with st.expander("조건 더 바꾸기"):
                noise = st.slider("평소 흔들림(%)", 2, 20, 6)
                shock = st.slider("심은 이상의 크기(%)", 10, 80, 35, 5)
            if st.button("다른 가상 데이터", icon=":material/casino:", width="stretch", key="b2"):
                st.session_state.seed2 += 1
            st.caption(f"가상 데이터 시드 {st.session_state.seed2}")
    days, y, truth, (g_mu, g_sd, fg), (d_mu, d_sd, fd) = anomaly(st.session_state.seed2, lift, noise, shock, ksig)
    T = int(truth.sum()); hg, hd = int((fg & truth).sum()), int((fd & truth).sum())
    ofg, ofd = int((fg & ~truth).sum()), int((fd & ~truth).sum())
    av = anomaly_avg(lift, noise, shock, ksig)
    write_head(head, f"② 요일 이상탐지 · 가상 데이터 · <a href='{REPO}/tree/main/03_store_clustering_anomaly'>03 폴더</a>",
               f"요일을 무시하면 심어 둔 이상 {T}개 중 <b>{hg}개</b>, 같은 요일과 비교하면 <b>{hd}개</b>를 잡습니다.",
               f"대가: 오탐 {ofd}건 — 정상 범위 폭(kσ)으로 조절하고, 실무에서는 사람이 대조했습니다. "
               f"20벌 평균: 전체 기간 {av[0]:.1f}개 잡음·오탐 {av[1]:.1f} / 요일별 {av[2]:.1f}개 잡음·오탐 {av[3]:.1f}")
    cards(kpi, [("잡은 이상 · 요일별", f"{hd} / {T}", f"{hd - hg:+d} vs 전체 기간", "normal", None),
                ("오탐 · 요일별", f"{ofd}건", f"{ofd - ofg:+d} vs 전체 기간", "inverse", "정상인 날을 이상으로 잡은 수"),
                ("놓친 이상 · 전체 기간", f"{T - hg} / {T}", None, "off", "요일 효과 때문에 밴드가 넓어져 놓친 수")])
    with right:
        with st.container(border=True):
            fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=.13, subplot_titles=(
                f"전체 기간 밴드 · 잡음 {hg}/{T} · 오탐 {ofg}", f"요일별 밴드 · 잡음 {hd}/{T} · 오탐 {ofd}"))
            for row, (mu, sd, f) in enumerate([(g_mu, g_sd, fg), (d_mu, d_sd, fd)], 1):
                leg = row == 1
                fig.add_trace(go.Scatter(x=days, y=mu + ksig * sd, line_width=0, hoverinfo="skip", showlegend=False), row, 1)
                fig.add_trace(go.Scatter(x=days, y=mu - ksig * sd, fill="tonexty", fillcolor="rgba(235,219,178,.10)", line_width=0,
                                         name=f"정상 범위 ±{ksig}σ", legendgroup="b", showlegend=leg, hoverinfo="skip"), row, 1)
                fig.add_trace(go.Scatter(x=days, y=y, mode="lines", line=dict(color=C["muted"], width=1.4),
                                         name="일매출(가상)", legendgroup="y", showlegend=leg), row, 1)
                for nm, m, mk in [("잡음", f & truth, dict(symbol="circle", size=10, color=C["true"], line=dict(color=C["surface"], width=2))),
                                  ("놓침", ~f & truth, dict(symbol="circle-open", size=12, color=C["true"], line=dict(width=2, color=C["true"]))),
                                  ("오탐", f & ~truth, dict(symbol="x-thin", size=9, line=dict(color=C["false"], width=2)))]:
                    fig.add_trace(go.Scatter(x=days[m], y=y[m], mode="markers", marker=mk, name=nm, legendgroup=nm, showlegend=leg), row, 1)
                fig.add_vrect(x0=days[0], x1=days[59], fillcolor=C["border"], opacity=.35, line_width=0, row=row, col=1)
            fig.add_annotation(x=days[30], y=1.0, yref="paper", yanchor="bottom", text="기준을 쌓는 기간", showarrow=False,
                               font=dict(size=11, color=C["muted"]))
            fig.update_layout(hovermode="x unified")
            show(fig, 520)
    with st.expander("방법과 가정"):
        st.markdown("- **요일별 밴드**: 같은 요일 과거 12주(n≥3)의 평균 ± kσ. **전체 기간 밴드**: 과거 56일 전체의 평균 ± kσ. 둘 다 당일 값은 쓰지 않습니다(shift 1).\n"
                    "- ±kσ 밴드는 정상인 날도 일부를 이상으로 잡습니다(정규분포면 ±1.5σ 밖이 약 13%). 실제 업무에서는 ±1.5σ를 시작으로 편차 크기에 따라 심각도를 나누고, 프로모션 일정과 사람이 대조했습니다.\n"
                    "- 모든 값은 가상 데이터입니다.")


# ================================================================ ③ 서울 핫스팟
@st.cache_data
def load_seoul():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "data")) if f.startswith("seoul_dong_pop_"))
    pop = pd.read_csv(os.path.join(HERE, "data", files[-1]), dtype={"dong_code": str})
    with open(os.path.join(HERE, "data", "seoul_dong.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    return pop, gj, files[-1].split("_")[-1][:6]


@st.cache_data
def lisa(day_type, hour, k):
    pop, _, _ = load_seoul()
    d = pop[(pop["day_type"] == day_type) & (pop["hour"] == hour)].reset_index(drop=True)
    d["density"] = d["pop"] / d["area_km2"]
    x = np.log1p(d["density"].to_numpy())
    xy = np.c_[d["lon"].to_numpy() * np.cos(np.radians(37.55)) * 111.0, d["lat"].to_numpy() * 111.0]
    nb = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    z = (x - x.mean()) / x.std(); lag = z[nb].mean(axis=1)
    I = float((z * lag).sum() / (z * z).sum())
    rng = np.random.default_rng(hour)
    sims = np.array([(z * rng.permutation(z)[nb].mean(axis=1)).sum() / (z * z).sum() for _ in range(999)])
    pg = (1 + (np.abs(sims) >= abs(I)).sum()) / 1000
    n = len(z); li = z * lag
    others = np.array([np.r_[0:i, i + 1:n] for i in range(n)])
    cnt = np.zeros(n)
    for _ in range(199):
        pick = others[np.arange(n)[:, None], rng.integers(0, n - 1, size=(n, k))]
        cnt += np.abs(z * z[pick].mean(axis=1)) >= np.abs(li)
    pl = (cnt + 1) / 200
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    d["사분면"] = quad
    d["군집"] = np.select([(pl < .05) & (quad == "HH"), (pl < .05) & (quad == "LL"), pl < .05],
                        ["핫스팟 (HH)", "콜드스팟 (LL)", "섞임 (HL·LH)"], "유의하지 않음")
    return d, I, pg


PRESET = {"새벽 4시": 4, "출근 9시": 9, "오후 2시": 14, "밤 10시": 22}


def demo_seoul():
    pop, gj, month = load_seoul()
    head, kpi = st.container(), st.container()
    st.session_state.setdefault("hour", 14)
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.container(border=True):
            day_type = st.segmented_control("요일", ["주중", "주말"], default="주중") or "주중"
            st.pills("시간 바로가기", list(PRESET), key="hp",
                     on_change=lambda: st.session_state.update(hour=PRESET[st.session_state.hp]) if st.session_state.hp else None)
            hour = st.slider("시간대", 0, 23, key="hour")
            view = st.segmented_control("지도", ["핫스팟", "밀도"], default="핫스팟") or "핫스팟"
            with st.expander("조건 더 바꾸기"):
                k = st.slider("이웃 수 (KNN)", 4, 10, 6)
    d, I, pg = lisa(day_type, hour, k)
    n_more = max(int(round(pg * 1000)) - 1, 0)
    write_head(head, f"③ 서울 핫스팟 · 서울시 공공데이터 {month[:4]}-{month[4:]} · <a href='{REPO}/tree/main/02_spatial_esda'>02 폴더</a>",
               f"{day_type} {hour}시 — 사람이 몰린 동끼리 <b>붙어 있습니다</b>. 무작위로 섞어 본 999번 중 이만큼 뭉친 경우는 <b>{n_more}번</b>.",
               "시간을 새벽 ↔ 오후로 바꿔 보세요. 낮의 업무지구와 밤의 주거지가 뒤바뀝니다.")
    hh, ll = int((d["군집"] == "핫스팟 (HH)").sum()), int((d["군집"] == "콜드스팟 (LL)").sum())
    cards(kpi, [("뭉침 정도 · Moran's I", f"{I:.2f}", None, "off", "무작위로 흩어져 있으면 0 근처, 비슷한 동끼리 붙어 있을수록 1에 가까움"),
                ("핫스팟 동", f"{hh}개", None, "off", "주변까지 밀도가 높은 동 (LISA, p<0.05)"),
                ("콜드스팟 동", f"{ll}개", None, "off", "주변까지 밀도가 낮은 동 (LISA, p<0.05)")])
    with right:
        with st.container(border=True):
            common = dict(geojson=gj, locations="dong_code", hover_name="dong_name",
                          hover_data={"density": ":,.0f", "사분면": True, "dong_code": False},
                          map_style="carto-darkmatter-nolabels", center=dict(lat=37.56, lon=126.99), zoom=10.1, opacity=.85)
            if view == "핫스팟":
                fig = px.choropleth_map(d, color="군집", category_orders={"군집": ["핫스팟 (HH)", "콜드스팟 (LL)", "섞임 (HL·LH)", "유의하지 않음"]},
                                        color_discrete_map={"핫스팟 (HH)": C["false"], "콜드스팟 (LL)": C["true"],
                                                            "섞임 (HL·LH)": C["mixed"], "유의하지 않음": "rgba(80,73,69,.35)"}, **common)
            else:
                fig = px.choropleth_map(d, color=np.log10(d["density"]), color_continuous_scale=SEQ, **common)
                fig.update_layout(coloraxis_colorbar=dict(orientation="h", y=-.02, yanchor="top", thickness=10, len=.6,
                                                          tickvals=np.log10([1e3, 1e4, 1e5]), ticktext=["1천", "1만", "10만"],
                                                          title="밀도(명/㎢)"))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            show(fig, 520, scrollZoom=False)
            top = d[d["군집"] == "핫스팟 (HH)"].nlargest(6, "density")["dong_name"].tolist()
            if top:
                st.markdown(" ".join(f'<span class="chip">{n}</span>' for n in top), unsafe_allow_html=True)
            st.caption(f"출처: 서울특별시 열린데이터광장 — 행정동 단위 서울 생활인구(내국인) {month[:4]}-{month[4:]}, "
                       "서울시 상권분석서비스(영역-행정동). 공공누리 1유형(출처표시).")
    with st.expander("방법과 가정"):
        st.markdown(f"- 값: 행정동 × {day_type} × {hour}시 평균 생활인구 ÷ 면적, 로그 변환.\n"
                    f"- 가중치: 동 대표점 기준 KNN(k={k}), 행표준화. Global Moran's I는 순열 999회, LISA는 조건부 순열 199회, p<0.05만 색칠.\n"
                    "- 섞임(HL·LH)은 주변과 반대로 튀는 동입니다. 원래 사분면은 지도 위 hover에 남겨 두었습니다.\n"
                    "- 행정동 개편으로 코드가 바뀐 7개 동은 생활인구와 연결되지 않아 비어 있습니다.")


{"1": demo_bh, "2": demo_anomaly, "3": demo_seoul}[key]()
