"""신홍조 · 판단 데모 — 포트폴리오의 판단을 직접 바꿔 보는 화면.

① 다중검정 보정(가상 데이터) ② 요일 조건부 이상탐지(가상 데이터) ③ 서울 생활인구 핫스팟(서울시 공공데이터).
회사 데이터는 쓰지 않는다.  streamlit run demos/app/streamlit_app.py   (특정 데모: ?demo=1|2|3)

화면 틀(모든 데모 공통):  머리줄 → 데모 고르기 → [결론 문장] → [조건 칸 | 숫자 카드 · 차트 카드 · 방법] → 출처
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
C = dict(bg="#1d2021", surface="#282828", raise_="#32302f", border="#3c3836", grid="#32302f",
         text="#ebdbb2", strong="#fbf1c7", text2="#bdae93", muted="#a89984", accent="#fabd2f",
         good="#b8bb26", true="#4a90c2", false="#fb4934", rest="#7c6f64", mixed="#b16286")
SEQ = ["#7a5230", "#a8581a", "#d65d0e", "#fe8019", "#fabd2f"]        # 밀도: 주황 한 계열

st.markdown(f"""<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
html, body, .stMarkdown, button, input, label, p {{font-family:'Pretendard Variable',Pretendard,'Noto Sans KR',sans-serif;}}
.block-container {{padding:1.6rem 1.5rem 3rem; max-width:1180px;}}
.stMarkdown, .stCaption {{word-break:keep-all; overflow-wrap:break-word;}}
a {{color:{C['accent']};}}

/* 머리줄 */
.topbar {{display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
          margin-bottom:18px;}}
.prompt {{font-family:ui-monospace,Menlo,monospace; font-size:12px; color:{C['muted']};}}
.brand {{font-size:1.75rem; font-weight:750; color:{C['strong']}; line-height:1.25; margin:2px 0 6px;}}
.brand span {{color:{C['text2']}; font-weight:500;}}
.lede {{font-size:.98rem; color:{C['text2']}; line-height:1.6;}}
.links {{display:flex; gap:8px; flex-shrink:0;}}
.btn {{display:inline-block; font-size:13px; font-weight:600; padding:7px 14px; border-radius:8px; text-decoration:none !important;
       border:1px solid {C['border']}; background:{C['surface']}; color:{C['text']} !important;}}
.btn:hover {{border-color:{C['accent']};}}

/* 데모 머리 */
.sec {{margin:14px 0 18px;}}
.tags {{display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px;}}
.tag {{font-size:12px; padding:2px 9px; border-radius:6px; background:{C['raise_']}; color:{C['text2']}; text-decoration:none !important;}}
a.tag {{color:{C['accent']} !important;}}
.headline {{font-size:1.45rem; font-weight:700; line-height:1.45; color:{C['text']}; margin:0;}}
.headline b {{color:{C['accent']};}}
.sub {{color:{C['muted']}; font-size:.92rem; line-height:1.6; margin-top:6px;}}

/* 칸 제목 */
.ptitle {{font-size:15px; font-weight:700; color:{C['text']}; margin:0;}}
.psub {{font-size:12.5px; color:{C['muted']}; margin:2px 0 4px; line-height:1.5;}}
.rail-head {{padding-bottom:12px; margin-bottom:4px; border-bottom:1px solid {C['border']};}}

/* 숫자 카드 */
.kpis {{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:16px;}}
.kpi {{background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; padding:14px 16px 12px;}}
.k-label {{font-size:13px; font-weight:600; color:{C['text2']};}}
.k-val {{font-size:1.9rem; font-weight:700; color:{C['strong']}; line-height:1.2; margin:6px 0 4px; font-variant-numeric:tabular-nums;}}
.k-ref {{font-size:12.5px; color:{C['muted']};}}
.k-ref .good {{color:{C['good']}; font-weight:600;}} .k-ref .bad {{color:{C['false']}; font-weight:600;}}
.k-note {{font-size:12px; color:{C['muted']}; line-height:1.5; margin-top:8px; padding-top:8px; border-top:1px solid {C['border']};}}

/* 범례(차트 위) */
.lg {{display:flex; flex-wrap:wrap; gap:4px 16px; font-size:12.5px; color:{C['text2']}; margin:6px 0 0;}}
.lg i {{display:inline-block; width:11px; height:11px; margin-right:6px; vertical-align:-1px; font-style:normal;}}
.chips {{display:flex; flex-wrap:wrap; gap:6px; align-items:center; font-size:12.5px; color:{C['muted']};}}
.chip {{font-size:12px; padding:2px 9px; border-radius:999px; border:1px solid {C['border']}; color:{C['text2']};}}

.foot {{margin-top:28px; padding-top:14px; border-top:1px solid {C['border']}; font-size:12px; color:{C['muted']}; line-height:1.7;}}
[data-testid="stExpander"] details {{border-color:{C['border']};}}

@media (max-width: 640px) {{
  .block-container {{padding:1rem 1rem 2rem;}}
  .topbar {{flex-direction:column; align-items:flex-start;}}
  .brand {{font-size:1.4rem;}} .headline {{font-size:1.15rem;}}
  .kpi {{padding:10px 10px 9px;}} .k-label {{font-size:11.5px;}} .k-val {{font-size:1.3rem;}} .k-ref, .k-note {{font-size:11px;}}
  .kpis {{gap:8px;}} .k-note {{display:none;}} .sec {{margin:10px 0 14px;}}
  [data-testid="stButtonGroup"] button p {{font-size:12.5px;}}
}}
</style>""", unsafe_allow_html=True)

pio.templates["gruv"] = go.layout.Template(layout=dict(
    font=dict(family="Pretendard Variable, Pretendard, 'Noto Sans KR', sans-serif", size=12.5, color=C["text2"]),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=4, r=4, t=28, b=4), showlegend=False,
    xaxis=dict(gridcolor=C["grid"], linecolor=C["border"], zeroline=False, ticks="", automargin=True),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["border"], zeroline=False, ticks="", automargin=True),
    hoverlabel=dict(bgcolor=C["bg"], bordercolor=C["border"], font_color=C["text"])))
pio.templates.default = "gruv"


def show(fig, h=380, **cfg):
    fig.update_layout(height=h)
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False, **cfg})


def left_titles(fig, axes):
    """subplot 제목을 각 칸 왼쪽 끝에 맞춘다(가운데 정렬은 차트 칸 경계와 어긋나 보임)."""
    for a, ax in zip(fig.layout.annotations, axes):
        a.update(x=ax.domain[0], xanchor="left", align="left", font=dict(size=13, color=C["text"]))


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def head(box, tags, headline, sub=None):
    t = "".join(f'<a class="tag" href="{u}">{x} ↗</a>' if u else f'<span class="tag">{x}</span>' for x, u in tags)
    with box:
        html(f'<div class="sec"><div class="tags">{t}</div><div class="headline">{headline}</div>'
             + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def kpis(box, items):
    """items: (제목, 값, 비교 화살표, 좋음/나쁨, 비교 대상, 설명)"""
    cells = []
    for label, value, arrow, tone, ref, note in items:
        r = (ref or "") + (f' 대비 <span class="{tone}">{arrow}</span>' if arrow else "")
        cells.append(f'<div class="kpi"><div class="k-label">{label}</div><div class="k-val">{value}</div>'
                     f'<div class="k-ref">{r}</div>' + (f'<div class="k-note">{note}</div>' if note else "") + "</div>")
    with box:
        html(f'<div class="kpis">{"".join(cells)}</div>')


def diff(d, fmt, unit, better):
    if round(d, 6) == 0:
        return "＝", "k-ref"
    good = (d < 0) == (better == "lower")
    return f"{'↓' if d < 0 else '↑'} {abs(d):{fmt}}{unit}", "good" if good else "bad"


def legend(items):
    """items: (모양, 색, 이름). 모양 fill|ring|dot|dotring|x|band|line"""
    sty = {"fill": "background:{c};border-radius:2px", "ring": "border:2px solid {c};border-radius:2px;width:7px;height:7px",
           "dot": "background:{c};border-radius:50%", "dotring": "border:2px solid {c};border-radius:50%;width:7px;height:7px",
           "band": "background:{c};border-radius:2px", "line": "background:{c};height:2px;vertical-align:3px"}
    out = []
    for shape, c, name in items:
        mark = f'<i style="color:{c};width:auto;font-weight:700">✕</i>' if shape == "x" else f'<i style="{sty[shape].format(c=c)}"></i>'
        out.append(f"<span>{mark}{name}</span>")
    html(f'<div class="lg">{"".join(out)}</div>')


def panel_title(title, sub=None):
    html(f'<div class="ptitle">{title}</div>' + (f'<div class="psub">{sub}</div>' if sub else ""))


def layout():
    """결론 자리 · [조건 | 결과] 두 칸. 휴대폰에서는 결론 → 조건 → 결과 순으로 쌓인다."""
    h = st.container()
    rail, main = st.columns([1, 3], gap="medium")
    with rail:
        box = st.container(border=True)
    with box:
        html('<div class="rail-head"><div class="ptitle">조건 바꾸기</div><div class="psub">움직이면 위 문장과 숫자가 바로 바뀝니다</div></div>')
    with main:
        k = st.container()
    return h, box, main, k


# ---------------------------------------------------------------- 머리줄 + 데모 고르기
html(f'<div class="topbar"><div><div class="prompt">~/portfolio/demos $ ./run</div>'
     '<div class="brand">신홍조 <span>· 데이터 사이언티스트</span></div>'
     '<div class="lede">"유의하다"는 결과를 믿기 전에 <b>우연·요일·공간</b> 때문에 생긴 착시인지 먼저 확인합니다. '
     '세 가지 판단을 직접 바꿔 보세요. 회사 데이터는 쓰지 않았습니다.</div></div>'
     f'<div class="links"><a class="btn" href="{REPO}">포트폴리오 ↗</a><a class="btn" href="{REPO}/tree/main/demos/app">코드 ↗</a></div></div>')

DEMOS = {"1": "① 다중검정 보정", "2": "② 요일 이상탐지", "3": "③ 서울 핫스팟"}
qp = st.query_params.get("demo", "1")
choice = st.segmented_control("데모 고르기", list(DEMOS.values()), default=DEMOS.get(qp, DEMOS["1"]),
                              label_visibility="collapsed", width="stretch") or DEMOS["1"]
key = [k for k, v in DEMOS.items() if v == choice][0]
if st.query_params.get("demo") != key:
    st.query_params["demo"] = key


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


LAB = {"hit": "찾은 진짜", "fp": "가짜 경보", "miss": "놓친 진짜", "rest": "효과 없음 · 제대로 걸러냄"}
MK = {"hit": dict(symbol="square", color=C["true"]), "fp": dict(symbol="square", color=C["false"]),
      "miss": dict(symbol="square-open", color=C["true"], line=dict(width=2, color=C["true"])),
      "rest": dict(symbol="square", color=C["rest"], opacity=.45)}


def waffle(r, cols=10):
    t = r["truly_effective"].to_numpy()
    tt = [f"{name}<br><span style='color:{C['muted']};font-size:12px'>찾음 {(r[c].to_numpy() & t).sum()}/{t.sum()} · 가짜 {(r[c].to_numpy() & ~t).sum()}</span>"
          for name, c in [("보정 없음 (p<0.05)", "naive_sig"), ("BH-FDR (q=0.10)", "bh_sig")]]
    fig = make_subplots(1, 2, horizontal_spacing=.06, subplot_titles=tt)
    for j, col in enumerate(["naive_sig", "bh_sig"], 1):
        s = r[col].to_numpy()
        o = np.select([s & t, s & ~t, ~s & t], ["hit", "fp", "miss"], "rest")
        idx = np.argsort(pd.Categorical(o, list(LAB)).codes, kind="stable"); o = o[idx]
        i = np.arange(len(o)); x, y = i % cols, -(i // cols)
        for k in LAB:
            m = o == k
            fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="markers", name=LAB[k], marker=dict(size=14, **MK[k]),
                                     customdata=np.c_[r["item"].to_numpy()[idx][m], r["p"].to_numpy()[idx][m]],
                                     hovertemplate="%{customdata[0]} · p=%{customdata[1]:.4f}<extra>" + LAB[k] + "</extra>"), 1, j)
    fig.update_xaxes(visible=False, range=[-.6, cols - .4]); fig.update_yaxes(visible=False)
    left_titles(fig, [fig.layout.xaxis, fig.layout.xaxis2])
    fig.update_layout(margin=dict(t=46))
    return fig, 62 + 25 * int(np.ceil(len(r) / cols))


def demo_bh():
    h, box, main, k = layout()
    st.session_state.setdefault("seed1", 7)
    with box:
        n_items = st.slider("검정한 항목 수", 10, 100, 60, 5, help="마케팅 점검 항목처럼, 하나씩 따로 검정하는 항목의 개수")
        n_true = st.slider("진짜 효과가 있는 항목", 0, 20, 8)
        with st.expander("조건 더 바꾸기"):
            n_stores = st.slider("매장 수", 40, 400, 120, 20)
            eff = st.slider("효과 크기(최대)", 0.2, 1.5, 1.0, 0.1)
        if st.button("다른 가상 데이터로", icon=":material/casino:", width="stretch"):
            st.session_state.seed1 += 1
        st.caption(f"지금 그림: 가상 데이터 {st.session_state.seed1}번")
    seed = st.session_state.seed1
    a = bh_avg(n_items, n_true, n_stores, eff)
    head(h, [("가상 데이터", None), ("04 폴더", f"{REPO}/tree/main/04_effect_testing_fdr")],
         f"보정 없이 고르면 '효과 있음' 중 <b>{a['nfdr']:.0%}가 가짜</b>입니다. BH-FDR은 이를 <b>{a['bfdr']:.0%}</b>로 줄이고, "
         f"진짜는 조금 덜 찾습니다.",
         "숫자는 가상 데이터 20벌의 평균입니다(한 번의 우연에 기대지 않기 위해). 아래 그림은 그중 한 벌.")
    d1, t1 = diff((a["bfdr"] - a["nfdr"]) * 100, ".0f", "%p", "lower")
    d2, t2 = diff(a["bf"] - a["nf"], ".1f", "개", "lower")
    d3, t3 = diff(a["bh"] - a["nh"], ".1f", "개", "higher")
    kpis(k, [("가짜 비율 · BH-FDR", f"{a['bfdr']:.0%}", d1, t1, f"보정 없음 {a['nfdr']:.0%}", "'효과 있음'으로 고른 것 중 실제로는 효과가 없는 비율"),
             ("가짜 경보 · 평균", f"{a['bf']:.1f}개", d2, t2, f"보정 없음 {a['nf']:.1f}개", "효과가 없는데 '있음'으로 잘못 고른 항목 수"),
             ("찾은 진짜 · 평균", f"{a['bh']:.1f} / {n_true}", d3, t3, f"보정 없음 {a['nh']:.1f}개", "보정의 대가: 진짜 효과도 조금 덜 찾는다")])
    ti = tuple(sorted(np.random.default_rng(seed + 1000).choice(n_items, size=min(n_true, n_items), replace=False).tolist()))
    r = bh_run(seed=seed, n_stores=n_stores, n_items=n_items, true_items=ti, effect_range=(eff * .6, eff)).reset_index(drop=True)
    with main:
        with st.container(border=True):
            panel_title("같은 항목을 두 기준으로 골랐을 때", f"한 칸 = 검정한 항목 하나 · 가상 데이터 {seed}번")
            legend([("fill", C["true"], LAB["hit"]), ("fill", C["false"], LAB["fp"]),
                    ("ring", C["true"], LAB["miss"]), ("fill", "rgba(124,111,100,.45)", LAB["rest"])])
            fig, hh = waffle(r)
            show(fig, hh)
        with st.expander("통계 그림으로 보기 — 정렬한 p값과 BH 기준선"):
            r["rank"] = np.arange(1, len(r) + 1)
            legend([("dot", C["true"], "진짜 효과"), ("dot", C["rest"], "효과 없음"), ("line", C["accent"], "BH 기준선 k/m·q"),
                    ("line", C["muted"], "p = 0.05")])
            fig = go.Figure()
            for m, col in [(r["truly_effective"], C["true"]), (~r["truly_effective"], C["rest"])]:
                fig.add_trace(go.Scatter(x=r["rank"][m], y=r["p"][m], mode="markers", marker=dict(size=8, color=col)))
            fig.add_trace(go.Scatter(x=r["rank"], y=r["rank"] / len(r) * .10, mode="lines", line=dict(color=C["accent"], width=2)))
            fig.add_hline(y=.05, line_dash="dot", line_color=C["muted"])
            fig.update_yaxes(type="log", title="p값 (로그)", exponentformat="power")
            fig.update_xaxes(title="p값 순위 (작은 것부터)")
            fig.update_layout(margin=dict(t=8))
            show(fig, 340)
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
    h, box, main, k = layout()
    st.session_state.setdefault("seed2", 3)
    with box:
        lift = st.slider("주말 매출 증가(%)", 0, 100, 45, 5, help="요일 효과의 크기. 0이면 두 방법의 차이가 거의 없다")
        ksig = st.slider("정상 범위 폭 (±kσ)", 1.0, 3.5, 2.5, 0.1, help="넓히면 오탐이 줄고, 좁히면 더 많이 잡지만 오탐이 는다")
        with st.expander("조건 더 바꾸기"):
            noise = st.slider("평소 흔들림(%)", 2, 20, 6)
            shock = st.slider("심은 이상의 크기(%)", 10, 80, 35, 5)
        if st.button("다른 가상 데이터로", icon=":material/casino:", width="stretch", key="b2"):
            st.session_state.seed2 += 1
        st.caption(f"지금 그림: 가상 데이터 {st.session_state.seed2}번")
    days, y, truth, (g_mu, g_sd, fg), (d_mu, d_sd, fd) = anomaly(st.session_state.seed2, lift, noise, shock, ksig)
    T = int(truth.sum()); hg, hd = int((fg & truth).sum()), int((fd & truth).sum())
    ofg, ofd = int((fg & ~truth).sum()), int((fd & ~truth).sum())
    av = anomaly_avg(lift, noise, shock, ksig)
    head(h, [("가상 데이터", None), ("03 폴더", f"{REPO}/tree/main/03_store_clustering_anomaly")],
         f"요일을 무시하면 심어 둔 이상 {T}개 중 <b>{hg}개</b>, 같은 요일과 비교하면 <b>{hd}개</b>를 잡습니다.",
         f"대가는 오탐 {ofd}건입니다. 정상 범위 폭(kσ)으로 조절하고, 실무에서는 걸린 날을 사람이 프로모션 일정과 대조했습니다.")
    d1, t1 = diff(hd - hg, "d", "개", "higher")
    d2, t2 = diff(ofd - ofg, "d", "건", "lower")
    kpis(k, [("잡은 이상 · 요일별", f"{hd} / {T}", d1, t1, f"전체 기간 {hg} / {T}", "같은 요일 과거 12주와 비교"),
             ("오탐 · 요일별", f"{ofd}건", d2, t2, f"전체 기간 {ofg}건", "정상인 날을 이상으로 잡은 수"),
             ("20벌 평균 · 요일별", f"{av[2]:.1f} / {T}", None, None, f"전체 기간 {av[0]:.1f} / {T}",
              f"평균 오탐: 요일별 {av[3]:.1f}건 · 전체 기간 {av[1]:.1f}건")])
    with main:
        with st.container(border=True):
            panel_title("두 기준의 정상 범위 비교", f"위: 과거 56일 전체 · 아래: 같은 요일 과거 12주 · 왼쪽 음영은 기준을 쌓는 기간(판정 안 함) · 가상 데이터 {st.session_state.seed2}번")
            legend([("line", C["muted"], "일매출(가상)"), ("band", "rgba(235,219,178,.22)", f"정상 범위 ±{ksig}σ"),
                    ("dot", C["true"], "잡은 이상"), ("dotring", C["true"], "놓친 이상"), ("x", C["false"], "오탐")])
            fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=.12, subplot_titles=(
                f"요일 무시 (전체 기간) · 잡음 {hg}/{T} · 오탐 {ofg}", f"같은 요일 비교 · 잡음 {hd}/{T} · 오탐 {ofd}"))
            for row, (mu, sd, f) in enumerate([(g_mu, g_sd, fg), (d_mu, d_sd, fd)], 1):
                fig.add_trace(go.Scatter(x=days, y=mu + ksig * sd, line_width=0, hoverinfo="skip"), row, 1)
                fig.add_trace(go.Scatter(x=days, y=mu - ksig * sd, fill="tonexty", fillcolor="rgba(235,219,178,.10)",
                                         line_width=0, hoverinfo="skip"), row, 1)
                fig.add_trace(go.Scatter(x=days, y=y, mode="lines", line=dict(color=C["muted"], width=1.4), name="일매출",
                                         hovertemplate="%{x|%m-%d (%a)} · %{y:.0f}<extra></extra>"), row, 1)
                for nm, m, mk in [("잡은 이상", f & truth, dict(symbol="circle", size=10, color=C["true"], line=dict(color=C["surface"], width=2))),
                                  ("놓친 이상", ~f & truth, dict(symbol="circle-open", size=12, color=C["true"], line=dict(width=2, color=C["true"]))),
                                  ("오탐", f & ~truth, dict(symbol="x-thin", size=9, line=dict(color=C["false"], width=2)))]:
                    fig.add_trace(go.Scatter(x=days[m], y=y[m], mode="markers", marker=mk, name=nm,
                                             hovertemplate="%{x|%m-%d (%a)}<extra>" + nm + "</extra>"), row, 1)
                fig.add_vrect(x0=days[0], x1=days[59], fillcolor=C["border"], opacity=.35, line_width=0, row=row, col=1)
            left_titles(fig, [fig.layout.xaxis, fig.layout.xaxis2])
            fig.update_layout(margin=dict(t=26))
            show(fig, 500)
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
CLS = {"핫스팟 (HH)": C["false"], "콜드스팟 (LL)": C["true"], "섞임 (HL·LH)": C["mixed"], "유의하지 않음": "rgba(80,73,69,.35)"}


def demo_seoul():
    pop, gj, month = load_seoul()
    ym = f"{month[:4]}-{month[4:]}"
    h, box, main, k = layout()
    st.session_state.setdefault("hour", 14)
    with box:
        day_type = st.segmented_control("요일", ["주중", "주말"], default="주중", width="stretch") or "주중"
        st.pills("시간 바로가기", list(PRESET), key="hp",
                 on_change=lambda: st.session_state.update(hour=PRESET[st.session_state.hp]) if st.session_state.hp else None)
        hour = st.slider("시간대", 0, 23, key="hour", format="%d시")
        view = st.segmented_control("지도", ["핫스팟", "밀도"], default="핫스팟", width="stretch") or "핫스팟"
        with st.expander("조건 더 바꾸기"):
            knn = st.slider("이웃 수 (KNN)", 4, 10, 6)
    d, I, pg = lisa(day_type, hour, knn)
    n_more = max(int(round(pg * 1000)) - 1, 0)
    head(h, [(f"서울시 공공데이터 {ym}", None), ("02 폴더", f"{REPO}/tree/main/02_spatial_esda")],
         f"{day_type} {hour}시 — 사람이 몰린 동끼리 <b>붙어 있습니다</b>. 무작위로 섞어 본 999번 중 이만큼 뭉친 경우는 <b>{n_more}번</b>.",
         "시간을 새벽 ↔ 오후로 바꿔 보세요. 낮의 업무지구와 밤의 주거지가 뒤바뀝니다.")
    hh, ll = int((d["군집"] == "핫스팟 (HH)").sum()), int((d["군집"] == "콜드스팟 (LL)").sum())
    kpis(k, [("뭉침 정도 · Moran's&nbsp;I", f"{I:.2f}", None, None, f"무작위 999번 중 이 이상 {n_more}번",
              "흩어져 있으면 0 근처, 비슷한 동끼리 붙을수록 1에 가까움"),
             ("핫스팟 동", f"{hh}개", None, None, f"전체 {len(d)}개 동 중", "자신과 주변이 모두 밀도가 높은 동 (p<0.05)"),
             ("콜드스팟 동", f"{ll}개", None, None, f"전체 {len(d)}개 동 중", "자신과 주변이 모두 밀도가 낮은 동 (p<0.05)")])
    with main:
        with st.container(border=True):
            if view == "핫스팟":
                panel_title(f"행정동 핫스팟 · {day_type} {hour}시", "LISA로 유의한(p<0.05) 동만 색칠 · 마우스를 올리면 동 이름과 밀도")
                legend([("fill", c, n) for n, c in CLS.items()])
            else:
                panel_title(f"행정동 인구 밀도 · {day_type} {hour}시", "평균 생활인구 ÷ 면적(명/㎢), 로그 눈금 · 진할수록 밀도가 높음")
            common = dict(geojson=gj, locations="dong_code", hover_name="dong_name",
                          hover_data={"density": ":,.0f", "사분면": True, "dong_code": False},
                          map_style="carto-darkmatter-nolabels", center=dict(lat=37.56, lon=126.99), zoom=10.1, opacity=.85)
            if view == "핫스팟":
                fig = px.choropleth_map(d, color="군집", category_orders={"군집": list(CLS)}, color_discrete_map=CLS, **common)
            else:
                fig = px.choropleth_map(d, color=np.log10(d["density"]), color_continuous_scale=SEQ, **common)
                fig.update_layout(coloraxis_colorbar=dict(orientation="h", x=.02, xanchor="left", y=.03, yanchor="bottom",
                                                          thickness=10, len=.45, bgcolor="rgba(29,32,33,.85)",
                                                          tickvals=np.log10([1e3, 1e4, 1e5]), ticktext=["1천", "1만", "10만"],
                                                          title=dict(text="명/㎢", side="top")))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            show(fig, 500, scrollZoom=False)
            top = d[d["군집"] == "핫스팟 (HH)"].nlargest(6, "density")["dong_name"].tolist()
            if top:
                html('<div class="chips"><span>밀도 높은 핫스팟</span>' + "".join(f'<span class="chip">{n}</span>' for n in top) + "</div>")
        with st.expander("방법과 가정"):
            st.markdown(f"- 값: 행정동 × {day_type} × {hour}시 평균 생활인구 ÷ 면적, 로그 변환.\n"
                        f"- 가중치: 동 대표점 기준 KNN(k={knn}), 행표준화. Global Moran's I는 순열 999회, LISA는 조건부 순열 199회, p<0.05만 색칠.\n"
                        "- 섞임(HL·LH)은 주변과 반대로 튀는 동입니다. 원래 사분면은 지도 위 hover에 남겨 두었습니다.\n"
                        "- 행정동 개편으로 코드가 바뀐 7개 동은 생활인구와 연결되지 않아 비어 있습니다.")
        st.caption(f"출처: 서울특별시 열린데이터광장 — 행정동 단위 서울 생활인구(내국인) {ym}, "
                   "서울시 상권분석서비스(영역-행정동). 공공누리 1유형(출처표시).")


{"1": demo_bh, "2": demo_anomaly, "3": demo_seoul}[key]()

html(f'<div class="foot">신홍조 · 데이터 사이언티스트 — ①② 가상 데이터, ③ 서울특별시 열린데이터광장(공공누리 1유형). 회사 데이터는 쓰지 않았습니다. '
     f'<a href="{REPO}">포트폴리오</a> · <a href="{REPO}/tree/main/demos/app">코드(MIT)</a></div>')
