import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy import stats
import io
import sqlite3
import tempfile
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdmitIQ — Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS Injection ────────────────────────────────────────────────────────────
_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:#07090f!important;color:#dde1f0!important}
.block-container{padding:0 2.5rem 4rem!important;max-width:1280px!important}
.hero{text-align:center;padding:3.2rem 2rem 2rem;position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;inset:0;background:radial-gradient(ellipse 65% 60% at 50% 0%,rgba(99,102,241,.22) 0%,transparent 70%),radial-gradient(ellipse 40% 35% at 15% 90%,rgba(236,72,153,.10) 0%,transparent 60%),radial-gradient(ellipse 40% 35% at 85% 90%,rgba(34,211,238,.09) 0%,transparent 60%);pointer-events:none}
.tag{display:inline-block;font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#818cf8;border:1px solid rgba(129,140,248,.3);padding:5px 16px;border-radius:999px;margin-bottom:1.4rem;position:relative;z-index:1}
.hero-title{font-family:'Playfair Display',serif;font-size:clamp(2.8rem,6vw,5rem);font-weight:900;line-height:1.06;letter-spacing:-1px;position:relative;z-index:1;background:linear-gradient(135deg,#fff 30%,#a5b4fc 65%,#f0abfc 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:.95rem;color:#64748b;margin-top:.9rem;font-weight:300;position:relative;z-index:1}
.sec-tag{font-size:9px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#6366f1;margin-bottom:.35rem}
.sec-title{font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;color:#f1f5f9;margin-bottom:1.4rem;padding-bottom:.65rem;border-bottom:1px solid rgba(99,102,241,.18)}
.gcard{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.065);border-radius:18px;padding:1.75rem 1.75rem 1.5rem;position:relative;overflow:hidden}
.gcard::before{content:"";position:absolute;inset:0;background:linear-gradient(135deg,rgba(99,102,241,.05) 0%,transparent 55%);pointer-events:none}
div[data-testid="stSlider"]>label{font-size:.75rem!important;font-weight:600!important;color:#64748b!important;letter-spacing:.8px!important;text-transform:uppercase!important}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]{background-color:#6366f1!important;border-color:#818cf8!important}
div[data-testid="stSelectbox"] label{font-size:.75rem!important;font-weight:600!important;color:#64748b!important;letter-spacing:.8px!important;text-transform:uppercase!important}
div[data-baseweb="select"]{background-color:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important}
div[data-baseweb="select"] span{color:#e2e8f0!important}
div[data-testid="stMetric"]{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:.8rem 1rem!important}
div[data-testid="stMetric"] label{color:#64748b!important;font-size:.7rem!important;font-weight:600!important;letter-spacing:1px!important;text-transform:uppercase!important}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{font-size:1.25rem!important;font-weight:600!important;color:#e2e8f0!important}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;letter-spacing:.3px}
.badge-good{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.25)}
.badge-avg{background:rgba(245,158,11,.13);color:#fbbf24;border:1px solid rgba(245,158,11,.22)}
.badge-low{background:rgba(239,68,68,.13);color:#f87171;border:1px solid rgba(239,68,68,.22)}
.res-banner{border-radius:18px;padding:2.2rem 2rem;text-align:center;position:relative;overflow:hidden}
.res-banner.excellent{background:linear-gradient(135deg,rgba(16,185,129,.16),rgba(5,150,105,.07));border:1px solid rgba(16,185,129,.3)}
.res-banner.good{background:linear-gradient(135deg,rgba(245,158,11,.16),rgba(217,119,6,.07));border:1px solid rgba(245,158,11,.3)}
.res-banner.low{background:linear-gradient(135deg,rgba(239,68,68,.16),rgba(185,28,28,.07));border:1px solid rgba(239,68,68,.3)}
.res-pct{font-family:'Playfair Display',serif;font-size:4.2rem;font-weight:900;line-height:1}
.res-pct.excellent{color:#34d399}.res-pct.good{color:#fbbf24}.res-pct.low{color:#f87171}
.res-label{font-size:.78rem;font-weight:600;color:#64748b;letter-spacing:2px;text-transform:uppercase;margin-top:.4rem}
.res-msg{font-size:.9rem;color:#94a3b8;margin-top:.8rem;line-height:1.65}
.pct-card{background:rgba(99,102,241,.09);border:1px solid rgba(99,102,241,.2);border-radius:14px;padding:1.25rem 1.5rem;text-align:center;margin-top:.9rem}
.pct-num{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#a5b4fc,#f0abfc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}
.pct-sub{font-size:.75rem;color:#64748b;margin-top:.3rem;letter-spacing:.5px}
.tip-box{background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.18);border-radius:12px;padding:.9rem 1.25rem;font-size:.83rem;color:#a5b4fc;line-height:1.7;margin-top:.9rem}
.tip-box strong{color:#c7d2fe}
.fdivider{border:none;height:1px;background:linear-gradient(to right,transparent,rgba(99,102,241,.35),transparent);margin:2rem 0}
.cwrap{background:rgba(255,255,255,.018);border:1px solid rgba(255,255,255,.055);border-radius:18px;padding:1.75rem}
.footer{text-align:center;color:#1e293b;font-size:.75rem;padding:2rem 0 1rem;letter-spacing:.4px}
#MainMenu,footer,header{visibility:hidden}
.bulk-info{background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.2);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;font-size:.85rem;color:#a5b4fc;line-height:1.8}
.bulk-info code{background:rgba(99,102,241,.18);color:#c7d2fe;padding:1px 6px;border-radius:4px;font-size:.82rem}
.bulk-stat{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.065);border-radius:12px;padding:.9rem 1rem;text-align:center}
.bulk-stat-num{font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;line-height:1}
.bulk-stat-lbl{font-size:.7rem;font-weight:600;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-top:.2rem}
</style>
"""

try:
    st.html(_CSS)
except AttributeError:
    st.markdown(_CSS, unsafe_allow_html=True)

# ─── Data & Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def setup():
    df = pd.read_csv("Admission_Predict.csv")
    df = df.drop(["Serial No."], axis=1)
    X = df.drop("Chance of Admit ", axis=1)
    y = df["Chance of Admit "]
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y, df

model, X, y, df = setup()

col_stats = {col: {"p33": df[col].quantile(.33), "p66": df[col].quantile(.66)} for col in X.columns}

def badge(val, col):
    p33, p66 = col_stats[col]["p33"], col_stats[col]["p66"]
    if val >= p66:   return '<span class="badge badge-good">✓ Strong</span>'
    elif val >= p33: return '<span class="badge badge-avg">~ Average</span>'
    else:            return '<span class="badge badge-low">↑ Improve</span>'

PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans", color="#94a3b8"))

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="tag">AI-Powered · Graduate Admissions</div>
  <div class="hero-title">AdmitIQ</div>
  <div class="hero-sub">Move any slider — your chances update instantly. No button needed.</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="fdivider">', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎓 Single Predictor", "📂 Bulk Scanner"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown('<div class="gcard">', unsafe_allow_html=True)
        st.markdown('<div class="sec-tag">Step 01</div><div class="sec-title">📝 Test Scores & Academic</div>', unsafe_allow_html=True)
        gre   = st.slider("GRE Score",   260, 340, 310)
        st.markdown(badge(gre,   "GRE Score"),   unsafe_allow_html=True); st.write("")
        toefl = st.slider("TOEFL Score",   0, 120, 105)
        st.markdown(badge(toefl, "TOEFL Score"), unsafe_allow_html=True); st.write("")
        cgpa  = st.slider("CGPA",         0.0, 10.0, 8.5, step=0.1)
        st.markdown(badge(cgpa,  "CGPA"),        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="gcard">', unsafe_allow_html=True)
        st.markdown('<div class="sec-tag">Step 02</div><div class="sec-title">🏛️ Profile & Background</div>', unsafe_allow_html=True)
        university = st.slider("University Rating", 1, 5, 3)
        st.markdown(badge(university, "University Rating"), unsafe_allow_html=True); st.write("")
        sop = st.slider("SOP Strength", 1.0, 5.0, 3.5, step=0.5)
        st.markdown(badge(sop, "SOP"),  unsafe_allow_html=True); st.write("")
        lor = st.slider("LOR Strength", 1.0, 5.0, 3.5, step=0.5)
        st.markdown(badge(lor, "LOR "), unsafe_allow_html=True); st.write("")
        research = st.selectbox("Research Experience", ["No", "Yes"])
        research_val = 1 if research == "Yes" else 0
        if research_val:
            st.markdown('<span class="badge badge-good">✓ Yes — great advantage</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-avg">○ No — consider gaining some</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    input_data = pd.DataFrame([[gre, toefl, university, sop, lor, cgpa, research_val]], columns=X.columns)
    prediction = float(np.clip(model.predict(input_data)[0], 0, 1))
    pct        = prediction * 100
    all_preds  = np.clip(model.predict(X), 0, 1) * 100
    percentile = float(stats.percentileofscore(all_preds, pct))

    if prediction > 0.75:   tier, msg = "excellent", "Strong profile — apply to your dream schools with confidence."
    elif prediction > 0.5:  tier, msg = "good",      "Solid profile — a few targeted improvements could push you into the top tier."
    else:                   tier, msg = "low",       "Needs strengthening — focus on GRE, CGPA, and research experience."

    gauge_color = {"excellent": "#34d399", "good": "#fbbf24", "low": "#f87171"}[tier]

    st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-tag" style="text-align:center;margin-bottom:.8rem">Live Result — Updates as you move sliders</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Admission Chance", f"{pct:.1f}%")
    m2.metric("Percentile Rank",  f"Top {100-percentile:.0f}%")
    m3.metric("vs. Dataset Avg",  f"{pct - all_preds.mean():+.1f}%")
    m4.metric("Research Bonus",   "Yes ✓" if research_val else "None")

    st.write("")

    res_col, vis_col = st.columns([1, 1], gap="large")

    with res_col:
        st.markdown(f"""
        <div class="res-banner {tier}">
          <div class="res-pct {tier}">{pct:.1f}%</div>
          <div class="res-label">Chance of Admission</div>
          <div class="res-msg">{msg}</div>
        </div>
        <div class="pct-card">
          <div class="pct-num">Top {100-percentile:.0f}%</div>
          <div class="pct-sub">You outperform {percentile:.0f}% of applicants in this dataset</div>
        </div>
        """, unsafe_allow_html=True)

        tips = []
        if gre        < col_stats["GRE Score"]["p66"]:   tips.append("📖 Raise <strong>GRE ≥ 320</strong> to reach the top third.")
        if toefl      < col_stats["TOEFL Score"]["p66"]: tips.append("🗣️ Aim for <strong>TOEFL ≥ 110</strong> for top schools.")
        if cgpa       < col_stats["CGPA"]["p66"]:        tips.append("📚 <strong>CGPA ≥ 8.8</strong> is where chances jump significantly.")
        if research_val == 0:                             tips.append("🔬 <strong>Research experience</strong> is a strong competitive edge.")
        if sop        < col_stats["SOP"]["p66"]:         tips.append("✍️ A polished <strong>SOP</strong> can compensate for weaker scores.")
        if tips:
            bullets = "".join(f"<div style='margin-top:4px'>• {t}</div>" for t in tips[:3])
            st.markdown(f'<div class="tip-box"><strong>💡 Top ways to improve</strong>{bullets}</div>', unsafe_allow_html=True)

    with vis_col:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pct,
            number={"suffix": "%", "font": {"size": 34, "color": "#e2e8f0", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#334155"}, "nticks": 6},
                "bar": {"color": gauge_color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "rgba(239,68,68,.07)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,.07)"},
                    {"range": [75,100], "color": "rgba(16,185,129,.07)"},
                ],
                "threshold": {"line": {"color": "#818cf8", "width": 2.5}, "thickness": .75, "value": pct},
            },
            title={"text": "Admission probability", "font": {"color": "#475569", "size": 12}},
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig_g.update_layout(**PL, height=250, margin=dict(l=20, r=20, t=40, b=5))
        st.plotly_chart(fig_g, use_container_width=True)

        norm = lambda v, mn, mx: (v - mn) / (mx - mn)
        labels = ["GRE", "TOEFL", "Univ.", "SOP", "LOR", "CGPA", "Research"]
        user_v = [norm(gre,260,340), norm(toefl,0,120), norm(university,1,5),
                  norm(sop,1,5), norm(lor,1,5), norm(cgpa,0,10), float(research_val)]
        avg_v  = [norm(df["GRE Score"].mean(),260,340), norm(df["TOEFL Score"].mean(),0,120),
                  norm(df["University Rating"].mean(),1,5), norm(df["SOP"].mean(),1,5),
                  norm(df["LOR "].mean(),1,5), norm(df["CGPA"].mean(),0,10), df["Research"].mean()]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=avg_v+[avg_v[0]], theta=labels+[labels[0]],
            fill="toself", fillcolor="rgba(100,116,139,.12)",
            line=dict(color="#475569", width=1.5, dash="dot"), name="Dataset avg"))
        fig_r.add_trace(go.Scatterpolar(r=user_v+[user_v[0]], theta=labels+[labels[0]],
            fill="toself", fillcolor="rgba(99,102,241,.18)",
            line=dict(color="#818cf8", width=2), name="Your profile"))
        fig_r.update_layout(**PL, height=270, margin=dict(l=20,r=20,t=25,b=20),
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#334155",size=9),
                    gridcolor="rgba(255,255,255,.06)", linecolor="rgba(255,255,255,.06)"),
                angularaxis=dict(tickfont=dict(color="#94a3b8",size=11),
                    gridcolor="rgba(255,255,255,.06)", linecolor="rgba(255,255,255,.06)")),
            legend=dict(font=dict(color="#64748b",size=11), bgcolor="rgba(0,0,0,0)",
                x=0.5, xanchor="center", y=-0.08, orientation="h"), showlegend=True)
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown('<div class="cwrap">', unsafe_allow_html=True)
        st.markdown('<div class="sec-tag">Model Insights</div><div class="sec-title">📊 Feature Impact (Model Coefficients)</div>', unsafe_allow_html=True)
        pretty = {"GRE Score":"GRE","TOEFL Score":"TOEFL","University Rating":"Univ.","SOP":"SOP","LOR ":"LOR","CGPA":"CGPA","Research":"Research"}
        cdf = pd.DataFrame({"Feature":[pretty.get(f,f) for f in X.columns],"Impact":model.coef_}).sort_values("Impact")
        fig_bar = go.Figure(go.Bar(x=cdf["Impact"], y=cdf["Feature"], orientation="h",
            marker_color=["#f87171" if v<0 else "#6366f1" for v in cdf["Impact"]], marker_line_width=0,
            text=[f"{v:.4f}" for v in cdf["Impact"]], textposition="outside",
            textfont=dict(color="#475569", size=10)))
        fig_bar.update_layout(**PL, height=290, margin=dict(l=10,r=55,t=5,b=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)",
                zeroline=True, zerolinecolor="rgba(255,255,255,.08)", tickfont=dict(color="#334155")),
            yaxis=dict(tickfont=dict(color="#cbd5e1", size=12)))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="cwrap">', unsafe_allow_html=True)
        st.markdown('<div class="sec-tag">Dataset View</div><div class="sec-title">🎯 CGPA vs Chance — You Are Here</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=df["CGPA"], y=df["Chance of Admit "]*100,
            mode="markers", marker=dict(color="rgba(99,102,241,.35)", size=5, line=dict(width=0)), name="Applicants"))
        fig_sc.add_trace(go.Scatter(x=[cgpa], y=[pct], mode="markers+text",
            marker=dict(color="#f0abfc", size=14, symbol="star", line=dict(color="#fff",width=1.5)),
            text=["You"], textposition="top center", textfont=dict(color="#f0abfc",size=11), name="You"))
        fig_sc.update_layout(**PL, height=290, margin=dict(l=10,r=10,t=5,b=30),
            xaxis=dict(title="CGPA", showgrid=True, gridcolor="rgba(255,255,255,.04)",
                tickfont=dict(color="#334155"), title_font=dict(color="#475569")),
            yaxis=dict(title="Chance (%)", showgrid=True, gridcolor="rgba(255,255,255,.04)",
                tickfont=dict(color="#334155"), title_font=dict(color="#475569")),
            legend=dict(font=dict(color="#64748b",size=11), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cwrap">', unsafe_allow_html=True)
    st.markdown('<div class="sec-tag">Benchmark</div><div class="sec-title">📈 Applicant Distribution — Your Position</div>', unsafe_allow_html=True)
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=all_preds, nbinsx=35,
        marker_color="#6366f1", marker_line_width=0, opacity=0.7))
    fig_h.add_vline(x=pct, line_color="#f0abfc", line_width=2, line_dash="dash",
        annotation_text=f"  You {pct:.1f}%", annotation_font_color="#f0abfc",
        annotation_font_size=12, annotation_position="top right")
    fig_h.update_layout(**PL, height=230, margin=dict(l=10,r=10,t=10,b=30), showlegend=False, bargap=0.04,
        xaxis=dict(title="Admission Chance (%)", showgrid=True, gridcolor="rgba(255,255,255,.04)",
            tickfont=dict(color="#334155"), title_font=dict(color="#475569")),
        yaxis=dict(title="# Applicants", showgrid=True, gridcolor="rgba(255,255,255,.04)",
            tickfont=dict(color="#334155"), title_font=dict(color="#475569")))
    st.plotly_chart(fig_h, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
    st.markdown('<div class="cwrap">', unsafe_allow_html=True)
    st.markdown('<div class="sec-tag">Multi-Dimensional View</div><div class="sec-title">📡 GRE vs TOEFL — Colored by Admission Chance</div>', unsafe_allow_html=True)
    fig_gre = go.Figure()
    fig_gre.add_trace(go.Scatter(x=df["GRE Score"], y=df["TOEFL Score"], mode="markers",
        marker=dict(
            color=df["Chance of Admit "]*100,
            colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#34d399"]],
            size=6, opacity=.65,
            colorbar=dict(
                title=dict(text="Chance %", font=dict(color="#64748b")),
                tickfont=dict(color="#64748b", size=10),
                thickness=12, len=0.75
            ),
            line=dict(width=0)
        ), name="Applicants"))
    fig_gre.add_trace(go.Scatter(x=[gre], y=[toefl], mode="markers+text",
        marker=dict(color="#f0abfc", size=15, symbol="star", line=dict(color="#fff",width=2)),
        text=["You"], textposition="top center", textfont=dict(color="#f0abfc",size=11), name="You"))
    fig_gre.update_layout(**PL, height=320, margin=dict(l=10,r=10,t=5,b=30),
        xaxis=dict(title="GRE Score", showgrid=True, gridcolor="rgba(255,255,255,.04)",
            tickfont=dict(color="#334155"), title_font=dict(color="#475569")),
        yaxis=dict(title="TOEFL Score", showgrid=True, gridcolor="rgba(255,255,255,.04)",
            tickfont=dict(color="#334155"), title_font=dict(color="#475569")),
        legend=dict(font=dict(color="#64748b",size=11), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_gre, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Scanner
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    EXPECTED_COLS = list(X.columns)

    # Flexible column name aliases for auto-detection
    COL_ALIASES = {
        "GRE Score":         ["gre", "gre score", "gre_score", "grescore"],
        "TOEFL Score":       ["toefl", "toefl score", "toefl_score", "toeflscore"],
        "University Rating": ["university rating", "univ rating", "university_rating",
                              "univ_rating", "university", "univ", "uni_rating"],
        "SOP":               ["sop", "sop strength", "sop_strength", "statement of purpose"],
        "LOR ":              ["lor", "lor ", "lor strength", "lor_strength", "letter of recommendation"],
        "CGPA":              ["cgpa", "gpa", "grade", "cumulative gpa"],
        "Research":          ["research", "research experience", "research_experience",
                              "has research", "research_exp"],
    }

    def auto_map_columns(upload_cols):
        mapping = {}
        up_lower = {c.lower().strip(): c for c in upload_cols}
        for expected, aliases in COL_ALIASES.items():
            for alias in aliases:
                if alias in up_lower:
                    mapping[expected] = up_lower[alias]
                    break
        return mapping

    def load_file(uploaded_file):
        """Load CSV / Excel / JSON / SQLite into a DataFrame. Returns (df, error)."""
        name = uploaded_file.name.lower()
        try:
            # ── CSV ──────────────────────────────────────────────────────────
            if name.endswith(".csv"):
                return pd.read_csv(uploaded_file), None

            # ── Excel ────────────────────────────────────────────────────────
            elif name.endswith((".xlsx", ".xls")):
                xf = pd.ExcelFile(uploaded_file)
                sheet = (
                    st.selectbox("📋 Multiple sheets — choose one:", xf.sheet_names, key="sheet_sel")
                    if len(xf.sheet_names) > 1 else xf.sheet_names[0]
                )
                return pd.read_excel(uploaded_file, sheet_name=sheet), None

            # ── JSON ─────────────────────────────────────────────────────────
            elif name.endswith(".json"):
                raw = pd.read_json(uploaded_file)
                # Flatten one nesting level if values are dicts
                if len(raw) > 0 and isinstance(raw.iloc[0, 0], dict):
                    raw = pd.json_normalize(raw.to_dict(orient="records"))
                return raw, None

            # ── SQLite / .db / .sql ──────────────────────────────────────────
            elif name.endswith((".db", ".sqlite", ".sql")):
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    conn = sqlite3.connect(tmp_path)
                    tables = pd.read_sql(
                        "SELECT name FROM sqlite_master WHERE type='table'", conn
                    )["name"].tolist()
                    if not tables:
                        conn.close()
                        return None, "No tables found in the SQLite database."
                    tbl = (
                        st.selectbox("🗄️ Multiple tables — choose one:", tables, key="sql_tbl")
                        if len(tables) > 1 else tables[0]
                    )
                    result_df = pd.read_sql(f'SELECT * FROM "{tbl}"', conn)
                    conn.close()
                    return result_df, None
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            else:
                return None, f"Unsupported file type: `{uploaded_file.name}`"

        except Exception as e:
            return None, str(e)

    def tier_label(p):
        if p >= 75:  return "🟢 Excellent"
        elif p >= 50: return "🟡 Good"
        else:         return "🔴 Needs Work"

    # ── UI ────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-tag">Bulk Scanner</div>'
        '<div class="sec-title">📂 Upload a file — scan &amp; predict all applicants at once</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="bulk-info">
      <strong>Supported formats:</strong>
      &nbsp;📄 <code>.csv</code>
      &nbsp;📊 <code>.xlsx&nbsp;/&nbsp;.xls</code>
      &nbsp;🗃️ <code>.db&nbsp;/&nbsp;.sqlite</code> (SQLite)
      &nbsp;🔷 <code>.json</code><br><br>
      <strong>Required columns (flexible naming):</strong><br>
      <code>GRE Score</code> · <code>TOEFL Score</code> · <code>University Rating</code> ·
      <code>SOP</code> · <code>LOR</code> · <code>CGPA</code> · <code>Research</code><br><br>
      Column names are matched automatically — e.g. <code>gre</code>, <code>gre_score</code>,
      <code>GRE Score</code> all resolve correctly. Missing columns can be mapped manually.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your file here",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite", "sql"],
        label_visibility="collapsed",
        key="bulk_uploader"
    )

    # ── Sample Data ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:1.4rem">
      <div class="sec-tag">Sample Data</div>
      <div class="sec-title">📥 Try it with Sample Data</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="bulk-info" style="margin-bottom:.9rem">
      Not sure about the format? Download this ready-to-use <strong>Excel file with 5 sample applicants</strong>
      — upload it above to see the bulk scanner in action instantly.
    </div>
    """, unsafe_allow_html=True)

    _sample_df = pd.DataFrame({
        "GRE Score":         [337, 316, 322, 295, 311],
        "TOEFL Score":       [118, 104, 110,  97, 107],
        "University Rating": [  4,   3,   4,   2,   3],
        "SOP":               [4.5, 3.0, 4.0, 2.5, 3.5],
        "LOR ":              [4.5, 3.5, 4.0, 3.0, 3.0],
        "CGPA":              [9.65, 8.00, 8.67, 7.10, 8.30],
        "Research":          [  1,   1,   1,   0,   1],
    })
    _sample_buf = io.BytesIO()
    with pd.ExcelWriter(_sample_buf, engine="openpyxl") as _writer:
        _sample_df.to_excel(_writer, index=False, sheet_name="Applicants")
    st.download_button(
        label="⬇️ Download Sample Excel (5 applicants)",
        data=_sample_buf.getvalue(),
        file_name="admitiq_sample_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="sample_dl"
    )
    st.markdown('<hr class="fdivider" style="margin:1.6rem 0">', unsafe_allow_html=True)

    if uploaded is not None:
        raw_df, err = load_file(uploaded)

        if err:
            st.error(f"❌ Could not load file: {err}")
        else:
            st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sec-tag">File loaded</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f"**{uploaded.name}** — {len(raw_df):,} rows · {len(raw_df.columns)} columns"
            )

            with st.expander("👁️ Preview raw data (first 5 rows)"):
                st.dataframe(raw_df.head(), use_container_width=True)

            # ── Column mapping ─────────────────────────────────────────────
            auto_map = auto_map_columns(raw_df.columns.tolist())
            missing  = [c for c in EXPECTED_COLS if c not in auto_map]
            final_map = dict(auto_map)

            st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
            st.markdown('<div class="sec-tag">Column Mapping</div>', unsafe_allow_html=True)

            if missing:
                st.warning(
                    f"⚠️ Could not auto-detect **{len(missing)}** column(s). Map them below:"
                )
                avail = ["— skip —"] + raw_df.columns.tolist()
                for ec in missing:
                    chosen = st.selectbox(f"Map **{ec}** →", avail, key=f"map_{ec}")
                    if chosen != "— skip —":
                        final_map[ec] = chosen
            else:
                st.success("✅ All columns auto-detected successfully!")

            with st.expander("🗺️ Column mapping details"):
                st.dataframe(
                    pd.DataFrame([
                        {"Model expects": ec, "Your file column": final_map.get(ec, "⚠️ Not mapped")}
                        for ec in EXPECTED_COLS
                    ]),
                    use_container_width=True,
                    hide_index=True
                )

            # ── Run predictions ────────────────────────────────────────────
            still_missing = [c for c in EXPECTED_COLS if c not in final_map]
            if still_missing:
                st.error(f"❌ Still missing: {still_missing}. Cannot predict without all columns.")
            else:
                aligned = pd.DataFrame({
                    ec: pd.to_numeric(raw_df[final_map[ec]], errors="coerce")
                    for ec in EXPECTED_COLS
                })

                n_bad = aligned.isnull().any(axis=1).sum()
                if n_bad:
                    st.warning(f"⚠️ {n_bad} row(s) had non-numeric / missing values and were skipped.")
                aligned = aligned.dropna()

                if len(aligned) == 0:
                    st.error("❌ No valid rows after cleaning — check your data.")
                else:
                    chances   = np.clip(model.predict(aligned[EXPECTED_COLS]), 0, 1) * 100
                    result_df = raw_df.loc[aligned.index].copy().reset_index(drop=True)
                    result_df["Predicted Chance (%)"] = np.round(chances, 2)
                    result_df["Tier"]       = [tier_label(p) for p in chances]
                    result_df["Percentile"] = [
                        f"Top {100 - float(stats.percentileofscore(chances, p)):.0f}%"
                        for p in chances
                    ]

                    st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="sec-tag">Scan Results</div>'
                        '<div class="sec-title">🔍 Predictions Complete</div>',
                        unsafe_allow_html=True
                    )

                    # ── Summary stat cards ─────────────────────────────────
                    n_exc = int((chances >= 75).sum())
                    n_gd  = int(((chances >= 50) & (chances < 75)).sum())
                    n_low = int((chances < 50).sum())
                    avg_c = float(chances.mean())

                    s1, s2, s3, s4 = st.columns(4)
                    s1.markdown(f'<div class="bulk-stat"><div class="bulk-stat-num" style="color:#34d399">{n_exc}</div><div class="bulk-stat-lbl">Excellent ≥75%</div></div>', unsafe_allow_html=True)
                    s2.markdown(f'<div class="bulk-stat"><div class="bulk-stat-num" style="color:#fbbf24">{n_gd}</div><div class="bulk-stat-lbl">Good 50–75%</div></div>', unsafe_allow_html=True)
                    s3.markdown(f'<div class="bulk-stat"><div class="bulk-stat-num" style="color:#f87171">{n_low}</div><div class="bulk-stat-lbl">Needs Work &lt;50%</div></div>', unsafe_allow_html=True)
                    s4.markdown(f'<div class="bulk-stat"><div class="bulk-stat-num" style="color:#a5b4fc">{avg_c:.1f}%</div><div class="bulk-stat-lbl">Avg Chance</div></div>', unsafe_allow_html=True)

                    st.write("")

                    # ── Distribution chart ─────────────────────────────────
                    st.markdown('<div class="cwrap">', unsafe_allow_html=True)
                    st.markdown('<div class="sec-tag">Distribution</div><div class="sec-title">📊 Predicted Chance Distribution</div>', unsafe_allow_html=True)
                    fig_bulk = go.Figure()
                    fig_bulk.add_trace(go.Histogram(x=chances, nbinsx=20,
                        marker_color="#6366f1", marker_line_width=0, opacity=0.75))
                    fig_bulk.add_vline(x=avg_c, line_color="#f0abfc", line_width=2, line_dash="dash",
                        annotation_text=f"  Avg {avg_c:.1f}%", annotation_font_color="#f0abfc",
                        annotation_font_size=12, annotation_position="top right")
                    fig_bulk.update_layout(**PL, height=210, margin=dict(l=10,r=10,t=10,b=30),
                        showlegend=False, bargap=0.05,
                        xaxis=dict(title="Predicted Chance (%)", gridcolor="rgba(255,255,255,.04)",
                            tickfont=dict(color="#334155"), title_font=dict(color="#475569")),
                        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,.04)",
                            tickfont=dict(color="#334155"), title_font=dict(color="#475569")))
                    st.plotly_chart(fig_bulk, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown('<hr class="fdivider">', unsafe_allow_html=True)

                    # ── Filter + sort controls ─────────────────────────────
                    st.markdown('<div class="sec-tag" style="margin-bottom:.5rem">Individual Results</div>', unsafe_allow_html=True)
                    fc1, fc2, fc3 = st.columns([2, 1, 1])
                    with fc1:
                        tier_filter = st.multiselect(
                            "Filter by tier",
                            ["🟢 Excellent", "🟡 Good", "🔴 Needs Work"],
                            default=["🟢 Excellent", "🟡 Good", "🔴 Needs Work"],
                            key="tier_filter"
                        )
                    with fc2:
                        sort_sel = st.selectbox(
                            "Sort by",
                            ["Predicted Chance (%) ↓", "Predicted Chance (%) ↑"],
                            key="sort_sel"
                        )
                    with fc3:
                        per_page = st.selectbox("Per page", [3, 6, 9, 12], index=1, key="per_page")

                    display_df = result_df[result_df["Tier"].isin(tier_filter)].copy()
                    display_df = display_df.sort_values(
                        "Predicted Chance (%)", ascending=sort_sel.endswith("↑")
                    ).reset_index(drop=True)

                    total     = len(display_df)
                    n_pages   = max(1, -(-total // per_page))   # ceiling division
                    page_idx  = st.session_state.get("bulk_page", 0)
                    page_idx  = min(page_idx, n_pages - 1)

                    # Pagination buttons
                    pg_l, pg_mid, pg_r = st.columns([1, 3, 1])
                    with pg_l:
                        if st.button("← Prev", disabled=(page_idx == 0), key="pg_prev"):
                            st.session_state["bulk_page"] = page_idx - 1
                            st.rerun()
                    with pg_mid:
                        st.markdown(
                            f'<div style="text-align:center;color:#64748b;font-size:.8rem;padding:.5rem 0">'
                            f'Page {page_idx+1} of {n_pages} &nbsp;·&nbsp; {total} applicants</div>',
                            unsafe_allow_html=True
                        )
                    with pg_r:
                        if st.button("Next →", disabled=(page_idx >= n_pages - 1), key="pg_next"):
                            st.session_state["bulk_page"] = page_idx + 1
                            st.rerun()

                    # Reset page when filters change
                    if "bulk_page" not in st.session_state:
                        st.session_state["bulk_page"] = 0

                    page_df = display_df.iloc[page_idx * per_page : (page_idx + 1) * per_page]

                    # ── Helper: build tip list for one row ─────────────────
                    def row_tips(row):
                        tips = []
                        if row["GRE Score"]         < col_stats["GRE Score"]["p66"]:   tips.append("📖 Raise <strong>GRE ≥ 320</strong> to reach the top third.")
                        if row["TOEFL Score"]        < col_stats["TOEFL Score"]["p66"]: tips.append("🗣️ Aim for <strong>TOEFL ≥ 110</strong> for top schools.")
                        if row["CGPA"]               < col_stats["CGPA"]["p66"]:        tips.append("📚 <strong>CGPA ≥ 8.8</strong> is where chances jump significantly.")
                        if row["Research"]           == 0:                              tips.append("🔬 <strong>Research experience</strong> is a strong competitive edge.")
                        if row["SOP"]                < col_stats["SOP"]["p66"]:         tips.append("✍️ A polished <strong>SOP</strong> can compensate for weaker scores.")
                        return tips[:3]

                    # ── Render cards: 2 per row ────────────────────────────
                    rows_iter = list(page_df.iterrows())
                    for pair_start in range(0, len(rows_iter), 2):
                        pair = rows_iter[pair_start : pair_start + 2]
                        cols = st.columns(len(pair), gap="large")

                        for col_widget, (_, row) in zip(cols, pair):
                            with col_widget:
                                p       = float(row["Predicted Chance (%)"])
                                pctile  = float(stats.percentileofscore(chances, p))
                                t       = "excellent" if p >= 75 else ("good" if p >= 50 else "low")
                                gc      = {"excellent": "#34d399", "good": "#fbbf24", "low": "#f87171"}[t]

                                # Try to find a name column for the card title
                                name_cols = [c for c in raw_df.columns if c.lower() in
                                             ("name","applicant","student","candidate","id","serial no.","serial no")]
                                label = f"Applicant #{int(row.name)+1}"
                                if name_cols:
                                    label = str(row[name_cols[0]])

                                # Result banner
                                st.markdown(f"""
                                <div class="res-banner {t}" style="margin-bottom:.6rem">
                                  <div style="font-size:.72rem;font-weight:700;letter-spacing:2px;
                                              text-transform:uppercase;color:#64748b;margin-bottom:.4rem">
                                    {label}
                                  </div>
                                  <div class="res-pct {t}">{p:.1f}%</div>
                                  <div class="res-label">Chance of Admission</div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Gauge
                                fig_card = go.Figure(go.Indicator(
                                    mode="gauge+number", value=p,
                                    number={"suffix": "%", "font": {"size": 26, "color": "#e2e8f0", "family": "DM Sans"}},
                                    gauge={
                                        "axis": {"range": [0, 100], "tickcolor": "#334155",
                                                 "tickfont": {"color": "#334155"}, "nticks": 5},
                                        "bar": {"color": gc, "thickness": 0.3},
                                        "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                                        "steps": [
                                            {"range": [0,  50], "color": "rgba(239,68,68,.07)"},
                                            {"range": [50, 75], "color": "rgba(245,158,11,.07)"},
                                            {"range": [75,100], "color": "rgba(16,185,129,.07)"},
                                        ],
                                        "threshold": {"line": {"color": "#818cf8", "width": 2},
                                                      "thickness": .75, "value": p},
                                    },
                                    title={"text": "Admission probability",
                                           "font": {"color": "#475569", "size": 11}},
                                    domain={"x": [0, 1], "y": [0, 1]},
                                ))
                                fig_card.update_layout(**PL, height=200,
                                    margin=dict(l=15, r=15, t=35, b=5))
                                st.plotly_chart(fig_card, use_container_width=True,
                                                key=f"gauge_{int(row.name)}")

                                # Percentile card
                                st.markdown(f"""
                                <div class="pct-card">
                                  <div class="pct-num">Top {100-pctile:.0f}%</div>
                                  <div class="pct-sub">Outperforms {pctile:.0f}% of this batch</div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Metrics row
                                ma, mb = st.columns(2)
                                ma.metric("GRE",  int(row["GRE Score"]))
                                mb.metric("TOEFL", int(row["TOEFL Score"]))
                                mc, md = st.columns(2)
                                mc.metric("CGPA", f"{row['CGPA']:.1f}")
                                md.metric("Research", "Yes ✓" if int(row["Research"]) else "None")

                                # Tips
                                tips = row_tips(row)
                                if tips:
                                    bullets = "".join(f"<div style='margin-top:3px'>• {t_}</div>" for t_ in tips)
                                    st.markdown(
                                        f'<div class="tip-box"><strong>💡 Improvements</strong>{bullets}</div>',
                                        unsafe_allow_html=True
                                    )

                                st.markdown('<hr class="fdivider" style="margin:1.2rem 0">', unsafe_allow_html=True)

                    # ── Download buttons ───────────────────────────────────
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            label="⬇️ Download results as CSV",
                            data=result_df.to_csv(index=False).encode("utf-8"),
                            file_name="admitiq_bulk_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with dl2:
                        xlsx_buf = io.BytesIO()
                        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                            result_df.to_excel(writer, index=False, sheet_name="Predictions")
                            pd.DataFrame({
                                "Tier":  ["🟢 Excellent (≥75%)", "🟡 Good (50–75%)", "🔴 Needs Work (<50%)", "Average Chance"],
                                "Count": [n_exc, n_gd, n_low, f"{avg_c:.1f}%"]
                            }).to_excel(writer, index=False, sheet_name="Summary")
                        st.download_button(
                            label="⬇️ Download results as Excel",
                            data=xlsx_buf.getvalue(),
                            file_name="admitiq_bulk_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">AdmitIQ · Streamlit + Plotly · Scikit-learn Linear Regression<br>
<span style="color:#0f172a">© 2024 · For academic use only</span></div>
""", unsafe_allow_html=True)
