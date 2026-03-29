import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy import stats

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdmitIQ — Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS Injection ────────────────────────────────────────────────────────────
# st.html() bypasses Streamlit's markdown sanitiser and never leaks as visible text.
# Falls back to st.markdown for Streamlit < 1.31.
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
</style>
"""

try:
    st.html(_CSS)          # Streamlit >= 1.31 — never leaks as visible text
except AttributeError:
    st.markdown(_CSS, unsafe_allow_html=True)   # older Streamlit fallback

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

# ─── Inputs ───────────────────────────────────────────────────────────────────
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

# ─── Live Prediction ──────────────────────────────────────────────────────────
input_data = pd.DataFrame([[gre, toefl, university, sop, lor, cgpa, research_val]], columns=X.columns)
prediction = float(np.clip(model.predict(input_data)[0], 0, 1))
pct        = prediction * 100
all_preds  = np.clip(model.predict(X), 0, 1) * 100
percentile = float(stats.percentileofscore(all_preds, pct))

if prediction > 0.75:   tier, msg = "excellent", "Strong profile — apply to your dream schools with confidence."
elif prediction > 0.5:  tier, msg = "good",      "Solid profile — a few targeted improvements could push you into the top tier."
else:                   tier, msg = "low",       "Needs strengthening — focus on GRE, CGPA, and research experience."

gauge_color = {"excellent": "#34d399", "good": "#fbbf24", "low": "#f87171"}[tier]

# ─── Metric Row ───────────────────────────────────────────────────────────────
st.markdown('<hr class="fdivider">', unsafe_allow_html=True)
st.markdown('<div class="sec-tag" style="text-align:center;margin-bottom:.8rem">Live Result — Updates as you move sliders</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Admission Chance", f"{pct:.1f}%")
m2.metric("Percentile Rank",  f"Top {100-percentile:.0f}%")
m3.metric("vs. Dataset Avg",  f"{pct - all_preds.mean():+.1f}%")
m4.metric("Research Bonus",   "Yes ✓" if research_val else "None")

st.write("")

# ─── Result + Radar ───────────────────────────────────────────────────────────
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
    # Gauge
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

    # Radar
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

# ─── Charts Row ───────────────────────────────────────────────────────────────
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

# ─── Distribution ──────────────────────────────────────────────────────────────
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

# ─── GRE vs TOEFL ──────────────────────────────────────────────────────────────
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

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">AdmitIQ · Streamlit + Plotly · Scikit-learn Linear Regression<br>
<span style="color:#0f172a">© 2024 · For academic use only</span></div>
""", unsafe_allow_html=True)