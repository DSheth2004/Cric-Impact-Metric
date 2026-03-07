"""
Tensor Titans — CricImpact IPL T20 Player Impact Dashboard
Dual-theme (light/dark), responsive, cricket-domain aesthetic with glassmorphism UI.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Paths ---
DIR = "cricket_impact"
P_FILE   = os.path.join(DIR, "all_players_impact_T20.csv")
M_FILE   = os.path.join(DIR, "match_rankings.csv")
BAT_FILE = os.path.join(DIR, "batting_impact.csv")
BWL_FILE = os.path.join(DIR, "bowling_impact.csv")
RIV_FILE = os.path.join(DIR, "batter_vs_bowler_rivalry.csv")


# ═══════════════════════════════════════════════════════════
# THEME ENGINE
# ═══════════════════════════════════════════════════════════
DARK_THEME = dict(
    BG="#0a0e14", SURFACE="#141a24", CARD_BG="rgba(20,26,36,0.88)",
    GLASS="rgba(255,255,255,0.035)", BORDER="rgba(255,255,255,0.08)",
    TEXT="#e6edf3", TEXT2="#8b949e",
    GRAD1="#0a0e14", GRAD2="#111927",
    PLOTLY_GRID="rgba(255,255,255,0.04)", PLOTLY_ZERO="rgba(255,255,255,0.06)",
    PLOTLY_BG="rgba(0,0,0,0)",
    SIDEBAR_TOP="#0a0e14", SIDEBAR_BOT="#141a24",
    HOVER_GLOW="rgba(88,166,255,0.15)",
)
LIGHT_THEME = dict(
    BG="#f4f6f9", SURFACE="#ffffff", CARD_BG="rgba(255,255,255,0.92)",
    GLASS="rgba(0,0,0,0.025)", BORDER="rgba(0,0,0,0.08)",
    TEXT="#1a1a2e", TEXT2="#555770",
    GRAD1="#f4f6f9", GRAD2="#e8ecf2",
    PLOTLY_GRID="rgba(0,0,0,0.05)", PLOTLY_ZERO="rgba(0,0,0,0.08)",
    PLOTLY_BG="rgba(0,0,0,0)",
    SIDEBAR_TOP="#ffffff", SIDEBAR_BOT="#f0f2f5",
    HOVER_GLOW="rgba(30,80,160,0.10)",
)

# Accent palette (shared across themes)
CYAN     = "#3b82f6"
ORANGE   = "#f59e0b"
GREEN    = "#22c55e"
PURPLE   = "#a855f7"
PINK     = "#ec4899"
RED      = "#ef4444"
GOLD     = "#eab308"
TEAL     = "#14b8a6"
CRIMSON  = "#dc2626"

# Init theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def T():
    """Return current theme dict."""
    return DARK_THEME if st.session_state.theme == "dark" else LIGHT_THEME

def fig_layout():
    t = T()
    return dict(
        paper_bgcolor=t["PLOTLY_BG"], plot_bgcolor=t["PLOTLY_BG"],
        font=dict(family="'Inter', 'Segoe UI', system-ui, sans-serif", color=t["TEXT"], size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=t["SURFACE"], font_size=12, font_color=t["TEXT"],
                        bordercolor=t["BORDER"]),
        transition=dict(duration=700, easing="cubic-in-out"),
    )

def fig_grid():
    t = T()
    return dict(
        xaxis=dict(gridcolor=t["PLOTLY_GRID"], zerolinecolor=t["PLOTLY_ZERO"]),
        yaxis=dict(gridcolor=t["PLOTLY_GRID"], zerolinecolor=t["PLOTLY_ZERO"]),
    )

PLOTLY_CFG = dict(scrollZoom=False, displayModeBar="hover",
                  modeBarButtonsToRemove=["lasso2d","select2d"],
                  displaylogo=False)


# --- Helpers ---
def clipped_mean(s):
    """Mean after clipping values outside 1.5*IQR."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr).mean()

@st.cache_data(show_spinner=False)
def load(p):
    return pd.read_csv(p) if os.path.exists(p) else None

def si(v, d=0):
    try: return int(v)
    except: return d

def role_of(row):
    b, w = si(row.get("bat_innings")), si(row.get("bowl_innings"))
    if b >= 10 and w >= 10 and min(b,w)/max(b,w) >= 0.4:
        return "All-rounder"
    return "Batter" if b >= w else "Bowler"

def get_trend(name, bat, bowl, season=None):
    df = None
    if bat is not None and "player" in bat.columns:
        df = bat[bat["player"] == name].copy()
    if (df is None or len(df) < 3) and bowl is not None and "player" in bowl.columns:
        df = bowl[bowl["player"] == name].copy()
    if df is None or "impact_score" not in df.columns:
        return None
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if season and season != "All" and "season" in df.columns:
        df = df[df["season"].astype(str) == season]
    return df.sort_values("start_date").tail(15)

def radar(row, df):
    bm = max(df["mean_bat_impact"].max(), 1e-9)
    wm = max(df["mean_bowl_impact"].max(), 1e-9)
    cm = max((df["total_clutch_bat"].abs()+df["total_clutch_bowl"].abs()).max(), 1e-9)
    dbowl = max(df["dot_pct_bowled"].max(), 1e-9)
    dbat  = max(df["dot_pct_faced"].max(), 1e-9)
    pm = max(df["pom_awards"].max(), 1e-9)
    return {
        "Batting":   min(row.get("mean_bat_impact",0)/bm, 1),
        "Bowling":   min(row.get("mean_bowl_impact",0)/wm, 1),
        "Clutch":    min(abs(row.get("total_clutch_bat",0)+row.get("total_clutch_bowl",0))/cm, 1),
        "Consistency": min(row.get("not_out_pct",0), 1),
        "Win Impact":row.get("win_rate", 0),
        "Dot Eff":   min(row.get("dot_pct_bowled",0)/dbowl,1) if row.get("bowl_innings",0)>5
                     else 1-min(row.get("dot_pct_faced",0)/dbat,1),
        "Awards":    min(row.get("pom_awards",0)/pm, 1),
    }

def build_match_labels(mf):
    out = {}
    for mid, g in mf.groupby("match_id"):
        ts = g["player_team"].unique().tolist()
        v  = g["venue"].iloc[0].split(",")[0].strip() if "venue" in g.columns else ""
        s  = str(g["season"].iloc[0]) if "season" in g.columns else ""
        if len(ts) >= 2:
            lbl = f"{ts[0]}  vs  {ts[1]}"
        else:
            opp = g["opposition"].iloc[0] if "opposition" in g.columns else "?"
            lbl = f"{ts[0]}  vs  {opp}"
        if v: lbl += f" | {v}"
        if s: lbl += f" ({s})"
        out[mid] = lbl
    return out

def kpi_card(label, value, color=CYAN):
    return f"""
<div class="kpi-card" style="background:{CARD_BG};backdrop-filter:blur(12px);border:1px solid {BORDER};
            border-radius:14px;padding:18px 14px;text-align:center;
            box-shadow:0 0 20px rgba(0,0,0,0.3);">
  <div style="font-size:0.7rem;color:{TEXT2};letter-spacing:1px;text-transform:uppercase;
              margin-bottom:6px;">{label}</div>
  <div style="font-size:1.6rem;font-weight:700;color:{color};
              text-shadow:0 0 18px {color}40;">{value}</div>
</div>"""

def stat_row(label, value, color=CYAN):
    return f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 12px;
            background:{GLASS};border-radius:8px;margin-bottom:5px;
            border-left:3px solid {color};">
  <span style="flex:1;font-size:0.82rem;color:{TEXT2};">{label}</span>
  <span style="font-weight:600;color:{TEXT};font-size:0.95rem;">{value}</span>
</div>"""


# --- Page setup ---
st.set_page_config(page_title="CricImpact - IPL T20", layout="wide", page_icon="C")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  .stApp {{ background: {BG}; }}
  .block-container {{ padding-top:.8rem; max-width:1380px; }}
  h1,h2,h3,h4 {{ color:{TEXT}; font-family:'Inter',sans-serif; }}
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
      border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] .stNumberInput > div > div > input,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] [data-baseweb="select"] {{
      background-color: {SURFACE} !important;
      color: {TEXT} !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {TEXT} !important; }}
  section[data-testid="stSidebar"] [data-baseweb="input"] {{
      background-color: {SURFACE} !important;
      color: {TEXT} !important;
      border-color: {BORDER} !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="input"] input {{ color: {TEXT} !important; }}
  section[data-testid="stSidebar"] .stRadio label span {{ color: {TEXT} !important; }}
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ color: {TEXT} !important; }}
  section[data-testid="stSidebar"] button {{ background-color: {SURFACE} !important; color: {TEXT} !important; border-color: {BORDER} !important; }}
  [data-testid="stMetric"] {{
      background:{CARD_BG}; backdrop-filter:blur(12px);
      border:1px solid {BORDER}; border-radius:12px;
      padding:14px 16px; box-shadow:0 4px 20px rgba(0,0,0,0.3);
  }}
  [data-testid="stMetricValue"] {{ font-size:1.3rem; color:{CYAN}; font-weight:700; }}
  [data-testid="stMetricLabel"] {{ color:{TEXT2}; font-size:0.75rem; text-transform:uppercase; letter-spacing:.5px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{
      background:{CARD_BG}; border-radius:10px 10px 0 0;
      padding:10px 22px; font-weight:600; color:{TEXT2};
      border:1px solid {BORDER}; border-bottom:none; transition: all .2s;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color:{TEXT}; background:rgba(88,166,255,0.08); }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(135deg, {CYAN}20, {PURPLE}20) !important;
      color:{CYAN} !important; border-color:{CYAN}40 !important;
      box-shadow: 0 0 15px {CYAN}15;
  }}
  .glass {{ background:{CARD_BG}; backdrop-filter:blur(12px);
            border:1px solid {BORDER}; border-radius:14px;
            padding:20px; margin-bottom:12px;
            box-shadow:0 4px 30px rgba(0,0,0,0.25); }}
  [data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
  .hero {{ text-align:center; padding:10px 0 6px; }}
  .hero h1 {{ font-size:2.2rem; font-weight:800;
              background:linear-gradient(135deg,{CYAN},{PURPLE},{PINK});
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              margin-bottom:0; }}
  .hero p {{ color:{TEXT2}; font-size:0.9rem; margin-top:2px; }}
  .sep {{ border:none; border-top:1px solid {BORDER}; margin:18px 0; }}
  #MainMenu, footer, header {{ visibility:hidden; }}

  /* --- Chart animations --- */
  @keyframes fadeSlideUp {{
      from {{ opacity:0; transform:translateY(24px); }}
      to   {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes gaugeNeedle {{
      from {{ opacity:0; transform:rotate(-120deg); }}
      to   {{ opacity:1; transform:rotate(0deg); }}
  }}
  [data-testid="stPlotlyChart"] {{
      animation: fadeSlideUp 0.7s ease-out both;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
  }}
  [data-testid="stPlotlyChart"]:hover {{
      transform: scale(1.015);
      box-shadow: 0 8px 32px rgba(88,166,255,0.15);
  }}
  [data-testid="stDataFrame"] {{
      animation: fadeSlideUp 0.6s ease-out both;
      transition: transform 0.25s ease;
  }}
  [data-testid="stDataFrame"]:hover {{
      transform: scale(1.008);
  }}
  .kpi-card {{
      animation: fadeSlideUp 0.5s ease-out both;
      transition: transform 0.25s ease, box-shadow 0.25s ease;
  }}
  .kpi-card:hover {{
      transform: translateY(-4px) scale(1.03);
      box-shadow: 0 8px 24px rgba(88,166,255,0.2);
  }}
  .glass {{
      animation: fadeSlideUp 0.6s ease-out both;
  }}
</style>
""", unsafe_allow_html=True)


# --- Load ---
pdf_raw = load(P_FILE)
mdf     = load(M_FILE)
bat_df  = load(BAT_FILE)
bwl_df  = load(BWL_FILE)
riv_df  = load(RIV_FILE)

if pdf_raw is None:
    st.error("Run the pipeline first - data not found.")
    st.stop()

# --- Sidebar ---
st.sidebar.markdown("### CricImpact Filters")
min_inn = st.sidebar.slider("Min innings played", 0, 100, 10)
role_pick = st.sidebar.radio("Role", ["All", "Batter", "Bowler", "All-rounder"], horizontal=True)

all_seasons = []
if mdf is not None:
    all_seasons = sorted(mdf["season"].dropna().astype(str).unique().tolist())
sb_season = st.sidebar.selectbox("Season", ["All"] + all_seasons, key="sb_season")
min_runs = st.sidebar.number_input("Min career runs", 0, 10000, 0, step=100)
min_wkts = st.sidebar.number_input("Min career wickets", 0, 500, 0, step=5)

# --- Build player dataframe ---
# If a season is selected and match data exists, compute season-level averages
season_mode = sb_season != "All" and mdf is not None

if season_mode:
    sm = mdf[mdf["season"].astype(str) == sb_season]
    season_players = sm.groupby("player").agg(
        season_avg_impact=("impact_score", "mean"),
        season_matches=("match_id", "nunique"),
        season_total_impact=("impact_score", "sum"),
        season_avg_perf=("performance_score", "mean"),
    ).reset_index()
    pdf = pdf_raw.copy()
    pdf["role"] = pdf.apply(role_of, axis=1)
    pdf["total_innings"] = pdf["bat_innings"] + pdf["bowl_innings"]
    # Keep only players who played in this season
    pdf = pdf[pdf["player"].isin(season_players["player"])].copy()
    pdf = pdf.merge(season_players, on="player", how="left")
else:
    pdf = pdf_raw.copy()
    pdf["role"] = pdf.apply(role_of, axis=1)
    pdf["total_innings"] = pdf["bat_innings"] + pdf["bowl_innings"]

pdf = pdf[pdf["total_innings"] >= min_inn]
if min_runs > 0:  pdf = pdf[pdf["total_runs"]    >= min_runs]
if min_wkts > 0:  pdf = pdf[pdf["total_wickets"]  >= min_wkts]

if role_pick == "Batter":
    pdf = pdf[pdf["role"].isin(["Batter","All-rounder"])].copy()
    log_vol = np.log1p(pdf["total_balls_faced"]) / np.log1p(pdf["total_balls_faced"].max())
    raw = pdf["mean_bat_impact"] * log_vol
    bmax = raw.max() or 1
    pdf["display_impact"] = raw / bmax * 100; imp_label = "Batting Impact"
elif role_pick == "Bowler":
    pdf = pdf[pdf["role"].isin(["Bowler","All-rounder"])].copy()
    log_vol = np.log1p(pdf["total_balls_bowled"]) / np.log1p(pdf["total_balls_bowled"].max())
    raw = pdf["mean_bowl_impact"] * log_vol
    wmax = raw.max() or 1
    pdf["display_impact"] = raw / wmax * 100; imp_label = "Bowling Impact"
elif role_pick == "All-rounder":
    pdf = pdf[pdf["role"]=="All-rounder"].copy()
    bvol = np.log1p(pdf["total_balls_faced"]) / np.log1p(pdf["total_balls_faced"].max())
    wvol = np.log1p(pdf["total_balls_bowled"]) / np.log1p(pdf["total_balls_bowled"].max())
    raw = pdf["mean_bat_impact"] * bvol + pdf["mean_bowl_impact"] * wvol
    cmax = raw.max() or 1
    pdf["display_impact"] = raw / cmax * 100; imp_label = "Combined Impact"
else:
    if season_mode:
        pdf["display_impact"] = pdf["season_avg_impact"]; imp_label = "Season Avg Impact"
    else:
        pdf["display_impact"] = pdf["overall_impact"]; imp_label = "Overall Impact"

pdf = pdf.sort_values("display_impact", ascending=False).reset_index(drop=True)

season_tag = f" ({sb_season})" if season_mode else ""
st.sidebar.markdown(f"**{len(pdf)}** players{season_tag}  |  Ranked by **{imp_label}**")


# --- Hero header ---
st.markdown("""
<div class="hero">
  <h1>CricImpact</h1>
  <p>IPL T20 Player Impact Analytics Dashboard</p>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Player Profile", "Head-to-Head", "Match Explorer"])


# ==================== TAB 1 - OVERVIEW ====================
with tabs[0]:

    top = pdf.iloc[0] if len(pdf) > 0 else None
    cols = st.columns(4)
    with cols[0]: st.markdown(kpi_card("Players", len(pdf), CYAN), unsafe_allow_html=True)
    with cols[1]: st.markdown(kpi_card(f"Mean {imp_label}", f"{clipped_mean(pdf['display_impact']):.1f}", GOLD), unsafe_allow_html=True)
    with cols[2]: st.markdown(kpi_card("Top Player", top["player"] if top is not None else "-", GREEN), unsafe_allow_html=True)
    with cols[3]: st.markdown(kpi_card(f"Top {imp_label}", f"{top['display_impact']:.1f}" if top is not None else "-", ORANGE), unsafe_allow_html=True)

    # Season highlights
    if season_mode:
        sm = mdf[mdf["season"].astype(str) == sb_season]
        if len(sm) > 0:
            st.markdown('<hr class="sep">', unsafe_allow_html=True)
            st.markdown(f"#### Top Performers - {sb_season}")
            stop = sm.groupby("player")["impact_score"].mean().nlargest(5).reset_index()
            stop.columns = ["Player", "Avg Impact"]
            cs = st.columns(5)
            for i, (_, r) in enumerate(stop.iterrows()):
                with cs[i]:
                    st.markdown(kpi_card(r["Player"], f"{r['Avg Impact']:.1f}",
                                         [CYAN,ORANGE,GREEN,PURPLE,PINK][i]), unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Top 15 chart
    c1, c2 = st.columns(2)
    with c1:
        t15 = pdf.head(15).iloc[::-1]
        colors = [f"rgba(88,166,255,{0.4 + 0.6*i/14})" for i in range(15)]
        fig = go.Figure(go.Bar(
            x=t15["display_impact"], y=t15["player"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=t15["display_impact"].round(1), textposition="outside",
            textfont=dict(size=11, color=CYAN),
        ))
        fig.update_layout(**FIG_LAYOUT, **FIG_GRID, height=420,
                          title=dict(text=f"Top 15 - {imp_label}{season_tag}", font=dict(size=15, color=TEXT)),
                          xaxis_title=imp_label)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

    # Distribution
    with c2:
        fig = go.Figure(go.Histogram(
            x=pdf["display_impact"], nbinsx=35,
            marker=dict(color=CYAN, line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
            opacity=0.85,
        ))
        cmean = clipped_mean(pdf["display_impact"])
        fig.add_vline(x=cmean, line_dash="dash", line_color=GOLD, line_width=2,
                      annotation_text=f"Mean: {cmean:.1f}", annotation_font_color=GOLD)
        fig.update_layout(**FIG_LAYOUT, **FIG_GRID, height=420,
                          title=dict(text=f"{imp_label} Distribution{season_tag}", font=dict(size=15, color=TEXT)),
                          xaxis_title=imp_label, yaxis_title="Players", bargap=0.08)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

    c3, c4 = st.columns(2)
    # Role pie
    with c3:
        rc = pdf["role"].value_counts().reset_index()
        rc.columns = ["Role", "Count"]
        fig = go.Figure(go.Pie(
            labels=rc["Role"], values=rc["Count"], hole=0.55,
            marker=dict(colors=[CYAN, ORANGE, GREEN, PURPLE][:len(rc)],
                        line=dict(color=BG, width=2)),
            textfont=dict(color=TEXT, size=12),
            hoverinfo="label+percent+value",
        ))
        fig.update_layout(**FIG_LAYOUT, height=370,
                          title=dict(text="Role Distribution", font=dict(size=15, color=TEXT)),
                          showlegend=True,
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2)))
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

    # Bat vs Bowl scatter
    with c4:
        ar = pdf[(pdf["mean_bat_impact"] > 0) & (pdf["mean_bowl_impact"] > 0)]
        if len(ar) > 0:
            fig = px.scatter(ar, x="mean_bat_impact", y="mean_bowl_impact",
                             color="display_impact", size="total_innings",
                             hover_name="player",
                             color_continuous_scale=[[0,BG],[0.4,"#0d4a6b"],[1,CYAN]])
            fig.update_traces(marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.15)")))
            fig.update_layout(**FIG_LAYOUT, **FIG_GRID, height=370,
                              title=dict(text="Batting vs Bowling Impact", font=dict(size=15, color=TEXT)),
                              xaxis_title="Batting Impact", yaxis_title="Bowling Impact",
                              coloraxis_colorbar=dict(title=imp_label, tickfont=dict(color=TEXT2)))
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown("#### Leaderboards")
    c5, c6 = st.columns(2)

    with c5:
        tb = pdf[pdf["bat_innings"] >= 20].nlargest(10, "mean_bat_impact")
        fig = go.Figure(go.Bar(
            x=tb["mean_bat_impact"], y=tb["player"], orientation="h",
            marker=dict(color=tb["mean_bat_impact"],
                        colorscale=[[0,"#1a1e24"],[0.4,"#6b3a0d"],[1,ORANGE]],
                        line=dict(width=0)),
            text=tb["mean_bat_impact"].round(3), textposition="outside",
            textfont=dict(color=ORANGE, size=10),
        ))
        fig.update_layout(**FIG_LAYOUT, height=360,
                          title=dict(text="Top 10 Batters", font=dict(size=14, color=TEXT)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

    with c6:
        tw = pdf[pdf["bowl_innings"] >= 20].nlargest(10, "mean_bowl_impact")
        fig = go.Figure(go.Bar(
            x=tw["mean_bowl_impact"], y=tw["player"], orientation="h",
            marker=dict(color=tw["mean_bowl_impact"],
                        colorscale=[[0,"#1a1e24"],[0.4,"#2d1a4e"],[1,PURPLE]],
                        line=dict(width=0)),
            text=tw["mean_bowl_impact"].round(3), textposition="outside",
            textfont=dict(color=PURPLE, size=10),
        ))
        fig.update_layout(**FIG_LAYOUT, height=360,
                          title=dict(text="Top 10 Bowlers", font=dict(size=14, color=TEXT)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

    # Full table
    with st.expander("Full Player Table"):
        srch = st.text_input("Search player", "", key="tbl_search")
        tcols = [c for c in ["player","role","display_impact","bat_innings","total_runs",
                              "avg_bat_sr","true_bat_avg","bowl_innings","total_wickets",
                              "avg_economy","pom_awards","win_rate"] if c in pdf.columns]
        td = pdf[tcols].rename(columns={"display_impact": imp_label})
        if srch:
            td = td[td["player"].str.contains(srch, case=False, na=False)]
        st.dataframe(td.style.format({
            imp_label: "{:.1f}", "avg_bat_sr": "{:.1f}", "true_bat_avg": "{:.1f}",
            "avg_economy": "{:.2f}", "win_rate": "{:.2f}",
        }).background_gradient(subset=[imp_label], cmap="cool"),
        width='stretch', height=440)


# ==================== TAB 2 - PLAYER PROFILE ====================
with tabs[1]:
    pc1, pc2 = st.columns([3, 1])
    with pc1:
        sel = st.selectbox("Search player", pdf["player"].tolist(), key="prof_sel")
    with pc2:
        sel_season = st.selectbox("Season filter", ["All"] + all_seasons, key="prof_season")

    if sel:
        p = pdf[pdf["player"] == sel].iloc[0]
        rank_idx = pdf.index[pdf["player"] == sel][0] + 1

        # Compute season-specific impact if a season is selected
        if sel_season != "All" and mdf is not None:
            sm = mdf[(mdf["player"] == sel) & (mdf["season"].astype(str) == sel_season)]
            prof_impact = sm["impact_score"].mean() if len(sm) > 0 else p["overall_impact"]
            prof_label = f"Impact ({sel_season})"
        else:
            prof_impact = p["overall_impact"]
            prof_label = "Impact"

        # Hero KPIs
        kcols = st.columns(6)
        kpi_data = [
            ("Rank", f"#{rank_idx}", GOLD),
            (prof_label, f"{prof_impact:.1f}", CYAN),
            ("Role", p["role"], PURPLE),
            ("Bat Inn", si(p["bat_innings"]), ORANGE),
            ("Bowl Inn", si(p["bowl_innings"]), GREEN),
            ("POM", si(p["pom_awards"]), PINK),
        ]
        for i, (l, v, co) in enumerate(kpi_data):
            with kcols[i]:
                st.markdown(kpi_card(l, v, co), unsafe_allow_html=True)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Gauge + Stats
        g_col, bat_col, bwl_col = st.columns([1.2, 1, 1])

        with g_col:
            score = prof_impact
            ref_mean = clipped_mean(pdf["overall_impact"])
            if score >= 80:   g_color = GREEN
            elif score >= 60: g_color = GOLD
            elif score >= 40: g_color = ORANGE
            else:             g_color = RED

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=score,
                delta=dict(reference=ref_mean, valueformat=".1f",
                           increasing=dict(color=GREEN), decreasing=dict(color=RED)),
                number=dict(font=dict(size=42, color=g_color)),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor=TEXT, tickfont=dict(size=11, color=TEXT)),
                    bar=dict(color=g_color, thickness=0.7),
                    bgcolor=SURFACE,
                    bordercolor=BORDER,
                    steps=[
                        dict(range=[0,40],  color="rgba(255,123,114,0.1)"),
                        dict(range=[40,60], color="rgba(227,179,65,0.1)"),
                        dict(range=[60,80], color="rgba(63,185,80,0.1)"),
                        dict(range=[80,100],color="rgba(88,166,255,0.1)"),
                    ],
                    threshold=dict(line=dict(color=PINK, width=3), value=score),
                ),
                title=dict(text="Impact Meter", font=dict(size=14, color=TEXT)),
            ))
            gauge_layout = {k: v for k, v in FIG_LAYOUT.items() if k != "margin"}
            fig.update_layout(**gauge_layout, height=280,
                              margin=dict(l=30, r=30, t=60, b=20))
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

        with bat_col:
            st.markdown(f"<h4 style='color:{ORANGE};margin-bottom:8px;'>Batting</h4>", unsafe_allow_html=True)
            bstats = [
                ("Runs", si(p.get("total_runs")), ORANGE),
                ("Average", f"{p.get('true_bat_avg',0):.1f}", GOLD),
                ("Strike Rate", f"{p.get('avg_bat_sr',0):.1f}", CYAN),
                ("4s / 6s", f"{si(p.get('total_fours'))} / {si(p.get('total_sixes'))}", GREEN),
                ("Dot %", f"{p.get('dot_pct_faced',0):.1f}%", TEXT2),
                ("Clutch WPA", f"{p.get('total_clutch_bat',0):.3f}", PINK),
            ]
            for l, v, co in bstats:
                st.markdown(stat_row(l, v, co), unsafe_allow_html=True)

        with bwl_col:
            st.markdown(f"<h4 style='color:{PURPLE};margin-bottom:8px;'>Bowling</h4>", unsafe_allow_html=True)
            wstats = [
                ("Wickets", si(p.get("total_wickets")), PURPLE),
                ("Economy", f"{p.get('avg_economy',0):.2f}", GREEN),
                ("Balls", si(p.get("total_balls_bowled")), CYAN),
                ("Death Wkts", si(p.get("total_death_wickets")), RED),
                ("PP Wkts", si(p.get("total_pp_wickets")), ORANGE),
                ("Clutch WPA", f"{p.get('total_clutch_bowl',0):.3f}", PINK),
            ]
            for l, v, co in wstats:
                st.markdown(stat_row(l, v, co), unsafe_allow_html=True)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Radar + Trend
        r_col, t_col = st.columns(2)
        with r_col:
            rv = radar(p, pdf)
            cats = list(rv.keys()); vals = list(rv.values())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(88,166,255,0.15)",
                line=dict(color=CYAN, width=2.5),
                marker=dict(size=6, color=CYAN, line=dict(width=1, color=BG)),
            ))
            fig.update_layout(**FIG_LAYOUT, height=370,
                              polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(visible=True, range=[0,1],
                                                         gridcolor="rgba(255,255,255,0.06)",
                                                         tickfont=dict(size=9, color=TEXT2)),
                                         angularaxis=dict(tickfont=dict(size=11, color=TEXT),
                                                          gridcolor="rgba(255,255,255,0.06)")),
                              title=dict(text="Strength Radar", font=dict(size=14, color=TEXT)))
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

        with t_col:
            trend = get_trend(sel, bat_df, bwl_df,
                              season=sel_season if sel_season != "All" else None)
            if trend is not None and len(trend) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend["start_date"], y=trend["impact_score"],
                    mode="lines+markers",
                    line=dict(color=CYAN, width=3, shape="spline"),
                    marker=dict(size=8, color=CYAN, line=dict(width=2, color=BG)),
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
                ))
                if len(trend) >= 5:
                    trend["ma"] = trend["impact_score"].rolling(3, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=trend["start_date"], y=trend["ma"],
                        mode="lines", name="3-inn avg",
                        line=dict(color=GOLD, width=1.5, dash="dot"),
                    ))
                sfx = f" ({sel_season})" if sel_season != "All" else " - Recent Form"
                fig.update_layout(**FIG_LAYOUT, **FIG_GRID, height=370,
                                  title=dict(text=f"{sel}{sfx}", font=dict(size=14, color=TEXT)),
                                  xaxis_title="Date", yaxis_title="Impact Score",
                                  showlegend=True,
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2),
                                              orientation="h", y=1.08))
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)
            else:
                st.info("Not enough innings for a trend chart.")


# ==================== TAB 3 - HEAD-TO-HEAD ====================
with tabs[2]:
    plist = pdf["player"].tolist()
    ha, hb = st.columns(2)
    with ha: pa = st.selectbox("Player A", plist, key="h2h_a")
    with hb: pb = st.selectbox("Player B", plist, index=min(1,len(plist)-1), key="h2h_b")

    if pa and pb and pa != pb:
        a = pdf[pdf["player"]==pa].iloc[0]
        b = pdf[pdf["player"]==pb].iloc[0]

        # Side KPIs
        lc, mc_col, rc_col = st.columns([1, 2.5, 1])
        with lc:
            st.markdown(f"<h3 style='color:{CYAN};text-align:center;'>{pa}</h3>", unsafe_allow_html=True)
            st.markdown(kpi_card("Impact", f"{a['overall_impact']:.1f}", CYAN), unsafe_allow_html=True)
            st.markdown(kpi_card("Runs", si(a["total_runs"]), ORANGE), unsafe_allow_html=True)
            st.markdown(kpi_card("Wickets", si(a["total_wickets"]), PURPLE), unsafe_allow_html=True)
            st.markdown(kpi_card("Win Rate", f"{a['win_rate']:.0%}", GREEN), unsafe_allow_html=True)

        with rc_col:
            st.markdown(f"<h3 style='color:{ORANGE};text-align:center;'>{pb}</h3>", unsafe_allow_html=True)
            st.markdown(kpi_card("Impact", f"{b['overall_impact']:.1f}", CYAN), unsafe_allow_html=True)
            st.markdown(kpi_card("Runs", si(b["total_runs"]), ORANGE), unsafe_allow_html=True)
            st.markdown(kpi_card("Wickets", si(b["total_wickets"]), PURPLE), unsafe_allow_html=True)
            st.markdown(kpi_card("Win Rate", f"{b['win_rate']:.0%}", GREEN), unsafe_allow_html=True)

        # Dumbbell chart
        with mc_col:
            metrics = [("Overall Impact","overall_impact"), ("Batting Avg","true_bat_avg"),
                       ("Strike Rate","avg_bat_sr"), ("Economy","avg_economy"),
                       ("Clutch","total_clutch_bat"), ("Win Rate","win_rate"),
                       ("POM Awards","pom_awards")]
            fig = go.Figure()
            y_labels, xa_vals, xb_vals = [], [], []
            for label, col in metrics:
                if col not in pdf.columns: continue
                va, vb = float(a.get(col,0)), float(b.get(col,0))
                mx = max(abs(va), abs(vb), 1e-9)
                if col == "avg_economy":
                    na = (1-va/mx)*100; nb = (1-vb/mx)*100
                else:
                    na = va/mx*100; nb = vb/mx*100
                y_labels.append(label)
                xa_vals.append(na); xb_vals.append(nb)
                fig.add_trace(go.Scatter(
                    x=[na, nb], y=[label, label], mode="lines",
                    line=dict(color="rgba(255,255,255,0.15)", width=3),
                    showlegend=False, hoverinfo="skip",
                ))

            fig.add_trace(go.Scatter(
                x=xa_vals, y=y_labels, mode="markers+text",
                marker=dict(size=14, color=CYAN, line=dict(width=2, color=BG), symbol="circle"),
                text=[f"{v:.0f}" for v in xa_vals], textposition="top center",
                textfont=dict(size=9, color=CYAN), name=pa,
            ))
            fig.add_trace(go.Scatter(
                x=xb_vals, y=y_labels, mode="markers+text",
                marker=dict(size=14, color=ORANGE, line=dict(width=2, color=BG), symbol="diamond"),
                text=[f"{v:.0f}" for v in xb_vals], textposition="bottom center",
                textfont=dict(size=9, color=ORANGE), name=pb,
            ))
            fig.update_layout(**FIG_LAYOUT, height=400,
                              title=dict(text="Normalized Comparison", font=dict(size=14, color=TEXT)),
                              xaxis=dict(title="% of better player", range=[0, 115],
                                         gridcolor="rgba(255,255,255,0.04)"),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2),
                                          orientation="h", y=1.08))
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Radar + Trend
        rr_col, tt_col = st.columns(2)
        with rr_col:
            rva, rvb = radar(a, pdf), radar(b, pdf)
            cats = list(rva.keys())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(rva.values()) + [list(rva.values())[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(88,166,255,0.12)",
                line=dict(color=CYAN, width=2.5), name=pa,
                marker=dict(size=5, color=CYAN),
            ))
            fig.add_trace(go.Scatterpolar(
                r=list(rvb.values()) + [list(rvb.values())[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(240,136,62,0.12)",
                line=dict(color=ORANGE, width=2.5), name=pb,
                marker=dict(size=5, color=ORANGE),
            ))
            fig.update_layout(**FIG_LAYOUT, height=400,
                              polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(visible=True, range=[0,1],
                                                         gridcolor="rgba(255,255,255,0.06)",
                                                         tickfont=dict(size=9, color=TEXT2)),
                                         angularaxis=dict(tickfont=dict(size=11, color=TEXT),
                                                          gridcolor="rgba(255,255,255,0.06)")),
                              title=dict(text="Radar Comparison", font=dict(size=14, color=TEXT)),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2),
                                          orientation="h", y=-0.08))
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

        with tt_col:
            ta = get_trend(pa, bat_df, bwl_df)
            tb_t = get_trend(pb, bat_df, bwl_df)
            if ta is not None and tb_t is not None and len(ta) >= 3 and len(tb_t) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ta["start_date"], y=ta["impact_score"],
                    mode="lines+markers", name=pa,
                    line=dict(color=CYAN, width=2.5, shape="spline"),
                    marker=dict(size=7, color=CYAN, line=dict(width=1, color=BG)),
                ))
                fig.add_trace(go.Scatter(
                    x=tb_t["start_date"], y=tb_t["impact_score"],
                    mode="lines+markers", name=pb,
                    line=dict(color=ORANGE, width=2.5, shape="spline"),
                    marker=dict(size=7, color=ORANGE, line=dict(width=1, color=BG)),
                ))
                fig.update_layout(**FIG_LAYOUT, **FIG_GRID, height=400,
                                  title=dict(text="Form Comparison", font=dict(size=14, color=TEXT)),
                                  xaxis_title="Date", yaxis_title="Impact Score",
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2),
                                              orientation="h", y=1.08))
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)
            else:
                st.info("Not enough innings data for both players.")

        # Raw stats table
        raw_l, raw_a, raw_b = [], [], []
        metrics_raw = [("Overall Impact","overall_impact"), ("Batting Avg","true_bat_avg"),
                       ("Strike Rate","avg_bat_sr"), ("Economy","avg_economy"),
                       ("Clutch","total_clutch_bat"), ("Win Rate","win_rate"), ("POM","pom_awards")]
        for label,col in metrics_raw:
            if col in pdf.columns:
                raw_l.append(label); raw_a.append(float(a.get(col,0))); raw_b.append(float(b.get(col,0)))
        comp = pd.DataFrame({"Metric": raw_l, pa: raw_a, pb: raw_b})
        comp["Better"] = comp.apply(
            lambda r: pa if (r[pa] > r[pb] if r["Metric"] != "Economy" else r[pa] < r[pb]) else pb, axis=1)
        st.dataframe(comp.style.format({pa:"{:.2f}", pb:"{:.2f}"}),
                     width='stretch', hide_index=True)

        # Rivalry
        if riv_df is not None:
            h2h = riv_df[((riv_df["batter"]==pa)&(riv_df["bowler_name"]==pb)) |
                         ((riv_df["batter"]==pb)&(riv_df["bowler_name"]==pa))]
            if len(h2h) > 0:
                st.markdown("#### Batter vs Bowler Rivalry")
                st.dataframe(h2h, width='stretch')

    elif pa == pb:
        st.warning("Pick two different players to compare.")


# ==================== TAB 4 - MATCH EXPLORER ====================
with tabs[3]:
    if mdf is None:
        st.info("match_rankings.csv not found - run the pipeline first.")
    else:
        mf = mdf.copy()
        seasons = sorted(mf["season"].dropna().astype(str).unique())

        f1, f2 = st.columns(2)
        with f1:
            m_season = st.selectbox("Season", ["All"]+seasons, key="mx_season")
        with f2:
            mf_t = mf[mf["season"].astype(str)==m_season] if m_season!="All" else mf
            teams = sorted(mf_t["player_team"].dropna().astype(str).unique())
            m_team = st.selectbox("Team", ["All"]+teams, key="mx_team")

        if m_season != "All": mf = mf[mf["season"].astype(str) == m_season]
        if m_team  != "All":  mf = mf[mf["player_team"].astype(str) == m_team]

        mlabels = build_match_labels(mf)
        mids = sorted(mlabels.keys())

        if mids:
            id_map = {v:k for k,v in mlabels.items()}
            disp = [mlabels[m] for m in mids]
            pick = st.selectbox("Select Match", disp, key="mx_pick")
            sel_id = id_map[pick]

            m = mf[mf["match_id"]==sel_id].sort_values("rank").head(22)
            venue = m["venue"].iloc[0] if "venue" in m.columns else ""
            seas  = m["season"].iloc[0] if "season" in m.columns else ""
            tms   = m["player_team"].unique().tolist()

            st.markdown(f"""
            <div class="glass" style="text-align:center;padding:12px 20px;">
                <span style="color:{CYAN};font-weight:700;font-size:1.1rem;">
                    {' vs '.join(tms)}
                </span>
                <br><span style="color:{TEXT2};font-size:0.85rem;">{venue}  |  Season {seas}</span>
            </div>""", unsafe_allow_html=True)

            top11 = m.head(11)
            team_colors = {tms[i]: [CYAN, ORANGE, GREEN, PURPLE][i % 4] for i in range(len(tms))}
            bar_colors = [team_colors.get(t, CYAN) for t in top11["player_team"]]

            fig = go.Figure(go.Bar(
                x=top11["performance_score"], y=top11["player"], orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=top11["key_stat"], textposition="outside",
                textfont=dict(size=10, color=TEXT2),
            ))
            fig.update_layout(**FIG_LAYOUT, height=380,
                              title=dict(text="Top Performers", font=dict(size=14, color=TEXT)),
                              xaxis_title="Performance Score")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CFG)

            show_c = [c for c in ["player","role","player_team","key_stat",
                                  "performance_score","impact_score","impact_category",
                                  "wpa_score","clutch_score","rank"] if c in m.columns]
            st.dataframe(m[show_c], width='stretch', height=400, hide_index=True)
        else:
            st.caption("No matches for the selected filters.")

