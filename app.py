"""
IPL T20 Impact Dashboard — Professional Analysis View
Uses Plotly for interactive charts with a soft, warm color palette.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = "cricket_impact"
PLAYER_FILE = os.path.join(DATA_DIR, "all_players_impact_T20.csv")
MATCH_FILE = os.path.join(DATA_DIR, "match_rankings.csv")
BAT_FILE = os.path.join(DATA_DIR, "batting_impact.csv")
BOWL_FILE = os.path.join(DATA_DIR, "bowling_impact.csv")
RIVALRY_FILE = os.path.join(DATA_DIR, "batter_vs_bowler_rivalry.csv")

# ─── Color palette ────────────────────────────────────────────────────────────
PAL = px.colors.qualitative.Pastel
BG = "#f8f5f0"
CARD = "#ffffff"
TEXT = "#3d3d3d"
ACCENT = "#5b9bd5"
ACCENT2 = "#ed7d6a"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor="#fdfbf7",
    font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
    margin=dict(l=40, r=20, t=40, b=30),
    hoverlabel=dict(bgcolor=CARD, font_size=12),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def safe_int(v, d=0):
    try:
        return int(v)
    except Exception:
        return d


def role_of(row):
    b, w = safe_int(row.get("bat_innings")), safe_int(row.get("bowl_innings"))
    if b >= 10 and w >= 10:
        return "All-rounder"
    return "Batter" if b >= w else "Bowler"


def player_trend(name, bat, bowl):
    df = None
    if bat is not None and "player" in bat.columns:
        df = bat[bat["player"] == name].copy()
    if (df is None or len(df) < 3) and bowl is not None and "player" in bowl.columns:
        df = bowl[bowl["player"] == name].copy()
    if df is None or "impact_score" not in df.columns:
        return None
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    return df.sort_values("start_date").tail(10)


def radar_vals(row, players):
    bm = max(players["mean_bat_impact"].max(), 1e-9)
    wm = max(players["mean_bowl_impact"].max(), 1e-9)
    cm = max((players["total_clutch_bat"].abs() + players["total_clutch_bowl"].abs()).max(), 1e-9)
    return [
        min(row.get("mean_bat_impact", 0) / bm, 1),
        min(row.get("mean_bowl_impact", 0) / wm, 1),
        min(abs(row.get("total_clutch_bat", 0) + row.get("total_clutch_bowl", 0)) / cm, 1),
        min(row.get("not_out_pct", 0), 1),
        row.get("win_rate", 0),
    ]


# ─── Page config & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="IPL T20 Impact Dashboard", layout="wide", page_icon="🏏")

st.markdown(f"""
<style>
  .stApp {{ background: {BG}; }}
  .block-container {{ padding-top: 1rem; max-width: 1280px; }}
  h1, h2, h3 {{ color: {TEXT}; }}
  [data-testid="stMetric"] {{
      background: {CARD}; border-radius: 10px; padding: 12px 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  [data-testid="stMetricValue"] {{ font-size: 1.3rem; color: {TEXT}; }}
  [data-testid="stMetricLabel"] {{ color: #777; font-size: 0.8rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{
      background: {CARD}; border-radius: 8px 8px 0 0;
      padding: 8px 18px; font-weight: 600; color: {TEXT};
      border: 1px solid #e8e4de; border-bottom: none;
  }}
  .stTabs [aria-selected="true"] {{
      background: {ACCENT}; color: #fff; border-color: {ACCENT};
  }}
  section[data-testid="stSidebar"] {{ background: #f0ece6; }}
</style>
""", unsafe_allow_html=True)


# ─── Load data ────────────────────────────────────────────────────────────────
players_raw = load(PLAYER_FILE)
match_df = load(MATCH_FILE)
bat_df = load(BAT_FILE)
bowl_df = load(BOWL_FILE)
rivalry_df = load(RIVALRY_FILE)

if players_raw is None:
    st.error("Run the pipeline first — `all_players_impact_T20.csv` not found.")
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏏 Filters")
min_inn = st.sidebar.slider("Min innings played", 0, 100, 10)
role_opts = ["All", "Batter", "Bowler", "All-rounder"]
role_pick = st.sidebar.radio("Role", role_opts, horizontal=True)

players = players_raw.copy()
players["role"] = players.apply(role_of, axis=1)
players["total_innings"] = players["bat_innings"] + players["bowl_innings"]
players = players[players["total_innings"] >= min_inn].copy()
if role_pick != "All":
    players = players[players["role"] == role_pick].copy()
players = players.sort_values("overall_impact", ascending=False).reset_index(drop=True)
st.sidebar.markdown(f"**{len(players)}** players shown")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏏 IPL T20 — Player Impact Dashboard")

tabs = st.tabs(["📊 Overview", "🧑 Player Profile", "⚔️ Head-to-Head", "🏟️ Match Explorer"])


# ═══════════════════════ TAB 1 — OVERVIEW ═════════════════════════════════════
with tabs[0]:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players", len(players))
    m2.metric("Median Impact", f"{players['overall_impact'].median():.1f}")
    top1 = players.iloc[0] if len(players) > 0 else None
    m3.metric("🏆 Top Player", top1["player"] if top1 is not None else "-")
    m4.metric("Top Score", f"{top1['overall_impact']:.1f}" if top1 is not None else "-")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        top15 = players.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top15["overall_impact"], y=top15["player"], orientation="h",
            marker=dict(color=top15["overall_impact"], colorscale="Tealgrn", showscale=False),
            text=top15["overall_impact"].round(1), textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title="Top 15 Players by Impact",
                          xaxis_title="Overall Impact", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(players, x="overall_impact", nbins=35,
                           color_discrete_sequence=[ACCENT], title="Impact Score Distribution")
        fig.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Overall Impact", yaxis_title="Players")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        rc = players["role"].value_counts().reset_index()
        rc.columns = ["Role", "Count"]
        fig = px.pie(rc, names="Role", values="Count", color_discrete_sequence=PAL,
                     title="Role Distribution", hole=0.45)
        fig.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        ar = players[(players["mean_bat_impact"] > 0) & (players["mean_bowl_impact"] > 0)]
        if len(ar) > 0:
            fig = px.scatter(ar, x="mean_bat_impact", y="mean_bowl_impact",
                             color="overall_impact", size="total_innings", hover_name="player",
                             color_continuous_scale="Tealgrn", title="Batting vs Bowling Impact")
            fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Batting Impact", yaxis_title="Bowling Impact")
            st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        top_bat = players[players["bat_innings"] >= 20].nlargest(10, "mean_bat_impact")
        fig = px.bar(top_bat, x="mean_bat_impact", y="player", orientation="h",
                     color="total_runs", color_continuous_scale="Sunset",
                     title="Top 10 Batters (Batting Impact)", text=top_bat["mean_bat_impact"].round(3))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        top_bowl = players[players["bowl_innings"] >= 20].nlargest(10, "mean_bowl_impact")
        fig = px.bar(top_bowl, x="mean_bowl_impact", y="player", orientation="h",
                     color="total_wickets", color_continuous_scale="Purp",
                     title="Top 10 Bowlers (Bowling Impact)", text=top_bowl["mean_bowl_impact"].round(3))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Player Table"):
        show_cols = [c for c in ["player", "role", "overall_impact", "bat_innings", "total_runs",
                                 "avg_bat_sr", "true_bat_avg", "bowl_innings", "total_wickets",
                                 "avg_economy", "pom_awards", "win_rate"] if c in players.columns]
        st.dataframe(
            players[show_cols].style.format({
                "overall_impact": "{:.1f}", "avg_bat_sr": "{:.1f}",
                "true_bat_avg": "{:.1f}", "avg_economy": "{:.2f}", "win_rate": "{:.2f}",
            }).background_gradient(subset=["overall_impact"], cmap="YlGnBu"),
            use_container_width=True, height=420,
        )


# ═══════════════════════ TAB 2 — PLAYER PROFILE ══════════════════════════════
with tabs[1]:
    sel = st.selectbox("Search player", players["player"].tolist(), key="profile_sel")
    if sel:
        p = players[players["player"] == sel].iloc[0]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Impact", f"{p['overall_impact']:.1f}")
        k2.metric("Role", p["role"])
        k3.metric("Bat Inn", safe_int(p["bat_innings"]))
        k4.metric("Bowl Inn", safe_int(p["bowl_innings"]))
        k5.metric("POM Awards", safe_int(p["pom_awards"]))

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p["overall_impact"],
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TEXT),
                bar=dict(color=ACCENT), bgcolor="#f0ece6",
                steps=[
                    dict(range=[0, 40], color="#fde8e4"),
                    dict(range=[40, 60], color="#fef3e0"),
                    dict(range=[60, 80], color="#e6f4d7"),
                    dict(range=[80, 100], color="#d4eaf7"),
                ],
                threshold=dict(line=dict(color=ACCENT2, width=3), value=p["overall_impact"]),
            ),
            title=dict(text="Impact Meter", font=dict(size=14)),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("#### 🏏 Batting Profile")
            for k, v in {
                "Runs": safe_int(p.get("total_runs")),
                "Innings": safe_int(p.get("bat_innings")),
                "Avg": f"{p.get('true_bat_avg', 0):.1f}",
                "SR": f"{p.get('avg_bat_sr', 0):.1f}",
                "4s / 6s": f"{safe_int(p.get('total_fours'))} / {safe_int(p.get('total_sixes'))}",
                "Dot %": f"{p.get('dot_pct_faced', 0):.1f}%",
                "Clutch WPA": f"{p.get('total_clutch_bat', 0):.3f}",
            }.items():
                st.markdown(f"**{k}:** {v}")

        with right:
            st.markdown("#### 🎯 Bowling Profile")
            for k, v in {
                "Wickets": safe_int(p.get("total_wickets")),
                "Innings": safe_int(p.get("bowl_innings")),
                "Economy": f"{p.get('avg_economy', 0):.2f}",
                "Balls Bowled": safe_int(p.get("total_balls_bowled")),
                "Death Wkts": safe_int(p.get("total_death_wickets")),
                "PP Wkts": safe_int(p.get("total_pp_wickets")),
                "Clutch WPA": f"{p.get('total_clutch_bowl', 0):.3f}",
            }.items():
                st.markdown(f"**{k}:** {v}")

        cats = ["Batting", "Bowling", "Clutch", "Consistency", "Win Impact"]
        rv = radar_vals(p, players)
        fig = go.Figure(go.Scatterpolar(
            r=rv + [rv[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(91,155,213,0.25)",
            line=dict(color=ACCENT, width=2), marker=dict(size=5),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          polar=dict(bgcolor="#fdfbf7",
                                     radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                                     angularaxis=dict(tickfont=dict(size=11))),
                          title="Strength Radar")
        st.plotly_chart(fig, use_container_width=True)

        trend = player_trend(sel, bat_df, bowl_df)
        if trend is not None and len(trend) >= 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend["start_date"], y=trend["impact_score"],
                mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                marker=dict(size=7, color=ACCENT, line=dict(width=1, color=CARD)),
                fill="tozeroy", fillcolor="rgba(91,155,213,0.12)",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                              title=f"{sel} — Last 10 Innings Trend",
                              xaxis_title="Date", yaxis_title="Impact Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough innings for a trend chart.")


# ═══════════════════════ TAB 3 — HEAD-TO-HEAD ═════════════════════════════════
with tabs[2]:
    plist = players["player"].tolist()
    ca, cb = st.columns(2)
    with ca:
        pa = st.selectbox("Player A", plist, key="cmp_a")
    with cb:
        pb = st.selectbox("Player B", plist, index=min(1, len(plist) - 1), key="cmp_b")

    if pa and pb and pa != pb:
        a = players[players["player"] == pa].iloc[0]
        b = players[players["player"] == pb].iloc[0]

        mc = st.columns([1, 2, 1])
        with mc[0]:
            st.markdown(f"### {pa}")
            st.metric("Impact", f"{a['overall_impact']:.1f}")
            st.metric("Runs", safe_int(a["total_runs"]))
            st.metric("Wickets", safe_int(a["total_wickets"]))
            st.metric("Win Rate", f"{a['win_rate']:.0%}")
        with mc[2]:
            st.markdown(f"### {pb}")
            st.metric("Impact", f"{b['overall_impact']:.1f}")
            st.metric("Runs", safe_int(b["total_runs"]))
            st.metric("Wickets", safe_int(b["total_wickets"]))
            st.metric("Win Rate", f"{b['win_rate']:.0%}")

        metrics = [("Overall Impact", "overall_impact"), ("Batting Avg", "true_bat_avg"),
                   ("Strike Rate", "avg_bat_sr"), ("Economy", "avg_economy"),
                   ("Clutch (Bat)", "total_clutch_bat"), ("Win Rate", "win_rate"),
                   ("POM Awards", "pom_awards")]
        labels, va, vb = [], [], []
        for label, col in metrics:
            if col in players.columns:
                labels.append(label)
                va.append(float(a.get(col, 0)))
                vb.append(float(b.get(col, 0)))

        with mc[1]:
            fig = go.Figure()
            fig.add_trace(go.Bar(name=pa, x=labels, y=va, marker_color=ACCENT,
                                 text=[f"{v:.1f}" for v in va], textposition="outside"))
            fig.add_trace(go.Bar(name=pb, x=labels, y=vb, marker_color=ACCENT2,
                                 text=[f"{v:.1f}" for v in vb], textposition="outside"))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=380,
                              title="Side-by-Side Comparison", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        cats = ["Batting", "Bowling", "Clutch", "Consistency", "Win Impact"]
        rv_a, rv_b = radar_vals(a, players), radar_vals(b, players)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=rv_a + [rv_a[0]], theta=cats + [cats[0]],
                                       fill="toself", fillcolor="rgba(91,155,213,0.2)",
                                       line=dict(color=ACCENT, width=2), name=pa))
        fig.add_trace(go.Scatterpolar(r=rv_b + [rv_b[0]], theta=cats + [cats[0]],
                                       fill="toself", fillcolor="rgba(237,125,106,0.2)",
                                       line=dict(color=ACCENT2, width=2), name=pb))
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                          polar=dict(bgcolor="#fdfbf7",
                                     radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                                     angularaxis=dict(tickfont=dict(size=11))),
                          title="Radar Comparison", legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

        ta, tb = player_trend(pa, bat_df, bowl_df), player_trend(pb, bat_df, bowl_df)
        if ta is not None and tb is not None and len(ta) >= 3 and len(tb) >= 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ta["start_date"], y=ta["impact_score"],
                                     mode="lines+markers", name=pa,
                                     line=dict(color=ACCENT, width=2.5), marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=tb["start_date"], y=tb["impact_score"],
                                     mode="lines+markers", name=pb,
                                     line=dict(color=ACCENT2, width=2.5), marker=dict(size=6)))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              title="Last 10 Innings — Trend Comparison",
                              xaxis_title="Date", yaxis_title="Impact Score",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        if rivalry_df is not None:
            h2h = rivalry_df[
                ((rivalry_df["batter"] == pa) & (rivalry_df["bowler_name"] == pb)) |
                ((rivalry_df["batter"] == pb) & (rivalry_df["bowler_name"] == pa))
            ]
            if len(h2h) > 0:
                st.markdown("#### 🔥 Batter vs Bowler Rivalry")
                st.dataframe(h2h, use_container_width=True)

    elif pa == pb:
        st.warning("Pick two different players to compare.")


# ═══════════════════════ TAB 4 — MATCH EXPLORER ══════════════════════════════
with tabs[3]:
    if match_df is None:
        st.info("match_rankings.csv not found — run the pipeline first.")
    else:
        mf = match_df.copy()
        seasons = sorted(mf["season"].dropna().astype(str).unique())
        sel_season = st.selectbox("Season", ["All"] + seasons, key="match_season")
        if sel_season != "All":
            mf = mf[mf["season"].astype(str) == sel_season]

        teams = sorted(mf["player_team"].dropna().astype(str).unique())
        sel_team = st.selectbox("Team", ["All"] + teams, key="match_team")
        if sel_team != "All":
            mf = mf[mf["player_team"].astype(str) == sel_team]

        match_ids = sorted(mf["match_id"].unique().tolist())
        if match_ids:
            sel_mid = st.selectbox("Match ID", match_ids, key="match_id_sel")
            m = mf[mf["match_id"] == sel_mid].sort_values("rank").head(22)

            fig = px.bar(m.head(11), x="performance_score", y="player", orientation="h",
                         color="role", color_discrete_sequence=PAL,
                         title=f"Match {sel_mid} — Top Performers",
                         text="key_stat", hover_data=["player_team", "opposition"])
            fig.update_layout(**PLOTLY_LAYOUT, height=360,
                              yaxis=dict(autorange="reversed"),
                              xaxis_title="Performance Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(m, use_container_width=True, height=400)
        else:
            st.caption("No matches for the selected filters.")
