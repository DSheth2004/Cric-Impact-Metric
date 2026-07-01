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
from plotly.subplots import make_subplots

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
GREEN = "#4caf50"
PURPLE = "#9c6ade"

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
        ratio = min(b, w) / max(b, w) if max(b, w) > 0 else 0
        if ratio >= 0.4:
            return "All-rounder"
    return "Batter" if b >= w else "Bowler"


def player_trend(name, bat, bowl, season=None):
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


def radar_vals(row, players):
    bm = max(players["mean_bat_impact"].max(), 1e-9)
    wm = max(players["mean_bowl_impact"].max(), 1e-9)
    cm = max((players["total_clutch_bat"].abs() + players["total_clutch_bowl"].abs()).max(), 1e-9)
    dm_bat = max(players["dot_pct_faced"].max(), 1e-9)
    dm_bowl = max(players["dot_pct_bowled"].max(), 1e-9)
    pm = max(players["pom_awards"].max(), 1e-9)
    return {
        "Batting":      min(row.get("mean_bat_impact", 0) / bm, 1),
        "Bowling":      min(row.get("mean_bowl_impact", 0) / wm, 1),
        "Clutch":       min(abs(row.get("total_clutch_bat", 0) + row.get("total_clutch_bowl", 0)) / cm, 1),
        "Consistency":  min(row.get("not_out_pct", 0), 1),
        "Win Impact":   row.get("win_rate", 0),
        "Dot Efficiency": min(row.get("dot_pct_bowled", 0) / dm_bowl, 1) if row.get("bowl_innings", 0) > 5
                          else 1 - min(row.get("dot_pct_faced", 0) / dm_bat, 1),
        "Awards":       min(row.get("pom_awards", 0) / pm, 1),
    }


def stat_card_html(icon, label, value, color=ACCENT):
    return f"""
    <div style="background:{CARD};border-radius:10px;padding:14px 16px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);text-align:center;
                border-left:4px solid {color};">
        <div style="font-size:0.75rem;color:#888;margin-bottom:2px;">{icon} {label}</div>
        <div style="font-size:1.25rem;font-weight:700;color:{TEXT};">{value}</div>
    </div>"""


def build_match_labels(mf):
    """Build human-readable match labels: 'Team A vs Team B @ Venue'."""
    labels = {}
    for mid, grp in mf.groupby("match_id"):
        teams = grp["player_team"].unique().tolist()
        venue = grp["venue"].iloc[0] if "venue" in grp.columns else ""
        season = grp["season"].iloc[0] if "season" in grp.columns else ""
        if len(teams) >= 2:
            label = f"{teams[0]} vs {teams[1]}"
        elif len(teams) == 1:
            opp = grp["opposition"].iloc[0] if "opposition" in grp.columns else "?"
            label = f"{teams[0]} vs {opp}"
        else:
            label = str(mid)
        if venue:
            short_venue = venue.split(",")[0].strip()
            label += f"  •  {short_venue}"
        if season:
            label += f"  ({season})"
        labels[mid] = label
    return labels


# ─── Page config & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="IPL T20 Impact Dashboard", layout="wide", page_icon="🏏")

st.markdown(f"""
<style>
  .stApp {{ background: {BG}; }}
  .block-container {{ padding-top: 1rem; max-width: 1320px; }}
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
  .section-divider {{ border: none; border-top: 2px solid #e8e4de; margin: 1.2rem 0; }}
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

# Season filter in sidebar (applies to Overview tab)
all_seasons = []
if match_df is not None:
    all_seasons = sorted(match_df["season"].dropna().astype(str).unique().tolist())
sidebar_season = st.sidebar.selectbox("Season (Overview)", ["All"] + all_seasons, key="sidebar_season")

min_runs = st.sidebar.number_input("Min career runs", 0, 10000, 0, step=100)
min_wkts = st.sidebar.number_input("Min career wickets", 0, 500, 0, step=5)

players = players_raw.copy()
players["role"] = players.apply(role_of, axis=1)
players["total_innings"] = players["bat_innings"] + players["bowl_innings"]
players = players[players["total_innings"] >= min_inn].copy()
if min_runs > 0:
    players = players[players["total_runs"] >= min_runs].copy()
if min_wkts > 0:
    players = players[players["total_wickets"] >= min_wkts].copy()

# Role filter: Batter/Bowler also includes All-rounders; ranked by relevant skill
if role_pick == "Batter":
    players = players[players["role"].isin(["Batter", "All-rounder"])].copy()
    players["display_impact"] = players["mean_bat_impact"]
    impact_label = "Batting Impact"
elif role_pick == "Bowler":
    players = players[players["role"].isin(["Bowler", "All-rounder"])].copy()
    players["display_impact"] = players["mean_bowl_impact"]
    impact_label = "Bowling Impact"
elif role_pick == "All-rounder":
    players = players[players["role"] == "All-rounder"].copy()
    players["display_impact"] = players["mean_bat_impact"] + players["mean_bowl_impact"]
    impact_label = "Combined Impact"
else:
    players["display_impact"] = players["overall_impact"]
    impact_label = "Overall Impact"

players = players.sort_values("display_impact", ascending=False).reset_index(drop=True)
st.sidebar.markdown(f"**{len(players)}** players shown &nbsp;|&nbsp; Ranked by **{impact_label}**")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏏 IPL T20 — Player Impact Dashboard")

tabs = st.tabs(["Overview", "Player Profile", "Head-to-Head", "Match Explorer"])


# ═══════════════════════ TAB 1 — OVERVIEW ═════════════════════════════════════
with tabs[0]:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players", len(players))
    m2.metric(f"Median {impact_label}", f"{players['display_impact'].median():.1f}")
    top1 = players.iloc[0] if len(players) > 0 else None
    m3.metric("Top Player", top1["player"] if top1 is not None else "-")
    m4.metric(f"Top {impact_label}", f"{top1['display_impact']:.1f}" if top1 is not None else "-")

    # ── Top performers this season ──
    if sidebar_season != "All" and match_df is not None:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        season_matches = match_df[match_df["season"].astype(str) == sidebar_season]
        if len(season_matches) > 0:
            season_top = (season_matches.groupby("player")["impact_score"]
                          .mean().nlargest(5).reset_index())
            season_top.columns = ["Player", "Avg Impact"]
            st.markdown(f"#### ⭐ Top Performers — {sidebar_season}")
            cols = st.columns(5)
            for i, (_, row) in enumerate(season_top.iterrows()):
                with cols[i]:
                    st.metric(row["Player"], f"{row['Avg Impact']:.1f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        top15 = players.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top15["display_impact"], y=top15["player"], orientation="h",
            marker=dict(color=top15["display_impact"], colorscale="Tealgrn", showscale=False),
            text=top15["display_impact"].round(1), textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title=f"Top 15 Players by {impact_label}",
                          xaxis_title=impact_label, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(players, x="display_impact", nbins=35,
                           color_discrete_sequence=[ACCENT], title=f"{impact_label} Distribution")
        fig.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title=impact_label, yaxis_title="Players")
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
                             color="display_impact", size="total_innings", hover_name="player",
                             color_continuous_scale="Tealgrn", title="Batting vs Bowling Impact")
            fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Batting Impact", yaxis_title="Bowling Impact")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Leaderboards")
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
        search_text = st.text_input("Search player name", "", key="table_search")
        show_cols = [c for c in ["player", "role", "display_impact", "bat_innings", "total_runs",
                                 "avg_bat_sr", "true_bat_avg", "bowl_innings", "total_wickets",
                                 "avg_economy", "pom_awards", "win_rate"] if c in players.columns]
        table_data = players[show_cols].copy()
        table_data = table_data.rename(columns={"display_impact": impact_label})
        if search_text:
            table_data = table_data[table_data["player"].str.contains(search_text, case=False, na=False)]
        st.dataframe(
            table_data.style.format({
                impact_label: "{:.1f}", "avg_bat_sr": "{:.1f}",
                "true_bat_avg": "{:.1f}", "avg_economy": "{:.2f}", "win_rate": "{:.2f}",
            }).background_gradient(subset=[impact_label], cmap="YlGnBu"),
            use_container_width=True, height=420,
        )


# ═══════════════════════ TAB 2 — PLAYER PROFILE ══════════════════════════════
with tabs[1]:
    prof_c1, prof_c2 = st.columns([3, 1])
    with prof_c1:
        sel = st.selectbox("Search player", players["player"].tolist(), key="profile_sel")
    with prof_c2:
        prof_seasons = ["All"] + all_seasons
        sel_prof_season = st.selectbox("Season", prof_seasons, key="profile_season")

    if sel:
        p = players[players["player"] == sel].iloc[0]

        # ── Top metrics row ──
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Impact", f"{p['overall_impact']:.1f}")
        k2.metric("Role", p["role"])
        k3.metric("Bat Inn", safe_int(p["bat_innings"]))
        k4.metric("Bowl Inn", safe_int(p["bowl_innings"]))
        k5.metric("POM Awards", safe_int(p["pom_awards"]))

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Gauge + Batting + Bowling in 3 columns ──
        gauge_col, bat_col, bowl_col = st.columns([1, 1, 1])

        with gauge_col:
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
                title=dict(text="Impact Meter", font=dict(size=13)),
            ))
            gauge_layout = {**PLOTLY_LAYOUT, "margin": dict(l=30, r=30, t=50, b=10)}
            fig.update_layout(**gauge_layout, height=230)
            st.plotly_chart(fig, use_container_width=True)

        with bat_col:
            st.markdown("#### 🏏 Batting")
            bat_stats = [
                ("🏃", "Runs", safe_int(p.get("total_runs")), ACCENT),
                ("📊", "Avg", f"{p.get('true_bat_avg', 0):.1f}", GREEN),
                ("⚡", "SR", f"{p.get('avg_bat_sr', 0):.1f}", ACCENT2),
                ("4️⃣", "4s / 6s", f"{safe_int(p.get('total_fours'))} / {safe_int(p.get('total_sixes'))}", PURPLE),
                ("⏺️", "Dot %", f"{p.get('dot_pct_faced', 0):.1f}%", "#888"),
                ("🎯", "Clutch WPA", f"{p.get('total_clutch_bat', 0):.3f}", ACCENT),
            ]
            for icon, label, value, color in bat_stats:
                st.markdown(stat_card_html(icon, label, value, color), unsafe_allow_html=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        with bowl_col:
            st.markdown("#### 🎯 Bowling")
            bowl_stats = [
                ("🎳", "Wickets", safe_int(p.get("total_wickets")), ACCENT2),
                ("💰", "Economy", f"{p.get('avg_economy', 0):.2f}", GREEN),
                ("🔴", "Balls", safe_int(p.get("total_balls_bowled")), ACCENT),
                ("💀", "Death Wkts", safe_int(p.get("total_death_wickets")), PURPLE),
                ("🌅", "PP Wkts", safe_int(p.get("total_pp_wickets")), "#888"),
                ("🎯", "Clutch WPA", f"{p.get('total_clutch_bowl', 0):.3f}", ACCENT2),
            ]
            for icon, label, value, color in bowl_stats:
                st.markdown(stat_card_html(icon, label, value, color), unsafe_allow_html=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Radar + Trend side by side ──
        radar_col, trend_col = st.columns(2)

        with radar_col:
            rv = radar_vals(p, players)
            cats = list(rv.keys())
            vals = list(rv.values())
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(91,155,213,0.25)",
                line=dict(color=ACCENT, width=2), marker=dict(size=5),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=340,
                              polar=dict(bgcolor="#fdfbf7",
                                         radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                                         angularaxis=dict(tickfont=dict(size=11))),
                              title="Strength Radar")
            st.plotly_chart(fig, use_container_width=True)

        with trend_col:
            trend = player_trend(sel, bat_df, bowl_df,
                                 season=sel_prof_season if sel_prof_season != "All" else None)
            if trend is not None and len(trend) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend["start_date"], y=trend["impact_score"],
                    mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                    marker=dict(size=7, color=ACCENT, line=dict(width=1, color=CARD)),
                    fill="tozeroy", fillcolor="rgba(91,155,213,0.12)",
                ))
                title_suffix = f" ({sel_prof_season})" if sel_prof_season != "All" else " — Last 15 Innings"
                fig.update_layout(**PLOTLY_LAYOUT, height=340,
                                  title=f"{sel}{title_suffix}",
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

        # ── Side-by-side metric cards ──
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

        # ── Normalized comparison bars (each metric scaled 0-100%) ──
        metrics = [("Overall Impact", "overall_impact"), ("Batting Avg", "true_bat_avg"),
                   ("Strike Rate", "avg_bat_sr"), ("Economy", "avg_economy"),
                   ("Clutch (Bat)", "total_clutch_bat"), ("Win Rate", "win_rate"),
                   ("POM Awards", "pom_awards")]

        with mc[1]:
            norm_labels, norm_a, norm_b = [], [], []
            for label, col in metrics:
                if col not in players.columns:
                    continue
                val_a, val_b = float(a.get(col, 0)), float(b.get(col, 0))
                mx = max(abs(val_a), abs(val_b), 1e-9)
                # For economy, lower is better — invert
                if col == "avg_economy":
                    na = (1 - val_a / mx) * 100 if mx else 0
                    nb = (1 - val_b / mx) * 100 if mx else 0
                else:
                    na = val_a / mx * 100
                    nb = val_b / mx * 100
                norm_labels.append(label)
                norm_a.append(na)
                norm_b.append(nb)

            fig = go.Figure()
            fig.add_trace(go.Bar(name=pa, x=norm_labels, y=norm_a, marker_color=ACCENT,
                                 text=[f"{v:.0f}%" for v in norm_a], textposition="outside"))
            fig.add_trace(go.Bar(name=pb, x=norm_labels, y=norm_b, marker_color=ACCENT2,
                                 text=[f"{v:.0f}%" for v in norm_b], textposition="outside"))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=380,
                              title="Normalized Comparison (relative to each other)",
                              yaxis_title="% of better player",
                              yaxis=dict(range=[0, 115]),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Radar + Trend side by side ──
        radar_c, trend_c = st.columns(2)

        with radar_c:
            rv_a = radar_vals(a, players)
            rv_b = radar_vals(b, players)
            cats = list(rv_a.keys())
            va_r = list(rv_a.values())
            vb_r = list(rv_b.values())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=va_r + [va_r[0]], theta=cats + [cats[0]],
                                           fill="toself", fillcolor="rgba(91,155,213,0.2)",
                                           line=dict(color=ACCENT, width=2), name=pa))
            fig.add_trace(go.Scatterpolar(r=vb_r + [vb_r[0]], theta=cats + [cats[0]],
                                           fill="toself", fillcolor="rgba(237,125,106,0.2)",
                                           line=dict(color=ACCENT2, width=2), name=pb))
            fig.update_layout(**PLOTLY_LAYOUT, height=370,
                              polar=dict(bgcolor="#fdfbf7",
                                         radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                                         angularaxis=dict(tickfont=dict(size=11))),
                              title="Radar Comparison", legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True)

        with trend_c:
            ta, tb = player_trend(pa, bat_df, bowl_df), player_trend(pb, bat_df, bowl_df)
            if ta is not None and tb is not None and len(ta) >= 3 and len(tb) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ta["start_date"], y=ta["impact_score"],
                                         mode="lines+markers", name=pa,
                                         line=dict(color=ACCENT, width=2.5), marker=dict(size=6)))
                fig.add_trace(go.Scatter(x=tb["start_date"], y=tb["impact_score"],
                                         mode="lines+markers", name=pb,
                                         line=dict(color=ACCENT2, width=2.5), marker=dict(size=6)))
                fig.update_layout(**PLOTLY_LAYOUT, height=370,
                                  title="Recent Innings — Trend Comparison",
                                  xaxis_title="Date", yaxis_title="Impact Score",
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough innings data for both players to show trends.")

        # ── Raw stats comparison table ──
        if len(players) > 0:
            raw_labels, raw_a, raw_b = [], [], []
            for label, col in metrics:
                if col in players.columns:
                    raw_labels.append(label)
                    raw_a.append(float(a.get(col, 0)))
                    raw_b.append(float(b.get(col, 0)))
            comp_df = pd.DataFrame({"Metric": raw_labels, pa: raw_a, pb: raw_b})
            comp_df["Better"] = comp_df.apply(
                lambda r: pa if (r[pa] > r[pb] if r["Metric"] != "Economy" else r[pa] < r[pb]) else pb, axis=1)
            st.dataframe(comp_df.style.format({pa: "{:.2f}", pb: "{:.2f}"}),
                         use_container_width=True, hide_index=True)

        # ── Rivalry records ──
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

        f1, f2 = st.columns(2)
        with f1:
            sel_season = st.selectbox("Season", ["All"] + seasons, key="match_season")
        with f2:
            if sel_season != "All":
                mf_filtered = mf[mf["season"].astype(str) == sel_season]
            else:
                mf_filtered = mf
            teams = sorted(mf_filtered["player_team"].dropna().astype(str).unique())
            sel_team = st.selectbox("Team", ["All"] + teams, key="match_team")

        if sel_season != "All":
            mf = mf[mf["season"].astype(str) == sel_season]
        if sel_team != "All":
            mf = mf[mf["player_team"].astype(str) == sel_team]

        # Build human-readable match labels
        match_labels = build_match_labels(mf)
        match_ids = sorted(match_labels.keys())

        if match_ids:
            label_to_id = {v: k for k, v in match_labels.items()}
            display_labels = [match_labels[mid] for mid in match_ids]
            sel_label = st.selectbox("Select Match", display_labels, key="match_label_sel")
            sel_mid = label_to_id[sel_label]

            m = mf[mf["match_id"] == sel_mid].sort_values("rank").head(22)

            # Match header
            match_venue = m["venue"].iloc[0] if "venue" in m.columns else ""
            match_season = m["season"].iloc[0] if "season" in m.columns else ""
            teams_in_match = m["player_team"].unique().tolist()
            st.markdown(f"**📍 {match_venue}** &nbsp;|&nbsp; **Season: {match_season}** "
                        f"&nbsp;|&nbsp; **Teams: {', '.join(teams_in_match)}**")

            fig = px.bar(m.head(11), x="performance_score", y="player", orientation="h",
                         color="role", color_discrete_sequence=PAL,
                         title=f"Top Performers",
                         text="key_stat", hover_data=["player_team", "opposition"])
            fig.update_layout(**PLOTLY_LAYOUT, height=360,
                              yaxis=dict(autorange="reversed"),
                              xaxis_title="Performance Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            show_match_cols = [c for c in ["player", "role", "player_team", "key_stat",
                                           "performance_score", "impact_score", "impact_category",
                                           "wpa_score", "clutch_score", "rank"]
                               if c in m.columns]
            st.dataframe(m[show_match_cols], use_container_width=True, height=400, hide_index=True)
        else:
            st.caption("No matches for the selected filters.")
