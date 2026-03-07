"""
Tensor Titans — CricImpact IPL T20 Player Impact Dashboard
Dual-theme (light/dark), responsive, cricket-domain aesthetic with glassmorphism UI.
"""
import os
import json
import tempfile
import html as html_lib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from match_impact_from_files import compute_match_rankings_from_files

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
DARK = dict(
    BG="#0a0e14", SURFACE="#141a24", CARD="rgba(20,26,36,0.88)",
    GLASS="rgba(255,255,255,0.035)", BORDER="rgba(255,255,255,0.08)",
    TEXT="#e6edf3", TEXT2="#8b949e",
    G1="#0a0e14", G2="#111927",
    PGRID="rgba(255,255,255,0.04)", PZERO="rgba(255,255,255,0.06)",
    PBG="rgba(0,0,0,0)", SB_T="#0a0e14", SB_B="#141a24",
    GLOW="rgba(59,130,246,0.15)", LINE_CON="rgba(255,255,255,0.15)",
)
LIGHT = dict(
    BG="#f4f6f9", SURFACE="#ffffff", CARD="rgba(255,255,255,0.92)",
    GLASS="rgba(0,0,0,0.025)", BORDER="rgba(0,0,0,0.09)",
    TEXT="#1a1a2e", TEXT2="#555770",
    G1="#f4f6f9", G2="#e8ecf2",
    PGRID="rgba(0,0,0,0.06)", PZERO="rgba(0,0,0,0.08)",
    PBG="rgba(0,0,0,0)", SB_T="#ffffff", SB_B="#f0f2f5",
    GLOW="rgba(30,80,160,0.10)", LINE_CON="rgba(0,0,0,0.12)",
)

CYAN="#3b82f6"; ORANGE="#f59e0b"; GREEN="#22c55e"; PURPLE="#a855f7"
PINK="#ec4899"; RED="#ef4444"; GOLD="#eab308"; TEAL="#14b8a6"; CRIMSON="#dc2626"

def rgba(hex_color, alpha):
    """Convert hex color + alpha (0-1) to rgba() string for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def _toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def TH():
    return DARK if st.session_state.theme == "dark" else LIGHT

def FL():
    t = TH()
    return dict(
        paper_bgcolor=t["PBG"], plot_bgcolor=t["PBG"],
        font=dict(family="'Inter','Segoe UI',system-ui,sans-serif", color=t["TEXT"], size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=t["SURFACE"], font_size=12, font_color=t["TEXT"],
                        bordercolor=t["BORDER"]),
        transition=dict(duration=700, easing="cubic-in-out"),
    )

def FG():
    t = TH()
    return dict(
        xaxis=dict(gridcolor=t["PGRID"], zerolinecolor=t["PZERO"]),
        yaxis=dict(gridcolor=t["PGRID"], zerolinecolor=t["PZERO"]),
    )

PCFG = dict(scrollZoom=False, displayModeBar="hover",
            modeBarButtonsToRemove=["lasso2d","select2d"], displaylogo=False)


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def clipped_mean(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr).mean()

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

def role_icon(r):
    return {"Batter": "\U0001F3CF", "Bowler": "\U0001F3B3", "All-rounder": "\u2B50"}.get(r, "\U0001F3CF")

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
        lbl = f"{ts[0]}  vs  {ts[1]}" if len(ts) >= 2 else f"{ts[0]}  vs  {g['opposition'].iloc[0] if 'opposition' in g.columns else '?'}"
        if v: lbl += f" | {v}"
        if s: lbl += f" ({s})"
        out[mid] = lbl
    return out

def kpi_card(label, value, color=CYAN):
    t = TH()
    return f"""
<div class="kpi-card" style="
    background:{t['CARD']};backdrop-filter:blur(16px) saturate(180%);
    border:1px solid {t['BORDER']};border-radius:16px;padding:20px 16px;
    text-align:center;position:relative;overflow:hidden;
    box-shadow:0 4px 24px rgba(0,0,0,0.12);
">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,{color}00,{color},{color}00);"></div>
  <div style="font-size:0.68rem;color:{t['TEXT2']};letter-spacing:1.5px;
              text-transform:uppercase;margin-bottom:8px;font-weight:600;">{label}</div>
  <div style="font-size:1.7rem;font-weight:800;color:{color};
              text-shadow:0 0 20px {color}35;">{value}</div>
</div>"""

def stat_row(label, value, color=CYAN):
    t = TH()
    return f"""
<div class="stat-row" style="display:flex;align-items:center;gap:10px;padding:9px 14px;
            background:{t['GLASS']};border-radius:10px;margin-bottom:6px;
            border-left:3px solid {color};transition:all 0.3s cubic-bezier(.34,1.56,.64,1);">
  <span style="flex:1;font-size:0.82rem;color:{t['TEXT2']};font-weight:500;">{label}</span>
  <span style="font-weight:700;color:{t['TEXT']};font-size:0.95rem;">{value}</span>
</div>"""


def section_header(title, icon=""):
    t = TH()
    return f"""
<div style="display:flex;align-items:center;gap:10px;margin:24px 0 14px;">
  <span style="font-size:1.3rem;">{icon}</span>
  <h3 style="margin:0;color:{t['TEXT']};font-weight:700;font-size:1.15rem;
             letter-spacing:-0.3px;">{title}</h3>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,{t['BORDER']},{t['BORDER']}00);
              margin-left:8px;"></div>
</div>"""


def get_groq_key():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def groq_generate(prompt, api_key, model=None):
    if model is None:
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    try:
        from groq import Groq
    except Exception as exc:
        return None, f"Missing dependency 'groq': {exc}"

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an IPL analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_completion_tokens=1200,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content, None
    except Exception as exc:
        return None, str(exc)


def parse_explanations(text):
    rows = []
    current_team = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "|" not in line:
            current_team = line.replace("**", "").strip()
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            rows.append({
                "Team": (current_team or "").replace("**", "").strip(),
                "Player": parts[0],
                "Role": parts[1],
                "Key Stat": parts[2],
                "Impact Explanation": " | ".join(parts[3:]),
            })
    return pd.DataFrame(rows)


def render_team_table(team_name, team_df):
    cols = ["No.", "Player", "Role", "Key Stat", "Impact Explanation"]
    head = "".join(f"<th>{html_lib.escape(c)}</th>" for c in cols)
    rows = []
    for _, row in team_df.iterrows():
        cells = []
        for col in cols:
            val = "" if pd.isna(row[col]) else str(row[col])
            cells.append(f"<td>{html_lib.escape(val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows)
    return (
        f"<div class='impact-table-wrap'>"
        f"<div class='impact-team'>{html_lib.escape(team_name)}</div>"
        f"<table class='impact-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        f"</table></div>"
    )


# --- Page setup ---
st.set_page_config(page_title="CricImpact - IPL T20", layout="wide", page_icon="C")

t = TH()
is_dark = st.session_state.theme == "dark"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&display=swap');

  :root {{
    --bg:{t['BG']}; --surface:{t['SURFACE']}; --card:{t['CARD']};
    --text:{t['TEXT']}; --text2:{t['TEXT2']}; --border:{t['BORDER']};
    --glass:{t['GLASS']}; --glow:{t['GLOW']};
  }}

  .stApp {{
    background:linear-gradient(160deg,{t['G1']} 0%,{t['G2']} 50%,{t['G1']} 100%);
    font-family:'Inter','Segoe UI',system-ui,sans-serif;
  }}
  .block-container {{ padding-top:.6rem; max-width:1420px; }}
  html, body, .stApp, .stApp > div {{ color:{t['TEXT']} !important; }}
  h1,h2,h3,h4 {{ color:{t['TEXT']} !important; font-family:'Outfit','Inter',sans-serif; }}
  p, span, label, .stMarkdown {{ color:{t['TEXT']}; }}

  /* Cricket pitch stripe */
  .stApp::before {{
    content:''; position:fixed; top:0; left:50%; width:100px; height:100vh;
    transform:translateX(-50%);
    background:linear-gradient(180deg,
      {'rgba(34,197,94,0.03)' if is_dark else 'rgba(34,197,94,0.045)'} 0%,
      {'rgba(34,197,94,0.008)' if is_dark else 'rgba(34,197,94,0.015)'} 100%);
    pointer-events:none; z-index:0;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{t['SB_T']} 0%,{t['SB_B']} 100%) !important;
    border-right:1px solid {t['BORDER']};
  }}
  section[data-testid="stSidebar"] * {{ color:{t['TEXT']} !important; }}
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] .stNumberInput > div > div > input,
  section[data-testid="stSidebar"] [data-baseweb="select"],
  section[data-testid="stSidebar"] [data-baseweb="input"] {{
    background-color:{t['SURFACE']} !important;
    color:{t['TEXT']} !important;
    border-color:{t['BORDER']} !important;
    border-radius:10px !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color:{t['TEXT']} !important; }}
  section[data-testid="stSidebar"] [data-baseweb="input"] input {{ color:{t['TEXT']} !important; }}
  section[data-testid="stSidebar"] .stRadio label span {{ color:{t['TEXT']} !important; }}
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ color:{t['TEXT']} !important; }}
  section[data-testid="stSidebar"] button {{
    background:linear-gradient(135deg,{CYAN}18,{PURPLE}18) !important;
    color:{t['TEXT']} !important;
    border:1px solid {CYAN}30 !important; border-radius:12px !important;
    font-weight:600 !important; letter-spacing:.3px;
    transition:all .35s cubic-bezier(.34,1.56,.64,1) !important;
    padding:10px 16px !important;
  }}
  section[data-testid="stSidebar"] button:hover {{
    background:linear-gradient(135deg,{CYAN}30,{PURPLE}30) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 4px 20px {CYAN}25 !important;
  }}
  section[data-testid="stSidebar"] button:active {{
    transform:scale(0.97) !important;
  }}

  /* Sidebar collapse/expand button */
  button[data-testid="stSidebarCollapseButton"],
  button[data-testid="stSidebarExpandButton"] {{
    background:{t['CARD']} !important;
    color:{t['TEXT']} !important;
    border:1px solid {t['BORDER']} !important;
    border-radius:10px !important;
    transition:all .3s ease !important;
    z-index:999 !important;
  }}
  button[data-testid="stSidebarCollapseButton"]:hover,
  button[data-testid="stSidebarExpandButton"]:hover {{
    background:{CYAN}20 !important;
    border-color:{CYAN}50 !important;
    box-shadow:0 2px 12px {CYAN}20 !important;
  }}

  /* Metric cards */
  [data-testid="stMetric"] {{
    background:{t['CARD']}; backdrop-filter:blur(16px) saturate(180%);
    border:1px solid {t['BORDER']}; border-radius:14px;
    padding:14px 16px; box-shadow:0 4px 20px rgba(0,0,0,0.12);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),box-shadow .35s ease;
  }}
  [data-testid="stMetric"]:hover {{
    transform:translateY(-3px); box-shadow:0 8px 28px {t['GLOW']};
  }}
  [data-testid="stMetricValue"] {{ font-size:1.3rem;color:{CYAN};font-weight:700; }}
  [data-testid="stMetricLabel"] {{ color:{t['TEXT2']};font-size:.72rem;text-transform:uppercase;letter-spacing:.8px; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap:8px; border-bottom:2px solid {t['BORDER']}; }}
  .stTabs [data-baseweb="tab"] {{
    background:{t['CARD']}; border-radius:12px 12px 0 0;
    padding:11px 24px; font-weight:600; color:{t['TEXT2']};
    border:1px solid {t['BORDER']}; border-bottom:none;
    transition:all .35s cubic-bezier(.34,1.56,.64,1);
    font-family:'Outfit','Inter',sans-serif; letter-spacing:-0.2px;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color:{t['TEXT']}; background:{t['GLASS']};
    transform:translateY(-2px); box-shadow:0 4px 16px {t['GLOW']};
  }}
  .stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,{CYAN}18,{PURPLE}18) !important;
    color:{CYAN} !important; border-color:{CYAN}40 !important;
    box-shadow:0 0 20px {CYAN}12,inset 0 -2px 0 {CYAN};
  }}

  /* Glass panels */
  .glass {{
    background:{t['CARD']}; backdrop-filter:blur(16px) saturate(180%);
    border:1px solid {t['BORDER']}; border-radius:16px;
    padding:22px; margin-bottom:14px;
    box-shadow:0 4px 30px rgba(0,0,0,0.12);
    transition:transform .35s ease,box-shadow .35s ease;
  }}
  .glass:hover {{ transform:translateY(-2px); box-shadow:0 8px 36px {t['GLOW']}; }}
  [data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; }}

  /* Hero */
  .hero {{ text-align:center; padding:18px 0 8px; position:relative; }}
  .hero h1 {{
    font-size:2.8rem; font-weight:900; letter-spacing:-1.5px;
    font-family:'Outfit','Inter',sans-serif;
    background:linear-gradient(135deg,{CYAN},{PURPLE},{PINK},{ORANGE});
    background-size:300% 300%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:gradShift 6s ease infinite; margin-bottom:2px;
    line-height:1.1;
  }}
  .hero .team-badge {{
    display:inline-flex; align-items:center; gap:8px;
    background:linear-gradient(135deg,{CYAN}15,{PURPLE}15);
    border:1px solid {CYAN}30; border-radius:40px;
    padding:7px 22px; margin-top:8px;
    font-size:.78rem; font-weight:700; color:{CYAN};
    letter-spacing:1.2px; text-transform:uppercase;
    backdrop-filter:blur(8px);
    animation:pulseGlow 4s ease-in-out infinite;
  }}
  .hero p {{ color:{t['TEXT2']};font-size:.88rem;margin-top:6px;font-weight:500;letter-spacing:.3px; }}
  .hero .sub-line {{
    display:flex; align-items:center; justify-content:center; gap:12px;
    margin-top:4px;
  }}
  .hero .sub-dot {{
    width:4px; height:4px; border-radius:50%; background:{CYAN};
    animation:pulseGlow 3s ease-in-out infinite;
  }}

  .sep {{
    border:none; margin:20px 0; height:1px;
    background:linear-gradient(90deg,{t['BORDER']}00,{t['BORDER']},{t['BORDER']}00);
  }}
  #MainMenu {{ visibility:hidden; }}
  footer {{ visibility:hidden; }}

  /* Keep sidebar toggle button visible */
  header[data-testid="stHeader"] {{
    background:transparent !important;
    backdrop-filter:none !important;
  }}
  button[kind="header"] {{
    visibility:visible !important;
  }}

  /* Fix Streamlit default backgrounds for proper theming */
  .stApp > div > div {{ background:transparent !important; }}
  .element-container {{ color:{t['TEXT']}; }}

  /* COMPREHENSIVE WIDGET THEMING — force all form elements to match our theme */
  /* Labels */
  .stSelectbox label, .stNumberInput label, .stSlider label,
  .stRadio label, .stTextInput label, .stFileUploader label,
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] * {{ color:{t['TEXT']} !important; }}

  /* Select boxes — everywhere */
  [data-baseweb="select"],
  [data-baseweb="select"] > div,
  [data-baseweb="select"] > div > div,
  [data-baseweb="select"] [class*="control"],
  [data-baseweb="select"] [class*="ValueContainer"],
  [data-baseweb="select"] [class*="singleValue"],
  [data-baseweb="select"] [class*="placeholder"],
  .stSelectbox > div > div > div {{
    background-color:{t['SURFACE']} !important;
    color:{t['TEXT']} !important;
    border-color:{t['BORDER']} !important;
    border-radius:10px !important;
  }}
  [data-baseweb="select"] * {{ color:{t['TEXT']} !important; }}
  [data-baseweb="select"] svg {{ fill:{t['TEXT2']} !important; }}
  [data-baseweb="select"] input {{ color:{t['TEXT']} !important; background:transparent !important; }}
  [data-baseweb="select"] [data-baseweb="tag"] {{ background-color:{t['GLASS']} !important; }}

  /* Popover / dropdown menus */
  [data-baseweb="popover"] > div {{ background-color:{t['SURFACE']} !important; border:1px solid {t['BORDER']} !important; border-radius:12px !important; }}
  [data-baseweb="menu"] {{ background-color:{t['SURFACE']} !important; }}
  [data-baseweb="menu"] li {{ color:{t['TEXT']} !important; }}
  [data-baseweb="menu"] li:hover {{ background:{t['GLASS']} !important; }}

  /* Text inputs */
  [data-baseweb="input"],
  [data-baseweb="input"] > div {{ background-color:{t['SURFACE']} !important; border-color:{t['BORDER']} !important; border-radius:10px !important; }}
  [data-baseweb="input"] input {{ color:{t['TEXT']} !important; background:transparent !important; }}

  /* Number inputs — cover all nesting levels */
  .stNumberInput > div > div > input,
  .stNumberInput input[type="number"],
  .stNumberInput [data-baseweb="input"] input,
  [data-testid="stNumberInput"] input {{ background-color:{t['SURFACE']} !important; color:{t['TEXT']} !important; border-color:{t['BORDER']} !important; border-radius:10px !important; }}
  .stNumberInput button,
  .stNumberInput [data-testid="stNumberInputStepUp"],
  .stNumberInput [data-testid="stNumberInputStepDown"],
  [data-testid="stNumberInput"] button {{ background-color:{t['GLASS']} !important; color:{t['TEXT']} !important; border-color:{t['BORDER']} !important; }}
  .stNumberInput > div > div,
  .stNumberInput > div,
  [data-testid="stNumberInput"] > div > div {{ background-color:{t['SURFACE']} !important; border-color:{t['BORDER']} !important; border-radius:10px !important; }}

  /* Text input */
  .stTextInput > div > div > input,
  .stTextInput > div > div {{ background-color:{t['SURFACE']} !important; color:{t['TEXT']} !important; border-color:{t['BORDER']} !important; border-radius:10px !important; }}

  /* Slider */
  .stSlider [data-baseweb="slider"] {{ background:transparent; }}
  .stSlider [data-testid="stThumbValue"] {{ color:{t['TEXT']} !important; }}
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"] {{ color:{t['TEXT2']} !important; }}

  /* Radio */
  .stRadio > div {{ color:{t['TEXT']} !important; }}
  .stRadio label span {{ color:{t['TEXT']} !important; }}

  /* Expander */
  div[data-testid="stExpander"] {{ background:{t['CARD']}; border:1px solid {t['BORDER']}; border-radius:14px; }}
  div[data-testid="stExpander"] summary {{ color:{t['TEXT']} !important; }}
  div[data-testid="stExpander"] summary span {{ color:{t['TEXT']} !important; }}

  /* Alerts (info, warning, error boxes) */
  .stAlert,
  [data-testid="stAlert"] {{ border-radius:12px; }}

  /* Main area buttons */
  .stButton > button {{
    background:{t['SURFACE']} !important;
    color:{t['TEXT']} !important;
    border:1px solid {t['BORDER']} !important;
    border-radius:10px !important;
    transition:all .3s ease !important;
  }}
  .stButton > button:hover {{
    background:{'rgba(59,130,246,0.12)' if is_dark else 'rgba(59,130,246,0.08)'} !important;
    border-color:{CYAN} !important;
    color:{CYAN} !important;
  }}

  /* File uploader */
  [data-testid="stFileUploader"] section {{
    background:{t['SURFACE']} !important;
    border:1px solid {t['BORDER']} !important;
    border-radius:12px !important;
  }}
  [data-testid="stFileUploader"] section * {{ color:{t['TEXT']} !important; }}
  [data-testid="stFileUploader"] button {{
    background:{t['SURFACE']} !important;
    color:{CYAN} !important;
    border:1px solid {t['BORDER']} !important;
  }}
  [data-testid="stFileUploadDropzone"] {{
    background:{t['SURFACE']} !important;
    border-color:{t['BORDER']} !important;
  }}
  [data-testid="stFileUploadDropzone"] * {{ color:{t['TEXT2']} !important; }}
  [data-testid="stFileUploadDropzone"] button {{ color:{CYAN} !important; }}
  [data-testid="stUploadedFile"] {{
    background:{t['GLASS']} !important;
    border:1px solid {t['BORDER']} !important;
    border-radius:8px !important;
  }}
  [data-testid="stUploadedFile"] * {{ color:{t['TEXT']} !important; }}

  /* Animations */
  @keyframes fadeSlideUp {{
    from {{ opacity:0;transform:translateY(28px); }}
    to   {{ opacity:1;transform:translateY(0); }}
  }}
  @keyframes fadeSlideLeft {{
    from {{ opacity:0;transform:translateX(30px); }}
    to   {{ opacity:1;transform:translateX(0); }}
  }}
  @keyframes scaleIn {{
    from {{ opacity:0;transform:scale(0.92); }}
    to   {{ opacity:1;transform:scale(1); }}
  }}
  @keyframes gradShift {{
    0%   {{ background-position:0% 50%; }}
    50%  {{ background-position:100% 50%; }}
    100% {{ background-position:0% 50%; }}
  }}
  @keyframes pulseGlow {{
    0%,100% {{ box-shadow:0 0 8px {CYAN}20; }}
    50%     {{ box-shadow:0 0 24px {CYAN}35; }}
  }}
  @keyframes shimmer {{
    0%   {{ background-position:-200% 0; }}
    100% {{ background-position:200% 0; }}
  }}
  @keyframes float {{
    0%,100% {{ transform:translateY(0); }}
    50%     {{ transform:translateY(-6px); }}
  }}
  @keyframes cricketBounce {{
    0%,100% {{ transform:translateY(0) rotate(0deg); }}
    25%     {{ transform:translateY(-10px) rotate(90deg); }}
    50%     {{ transform:translateY(0) rotate(180deg); }}
    75%     {{ transform:translateY(-5px) rotate(270deg); }}
  }}

  [data-testid="stPlotlyChart"] {{
    animation:fadeSlideUp .7s cubic-bezier(.16,1,.3,1) both;
    transition:transform .4s cubic-bezier(.34,1.56,.64,1),box-shadow .4s ease;
    border-radius:14px;
  }}
  [data-testid="stPlotlyChart"]:hover {{
    transform:scale(1.012) translateY(-2px);
    box-shadow:0 12px 40px {t['GLOW']};
  }}
  [data-testid="stDataFrame"] {{
    animation:fadeSlideUp .65s cubic-bezier(.16,1,.3,1) both;
    transition:transform .3s ease;
  }}
  [data-testid="stDataFrame"]:hover {{ transform:scale(1.005); }}

  /* Dataframe / table — theme-aware */
  [data-testid="stDataFrame"] > div,
  [data-testid="stDataFrame"] > div > div {{
    background:{t['SURFACE']} !important;
    border-radius:14px !important;
    border:1px solid {t['BORDER']} !important;
  }}
  /* Glide Data Grid internal CSS variable overrides */
  [data-testid="stDataFrame"] {{
    --gdg-bg-cell:{t['SURFACE']};
    --gdg-bg-cell-medium:{'#1a2233' if is_dark else '#f0f2f5'};
    --gdg-bg-header:{'#1a2233' if is_dark else '#edf0f5'};
    --gdg-bg-header-has-focus:{'#1e2a3a' if is_dark else '#e4e8f0'};
    --gdg-bg-header-hovered:{'#1e2a3a' if is_dark else '#e8ecf2'};
    --gdg-text-dark:{t['TEXT']};
    --gdg-text-medium:{t['TEXT2']};
    --gdg-text-light:{t['TEXT2']};
    --gdg-text-header:{t['TEXT']};
    --gdg-border-color:{t['BORDER']};
    --gdg-bg-bubble:{'#1e2a3a' if is_dark else '#edf0f5'};
    --gdg-bg-bubble-selected:{'#253348' if is_dark else '#dde4f0'};
    --gdg-link-color:{CYAN};
    --gdg-accent-color:{CYAN};
    --gdg-accent-light:{'rgba(59,130,246,0.15)' if is_dark else 'rgba(59,130,246,0.12)'};
  }}
  [data-testid="stTable"] th,
  .stDataFrame thead th {{
    background-color:{'#1a2233' if is_dark else '#edf0f5'} !important;
    color:{t['TEXT']} !important;
    padding:12px 16px !important;
    font-weight:600 !important;
  }}
  [data-testid="stTable"] td,
  .stDataFrame tbody td {{
    background-color:{t['SURFACE']} !important;
    color:{t['TEXT']} !important;
    border-color:{t['BORDER']} !important;
    padding:10px 16px !important;
  }}
  /* Streamlit internal theme CSS variables */
  .stApp {{
    --primary-color:{CYAN};
    --background-color:{t['BG']};
    --secondary-background-color:{t['SURFACE']};
    --text-color:{t['TEXT']};
    --font:'Inter','Segoe UI',system-ui,sans-serif;
  }}
  .themed-table {{ width:100%;border-collapse:collapse;font-size:.88rem; }}
  .themed-table th {{
    text-align:left;padding:14px 16px;
    color:{CYAN};font-weight:700;font-size:.78rem;
    text-transform:uppercase;letter-spacing:.8px;
    border-bottom:2px solid {t['BORDER']};
    background:{t['GLASS']};
  }}
  .themed-table td {{
    padding:12px 16px;
    color:{t['TEXT']};
    border-bottom:1px solid {t['BORDER']};
  }}
  .themed-table tr:hover {{
    background:{t['GLASS']};
  }}

  .kpi-card {{
    animation:scaleIn .5s cubic-bezier(.16,1,.3,1) both;
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),box-shadow .35s ease;
  }}
  .kpi-card:hover {{
      transform: translateY(-4px) scale(1.03);
      box-shadow: 0 8px 24px rgba(88,166,255,0.2);
  }}
  .glass {{
      animation: fadeSlideUp 0.6s ease-out both;
  }}
    .impact-table-wrap {{ margin-bottom: 22px; }}
    .impact-team {{ font-weight: 700; font-size: 1.05rem; margin: 6px 0 10px; color: {t['TEXT']}; }}
    .impact-table {{ width: 100%; border-collapse: collapse; background: {t['CARD']}; border: 1px solid {t['BORDER']}; }}
    .impact-table th, .impact-table td {{
            padding: 10px 12px; border-bottom: 1px solid {t['BORDER']}; text-align: left; vertical-align: top;
    }}
    .impact-table th {{ background: {t['SURFACE']}; font-weight: 600; color: {t['TEXT']}; }}
    .impact-table td {{ white-space: normal; word-break: break-word; color: {t['TEXT']}; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
pdf_raw = load(P_FILE)
mdf     = load(M_FILE)
bat_df  = load(BAT_FILE)
bwl_df  = load(BWL_FILE)
riv_df  = load(RIV_FILE)

if pdf_raw is None:
    st.error("\U0001F3CF Run the pipeline first — data not found.")
    st.stop()


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
# Theme toggle — uses on_click callback so state updates before rerun
theme_icon = '🌙' if is_dark else '☀️'
theme_label = 'Switch to Light' if is_dark else 'Switch to Dark'
st.sidebar.button(
    f"{theme_icon}  {theme_label}",
    on_click=_toggle_theme,
    use_container_width=True,
    type="secondary",
)

st.sidebar.markdown(f"""
<div style="text-align:center;margin:10px 0 18px;">
  <div class="cricket-stumps">
    <div class="bail"></div>
  </div>
  <div class="cricket-stumps">
    <div class="stump"></div>
    <div class="stump"></div>
    <div class="stump"></div>
  </div>
  <div style="font-size:.72rem;color:{t['TEXT2']};letter-spacing:1.5px;
              text-transform:uppercase;font-weight:700;margin-top:6px;">
    Tensor Titans Analytics
  </div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown(f"### \U0001F3CF Filters")
min_inn = st.sidebar.slider("Min innings played", 0, 100, 10)
role_pick = st.sidebar.radio("Role", ["All", "Batter", "Bowler", "All-rounder"], horizontal=True)

all_seasons = []
if mdf is not None:
    all_seasons = sorted(mdf["season"].dropna().astype(str).unique().tolist())
sb_season = st.sidebar.selectbox("Season", ["All"] + all_seasons, key="sb_season")
min_runs = st.sidebar.number_input("Min career runs", 0, 10000, 0, step=100)
min_wkts = st.sidebar.number_input("Min career wickets", 0, 500, 0, step=5)


# ═══════════════════════════════════════════════════════════
# BUILD PLAYER DATAFRAME
# ═══════════════════════════════════════════════════════════
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
    pdf = pdf[pdf["player"].isin(season_players["player"])].copy()
    pdf = pdf.merge(season_players, on="player", how="left")
else:
    pdf = pdf_raw.copy()
    pdf["role"] = pdf.apply(role_of, axis=1)
    pdf["total_innings"] = pdf["bat_innings"] + pdf["bowl_innings"]

pdf["total_runs"] = pdf["total_runs"].fillna(0)
pdf["total_wickets"] = pdf["total_wickets"].fillna(0)
pdf = pdf[pdf["total_innings"] >= min_inn]
if min_runs > 0:  pdf = pdf[pdf["total_runs"]    >= min_runs]
if min_wkts > 0:  pdf = pdf[pdf["total_wickets"]  >= min_wkts]

if len(pdf) == 0:
    st.warning("No players match the current filters. Adjust sidebar filters.")
    st.stop()

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
st.sidebar.markdown(f"""
<div style="margin-top:16px;padding:10px 14px;background:{t['CARD']};border-radius:12px;
            border:1px solid {t['BORDER']};text-align:center;">
  <span style="font-size:.82rem;color:{t['TEXT']};font-weight:600;">
    {len(pdf)} players{season_tag}
  </span>
  <br><span style="font-size:.7rem;color:{t['TEXT2']};">Ranked by {imp_label}</span>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>\U0001F3CF CricImpact</h1>
  <div class="team-badge">
    <span style="font-size:1rem;">⚡</span> Tensor Titans
  </div>
  <div class="sub-line">
    <div class="sub-dot"></div>
    <p>IPL T20 Player Impact Analytics Dashboard</p>
    <div class="sub-dot"></div>
  </div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Player Profile", "Head-to-Head", "Match Explorer", "Match Impact Upload"])


# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tabs[0]:
    top = pdf.iloc[0] if len(pdf) > 0 else None
    cols = st.columns(4)
    with cols[0]: st.markdown(kpi_card("\U0001F465 Players", len(pdf), CYAN), unsafe_allow_html=True)
    with cols[1]: st.markdown(kpi_card(f"\U0001F4C8 Mean {imp_label}", f"{clipped_mean(pdf['display_impact']):.1f}", GOLD), unsafe_allow_html=True)
    with cols[2]: st.markdown(kpi_card("\U0001F3C6 Top Player", top["player"] if top is not None else "-", GREEN), unsafe_allow_html=True)
    with cols[3]: st.markdown(kpi_card(f"\U0001F31F Top {imp_label}", f"{top['display_impact']:.1f}" if top is not None else "-", ORANGE), unsafe_allow_html=True)

    # Season highlights
    if season_mode:
        sm = mdf[mdf["season"].astype(str) == sb_season]
        if len(sm) > 0:
            st.markdown('<hr class="sep">', unsafe_allow_html=True)
            st.markdown(section_header(f"Top Performers \u2014 {sb_season}", "\U0001F525"), unsafe_allow_html=True)
            stop = sm.groupby("player")["impact_score"].mean().nlargest(5).reset_index()
            stop.columns = ["Player", "Avg Impact"]
            cs = st.columns(5)
            for i, (_, r) in enumerate(stop.iterrows()):
                with cs[i]:
                    st.markdown(kpi_card(r["Player"], f"{r['Avg Impact']:.1f}",
                                         [CYAN,ORANGE,GREEN,PURPLE,PINK][i]), unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Top 15 + Distribution
    c1, c2 = st.columns(2)
    with c1:
        t15 = pdf.head(15).iloc[::-1]
        colors = [f"rgba(59,130,246,{0.35 + 0.65*i/14})" for i in range(15)]
        fig = go.Figure(go.Bar(
            x=t15["display_impact"], y=t15["player"], orientation="h",
            marker=dict(color=colors, line=dict(width=0),
                        cornerradius=6),
            text=t15["display_impact"].round(1), textposition="outside",
            textfont=dict(size=11, color=CYAN, family="Inter"),
        ))
        fig.update_layout(**FL(), **FG(), height=440,
                          title=dict(text=f"\U0001F3CF Top 15 \u2014 {imp_label}{season_tag}",
                                     font=dict(size=15, color=t['TEXT'])),
                          xaxis_title=imp_label)
        fig.update_xaxes(showgrid=True, gridcolor=t["PGRID"])
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    with c2:
        fig = go.Figure(go.Histogram(
            x=pdf["display_impact"], nbinsx=35,
            marker=dict(color=CYAN, line=dict(width=0.5, color=t["BORDER"])),
            opacity=0.85,
        ))
        cmean = clipped_mean(pdf["display_impact"])
        fig.add_vline(x=cmean, line_dash="dash", line_color=GOLD, line_width=2,
                      annotation_text=f"Mean: {cmean:.1f}", annotation_font_color=GOLD)
        fig.update_layout(**FL(), **FG(), height=440,
                          title=dict(text=f"\U0001F4CA {imp_label} Distribution{season_tag}",
                                     font=dict(size=15, color=t['TEXT'])),
                          xaxis_title=imp_label, yaxis_title="Players", bargap=0.08)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    c3, c4 = st.columns(2)
    with c3:
        rc = pdf["role"].value_counts().reset_index()
        rc.columns = ["Role", "Count"]
        fig = go.Figure(go.Pie(
            labels=rc["Role"], values=rc["Count"], hole=0.58,
            marker=dict(colors=[CYAN, ORANGE, GREEN, PURPLE][:len(rc)],
                        line=dict(color=t["BG"], width=2.5)),
            textfont=dict(color=t["TEXT"], size=12),
            hoverinfo="label+percent+value",
        ))
        fig.update_layout(**FL(), height=380,
                          title=dict(text="\U0001F3AF Role Distribution",
                                     font=dict(size=15, color=t['TEXT'])),
                          showlegend=True,
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t['TEXT2'])))
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    with c4:
        ar = pdf[(pdf["mean_bat_impact"] > 0) & (pdf["mean_bowl_impact"] > 0)]
        if len(ar) > 0:
            scatter_scale = [[0, t["BG"]], [0.4, "#1e40af"], [1, CYAN]]
            fig = px.scatter(ar, x="mean_bat_impact", y="mean_bowl_impact",
                             color="display_impact", size="total_innings",
                             hover_name="player",
                             color_continuous_scale=scatter_scale)
            fig.update_traces(marker=dict(line=dict(width=0.5, color=t["LINE_CON"])))
            fig.update_layout(**FL(), **FG(), height=380,
                              title=dict(text="\U0001F3CF Batting vs Bowling Impact",
                                         font=dict(size=15, color=t['TEXT'])),
                              xaxis_title="Batting Impact", yaxis_title="Bowling Impact",
                              coloraxis_colorbar=dict(title=imp_label,
                                                      tickfont=dict(color=t['TEXT2'])))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.markdown(section_header("Leaderboards", "\U0001F451"), unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        tb = pdf[pdf["bat_innings"] >= 20].nlargest(10, "mean_bat_impact")
        fig = go.Figure(go.Bar(
            x=tb["mean_bat_impact"], y=tb["player"], orientation="h",
            marker=dict(color=tb["mean_bat_impact"],
                        colorscale=[[0, t["SURFACE"]], [0.4, "#92400e"], [1, ORANGE]],
                        line=dict(width=0), cornerradius=5),
            text=tb["mean_bat_impact"].round(3), textposition="outside",
            textfont=dict(color=ORANGE, size=10),
        ))
        fig.update_layout(**FL(), height=370,
                          title=dict(text="\U0001F3CF Top 10 Batters",
                                     font=dict(size=14, color=t['TEXT'])))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    with c6:
        tw = pdf[pdf["bowl_innings"] >= 20].nlargest(10, "mean_bowl_impact")
        fig = go.Figure(go.Bar(
            x=tw["mean_bowl_impact"], y=tw["player"], orientation="h",
            marker=dict(color=tw["mean_bowl_impact"],
                        colorscale=[[0, t["SURFACE"]], [0.4, "#581c87"], [1, PURPLE]],
                        line=dict(width=0), cornerradius=5),
            text=tw["mean_bowl_impact"].round(3), textposition="outside",
            textfont=dict(color=PURPLE, size=10),
        ))
        fig.update_layout(**FL(), height=370,
                          title=dict(text="\U0001F3B3 Top 10 Bowlers",
                                     font=dict(size=14, color=t['TEXT'])))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # Full table
    with st.expander("\U0001F4CB Full Player Table"):
        srch = st.text_input("Search player", "", key="tbl_search")
        tcols = [c for c in ["player","role","display_impact","bat_innings","total_runs",
                              "avg_bat_sr","true_bat_avg","bowl_innings","total_wickets",
                              "avg_economy","pom_awards","win_rate"] if c in pdf.columns]
        td = pdf[tcols].copy()
        for _ic in ["bat_innings","total_runs","bowl_innings","total_wickets","pom_awards"]:
            if _ic in td.columns:
                td[_ic] = td[_ic].fillna(0).astype(int)
        td = td.rename(columns={"display_impact": imp_label})
        if srch:
            td = td[td["player"].str.contains(srch, case=False, na=False)]
        _col_cfg = {
            imp_label: st.column_config.ProgressColumn(
                imp_label, min_value=0, max_value=100, format="%.1f"),
            "avg_bat_sr": st.column_config.NumberColumn("avg_bat_sr", format="%.1f"),
            "true_bat_avg": st.column_config.NumberColumn("true_bat_avg", format="%.1f"),
            "avg_economy": st.column_config.NumberColumn("avg_economy", format="%.2f"),
            "win_rate": st.column_config.NumberColumn("win_rate", format="%.2f"),
        }
        st.dataframe(td, column_config=_col_cfg,
                     use_container_width=True, height=440)


# ════════════════════════════════════════════════════════
# TAB 2 — PLAYER PROFILE
# ════════════════════════════════════════════════════════
with tabs[1]:
    pc1, pc2 = st.columns([3, 1])
    with pc1:
        sel = st.selectbox("\U0001F50D Search player", pdf["player"].tolist(), key="prof_sel")
    with pc2:
        sel_season = st.selectbox("Season filter", ["All"] + all_seasons, key="prof_season")

    if sel:
        p = pdf[pdf["player"] == sel].iloc[0]
        rank_idx = pdf.index[pdf["player"] == sel][0] + 1

        if sel_season != "All" and mdf is not None:
            sm = mdf[(mdf["player"] == sel) & (mdf["season"].astype(str) == sel_season)]
            prof_impact = sm["impact_score"].mean() if len(sm) > 0 else p["overall_impact"]
            prof_label = f"Impact ({sel_season})"
        else:
            prof_impact = p["overall_impact"]
            prof_label = "Impact"

        # Player name header with role icon
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
          <span style="font-size:2rem;">{role_icon(p['role'])}</span>
          <div>
            <span style="font-size:1.5rem;font-weight:800;color:{t['TEXT']};
                         font-family:'Outfit','Inter',sans-serif;letter-spacing:-0.5px;">
              {sel}
            </span>
            <span style="margin-left:10px;font-size:.75rem;padding:3px 10px;
                         background:{PURPLE}20;color:{PURPLE};border-radius:20px;
                         font-weight:600;letter-spacing:.5px;">{p['role']}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Hero KPIs
        kcols = st.columns(6)
        kpi_data = [
            ("\U0001F3C5 Rank", f"#{rank_idx}", GOLD),
            (f"\U0001F4CA {prof_label}", f"{prof_impact:.1f}", CYAN),
            ("\U0001F3AF Role", p["role"], PURPLE),
            ("\U0001F3CF Bat Inn", si(p["bat_innings"]), ORANGE),
            ("\U0001F3B3 Bowl Inn", si(p["bowl_innings"]), GREEN),
            ("\U0001F3C6 POM", si(p["pom_awards"]), PINK),
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
                number=dict(font=dict(size=44, color=g_color, family="Outfit")),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor=t["TEXT"],
                              tickfont=dict(size=11, color=t["TEXT"])),
                    bar=dict(color=g_color, thickness=0.7),
                    bgcolor=t["SURFACE"], bordercolor=t["BORDER"],
                    steps=[
                        dict(range=[0,40],  color=rgba(RED, 0.07)),
                        dict(range=[40,60], color=rgba(GOLD, 0.07)),
                        dict(range=[60,80], color=rgba(GREEN, 0.07)),
                        dict(range=[80,100],color=rgba(CYAN, 0.07)),
                    ],
                    threshold=dict(line=dict(color=PINK, width=3), value=score),
                ),
                title=dict(text="\U0001F3AF Impact Meter",
                           font=dict(size=14, color=t["TEXT"])),
            ))
            gauge_layout = {k: v for k, v in FL().items() if k != "margin"}
            fig.update_layout(**gauge_layout, height=290,
                              margin=dict(l=30, r=30, t=65, b=20))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)

        with bat_col:
            st.markdown(section_header("Batting", "\U0001F3CF"), unsafe_allow_html=True)
            bstats = [
                ("Runs", si(p.get("total_runs")), ORANGE),
                ("Average", f"{p.get('true_bat_avg',0):.1f}", GOLD),
                ("Strike Rate", f"{p.get('avg_bat_sr',0):.1f}", CYAN),
                ("4s / 6s", f"{si(p.get('total_fours'))} / {si(p.get('total_sixes'))}", GREEN),
                ("Dot %", f"{p.get('dot_pct_faced',0):.1f}%", t['TEXT2']),
                ("Clutch WPA", f"{p.get('total_clutch_bat',0):.3f}", PINK),
            ]
            for l, v, co in bstats:
                st.markdown(stat_row(l, v, co), unsafe_allow_html=True)

        with bwl_col:
            st.markdown(section_header("Bowling", "\U0001F3B3"), unsafe_allow_html=True)
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
                fill="toself", fillcolor=rgba(CYAN, 0.09),
                line=dict(color=CYAN, width=2.5),
                marker=dict(size=7, color=CYAN, line=dict(width=1.5, color=t["BG"])),
            ))
            fig.update_layout(**FL(), height=380,
                              polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(visible=True, range=[0,1],
                                                         gridcolor=t["PGRID"],
                                                         tickfont=dict(size=9, color=t['TEXT2'])),
                                         angularaxis=dict(tickfont=dict(size=11, color=t['TEXT']),
                                                          gridcolor=t["PGRID"])),
                              title=dict(text="\U0001F4E1 Strength Radar",
                                         font=dict(size=14, color=t['TEXT'])))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)

        with t_col:
            trend = get_trend(sel, bat_df, bwl_df,
                              season=sel_season if sel_season != "All" else None)
            if trend is not None and len(trend) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend["start_date"], y=trend["impact_score"],
                    mode="lines+markers",
                    line=dict(color=CYAN, width=3, shape="spline"),
                    marker=dict(size=8, color=CYAN,
                                line=dict(width=2, color=t["BG"])),
                    fill="tozeroy", fillcolor=rgba(CYAN, 0.04),
                ))
                if len(trend) >= 5:
                    trend["ma"] = trend["impact_score"].rolling(3, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=trend["start_date"], y=trend["ma"],
                        mode="lines", name="3-inn avg",
                        line=dict(color=GOLD, width=1.5, dash="dot"),
                    ))
                sfx = f" ({sel_season})" if sel_season != "All" else " \u2014 Recent Form"
                fig.update_layout(**FL(), **FG(), height=380,
                                  title=dict(text=f"\U0001F4C8 {sel}{sfx}",
                                             font=dict(size=14, color=t['TEXT'])),
                                  xaxis_title="Date", yaxis_title="Impact Score",
                                  showlegend=True,
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t['TEXT2']),
                                              orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True, config=PCFG)
            else:
                st.info("Not enough innings for a trend chart.")


# ════════════════════════════════════════════════════════
# TAB 3 — HEAD-TO-HEAD
# ════════════════════════════════════════════════════════
with tabs[2]:
    plist = pdf["player"].tolist()
    ha, hb = st.columns(2)
    with ha: pa = st.selectbox("\U0001F535 Player A", plist, key="h2h_a")
    with hb: pb = st.selectbox("\U0001F7E0 Player B", plist, index=min(1,len(plist)-1), key="h2h_b")

    if pa and pb and pa != pb:
        a = pdf[pdf["player"]==pa].iloc[0]
        b = pdf[pdf["player"]==pb].iloc[0]

        # VS Header
        st.markdown(f"""
        <div style="text-align:center;margin:8px 0 16px;">
          <span style="font-size:1.2rem;font-weight:800;color:{CYAN};
                       font-family:'Outfit','Inter',sans-serif;">{pa}</span>
          <span style="margin:0 14px;font-size:1.5rem;font-weight:900;
                       background:linear-gradient(135deg,{GOLD},{ORANGE});
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            VS
          </span>
          <span style="font-size:1.2rem;font-weight:800;color:{ORANGE};
                       font-family:'Outfit','Inter',sans-serif;">{pb}</span>
        </div>""", unsafe_allow_html=True)

        # Side KPIs
        lc, mc_col, rc_col = st.columns([1, 2.5, 1])
        with lc:
            st.markdown(kpi_card("Impact", f"{a['overall_impact']:.1f}", CYAN), unsafe_allow_html=True)
            st.markdown(kpi_card("Runs", si(a["total_runs"]), ORANGE), unsafe_allow_html=True)
            st.markdown(kpi_card("Wickets", si(a["total_wickets"]), PURPLE), unsafe_allow_html=True)
            st.markdown(kpi_card("Win Rate", f"{a['win_rate']:.0%}", GREEN), unsafe_allow_html=True)

        with rc_col:
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
                na = (1-va/mx)*100 if col == "avg_economy" else va/mx*100
                nb = (1-vb/mx)*100 if col == "avg_economy" else vb/mx*100
                y_labels.append(label)
                xa_vals.append(na); xb_vals.append(nb)
                fig.add_trace(go.Scatter(
                    x=[na, nb], y=[label, label], mode="lines",
                    line=dict(color=t["LINE_CON"], width=3),
                    showlegend=False, hoverinfo="skip",
                ))

            fig.add_trace(go.Scatter(
                x=xa_vals, y=y_labels, mode="markers+text",
                marker=dict(size=14, color=CYAN, line=dict(width=2, color=t["BG"]),
                            symbol="circle"),
                text=[f"{v:.0f}" for v in xa_vals], textposition="top center",
                textfont=dict(size=9, color=CYAN), name=pa,
            ))
            fig.add_trace(go.Scatter(
                x=xb_vals, y=y_labels, mode="markers+text",
                marker=dict(size=14, color=ORANGE, line=dict(width=2, color=t["BG"]),
                            symbol="diamond"),
                text=[f"{v:.0f}" for v in xb_vals], textposition="bottom center",
                textfont=dict(size=9, color=ORANGE), name=pb,
            ))
            fig.update_layout(**FL(), height=400,
                              title=dict(text="\U0001F4CA Normalized Comparison",
                                         font=dict(size=14, color=t['TEXT'])),
                              xaxis=dict(title="% of better player", range=[0, 115],
                                         gridcolor=t["PGRID"]),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t['TEXT2']),
                                          orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Radar + Trend
        rr_col, tt_col = st.columns(2)
        with rr_col:
            rva, rvb = radar(a, pdf), radar(b, pdf)
            cats = list(rva.keys())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(rva.values()) + [list(rva.values())[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor=rgba(CYAN, 0.08),
                line=dict(color=CYAN, width=2.5), name=pa,
                marker=dict(size=5, color=CYAN),
            ))
            fig.add_trace(go.Scatterpolar(
                r=list(rvb.values()) + [list(rvb.values())[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor=rgba(ORANGE, 0.08),
                line=dict(color=ORANGE, width=2.5), name=pb,
                marker=dict(size=5, color=ORANGE),
            ))
            fig.update_layout(**FL(), height=400,
                              polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(visible=True, range=[0,1],
                                                         gridcolor=t["PGRID"],
                                                         tickfont=dict(size=9, color=t['TEXT2'])),
                                         angularaxis=dict(tickfont=dict(size=11, color=t['TEXT']),
                                                          gridcolor=t["PGRID"])),
                              title=dict(text="\U0001F4E1 Radar Comparison",
                                         font=dict(size=14, color=t['TEXT'])),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t['TEXT2']),
                                          orientation="h", y=-0.08))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)

        with tt_col:
            ta = get_trend(pa, bat_df, bwl_df)
            tb_t = get_trend(pb, bat_df, bwl_df)
            if ta is not None and tb_t is not None and len(ta) >= 3 and len(tb_t) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ta["start_date"], y=ta["impact_score"],
                    mode="lines+markers", name=pa,
                    line=dict(color=CYAN, width=2.5, shape="spline"),
                    marker=dict(size=7, color=CYAN, line=dict(width=1, color=t["BG"])),
                ))
                fig.add_trace(go.Scatter(
                    x=tb_t["start_date"], y=tb_t["impact_score"],
                    mode="lines+markers", name=pb,
                    line=dict(color=ORANGE, width=2.5, shape="spline"),
                    marker=dict(size=7, color=ORANGE, line=dict(width=1, color=t["BG"])),
                ))
                fig.update_layout(**FL(), **FG(), height=400,
                                  title=dict(text="\U0001F4C8 Form Comparison",
                                             font=dict(size=14, color=t['TEXT'])),
                                  xaxis_title="Date", yaxis_title="Impact Score",
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t['TEXT2']),
                                              orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True, config=PCFG)
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
        # Themed HTML comparison table
        _cmp_rows = ""
        for _, _cr in comp.iterrows():
            _bc = CYAN if _cr["Better"] == pa else ORANGE
            _cmp_rows += f"""<tr>
              <td style="font-weight:700">{_cr['Metric']}</td>
              <td style="text-align:right">{_cr[pa]:.2f}</td>
              <td style="text-align:right">{_cr[pb]:.2f}</td>
              <td style="color:{_bc};font-weight:600">{_cr['Better']}</td></tr>"""
        st.markdown(f"""<div class="glass" style="padding:0;overflow:hidden;">
        <table class="themed-table">
          <thead><tr>
            <th>Metric</th><th style="text-align:right">{pa}</th>
            <th style="text-align:right">{pb}</th><th>Better</th>
          </tr></thead>
          <tbody>{_cmp_rows}</tbody>
        </table></div>""", unsafe_allow_html=True)

        # Rivalry
        if riv_df is not None:
            h2h = riv_df[((riv_df["batter"]==pa)&(riv_df["bowler_name"]==pb)) |
                         ((riv_df["batter"]==pb)&(riv_df["bowler_name"]==pa))]
            if len(h2h) > 0:
                st.markdown(section_header("Batter vs Bowler Rivalry", "\u2694\uFE0F"),
                            unsafe_allow_html=True)
                st.dataframe(h2h, use_container_width=True)

    elif pa == pb:
        st.warning("Pick two different players to compare.")


# ════════════════════════════════════════════════════════
# TAB 4 — MATCH EXPLORER
# ════════════════════════════════════════════════════════
with tabs[3]:
    if mdf is None:
        st.info("match_rankings.csv not found \u2014 run the pipeline first.")
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
            pick = st.selectbox("\U0001F3DF\uFE0F Select Match", disp, key="mx_pick")
            sel_id = id_map[pick]

            m = mf[mf["match_id"]==sel_id].sort_values("rank").head(22)
            venue = m["venue"].iloc[0] if "venue" in m.columns else ""
            seas  = m["season"].iloc[0] if "season" in m.columns else ""
            tms   = m["player_team"].unique().tolist()

            st.markdown(f"""
            <div class="glass" style="text-align:center;padding:16px 24px;">
                <span style="color:{CYAN};font-weight:800;font-size:1.2rem;
                             font-family:'Outfit','Inter',sans-serif;">
                    {' vs '.join(tms)}
                </span>
                <br><span style="color:{t['TEXT2']};font-size:.85rem;font-weight:500;">
                    \U0001F3DF\uFE0F {venue}  \u2022  Season {seas}
                </span>
            </div>""", unsafe_allow_html=True)

            top11 = m.head(11)
            team_colors = {tms[i]: [CYAN, ORANGE, GREEN, PURPLE][i % 4] for i in range(len(tms))}
            bar_colors = [team_colors.get(tc, CYAN) for tc in top11["player_team"]]

            fig = go.Figure(go.Bar(
                x=top11["performance_score"], y=top11["player"], orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0), cornerradius=5),
                text=top11["key_stat"], textposition="outside",
                textfont=dict(size=10, color=t['TEXT2']),
            ))
            fig.update_layout(**FL(), height=400,
                              title=dict(text="\U0001F3C6 Top Performers",
                                         font=dict(size=14, color=t['TEXT'])),
                              xaxis_title="Performance Score")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, config=PCFG)

            show_c = [c for c in ["player","role","player_team","key_stat",
                                  "performance_score","impact_score","impact_category",
                                  "wpa_score","clutch_score","rank"] if c in m.columns]
            _mdf = m[show_c].copy()
            _mx_col_cfg = {}
            for _nc in ["performance_score","impact_score","wpa_score","clutch_score"]:
                if _nc in _mdf.columns:
                    _mx_col_cfg[_nc] = st.column_config.NumberColumn(_nc, format="%.2f")
            if "rank" in _mdf.columns:
                _mdf["rank"] = _mdf["rank"].fillna(0).astype(int)
            st.dataframe(_mdf, column_config=_mx_col_cfg,
                use_container_width=True, height=400, hide_index=True)

            # --- AI Match Summary ---
            api_key = get_groq_key()
            if api_key:
                if st.button("Generate Match Summary", key="mx_summary_btn"):
                    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
                    top_by_team = []
                    for _, grp in m.groupby("player_team", dropna=False):
                        top_by_team.append(grp.sort_values("impact_score", ascending=False).head(5))
                    top_df = pd.concat(top_by_team, ignore_index=True) if top_by_team else m.head(10)

                    context = {
                        "match_id": str(sel_id),
                        "teams": tms,
                        "venue": venue,
                        "season": str(seas),
                        "players": top_df[show_c].fillna("").to_dict(orient="records"),
                    }
                    prompt = (
                        "You are an IPL analyst. Write simple, professional explanations for the top 5 players from each team. "
                        "Use short, clear sentences and avoid heavy terms. "
                        "Output a structured list grouped by team. For each player, give 1-2 sentences "
                        "explaining why their impact_score is high or low using role, key_stat, and impact_category. "
                        "Use the numeric context signals in the data to describe pressure, momentum, and phase "
                        "(powerplay, middle, death) when implied, but do NOT mention terms like WPA or clutch scores. "
                        "Do not invent facts that are not present.\n\n"
                        "Output format (must be identical for every player):\n"
                        "Player | Role | Key Stat | Impact Explanation\n"
                        "Example: V Kohli | batter | 62 runs (SR 137.8) | Anchored a shaky start and kept the chase steady...\n\n"
                        + json.dumps(context, ensure_ascii=True)
                    )
                    with st.spinner("Generating match summary..."):
                        text, err = groq_generate(prompt, api_key, model=model)
                    if err:
                        st.error(f"Groq error: {err}")
                        st.info("Tip: set GROQ_MODEL in .env (e.g., openai/gpt-oss-120b).")
                    else:
                        table = parse_explanations(text)
                        if len(table) > 0:
                            for team_name, team_df in table.groupby("Team", dropna=False):
                                display_df = team_df.drop(columns=["Team"]).reset_index(drop=True)
                                display_df.insert(0, "No.", np.arange(1, len(display_df) + 1))
                                html_table = render_team_table(team_name, display_df)
                                st.markdown(html_table, unsafe_allow_html=True)
                        else:
                            st.markdown(text)
        else:
            st.caption("No matches for the selected filters.")


# ==================== TAB 5 - MATCH IMPACT UPLOAD ====================
with tabs[4]:
    st.markdown("#### Upload Match Files")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        ball_file = st.file_uploader("Match CSV (e.g., 335982.csv)", type=["csv"], key="ball_upload")
    with col_u2:
        info_file = st.file_uploader("Info CSV (e.g., 335982_info.csv)", type=["csv"], key="info_upload")

    if "upload_rankings" not in st.session_state:
        st.session_state.upload_rankings = None
    if "upload_show_cols" not in st.session_state:
        st.session_state.upload_show_cols = None

    if ball_file is None or info_file is None:
        st.info("Upload both files to compute match impact.")
    else:
        if not info_file.name.endswith("_info.csv"):
            st.warning("Info file name should end with _info.csv.")

        if st.button("Compute match impact", key="match_impact_btn"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f_ball:
                    f_ball.write(ball_file.getbuffer())
                    ball_path = f_ball.name
                with tempfile.NamedTemporaryFile(delete=False, suffix="_info.csv") as f_info:
                    f_info.write(info_file.getbuffer())
                    info_path = f_info.name

                rankings = compute_match_rankings_from_files(ball_path, info_path)
                show_cols = [c for c in [
                    "match_id", "player", "role", "player_team", "opposition", "key_stat",
                    "performance_score", "impact_score", "impact_category", "wpa_score", "clutch_score"
                ] if c in rankings.columns]
                st.session_state.upload_rankings = rankings
                st.session_state.upload_show_cols = show_cols
            except Exception as exc:
                st.error(f"Failed to compute impact: {exc}")

    if st.session_state.upload_rankings is not None:
        rankings = st.session_state.upload_rankings
        show_cols = st.session_state.upload_show_cols or []

        st.markdown("### All Players (Match Impact)")
        st.dataframe(rankings[show_cols], width='stretch', height=420, hide_index=True)

        api_key = get_groq_key()
        if not api_key:
            st.warning("Set GROQ_API_KEY in .env to generate explanations.")
        else:
            if st.button("Generate top 5 explanations", key="match_explain_btn"):
                model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
                top_by_team = []
                for _, grp in rankings.groupby("player_team", dropna=False):
                    top_by_team.append(grp.sort_values("impact_score", ascending=False).head(5))
                top_df = pd.concat(top_by_team, ignore_index=True) if top_by_team else rankings.head(10)

                context = {
                    "match_id": str(rankings["match_id"].iloc[0]) if "match_id" in rankings.columns else "",
                    "players": top_df[show_cols].fillna("").to_dict(orient="records"),
                }
                prompt = (
                    "You are an IPL analyst. Write simple, professional explanations for the top 5 players from each team. "
                    "Use short, clear sentences and avoid heavy terms. "
                    "Output a structured list grouped by team. For each player, give 1-2 sentences "
                    "explaining why their impact_score is high or low using role, key_stat, and impact_category. "
                    "Use the numeric context signals in the data to describe pressure, momentum, and phase "
                    "(powerplay, middle, death) when implied, but do NOT mention terms like WPA or clutch scores. "
                    "Do not invent facts that are not present.\n\n"
                    "Output format (must be identical for every player):\n"
                    "Player | Role | Key Stat | Impact Explanation\n"
                    "Example: V Kohli | batter | 62 runs (SR 137.8) | Anchored a shaky start and kept the chase steady...\n\n"
                    + json.dumps(context, ensure_ascii=True)
                )
                with st.spinner("Generating explanations with Groq..."):
                    text, err = groq_generate(prompt, api_key, model=model)
                if err:
                    st.error(f"Groq error: {err}")
                    st.info("Tip: set GROQ_MODEL in .env (e.g., openai/gpt-oss-120b).")
                else:
                    table = parse_explanations(text)
                    if len(table) > 0:
                        for team_name, team_df in table.groupby("Team", dropna=False):
                            display_df = team_df.drop(columns=["Team"]).reset_index(drop=True)
                            display_df.insert(0, "No.", np.arange(1, len(display_df) + 1))
                            html_table = render_team_table(team_name, display_df)
                            st.markdown(html_table, unsafe_allow_html=True)
                    else:
                        st.markdown(text)

