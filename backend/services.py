import os
import pandas as pd
import numpy as np
import tempfile
from typing import Dict, Any, List, Optional
import sys

# Add the parent directory to the path so we can import the original matcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from match_impact_from_files import compute_match_rankings_from_files
except ImportError:
    pass

# --- Paths relative to backend directory ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(BASE_DIR, "cricket_impact")
P_FILE   = os.path.join(DIR, "all_players_impact_T20.csv")
M_FILE   = os.path.join(DIR, "match_rankings.csv")
BAT_FILE = os.path.join(DIR, "batting_impact.csv")
BWL_FILE = os.path.join(DIR, "bowling_impact.csv")
RIV_FILE = os.path.join(DIR, "batter_vs_bowler_rivalry.csv")

# Global variables for caching in memory
_data = {
    "pdf": None,
    "mdf": None,
    "bat_df": None,
    "bwl_df": None,
    "riv_df": None
}

def load_data():
    if _data["pdf"] is None:
        _data["pdf"] = pd.read_csv(P_FILE) if os.path.exists(P_FILE) else None
        _data["mdf"] = pd.read_csv(M_FILE) if os.path.exists(M_FILE) else None
        _data["bat_df"] = pd.read_csv(BAT_FILE) if os.path.exists(BAT_FILE) else None
        _data["bwl_df"] = pd.read_csv(BWL_FILE) if os.path.exists(BWL_FILE) else None
        _data["riv_df"] = pd.read_csv(RIV_FILE) if os.path.exists(RIV_FILE) else None

def get_data() -> Dict[str, pd.DataFrame]:
    load_data()
    return _data

def si(v, d=0):
    try: return int(v)
    except: return d

def role_of(row):
    b, w = si(row.get("bat_innings")), si(row.get("bowl_innings"))
    if b >= 10 and w >= 10 and min(b,w)/max(b,w) >= 0.4:
        return "All-rounder"
    return "Batter" if b >= w else "Bowler"

def clipped_mean(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr).mean()

def process_players(min_inn: int = 10, role_pick: str = "All", 
                   season: str = "All", min_runs: int = 0, min_wkts: int = 0):
    
    data = get_data()
    pdf_raw = data["pdf"]
    mdf = data["mdf"]
    
    if pdf_raw is None:
        return []
        
    season_mode = season != "All" and mdf is not None

    if season_mode:
        sm = mdf[mdf["season"].astype(str) == str(season)]
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
    if min_runs > 0:  pdf = pdf[pdf["total_runs"] >= min_runs]
    if min_wkts > 0:  pdf = pdf[pdf["total_wickets"] >= min_wkts]

    if len(pdf) == 0:
        return []

    if role_pick == "Batter":
        pdf = pdf[pdf["role"].isin(["Batter","All-rounder"])].copy()
        log_vol = np.log1p(pdf["total_balls_faced"]) / np.log1p(pdf["total_balls_faced"].max())
        raw = pdf["mean_bat_impact"] * log_vol
        bmax = raw.max() or 1
        pdf["display_impact"] = raw / bmax * 100
    elif role_pick == "Bowler":
        pdf = pdf[pdf["role"].isin(["Bowler","All-rounder"])].copy()
        log_vol = np.log1p(pdf["total_balls_bowled"]) / np.log1p(pdf["total_balls_bowled"].max())
        raw = pdf["mean_bowl_impact"] * log_vol
        wmax = raw.max() or 1
        pdf["display_impact"] = raw / wmax * 100
    elif role_pick == "All-rounder":
        pdf = pdf[pdf["role"]=="All-rounder"].copy()
        bvol = np.log1p(pdf["total_balls_faced"]) / np.log1p(pdf["total_balls_faced"].max())
        wvol = np.log1p(pdf["total_balls_bowled"]) / np.log1p(pdf["total_balls_bowled"].max())
        raw = pdf["mean_bat_impact"] * bvol + pdf["mean_bowl_impact"] * wvol
        cmax = raw.max() or 1
        pdf["display_impact"] = raw / cmax * 100
    else:
        if season_mode:
            pdf["display_impact"] = pdf["season_avg_impact"]
        else:
            pdf["display_impact"] = pdf["overall_impact"]

    pdf = pdf.sort_values("display_impact", ascending=False).reset_index(drop=True)
    
    # Replace NaN with None for JSON serialization
    pdf = pdf.where(pd.notnull(pdf), None)
    return pdf.to_dict(orient="records")

def get_seasons():
    mdf = get_data()["mdf"]
    if mdf is not None:
        return sorted(mdf["season"].dropna().astype(str).unique().tolist())
    return []

def get_player_trend(name: str, season: Optional[str] = None):
    data = get_data()
    bat = data["bat_df"]
    bowl = data["bwl_df"]
    
    df = None
    if bat is not None and "player" in bat.columns:
        df = bat[bat["player"] == name].copy()
    if (df is None or len(df) < 3) and bowl is not None and "player" in bowl.columns:
        df = bowl[bowl["player"] == name].copy()
    
    if df is None or "impact_score" not in df.columns:
        return []
        
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if season and season != "All" and "season" in df.columns:
        df = df[df["season"].astype(str) == season]
    
    df = df.sort_values("start_date").tail(15)
    df["start_date"] = df["start_date"].dt.strftime("%Y-%m-%d")
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")

def get_matches(season: str = "All", team: str = "All"):
    mdf = get_data()["mdf"]
    if mdf is None:
        return []
        
    mf = mdf.copy()
    if season and season != "All":
        mf = mf[mf["season"].astype(str) == str(season)]
    if team and team != "All":
        mf = mf[mf["player_team"].astype(str) == team]
        
    out = []
    for mid, g in mf.groupby("match_id"):
        ts = g["player_team"].unique().tolist()
        v  = g["venue"].iloc[0].split(",")[0].strip() if "venue" in g.columns else ""
        s  = str(g["season"].iloc[0]) if "season" in g.columns else ""
        lbl = f"{ts[0]} vs {ts[1]}" if len(ts) >= 2 else f"{ts[0]} vs {g['opposition'].iloc[0] if 'opposition' in g.columns else '?'}"
        if v: lbl += f" | {v}"
        if s: lbl += f" ({s})"
        
        # Get top 11 players for the match
        top11 = g.sort_values("rank").head(11)
        top_players = top11.where(pd.notnull(top11), None).to_dict(orient="records")
        
        out.append({
            "match_id": int(mid),
            "label": lbl,
            "season": s,
            "venue": v,
            "teams": ts,
            "top_players": top_players
        })
        
    return sorted(out, key=lambda x: x["match_id"], reverse=True)

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
    return rows

def generate_ai_summary(context_data: dict, api_key: str):
    import json
    import os
    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile") # specific model
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        prompt = (
            "You are an IPL analyst. Write simple, professional explanations for the top players. "
            "Output format (must be identical for every player):\n"
            "Player | Role | Key Stat | Impact Explanation\n"
            "Example: V Kohli | batter | 62 runs (SR 137.8) | Anchored a shaky start...\n\n"
            + json.dumps(context_data, ensure_ascii=True)
        )
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an IPL analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
            top_p=1,
            stream=False,
        )
        text = completion.choices[0].message.content
        return {"success": True, "text": text, "parsed": parse_explanations(text)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
