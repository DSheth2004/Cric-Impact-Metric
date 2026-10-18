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

CTX_FILE = os.path.join(DIR, "player_match_context.csv")
_ctx = {"df": None}


def get_ctx():
    if _ctx["df"] is None and os.path.exists(CTX_FILE):
        df = pd.read_csv(CTX_FILE)
        df["season"] = df["season"].astype(str)
        _ctx["df"] = df
    return _ctx["df"]


def _model_card() -> Dict[str, Any]:
    import json
    path = os.path.join(DIR, "model_card.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


VIEW = {"All": "all", "Batter": "bat", "Bowler": "bowl", "All-rounder": "allround"}


def process_players(min_inn: int = 10, role_pick: str = "All",
                    season: str = "All", min_runs: int = 0, min_wkts: int = 0):
    """
    Leaderboard rows for any season x role combination, all from one formula
    (cricket_impact/rating.py). Season views use only that season's innings.
    """
    from cricket_impact.rating import rate, replacement_level, center

    ctx = get_ctx()
    pdf = get_data()["pdf"]
    if ctx is None or pdf is None:
        return []
    season_mode = season not in (None, "", "All")
    card = _model_card()
    full = center(ctx, card.get("discipline_means"))   # same baseline as the pipeline
    c = full[full["season"] == str(season)] if season_mode else full
    if c.empty:
        return []
    # Shrinkage strength and replacement level come from the whole career dataset, so a
    # single season is judged on the same scale (within one season everyone looks "fringe").
    view = VIEW.get(role_pick, "all")
    k = card.get("shrink_k")
    repl = (card.get("replacement_level") or {}).get(view)
    if repl is None:
        repl = replacement_level(full, view)
    # all-rounder classification needs fewer innings inside a single season
    r = rate(c, view, role_min_inn=4 if season_mode else 10, shrink_k=k, repl=repl)
    if r.empty:
        return []

    out = pd.DataFrame({
        "player": r.index,
        "role": r["role"].values,
        "display_impact": r["rating"].values,
        "rank": r["rank"].values,
        "total_innings": r["innings_rated"].values if role_pick in ("Batter", "Bowler") else (r["bat_innings"] + r["bowl_innings"]).values,
        "matches": r["appearances"].values,
        "total_runs": r["runs"].values,
        "total_balls_faced": r["balls_faced"].values,
        "avg_bat_sr": r["strike_rate"].values,
        "true_bat_avg": r["bat_avg"].values,
        "total_wickets": r["wickets"].values,
        "total_balls_bowled": r["balls_bowled"].values,
        "avg_economy": r["economy"].values,
        "win_rate": r["win_rate"].values,
        "pom_awards": r["pom"].values,
        "impact_per_inn": (r["impact_per_inn_shrunk"] * 100).values,  # win prob points per innings
        "impact_total": (r["impact_total"] * 100).values,
        "value_above_repl": (r["impact_above_repl"] * 100).values,  # win-prob points above replacement
        # per-discipline impact for the scatter; hidden below 5 innings, where one game dominates
        "bat_impact_per_inn": (r["bat_wpa"] / r["bat_innings"].where(r["bat_innings"] >= 5) * 100).values,
        "bowl_impact_per_inn": (r["bowl_wpa"] / r["bowl_innings"].where(r["bowl_innings"] >= 5) * 100).values,
        "bonus": (r["bonus"] * 100).values,
    })
    if min_inn > 0:
        out = out[out["total_innings"] >= min_inn]
    if min_runs > 0:
        out = out[out["total_runs"] >= min_runs]
    if min_wkts > 0:
        out = out[out["total_wickets"] >= min_wkts]
    out = out.sort_values("display_impact", ascending=False).reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notnull(out), None)
    return out.to_dict(orient="records")

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
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") # specific model
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
