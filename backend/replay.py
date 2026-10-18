"""Ball-by-ball win probability replay, model card and the runs-vs-impact insight."""
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services import DIR, data_file
from insights import clean, records, _players

BALL_FILE = data_file("ball_win_prob.csv")
CARD_FILE = data_file("model_card.json")
DATA_FOLDER = os.path.join(os.path.dirname(DIR), "ipl_male_csv2")

# Hand-picked thrillers for the home page replay
FEATURED = [
    (1181768, "2019 Final", "Mumbai win the title by one run off the last ball"),
    (1082650, "2017 Final", "Mumbai defend 129 and win by a single run"),
    (1473511, "2025 Final", "Bengaluru finally lift the trophy"),
    (1136610, "2018 Thriller", "The favourite flips eight times"),
]


@lru_cache(maxsize=1)
def _balls() -> Optional[pd.DataFrame]:
    if not os.path.exists(BALL_FILE):
        return None
    df = pd.read_csv(BALL_FILE)
    return df.set_index("match_id", drop=False).sort_index()  # fast per-match lookup


@lru_cache(maxsize=1)
def _info() -> pd.DataFrame:
    """Winner, margin, venue and Player of the Match per match: from the raw Cricsheet folder
    locally, or from the small match_info.csv(.gz) bundled for deployment."""
    if os.path.isdir(DATA_FOLDER):
        from cricket_impact.data_loader import load_match_info
        return load_match_info(DATA_FOLDER)
    path = data_file("match_info.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in ("winner", "player_of_match", "venue"):
            df[c] = df[c].fillna("")
        df["season"] = df["season"].astype(str)
        return df
    return pd.DataFrame()


def model_card() -> Dict[str, Any]:
    if os.path.exists(CARD_FILE):
        with open(CARD_FILE) as f:
            return json.load(f)
    return {}


def _describe(r: Dict[str, Any]) -> str:
    who = f"{r['bowler']} to {r['striker']}"
    if r["is_wicket"] == 1:
        kind = r.get("wicket_type") if isinstance(r.get("wicket_type"), str) and r.get("wicket_type") else "out"
        out = r.get("player_dismissed") if isinstance(r.get("player_dismissed"), str) and r.get("player_dismissed") else r["striker"]
        return f"{who}: WICKET, {out} {kind}"
    if r["runs_off_bat"] == 6:
        return f"{who}: SIX"
    if r["runs_off_bat"] == 4:
        return f"{who}: FOUR"
    n = int(r["total_runs"])
    return f"{who}: {n} run{'' if n == 1 else 's'}"


def _margin(rec: Optional[pd.Series]) -> str:
    if rec is None:
        return ""
    if rec["winner_runs"]:
        n = int(rec["winner_runs"])
        return f"won by {n} run{'' if n == 1 else 's'}"
    if rec["winner_wickets"]:
        n = int(rec["winner_wickets"])
        return f"won by {n} wicket{'' if n == 1 else 's'}"
    return "tied / no result"


@lru_cache(maxsize=256)
def worm(match_id: int) -> Dict[str, Any]:
    balls = _balls()
    if balls is None:
        raise FileNotFoundError("Run backend/build_win_prob.py first")
    if match_id not in balls.index:
        raise KeyError(match_id)
    g = balls.loc[[match_id]]
    g = g[g["innings"].isin([1, 2])].sort_values(["innings", "over", "ball_in_over"]).reset_index(drop=True)
    if g.empty or g["innings"].max() < 2:
        raise KeyError(match_id)
    team1 = g[g["innings"] == 1]["batting_team"].iloc[0]
    team2 = g[g["innings"] == 2]["batting_team"].iloc[0]
    # win probability of the chasing side throughout the match
    cwp = np.where(g["batting_team"] == team2, g["win_prob"], 1 - g["win_prob"])
    swing = np.diff(np.concatenate([[cwp[0]], cwp]))

    info = _info()
    hit = info[info["match_id"] == match_id] if len(info) else info
    rec = hit.iloc[0] if len(hit) else None
    winner = rec["winner"] if rec is not None else ""

    rows = g.to_dict("records")
    points = []
    for i, (r, wp, sw) in enumerate(zip(rows, cwp, swing)):
        ev = "W" if r["is_wicket"] == 1 else "6" if r["runs_off_bat"] == 6 else "4" if r["runs_off_bat"] == 4 else ""
        points.append({
            "i": i, "inn": int(r["innings"]), "over": f"{int(r['over'])}.{int(r['ball_in_over'])}",
            "score": f"{int(r['cumulative_runs'])}/{int(r['cumulative_wickets'])}",
            "wp": round(float(wp), 4), "swing": round(float(sw), 4), "ev": ev,
            "text": _describe(r), "batting": r["batting_team"],
        })

    # biggest single-ball swings; the first ball of the chase has no comparable prior state
    brk = int((g["innings"] == 1).sum())
    cand = [p for p in points if p["i"] not in (0, brk)]
    moments = sorted(sorted(cand, key=lambda p: abs(p["swing"]), reverse=True)[:5], key=lambda p: p["i"])

    # who added the most win probability for their own side in this match: uses the
    # pipeline's credited WPA (batting side's view), which includes the innings-break
    # share and the final step to the result
    df = pd.DataFrame({"striker": g["striker"].values, "bowler": g["bowler"].values,
                       "bat_team": g["batting_team"].values, "bowl_team": g["bowling_team"].values,
                       "gain": g["wpa"].values})
    bat = df.groupby(["striker", "bat_team"])["gain"].sum().reset_index()
    bat.columns = ["player", "team", "wpa"]
    bowl = df.groupby(["bowler", "bowl_team"])["gain"].sum().mul(-1).reset_index()
    bowl.columns = ["player", "team", "wpa"]
    contrib = pd.concat([bat, bowl]).groupby(["player", "team"])["wpa"].sum().reset_index()
    top = contrib.sort_values("wpa", ascending=False).head(5)

    inn1, inn2 = g[g["innings"] == 1], g[g["innings"] == 2]
    return clean({
        "match_id": match_id, "team1": team1, "team2": team2, "winner": winner, "margin": _margin(rec),
        "season": str(rec["season"]) if rec is not None else "", "venue": str(rec["venue"]) if rec is not None else "",
        "player_of_match": rec["player_of_match"] if rec is not None else "",
        "score1": f"{int(inn1['cumulative_runs'].iloc[-1])}/{int(inn1['cumulative_wickets'].iloc[-1])}",
        "score2": f"{int(inn2['cumulative_runs'].iloc[-1])}/{int(inn2['cumulative_wickets'].iloc[-1])}",
        "innings_break": brk,
        "final_wp": 1.0 if winner == team2 else 0.0 if winner == team1 else 0.5,
        "points": points, "moments": moments, "top_wpa": records(top),
    })


def featured() -> List[Dict[str, Any]]:
    out = []
    for mid, title, blurb in FEATURED:
        try:
            w = worm(mid)
        except Exception:
            continue
        out.append({"match_id": mid, "title": title, "blurb": blurb, "team1": w["team1"],
                    "team2": w["team2"], "winner": w["winner"], "margin": w["margin"]})
    return out


def runs_vs_impact(min_innings: int = 40) -> List[Dict[str, Any]]:
    """Career runs vs average win probability added per innings (percentage points)."""
    pdf = _players()
    b = pdf[pdf["bat_innings"] >= min_innings].copy()
    b["wpa_per_inn"] = b["total_bat_wpa"] / b["bat_innings"] * 100
    return records(b[["player", "role", "total_runs", "bat_innings", "avg_bat_sr", "wpa_per_inn"]])
