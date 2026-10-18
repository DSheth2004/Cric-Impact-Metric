"""Aggregated analytics endpoints used by the React frontend."""
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services import DIR, get_data, role_of, get_player_trend

PART_FILE = os.path.join(DIR, "batting_partnerships.csv")
TANDEM_FILE = os.path.join(DIR, "bowling_partnerships.csv")
_extra: Dict[str, Optional[pd.DataFrame]] = {"part": None, "tandem": None}


def _part() -> Optional[pd.DataFrame]:
    if _extra["part"] is None and os.path.exists(PART_FILE):
        _extra["part"] = pd.read_csv(PART_FILE)
    return _extra["part"]


def clean(obj: Any) -> Any:
    """Recursively convert numpy types and NaN/inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return clean(df.to_dict(orient="records"))


def _players() -> pd.DataFrame:
    pdf = get_data()["pdf"].copy()
    pdf["role"] = pdf.apply(role_of, axis=1)
    m = pdf[["bat_matches", "bowl_matches"]].max(axis=1).replace(0, np.nan)
    pdf["pom_rate"] = (pdf["pom_awards"] / m).fillna(0)
    pdf["impact_pct"] = pdf.get("impact_per_match", 0) * 100
    for c in ["bowling_rating", "impact_per_match_shrunk"]:
        if c not in pdf.columns:
            pdf[c] = 0.0
    return pdf


def _f(v, d=0.0) -> float:
    try:
        f = float(v)
        return d if math.isnan(f) else f
    except (TypeError, ValueError):
        return d


# ---------------------------------------------------------------- overview
def overview() -> Dict[str, Any]:
    pdf = _players()
    mdf = get_data()["mdf"]

    role_split = pdf["role"].value_counts().rename_axis("name").reset_index(name="value")
    cat_split = mdf["impact_category"].value_counts().rename_axis("name").reset_index(name="value")

    bins = list(range(0, 91, 10))
    hist = pd.cut(pdf["overall_impact"], bins=bins, include_lowest=True).value_counts().sort_index()
    histogram = [{"range": f"{int(i.left) if i.left > 0 else 0}-{int(i.right)}", "count": int(c)}
                 for i, c in hist.items()]

    season = mdf.groupby("season").agg(
        matches=("match_id", "nunique"),
        avg_impact=("impact_score", "mean"),
        players=("player", "nunique"),
    ).reset_index()
    season["season"] = season["season"].astype(str)

    def top(df, col, n=10):
        t = df.sort_values(col, ascending=False).head(n)
        out = t[["player", "role", "overall_impact", "total_runs", "total_wickets"]].copy()
        out["value"] = t[col]
        return records(out)

    bats = pdf[pdf["bat_innings"] >= 30]
    bowls = pdf[pdf["bowl_innings"] >= 30]
    ars = pdf[pdf["role"] == "All-rounder"]

    renamed = {"Royal Challengers Bengaluru": "Royal Challengers Bangalore",
               "Rising Pune Supergiant": "Rising Pune Supergiants",
               "Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings"}
    team = mdf.assign(player_team=mdf["player_team"].replace(renamed)).groupby("player_team").agg(
        avg_impact=("impact_score", "mean"),
        match_winners=("impact_category", lambda s: int((s == "Match Winner").sum())),
    ).reset_index().sort_values("avg_impact", ascending=False)

    totals = {
        "players": int(len(pdf)),
        "matches": int(mdf["match_id"].nunique()),
        "seasons": int(mdf["season"].nunique()),
        "runs": int(pdf["total_runs"].fillna(0).sum()),
        "wickets": int(pdf["total_wickets"].fillna(0).sum()),
        "sixes": int(pdf["total_sixes"].fillna(0).sum()),
        "fours": int(pdf["total_fours"].fillna(0).sum()),
        "performances": int(len(mdf)),
    }
    return clean({
        "totals": totals,
        "role_split": records(role_split),
        "category_split": records(cat_split),
        "histogram": histogram,
        "season_trend": records(season),
        "team_impact": records(team),
        "top_overall": top(pdf, "overall_impact"),
        "top_batters": top(bats, "batting_rating"),
        "top_bowlers": top(bowls, "bowling_rating"),
        "top_allrounders": top(ars, "overall_impact"),
        "boundary_split": [
            {"name": "Runs in 4s", "value": totals["fours"] * 4},
            {"name": "Runs in 6s", "value": totals["sixes"] * 6},
            {"name": "Running", "value": max(totals["runs"] - totals["fours"] * 4 - totals["sixes"] * 6, 0)},
        ],
    })


# ---------------------------------------------------------------- player
def _breakdowns(r: pd.Series) -> Dict[str, Any]:
    runs, fours, sixes = _f(r.get("total_runs")), _f(r.get("total_fours")), _f(r.get("total_sixes"))
    bat_inn, outs = _f(r.get("bat_innings")), _f(r.get("total_dismissals"))
    balls_faced, dots_faced = _f(r.get("total_balls_faced")), _f(r.get("total_dots_faced"))
    bb, death, pp = _f(r.get("total_balls_bowled")), _f(r.get("total_death_balls")), _f(r.get("total_pp_balls"))
    wk, dw, pw = _f(r.get("total_wickets")), _f(r.get("total_death_wickets")), _f(r.get("total_pp_wickets"))
    dots_b = _f(r.get("total_dots_bowled"))
    cap, cap_w = _f(r.get("matches_as_captain")), _f(r.get("wins_as_captain"))
    matches = max(_f(r.get("bat_matches")), _f(r.get("bowl_matches")))
    wins = _f(r.get("total_wins"))
    return {
        "run_composition": [
            {"name": "Fours", "value": fours * 4},
            {"name": "Sixes", "value": sixes * 6},
            {"name": "Running", "value": max(runs - fours * 4 - sixes * 6, 0)},
        ],
        "dismissals": [
            {"name": "Dismissed", "value": outs},
            {"name": "Not out", "value": max(bat_inn - outs, 0)},
        ],
        "balls_faced": [
            {"name": "Dot balls", "value": dots_faced},
            {"name": "Scoring balls", "value": max(balls_faced - dots_faced, 0)},
        ],
        "bowling_phases": [
            {"name": "Powerplay", "value": pp},
            {"name": "Middle", "value": max(bb - pp - death, 0)},
            {"name": "Death", "value": death},
        ],
        "wicket_phases": [
            {"name": "Powerplay", "value": pw},
            {"name": "Middle", "value": max(wk - pw - dw, 0)},
            {"name": "Death", "value": dw},
        ],
        "balls_bowled": [
            {"name": "Dot balls", "value": dots_b},
            {"name": "Scoring balls", "value": max(bb - dots_b, 0)},
        ],
        "results": [
            {"name": "Wins", "value": wins},
            {"name": "Losses / NR", "value": max(matches - wins, 0)},
        ],
        "captaincy": [
            {"name": "Won", "value": cap_w},
            {"name": "Lost / NR", "value": max(cap - cap_w, 0)},
        ],
    }


RADAR = [
    ("Impact", "overall_impact", False),
    ("Batting", "batting_rating", False),
    ("Impact / match", "impact_per_match_shrunk", False),
    ("Strike Rate", "avg_bat_sr", False),
    ("Bowling", "bowling_rating", False),
    ("Economy", "avg_economy", True),
    ("Win Rate", "win_rate", False),
    ("POM rate", "pom_rate", False),
]


def _radar(rows: List[pd.Series], pdf: pd.DataFrame) -> List[Dict[str, Any]]:
    """Percentile rank of each metric among qualified players (0-100)."""
    out = []
    for label, col, invert in RADAR:
        if col not in pdf.columns:
            continue
        s = pdf[col].astype(float)
        # compare each skill only among players who actually do it
        if col in ("avg_economy", "bowling_rating"):
            s = pdf.loc[pdf["bowl_innings"] >= 10, col].astype(float)
        elif col in ("batting_rating", "avg_bat_sr"):
            s = pdf.loc[pdf["bat_innings"] >= 10, col].astype(float)
        item = {"metric": label}
        for i, r in enumerate(rows):
            v = r.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)) or len(s.dropna()) == 0:
                pct = 0.0
            else:
                pct = float((s.dropna() < float(v)).mean() * 100)
                if invert:
                    pct = 100 - pct
            if col in ("avg_economy", "bowling_rating") and _f(r.get("bowl_innings")) < 5:
                pct = 0.0
            if col in ("batting_rating", "avg_bat_sr") and _f(r.get("bat_innings")) < 5:
                pct = 0.0
            item[f"p{i}"] = round(pct, 1)
        out.append(item)
    return out


def _find(pdf: pd.DataFrame, name: str) -> pd.Series:
    hit = pdf[pdf["player"] == name]
    if hit.empty:
        raise KeyError(name)
    return hit.iloc[0]


def player_list() -> List[Dict[str, Any]]:
    pdf = _players().sort_values("overall_impact", ascending=False)
    return records(pdf[["player", "role", "overall_impact", "total_innings"]])


def player_detail(name: str) -> Dict[str, Any]:
    data = get_data()
    pdf = _players()
    r = _find(pdf, name)
    rank = int((pdf["overall_impact"] > r["overall_impact"]).sum() + 1)

    mdf = data["mdf"]
    pm = mdf[mdf["player"] == name]
    cat = pm["impact_category"].value_counts().rename_axis("name").reset_index(name="value")
    seasons = pm.groupby("season").agg(
        avg_impact=("impact_score", "mean"), matches=("match_id", "nunique")
    ).reset_index()
    seasons["season"] = seasons["season"].astype(str)
    teams = pm.drop_duplicates("match_id")["player_team"].value_counts().rename_axis("name").reset_index(name="value")
    best = pm.sort_values("impact_score", ascending=False).head(8)[
        ["match_id", "season", "opposition", "key_stat", "impact_score", "role"]]

    riv = data["riv_df"]
    as_bat = riv[(riv["batter"] == name) & (riv["balls"] >= 12)]
    as_bowl = riv[(riv["bowler_name"] == name) & (riv["balls"] >= 12)]

    part = _part()
    parts = pd.DataFrame()
    if part is not None:
        pp = part[(part["p1"] == name) | (part["p2"] == name)].copy()
        if len(pp):
            pp["partner"] = np.where(pp["p1"] == name, pp["p2"], pp["p1"])
            parts = pp.groupby("partner").agg(
                runs=("partnership_total", "sum"), stands=("match_id", "count"),
                best=("partnership_total", "max"),
            ).reset_index().sort_values("runs", ascending=False).head(8)

    return clean({
        "player": {k: v for k, v in r.to_dict().items()},
        "rank": rank,
        "total_players": int(len(pdf)),
        "breakdowns": _breakdowns(r),
        "radar": _radar([r], pdf),
        "categories": records(cat),
        "season_impact": records(seasons),
        "teams": records(teams),
        "best_matches": records(best),
        "trend": _trend(name, r["role"]),
        "dominated_bowlers": records(as_bat.sort_values("dominance_index", ascending=False).head(6)),
        "nemesis_bowlers": records(as_bat.sort_values(["dismissals", "dominance_index"], ascending=[False, True]).head(6)),
        "dominated_batters": records(as_bowl.sort_values(["dismissals", "dominance_index"], ascending=[False, True]).head(6)),
        "partners": records(parts) if len(parts) else [],
    })


def _trend(name: str, role: str, n: int = 15) -> List[Dict[str, Any]]:
    """Last n innings in the player's primary discipline (bowling for bowlers)."""
    data = get_data()
    src = data["bwl_df"] if role == "Bowler" else data["bat_df"]
    df = src[src["player"] == name].copy()
    if len(df) < 3:
        return get_player_trend(name, "All")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.sort_values("start_date").tail(n)
    df["start_date"] = df["start_date"].dt.strftime("%Y-%m-%d")
    keep = [c for c in ["start_date", "opposition", "impact_score", "runs", "balls_faced", "wickets", "runs_conceded"] if c in df.columns]
    return records(df[keep])


COMPARE_STATS = [
    ("Overall Impact", "overall_impact"), ("Batting Rating", "batting_rating"),
    ("Bowling Rating", "bowling_rating"), ("Win Prob Added / match %", "impact_pct"),
    ("Runs", "total_runs"), ("Batting Avg", "true_bat_avg"), ("Strike Rate", "avg_bat_sr"),
    ("Fours", "total_fours"), ("Sixes", "total_sixes"), ("Wickets", "total_wickets"),
    ("Economy", "avg_economy"), ("Dot % Bowled", "dot_pct_bowled"), ("Win Rate", "win_rate"),
    ("POM Awards", "pom_awards"), ("Matches as Captain", "matches_as_captain"),
    ("Innings", "total_innings"),
]


def compare(a: str, b: str) -> Dict[str, Any]:
    data = get_data()
    pdf = _players()
    ra, rb = _find(pdf, a), _find(pdf, b)
    stats = [{"label": lbl, "key": col, "a": _f(ra.get(col)), "b": _f(rb.get(col))}
             for lbl, col in COMPARE_STATS if col in pdf.columns]
    riv = data["riv_df"]
    h2h = riv[((riv["batter"] == a) & (riv["bowler_name"] == b)) |
              ((riv["batter"] == b) & (riv["bowler_name"] == a))]
    mdf = data["mdf"]

    def season_series(n):
        s = mdf[mdf["player"] == n].groupby("season")["impact_score"].mean()
        return {str(k): float(v) for k, v in s.items()}

    sa, sb = season_series(a), season_series(b)
    seasons = sorted(set(sa) | set(sb))
    return clean({
        "a": ra.to_dict(), "b": rb.to_dict(),
        "breakdowns_a": _breakdowns(ra), "breakdowns_b": _breakdowns(rb),
        "radar": _radar([ra, rb], pdf),
        "stats": stats,
        "head_to_head": records(h2h),
        "season_trend": [{"season": s, "a": sa.get(s), "b": sb.get(s)} for s in seasons],
    })


# ---------------------------------------------------------------- match
def match_detail(match_id: int) -> Dict[str, Any]:
    mdf = get_data()["mdf"]
    g = mdf[mdf["match_id"] == match_id].sort_values("rank")
    if g.empty:
        raise KeyError(match_id)
    return clean({"match_id": match_id, **summarize_match(g)})


def summarize_match(g: pd.DataFrame) -> Dict[str, Any]:
    team = g.groupby("player_team")["impact_score"].sum().rename_axis("name").reset_index(name="value")
    cat = g["impact_category"].value_counts().rename_axis("name").reset_index(name="value")
    role = g.groupby("role")["impact_score"].sum().rename_axis("name").reset_index(name="value")
    first = g.iloc[0]
    return {
        "season": str(first.get("season", "")),
        "venue": str(first.get("venue", "")),
        "teams": g["player_team"].dropna().unique().tolist(),
        "players": records(g),
        "team_impact": records(team),
        "category_split": records(cat),
        "role_impact": records(role),
    }
