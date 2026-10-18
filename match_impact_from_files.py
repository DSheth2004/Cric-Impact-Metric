"""
Compute per-match impact from a single ball-by-ball CSV and its _info CSV.
Designed for IPL-style Cricsheet files (e.g., 335982.csv, 335982_info.csv).
"""

import os
import re
import pandas as pd

from cricket_impact.data_loader import clean_data
from cricket_impact.player_aggregation import (
    aggregate_batting,
    aggregate_bowling,
    compute_phase_performance,
    get_innings_totals,
    merge_runs_contribution,
)
from cricket_impact.context_features import add_all_context_features
from cricket_impact.win_prob_model import load_model, predict_win_prob
from cricket_impact.wpa_engine import (
    compute_per_ball_wpa,
    aggregate_batter_wpa,
    compute_batter_presence_impact,
    aggregate_bowler_wpa,
    compute_clutch_wpa,
)
from cricket_impact.impact_metric import compute_batting_impact, compute_bowling_impact
from cricket_impact.rolling_normalization import apply_rolling_normalization


def _detect_format(event_name, max_over):
    event = str(event_name or "").lower()
    if any(k in event for k in ["test", "trophy", "sheffield", "ranji", "county", "ashes"]):
        return "Test"
    if any(k in event for k in ["t20", "twenty20", "ipl", "bbl", "premier league", "cpl", "psl",
                                "hundred", "sa20", "blast"]):
        return "T20"
    if any(k in event for k in ["odi", "world cup", "asia cup", "champions trophy", "50-over"]):
        return "ODI"
    if max_over <= 20:
        return "T20"
    if max_over <= 50:
        return "ODI"
    return "Test"


def _parse_info_file(info_path, match_id):
    rec = {"match_id": match_id}
    teams = []
    roster = []
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3 or parts[0] != "info":
                    continue
                key = parts[1].strip()
                val = ",".join(parts[2:]).strip().strip("\"")
                if key == "winner":
                    rec["winner"] = val
                elif key == "winner_runs":
                    rec["winner_runs"] = int(val)
                elif key == "winner_wickets":
                    rec["winner_wickets"] = int(val)
                elif key == "player_of_match":
                    rec["player_of_match"] = val
                elif key == "toss_winner":
                    rec["toss_winner"] = val
                elif key == "toss_decision":
                    rec["toss_decision"] = val
                elif key == "venue":
                    rec["venue"] = val
                elif key == "city":
                    rec["city"] = val
                elif key == "season":
                    rec["season"] = val
                elif key == "date":
                    rec["start_date"] = val
                elif key == "outcome" and val == "no result":
                    rec["no_result"] = True
                elif key == "match_number":
                    try:
                        rec["match_number"] = int(val)
                    except ValueError:
                        pass
                elif key == "event":
                    rec["event"] = val
                elif key == "team":
                    teams.append(val)
                elif key == "player" and len(parts) >= 4:
                    team_name = parts[2].strip()
                    player_name = ",".join(parts[3:]).strip().strip("\"")
                    roster.append((team_name, player_name))
    except FileNotFoundError:
        raise

    rec.setdefault("winner", "")
    rec.setdefault("winner_runs", 0)
    rec.setdefault("winner_wickets", 0)
    rec.setdefault("player_of_match", "")
    rec.setdefault("toss_winner", "")
    rec.setdefault("toss_decision", "")
    rec.setdefault("venue", "")
    rec.setdefault("city", "")
    rec.setdefault("season", "")
    rec.setdefault("start_date", "")
    rec.setdefault("no_result", False)
    rec.setdefault("match_number", -1)
    rec["teams"] = teams
    rec["roster"] = roster

    info_df = pd.DataFrame([rec])
    info_df["winner_runs"] = info_df["winner_runs"].fillna(0).astype(int)
    info_df["winner_wickets"] = info_df["winner_wickets"].fillna(0).astype(int)
    info_df["no_result"] = info_df["no_result"].fillna(False)
    info_df["match_number"] = info_df["match_number"].fillna(-1).astype(int)
    info_df["is_playoff"] = (info_df["match_number"] == -1).astype(int)

    return info_df


def _extract_match_id(path_or_name):
    name = os.path.basename(path_or_name)
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def load_single_match(ball_path, info_path):
    ball_df = pd.read_csv(ball_path, low_memory=False)
    match_id = _extract_match_id(ball_path)
    if "match_id" not in ball_df.columns:
        if match_id is None:
            raise ValueError("match_id not found in filename or data")
        ball_df["match_id"] = match_id
    else:
        match_id = int(ball_df["match_id"].iloc[0])

    info_df = _parse_info_file(info_path, match_id)

    # Clean and derive columns
    ball_df = clean_data(ball_df)

    # Merge basic info into ball_df
    if "venue" not in ball_df.columns or ball_df["venue"].isna().all():
        ball_df["venue"] = info_df["venue"].iloc[0]
    if "season" not in ball_df.columns or ball_df["season"].isna().all():
        ball_df["season"] = info_df["season"].iloc[0]
    if "start_date" not in ball_df.columns or ball_df["start_date"].isna().all():
        ball_df["start_date"] = info_df["start_date"].iloc[0]

    # Detect format
    max_over = ball_df.groupby("match_id")["over"].max().iloc[0]
    fmt = _detect_format(info_df.get("event", [""])[0], max_over)
    info_df["match_format"] = fmt
    ball_df["match_format"] = fmt

    return ball_df, info_df, match_id


def _build_match_rankings(bat_impact, bowl_impact):
    bat_rows = bat_impact[[
        "match_id", "innings", "player", "player_team", "opposition",
        "season", "venue", "runs", "balls_faced", "strike_rate",
        "raw_impact", "impact_score", "impact_category",
        "total_wpa", "clutch_wpa"
    ]].copy()
    bat_rows["role"] = "batter"
    bat_rows["key_stat"] = bat_rows.apply(
        lambda r: f"{int(r['runs'])} runs ({r['strike_rate']:.1f} SR)" if r["balls_faced"] > 0 else "DNB",
        axis=1,
    )

    bowl_cols = [
        "match_id", "innings", "player", "player_team", "opposition",
        "season", "venue", "wickets", "runs_conceded", "economy",
        "balls_bowled", "raw_impact", "impact_score", "impact_category",
        "total_wpa", "clutch_wpa",
    ]
    bowl_cols_available = [c for c in bowl_cols if c in bowl_impact.columns]
    bowl_rows = bowl_impact[bowl_cols_available].copy()
    bowl_rows["role"] = "bowler"
    if "wickets" in bowl_rows.columns and "runs_conceded" in bowl_rows.columns:
        bowl_rows["key_stat"] = bowl_rows.apply(
            lambda r: f"{int(r['wickets'])}/{int(r['runs_conceded'])} ({r.get('economy', 0):.1f} econ)",
            axis=1,
        )
    else:
        bowl_rows["key_stat"] = ""

    for df in [bat_rows, bowl_rows]:
        for c in [
            "runs", "balls_faced", "strike_rate", "wickets", "runs_conceded",
            "economy", "balls_bowled",
        ]:
            if c not in df.columns:
                df[c] = 0

    common_cols = [
        "match_id", "innings", "player", "role", "player_team", "opposition",
        "season", "venue", "key_stat", "raw_impact", "impact_score",
        "impact_category", "total_wpa", "clutch_wpa",
    ]
    for df in [bat_rows, bowl_rows]:
        for c in common_cols:
            if c not in df.columns:
                df[c] = "" if c in [
                    "key_stat", "impact_category", "player", "role",
                    "player_team", "opposition", "season", "venue",
                ] else 0

    combined = pd.concat([bat_rows[common_cols], bowl_rows[common_cols]], ignore_index=True)
    combined["rank"] = combined.groupby("match_id")["raw_impact"].rank(
        ascending=False, method="min"
    ).astype(int)
    combined = combined.sort_values(["match_id", "rank"]).reset_index(drop=True)

    combined.rename(columns={
        "total_wpa": "wpa_score",
        "clutch_wpa": "clutch_score",
        "raw_impact": "performance_score",
    }, inplace=True)

    return combined


def compute_match_rankings_from_files(ball_path, info_path, model_dir=None):
    ball_df, info_df, _ = load_single_match(ball_path, info_path)

    # Aggregations
    bat_df = aggregate_batting(ball_df)
    bowl_df = aggregate_bowling(ball_df)
    compute_phase_performance(ball_df)
    innings_totals = get_innings_totals(ball_df)
    bat_df = merge_runs_contribution(bat_df, innings_totals)

    # Context features
    ball_df = add_all_context_features(ball_df, info_df)

    # Win prob model
    model = load_model(model_dir)
    if model is None:
        raise RuntimeError("win_prob_model.pkl not found. Run the pipeline first to train the model.")
    ball_df = predict_win_prob(ball_df, model)

    # WPA
    ball_df = compute_per_ball_wpa(ball_df, info_df)
    batter_wpa = aggregate_batter_wpa(ball_df)
    presence = compute_batter_presence_impact(ball_df)
    bowler_wpa = aggregate_bowler_wpa(ball_df)
    batter_clutch, bowler_clutch = compute_clutch_wpa(ball_df)

    bat_df = bat_df.merge(
        batter_wpa[[
            "match_id", "innings", "player", "total_wpa", "avg_wpa",
            "max_wpa", "min_wpa", "wpa_volatility", "positive_wpa_pct",
            "leverage_wpa", "consistency_wpa",
        ]],
        on=["match_id", "innings", "player"], how="left",
    )
    bat_df = bat_df.merge(
        presence[["match_id", "innings", "player", "presence_impact", "wp_at_arrival", "wp_at_departure"]],
        on=["match_id", "innings", "player"], how="left",
    )
    bat_df = bat_df.merge(
        batter_clutch[["match_id", "innings", "player", "clutch_wpa"]],
        on=["match_id", "innings", "player"], how="left",
    )

    bowl_df = bowl_df.merge(
        bowler_wpa[[
            "match_id", "innings", "player", "total_wpa", "avg_wpa",
            "wicket_wpa", "leverage_wpa", "spell_efficiency", "economy_wpa",
        ]],
        on=["match_id", "innings", "player"], how="left",
    )
    bowl_df = bowl_df.merge(
        bowler_clutch[["match_id", "innings", "player", "clutch_wpa"]],
        on=["match_id", "innings", "player"], how="left",
    )

    wpa_fill_cols = [
        "total_wpa", "avg_wpa", "max_wpa", "min_wpa", "wpa_volatility",
        "positive_wpa_pct", "leverage_wpa", "consistency_wpa",
        "presence_impact", "wp_at_arrival", "wp_at_departure",
        "clutch_wpa", "wicket_wpa", "spell_efficiency", "economy_wpa",
    ]
    for c in wpa_fill_cols:
        if c in bat_df.columns:
            bat_df[c] = bat_df[c].fillna(0)
        if c in bowl_df.columns:
            bowl_df[c] = bowl_df[c].fillna(0)

    if "match_format" not in bat_df.columns:
        bat_df["match_format"] = "T20"
    if "match_format" not in bowl_df.columns:
        bowl_df["match_format"] = "T20"

    bat_parts = []
    for fmt, grp in bat_df.groupby("match_format"):
        bat_parts.append(compute_batting_impact(grp, match_format=fmt))
    bat_impact = pd.concat(bat_parts, ignore_index=True) if bat_parts else bat_df.copy()

    bowl_parts = []
    for fmt, grp in bowl_df.groupby("match_format"):
        bowl_parts.append(compute_bowling_impact(grp, match_format=fmt))
    bowl_impact = pd.concat(bowl_parts, ignore_index=True) if bowl_parts else bowl_df.copy()

    bat_norm_parts = []
    for fmt, grp in bat_impact.groupby("match_format"):
        bat_norm_parts.append(apply_rolling_normalization(grp, impact_col="raw_impact"))
    bat_impact = pd.concat(bat_norm_parts, ignore_index=True) if bat_norm_parts else bat_impact

    bowl_norm_parts = []
    for fmt, grp in bowl_impact.groupby("match_format"):
        bowl_norm_parts.append(apply_rolling_normalization(grp, impact_col="raw_impact"))
    bowl_impact = pd.concat(bowl_norm_parts, ignore_index=True) if bowl_norm_parts else bowl_impact

    match_rankings = _build_match_rankings(bat_impact, bowl_impact)
    return match_rankings
