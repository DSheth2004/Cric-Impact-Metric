"""
Pipeline Runner: Orchestrates the full Cricket Impact Metric pipeline.
Usage: python run_pipeline.py --data_folder ./ipl_male_csv2 [--retrain]
"""

import os
import sys
import time
import argparse
import warnings
import pandas as pd
import numpy as np
import json

warnings.filterwarnings('ignore')

# Add parent dir to path so cricket_impact package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cricket_impact.data_loader import load_and_prepare
from cricket_impact.player_aggregation import (
    aggregate_batting, aggregate_bowling, compute_phase_performance,
    get_innings_totals, merge_runs_contribution
)
from cricket_impact.context_features import add_all_context_features
from cricket_impact.win_prob_model import cross_fit_win_prob, FEATURES
from cricket_impact.rating import build_player_match_context, rate, summarize, estimate_shrink_k, replacement_level, center, discipline_means
from cricket_impact.wpa_engine import (
    compute_per_ball_wpa, aggregate_batter_wpa, compute_batter_presence_impact,
    aggregate_bowler_wpa, compute_clutch_wpa
)
from cricket_impact.partnerships import (
    compute_batting_partnerships, get_best_partnership_per_player,
    compute_bowling_tandems, compute_batter_vs_bowler_rivalry,
    merge_rivalry_features
)
from cricket_impact.impact_metric import compute_batting_impact, compute_bowling_impact
from cricket_impact.rolling_normalization import apply_rolling_normalization
from cricket_impact.visualization import generate_all_visualizations
from cricket_impact.captaincy import (
    build_captaincy_data, compute_captaincy_stats,
    compute_win_bonus, compute_playoff_team_bonus
)


def build_match_rankings(bat_impact, bowl_impact):
    """Build per-match rankings: all players ranked by raw_impact within each match."""
    # Prepare batter rows
    bat_rows = bat_impact[['match_id', 'innings', 'player', 'player_team', 'opposition',
                            'season', 'venue', 'runs', 'balls_faced', 'strike_rate',
                            'raw_impact', 'impact_score', 'impact_category',
                            'total_wpa', 'clutch_wpa']].copy()
    bat_rows['role'] = 'batter'
    bat_rows['key_stat'] = bat_rows.apply(
        lambda r: f"{int(r['runs'])} runs ({r['strike_rate']:.1f} SR)" if r['balls_faced'] > 0 else "DNB",
        axis=1
    )

    # Prepare bowler rows
    bowl_cols = ['match_id', 'innings', 'player', 'player_team', 'opposition',
                 'season', 'venue', 'wickets', 'runs_conceded', 'economy',
                 'balls_bowled', 'raw_impact', 'impact_score', 'impact_category',
                 'total_wpa', 'clutch_wpa']
    bowl_cols_available = [c for c in bowl_cols if c in bowl_impact.columns]
    bowl_rows = bowl_impact[bowl_cols_available].copy()
    bowl_rows['role'] = 'bowler'
    if 'wickets' in bowl_rows.columns and 'runs_conceded' in bowl_rows.columns:
        bowl_rows['key_stat'] = bowl_rows.apply(
            lambda r: f"{int(r['wickets'])}/{int(r['runs_conceded'])} ({r.get('economy', 0):.1f} econ)",
            axis=1
        )
    else:
        bowl_rows['key_stat'] = ''

    # Standardize columns
    for df in [bat_rows, bowl_rows]:
        for c in ['runs', 'balls_faced', 'strike_rate', 'wickets', 'runs_conceded',
                   'economy', 'balls_bowled']:
            if c not in df.columns:
                df[c] = 0

    common_cols = ['match_id', 'innings', 'player', 'role', 'player_team', 'opposition',
                   'season', 'venue', 'key_stat', 'raw_impact', 'impact_score',
                   'impact_category', 'total_wpa', 'clutch_wpa']
    for df in [bat_rows, bowl_rows]:
        for c in common_cols:
            if c not in df.columns:
                df[c] = '' if c in ['key_stat', 'impact_category', 'player', 'role',
                                     'player_team', 'opposition', 'season', 'venue'] else 0

    combined = pd.concat([bat_rows[common_cols], bowl_rows[common_cols]], ignore_index=True)

    # Rank within each match
    combined['rank'] = combined.groupby('match_id')['raw_impact'].rank(
        ascending=False, method='min'
    ).astype(int)
    combined = combined.sort_values(['match_id', 'rank']).reset_index(drop=True)

    # Rename for output
    combined.rename(columns={
        'total_wpa': 'wpa_score',
        'clutch_wpa': 'clutch_score',
        'raw_impact': 'performance_score',
    }, inplace=True)

    return combined


def build_all_players_summary(bat_impact, bowl_impact, info_df, ball_df=None,
                               match_format=None, ctx=None):
    """Build career summary: one row per player, optionally filtered by format."""
    # ── Filter inputs to this format if specified ──
    if match_format:
        if 'match_format' in bat_impact.columns:
            bat_impact = bat_impact[bat_impact['match_format'] == match_format].copy()
        if 'match_format' in bowl_impact.columns:
            bowl_impact = bowl_impact[bowl_impact['match_format'] == match_format].copy()
        if info_df is not None and 'match_format' in info_df.columns:
            info_df = info_df[info_df['match_format'] == match_format].copy()
        if ball_df is not None and 'match_format' in ball_df.columns:
            ball_df = ball_df[ball_df['match_format'] == match_format].copy()

    # Batting career stats
    if len(bat_impact) > 0:
        bat_career = bat_impact.groupby('player').agg(
            bat_innings=('match_id', 'count'),
            total_runs=('runs', 'sum'),
            total_balls_faced=('balls_faced', 'sum'),
            total_fours=('fours', 'sum'),
            total_sixes=('sixes', 'sum'),
            avg_bat_sr=('strike_rate', 'mean'),
            mean_bat_impact=('raw_impact', 'mean'),
            max_bat_impact=('raw_impact', 'max'),
            mean_bat_wpa=('total_wpa', 'mean'),
            total_bat_wpa=('total_wpa', 'sum'),
            mean_clutch_bat=('clutch_wpa', 'mean'),
            total_clutch_bat=('clutch_wpa', 'sum'),
            total_dismissals=('was_dismissed', 'sum'),
            bat_matches=('match_id', 'nunique'),
            total_dots_faced=('dots_faced', 'sum'),
        ).reset_index()
        bat_career['dot_pct_faced'] = np.where(
            bat_career['total_balls_faced'] > 0,
            (bat_career['total_dots_faced'] / bat_career['total_balls_faced']) * 100, 0
        ).round(1)
        bat_career['career_bat_avg'] = np.where(
            bat_career['bat_innings'] > 0,
            bat_career['total_runs'] / bat_career['bat_innings'], 0
        )
        bat_career['not_out_pct'] = np.where(
            bat_career['bat_innings'] > 0,
            1 - (bat_career['total_dismissals'] / bat_career['bat_innings']), 0
        )
        bat_career['true_bat_avg'] = np.where(
            bat_career['total_dismissals'] > 0,
            bat_career['total_runs'] / bat_career['total_dismissals'],
            bat_career['total_runs']
        )
    else:
        bat_career = pd.DataFrame(columns=['player'])

    # Bowling career stats
    if len(bowl_impact) > 0:
        bowl_career = bowl_impact.groupby('player').agg(
            bowl_innings=('match_id', 'count'),
            total_wickets=('wickets', 'sum'),
            total_runs_conceded=('runs_conceded', 'sum'),
            total_balls_bowled=('balls_bowled', 'sum'),
            avg_economy=('economy', 'mean'),
            mean_bowl_impact=('raw_impact', 'mean'),
            max_bowl_impact=('raw_impact', 'max'),
            mean_bowl_wpa=('total_wpa', 'mean'),
            total_bowl_wpa=('total_wpa', 'sum'),
            mean_clutch_bowl=('clutch_wpa', 'mean'),
            total_clutch_bowl=('clutch_wpa', 'sum'),
            bowl_matches=('match_id', 'nunique'),
            total_death_balls=('death_balls', 'sum'),
            total_death_wickets=('death_wickets', 'sum'),
            total_pp_balls=('pp_balls', 'sum'),
            total_pp_wickets=('pp_wickets', 'sum'),
            avg_phase_difficulty=('phase_difficulty_bonus', 'mean'),
            total_dots_bowled=('dots_bowled', 'sum'),
        ).reset_index()
        bowl_career['dot_pct_bowled'] = np.where(
            bowl_career['total_balls_bowled'] > 0,
            (bowl_career['total_dots_bowled'] / bowl_career['total_balls_bowled']) * 100, 0
        ).round(1)
    else:
        bowl_career = pd.DataFrame(columns=['player'])

    # Merge
    all_players = bat_career.merge(bowl_career, on='player', how='outer')

    # Fill NaN
    for c in all_players.columns:
        if all_players[c].dtype in [np.float64, np.int64, float, int]:
            all_players[c] = all_players[c].fillna(0)

    # Player of match awards
    if info_df is not None and 'player_of_match' in info_df.columns:
        pom = info_df[info_df['player_of_match'] != ''].groupby('player_of_match').size().reset_index(name='pom_awards')
        pom.rename(columns={'player_of_match': 'player'}, inplace=True)
        all_players = all_players.merge(pom, on='player', how='left')
        all_players['pom_awards'] = all_players['pom_awards'].fillna(0).astype(int)
    else:
        all_players['pom_awards'] = 0

    # ── Captaincy stats ──
    if ball_df is not None and info_df is not None:
        captaincy_df = build_captaincy_data(info_df, ball_df)
        captaincy_stats = compute_captaincy_stats(captaincy_df)
        if len(captaincy_stats) > 0:
            all_players = all_players.merge(
                captaincy_stats[['player', 'matches_as_captain', 'wins_as_captain',
                                 'captain_playoff_matches', 'captaincy_score',
                                 'captaincy_score_norm']],
                on='player', how='left'
            )
    for c in ['matches_as_captain', 'wins_as_captain', 'captain_playoff_matches',
              'captaincy_score', 'captaincy_score_norm']:
        if c not in all_players.columns:
            all_players[c] = 0
        all_players[c] = all_players[c].fillna(0)

    # ── Win bonus: all players on winning teams get credit ──
    if ball_df is not None and info_df is not None:
        win_stats = compute_win_bonus(ball_df, info_df)
        if len(win_stats) > 0:
            all_players = all_players.merge(
                win_stats[['player', 'total_wins', 'win_rate', 'win_bonus_norm']],
                on='player', how='left'
            )
    for c in ['total_wins', 'win_rate', 'win_bonus_norm']:
        if c not in all_players.columns:
            all_players[c] = 0
        all_players[c] = all_players[c].fillna(0)

    # ── Playoff team bonus ──
    if ball_df is not None and info_df is not None:
        playoff_bonus = compute_playoff_team_bonus(info_df, ball_df)
        if len(playoff_bonus) > 0:
            all_players = all_players.merge(
                playoff_bonus[['player', 'playoff_seasons']],
                on='player', how='left'
            )
    if 'playoff_seasons' not in all_players.columns:
        all_players['playoff_seasons'] = 0
    all_players['playoff_seasons'] = all_players['playoff_seasons'].fillna(0)

    # ── v2 rating: Win Probability Added, shrunk, blended, plus a capped bonus ──
    # (see cricket_impact/rating.py). Strike rate and economy come from career totals.
    all_players['avg_bat_sr'] = np.where(all_players['total_balls_faced'] > 0,
                                         all_players['total_runs'] / all_players['total_balls_faced'] * 100, 0)
    all_players['avg_economy'] = np.where(all_players['total_balls_bowled'] > 0,
                                          all_players['total_runs_conceded'] / (all_players['total_balls_bowled'] / 6), 0)
    all_players['total_innings'] = all_players['bat_innings'] + all_players['bowl_innings']
    for c in ['overall_impact', 'batting_rating', 'bowling_rating', 'impact_per_match',
              'impact_per_match_shrunk', 'impact_total', 'rating_bonus']:
        all_players[c] = 0.0
    if ctx is not None and len(ctx):
        c = ctx[ctx['match_id'].isin(set(bat_impact['match_id']) | set(bowl_impact['match_id']))]
        c = center(c)  # batting vs average batting innings, bowling vs average spell
        summary = summarize(c)
        k = estimate_shrink_k(c)
        print(f"  Shrinkage strength (empirical Bayes): K = {k:.0f} matches")
        r_all = rate(c, 'all', players=summary, shrink_k=k)
        r_bat = rate(c, 'bat', players=summary, shrink_k=k)
        r_bowl = rate(c, 'bowl', players=summary, shrink_k=k)
        idx = all_players['player']
        all_players['overall_impact'] = idx.map(r_all['rating']).fillna(0).values
        all_players['impact_per_match'] = idx.map(r_all['impact_per_inn']).fillna(0).values
        all_players['impact_per_match_shrunk'] = idx.map(r_all['impact_per_inn_shrunk']).fillna(0).values
        all_players['impact_total'] = idx.map(r_all['impact_total']).fillna(0).values
        all_players['rating_bonus'] = idx.map(r_all['bonus']).fillna(0).values
        all_players['batting_rating'] = idx.map(r_bat['rating']).fillna(0).values
        all_players['bowling_rating'] = idx.map(r_bowl['rating']).fillna(0).values
    all_players = all_players.sort_values('overall_impact', ascending=False).reset_index(drop=True)

    return all_players


def run_pipeline(data_folder, retrain=False, output_dir='cricket_impact'):
    """Execute the full pipeline."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # ── Step 1: Load & Clean ──
    print("=" * 60)
    print("STEP 1: Loading and cleaning data")
    print("=" * 60)
    ball_df, info_df = load_and_prepare(data_folder)

    # ── Step 2: Aggregate batting/bowling ──
    print("\n" + "=" * 60)
    print("STEP 2: Aggregating player stats")
    print("=" * 60)
    bat_df = aggregate_batting(ball_df)
    print(f"  Batting records: {len(bat_df):,}")
    bowl_df = aggregate_bowling(ball_df)
    print(f"  Bowling records: {len(bowl_df):,}")
    bat_phase, bowl_phase = compute_phase_performance(ball_df)
    innings_totals = get_innings_totals(ball_df)
    print(f"  Phase batting: {len(bat_phase):,}, Phase bowling: {len(bowl_phase):,}")

    # ── Step 3: Merge runs_contribution ──
    bat_df = merge_runs_contribution(bat_df, innings_totals)
    print("  Runs contribution merged")

    # ── Step 4: Context features ──
    print("\n" + "=" * 60)
    print("STEP 4: Computing context features")
    print("=" * 60)
    ball_df = add_all_context_features(ball_df, info_df)

    # ── Step 5: Train or load ML model ──
    print("\n" + "=" * 60)
    print("STEP 5: Win probability model (5-fold cross-fitted, monotonic)")
    print("=" * 60)
    # Every match is scored by a model that never saw its result. The `retrain`
    # flag is kept for compatibility; the model is always refit (it takes seconds).
    ball_df, model_metrics = cross_fit_win_prob(ball_df, info_df, n_splits=5, model_dir=model_dir)
    print(f"  Win prob range: [{ball_df['win_prob'].min():.3f}, {ball_df['win_prob'].max():.3f}]")

    # ── Step 7: WPA ──
    print("\n" + "=" * 60)
    print("STEP 7: Computing WPA")
    print("=" * 60)
    ball_df = compute_per_ball_wpa(ball_df, info_df)  # full-match crediting incl. innings break
    print(f"  Per-ball WPA computed")

    batter_wpa = aggregate_batter_wpa(ball_df)
    presence = compute_batter_presence_impact(ball_df)
    bowler_wpa = aggregate_bowler_wpa(ball_df)
    batter_clutch, bowler_clutch = compute_clutch_wpa(ball_df)
    print(f"  Batter WPA records: {len(batter_wpa):,}")
    print(f"  Bowler WPA records: {len(bowler_wpa):,}")

    # Merge WPA into bat_df
    bat_df = bat_df.merge(
        batter_wpa[['match_id', 'innings', 'player', 'total_wpa', 'avg_wpa',
                     'max_wpa', 'min_wpa', 'wpa_volatility', 'positive_wpa_pct',
                     'leverage_wpa', 'consistency_wpa']],
        on=['match_id', 'innings', 'player'], how='left'
    )
    bat_df = bat_df.merge(
        presence[['match_id', 'innings', 'player', 'presence_impact',
                   'wp_at_arrival', 'wp_at_departure']],
        on=['match_id', 'innings', 'player'], how='left'
    )
    bat_df = bat_df.merge(
        batter_clutch[['match_id', 'innings', 'player', 'clutch_wpa']],
        on=['match_id', 'innings', 'player'], how='left'
    )

    # Merge WPA into bowl_df
    bowl_df = bowl_df.merge(
        bowler_wpa[['match_id', 'innings', 'player', 'total_wpa', 'avg_wpa',
                     'wicket_wpa', 'leverage_wpa', 'spell_efficiency', 'economy_wpa']],
        on=['match_id', 'innings', 'player'], how='left'
    )
    bowl_df = bowl_df.merge(
        bowler_clutch[['match_id', 'innings', 'player', 'clutch_wpa']],
        on=['match_id', 'innings', 'player'], how='left'
    )

    # Fill NaN WPA columns
    wpa_fill_cols = ['total_wpa', 'avg_wpa', 'max_wpa', 'min_wpa', 'wpa_volatility',
                     'positive_wpa_pct', 'leverage_wpa', 'consistency_wpa',
                     'presence_impact', 'wp_at_arrival', 'wp_at_departure',
                     'clutch_wpa', 'wicket_wpa', 'spell_efficiency', 'economy_wpa']
    for c in wpa_fill_cols:
        if c in bat_df.columns:
            bat_df[c] = bat_df[c].fillna(0)
        if c in bowl_df.columns:
            bowl_df[c] = bowl_df[c].fillna(0)

    # ── Step 8: Partnerships & Rivalries ──
    print("\n" + "=" * 60)
    print("STEP 8: Partnerships & Rivalries")
    print("=" * 60)
    partnerships = compute_batting_partnerships(ball_df)
    best_partnerships = get_best_partnership_per_player(partnerships)
    bowling_tandems = compute_bowling_tandems(ball_df)
    rivalry_df = compute_batter_vs_bowler_rivalry(ball_df)
    print(f"  Partnerships: {len(partnerships):,}")
    print(f"  Bowling tandems: {len(bowling_tandems):,}")
    print(f"  Rivalries: {len(rivalry_df):,}")

    # ── Step 9: Merge rivalry features ──
    bat_df, bowl_df = merge_rivalry_features(bat_df, bowl_df, rivalry_df)

    # Merge best partnership into bat_df
    if len(best_partnerships) > 0:
        bat_df = bat_df.merge(
            best_partnerships[['match_id', 'innings', 'player',
                                'best_partnership_runs', 'best_partnership_impact']],
            on=['match_id', 'innings', 'player'], how='left'
        )
        bat_df['best_partnership_runs'] = bat_df['best_partnership_runs'].fillna(0)
        bat_df['best_partnership_impact'] = bat_df['best_partnership_impact'].fillna(0)

    # ── Step 10: Compute impact scores (format-aware) ──
    print("\n" + "=" * 60)
    print("STEP 10: Computing impact scores (format-aware)")
    print("=" * 60)

    # Ensure match_format column exists
    if 'match_format' not in bat_df.columns:
        bat_df['match_format'] = 'T20'
    if 'match_format' not in bowl_df.columns:
        bowl_df['match_format'] = 'T20'

    # Process each format separately, then concatenate
    bat_parts = []
    for fmt, grp in bat_df.groupby('match_format'):
        print(f"  Batting: {fmt} — {len(grp):,} records")
        bat_parts.append(compute_batting_impact(grp, match_format=fmt))
    bat_impact = pd.concat(bat_parts, ignore_index=True) if bat_parts else bat_df.copy()

    bowl_parts = []
    for fmt, grp in bowl_df.groupby('match_format'):
        print(f"  Bowling: {fmt} — {len(grp):,} records")
        bowl_parts.append(compute_bowling_impact(grp, match_format=fmt))
    bowl_impact = pd.concat(bowl_parts, ignore_index=True) if bowl_parts else bowl_df.copy()

    print(f"  Batting impact computed for {len(bat_impact):,} records")
    print(f"  Bowling impact computed for {len(bowl_impact):,} records")

    # ── Step 11: Rolling normalization (per format) ──
    print("\n" + "=" * 60)
    print("STEP 11: Rolling normalization (0-100, per format)")
    print("=" * 60)
    # Normalize within each format so players are compared to same-format peers
    bat_norm_parts = []
    for fmt, grp in bat_impact.groupby('match_format'):
        bat_norm_parts.append(apply_rolling_normalization(grp, impact_col='raw_impact'))
    bat_impact = pd.concat(bat_norm_parts, ignore_index=True) if bat_norm_parts else bat_impact

    bowl_norm_parts = []
    for fmt, grp in bowl_impact.groupby('match_format'):
        bowl_norm_parts.append(apply_rolling_normalization(grp, impact_col='raw_impact'))
    bowl_impact = pd.concat(bowl_norm_parts, ignore_index=True) if bowl_norm_parts else bowl_impact
    print("  Rolling normalization applied (per format)")

    # ── Step 12: Save CSVs ──
    print("\n" + "=" * 60)
    print("STEP 12: Saving output files")
    print("=" * 60)

    bat_impact.to_csv(os.path.join(output_dir, 'batting_impact.csv'), index=False)
    print("  Saved batting_impact.csv")

    bowl_impact.to_csv(os.path.join(output_dir, 'bowling_impact.csv'), index=False)
    print("  Saved bowling_impact.csv")

    partnerships.to_csv(os.path.join(output_dir, 'batting_partnerships.csv'), index=False)
    print("  Saved batting_partnerships.csv")

    bowling_tandems.to_csv(os.path.join(output_dir, 'bowling_partnerships.csv'), index=False)
    print("  Saved bowling_partnerships.csv")

    rivalry_df.to_csv(os.path.join(output_dir, 'batter_vs_bowler_rivalry.csv'), index=False)
    print("  Saved batter_vs_bowler_rivalry.csv")

    # ── Step 13: Match rankings ──
    print("\n" + "=" * 60)
    print("STEP 13: Building match rankings")
    print("=" * 60)
    match_rankings = build_match_rankings(bat_impact, bowl_impact)
    match_rankings.to_csv(os.path.join(output_dir, 'match_rankings.csv'), index=False)
    print(f"  Match rankings: {len(match_rankings):,} rows")

    # ── Step 13b: player-match context (drives every leaderboard view) ──
    captains = build_captaincy_data(info_df, ball_df)
    ctx = build_player_match_context(ball_df, info_df, bat_impact, bowl_impact, captains)
    ctx.to_csv(os.path.join(output_dir, 'player_match_context.csv'), index=False)
    print(f"  Player-match context: {len(ctx):,} rows")

    # ── Step 13c: ball-by-ball win probability + model card (for replays) ──
    keep = ['match_id', 'innings', 'over', 'ball_in_over', 'batting_team', 'bowling_team',
            'striker', 'non_striker', 'bowler', 'runs_off_bat', 'extras', 'total_runs',
            'is_wicket', 'player_dismissed', 'wicket_type', 'cumulative_runs',
            'cumulative_wickets', 'target', 'win_prob', 'wpa']
    bw = ball_df[[c for c in keep if c in ball_df.columns]].copy()
    bw['win_prob'] = bw['win_prob'].round(4)
    bw['wpa'] = bw['wpa'].round(4)
    bw.to_csv(os.path.join(output_dir, 'ball_win_prob.csv'), index=False)
    card_path = os.path.join(output_dir, 'model_card.json')
    card = {}
    if os.path.exists(card_path):  # keep only the old-vs-new comparison from validate_model.py
        with open(card_path) as f:
            prev = json.load(f)
        if 'validation' in prev:
            card['validation'] = prev['validation']
    card.update({
        'model': model_metrics['model'], 'scoring': model_metrics['scoring'],
        'features': FEATURES, 'matches': int(ball_df['match_id'].nunique()),
        'deliveries': int(len(ball_df)), 'auc': model_metrics['oof_auc'],
        'brier': model_metrics['oof_brier'], 'log_loss': model_metrics['oof_log_loss'],
        'discipline_means': discipline_means(ctx),
        'shrink_k': round(estimate_shrink_k(center(ctx)), 1),
        'replacement_level': {v: round(replacement_level(center(ctx), v), 5) for v in ['all', 'bat', 'bowl', 'allround']},
    })
    with open(card_path, 'w') as f:
        json.dump(card, f, indent=2)
    print(f"  ball_win_prob.csv and model_card.json written")

    # ── Step 14: All-players career summary (per format) ──
    print("\n" + "=" * 60)
    print("STEP 14: Building all-players summary (per format)")
    print("=" * 60)

    # Detect all formats present
    formats_in_data = sorted(bat_impact['match_format'].unique()) if 'match_format' in bat_impact.columns else ['T20']
    print(f"  Formats detected: {', '.join(formats_in_data)}")

    all_players_by_format = {}
    for fmt in formats_in_data:
        fmt_players = build_all_players_summary(bat_impact, bowl_impact, info_df, ball_df,
                                                 match_format=fmt, ctx=ctx)
        fmt_players['match_format'] = fmt
        all_players_by_format[fmt] = fmt_players
        # Save per-format CSV
        fmt_file = f'all_players_impact_{fmt}.csv'
        fmt_players.to_csv(os.path.join(output_dir, fmt_file), index=False)
        print(f"  {fmt}: {len(fmt_players)} players → {fmt_file}")

    # Also save a combined file with format column for cross-format queries
    all_players_combined = pd.concat(all_players_by_format.values(), ignore_index=True)
    all_players_combined.to_csv(os.path.join(output_dir, 'all_players_impact.csv'), index=False)
    print(f"  Combined: {len(all_players_combined)} rows → all_players_impact.csv")

    # Use the primary format for validation (most matches)
    primary_format = max(formats_in_data,
                         key=lambda f: len(all_players_by_format[f]))
    all_players = all_players_by_format[primary_format]

    # ── Step 15: Visualizations ──
    print("\n" + "=" * 60)
    print("STEP 15: Generating visualizations")
    print("=" * 60)
    try:
        viz_paths = generate_all_visualizations(bat_impact, bowl_impact, all_players, output_dir)
    except Exception as e:
        print(f"  Visualization error (non-fatal): {e}")
        viz_paths = []

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Deliveries processed: {len(ball_df):,}")
    print(f"  Matches: {ball_df['match_id'].nunique()}")
    print(f"  Players: {len(all_players)} (primary format: {primary_format})")
    print(f"  Formats: {', '.join(formats_in_data)}")
    print(f"  Output directory: {output_dir}/")
    print(f"  Files generated:")
    base_files = ['batting_impact.csv', 'bowling_impact.csv', 'match_rankings.csv',
                   'batting_partnerships.csv', 'bowling_partnerships.csv',
                   'batter_vs_bowler_rivalry.csv', 'all_players_impact.csv']
    format_files = [f'all_players_impact_{fmt}.csv' for fmt in formats_in_data]
    for fname in base_files + format_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"    {fname} ({size / 1024:.0f} KB)")

    # ── Validation spot-checks ──
    print("\n" + "=" * 60)
    print(f"VALIDATION SPOT-CHECKS (format: {primary_format})")
    print("=" * 60)

    # Top players check (overall)
    top10 = all_players.nlargest(10, 'overall_impact')
    print("  Top 10 Overall Impact:")
    for i, (_, row) in enumerate(top10.iterrows()):
        print(f"    {i+1}. {row['player']} - {row['overall_impact']:.1f}")

    # Key player positions
    print("\n  Key Player Positions:")
    targets = [
        ('V Kohli', 50), ('DA Warner', 50), ('S Dhawan', 50),
        ('KA Pollard', 50), ('AD Russell', 50), ('DW Steyn', 50),
        ('JJ Bumrah', 50), ('SL Malinga', 50), ('YS Chahal', 50),
        ('HH Pandya', 50), ('F du Plessis', 50), ('MS Dhoni', 50),
        ('AB de Villiers', 50), ('RG Sharma', 50), ('RA Jadeja', 50),
        ('SP Narine', 50), ('CH Gayle', 50),
    ]
    for name, target in targets:
        m = all_players[all_players['player'].str.contains(name, case=False, na=False)]
        if len(m) > 0:
            rank = m.index[0] + 1
            score = m.iloc[0]['overall_impact']
            status = "PASS" if rank <= target else "WARN"
            bi = int(m.iloc[0].get('bat_innings', 0))
            boi = int(m.iloc[0].get('bowl_innings', 0))
            print(f"    [{status}] {m.iloc[0]['player']:25s} Rank={rank:3d} OI={score:5.1f} BI={bi:3d} BoI={boi:3d}")

    # Batters with 50+ innings check
    batters_50 = all_players[all_players['bat_innings'] >= 50].sort_values(
        'batting_rating', ascending=False
    ).reset_index(drop=True)
    print(f"\n  Batters with 50+ innings: {len(batters_50)}")
    if len(batters_50) > 0:
        print("  Top 10 (batting_rating):")
        for i, (_, row) in enumerate(batters_50.head(10).iterrows()):
            print(f"    {i+1}. {row['player']:25s} BR={row['batting_rating']:5.1f} ({int(row['bat_innings'])} inn, {int(row['total_runs'])} runs)")

    return {
        'bat_impact': bat_impact,
        'bowl_impact': bowl_impact,
        'match_rankings': match_rankings,
        'all_players': all_players,
        'all_players_by_format': all_players_by_format,
        'partnerships': partnerships,
        'bowling_tandems': bowling_tandems,
        'rivalry_df': rivalry_df,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cricket Impact Metric Pipeline')
    parser.add_argument('--data_folder', type=str, default='./ipl_male_csv2',
                        help='Path to folder containing IPL CSVs')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain the ML model')
    parser.add_argument('--output_dir', type=str, default='cricket_impact',
                        help='Output directory for results')
    args = parser.parse_args()

    run_pipeline(args.data_folder, retrain=args.retrain, output_dir=args.output_dir)
