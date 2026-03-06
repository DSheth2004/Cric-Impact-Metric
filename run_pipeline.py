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

warnings.filterwarnings('ignore')

# Add parent dir to path so cricket_impact package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cricket_impact.data_loader import load_and_prepare
from cricket_impact.player_aggregation import (
    aggregate_batting, aggregate_bowling, compute_phase_performance,
    get_innings_totals, merge_runs_contribution
)
from cricket_impact.context_features import add_all_context_features
from cricket_impact.win_prob_model import train_model, load_model, predict_win_prob, FEATURES
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
                               match_format=None):
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

    # ── Batting-specific rating (for "batters with 50+ innings" validation) ──
    bat_mask = all_players['bat_innings'] >= 5
    bat_rated = all_players[bat_mask].copy()
    if len(bat_rated) > 0:
        br_runs = np.log1p(bat_rated['total_runs'])
        br_runs_norm = br_runs / br_runs.max() if br_runs.max() > 0 else 0

        br_impact = bat_rated['mean_bat_impact']
        br_impact_pct = br_impact.rank(pct=True)

        br_wpa = bat_rated['total_bat_wpa']
        br_wpa_pct = br_wpa.rank(pct=True)

        br_avg = bat_rated['true_bat_avg'].clip(0, 100)
        br_avg_norm = br_avg / 100.0

        br_sr = bat_rated['avg_bat_sr'].clip(0, 250)
        br_sr_norm = br_sr / 250.0

        br_not_out = bat_rated['not_out_pct'].clip(0, 1)

        br_longevity = np.log1p(bat_rated['bat_innings']) / np.log1p(bat_rated['bat_innings'].max()) if bat_rated['bat_innings'].max() > 0 else 0

        br_pom = bat_rated['pom_awards'] / bat_rated['pom_awards'].max() if bat_rated['pom_awards'].max() > 0 else 0

        br_pom_rate = np.where(
            bat_rated['bat_matches'] > 0,
            bat_rated['pom_awards'] / bat_rated['bat_matches'], 0
        )
        br_pom_rate_norm = br_pom_rate / br_pom_rate.max() if br_pom_rate.max() > 0 else 0

        br_max = bat_rated['max_bat_impact']
        br_max_pct = br_max.rank(pct=True)

        bat_rated['batting_rating_raw'] = (
            0.20 * br_impact_pct.values +
            0.15 * br_wpa_pct.values +
            0.12 * br_runs_norm.values +
            0.10 * br_avg_norm.values +
            0.07 * br_sr_norm.values +
            0.08 * br_not_out.values +
            0.10 * br_longevity +
            0.06 * br_pom +
            0.07 * br_pom_rate_norm +
            0.05 * br_max_pct.values
        )

        bayes_w_bat = bat_rated['bat_innings'] / (bat_rated['bat_innings'] + 10)
        prior_bat = bat_rated['batting_rating_raw'].median()
        bat_rated['batting_rating'] = bayes_w_bat * bat_rated['batting_rating_raw'] + (1 - bayes_w_bat) * prior_bat
        bat_rated['batting_rating'] = bat_rated['batting_rating'].rank(pct=True) * 100

        all_players = all_players.merge(
            bat_rated[['player', 'batting_rating']],
            on='player', how='left'
        )
    if 'batting_rating' not in all_players.columns:
        all_players['batting_rating'] = 0
    all_players['batting_rating'] = all_players['batting_rating'].fillna(0)

    # ── Overall Impact: balanced multi-factor scoring ──
    total_innings = all_players['bat_innings'] + all_players['bowl_innings']
    all_players['total_innings'] = total_innings

    # --- Component 1: Batting contribution score ---
    bat_contrib = np.zeros(len(all_players))
    bat_mask_any = all_players['bat_innings'] > 0
    if bat_mask_any.any():
        # Runs (log-scaled)
        runs_log = np.log1p(all_players['total_runs'])
        runs_norm = runs_log / runs_log.max() if runs_log.max() > 0 else 0
        # Batting average
        avg_norm = all_players['true_bat_avg'].clip(0, 80) / 80.0
        # Strike rate
        sr_norm = all_players['avg_bat_sr'].clip(0, 200) / 200.0
        # Batting WPA
        bat_wpa_pct = pd.Series(all_players['total_bat_wpa']).rank(pct=True).values
        # Batting impact
        bat_imp_pct = pd.Series(all_players['mean_bat_impact']).rank(pct=True).values

        bat_contrib = (
            0.30 * runs_norm +
            0.20 * avg_norm +
            0.15 * sr_norm +
            0.20 * bat_wpa_pct +
            0.15 * bat_imp_pct
        )
        bat_contrib = np.where(bat_mask_any, bat_contrib, 0)

    # --- Component 2: Bowling contribution score ---
    bowl_contrib = np.zeros(len(all_players))
    bowl_mask_any = all_players['bowl_innings'] > 0
    if bowl_mask_any.any():
        # Wickets (log-scaled)
        wkts_log = np.log1p(all_players['total_wickets'])
        wkts_norm = wkts_log / wkts_log.max() if wkts_log.max() > 0 else 0
        # Economy (inverted — lower is better)
        econ = all_players['avg_economy'].clip(4, 14)
        econ_norm = 1 - ((econ - 4) / 10.0)
        econ_norm = np.where(bowl_mask_any, econ_norm, 0)
        # Bowling WPA
        bowl_wpa_pct = pd.Series(all_players['total_bowl_wpa']).rank(pct=True).values
        # Bowling impact
        bowl_imp_pct = pd.Series(all_players['mean_bowl_impact']).rank(pct=True).values
        # Phase difficulty: death/powerplay bowling contribution
        phase_diff = all_players.get('avg_phase_difficulty', pd.Series(0, index=all_players.index)).fillna(0)
        phase_diff_pct = pd.Series(phase_diff).rank(pct=True).values

        bowl_contrib = (
            0.22 * wkts_norm +
            0.15 * econ_norm +
            0.22 * bowl_wpa_pct +
            0.21 * bowl_imp_pct +
            0.20 * phase_diff_pct    # death/powerplay phase difficulty
        )
        bowl_contrib = np.where(bowl_mask_any, bowl_contrib, 0)

    # --- Component 3: All-rounder bonus ---
    # Players who genuinely contribute in both departments
    is_allrounder = (all_players['bat_innings'] >= 20) & (all_players['bowl_innings'] >= 20)
    allrounder_bonus = np.zeros(len(all_players))
    if is_allrounder.any():
        # Dual contribution: geometric mean of batting and bowling contrib
        dual = np.sqrt(np.maximum(bat_contrib, 0) * np.maximum(bowl_contrib, 0))
        allrounder_bonus = np.where(is_allrounder, dual, 0)

    # --- Component 4: Longevity ---
    total_matches = np.maximum(
        all_players['bat_matches'].fillna(0),
        all_players['bowl_matches'].fillna(0)
    )
    longevity = np.log1p(total_matches)
    longevity_norm = longevity / longevity.max() if longevity.max() > 0 else 0

    # --- Component 5: POM awards ---
    pom_norm = all_players['pom_awards'] / all_players['pom_awards'].max() if all_players['pom_awards'].max() > 0 else 0

    # POM rate (awards per match — rewards efficiency)
    pom_rate = np.where(total_matches > 0, all_players['pom_awards'] / total_matches, 0)
    pom_rate_norm = pom_rate / pom_rate.max() if pom_rate.max() > 0 else 0

    # --- Component 6: Clutch contribution ---
    total_clutch = all_players.get('total_clutch_bat', pd.Series(0, index=all_players.index)).fillna(0).abs() + \
                   all_players.get('total_clutch_bowl', pd.Series(0, index=all_players.index)).fillna(0).abs()
    clutch_pct = pd.Series(total_clutch).rank(pct=True).values

    # --- Weighted role-based contribution ---
    # Weight batting vs bowling by relative innings (so pure batters use batting, pure bowlers use bowling)
    bat_weight = np.where(total_innings > 0, all_players['bat_innings'] / total_innings, 0.5)
    bowl_weight = np.where(total_innings > 0, all_players['bowl_innings'] / total_innings, 0.5)

    role_score = bat_weight * bat_contrib + bowl_weight * bowl_contrib

    # --- Component 7: Win bonus (for all players, not just captains) ---
    win_bonus = all_players['win_bonus_norm'].values

    # --- Component 8: Captaincy bonus (only for captains, 0 for non-captains) ---
    captaincy_bonus = all_players['captaincy_score_norm'].values

    # --- Component 9: Playoff seasons bonus (for all players on playoff teams) ---
    playoff_raw = all_players['playoff_seasons']
    playoff_norm = playoff_raw / playoff_raw.max() if playoff_raw.max() > 0 else 0

    # --- Final composite ---
    all_players['overall_impact_raw'] = (
        0.28 * role_score +              # Core performance (role-weighted)
        0.10 * allrounder_bonus +         # All-rounder dual contribution
        0.12 * longevity_norm +           # Career longevity / consistency
        0.10 * pom_norm +                 # Match-winning ability
        0.06 * pom_rate_norm +            # Match-winning efficiency
        0.06 * clutch_pct +              # Clutch performance
        0.07 * np.where(bat_mask_any, runs_norm, wkts_norm) +  # Volume
        0.08 * win_bonus +               # Winning team bonus (all players)
        0.05 * captaincy_bonus +          # Captain-only bonus (0 for non-captains)
        0.04 * playoff_norm +             # Playoff team bonus
        0.04 * np.maximum(              # Bowling visibility boost
            pd.Series(all_players['mean_bowl_impact']).rank(pct=True).fillna(0).values
            * np.where(bowl_mask_any, 1, 0), 0)
    )

    # Bayesian dampening — sample size correction
    min_innings = 12
    bayes_w = total_innings / (total_innings + min_innings)
    prior = all_players['overall_impact_raw'].median() if len(all_players) > 0 else 0.35
    all_players['overall_impact_dampened'] = bayes_w * all_players['overall_impact_raw'] + (1 - bayes_w) * prior

    # Scale to 0-90 with spread — logistic curve so gauge meter displays well
    # Median player = 45, top player ≈ 88-90, bottom ≈ 2-5
    if len(all_players) > 1:
        pctile = all_players['overall_impact_dampened'].rank(pct=True)
        # Use power curve for better spread: score = 90 * pctile^0.7
        # This gives top=90, median≈55, bottom near 0
        all_players['overall_impact'] = 90.0 * (pctile ** 0.7)
    else:
        all_players['overall_impact'] = 45.0

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
    print("STEP 5: ML Win Probability Model")
    print("=" * 60)
    model = None
    if not retrain:
        model = load_model(model_dir)
        if model is not None:
            print("  Loaded saved model")

    if model is None:
        print("  Training new model...")
        model, metrics = train_model(ball_df, info_df, model_dir)

    # ── Step 6: Predict win_prob ──
    print("\n" + "=" * 60)
    print("STEP 6: Predicting win probabilities")
    print("=" * 60)
    ball_df = predict_win_prob(ball_df, model)
    print(f"  Win prob range: [{ball_df['win_prob'].min():.3f}, {ball_df['win_prob'].max():.3f}]")

    # ── Step 7: WPA ──
    print("\n" + "=" * 60)
    print("STEP 7: Computing WPA")
    print("=" * 60)
    ball_df = compute_per_ball_wpa(ball_df)
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
                                                 match_format=fmt)
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
