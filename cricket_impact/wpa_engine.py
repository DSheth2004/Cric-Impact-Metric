"""
Module 5: WPA Engine
Per-ball Win Probability Added, batter/bowler aggregation, clutch/leverage WPA.
"""

import pandas as pd
import numpy as np


def compute_per_ball_wpa(ball_df):
    """Compute per-ball WPA as diff of win_prob within each match-innings."""
    df = ball_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over']).reset_index(drop=True)
    df['wpa'] = df.groupby(['match_id', 'innings'])['win_prob'].diff().fillna(0)
    return df


def aggregate_batter_wpa(ball_df):
    """Aggregate WPA for batters per match-innings."""
    df = ball_df.copy()

    # Core WPA aggregation per striker
    batter_wpa = df.groupby(['match_id', 'innings', 'striker']).agg(
        total_wpa=('wpa', 'sum'),
        avg_wpa=('wpa', 'mean'),
        max_wpa=('wpa', 'max'),
        min_wpa=('wpa', 'min'),
        wpa_volatility=('wpa', 'std'),
        wpa_count=('wpa', 'count'),
    ).reset_index()

    # Positive WPA percentage
    pos_wpa = df[df['wpa'] > 0].groupby(['match_id', 'innings', 'striker']).size().reset_index(name='pos_wpa_balls')
    batter_wpa = batter_wpa.merge(pos_wpa, on=['match_id', 'innings', 'striker'], how='left')
    batter_wpa['pos_wpa_balls'] = batter_wpa['pos_wpa_balls'].fillna(0)
    batter_wpa['positive_wpa_pct'] = np.where(
        batter_wpa['wpa_count'] > 0,
        batter_wpa['pos_wpa_balls'] / batter_wpa['wpa_count'], 0
    )

    # Leverage WPA
    if 'leverage_index' in df.columns:
        df['leverage_wpa_val'] = df['wpa'] * df['leverage_index']
        lev_wpa = df.groupby(['match_id', 'innings', 'striker'])['leverage_wpa_val'].sum().reset_index()
        lev_wpa.columns = ['match_id', 'innings', 'striker', 'leverage_wpa']
        batter_wpa = batter_wpa.merge(lev_wpa, on=['match_id', 'innings', 'striker'], how='left')
    else:
        batter_wpa['leverage_wpa'] = batter_wpa['total_wpa']

    batter_wpa['leverage_wpa'] = batter_wpa['leverage_wpa'].fillna(0)
    batter_wpa['wpa_volatility'] = batter_wpa['wpa_volatility'].fillna(0)

    # Consistency WPA
    batter_wpa['consistency_wpa'] = batter_wpa['positive_wpa_pct'] * (1 - batter_wpa['wpa_volatility'].clip(0, 1))

    batter_wpa.rename(columns={'striker': 'player'}, inplace=True)
    return batter_wpa


def compute_batter_presence_impact(ball_df):
    """Compute wp_at_arrival, wp_at_departure, presence_impact per batter."""
    df = ball_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over']).reset_index(drop=True)

    # First and last ball per striker per innings
    first_ball = df.groupby(['match_id', 'innings', 'striker'])['win_prob'].first().reset_index()
    first_ball.columns = ['match_id', 'innings', 'striker', 'wp_at_arrival']

    last_ball = df.groupby(['match_id', 'innings', 'striker'])['win_prob'].last().reset_index()
    last_ball.columns = ['match_id', 'innings', 'striker', 'wp_at_departure']

    presence = first_ball.merge(last_ball, on=['match_id', 'innings', 'striker'])
    presence['presence_impact'] = presence['wp_at_departure'] - presence['wp_at_arrival']
    presence.rename(columns={'striker': 'player'}, inplace=True)

    return presence[['match_id', 'innings', 'player', 'wp_at_arrival', 'wp_at_departure', 'presence_impact']]


def aggregate_bowler_wpa(ball_df):
    """Aggregate WPA for bowlers per match-innings (negated: positive = good for bowler)."""
    df = ball_df.copy()

    bowler_wpa = df.groupby(['match_id', 'innings', 'bowler']).agg(
        raw_total_wpa=('wpa', 'sum'),
        raw_avg_wpa=('wpa', 'mean'),
        wpa_count=('wpa', 'count'),
    ).reset_index()

    # Negate for bowler perspective
    bowler_wpa['total_wpa'] = -bowler_wpa['raw_total_wpa']
    bowler_wpa['avg_wpa'] = -bowler_wpa['raw_avg_wpa']

    # Wicket WPA
    wicket_balls = df[df['is_wicket'] == 1].copy()
    if len(wicket_balls) > 0:
        wicket_wpa = wicket_balls.groupby(['match_id', 'innings', 'bowler'])['wpa'].sum().reset_index()
        wicket_wpa.columns = ['match_id', 'innings', 'bowler', 'wicket_wpa_raw']
        wicket_wpa['wicket_wpa'] = -wicket_wpa['wicket_wpa_raw']
        bowler_wpa = bowler_wpa.merge(wicket_wpa[['match_id', 'innings', 'bowler', 'wicket_wpa']],
                                       on=['match_id', 'innings', 'bowler'], how='left')
    else:
        bowler_wpa['wicket_wpa'] = 0

    bowler_wpa['wicket_wpa'] = bowler_wpa['wicket_wpa'].fillna(0)

    # Leverage WPA for bowler
    if 'leverage_index' in df.columns:
        df['bowler_lev_wpa'] = -df['wpa'] * df['leverage_index']
        lev_wpa = df.groupby(['match_id', 'innings', 'bowler'])['bowler_lev_wpa'].sum().reset_index()
        lev_wpa.columns = ['match_id', 'innings', 'bowler', 'leverage_wpa']
        bowler_wpa = bowler_wpa.merge(lev_wpa, on=['match_id', 'innings', 'bowler'], how='left')
    else:
        bowler_wpa['leverage_wpa'] = bowler_wpa['total_wpa']

    bowler_wpa['leverage_wpa'] = bowler_wpa['leverage_wpa'].fillna(0)

    # Spell efficiency
    bowler_wpa['spell_efficiency'] = np.where(
        bowler_wpa['wpa_count'] > 0,
        bowler_wpa['total_wpa'] / bowler_wpa['wpa_count'], 0
    )

    # Economy WPA: dot ball contribution
    dot_balls = df[df['is_dot'] == 1].copy()
    if len(dot_balls) > 0:
        dot_wpa = dot_balls.groupby(['match_id', 'innings', 'bowler'])['wpa'].sum().reset_index()
        dot_wpa.columns = ['match_id', 'innings', 'bowler', 'economy_wpa']
        dot_wpa['economy_wpa'] = -dot_wpa['economy_wpa']
        bowler_wpa = bowler_wpa.merge(dot_wpa, on=['match_id', 'innings', 'bowler'], how='left')
    else:
        bowler_wpa['economy_wpa'] = 0

    bowler_wpa['economy_wpa'] = bowler_wpa['economy_wpa'].fillna(0)

    bowler_wpa.rename(columns={'bowler': 'player'}, inplace=True)
    bowler_wpa.drop(columns=['raw_total_wpa', 'raw_avg_wpa'], inplace=True)

    return bowler_wpa


def compute_clutch_wpa(ball_df):
    """Compute clutch WPA for batters and bowlers — format-aware."""
    # Compute max over per match for dynamic clutch threshold
    match_max = ball_df.groupby('match_id')['over'].max()
    ball_df = ball_df.copy()
    ball_df['_match_total_overs'] = ball_df['match_id'].map(match_max) + 1  # 0-indexed
    ball_df['_match_total_overs'] = ball_df['_match_total_overs'].fillna(20)

    is_long_match = ball_df['_match_total_overs'] > 50

    # For ≤50 over matches: clutch = last 20% of overs with high pressure
    clutch_start = np.floor(ball_df['_match_total_overs'] * 0.80).astype(int)
    standard_clutch_mask = (
        (~is_long_match) &
        (ball_df['over'] >= clutch_start) &
        (
            (ball_df['pressure_index'] > 0.6) |
            (ball_df['wickets_remaining'] <= 3) |
            (ball_df['rr_pressure_gap'] > 3)
        )
    )

    # For >50 over matches (Tests): clutch = 10-over windows where a player's
    # WPA swings drastically (sustained impact), based on rolling WPA magnitude
    long_clutch_mask = pd.Series(False, index=ball_df.index)
    if is_long_match.any():
        long_df = ball_df[is_long_match].copy()
        # Rolling 60-ball (10 over) WPA sum per batter and bowler
        long_df = long_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over'])
        batter_roll = long_df.groupby(['match_id', 'innings', 'striker'])['wpa'].transform(
            lambda x: x.rolling(60, min_periods=6).sum().abs()
        ).fillna(0)
        bowler_roll = long_df.groupby(['match_id', 'innings', 'bowler'])['wpa'].transform(
            lambda x: x.rolling(60, min_periods=6).sum().abs()
        ).fillna(0)
        # Clutch if either batter or bowler rolling WPA exceeds threshold
        wpa_threshold = 0.15
        long_clutch_mask.loc[long_df.index] = (
            (batter_roll > wpa_threshold) | (bowler_roll > wpa_threshold)
        )

    clutch_mask = standard_clutch_mask | long_clutch_mask
    clutch_df = ball_df[clutch_mask].copy()

    # Batter clutch WPA
    if len(clutch_df) > 0:
        batter_clutch = clutch_df.groupby(['match_id', 'innings', 'striker'])['wpa'].sum().reset_index()
        batter_clutch.columns = ['match_id', 'innings', 'player', 'clutch_wpa']

        bowler_clutch = clutch_df.groupby(['match_id', 'innings', 'bowler'])['wpa'].sum().reset_index()
        bowler_clutch.columns = ['match_id', 'innings', 'player', 'clutch_wpa']
        bowler_clutch['clutch_wpa'] = -bowler_clutch['clutch_wpa']
    else:
        batter_clutch = pd.DataFrame(columns=['match_id', 'innings', 'player', 'clutch_wpa'])
        bowler_clutch = pd.DataFrame(columns=['match_id', 'innings', 'player', 'clutch_wpa'])

    ball_df.drop(columns=['_match_total_overs'], inplace=True, errors='ignore')
    return batter_clutch, bowler_clutch
