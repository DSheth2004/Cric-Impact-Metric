"""
Module 3: Context Feature Engineering
Venue, opposition, H2H, ball-level context, pressure, leverage, momentum.
"""

import pandas as pd
import numpy as np


def add_venue_features(ball_df, info_df):
    """Add venue-level scoring features."""
    # Compute innings totals per match-innings
    inn_totals = ball_df.groupby(['match_id', 'innings']).agg(
        inn_runs=('total_runs', 'sum'),
        inn_balls=('is_legal', 'sum'),
    ).reset_index()
    inn_totals['inn_overs'] = inn_totals['inn_balls'] / 6.0
    inn_totals['inn_rpo'] = np.where(inn_totals['inn_overs'] > 0,
                                     inn_totals['inn_runs'] / inn_totals['inn_overs'], 0)

    # Get venue per match
    venue_map = ball_df.groupby('match_id')['venue'].first().reset_index()
    inn_totals = inn_totals.merge(venue_map, on='match_id', how='left')

    # Venue stats
    venue_stats = inn_totals.groupby('venue').agg(
        venue_rpo=('inn_rpo', 'mean'),
        venue_avg_score=('inn_runs', 'mean'),
        venue_matches=('match_id', 'nunique'),
    ).reset_index()

    # Normalize venue_rpo to scoring index (0-1)
    rpo_min = venue_stats['venue_rpo'].min()
    rpo_max = venue_stats['venue_rpo'].max()
    rpo_range = rpo_max - rpo_min if rpo_max > rpo_min else 1.0
    venue_stats['venue_scoring_index'] = (venue_stats['venue_rpo'] - rpo_min) / rpo_range
    venue_stats['venue_difficulty'] = 1.0 - venue_stats['venue_scoring_index']

    # Chase success rate
    if info_df is not None and 'winner_wickets' in info_df.columns:
        chase_df = info_df[info_df['winner'] != ''].copy()
        chase_df['is_chase_win'] = (chase_df['winner_wickets'] > 0).astype(int)
        venue_chase_map = ball_df.groupby('match_id')['venue'].first().reset_index()
        chase_df = chase_df.merge(venue_chase_map, on='match_id', how='left')
        if 'venue' in chase_df.columns:
            chase_rates = chase_df.groupby('venue')['is_chase_win'].mean().reset_index()
            chase_rates.columns = ['venue', 'venue_chase_success_rate']
            venue_stats = venue_stats.merge(chase_rates, on='venue', how='left')
    if 'venue_chase_success_rate' not in venue_stats.columns:
        venue_stats['venue_chase_success_rate'] = 0.5

    venue_stats['venue_chase_success_rate'] = venue_stats['venue_chase_success_rate'].fillna(0.5)

    ball_df = ball_df.merge(
        venue_stats[['venue', 'venue_rpo', 'venue_scoring_index', 'venue_difficulty',
                      'venue_avg_score', 'venue_chase_success_rate']],
        on='venue', how='left'
    )
    for c in ['venue_rpo', 'venue_scoring_index', 'venue_difficulty',
              'venue_avg_score', 'venue_chase_success_rate']:
        ball_df[c] = ball_df[c].fillna(ball_df[c].median() if ball_df[c].notna().any() else 0)

    return ball_df, venue_stats


def add_opposition_strength(ball_df):
    """Per-team batting and bowling strength from innings totals."""
    inn = ball_df.groupby(['match_id', 'innings', 'batting_team']).agg(
        inn_runs=('total_runs', 'sum'),
        inn_balls=('is_legal', 'sum'),
        inn_wickets=('is_wicket', 'sum'),
    ).reset_index()
    inn['inn_rpo'] = np.where(inn['inn_balls'] > 0,
                              inn['inn_runs'] / (inn['inn_balls'] / 6.0), 0)

    team_bat = inn.groupby('batting_team').agg(
        team_bat_avg_rpo=('inn_rpo', 'mean'),
        team_bat_avg_score=('inn_runs', 'mean'),
    ).reset_index().rename(columns={'batting_team': 'team'})

    # Bowling strength (what teams concede when bowling)
    inn2 = ball_df.groupby(['match_id', 'innings', 'bowling_team']).agg(
        conc_runs=('total_runs', 'sum'),
        conc_balls=('is_legal', 'sum'),
    ).reset_index()
    inn2['conc_rpo'] = np.where(inn2['conc_balls'] > 0,
                                inn2['conc_runs'] / (inn2['conc_balls'] / 6.0), 0)
    team_bowl = inn2.groupby('bowling_team').agg(
        team_bowl_avg_rpo=('conc_rpo', 'mean'),
    ).reset_index().rename(columns={'bowling_team': 'team'})

    team_strength = team_bat.merge(team_bowl, on='team', how='outer')
    for c in ['team_bat_avg_rpo', 'team_bat_avg_score', 'team_bowl_avg_rpo']:
        team_strength[c] = team_strength[c].fillna(team_strength[c].median() if team_strength[c].notna().any() else 7.0)

    # Merge opposition strength for batting team (the bowling_team's strength)
    ball_df = ball_df.merge(
        team_strength.rename(columns={
            'team': 'bowling_team',
            'team_bowl_avg_rpo': 'opp_bowl_strength',
            'team_bat_avg_rpo': 'opp_bat_strength',
            'team_bat_avg_score': 'opp_bat_avg_score',
        }),
        on='bowling_team', how='left'
    )
    for c in ['opp_bowl_strength', 'opp_bat_strength', 'opp_bat_avg_score']:
        ball_df[c] = ball_df[c].fillna(7.0)

    return ball_df, team_strength


def add_ball_level_context(ball_df):
    """Add cumulative per match-innings context features."""
    ball_df = ball_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over']).reset_index(drop=True)

    # Cumulative stats within each match-innings
    grp = ball_df.groupby(['match_id', 'innings'])
    ball_df['cumulative_runs'] = grp['total_runs'].cumsum()
    ball_df['cumulative_wickets'] = grp['is_wicket'].cumsum()
    ball_df['cumulative_balls'] = grp['is_legal'].cumsum()

    ball_df['current_rr'] = np.where(
        ball_df['cumulative_balls'] > 0,
        ball_df['cumulative_runs'] / (ball_df['cumulative_balls'] / 6.0),
        0
    )

    # Target: innings-1 total + 1 for innings 2
    inn1_totals = ball_df[ball_df['innings'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    inn1_totals.columns = ['match_id', 'innings_1_total']
    ball_df = ball_df.merge(inn1_totals, on='match_id', how='left')
    ball_df['innings_1_total'] = ball_df['innings_1_total'].fillna(0)

    ball_df['target'] = np.where(ball_df['innings'] == 2,
                                 ball_df['innings_1_total'] + 1, 0)
    ball_df['is_chasing'] = (ball_df['innings'] == 2).astype(int)

    # Required run rate
    ball_df['runs_needed'] = np.where(
        ball_df['innings'] == 2,
        ball_df['target'] - ball_df['cumulative_runs'],
        0
    )
    ball_df['balls_remaining'] = np.where(
        ball_df['innings'] == 2,
        np.maximum(120 - ball_df['cumulative_balls'], 1),
        np.maximum(120 - ball_df['cumulative_balls'], 1)
    )
    ball_df['overs_remaining'] = ball_df['balls_remaining'] / 6.0
    ball_df['required_rr'] = np.where(
        (ball_df['innings'] == 2) & (ball_df['overs_remaining'] > 0),
        ball_df['runs_needed'] / ball_df['overs_remaining'],
        0
    )
    ball_df['required_rr'] = ball_df['required_rr'].clip(lower=0)

    ball_df['wickets_remaining'] = 10 - ball_df['cumulative_wickets']
    ball_df['rr_pressure_gap'] = np.where(
        ball_df['innings'] == 2,
        ball_df['required_rr'] - ball_df['current_rr'],
        0
    )

    return ball_df


def add_pressure_index(ball_df):
    """Compute pressure_index for each ball."""
    overs_remaining_inv = np.where(
        ball_df['overs_remaining'] > 0,
        1.0 / ball_df['overs_remaining'],
        1.0
    )
    # Normalize components
    rrr_norm = ball_df['required_rr'].clip(0, 36) / 36.0
    wick_norm = (10 - ball_df['wickets_remaining']) / 10.0
    ov_inv_norm = np.clip(overs_remaining_inv, 0, 1)

    ball_df['pressure_index'] = (0.4 * rrr_norm + 0.3 * wick_norm + 0.3 * ov_inv_norm)

    # For innings 1, use a simpler formula
    mask_inn1 = ball_df['innings'] == 1
    ball_df.loc[mask_inn1, 'pressure_index'] = (
        0.3 * wick_norm[mask_inn1] + 0.3 * ov_inv_norm[mask_inn1]
    )

    return ball_df


def add_clutch_indicator(ball_df):
    """Add clutch flags and factor."""
    ball_df['is_clutch'] = (ball_df['over'] >= 15).astype(int)
    ball_df['clutch_factor'] = np.where(
        ball_df['over'] >= 15,
        (ball_df['over'] - 14) / 5.0,  # 0.2 at over 15, 1.0 at over 19
        0
    )
    return ball_df


def add_leverage_index(ball_df):
    """Compute leverage_index."""
    rr_ratio = np.where(
        ball_df['current_rr'] > 0,
        ball_df['required_rr'] / ball_df['current_rr'],
        1.0
    )
    wick_frac = (10 - ball_df['wickets_remaining']) / 10.0
    ball_df['leverage_index'] = np.clip(rr_ratio + wick_frac, 0, 3)

    # Innings 1: heuristic leverage based on phase and wickets
    mask_inn1 = ball_df['innings'] == 1
    ball_df.loc[mask_inn1, 'leverage_index'] = np.clip(
        0.5 + wick_frac[mask_inn1] + ball_df.loc[mask_inn1, 'clutch_factor'] * 0.5,
        0, 3
    )
    return ball_df


def add_momentum_shift(ball_df):
    """12-ball rolling window tracking required_rr changes."""
    ball_df = ball_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over']).reset_index(drop=True)
    ball_df['momentum_shift'] = ball_df.groupby(['match_id', 'innings'])['required_rr'].transform(
        lambda x: x.diff().rolling(window=12, min_periods=1).mean()
    ).fillna(0)
    return ball_df


def add_all_context_features(ball_df, info_df):
    """Chain all context feature engineering."""
    print("  Adding venue features...")
    ball_df, venue_stats = add_venue_features(ball_df, info_df)
    print("  Adding opposition strength...")
    ball_df, team_strength = add_opposition_strength(ball_df)
    print("  Adding ball-level context...")
    ball_df = add_ball_level_context(ball_df)
    print("  Adding pressure index...")
    ball_df = add_pressure_index(ball_df)
    print("  Adding clutch indicator...")
    ball_df = add_clutch_indicator(ball_df)
    print("  Adding leverage index...")
    ball_df = add_leverage_index(ball_df)
    print("  Adding momentum shift...")
    ball_df = add_momentum_shift(ball_df)
    print("  Context features complete.")
    return ball_df
