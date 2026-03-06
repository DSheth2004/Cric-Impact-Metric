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

    # Compute max balls per match from actual overs bowled
    match_max_over = ball_df.groupby('match_id')['over'].max().reset_index()
    match_max_over.columns = ['match_id', '_max_over']
    ball_df = ball_df.merge(match_max_over, on='match_id', how='left')
    ball_df['_max_over'] = ball_df['_max_over'].fillna(20)
    ball_df['_total_balls'] = (ball_df['_max_over'] + 1) * 6  # overs are 0-indexed

    # Required run rate
    ball_df['runs_needed'] = np.where(
        ball_df['innings'] == 2,
        ball_df['target'] - ball_df['cumulative_runs'],
        0
    )
    ball_df['balls_remaining'] = np.maximum(
        ball_df['_total_balls'] - ball_df['cumulative_balls'], 1
    )
    ball_df['overs_remaining'] = ball_df['balls_remaining'] / 6.0
    ball_df['required_rr'] = np.where(
        (ball_df['innings'] == 2) & (ball_df['overs_remaining'] > 0),
        ball_df['runs_needed'] / ball_df['overs_remaining'],
        0
    )
    ball_df['required_rr'] = ball_df['required_rr'].clip(lower=0)

    # Cap required_rr based on format to avoid distorted pressure
    fmt = ball_df.get('match_format', pd.Series('T20', index=ball_df.index))
    is_test = fmt == 'Test'
    is_odi = fmt == 'ODI'
    # T20: cap at 15 when > 4 overs remain; ODI: cap at 9 when > 20 overs remain
    # Test: drop required_rr entirely (most Test innings aren't chasing)
    ball_df['required_rr'] = np.where(
        is_test, 0,
        np.where(
            is_odi & (ball_df['overs_remaining'] > 20),
            ball_df['required_rr'].clip(upper=9),
            np.where(
                (~is_odi) & (~is_test) & (ball_df['overs_remaining'] > 4),
                ball_df['required_rr'].clip(upper=15),
                ball_df['required_rr']
            )
        )
    )

    ball_df.drop(columns=['_max_over', '_total_balls'], inplace=True)

    ball_df['wickets_remaining'] = 10 - ball_df['cumulative_wickets']
    ball_df['rr_pressure_gap'] = np.where(
        ball_df['innings'] == 2,
        ball_df['required_rr'] - ball_df['current_rr'],
        0
    )

    return ball_df


def add_pressure_index(ball_df):
    """Compute pressure_index for each ball — format-aware."""
    overs_remaining_inv = np.where(
        ball_df['overs_remaining'] > 0,
        1.0 / ball_df['overs_remaining'],
        1.0
    )

    fmt = ball_df.get('match_format', pd.Series('T20', index=ball_df.index))
    is_test = fmt == 'Test'
    is_odi = fmt == 'ODI'

    # Format-aware required_rr cap for normalization
    # T20: max meaningful RRR ~15, ODI: ~9, Test: 0 (pressure from wickets)
    rrr_cap = np.where(is_test, 1, np.where(is_odi, 9, 15)).astype(float)
    rrr_norm = np.where(rrr_cap > 0,
                        ball_df['required_rr'].clip(0, rrr_cap) / rrr_cap, 0)
    wick_norm = (10 - ball_df['wickets_remaining']) / 10.0
    ov_inv_norm = np.clip(overs_remaining_inv, 0, 1)

    # Test: pressure dominated by wickets (no chase RRR usually)
    # ODI/T20: standard formula
    ball_df['pressure_index'] = np.where(
        is_test,
        0.6 * wick_norm + 0.4 * ov_inv_norm,
        0.4 * rrr_norm + 0.3 * wick_norm + 0.3 * ov_inv_norm
    )

    # For innings 1, use a simpler formula (no chase pressure)
    mask_inn1 = ball_df['innings'] == 1
    ball_df.loc[mask_inn1, 'pressure_index'] = (
        0.3 * wick_norm[mask_inn1] + 0.3 * ov_inv_norm[mask_inn1]
    )

    return ball_df


def add_clutch_indicator(ball_df):
    """Add clutch flags and factor — dynamic based on match length."""
    # Compute max over per match (0-indexed)
    match_max = ball_df.groupby('match_id')['over'].max().reset_index()
    match_max.columns = ['match_id', '_match_max_over']
    ball_df = ball_df.merge(match_max, on='match_id', how='left')
    ball_df['_match_max_over'] = ball_df['_match_max_over'].fillna(19)
    total_overs = ball_df['_match_max_over'] + 1  # 0-indexed to count

    # For matches <= 50 overs: clutch starts at last 20% of overs
    # For matches > 50 overs (Tests): no fixed clutch zone — handled via WPA swing
    clutch_start = np.floor(total_overs * 0.80).astype(int)
    is_long_match = total_overs > 50

    # Standard clutch: last 20% of overs, factor ramps 0→1 over that zone
    clutch_zone_size = np.maximum(total_overs - clutch_start, 1)
    ball_df['is_clutch'] = np.where(
        is_long_match, 0,
        (ball_df['over'] >= clutch_start).astype(int)
    )
    ball_df['clutch_factor'] = np.where(
        is_long_match, 0,
        np.where(
            ball_df['over'] >= clutch_start,
            np.clip((ball_df['over'] - clutch_start + 1) / clutch_zone_size, 0, 1),
            0
        )
    )

    ball_df.drop(columns=['_match_max_over'], inplace=True)
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
