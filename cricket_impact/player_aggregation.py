"""
Module 2: Player-Match Aggregation
Batting, bowling, and phase-level stats per player per match-innings.
"""

import pandas as pd
import numpy as np


def derive_batting_position(ball_df):
    """Derive batting position from ball-by-ball data.
    For each match-innings, track the order in which batters first appear."""
    df = ball_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over']).copy()
    positions = []
    for (mid, inn), grp in df.groupby(['match_id', 'innings']):
        seen = {}
        pos = 1
        for _, row in grp.iterrows():
            striker = row['striker']
            non_striker = row['non_striker']
            if striker not in seen:
                seen[striker] = pos
                pos += 1
            if non_striker not in seen:
                seen[non_striker] = pos
                pos += 1
        for player, bat_pos in seen.items():
            positions.append({
                'match_id': mid, 'innings': inn,
                'striker': player, 'batting_position': min(bat_pos, 11)
            })
    return pd.DataFrame(positions)


def aggregate_batting(ball_df):
    """Aggregate batting stats per player per match-innings."""
    # Filter to legal balls and all balls for different stats
    df = ball_df.copy()

    # Ensure match_format column exists
    if 'match_format' not in df.columns:
        df['match_format'] = 'T20'

    agg = df.groupby(['match_id', 'innings', 'striker', 'batting_team',
                       'bowling_team', 'season', 'venue', 'start_date',
                       'match_format']).agg(
        runs=('runs_off_bat', 'sum'),
        balls_faced=('is_legal', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots_faced=('is_dot', 'sum'),
        singles=('runs_off_bat', lambda x: (x == 1).sum()),
        doubles=('runs_off_bat', lambda x: (x == 2).sum()),
        threes=('runs_off_bat', lambda x: (x == 3).sum()),
        total_deliveries=('runs_off_bat', 'count'),
    ).reset_index()

    # Derived stats
    agg['strike_rate'] = np.where(agg['balls_faced'] > 0,
                                  (agg['runs'] / agg['balls_faced']) * 100, 0)
    agg['boundary_pct'] = np.where(agg['balls_faced'] > 0,
                                   ((agg['fours'] + agg['sixes']) / agg['balls_faced']) * 100, 0)
    agg['dot_pct_faced'] = np.where(agg['balls_faced'] > 0,
                                    (agg['dots_faced'] / agg['balls_faced']) * 100, 0)
    agg['rotation_balls'] = agg['singles'] + agg['doubles'] + agg['threes']
    agg['rotation_pct'] = np.where(agg['balls_faced'] > 0,
                                   (agg['rotation_balls'] / agg['balls_faced']) * 100, 0)
    agg['rotation_runs'] = agg['singles'] + agg['doubles'] * 2 + agg['threes'] * 3
    agg['boundary_runs'] = agg['fours'] * 4 + agg['sixes'] * 6

    # Dismissal info
    dismissals = df[df['player_dismissed'] != ''].groupby(
        ['match_id', 'innings', 'player_dismissed']
    ).size().reset_index(name='was_dismissed')
    dismissals['was_dismissed'] = 1
    dismissals.rename(columns={'player_dismissed': 'striker'}, inplace=True)

    agg = agg.merge(dismissals[['match_id', 'innings', 'striker', 'was_dismissed']],
                     on=['match_id', 'innings', 'striker'], how='left')
    agg['was_dismissed'] = agg['was_dismissed'].fillna(0).astype(int)

    # Derive and merge batting position
    bat_pos_df = derive_batting_position(ball_df)
    agg = agg.merge(bat_pos_df, on=['match_id', 'innings', 'striker'], how='left')
    agg['batting_position'] = agg['batting_position'].fillna(6).astype(int)

    # Rename
    agg.rename(columns={'striker': 'player', 'batting_team': 'player_team',
                         'bowling_team': 'opposition'}, inplace=True)
    agg['role'] = 'batter'

    return agg


def aggregate_bowling(ball_df):
    """Aggregate bowling stats per bowler per match-innings."""
    df = ball_df.copy()

    # Flag difficult phases: death (overs 16-19) and powerplay (overs 0-5)
    df['is_death'] = (df['over'] >= 16).astype(int)
    df['is_powerplay'] = (df['over'] < 6).astype(int)
    df['death_legal'] = df['is_death'] * df['is_legal']
    df['death_wicket'] = df['is_death'] * df['is_wicket']
    df['death_runs'] = df['is_death'] * df['total_runs']
    df['pp_legal'] = df['is_powerplay'] * df['is_legal']
    df['pp_wicket'] = df['is_powerplay'] * df['is_wicket']
    df['pp_runs'] = df['is_powerplay'] * df['total_runs']

    # Ensure match_format column exists
    if 'match_format' not in df.columns:
        df['match_format'] = 'T20'

    agg = df.groupby(['match_id', 'innings', 'bowler', 'batting_team',
                       'bowling_team', 'season', 'venue', 'start_date',
                       'match_format']).agg(
        runs_conceded=('total_runs', 'sum'),
        balls_bowled=('is_legal', 'sum'),
        wickets=('is_wicket', 'sum'),
        dots_bowled=('is_dot', 'sum'),
        fours_conceded=('is_four', 'sum'),
        sixes_conceded=('is_six', 'sum'),
        wides_given=('wides', lambda x: (x > 0).sum()),
        noballs_given=('noballs', lambda x: (x > 0).sum()),
        total_deliveries=('total_runs', 'count'),
        death_balls=('death_legal', 'sum'),
        death_wickets=('death_wicket', 'sum'),
        death_runs_conceded=('death_runs', 'sum'),
        pp_balls=('pp_legal', 'sum'),
        pp_wickets=('pp_wicket', 'sum'),
        pp_runs_conceded=('pp_runs', 'sum'),
    ).reset_index()

    # Derived stats
    agg['overs_bowled'] = agg['balls_bowled'] / 6.0
    agg['economy'] = np.where(agg['overs_bowled'] > 0,
                              agg['runs_conceded'] / agg['overs_bowled'], 0)
    agg['dot_pct'] = np.where(agg['balls_bowled'] > 0,
                              (agg['dots_bowled'] / agg['balls_bowled']) * 100, 0)
    agg['bowling_sr'] = np.where(agg['wickets'] > 0,
                                 agg['balls_bowled'] / agg['wickets'], 999)
    agg['bowling_avg'] = np.where(agg['wickets'] > 0,
                                  agg['runs_conceded'] / agg['wickets'], 999)
    agg['boundary_conceded_rate'] = np.where(
        agg['overs_bowled'] > 0,
        (agg['fours_conceded'] + agg['sixes_conceded']) / agg['overs_bowled'], 0
    )

    # Rename
    agg.rename(columns={'bowler': 'player', 'bowling_team': 'player_team',
                         'batting_team': 'opposition'}, inplace=True)
    agg['role'] = 'bowler'

    return agg


def compute_phase_performance(ball_df):
    """Per-phase stats for batters and bowlers."""
    df = ball_df.copy()

    # Batting phase stats
    bat_phase = df.groupby(['match_id', 'innings', 'striker', 'phase']).agg(
        phase_runs=('runs_off_bat', 'sum'),
        phase_balls=('is_legal', 'sum'),
        phase_fours=('is_four', 'sum'),
        phase_sixes=('is_six', 'sum'),
        phase_dots=('is_dot', 'sum'),
    ).reset_index()
    bat_phase['phase_sr'] = np.where(bat_phase['phase_balls'] > 0,
                                     (bat_phase['phase_runs'] / bat_phase['phase_balls']) * 100, 0)
    bat_phase.rename(columns={'striker': 'player'}, inplace=True)
    bat_phase['role'] = 'batter'

    # Bowling phase stats
    bowl_phase = df.groupby(['match_id', 'innings', 'bowler', 'phase']).agg(
        phase_runs_conceded=('total_runs', 'sum'),
        phase_balls=('is_legal', 'sum'),
        phase_wickets=('is_wicket', 'sum'),
        phase_dots=('is_dot', 'sum'),
    ).reset_index()
    bowl_phase['phase_economy'] = np.where(
        bowl_phase['phase_balls'] > 0,
        bowl_phase['phase_runs_conceded'] / (bowl_phase['phase_balls'] / 6.0), 0
    )
    bowl_phase.rename(columns={'bowler': 'player'}, inplace=True)
    bowl_phase['role'] = 'bowler'

    return bat_phase, bowl_phase


def get_innings_totals(ball_df):
    """Per match-innings totals."""
    df = ball_df.copy()
    totals = df.groupby(['match_id', 'innings']).agg(
        innings_runs=('total_runs', 'sum'),
        innings_wickets=('is_wicket', 'sum'),
        innings_balls=('is_legal', 'sum'),
    ).reset_index()
    totals['innings_rr'] = np.where(
        totals['innings_balls'] > 0,
        totals['innings_runs'] / (totals['innings_balls'] / 6.0), 0
    )
    return totals


def merge_runs_contribution(bat_df, innings_totals):
    """Add runs_contribution = player_runs / innings_total_runs * 100."""
    merged = bat_df.merge(
        innings_totals[['match_id', 'innings', 'innings_runs']],
        on=['match_id', 'innings'], how='left'
    )
    merged['runs_contribution'] = np.where(
        merged['innings_runs'] > 0,
        (merged['runs'] / merged['innings_runs']) * 100, 0
    )
    return merged
