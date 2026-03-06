"""
Module 6: Partnerships & Rivalries
Batting partnerships, bowling tandems, batter-vs-bowler H2H.
"""

import pandas as pd
import numpy as np


def compute_batting_partnerships(ball_df):
    """Compute batting partnership stats per match-innings."""
    df = ball_df.copy()

    # Normalize pair: alphabetical order
    df['p1'] = df[['striker', 'non_striker']].min(axis=1)
    df['p2'] = df[['striker', 'non_striker']].max(axis=1)

    partnerships = df.groupby(['match_id', 'innings', 'batting_team', 'p1', 'p2']).agg(
        partnership_runs=('runs_off_bat', 'sum'),
        partnership_extras=('extras', 'sum'),
        partnership_balls=('is_legal', 'sum'),
        partnership_fours=('is_four', 'sum'),
        partnership_sixes=('is_six', 'sum'),
        wickets_fallen=('is_wicket', 'sum'),
    ).reset_index()

    partnerships['partnership_total'] = partnerships['partnership_runs'] + partnerships['partnership_extras']
    partnerships['partnership_sr'] = np.where(
        partnerships['partnership_balls'] > 0,
        (partnerships['partnership_total'] / partnerships['partnership_balls']) * 100, 0
    )
    partnerships['boundary_count'] = partnerships['partnership_fours'] + partnerships['partnership_sixes']

    # Normalize for impact calculation
    for col in ['partnership_total', 'partnership_sr', 'boundary_count']:
        max_val = partnerships[col].max()
        if max_val > 0:
            partnerships[f'{col}_norm'] = partnerships[col] / max_val
        else:
            partnerships[f'{col}_norm'] = 0

    partnerships['partnership_impact'] = (
        0.5 * partnerships['partnership_total_norm'] +
        0.3 * partnerships['partnership_sr_norm'] +
        0.2 * partnerships['boundary_count_norm']
    )

    return partnerships


def get_best_partnership_per_player(partnerships):
    """For each player-match, find their best partnership by runs."""
    rows = []
    for _, row in partnerships.iterrows():
        for p in [row['p1'], row['p2']]:
            rows.append({
                'match_id': row['match_id'],
                'innings': row['innings'],
                'player': p,
                'best_partnership_runs': row['partnership_total'],
                'best_partnership_sr': row['partnership_sr'],
                'best_partnership_impact': row['partnership_impact'],
                'partner': row['p2'] if p == row['p1'] else row['p1'],
            })
    best_df = pd.DataFrame(rows)
    if len(best_df) > 0:
        best_df = best_df.sort_values('best_partnership_runs', ascending=False)
        best_df = best_df.drop_duplicates(subset=['match_id', 'innings', 'player'], keep='first')
    return best_df


def compute_bowling_tandems(ball_df):
    """Pair bowlers bowling in consecutive overs."""
    df = ball_df.copy()

    # Get bowler per over per match-innings
    over_bowlers = df.groupby(['match_id', 'innings', 'over', 'bowling_team'])['bowler'].first().reset_index()
    over_bowlers = over_bowlers.sort_values(['match_id', 'innings', 'over']).reset_index(drop=True)

    # Create pairs from consecutive overs (partners at different ends)
    pairs = []
    for (mid, inn), grp in over_bowlers.groupby(['match_id', 'innings']):
        bowlers = grp['bowler'].values
        team = grp['bowling_team'].values[0]
        for i in range(len(bowlers) - 1):
            if bowlers[i] != bowlers[i + 1]:
                b1, b2 = sorted([bowlers[i], bowlers[i + 1]])
                pairs.append({
                    'match_id': mid, 'innings': inn,
                    'bowling_team': team,
                    'bowler1': b1, 'bowler2': b2,
                })

    if not pairs:
        return pd.DataFrame(columns=['match_id', 'innings', 'bowling_team',
                                      'bowler1', 'bowler2', 'tandem_wickets',
                                      'tandem_runs', 'tandem_economy', 'tandem_dot_pct'])

    pair_df = pd.DataFrame(pairs).drop_duplicates()

    # Stats per bowler per match-innings
    bowler_stats = df.groupby(['match_id', 'innings', 'bowler']).agg(
        wickets=('is_wicket', 'sum'),
        runs_conceded=('total_runs', 'sum'),
        balls=('is_legal', 'sum'),
        dots=('is_dot', 'sum'),
    ).reset_index()

    # Merge stats for both bowlers in tandem
    tandem = pair_df.merge(bowler_stats.rename(columns={'bowler': 'bowler1',
                                                         'wickets': 'w1', 'runs_conceded': 'r1',
                                                         'balls': 'b1', 'dots': 'd1'}),
                           on=['match_id', 'innings', 'bowler1'], how='left')
    tandem = tandem.merge(bowler_stats.rename(columns={'bowler': 'bowler2',
                                                        'wickets': 'w2', 'runs_conceded': 'r2',
                                                        'balls': 'b2', 'dots': 'd2'}),
                          on=['match_id', 'innings', 'bowler2'], how='left')

    for c in ['w1', 'r1', 'b1', 'd1', 'w2', 'r2', 'b2', 'd2']:
        tandem[c] = tandem[c].fillna(0)

    tandem['tandem_wickets'] = tandem['w1'] + tandem['w2']
    tandem['tandem_runs'] = tandem['r1'] + tandem['r2']
    tandem['tandem_balls'] = tandem['b1'] + tandem['b2']
    tandem['tandem_dots'] = tandem['d1'] + tandem['d2']
    tandem['tandem_economy'] = np.where(
        tandem['tandem_balls'] > 0,
        tandem['tandem_runs'] / (tandem['tandem_balls'] / 6.0), 0
    )
    tandem['tandem_dot_pct'] = np.where(
        tandem['tandem_balls'] > 0,
        (tandem['tandem_dots'] / tandem['tandem_balls']) * 100, 0
    )

    return tandem[['match_id', 'innings', 'bowling_team', 'bowler1', 'bowler2',
                    'tandem_wickets', 'tandem_runs', 'tandem_economy', 'tandem_dot_pct']]


def compute_batter_vs_bowler_rivalry(ball_df):
    """Career-level batter vs bowler H2H stats."""
    df = ball_df.copy()

    rivalry = df.groupby(['striker', 'bowler']).agg(
        matches=('match_id', 'nunique'),
        balls=('is_legal', 'sum'),
        runs=('runs_off_bat', 'sum'),
        dismissals=('is_wicket', 'sum'),
        dots=('is_dot', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
    ).reset_index()

    # Filter: minimum 6 balls
    rivalry = rivalry[rivalry['balls'] >= 6].copy()

    rivalry['sr'] = np.where(rivalry['balls'] > 0,
                             (rivalry['runs'] / rivalry['balls']) * 100, 0)
    rivalry['dot_pct'] = np.where(rivalry['balls'] > 0,
                                  (rivalry['dots'] / rivalry['balls']) * 100, 0)
    rivalry['boundary_pct'] = np.where(rivalry['balls'] > 0,
                                       ((rivalry['fours'] + rivalry['sixes']) / rivalry['balls']) * 100, 0)
    rivalry['dismissal_rate'] = np.where(rivalry['balls'] > 0,
                                         (rivalry['dismissals'] / rivalry['balls']) * 100, 0)

    # Dominance index: higher = batter dominates
    # Normalize each component
    sr_norm = rivalry['sr'] / 200.0  # typical T20 SR
    dot_norm = 1 - (rivalry['dot_pct'] / 100.0)  # lower dots = batter better
    boundary_norm = rivalry['boundary_pct'] / 50.0
    dismiss_norm = 1 - (rivalry['dismissal_rate'] / 20.0)  # lower dismissal = batter better

    rivalry['dominance_index'] = (
        0.4 * sr_norm.clip(0, 1) +
        0.2 * dot_norm.clip(0, 1) +
        0.2 * boundary_norm.clip(0, 1) +
        0.2 * dismiss_norm.clip(0, 1)
    )

    rivalry.rename(columns={'striker': 'batter', 'bowler': 'bowler_name'}, inplace=True)
    return rivalry


def merge_rivalry_features(bat_df, bowl_df, rivalry_df):
    """Merge average dominance features into player DataFrames."""
    if rivalry_df is None or len(rivalry_df) == 0:
        bat_df['avg_dominance_vs_bowlers'] = 0.5
        bowl_df['avg_dominance_vs_batters'] = 0.5
        return bat_df, bowl_df

    # For batters: average dominance vs all bowlers they faced
    batter_dom = rivalry_df.groupby('batter')['dominance_index'].mean().reset_index()
    batter_dom.columns = ['player', 'avg_dominance_vs_bowlers']
    bat_df = bat_df.merge(batter_dom, on='player', how='left')
    bat_df['avg_dominance_vs_bowlers'] = bat_df['avg_dominance_vs_bowlers'].fillna(0.5)

    # For bowlers: average dominance of batters against them (inverted)
    bowler_dom = rivalry_df.groupby('bowler_name')['dominance_index'].mean().reset_index()
    bowler_dom.columns = ['player', 'avg_dominance_vs_batters']
    bowler_dom['avg_dominance_vs_batters'] = 1 - bowler_dom['avg_dominance_vs_batters']
    bowl_df = bowl_df.merge(bowler_dom, on='player', how='left')
    bowl_df['avg_dominance_vs_batters'] = bowl_df['avg_dominance_vs_batters'].fillna(0.5)

    return bat_df, bowl_df
