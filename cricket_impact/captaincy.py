"""
Module: Captaincy, Win Bonus, and Playoff Features
Provides captain mapping, win/loss credits, and playoff qualification bonus.
"""

import numpy as np
import pandas as pd


# ── IPL Captain Mapping: (season, team) -> captain name ──
# Uses Cricsheet name format. Primary captains for each season.
IPL_CAPTAINS = {
    # 2007/08
    ('2007/08', 'Chennai Super Kings'): 'MS Dhoni',
    ('2007/08', 'Deccan Chargers'): 'AC Gilchrist',
    ('2007/08', 'Delhi Daredevils'): 'V Sehwag',
    ('2007/08', 'Kings XI Punjab'): 'Yuvraj Singh',
    ('2007/08', 'Kolkata Knight Riders'): 'SC Ganguly',
    ('2007/08', 'Mumbai Indians'): 'SR Tendulkar',
    ('2007/08', 'Rajasthan Royals'): 'SK Warne',
    ('2007/08', 'Royal Challengers Bangalore'): 'R Dravid',
    # 2009
    ('2009', 'Chennai Super Kings'): 'MS Dhoni',
    ('2009', 'Deccan Chargers'): 'AC Gilchrist',
    ('2009', 'Delhi Daredevils'): 'V Sehwag',
    ('2009', 'Kings XI Punjab'): 'Yuvraj Singh',
    ('2009', 'Kolkata Knight Riders'): 'SC Ganguly',
    ('2009', 'Mumbai Indians'): 'SR Tendulkar',
    ('2009', 'Rajasthan Royals'): 'SK Warne',
    ('2009', 'Royal Challengers Bangalore'): 'KP Pietersen',
    # 2009/10
    ('2009/10', 'Chennai Super Kings'): 'MS Dhoni',
    ('2009/10', 'Deccan Chargers'): 'AC Gilchrist',
    ('2009/10', 'Delhi Daredevils'): 'V Sehwag',
    ('2009/10', 'Kings XI Punjab'): 'Yuvraj Singh',
    ('2009/10', 'Kolkata Knight Riders'): 'SC Ganguly',
    ('2009/10', 'Mumbai Indians'): 'SR Tendulkar',
    ('2009/10', 'Rajasthan Royals'): 'SK Warne',
    ('2009/10', 'Royal Challengers Bangalore'): 'A Kumble',
    # 2011
    ('2011', 'Chennai Super Kings'): 'MS Dhoni',
    ('2011', 'Deccan Chargers'): 'S Dhawan',
    ('2011', 'Delhi Daredevils'): 'V Sehwag',
    ('2011', 'Kings XI Punjab'): 'AC Gilchrist',
    ('2011', 'Kochi Tuskers Kerala'): 'DPMD Jayawardene',
    ('2011', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2011', 'Mumbai Indians'): 'SR Tendulkar',
    ('2011', 'Pune Warriors'): 'Yuvraj Singh',
    ('2011', 'Rajasthan Royals'): 'SR Watson',
    ('2011', 'Royal Challengers Bangalore'): 'V Kohli',
    # 2012
    ('2012', 'Chennai Super Kings'): 'MS Dhoni',
    ('2012', 'Deccan Chargers'): 'KD Karthik',
    ('2012', 'Delhi Daredevils'): 'V Sehwag',
    ('2012', 'Kings XI Punjab'): 'AC Gilchrist',
    ('2012', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2012', 'Mumbai Indians'): 'SR Tendulkar',
    ('2012', 'Pune Warriors'): 'SC Ganguly',
    ('2012', 'Rajasthan Royals'): 'R Dravid',
    ('2012', 'Royal Challengers Bangalore'): 'V Kohli',
    # 2013
    ('2013', 'Chennai Super Kings'): 'MS Dhoni',
    ('2013', 'Delhi Daredevils'): 'V Sehwag',
    ('2013', 'Kings XI Punjab'): 'AC Gilchrist',
    ('2013', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2013', 'Mumbai Indians'): 'RG Sharma',
    ('2013', 'Pune Warriors'): 'AJ Finch',
    ('2013', 'Rajasthan Royals'): 'R Dravid',
    ('2013', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2013', 'Sunrisers Hyderabad'): 'KD Karthik',
    # 2014
    ('2014', 'Chennai Super Kings'): 'MS Dhoni',
    ('2014', 'Delhi Daredevils'): 'KP Pietersen',
    ('2014', 'Kings XI Punjab'): 'GJ Bailey',
    ('2014', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2014', 'Mumbai Indians'): 'RG Sharma',
    ('2014', 'Rajasthan Royals'): 'SR Watson',
    ('2014', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2014', 'Sunrisers Hyderabad'): 'S Dhawan',
    # 2015
    ('2015', 'Chennai Super Kings'): 'MS Dhoni',
    ('2015', 'Delhi Daredevils'): 'JP Duminy',
    ('2015', 'Kings XI Punjab'): 'GJ Bailey',
    ('2015', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2015', 'Mumbai Indians'): 'RG Sharma',
    ('2015', 'Rajasthan Royals'): 'SR Watson',
    ('2015', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2015', 'Sunrisers Hyderabad'): 'DA Warner',
    # 2016
    ('2016', 'Delhi Daredevils'): 'Z Khan',
    ('2016', 'Gujarat Lions'): 'SK Raina',
    ('2016', 'Kings XI Punjab'): 'DA Miller',
    ('2016', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2016', 'Mumbai Indians'): 'RG Sharma',
    ('2016', 'Rising Pune Supergiants'): 'MS Dhoni',
    ('2016', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2016', 'Sunrisers Hyderabad'): 'DA Warner',
    # 2017
    ('2017', 'Delhi Daredevils'): 'Z Khan',
    ('2017', 'Gujarat Lions'): 'SK Raina',
    ('2017', 'Kings XI Punjab'): 'GJ Maxwell',
    ('2017', 'Kolkata Knight Riders'): 'G Gambhir',
    ('2017', 'Mumbai Indians'): 'RG Sharma',
    ('2017', 'Rising Pune Supergiant'): 'SPD Smith',
    ('2017', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2017', 'Sunrisers Hyderabad'): 'DA Warner',
    # 2018
    ('2018', 'Chennai Super Kings'): 'MS Dhoni',
    ('2018', 'Delhi Daredevils'): 'SS Iyer',
    ('2018', 'Kings XI Punjab'): 'R Ashwin',
    ('2018', 'Kolkata Knight Riders'): 'KD Karthik',
    ('2018', 'Mumbai Indians'): 'RG Sharma',
    ('2018', 'Rajasthan Royals'): 'AM Rahane',
    ('2018', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2018', 'Sunrisers Hyderabad'): 'KS Williamson',
    # 2019
    ('2019', 'Chennai Super Kings'): 'MS Dhoni',
    ('2019', 'Delhi Capitals'): 'SS Iyer',
    ('2019', 'Kings XI Punjab'): 'R Ashwin',
    ('2019', 'Kolkata Knight Riders'): 'KD Karthik',
    ('2019', 'Mumbai Indians'): 'RG Sharma',
    ('2019', 'Rajasthan Royals'): 'AM Rahane',
    ('2019', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2019', 'Sunrisers Hyderabad'): 'DA Warner',
    # 2020/21
    ('2020/21', 'Chennai Super Kings'): 'MS Dhoni',
    ('2020/21', 'Delhi Capitals'): 'SS Iyer',
    ('2020/21', 'Kings XI Punjab'): 'KL Rahul',
    ('2020/21', 'Kolkata Knight Riders'): 'KD Karthik',
    ('2020/21', 'Mumbai Indians'): 'RG Sharma',
    ('2020/21', 'Rajasthan Royals'): 'SPD Smith',
    ('2020/21', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2020/21', 'Sunrisers Hyderabad'): 'DA Warner',
    # 2021
    ('2021', 'Chennai Super Kings'): 'MS Dhoni',
    ('2021', 'Delhi Capitals'): 'RR Pant',
    ('2021', 'Kolkata Knight Riders'): 'DE Morgan',
    ('2021', 'Mumbai Indians'): 'RG Sharma',
    ('2021', 'Punjab Kings'): 'KL Rahul',
    ('2021', 'Rajasthan Royals'): 'SS Samson',
    ('2021', 'Royal Challengers Bangalore'): 'V Kohli',
    ('2021', 'Sunrisers Hyderabad'): 'KS Williamson',
    # 2022
    ('2022', 'Chennai Super Kings'): 'MS Dhoni',
    ('2022', 'Delhi Capitals'): 'RR Pant',
    ('2022', 'Gujarat Titans'): 'HH Pandya',
    ('2022', 'Kolkata Knight Riders'): 'SS Iyer',
    ('2022', 'Lucknow Super Giants'): 'KL Rahul',
    ('2022', 'Mumbai Indians'): 'RG Sharma',
    ('2022', 'Punjab Kings'): 'Mayank Agarwal',
    ('2022', 'Rajasthan Royals'): 'SS Samson',
    ('2022', 'Royal Challengers Bangalore'): 'F du Plessis',
    ('2022', 'Sunrisers Hyderabad'): 'KS Williamson',
    # 2023
    ('2023', 'Chennai Super Kings'): 'MS Dhoni',
    ('2023', 'Delhi Capitals'): 'DA Warner',
    ('2023', 'Gujarat Titans'): 'HH Pandya',
    ('2023', 'Kolkata Knight Riders'): 'N Rana',
    ('2023', 'Lucknow Super Giants'): 'KL Rahul',
    ('2023', 'Mumbai Indians'): 'RG Sharma',
    ('2023', 'Punjab Kings'): 'S Dhawan',
    ('2023', 'Rajasthan Royals'): 'SS Samson',
    ('2023', 'Royal Challengers Bangalore'): 'F du Plessis',
    ('2023', 'Sunrisers Hyderabad'): 'A Markram',
    # 2024
    ('2024', 'Chennai Super Kings'): 'RD Gaikwad',
    ('2024', 'Delhi Capitals'): 'RR Pant',
    ('2024', 'Gujarat Titans'): 'SA Gill',
    ('2024', 'Kolkata Knight Riders'): 'SS Iyer',
    ('2024', 'Lucknow Super Giants'): 'KL Rahul',
    ('2024', 'Mumbai Indians'): 'HH Pandya',
    ('2024', 'Punjab Kings'): 'S Dhawan',
    ('2024', 'Rajasthan Royals'): 'SS Samson',
    ('2024', 'Royal Challengers Bengaluru'): 'F du Plessis',
    ('2024', 'Sunrisers Hyderabad'): 'PJ Cummins',
    # 2025
    ('2025', 'Chennai Super Kings'): 'RD Gaikwad',
    ('2025', 'Delhi Capitals'): 'KL Rahul',
    ('2025', 'Gujarat Titans'): 'SA Gill',
    ('2025', 'Kolkata Knight Riders'): 'AM Rahane',
    ('2025', 'Lucknow Super Giants'): 'RR Pant',
    ('2025', 'Mumbai Indians'): 'HH Pandya',
    ('2025', 'Punjab Kings'): 'SS Iyer',
    ('2025', 'Rajasthan Royals'): 'SS Samson',
    ('2025', 'Royal Challengers Bengaluru'): 'V Kohli',
    ('2025', 'Sunrisers Hyderabad'): 'PJ Cummins',
}


def get_captain_for_match(season, team):
    """Look up captain for a given season and team."""
    return IPL_CAPTAINS.get((str(season), team), None)


def build_captaincy_data(info_df, ball_df):
    """
    Build per-match captaincy information from info_df.
    Returns a DataFrame with: match_id, season, team1, team2, captain1, captain2,
    winner, is_playoff, and per-team rosters if available.
    """
    rows = []
    for _, match in info_df.iterrows():
        mid = match['match_id']
        season = str(match.get('season', ''))
        winner = str(match.get('winner', ''))
        is_playoff = int(match.get('is_playoff', 0))

        # Get teams from ball_df for this match
        match_balls = ball_df[ball_df['match_id'] == mid]
        teams = list(match_balls['batting_team'].unique())
        if len(teams) < 2:
            continue

        for team in teams:
            captain = get_captain_for_match(season, team)
            if captain is None:
                continue
            is_winner = 1 if team == winner else 0
            rows.append({
                'match_id': mid,
                'season': season,
                'team': team,
                'captain': captain,
                'is_winner': is_winner,
                'is_playoff': is_playoff,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['match_id', 'season', 'team', 'captain', 'is_winner', 'is_playoff']
    )


def compute_captaincy_stats(captaincy_df):
    """
    Compute per-captain career stats:
    - matches_as_captain, wins_as_captain, losses_as_captain, win_rate
    - playoff_matches_as_captain
    - captaincy_score: small net credit (wins - losses weighted)
    Returns DataFrame with player, captaincy columns.
    """
    if len(captaincy_df) == 0:
        return pd.DataFrame(columns=['player', 'captaincy_score', 'matches_as_captain',
                                     'wins_as_captain', 'captain_playoff_matches'])

    stats = captaincy_df.groupby('captain').agg(
        matches_as_captain=('match_id', 'nunique'),
        wins_as_captain=('is_winner', 'sum'),
        captain_playoff_matches=('is_playoff', 'sum'),
    ).reset_index()
    stats.rename(columns={'captain': 'player'}, inplace=True)

    stats['losses_as_captain'] = stats['matches_as_captain'] - stats['wins_as_captain']

    # Captaincy score: +2 per win, -1 per loss (net positive for winning captains)
    # Plus bonus for playoff appearances
    stats['captaincy_score'] = (
        2.0 * stats['wins_as_captain']
        - 1.0 * stats['losses_as_captain']
        + 3.0 * stats['captain_playoff_matches']
    )

    # Normalize to 0-1 range
    cs_max = stats['captaincy_score'].max()
    if cs_max > 0:
        stats['captaincy_score_norm'] = stats['captaincy_score'] / cs_max
    else:
        stats['captaincy_score_norm'] = 0

    return stats


def compute_win_bonus(ball_df, info_df):
    """
    Compute per-player win bonus: how many wins each player was part of.
    This is a small additive bonus — 0 for non-winners, positive for frequent winners.
    Returns DataFrame with player, total_wins, win_rate.
    """
    # Merge winner info
    if 'winner' not in info_df.columns:
        return pd.DataFrame(columns=['player', 'total_wins', 'win_rate', 'win_bonus_norm'])

    match_winner = info_df[['match_id', 'winner']].drop_duplicates()

    # Get unique player-match-team combinations
    bat_players = ball_df[['match_id', 'batting_team', 'striker']].rename(
        columns={'striker': 'player', 'batting_team': 'team'}
    ).drop_duplicates()
    bowl_players = ball_df[['match_id', 'bowling_team', 'bowler']].rename(
        columns={'bowler': 'player', 'bowling_team': 'team'}
    ).drop_duplicates()

    all_players = pd.concat([bat_players, bowl_players], ignore_index=True).drop_duplicates()
    all_players = all_players.merge(match_winner, on='match_id', how='left')
    all_players['is_winner'] = (all_players['team'] == all_players['winner']).astype(int)

    # Aggregate per player
    win_stats = all_players.groupby('player').agg(
        total_matches_played=('match_id', 'nunique'),
        total_wins=('is_winner', 'sum'),
    ).reset_index()
    win_stats['win_rate'] = np.where(
        win_stats['total_matches_played'] > 0,
        win_stats['total_wins'] / win_stats['total_matches_played'], 0
    )

    # Normalize: win bonus = log(1 + wins) scaled
    win_stats['win_bonus_raw'] = np.log1p(win_stats['total_wins'])
    wb_max = win_stats['win_bonus_raw'].max()
    win_stats['win_bonus_norm'] = win_stats['win_bonus_raw'] / wb_max if wb_max > 0 else 0

    return win_stats


def compute_playoff_team_bonus(info_df, ball_df):
    """
    Identify teams that played in playoff matches each season.
    Players on those teams get a small bonus.
    Returns DataFrame with player, playoff_seasons (count of seasons where
    the player's team made playoffs).
    """
    playoff_matches = info_df[info_df['is_playoff'] == 1]
    if len(playoff_matches) == 0:
        return pd.DataFrame(columns=['player', 'playoff_seasons'])

    # Get teams in playoff matches
    playoff_teams = set()
    for _, row in playoff_matches.iterrows():
        mid = row['match_id']
        season = str(row.get('season', ''))
        match_balls = ball_df[ball_df['match_id'] == mid]
        for team in match_balls['batting_team'].unique():
            playoff_teams.add((season, team))

    # For each player, count how many seasons their team made playoffs
    bat_players = ball_df[['match_id', 'batting_team', 'striker']].rename(
        columns={'striker': 'player', 'batting_team': 'team'})
    bowl_players = ball_df[['match_id', 'bowling_team', 'bowler']].rename(
        columns={'bowler': 'player', 'bowling_team': 'team'})
    all_players = pd.concat([bat_players, bowl_players], ignore_index=True)

    # Merge season
    season_map = info_df[['match_id', 'season']].drop_duplicates()
    all_players = all_players.merge(season_map, on='match_id', how='left')
    all_players['season'] = all_players['season'].astype(str)

    # Check if (season, team) is in playoff_teams
    all_players['in_playoff_team'] = all_players.apply(
        lambda r: 1 if (r['season'], r['team']) in playoff_teams else 0, axis=1
    )

    # Count distinct playoff seasons per player
    playoff_player = all_players[all_players['in_playoff_team'] == 1].groupby('player').agg(
        playoff_seasons=('season', 'nunique')
    ).reset_index()

    return playoff_player
