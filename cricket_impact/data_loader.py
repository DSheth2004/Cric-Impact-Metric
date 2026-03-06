"""
Module 1: Data Ingestion & Cleaning
Load all ball-by-ball and info CSVs, clean and prepare for analysis.
"""

import os
import re
import pandas as pd
import numpy as np


def load_ball_by_ball(data_folder):
    """Read all non-_info CSVs from folder, combine into one DataFrame."""
    frames = []
    for fname in os.listdir(data_folder):
        if fname.endswith('.csv') and '_info' not in fname:
            match_id = int(fname.replace('.csv', ''))
            fpath = os.path.join(data_folder, fname)
            try:
                df = pd.read_csv(fpath, low_memory=False)
                df['match_id'] = match_id
                frames.append(df)
            except Exception:
                continue
    if not frames:
        raise ValueError(f"No ball-by-ball CSVs found in {data_folder}")
    ball_df = pd.concat(frames, ignore_index=True)
    return ball_df


def load_match_info(data_folder):
    """Parse each _info.csv for match metadata."""
    records = []
    for fname in os.listdir(data_folder):
        if fname.endswith('_info.csv'):
            match_id = int(fname.replace('_info.csv', ''))
            fpath = os.path.join(data_folder, fname)
            rec = {'match_id': match_id}
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 3 or parts[0] != 'info':
                            continue
                        key = parts[1].strip()
                        val = ','.join(parts[2:]).strip().strip('"')
                        if key == 'winner':
                            rec['winner'] = val
                        elif key == 'winner_runs':
                            rec['winner_runs'] = int(val)
                        elif key == 'winner_wickets':
                            rec['winner_wickets'] = int(val)
                        elif key == 'player_of_match':
                            rec['player_of_match'] = val
                        elif key == 'toss_winner':
                            rec['toss_winner'] = val
                        elif key == 'toss_decision':
                            rec['toss_decision'] = val
                        elif key == 'venue':
                            rec['venue'] = val
                        elif key == 'city':
                            rec['city'] = val
                        elif key == 'season':
                            rec['season'] = val
                        elif key == 'date':
                            rec['start_date'] = val
                        elif key == 'outcome' and val == 'no result':
                            rec['no_result'] = True
                        elif key == 'match_number':
                            try:
                                rec['match_number'] = int(val)
                            except ValueError:
                                pass
                        elif key == 'event':
                            rec['event'] = val
                        elif key == 'team':
                            rec.setdefault('teams', []).append(val)
                        elif key == 'player' and len(parts) >= 4:
                            team_name = parts[2].strip()
                            player_name = ','.join(parts[3:]).strip().strip('"')
                            rec.setdefault('roster', []).append((team_name, player_name))
            except Exception:
                continue
            records.append(rec)
    info_df = pd.DataFrame(records)
    for col in ['winner', 'player_of_match', 'toss_winner', 'toss_decision',
                'venue', 'city', 'season', 'start_date']:
        if col not in info_df.columns:
            info_df[col] = ''
    for col in ['winner_runs', 'winner_wickets']:
        if col not in info_df.columns:
            info_df[col] = 0
    if 'no_result' not in info_df.columns:
        info_df['no_result'] = False
    info_df['winner_runs'] = info_df['winner_runs'].fillna(0).astype(int)
    info_df['winner_wickets'] = info_df['winner_wickets'].fillna(0).astype(int)
    info_df['no_result'] = info_df['no_result'].fillna(False)
    if 'match_number' not in info_df.columns:
        info_df['match_number'] = -1
    info_df['match_number'] = info_df['match_number'].fillna(-1).astype(int)
    # is_playoff: matches without a match_number are playoff matches
    info_df['is_playoff'] = (info_df['match_number'] == -1).astype(int)

    # Detect match format from event name or balls_per_over hint
    def _detect_format(row):
        event = str(row.get('event', '')).lower()
        if any(k in event for k in ['test', 'trophy', 'sheffield',
                                     'ranji', 'county', 'ashes']):
            return 'Test'
        if any(k in event for k in ['t20', 'twenty20', 'ipl', 'bbl',
                                     'premier league', 'cpl', 'psl',
                                     'hundred', 'sa20', 'blast']):
            return 'T20'
        if any(k in event for k in ['odi', 'world cup', 'asia cup',
                                     'champions trophy', '50-over']):
            return 'ODI'
        return ''  # unknown — will be resolved from ball data
    if 'event' in info_df.columns:
        info_df['match_format'] = info_df.apply(_detect_format, axis=1)
    else:
        info_df['match_format'] = ''

    return info_df


def clean_data(ball_df):
    """Clean and derive columns from raw ball-by-ball data."""
    # Fill missing values
    cat_cols = ['batting_team', 'bowling_team', 'striker', 'non_striker',
                'bowler', 'wicket_type', 'player_dismissed',
                'other_wicket_type', 'other_player_dismissed']
    for c in cat_cols:
        if c in ball_df.columns:
            ball_df[c] = ball_df[c].fillna('').astype(str)

    num_cols = ['runs_off_bat', 'extras', 'wides', 'noballs', 'byes',
                'legbyes', 'penalty']
    for c in num_cols:
        if c in ball_df.columns:
            ball_df[c] = pd.to_numeric(ball_df[c], errors='coerce').fillna(0).astype(int)

    # Extract over and ball_in_over from ball column
    ball_df['ball'] = pd.to_numeric(ball_df['ball'], errors='coerce').fillna(0)
    ball_df['over'] = ball_df['ball'].apply(lambda x: int(x))
    ball_df['ball_in_over'] = ball_df['ball'].apply(
        lambda x: round((x - int(x)) * 10)
    )

    # Derived columns
    ball_df['total_runs'] = ball_df['runs_off_bat'] + ball_df['extras']
    ball_df['is_four'] = (ball_df['runs_off_bat'] == 4).astype(int)
    ball_df['is_six'] = (ball_df['runs_off_bat'] == 6).astype(int)
    ball_df['is_dot'] = ((ball_df['runs_off_bat'] == 0) & (ball_df['extras'] == 0)).astype(int)
    ball_df['is_wicket'] = (ball_df['wicket_type'] != '').astype(int)
    ball_df['is_legal'] = ((ball_df['wides'] == 0) & (ball_df['noballs'] == 0)).astype(int)

    # Phase
    def get_phase(over):
        if over < 6:
            return 'Powerplay'
        elif over < 15:
            return 'Middle'
        else:
            return 'Death'
    ball_df['phase'] = ball_df['over'].apply(get_phase)
    ball_df['phase_encoded'] = ball_df['phase'].map(
        {'Powerplay': 0, 'Middle': 1, 'Death': 2}
    )

    # Ensure innings is numeric
    ball_df['innings'] = pd.to_numeric(ball_df['innings'], errors='coerce').fillna(1).astype(int)

    # Sort
    ball_df = ball_df.sort_values(['match_id', 'innings', 'over', 'ball_in_over']).reset_index(drop=True)

    return ball_df


def load_and_prepare(data_folder):
    """Orchestrate full loading. Returns ball_df, info_df."""
    print("Loading ball-by-ball data...")
    ball_df = load_ball_by_ball(data_folder)
    print(f"  Loaded {len(ball_df):,} deliveries from {ball_df['match_id'].nunique()} matches")

    print("Loading match info...")
    info_df = load_match_info(data_folder)
    print(f"  Loaded info for {len(info_df)} matches")

    print("Cleaning data...")
    ball_df = clean_data(ball_df)
    print("  Data cleaned and derived columns added")

    # Resolve match format: use info event name, fallback to max overs in ball data
    if 'match_format' not in info_df.columns:
        info_df['match_format'] = ''
    # For unresolved formats, detect from max over in ball data
    max_overs = ball_df.groupby('match_id')['over'].max().reset_index()
    max_overs.columns = ['match_id', 'max_over']
    info_df = info_df.merge(max_overs, on='match_id', how='left')
    info_df['max_over'] = info_df['max_over'].fillna(19)
    unknown = info_df['match_format'] == ''
    info_df.loc[unknown & (info_df['max_over'] <= 20), 'match_format'] = 'T20'
    info_df.loc[unknown & (info_df['max_over'] > 20) & (info_df['max_over'] <= 50), 'match_format'] = 'ODI'
    info_df.loc[unknown & (info_df['max_over'] > 50), 'match_format'] = 'Test'
    info_df.drop(columns=['max_over'], inplace=True)

    # Merge format into ball_df
    ball_df = ball_df.merge(info_df[['match_id', 'match_format']], on='match_id', how='left')
    ball_df['match_format'] = ball_df['match_format'].fillna('T20')

    return ball_df, info_df
