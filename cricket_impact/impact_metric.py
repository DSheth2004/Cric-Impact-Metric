"""
Module 7: Impact Metric Computation
WPA-derived format-aware formula + volume multiplier for batting and bowling.
Supports T20, ODI, and Test formats with different weight profiles.
"""

import pandas as pd
import numpy as np

# ── Format-specific weight profiles ──
# Batting: [total_wpa, presence, leverage, runs_contrib, clutch, consistency, dot_efficiency]
BAT_WEIGHTS = {
    'T20':  [0.27, 0.13, 0.13, 0.14, 0.09, 0.13, 0.11],
    'ODI':  [0.25, 0.16, 0.12, 0.16, 0.08, 0.13, 0.10],
    'Test': [0.20, 0.22, 0.10, 0.20, 0.08, 0.13, 0.05],  # 2% redistributed to presence+runs
}

# Bowling: [total_wpa, wicket_wpa, leverage, economy, clutch, spell_eff, phase_diff, dot_pressure]
BOWL_WEIGHTS = {
    'T20':  [0.25, 0.15, 0.12, 0.07, 0.08, 0.07, 0.15, 0.11],
    'ODI':  [0.23, 0.17, 0.10, 0.10, 0.07, 0.08, 0.12, 0.13],
    'Test': [0.20, 0.25, 0.08, 0.05, 0.05, 0.10, 0.08, 0.19],
}

# Economy benchmarks: (elite, bad) — used for death/PP normalization
DEATH_ECON_RANGE = {'T20': (6, 15), 'ODI': (5, 12), 'Test': (3, 7)}
PP_ECON_RANGE = {'T20': (5, 12), 'ODI': (4, 10), 'Test': (2, 5)}


def percentile_rank(series):
    """Percentile rank normalization (0-1)."""
    if series.nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, method='average')


def minmax_norm(series):
    """Min-max normalization (0-1)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def compute_batting_impact(bat_df, match_format='T20'):
    """
    Compute batting impact score with format-aware weights.
    bat_df must already have WPA columns merged in.
    """
    df = bat_df.copy()
    fmt = match_format if match_format in BAT_WEIGHTS else 'T20'
    w = BAT_WEIGHTS[fmt]

    # Ensure all required columns exist with defaults
    wpa_cols = {
        'total_wpa': 0, 'presence_impact': 0, 'leverage_wpa': 0,
        'runs_contribution': 0, 'clutch_wpa': 0, 'consistency_wpa': 0,
        'positive_wpa_pct': 0.5, 'wpa_volatility': 0,
    }
    for col, default in wpa_cols.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    # Normalize components
    df['total_wpa_norm'] = percentile_rank(df['total_wpa'])
    df['presence_impact_norm'] = percentile_rank(df['presence_impact'])
    df['leverage_wpa_norm'] = percentile_rank(df['leverage_wpa'])
    df['runs_contribution_norm'] = percentile_rank(df['runs_contribution'])
    df['clutch_wpa_norm'] = percentile_rank(df['clutch_wpa'])
    df['consistency_wpa_norm'] = percentile_rank(df['consistency_wpa'])

    # Dot ball efficiency: fewer dots faced = better for batsmen (inverted)
    if 'dot_pct_faced' not in df.columns:
        if 'dots_faced' in df.columns and 'balls_faced' in df.columns:
            df['dot_pct_faced'] = np.where(df['balls_faced'] > 0,
                                           (df['dots_faced'] / df['balls_faced']) * 100, 50)
        else:
            df['dot_pct_faced'] = 50
    df['dot_pct_faced'] = pd.to_numeric(df['dot_pct_faced'], errors='coerce').fillna(50)
    df['dot_efficiency_norm'] = 1 - percentile_rank(df['dot_pct_faced'])

    # 7-component batting impact formula — format-aware weights
    df['batting_raw_impact'] = (
        w[0] * df['total_wpa_norm'] +
        w[1] * df['presence_impact_norm'] +
        w[2] * df['leverage_wpa_norm'] +
        w[3] * df['runs_contribution_norm'] +
        w[4] * df['clutch_wpa_norm'] +
        w[5] * df['consistency_wpa_norm'] +
        w[6] * df['dot_efficiency_norm']
    )

    # ── Batting position reward ──
    # Expected run share by position: lower-order scoring more than expected = bonus
    EXPECTED_SHARE = {
        1: 14.0, 2: 13.0, 3: 13.0, 4: 12.0, 5: 11.0,
        6: 9.0, 7: 7.0, 8: 6.0, 9: 4.0, 10: 3.0, 11: 2.0
    }
    if 'batting_position' not in df.columns:
        df['batting_position'] = 5  # default middle order
    bat_pos = df['batting_position'].clip(1, 11).astype(int)
    expected_share = bat_pos.map(EXPECTED_SHARE).fillna(9.0)
    actual_share = df['runs_contribution'].clip(0, 100)  # already in %

    # Position multiplier: reward batters who outperform their expected share
    # ratio > 1 means outperforming, < 1 means underperforming
    share_ratio = np.where(expected_share > 0, actual_share / expected_share, 1.0)
    # Clamp ratio to [0.5, 2.5] and convert to multiplier centered at 1.0
    # Bonus range: 0.85 (underperformer) to 1.25 (overperformer)
    df['position_multiplier'] = 0.85 + 0.16 * np.clip(share_ratio, 0.5, 2.5)

    # Volume multiplier for batters — format-aware
    contrib_pct = (df['runs_contribution'] / 100.0).clip(0, 1)
    if 'balls_faced' in df.columns:
        balls_col = pd.to_numeric(df['balls_faced'], errors='coerce').fillna(0)
    else:
        balls_col = pd.Series(0, index=df.index)

    if fmt == 'Test':
        # Tests: staying power matters most — balls_faced/150 (half a session)
        balls_frac = (balls_col / 150.0).clip(0, 1.5)
        df['volume_mult'] = 0.30 + 0.50 * contrib_pct + 0.50 * balls_frac
    elif fmt == 'ODI':
        # ODIs: balanced — balls_faced/100
        balls_frac = (balls_col / 100.0).clip(0, 1.5)
        df['volume_mult'] = 0.35 + 0.65 * contrib_pct + 0.40 * balls_frac
    else:
        # T20: current formula
        balls_frac = (balls_col / 60.0).clip(0, 1.5)
        df['volume_mult'] = 0.40 + 0.80 * contrib_pct + 0.30 * balls_frac

    # Apply volume multiplier AND position reward
    df['raw_impact'] = df['batting_raw_impact'] * df['volume_mult'] * df['position_multiplier']

    return df


def compute_bowling_impact(bowl_df, match_format='T20'):
    """
    Compute bowling impact score with format-aware weights.
    bowl_df must already have WPA columns merged in.
    """
    df = bowl_df.copy()
    fmt = match_format if match_format in BOWL_WEIGHTS else 'T20'
    w = BOWL_WEIGHTS[fmt]

    # Ensure all required columns exist with defaults
    wpa_cols = {
        'total_wpa': 0, 'wicket_wpa': 0, 'leverage_wpa': 0,
        'economy_wpa': 0, 'clutch_wpa': 0, 'spell_efficiency': 0,
        'balls_bowled': 0,
    }
    for col, default in wpa_cols.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    # Normalize components
    df['total_wpa_norm'] = percentile_rank(df['total_wpa'])
    df['wicket_wpa_norm'] = percentile_rank(df['wicket_wpa'])
    df['leverage_wpa_norm'] = percentile_rank(df['leverage_wpa'])
    df['economy_wpa_norm'] = percentile_rank(df['economy_wpa'])
    df['clutch_wpa_norm'] = percentile_rank(df['clutch_wpa'])
    df['spell_efficiency_norm'] = percentile_rank(df['spell_efficiency'])

    # ── Phase difficulty bonus ──
    # Bowling in death overs (16-19) and powerplay (0-5) is significantly harder.
    # Reward bowlers who bowl in these phases and perform well.
    for col in ['death_balls', 'death_wickets', 'death_runs_conceded',
                'pp_balls', 'pp_wickets', 'pp_runs_conceded']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Death overs contribution: fraction of balls bowled in death + wicket rate there
    death_frac = np.where(df['balls_bowled'] > 0,
                          df['death_balls'] / df['balls_bowled'], 0)
    d_elite, d_bad = DEATH_ECON_RANGE.get(fmt, (6, 15))
    death_econ = np.where(df['death_balls'] > 0,
                          df['death_runs_conceded'] / (df['death_balls'] / 6.0), float(d_bad))
    death_econ_score = (1 - ((np.clip(death_econ, d_elite, d_bad) - d_elite) / max(d_bad - d_elite, 1)))
    death_wkt_rate = np.where(df['death_balls'] > 0,
                              df['death_wickets'] / (df['death_balls'] / 6.0), 0)
    death_wkt_score = np.minimum(death_wkt_rate / 2.0, 1.0)

    # Powerplay contribution
    pp_frac = np.where(df['balls_bowled'] > 0,
                       df['pp_balls'] / df['balls_bowled'], 0)
    p_elite, p_bad = PP_ECON_RANGE.get(fmt, (5, 12))
    pp_econ = np.where(df['pp_balls'] > 0,
                       df['pp_runs_conceded'] / (df['pp_balls'] / 6.0), float(p_bad))
    pp_econ_score = (1 - ((np.clip(pp_econ, p_elite, p_bad) - p_elite) / max(p_bad - p_elite, 1)))
    pp_wkt_rate = np.where(df['pp_balls'] > 0,
                           df['pp_wickets'] / (df['pp_balls'] / 6.0), 0)
    pp_wkt_score = np.minimum(pp_wkt_rate / 2.0, 1.0)

    # Combined phase difficulty bonus: weighted by fraction of balls in that phase
    df['phase_difficulty_bonus'] = (
        death_frac * (0.5 * death_econ_score + 0.5 * death_wkt_score) +
        pp_frac * (0.5 * pp_econ_score + 0.5 * pp_wkt_score)
    )
    df['phase_difficulty_norm'] = percentile_rank(df['phase_difficulty_bonus'])

    # Dot ball pressure: more dots bowled = better for bowlers
    if 'dot_pct' not in df.columns:
        if 'dots_bowled' in df.columns:
            df['dot_pct'] = np.where(df['balls_bowled'] > 0,
                                     (df['dots_bowled'] / df['balls_bowled']) * 100, 40)
        else:
            df['dot_pct'] = 40
    df['dot_pct'] = pd.to_numeric(df['dot_pct'], errors='coerce').fillna(40)
    df['dot_pressure_norm'] = percentile_rank(df['dot_pct'])

    # 8-component bowling impact formula — format-aware weights
    df['bowling_raw_impact'] = (
        w[0] * df['total_wpa_norm'] +
        w[1] * df['wicket_wpa_norm'] +
        w[2] * df['leverage_wpa_norm'] +
        w[3] * df['economy_wpa_norm'] +
        w[4] * df['clutch_wpa_norm'] +
        w[5] * df['spell_efficiency_norm'] +
        w[6] * df['phase_difficulty_norm'] +
        w[7] * df['dot_pressure_norm']
    )

    # Volume multiplier for bowlers — format-aware
    if fmt == 'Test':
        # Tests: bowl long spells, max overs ~30 per innings
        overs_frac = (df['balls_bowled'] / 90.0).clip(0, 1.25)
        wicket_bonus = np.minimum(df.get('wickets', pd.Series(0, index=df.index)).fillna(0) / 5.0, 1.0)
        df['volume_mult'] = 0.45 + 0.35 * overs_frac + 0.30 * wicket_bonus
    elif fmt == 'ODI':
        # ODIs: max 10 overs
        overs_frac = (df['balls_bowled'] / 50.0).clip(0, 1.25)
        wicket_bonus = np.minimum(df.get('wickets', pd.Series(0, index=df.index)).fillna(0) / 4.0, 1.0)
        df['volume_mult'] = 0.50 + 0.35 * overs_frac + 0.28 * wicket_bonus
    else:
        # T20: max 4 overs
        overs_frac = (df['balls_bowled'] / 24.0).clip(0, 1.25)
        wicket_bonus = np.minimum(df.get('wickets', pd.Series(0, index=df.index)).fillna(0) / 3.0, 1.0)
        df['volume_mult'] = 0.55 + 0.40 * overs_frac + 0.25 * wicket_bonus

    # Apply volume multiplier
    df['raw_impact'] = df['bowling_raw_impact'] * df['volume_mult']

    return df
