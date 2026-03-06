"""
Module 8: Rolling Last-10-Innings Normalization
Recency-weighted rolling impact, 0-100 scale with Bayesian dampening.
"""

import pandas as pd
import numpy as np


RECENCY_WEIGHTS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


def compute_rolling_impact(player_df, impact_col='raw_impact', window=10):
    """
    Compute recency-weighted rolling impact over last N innings per player.
    player_df must have columns: player, start_date, match_id, raw_impact.
    """
    df = player_df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df.sort_values(['player', 'start_date', 'match_id']).reset_index(drop=True)

    results = []
    for player, grp in df.groupby('player'):
        grp = grp.sort_values(['start_date', 'match_id']).reset_index(drop=True)
        impacts = grp[impact_col].values
        n = len(impacts)

        rolling_vals = []
        consistency_vals = []
        comeback_vals = []

        for i in range(n):
            start = max(0, i - window + 1)
            recent = impacts[start:i + 1]
            k = len(recent)

            # Get recency weights for this window
            weights = RECENCY_WEIGHTS[:k][::-1]  # most recent gets highest weight
            weights = weights / weights.sum()

            weighted_avg = np.average(recent, weights=weights)
            rolling_vals.append(weighted_avg)

            # Consistency modifier: 1 - (std/mean) — consistent players rewarded
            if k >= 3 and np.mean(recent) > 0:
                consistency = max(0, 1 - (np.std(recent) / (np.mean(recent) + 1e-8)))
            else:
                consistency = 0.5
            consistency_vals.append(consistency)

            # Comeback bonus: recent 5 > prior 5 by >20%
            if k >= 10:
                recent_5 = np.mean(recent[-5:])
                prior_5 = np.mean(recent[:5])
                if prior_5 > 0 and (recent_5 - prior_5) / prior_5 > 0.2:
                    comeback_vals.append(1.1)
                else:
                    comeback_vals.append(1.0)
            else:
                comeback_vals.append(1.0)

        grp['rolling_impact'] = rolling_vals
        grp['consistency_modifier'] = consistency_vals
        grp['comeback_bonus'] = comeback_vals
        results.append(grp)

    if not results:
        return df

    result_df = pd.concat(results, ignore_index=True)

    # Apply consistency and comeback
    result_df['adjusted_impact'] = (
        result_df['rolling_impact'] *
        (0.8 + 0.2 * result_df['consistency_modifier']) *
        result_df['comeback_bonus']
    )

    return result_df


def bayesian_dampening(df, impact_col='adjusted_impact', prior_mean=0.35, min_innings=5):
    """
    Bayesian shrinkage: pull toward prior_mean for players with few innings.
    """
    df = df.copy()

    # Count innings per player up to this point
    df['innings_count'] = df.groupby('player').cumcount() + 1

    # Bayesian weight: more innings → more trust in observed impact
    df['bayes_weight'] = df['innings_count'] / (df['innings_count'] + min_innings)
    df['dampened_impact'] = (
        df['bayes_weight'] * df[impact_col] +
        (1 - df['bayes_weight']) * prior_mean
    )

    return df


def normalize_to_100(df, impact_col='dampened_impact'):
    """
    Normalize to 0-100 scale using percentile + logistic normalization.
    """
    df = df.copy()

    # Percentile rank (0-1)
    df['impact_pctile'] = df[impact_col].rank(pct=True, method='average')

    # Logistic transformation centered at 0.5
    def logistic_scale(x, center=0.5, steepness=8):
        return 100 / (1 + np.exp(-steepness * (x - center)))

    df['impact_score'] = df['impact_pctile'].apply(logistic_scale)

    # Categorize
    def categorize(score):
        if score < 40:
            return 'Low Impact'
        elif score < 60:
            return 'Neutral'
        elif score < 80:
            return 'High Impact'
        else:
            return 'Match Winner'

    df['impact_category'] = df['impact_score'].apply(categorize)

    return df


def apply_rolling_normalization(player_df, impact_col='raw_impact'):
    """Full pipeline: rolling → Bayesian → 0-100 normalization."""
    df = compute_rolling_impact(player_df, impact_col=impact_col)
    df = bayesian_dampening(df)
    df = normalize_to_100(df)
    return df
