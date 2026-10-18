"""
Player rating (v2): one formula for career, season and role leaderboards.

Inputs are player-match rows (see build_player_match_context) carrying each player's
batting and bowling Win Probability Added (WPA) for that match plus result context.

    rating = 90% impact + 10% bonus, rescaled to 0-100

    impact = 50% * z(impact per innings, shrunk toward the league average)
           + 50% * z(career impact above a replacement-level player)

Shrinkage strength K is estimated from the data (empirical Bayes): per-match impact is
very noisy compared with the real spread between players, so a player needs about K
matches before their own average outweighs the league average.
Career value is measured above replacement (the fringe players a side would otherwise
pick), as in baseball's WAR, so a long career at a solid level earns credit.

    bonus  = 6% Player of the Match rate
           + 4/3% each: captaincy record, team win rate, share of seasons in the playoffs

The 0-100 scale is a min-max of the raw score over the players in the view, so real
gaps between players stay visible. Rank is reported separately.
"""
from typing import Optional

import numpy as np
import pandas as pd

SHRINK_K = 87          # default; estimate_shrink_k() recomputes it from the data each pipeline run
CAPTAIN_K = 10         # pseudo-matches at 50% when shrinking a captain's win rate
CAPTAIN_FULL = 50      # matches as captain for full captaincy credit
W_RATE, W_TOTAL = 0.50, 0.50
W_IMPACT = 0.90
W_POM = 0.06
W_OTHER = 0.04 / 3     # captaincy, team win rate, playoffs


def role_of(bat_inn: pd.Series, bowl_inn: pd.Series, min_inn: int = 10) -> pd.Series:
    """Batter / Bowler / All-rounder from innings counts (same rule as the original app)."""
    b, w = bat_inn.fillna(0), bowl_inn.fillna(0)
    lo, hi = np.minimum(b, w), np.maximum(b, w).replace(0, np.nan)
    allround = (b >= min_inn) & (w >= min_inn) & ((lo / hi) >= 0.4)
    return pd.Series(np.where(allround, 'All-rounder', np.where(b >= w, 'Batter', 'Bowler')), index=b.index)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5


def summarize(ctx: pd.DataFrame, role_min_inn: int = 10) -> pd.DataFrame:
    """Per-player totals and standard cricket stats from player-match rows."""
    g = ctx.groupby('player').agg(
        appearances=('match_id', 'nunique'),
        bat_innings=('bat_inn', 'sum'), bowl_innings=('bowl_inn', 'sum'),
        bat_wpa=('bat_wpa', 'sum'), bowl_wpa=('bowl_wpa', 'sum'),
        runs=('runs', 'sum'), balls_faced=('balls_faced', 'sum'), dismissals=('dismissed', 'sum'),
        fours=('fours', 'sum'), sixes=('sixes', 'sum'),
        wickets=('wickets', 'sum'), runs_conceded=('runs_conceded', 'sum'), balls_bowled=('balls_bowled', 'sum'),
        wins=('won', 'sum'), pom=('is_pom', 'sum'),
        captain_matches=('is_captain', 'sum'), captain_wins=('captain_won', 'sum'),
        seasons=('season', 'nunique'),
        team=('team', lambda s: s.value_counts().index[0]),
    )
    po = ctx[ctx['team_playoffs'] == 1].groupby('player')['season'].nunique()
    g['playoff_seasons'] = po.reindex(g.index).fillna(0)
    g['role'] = role_of(g['bat_innings'], g['bowl_innings'], role_min_inn)
    # standard stats from totals
    g['strike_rate'] = np.where(g['balls_faced'] > 0, g['runs'] / g['balls_faced'] * 100, np.nan)
    g['bat_avg'] = np.where(g['dismissals'] > 0, g['runs'] / g['dismissals'], np.where(g['runs'] > 0, g['runs'], np.nan))
    g['economy'] = np.where(g['balls_bowled'] > 0, g['runs_conceded'] / (g['balls_bowled'] / 6), np.nan)
    g['win_rate'] = np.where(g['appearances'] > 0, g['wins'] / g['appearances'], 0)
    return g


def discipline_means(ctx: pd.DataFrame) -> dict:
    """League-average WPA per batting innings and per bowling innings."""
    return {
        'bat': float(ctx['bat_wpa'].sum() / max(ctx['bat_inn'].sum(), 1)),
        'bowl': float(ctx['bowl_wpa'].sum() / max(ctx['bowl_inn'].sum(), 1)),
    }


def center(ctx: pd.DataFrame, means: Optional[dict] = None) -> pd.DataFrame:
    """
    Express WPA relative to an average batting innings and an average bowling spell.
    The win probability model can leave batting and bowling credit on different baselines
    (e.g. batting averaging slightly negative); centring each discipline puts batters and
    bowlers on the same footing before they are compared in one table.
    """
    m = discipline_means(ctx) if means is None else means
    out = ctx.copy()
    out['bat_wpa'] = out['bat_wpa'] - out['bat_inn'] * m['bat']
    out['bowl_wpa'] = out['bowl_wpa'] - out['bowl_inn'] * m['bowl']
    return out


def estimate_shrink_k(ctx: pd.DataFrame, min_matches: int = 10) -> float:
    """Empirical-Bayes K = within-player variance of per-match impact / between-player variance."""
    x = ctx.assign(w=ctx['bat_wpa'] + ctx['bowl_wpa'])
    g = x.groupby('player')['w'].agg(['mean', 'var', 'count'])
    g = g[g['count'] >= min_matches]
    if len(g) < 20:
        return float(SHRINK_K)
    sig2 = (g['var'] * (g['count'] - 1)).sum() / (g['count'] - 1).sum()
    tau2 = g['mean'].var() - (sig2 / g['count']).mean()
    return float(np.clip(sig2 / tau2, 5, 500)) if tau2 > 0 else float(SHRINK_K)


def replacement_level(ctx: pd.DataFrame, view: str = 'all', max_matches: int = 25) -> float:
    """
    WPA per innings of a replacement-level player: the pooled average of fringe players
    (fewer than max_matches career appearances), i.e. who a side picks when its regular is missing.
    """
    app = ctx.groupby('player')['match_id'].nunique()
    fringe = ctx[ctx['player'].map(app) < max_matches]
    if view == 'bat':
        n, w = fringe['bat_inn'].sum(), fringe['bat_wpa'].sum()
    elif view == 'bowl':
        n, w = fringe['bowl_inn'].sum(), fringe['bowl_wpa'].sum()
    else:
        n, w = len(fringe), (fringe['bat_wpa'] + fringe['bowl_wpa']).sum()
    return float(w / n) if n else 0.0


def rate(ctx: pd.DataFrame, view: str = 'all', role_min_inn: int = 10,
         players: Optional[pd.DataFrame] = None, shrink_k: Optional[float] = None,
         w_rate: float = None, w_total: float = None, repl: Optional[float] = None) -> pd.DataFrame:
    """
    Rate players for one view: 'all', 'bat', 'bowl' or 'allround'.
    ctx: player-match rows already filtered to the season(s) of interest.
    Returns one row per eligible player with rating (0-100), rank and components.
    """
    g = summarize(ctx, role_min_inn) if players is None else players.copy()

    if view == 'bat':
        g = g[g['role'].isin(['Batter', 'All-rounder']) & (g['bat_innings'] > 0)].copy()
        n, tot = g['bat_innings'], g['bat_wpa']
    elif view == 'bowl':
        g = g[g['role'].isin(['Bowler', 'All-rounder']) & (g['bowl_innings'] > 0)].copy()
        n, tot = g['bowl_innings'], g['bowl_wpa']
    elif view == 'allround':
        g = g[g['role'] == 'All-rounder'].copy()
        n, tot = g['appearances'], g['bat_wpa'] + g['bowl_wpa']
    else:
        g = g[g['appearances'] > 0].copy()
        n, tot = g['appearances'], g['bat_wpa'] + g['bowl_wpa']
    if g.empty:
        return g

    k = SHRINK_K if shrink_k is None else shrink_k
    wr = W_RATE if w_rate is None else w_rate
    wt = W_TOTAL if w_total is None else w_total
    league = tot.sum() / max(n.sum(), 1)
    r0 = replacement_level(ctx, view) if repl is None else repl
    g['innings_rated'] = n
    g['impact_total'] = tot
    g['impact_per_inn'] = tot / n.where(n > 0, np.nan)
    g['impact_per_inn_shrunk'] = (tot + k * league) / (n + k)
    # career value above a replacement-level player (WAR-style), using the shrunk rate
    g['replacement_level'] = r0
    g['impact_above_repl'] = (g['impact_per_inn_shrunk'] - r0) * n
    g['impact_blend'] = wr * _z(g['impact_per_inn_shrunk']) + wt * _z(g['impact_above_repl'])
    g['impact_norm'] = _minmax(g['impact_blend'])

    # bonus parts, each 0-1, all shrunk so tiny samples can't spike
    app = g['appearances']
    lg_pom = g['pom'].sum() / max(app.sum(), 1)
    pom_rate = (g['pom'] + SHRINK_K * lg_pom) / (app + SHRINK_K)
    g['pom_norm'] = pom_rate / pom_rate.max() if pom_rate.max() > 0 else 0
    cap_wr = (g['captain_wins'] + CAPTAIN_K * 0.5) / (g['captain_matches'] + CAPTAIN_K)
    cap = np.where(g['captain_matches'] > 0, cap_wr * np.minimum(1, g['captain_matches'] / CAPTAIN_FULL), 0)
    g['captain_norm'] = cap / cap.max() if cap.max() > 0 else 0
    g['team_win_norm'] = _minmax((g['wins'] + SHRINK_K * 0.5) / (app + SHRINK_K))
    g['playoff_share'] = np.where(g['seasons'] > 0, g['playoff_seasons'] / g['seasons'], 0)

    g['bonus'] = (W_POM * g['pom_norm'] + W_OTHER * (g['captain_norm'] + g['team_win_norm'] + g['playoff_share']))
    g['rating_raw'] = W_IMPACT * g['impact_norm'] + g['bonus']
    g['rating'] = _minmax(g['rating_raw']) * 100
    g['rank'] = g['rating'].rank(ascending=False, method='min').astype(int)
    return g.sort_values('rating', ascending=False)


def build_player_match_context(ball_df: pd.DataFrame, info_df: pd.DataFrame,
                               bat_impact: pd.DataFrame, bowl_impact: pd.DataFrame,
                               captains: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per player per match: WPA, basic stats and result context."""
    bat = ball_df[['match_id', 'batting_team', 'striker']].rename(columns={'striker': 'player', 'batting_team': 'team'})
    nst = ball_df[['match_id', 'batting_team', 'non_striker']].rename(columns={'non_striker': 'player', 'batting_team': 'team'}) \
        if 'non_striker' in ball_df.columns else bat.iloc[0:0]
    bowl = ball_df[['match_id', 'bowling_team', 'bowler']].rename(columns={'bowler': 'player', 'bowling_team': 'team'})
    ctx = pd.concat([bat, nst, bowl], ignore_index=True).drop_duplicates(['match_id', 'player'])

    info = info_df[['match_id', 'season', 'winner', 'player_of_match', 'is_playoff']].copy()
    info['season'] = info['season'].astype(str)
    ctx = ctx.merge(info, on='match_id', how='left')
    ctx['won'] = (ctx['team'] == ctx['winner']).astype(int)
    ctx['is_pom'] = (ctx['player'] == ctx['player_of_match']).astype(int)

    # team reached the playoffs that season
    po = ctx[ctx['is_playoff'] == 1][['season', 'team']].drop_duplicates().assign(team_playoffs=1)
    ctx = ctx.merge(po, on=['season', 'team'], how='left')
    ctx['team_playoffs'] = ctx['team_playoffs'].fillna(0).astype(int)

    if captains is not None and len(captains):
        cap = captains[['match_id', 'team', 'captain']].rename(columns={'captain': 'player'}).assign(is_captain=1)
        ctx = ctx.merge(cap, on=['match_id', 'team', 'player'], how='left')
    ctx['is_captain'] = ctx.get('is_captain', 0)
    ctx['is_captain'] = pd.Series(ctx['is_captain']).fillna(0).astype(int)
    ctx['captain_won'] = ctx['is_captain'] * ctx['won']

    b = bat_impact.groupby(['match_id', 'player']).agg(
        bat_inn=('total_wpa', 'size'), bat_wpa=('total_wpa', 'sum'), runs=('runs', 'sum'),
        balls_faced=('balls_faced', 'sum'), dismissed=('was_dismissed', 'sum'),
        fours=('fours', 'sum'), sixes=('sixes', 'sum')).reset_index()
    w = bowl_impact.groupby(['match_id', 'player']).agg(
        bowl_inn=('total_wpa', 'size'), bowl_wpa=('total_wpa', 'sum'), wickets=('wickets', 'sum'),
        runs_conceded=('runs_conceded', 'sum'), balls_bowled=('balls_bowled', 'sum')).reset_index()
    ctx = ctx.merge(b, on=['match_id', 'player'], how='left').merge(w, on=['match_id', 'player'], how='left')
    num = ['bat_inn', 'bat_wpa', 'runs', 'balls_faced', 'dismissed', 'fours', 'sixes',
           'bowl_inn', 'bowl_wpa', 'wickets', 'runs_conceded', 'balls_bowled']
    ctx[num] = ctx[num].fillna(0)
    keep = ['match_id', 'season', 'player', 'team', 'won', 'is_pom', 'is_captain', 'captain_won',
            'team_playoffs'] + num
    return ctx[keep]
