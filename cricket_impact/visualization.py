"""
Module 9: Visualization
Gauge charts, trend lines, comparisons, distributions, all-rounder scatter.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def gauge_chart(player_name, score, output_dir='cricket_impact'):
    """Current impact score gauge meter (0-100 with color zones)."""
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': 'polar'})

    # Gauge setup (half circle from pi to 0)
    score = np.clip(score, 0, 100)
    theta_range = np.pi  # half circle

    # Color zones
    zones = [
        (0, 40, '#ff4444', 'Low Impact'),
        (40, 60, '#ffaa00', 'Neutral'),
        (60, 80, '#44bb44', 'High Impact'),
        (80, 100, '#2266ff', 'Match Winner'),
    ]

    for lo, hi, color, label in zones:
        start = np.pi - (lo / 100) * theta_range
        end = np.pi - (hi / 100) * theta_range
        theta = np.linspace(start, end, 50)
        ax.fill_between(theta, 0.7, 1.0, color=color, alpha=0.3)

    # Needle
    needle_angle = np.pi - (score / 100) * theta_range
    ax.plot([needle_angle, needle_angle], [0, 0.9], color='black', linewidth=3)
    ax.plot(needle_angle, 0.9, 'ko', markersize=8)

    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'{player_name}\nImpact Score: {score:.1f}', fontsize=14, fontweight='bold', pad=20)

    # Legend
    patches = [mpatches.Patch(color=c, label=l, alpha=0.5) for _, _, c, l in zones]
    ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f'gauge_{player_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def trend_line(player_name, player_data, output_dir='cricket_impact'):
    """Impact trend over last innings with rolling average."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scores = player_data['impact_score'].values
    x = range(1, len(scores) + 1)

    ax.plot(x, scores, 'o-', color='#2266ff', linewidth=2, markersize=6, label='Impact Score')

    # Rolling average
    if len(scores) >= 3:
        rolling = pd.Series(scores).rolling(window=min(5, len(scores)), min_periods=1).mean().values
        ax.plot(x, rolling, '--', color='#ff6600', linewidth=2, label='Rolling Avg (5)')

    # Zone backgrounds
    ax.axhspan(0, 40, alpha=0.05, color='red')
    ax.axhspan(40, 60, alpha=0.05, color='orange')
    ax.axhspan(60, 80, alpha=0.05, color='green')
    ax.axhspan(80, 100, alpha=0.05, color='blue')

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Neutral (50)')

    ax.set_xlabel('Innings Number', fontsize=12)
    ax.set_ylabel('Impact Score', fontsize=12)
    ax.set_title(f'{player_name} - Impact Score Trend', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'trend_{player_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def player_comparison(players_data, output_dir='cricket_impact'):
    """Bar chart comparing multiple players' component scores."""
    fig, ax = plt.subplots(figsize=(12, 6))

    components = ['total_wpa_norm', 'presence_impact_norm', 'leverage_wpa_norm',
                  'clutch_wpa_norm', 'consistency_wpa_norm', 'runs_contribution_norm']
    # Use shorter labels
    comp_labels = ['Total WPA', 'Presence', 'Leverage', 'Clutch', 'Consistency', 'Runs Contrib']

    # Filter to available components
    available = [c for c in components if c in players_data.columns]
    avail_labels = [comp_labels[components.index(c)] for c in available]

    if not available:
        plt.close()
        return None

    players = players_data['player'].unique()[:8]
    x = np.arange(len(available))
    width = 0.8 / len(players)

    colors = plt.cm.Set2(np.linspace(0, 1, len(players)))

    for i, player in enumerate(players):
        pdata = players_data[players_data['player'] == player].iloc[0]
        vals = [pdata.get(c, 0) for c in available]
        ax.bar(x + i * width, vals, width, label=player, color=colors[i])

    ax.set_xticks(x + width * len(players) / 2)
    ax.set_xticklabels(avail_labels, rotation=30, ha='right')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Player Impact Component Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'player_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def impact_distribution(all_players, output_dir='cricket_impact'):
    """Histogram of impact scores across all players."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scores = all_players['overall_impact'].dropna()
    ax.hist(scores, bins=40, color='#2266ff', alpha=0.7, edgecolor='white')

    # Zone markers
    for threshold, label, color in [(40, 'Low/Neutral', 'red'),
                                     (60, 'Neutral/High', 'orange'),
                                     (80, 'High/Winner', 'green')]:
        ax.axvline(x=threshold, color=color, linestyle='--', linewidth=2, label=f'{label} ({threshold})')

    ax.set_xlabel('Impact Score', fontsize=12)
    ax.set_ylabel('Number of Players', fontsize=12)
    ax.set_title('Impact Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'impact_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def allrounder_scatter(all_players, output_dir='cricket_impact'):
    """Scatter plot: batting impact vs bowling impact for all-rounders."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Need players with both batting and bowling data
    df = all_players.copy()
    if 'mean_bat_impact' not in df.columns or 'mean_bowl_impact' not in df.columns:
        plt.close()
        return None

    # Filter players with meaningful contributions in both
    both = df[(df['mean_bat_impact'] > 0) & (df['mean_bowl_impact'] > 0)].copy()

    if len(both) == 0:
        plt.close()
        return None

    ax.scatter(both['mean_bat_impact'], both['mean_bowl_impact'],
               c=both['overall_impact'], cmap='viridis', s=60, alpha=0.7, edgecolors='gray')

    # Label top all-rounders
    top = both.nlargest(15, 'overall_impact')
    for _, row in top.iterrows():
        ax.annotate(row['player'], (row['mean_bat_impact'], row['mean_bowl_impact']),
                    fontsize=7, alpha=0.8, ha='left')

    ax.set_xlabel('Batting Impact', fontsize=12)
    ax.set_ylabel('Bowling Impact', fontsize=12)
    ax.set_title('All-Rounder Impact Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(ax.collections[0], label='Overall Impact')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'allrounder_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def generate_all_visualizations(bat_impact, bowl_impact, all_players, output_dir='cricket_impact'):
    """Generate all visualization charts."""
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    print("  Generating visualizations...")

    # 1. Impact distribution
    if 'overall_impact' in all_players.columns:
        path = impact_distribution(all_players, output_dir)
        if path:
            generated.append(path)

    # 2. All-rounder scatter
    path = allrounder_scatter(all_players, output_dir)
    if path:
        generated.append(path)

    # 3. Top player gauges
    if 'overall_impact' in all_players.columns:
        top5 = all_players.nlargest(5, 'overall_impact')
        for _, row in top5.iterrows():
            path = gauge_chart(row['player'], row['overall_impact'], output_dir)
            generated.append(path)

    # 4. Top player trends
    for _, row in all_players.nlargest(3, 'overall_impact').iterrows():
        p = row['player']
        pdata = bat_impact[bat_impact['player'] == p].copy()
        if len(pdata) < 3:
            pdata = bowl_impact[bowl_impact['player'] == p].copy()
        if len(pdata) >= 3 and 'impact_score' in pdata.columns:
            path = trend_line(p, pdata, output_dir)
            generated.append(path)

    # 5. Player comparison (top 6)
    if len(bat_impact) > 0 and 'total_wpa_norm' in bat_impact.columns:
        # Get latest innings for top players
        top_players = all_players.nlargest(6, 'overall_impact')['player'].values
        comp_data = bat_impact[bat_impact['player'].isin(top_players)].copy()
        if len(comp_data) > 0:
            comp_latest = comp_data.sort_values('start_date').drop_duplicates('player', keep='last')
            path = player_comparison(comp_latest, output_dir)
            if path:
                generated.append(path)

    print(f"  Generated {len(generated)} visualizations")
    return generated
