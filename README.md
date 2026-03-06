# Cricket Player Impact Metric

A data-driven system that quantifies cricket player performance using **Win Probability Added (WPA)**, machine learning, and multi-factor impact scoring. Built for the CricHeroes HackaMined Data Science Hackathon.

## Overview

The pipeline processes ball-by-ball match data to produce holistic player impact scores (0–90 scale) that go beyond traditional stats. It uses a **Random Forest win probability model** to compute each delivery's impact on match outcome, then aggregates across batting, bowling, partnerships, and contextual factors.

### Key Features

- **WPA-Based Scoring** — Every ball's impact measured via ML win probability shifts
- **Format-Aware** — Separate weight profiles and player rankings for T20, ODI, and Test
- **Multi-Factor Impact** — Batting, bowling, all-rounder bonus, captaincy, clutch performance, dot ball efficiency, phase difficulty, partnerships, and more
- **Rolling Normalization** — Season-aware normalization so players are compared fairly across eras
- **Per-Format Player Lists** — Independent rankings for each format (T20, ODI, Test)

## Project Structure

```
├── run_pipeline.py                  # Main orchestrator (CLI entry point)
├── cricket_impact/
│   ├── data_loader.py               # Load & clean ball-by-ball + info CSVs, format detection
│   ├── player_aggregation.py        # Aggregate batting/bowling stats per match-innings
│   ├── context_features.py          # Venue, opposition strength, pressure, leverage, momentum
│   ├── win_prob_model.py            # Random Forest win probability model (train/predict)
│   ├── wpa_engine.py                # Per-ball WPA, batter/bowler WPA aggregation, clutch WPA
│   ├── partnerships.py              # Batting partnerships, bowling tandems, batter-vs-bowler rivalry
│   ├── impact_metric.py             # Format-aware batting & bowling impact formulas
│   ├── rolling_normalization.py     # Season-based rolling normalization (0–100)
│   ├── captaincy.py                 # Captaincy stats, win bonus, playoff team bonus
│   └── visualization.py             # Gauge meters, trends, scatter plots
├── ipl_male_csv2/                   # Ball-by-ball CSVs (not tracked in git)
└── Impact metric hackathon challenge_cricheros.pdf
```

## Setup

### Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Data

Place ball-by-ball CSV files in a data folder (e.g. `ipl_male_csv2/`). Each match needs:
- `{match_id}.csv` — ball-by-ball data (striker, bowler, runs, wickets, etc.)
- `{match_id}_info.csv` — match metadata (teams, venue, winner, player of match, etc.)

## Usage

```bash
# Run with default data folder
python run_pipeline.py --data_folder ./ipl_male_csv2

# Force retrain the ML model
python run_pipeline.py --data_folder ./ipl_male_csv2 --retrain

# Custom output directory
python run_pipeline.py --data_folder ./ipl_male_csv2 --output_dir ./output
```

## Output Files

| File | Description |
|------|-------------|
| `all_players_impact_{format}.csv` | Per-format player rankings (e.g. `all_players_impact_T20.csv`) |
| `all_players_impact.csv` | Combined cross-format player rankings |
| `batting_impact.csv` | Per-match batting impact scores |
| `bowling_impact.csv` | Per-match bowling impact scores |
| `match_rankings.csv` | Per-match player rankings |
| `batting_partnerships.csv` | Partnership analysis |
| `bowling_partnerships.csv` | Bowling tandem analysis |
| `batter_vs_bowler_rivalry.csv` | Head-to-head matchup stats |

## Impact Scoring Components

### Batting Impact (7 components)
WPA contribution, wicket-preserving WPA, leverage-weighted WPA, economy WPA, clutch WPA, batting position multiplier, dot ball efficiency

### Bowling Impact (8 components)
Total WPA, wicket WPA, leverage WPA, economy WPA, clutch WPA, spell efficiency, death/powerplay phase difficulty, dot ball pressure

### Overall Impact (11 components)
Role-weighted performance, all-rounder bonus, longevity, Player of Match awards, POM rate, clutch contribution, volume, win bonus, captaincy bonus, playoff team bonus, bowling visibility

### Format-Specific Weights
Each format (T20/ODI/Test) uses different weight profiles:
- **T20**: Strike rate and boundary hitting emphasized
- **ODI**: Balance between accumulation and acceleration
- **Test**: Runs scored and balls faced prioritized over strike rate

## Pipeline Steps

1. **Load & Clean** — Parse ball-by-ball CSVs, detect match format
2. **Aggregate** — Per-player per-match batting and bowling stats
3. **Context Features** — Venue difficulty, opposition strength, pressure index
4. **Win Probability Model** — Train/load Random Forest classifier
5. **WPA Computation** — Per-ball win probability changes
6. **Partnerships** — Batting partnerships, bowling tandems, rivalries
7. **Impact Scoring** — Format-aware batting and bowling impact formulas
8. **Rolling Normalization** — Season-aware 0–100 scaling
9. **Career Summary** — Per-format player rankings with Bayesian dampening
10. **Visualizations** — Gauge meters, trend charts, comparison plots

## Sample Results (IPL T20)

| Rank | Player | Overall Impact |
|------|--------|---------------|
| 1 | RG Sharma | 90.0 |
| 2 | V Kohli | 89.9 |
| 3 | SP Narine | 89.8 |
| 4 | RA Jadeja | 89.8 |
| 5 | MS Dhoni | 89.7 |

*Based on 1,169 IPL matches, 278,205 deliveries, 768 players*
