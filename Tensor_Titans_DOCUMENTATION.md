# CricImpact — Complete Project Documentation

## Manual / Report / Documentation

> A data-driven cricket analytics platform that quantifies player performance using **Win Probability Added (WPA)**, machine learning, and multi-factor impact scoring. Built for the **CricHeroes HackaMined Data Science Hackathon**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation & Setup](#2-installation--setup)
3. [Data Requirements](#3-data-requirements)
4. [Execution Instructions](#4-execution-instructions)
5. [Project Architecture](#5-project-architecture)
6. [Core Data Pipeline (15 Steps)](#6-core-data-pipeline-15-steps)
7. [Module-by-Module Feature Documentation](#7-module-by-module-feature-documentation)
   - 7.1 [Data Loader](#71-data-loader-data_loaderpy)
   - 7.2 [Player Aggregation](#72-player-aggregation-player_aggregationpy)
   - 7.3 [Context Features](#73-context-features-context_featurespy)
   - 7.4 [Win Probability Model](#74-win-probability-model-win_prob_modelpy)
   - 7.5 [WPA Engine](#75-wpa-engine-wpa_enginepy)
   - 7.6 [Partnerships & Rivalry](#76-partnerships--rivalry-partnershipspy)
   - 7.7 [Impact Metric](#77-impact-metric-impact_metricpy)
   - 7.8 [Rolling Normalization](#78-rolling-normalization-rolling_normalizationpy)
   - 7.9 [Captaincy & Win Bonus](#79-captaincy--win-bonus-captaincypy)
   - 7.10 [Visualization](#710-visualization-visualizationpy)
8. [Impact Scoring — Detailed Formulas](#8-impact-scoring--detailed-formulas)
9. [Overall Impact — 11-Factor Composite Score](#9-overall-impact--11-factor-composite-score)
10. [Streamlit Dashboard (app.py)](#10-streamlit-dashboard-apppy)
11. [Output Files Reference](#11-output-files-reference)
12. [Troubleshooting & FAQ](#12-troubleshooting--faq)
13. [Sample Results](#13-sample-results)
14. [Thought Process & Cricket Logic — Step-by-Step Derivation](#14-thought-process--cricket-logic--step-by-step-derivation-of-the-impact-metric)
    - 14.1 [Why WPA Instead of Traditional Stats?](#141-why-win-probability-added-wpa-instead-of-traditional-stats)
    - 14.2 [The ML Win Probability Model — Why XGBoost with 13 Features?](#142-the-ml-win-probability-model--why-xgboost-with-these-13-features)
    - 14.3 [Context Features — Why Each One Matters](#143-context-features--why-each-one-matters-in-real-cricket)
    - 14.4 [WPA Attribution — Crediting Batters and Bowlers](#144-wpa-attribution--why-we-credit-both-batters-and-bowlers-differently)
    - 14.5 [Batting Impact — Why 7 Components](#145-batting-impact--why-7-components-and-these-specific-weights)
    - 14.6 [Why Format-Specific Weights?](#146-why-format-specific-weights)
    - 14.7 [Position Multiplier](#147-position-multiplier--rewarding-over-performance-vs-expected-contribution)
    - 14.8 [Volume Multiplier](#148-volume-multiplier--why-raw-impact-needs-scaling-by-contribution-volume)
    - 14.9 [Bowling Impact — Why 8 Components](#149-bowling-impact--why-8-components)
    - 14.10 [Rolling Normalization — Recency + Bayesian](#1410-rolling-normalization--why-recency-weighted-with-bayesian-dampening)
    - 14.11 [Overall Impact — Why 11 Factors](#1411-overall-impact-090--why-11-factors-and-these-specific-weights)
    - 14.12 [Final Score: Dampening + Scaling](#1412-final-score-bayesian-dampening--power-curve-scaling-090)
    - 14.13 [Cricket Scenarios — How CricImpact Handles Them](#1413-summary-of-cricket-scenarios-and-how-cricimpact-handles-them)

---

## 1. Project Overview

**CricImpact** is a comprehensive cricket analytics system that goes far beyond traditional batting averages and bowling strike rates. It processes **ball-by-ball match data** to produce holistic player impact scores on a **0–90 scale**, capturing the true match-changing ability of every player.

### What Makes CricImpact Different?

| Traditional Stats | CricImpact Approach |
|---|---|
| Batting average, strike rate | Win Probability Added (WPA) per delivery |
| Bowling economy, wickets | Context-aware impact (pressure, leverage, phase) |
| Static career aggregates | Recency-weighted rolling scores with Bayesian dampening |
| One-size-fits-all | Format-specific weights (T20 / ODI / Test) |
| Individual stats only | Partnerships, rivalries, captaincy, team context |

### Key Capabilities

- **WPA-Based Scoring** — Every ball's impact measured via ML-driven win probability shifts
- **Format-Aware Weights** — Separate weight profiles for T20, ODI, and Test cricket
- **Multi-Factor Impact** — 11 components: batting, bowling, all-rounder bonus, captaincy, clutch performance, dot ball efficiency, phase difficulty, partnerships, win bonus, playoff bonus, and more
- **Rolling Normalization** — Recency-weighted normalization so players are compared fairly across seasons
- **Per-Format Rankings** — Independent rankings generated for each format
- **Interactive Dashboard** — Dark-themed Streamlit UI with Plotly charts, radar plots, head-to-head comparison, and match explorer
- **Partnership Analysis** — Batting partnerships, bowling tandems, and batter-vs-bowler rivalry tracking
- **Captaincy Evaluation** — Captain-specific bonuses based on win rate, playoff appearances, and leadership impact

### Scale of Analysis

- **1,169+ IPL matches** processed
- **278,205+ deliveries** analyzed
- **768+ players** ranked
- **XGBoost ML model** for win probability prediction

---

## 2. Installation & Setup

### 2.1 Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10 or higher |
| **pip** | Latest recommended |
| **OS** | Windows / macOS / Linux |
| **RAM** | 4 GB minimum (8 GB recommended for large datasets) |
| **Disk** | ~500 MB for data + models |

### 2.2 Step-by-Step Installation

#### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository_url>
cd HackaMined-Data-Science

# Or download and extract the ZIP file, then navigate to the folder
cd "HackaMined-Data Science"
```

#### Step 2: (Recommended) Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib plotly streamlit joblib
```

**Full dependency list:**

| Package | Purpose | Minimum Version |
|---|---|---|
| `pandas` | Data manipulation and CSV processing | 1.5+ |
| `numpy` | Numerical computation | 1.23+ |
| `scikit-learn` | ML models (Random Forest, Logistic Regression, calibration) | 1.2+ |
| `xgboost` | Primary win probability classifier | 1.7+ |
| `matplotlib` | Static chart generation (gauges, trends) | 3.6+ |
| `plotly` | Interactive charts in Streamlit dashboard | 5.15+ |
| `streamlit` | Web-based interactive dashboard | 1.28+ |
| `joblib` | Model serialization (save/load `.pkl` files) | 1.2+ |

> **Note:** If `xgboost` cannot be installed (e.g., on certain ARM architectures), the system automatically falls back to `Random Forest` as the win probability model.

#### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, matplotlib, plotly, streamlit, joblib; print('All dependencies installed successfully!')"
```

### 2.3 Data Setup

Place the ball-by-ball match data in a folder (default: `ipl_male_csv2/`). The data comes from [Cricsheet](https://cricsheet.org/) in CSV format.

Each match requires **two CSV files**:
- `{match_id}.csv` — Ball-by-ball delivery data (striker, bowler, runs, wickets, etc.)
- `{match_id}_info.csv` — Match metadata (teams, venue, winner, date, player of match, etc.)

**Example:**
```
ipl_male_csv2/
├── 1082591.csv          # Ball-by-ball data for match 1082591
├── 1082591_info.csv     # Metadata for match 1082591
├── 1082592.csv
├── 1082592_info.csv
├── ...                  # (1,169+ match pairs)
```

---

## 3. Data Requirements

### 3.1 Ball-by-Ball CSV Format (`{match_id}.csv`)

Each row represents a single delivery. Key columns:

| Column | Description | Example |
|---|---|---|
| `match_id` | Unique match identifier | 1082591 |
| `innings` | Innings number (1 or 2) | 1 |
| `ball` | Over.ball notation | 0.1, 0.2, ..., 19.6 |
| `batting_team` | Team batting | Mumbai Indians |
| `bowling_team` | Team bowling | Chennai Super Kings |
| `striker` | Batter on strike | V Kohli |
| `non_striker` | Batter at non-striker end | RG Sharma |
| `bowler` | Bowler delivering | JJ Bumrah |
| `runs_off_bat` | Runs scored off bat | 4 |
| `extras` | Extra runs (wides, no-balls, etc.) | 1 |
| `wicket_type` | Type of dismissal (if any) | caught |
| `player_dismissed` | Player who got out (if any) | SK Raina |

### 3.2 Match Info CSV Format (`{match_id}_info.csv`)

Two-column format with key-value pairs:

| Key | Value Example |
|---|---|
| `team` | Mumbai Indians |
| `team` | Chennai Super Kings |
| `venue` | Wankhede Stadium |
| `winner` | Mumbai Indians |
| `player_of_match` | V Kohli |
| `date` | 2023-04-09 |
| `season` | 2023 |
| `match_number` | 1 |
| `event` | Indian Premier League |

### 3.3 Format Detection

The system **automatically detects match format** based on the data:
- **T20** — Max 20 overs per innings
- **ODI** — Max 50 overs per innings
- **Test** — More than 50 overs or 4 innings

---

## 4. Execution Instructions

### 4.1 Running the Full Pipeline

The pipeline processes raw CSV data and generates all impact scores, rankings, and visualizations.

```bash
# Basic run (uses default data folder)
python run_pipeline.py --data_folder ./ipl_male_csv2

# Force re-training of the ML model (use when data changes significantly)
python run_pipeline.py --data_folder ./ipl_male_csv2 --retrain

# Specify custom output directory
python run_pipeline.py --data_folder ./ipl_male_csv2 --output_dir ./output
```

**Command-Line Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_folder` | Yes | — | Path to folder containing match CSV files |
| `--retrain` | No | `False` | Force re-training the win probability model |
| `--output_dir` | No | `cricket_impact` | Directory to save output CSVs and models |

**Pipeline Output (printed to console):**
```
Step  1/15  Loading & cleaning data from ./ipl_male_csv2 ...
            → 278,205 deliveries across 1,169 matches
Step  2/15  Aggregating batting stats ...
Step  3/15  Merging runs contribution ...
Step  4/15  Adding context features ...
Step  5/15  Training win probability model ...
            → Model: XGBoost | AUC: 0.72 | Brier: 0.21
Step  6/15  Predicting win probabilities ...
Step  7/15  Computing per-ball WPA ...
Step  8/15  Computing partnerships & rivalries ...
Step  9/15  Computing batting impact scores ...
Step 10/15  Computing bowling impact scores ...
Step 11/15  Applying rolling normalization ...
Step 12/15  Saving per-match CSVs ...
Step 13/15  Building match rankings ...
Step 14/15  Building all-players career summary ...
Step 15/15  Generating visualizations ...
Pipeline complete. Files saved to cricket_impact/
```

### 4.2 Launching the Streamlit Dashboard

After the pipeline has been run (and CSV output files exist in `cricket_impact/`), launch the interactive dashboard:

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` with the full CricImpact dashboard.

> **Important:** The dashboard reads pre-computed CSV files from the `cricket_impact/` directory. You must run the pipeline first before launching the dashboard.

### 4.3 Verifying Rankings

Use the validation script to spot-check output integrity:

```bash
python check_rankings.py
```

### 4.4 Execution Order Summary

```
1. Place data in ipl_male_csv2/ (or your chosen folder)
2. python run_pipeline.py --data_folder ./ipl_male_csv2
3. (Wait for pipeline to complete — outputs saved to cricket_impact/)
4. streamlit run app.py
5. Open http://localhost:8501 in your browser
```

---

## 5. Project Architecture

### 5.1 Directory Structure

```
HackaMined-Data Science/
│
├── app.py                              # Streamlit dashboard (~1000 lines)
├── run_pipeline.py                     # Pipeline orchestrator, CLI entry point (~800 lines)
├── check_rankings.py                   # Validation/ranking verification script
├── README.md                           # Quick-start documentation
├── README.txt                          # Data source info (Cricsheet)
│
├── cricket_impact/                     # Main Python package
│   ├── __init__.py                     # Package initializer
│   ├── data_loader.py                  # Module 1: Data ingestion & cleaning
│   ├── player_aggregation.py           # Module 2: Per-player per-match stats
│   ├── context_features.py             # Module 3: Context engineering (7 features)
│   ├── win_prob_model.py               # Module 4: ML win probability model
│   ├── wpa_engine.py                   # Module 5: Win Probability Added computation
│   ├── partnerships.py                 # Module 6: Partnerships & rivalry analysis
│   ├── impact_metric.py               # Module 7: Format-aware impact formulas
│   ├── rolling_normalization.py        # Module 8: Recency-weighted normalization
│   ├── captaincy.py                    # Module 9: Captaincy stats & bonuses
│   ├── visualization.py               # Module 10: Static chart generation
│   │
│   ├── models/
│   │   └── win_prob_model.pkl          # Serialized ML model (auto-generated)
│   │
│   ├── all_players_impact_T20.csv      # Career rankings (T20 format)
│   ├── all_players_impact.csv          # Combined cross-format rankings
│   ├── batting_impact.csv              # Per-match batting impact scores
│   ├── bowling_impact.csv              # Per-match bowling impact scores
│   ├── match_rankings.csv              # Per-match player rankings
│   ├── batting_partnerships.csv        # Partnership analysis output
│   ├── bowling_partnerships.csv        # Bowling tandem analysis output
│   └── batter_vs_bowler_rivalry.csv    # Head-to-head matchup stats
│
└── ipl_male_csv2/                      # Raw data folder (not tracked in Git)
    ├── 1082591.csv                     # Ball-by-ball data
    ├── 1082591_info.csv                # Match metadata
    └── ...                             # (1,169+ match pairs)
```

### 5.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAW DATA (ipl_male_csv2/)                         │
│                  Ball-by-ball CSVs + Match Info CSVs                       │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   1. DATA LOADER      │
                    │   load_and_prepare()  │
                    │   278K+ deliveries    │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                   │
    ┌─────────▼────────┐ ┌─────▼──────────┐ ┌──────▼──────────┐
    │ 2. BATTING AGG   │ │ 3. BOWLING AGG │ │ 4. CONTEXT      │
    │ runs, SR, dots,  │ │ wkts, econ,    │ │ venue, pressure,│
    │ bf, position     │ │ dots, phases   │ │ leverage,       │
    └────────┬─────────┘ └──────┬─────────┘ │ momentum        │
             │                  │            └────────┬────────┘
             └──────────┬───────┘                     │
                        │                             │
              ┌─────────▼─────────────────────────────▼───────────┐
              │          5. WIN PROBABILITY MODEL (ML)            │
              │   XGBoost (300 trees, depth=6) → P(win) per ball  │
              └──────────────────────┬────────────────────────────┘
                                     │
              ┌──────────────────────▼───────────┐
              │     6. WPA ENGINE                │
              │  ΔP(win) per ball for batters    │
              │  & bowlers, clutch WPA,          │
              │  leverage WPA, presence impact   │
              └──────────┬──────────┬────────────┘
                         │          │
           ┌─────────────▼──┐  ┌───▼──────────────┐
           │ 7. PARTNERSHIPS│  │ 8. IMPACT METRIC │
           │ pairs, tandems,│  │ 7-factor batting │
           │ H2H rivalry    │  │ 8-factor bowling │
           └────────────────┘  └────────┬─────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │ 9. ROLLING NORMALIZATION    │
                         │ Recency weights, Bayesian   │
                         │ dampening, logistic 0-100   │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────▼─────────────────────────┐
              │        10. CAREER SUMMARY (build_all_players)    │
              │  11-factor composite → overall_impact (0-90)      │
              │  + captaincy bonus + win bonus + playoff bonus    │
              └──────────┬───────────────────────┬────────────────┘
                         │                       │
              ┌──────────▼──────────┐  ┌─────────▼──────────────┐
              │ 11. OUTPUT CSVs    │  │ 12. VISUALIZATIONS     │
              │ rankings, impacts, │  │ gauges, trends, radar, │
              │ partnerships       │  │ scatter, comparison    │
              └─────────────────────┘  └────────────────────────┘
                         │
              ┌──────────▼──────────────────────────────────────┐
              │         13. STREAMLIT DASHBOARD (app.py)        │
              │  4 tabs: Overview | Player Profile | H2H |     │
              │          Match Explorer                         │
              └─────────────────────────────────────────────────┘
```

---

## 6. Core Data Pipeline (15 Steps)

The `run_pipeline.py` script orchestrates the entire computation in 15 sequential steps:

| Step | Operation | Module | Description |
|------|-----------|--------|-------------|
| 1 | **Load & Clean** | `data_loader.py` | Parse all ball-by-ball CSVs, combine into single DataFrame; parse info CSVs for metadata (winner, venue, season, player of match). Derive columns: overs, dot balls, wickets, match phases. Auto-detect match format (T20/ODI/Test). |
| 2 | **Aggregate Batting** | `player_aggregation.py` | Group by striker per match-innings: runs, balls faced, strike rate, boundaries (4s/6s), dot balls, dismissals, batting position (derived from appearance order). |
| 3 | **Aggregate Bowling** | `player_aggregation.py` | Group by bowler per match-innings: wickets, runs conceded, economy rate, dot balls, phase-specific stats (powerplay/middle/death). |
| 4 | **Merge Runs Contribution** | `player_aggregation.py` | Compute each batter's share of their team's innings total as a percentage. |
| 5 | **Add Context Features** | `context_features.py` | Engineer 7 contextual features: venue difficulty index, opposition strength (team batting/bowling averages), cumulative ball-level stats (runs, wickets, run rate, target), pressure index, clutch indicator, leverage index, and momentum shift. |
| 6 | **Train/Load ML Model** | `win_prob_model.py` | Train XGBoost win probability classifier on 13 features, or load previously saved model from `models/win_prob_model.pkl`. Uses 80/20 match-level train/test split. Evaluates AUC-ROC, Log Loss, and Brier Score. |
| 7 | **Predict Win Probabilities** | `win_prob_model.py` | Add `win_prob` column to every delivery in ball_df using the trained model. |
| 8 | **Compute Per-Ball WPA** | `wpa_engine.py` | Calculate Win Probability Added for each delivery: `WPA[i] = win_prob[i+1] - win_prob[i]`. Attribute WPA to both striker (batter) and bowler. |
| 9 | **Aggregate WPA** | `wpa_engine.py` | Per-player per-match: total WPA, average WPA, leverage-weighted WPA, consistency WPA, clutch WPA, presence impact (WPA at arrival vs departure), and bowler-specific metrics (wicket WPA, spell efficiency). |
| 10 | **Compute Partnerships** | `partnerships.py` | Batting partnerships (pairs of batters: runs, SR, boundaries), bowling tandems (consecutive-over bowler pairs), and batter-vs-bowler career head-to-head rivalry stats (SR, dominance index). |
| 11 | **Compute Impact Scores** | `impact_metric.py` | Format-aware batting impact (7 components) and bowling impact (8 components) with format-specific weights for T20/ODI/Test. Includes position multiplier and volume multiplier. |
| 12 | **Apply Rolling Normalization** | `rolling_normalization.py` | Recency-weighted average over last 10 innings, consistency bonus, comeback bonus. Bayesian dampening for players with few innings. Logistic normalization to 0–100 scale. |
| 13 | **Save Per-Match CSVs** | `run_pipeline.py` | Write `batting_impact.csv`, `bowling_impact.csv`, `batting_partnerships.csv`, `bowling_partnerships.csv`, `batter_vs_bowler_rivalry.csv`. |
| 14 | **Build Match Rankings** | `run_pipeline.py` | Combine batter and bowler rows per match, rank by `raw_impact` within each match. Save `match_rankings.csv`. |
| 15 | **Build Career Summary** | `run_pipeline.py` | Aggregate per-player across all matches: 11-factor composite `overall_impact` score (0–90). Add captaincy stats, win bonuses, playoff bonuses. Generate visualizations. Save `all_players_impact_T20.csv` and `all_players_impact.csv`. |

---

## 7. Module-by-Module Feature Documentation

### 7.1 Data Loader (`data_loader.py`)

**Purpose:** Load raw Cricsheet CSV files and prepare a clean, unified DataFrame.

**Functions:**

| Function | Description |
|---|---|
| `load_ball_by_ball(data_folder)` | Scans `data_folder` for all `{match_id}.csv` files, reads each into a DataFrame, concatenates into a single ball-by-ball DataFrame with `match_id` column. |
| `load_match_info(data_folder)` | Parses all `{match_id}_info.csv` files. Extracts key-value metadata: teams, venue, winner, season, date, player of match, and team rosters. |
| `clean_data(ball_df, info_df)` | Derives computed columns: `over` (integer from ball notation), `is_dot` (no runs off bat), `is_wicket` (dismissal occurred), `phase` (Powerplay: 0–5, Middle: 6–14, Death: 15–19 for T20). Detects and attaches `match_format`. |
| `load_and_prepare(data_folder)` | Orchestrator: calls all three functions above in sequence, returns `(ball_df, info_df, match_format)`. |

**Key Derived Columns:**

| Column | Formula |
|---|---|
| `over` | `floor(ball)` |
| `is_dot` | `1 if runs_off_bat == 0 and extras == 0 else 0` |
| `is_wicket` | `1 if wicket_type is not NaN else 0` |
| `phase` | `Powerplay` (overs 0–5), `Middle` (6–14), `Death` (15–19) |
| `match_format` | Auto-detected: T20 / ODI / Test |

---

### 7.2 Player Aggregation (`player_aggregation.py`)

**Purpose:** Compute per-player per-match statistics from ball-level data.

**Functions:**

| Function | Description |
|---|---|
| `derive_batting_position(ball_df)` | Orders batters by their first appearance (first ball faced) to derive batting position (1 = opener, 11 = tail). |
| `aggregate_batting(ball_df)` | Groups by `(match_id, innings, striker)` to compute: runs, balls_faced, fours, sixes, strike_rate, dots, dot_pct, is_dismissed. |
| `aggregate_bowling(ball_df)` | Groups by `(match_id, innings, bowler)` to compute: wickets, runs_conceded, balls_bowled, economy, dots, maidens, phase-specific stats. |
| `compute_phase_performance(ball_df)` | Splits stats into powerplay/middle/death phases for both batters and bowlers. |
| `get_innings_totals(ball_df)` | Computes team-level innings totals: total runs, total wickets. Used for runs_contribution. |
| `merge_runs_contribution(bat_agg, innings_totals)` | Adds `runs_contribution` column: each batter's share of their team's total runs as a percentage. |

**Batting Aggregation Outputs:**

| Column | Description |
|---|---|
| `runs` | Total runs scored by batter in the innings |
| `balls_faced` | Legal deliveries faced |
| `fours` | Number of boundaries (4 runs off bat) |
| `sixes` | Number of maximums (6 runs off bat) |
| `strike_rate` | `(runs / balls_faced) × 100` |
| `dots` | Number of dot balls faced |
| `dot_pct` | `(dots / balls_faced) × 100` |
| `is_dismissed` | 1 if the batter was dismissed, 0 otherwise |
| `batting_position` | Order of appearance (1 = opener) |
| `runs_contribution` | `runs / team_innings_total × 100` |

**Bowling Aggregation Outputs:**

| Column | Description |
|---|---|
| `wickets` | Wickets taken |
| `runs_conceded` | Runs conceded while bowling |
| `balls_bowled` | Legal deliveries bowled |
| `economy` | `(runs_conceded / balls_bowled) × 6` |
| `dots` | Dot balls bowled |
| `dot_pct` | `(dots / balls_bowled) × 100` |
| `pp_wickets` / `pp_economy` | Powerplay-specific stats |
| `death_wickets` / `death_economy` | Death overs stats |

---

### 7.3 Context Features (`context_features.py`)

**Purpose:** Engineer contextual features that capture match situation, pressure, and difficulty beyond raw stats.

**7 Context Features:**

#### 1. Venue Features (`add_venue_features`)
- **Venue Scoring Index:** Historical average first-innings total at each venue, normalized to 0–1.
- **Chase Success Rate:** Historical percentage of successful chases at each venue.
- **Venue Difficulty:** `1 - venue_scoring_index` (higher = harder to score).

#### 2. Opposition Strength (`add_opposition_strength`)
- **Team Batting Strength:** Average runs scored per match by the opposition's batting lineup.
- **Team Bowling Strength:** Average wickets per match by the opposition's bowling lineup.
- Computed as rolling averages across all prior matches.

#### 3. Ball-Level Context (`add_ball_level_context`)
- **Cumulative Runs:** Running total of team runs at each ball.
- **Cumulative Wickets:** Running total of wickets lost at each ball.
- **Cumulative Balls:** Legal balls delivered so far.
- **Current Run Rate:** `cumulative_runs / (cumulative_balls / 6)`.
- **Required Run Rate (chase only):** `(target - cumulative_runs) / overs_remaining`.
- **RR Pressure Gap:** `required_rr - current_rr` (positive = behind the rate).
- **Target:** First innings total + 1 (for second innings); set to 0 for first innings.

#### 4. Pressure Index (`add_pressure_index`)
- Format-aware composite pressure metric:
  - **Short formats (T20/ODI):** Combines `rr_pressure_gap`, wickets lost ratio, and overs consumed ratio.
  - **Long formats (Test):** Uses session-based pressure windows instead of RRR.
- Scaled 0–1 (0 = no pressure, 1 = maximum pressure).

#### 5. Clutch Indicator (`add_clutch_indicator`)
- Marks deliveries as "clutch" when in high-pressure situations:
  - **T20/ODI:** Last 20% of overs AND high pressure index (>0.6).
  - **Test:** Rolling 60-ball windows where team WPA exceeds threshold.

#### 6. Leverage Index (`add_leverage_index`)
- Measures how much a single delivery can swing the match:
  - `leverage_index = rr_pressure_gap_normalized × wicket_fraction × phase_difficulty_factor`
  - Higher in death overs with close margins and few wickets in hand.

#### 7. Momentum Shift (`add_momentum_shift`)
- Rolling 12-ball window change in required run rate.
- Captures sudden surges or collapses in match flow.
- `momentum_shift = RRR[t] - RRR[t-12]` (positive = chase getting harder).

---

### 7.4 Win Probability Model (`win_prob_model.py`)

**Purpose:** Train a machine learning classifier to predict the probability of the batting team winning at any given delivery.

#### Model Features (13 input variables)

| # | Feature | Description |
|---|---|---|
| 1 | `innings` | 1 or 2 |
| 2 | `over` | Current over number (0–19 for T20) |
| 3 | `cumulative_runs` | Team runs scored so far |
| 4 | `cumulative_wickets` | Wickets lost so far |
| 5 | `cumulative_balls` | Legal balls bowled so far |
| 6 | `current_rr` | Current run rate |
| 7 | `required_rr` | Required run rate (chase only) |
| 8 | `rr_pressure_gap` | `required_rr - current_rr` |
| 9 | `wickets_remaining` | `10 - cumulative_wickets` |
| 10 | `target` | First inning score + 1 (second innings) |
| 11 | `venue_difficulty` | Inverted scoring index (0–1) |
| 12 | `phase_encoded` | 0 = Powerplay, 1 = Middle, 2 = Death |
| 13 | `is_chasing` | 1 if second innings, 0 otherwise |

#### Models Trained

| Model | Configuration | Role |
|---|---|---|
| Logistic Regression | Default | Baseline comparison |
| Random Forest | 200 trees, max_depth=10 | Fallback model |
| **XGBoost** | **300 trees, max_depth=6, learning_rate=0.1** | **Primary model** |

#### Training Process
1. **Data split:** 80% matches for training, 20% for testing (match-level split to prevent data leakage).
2. **Training:** All three models trained on the same features.
3. **Evaluation:** AUC-ROC, Log Loss, Brier Score computed on test set.
4. **Calibration:** If Brier Score > 0.22, XGBoost predictions are calibrated using isotonic regression.
5. **Serialization:** Best model saved as `cricket_impact/models/win_prob_model.pkl` using joblib.

#### Output
- Adds `win_prob` column (0.0 to 1.0) to every delivery in the ball DataFrame.

---

### 7.5 WPA Engine (`wpa_engine.py`)

**Purpose:** Compute Win Probability Added — the change in win probability caused by each delivery, attributed to both batter and bowler.

#### Core Concept

```
WPA[i] = win_prob[i+1] - win_prob[i]
```

A positive WPA for the batter means that delivery increased the batting team's win probability (e.g., hitting a six). The same delivery has negative WPA for the bowler.

#### Functions

| Function | Description |
|---|---|
| `compute_per_ball_wpa(ball_df)` | Calculates `wpa = win_prob.shift(-1) - win_prob` for each delivery within each match-innings, grouped by `(match_id, innings)`. |
| `aggregate_batter_wpa(ball_df)` | Aggregates per-ball WPA to per-batter per-match: `total_wpa` (sum), `avg_wpa` (mean), `max_wpa`, `min_wpa`, `positive_wpa_pct`, `leverage_wpa` (WPA × leverage_index), `consistency_wpa` (pct × (1 - volatility)). |
| `compute_batter_presence_impact(ball_df)` | Measures win probability at batter's arrival vs departure: `presence_impact = win_prob_last - win_prob_first`. Captures batting innings-level contribution. |
| `aggregate_bowler_wpa(ball_df)` | Negates WPA for bowler perspective: positive = bowler reduced batting team's win probability. Computes `total_wpa`, `wicket_wpa` (WPA on dismissal balls only), `economy_wpa` (WPA from dot balls), `spell_efficiency` (WPA per ball bowled). |
| `compute_clutch_wpa(ball_df)` | WPA earned in high-pressure "clutch" situations: short formats (last 20% of overs + high pressure_index), long formats (rolling 60-ball windows exceeding WPA threshold). |

#### WPA Aggregation Outputs (Batter)

| Metric | Description |
|---|---|
| `total_wpa` | Sum of all delivery WPAs. Positive = added to team's win chance. |
| `avg_wpa` | Mean WPA per delivery. How impactful per ball. |
| `max_wpa` | Single most positive-impact delivery. |
| `min_wpa` | Single most negative-impact delivery. |
| `positive_wpa_pct` | % of deliveries with positive WPA (good balls). |
| `leverage_wpa` | Σ(WPA × leverage_index). Impact in high-stakes moments. |
| `consistency_wpa` | `positive_wpa_pct × (1 - std(wpa)/mean(|wpa|))`. Rewards steady performance. |
| `presence_impact` | Win prob at departure minus win prob at arrival. Total innings-level shift. |
| `clutch_wpa` | WPA earned only in clutch situations. |

#### WPA Aggregation Outputs (Bowler)

| Metric | Description |
|---|---|
| `total_wpa` | Negated sum: positive = bowling team gained advantage. |
| `wicket_wpa` | WPA on dismissal deliveries only. Big swing for big wickets. |
| `economy_wpa` | WPA from dot balls. Pressure building through containment. |
| `spell_efficiency` | `total_wpa / balls_bowled`. Quality per delivery. |

---

### 7.6 Partnerships & Rivalry (`partnerships.py`)

**Purpose:** Analyze batting partnerships, bowling tandems, and batter-vs-bowler career matchups.

#### Functions

| Function | Description |
|---|---|
| `compute_batting_partnerships(ball_df)` | For every pair of batters at the crease together: total runs scored, strike rate, boundaries, balls faced, partnership impact. Groups by striker+non_striker. |
| `get_best_partnership_per_player(partnerships_df)` | For each player per match, identifies their best partnership. |
| `compute_bowling_tandems(ball_df)` | Finds pairs of bowlers who bowl in consecutive overs. Computes tandem wickets, runs conceded, economy, dot percentage. |
| `compute_batter_vs_bowler_rivalry(ball_df)` | Career head-to-head stats: balls, runs, dismissals, dots, boundaries, SR, dominance_index. |
| `merge_rivalry_features(bat_agg, rivalry_df)` | Merges average dominance metrics back into batting aggregation for impact scoring. |

#### Rivalry Output — Dominance Index

```
dominance_index = (batter_sr / 100) × (1 - dismissal_rate) × boundary_pct_factor
```

- **> 1.0** — Batter dominates the bowler
- **< 1.0** — Bowler dominates the batter
- **= 1.0** — Even contest

---

### 7.7 Impact Metric (`impact_metric.py`)

**Purpose:** Compute format-aware batting and bowling impact scores using weighted multi-factor formulas.

#### Batting Impact — 7 Components

| # | Component | T20 Weight | ODI Weight | Test Weight | Description |
|---|---|---|---|---|---|
| 1 | Total WPA | 0.27 | 0.25 | 0.20 | Win Probability Added (percentile-ranked) |
| 2 | Presence Impact | 0.13 | 0.16 | 0.22 | Win prob change from arrival to departure |
| 3 | Leverage WPA | 0.13 | 0.12 | 0.10 | WPA in high-leverage situations |
| 4 | Runs Contribution | 0.14 | 0.16 | 0.20 | % of team's innings runs |
| 5 | Clutch WPA | 0.09 | 0.08 | 0.08 | WPA in clutch (high-pressure) situations |
| 6 | Consistency | 0.13 | 0.13 | 0.13 | `1 - (σ / μ)` modifier — rewards steady scoring |
| 7 | Dot Efficiency | 0.11 | 0.10 | 0.05 | `1 - (dot_pct / 100)` — penalizes dot balls |

#### Bowling Impact — 8 Components

| # | Component | T20 Weight | ODI Weight | Test Weight | Description |
|---|---|---|---|---|---|
| 1 | Total WPA | 0.25 | 0.23 | 0.20 | Win Probability Added (bowler perspective) |
| 2 | Wicket WPA | 0.15 | 0.17 | 0.25 | WPA earned specifically on wicket-taking balls |
| 3 | Leverage WPA | 0.12 | 0.10 | 0.08 | WPA in high-leverage moments |
| 4 | Economy WPA | 0.07 | 0.10 | 0.05 | WPA from containment (dot balls) |
| 5 | Clutch WPA | 0.08 | 0.07 | 0.05 | WPA in clutch situations |
| 6 | Spell Efficiency | 0.07 | 0.08 | 0.10 | WPA per ball — quality per delivery |
| 7 | Phase Difficulty | 0.15 | 0.12 | 0.08 | Bonus for performing in death/powerplay overs |
| 8 | Dot Pressure | 0.11 | 0.13 | 0.19 | Building pressure through dot balls |

#### Position Multiplier (Batting)

Rewards batters who outperform their expected contribution based on batting position:

```
expected_share = {1: 0.18, 2: 0.16, 3: 0.15, 4: 0.14, 5: 0.12, ...}
position_mult = 0.85 + 0.16 × clamp(actual_share / expected_share, 0.5, 2.5)
```

A lower-order batter (e.g., #7) scoring heavily gets a larger multiplier than an opener with the same runs.

#### Volume Multiplier (Format-Specific)

Adjusts for the absolute volume of contribution:

| Format | Formula |
|---|---|
| **T20** | `0.40 + 0.80 × runs_contribution + 0.30 × (balls_faced / 60)` |
| **ODI** | `0.35 + 0.65 × runs_contribution + 0.40 × (balls_faced / 100)` |
| **Test** | `0.30 + 0.50 × runs_contribution + 0.50 × (balls_faced / 150)` |

#### Phase Difficulty Bonus (Bowling)

Bowlers who perform well in the most difficult phases receive a bonus:

- **Death Overs (16–19 in T20):** Economy rate and wicket rate normalized against elite/bad benchmarks.
- **Powerplay (0–5):** Similar normalization with different thresholds.
- The bonus is scaled 0–1 based on how the bowler's stats compare to format-specific elite/bad ranges.

---

### 7.8 Rolling Normalization (`rolling_normalization.py`)

**Purpose:** Apply recency-weighted rolling normalization so that recent form matters more than distant past, with Bayesian correction for small samples.

#### Recency Weights

```python
RECENCY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# Index 0 = most recent match, Index 9 = 10 matches ago
```

#### Process

1. **Rolling Impact:**
   - Take the player's last 10 innings raw impact scores.
   - Compute weighted average using recency weights (most recent innings weighted 1.0, oldest weighted 0.1).

2. **Consistency Bonus:**
   ```
   consistency = 1 - (std_dev / mean)  if mean > 0  else 0.5
   ```
   - Rewards players who deliver consistently rather than having a few big games and many mediocre ones.

3. **Comeback Bonus:**
   ```
   If avg(last 5 innings) > avg(prior 5 innings) by 20%+:
       comeback_bonus = 1.10  (10% uplift)
   Else:
       comeback_bonus = 1.0
   ```
   - Rewards players on an upward trajectory.

4. **Adjusted Impact:**
   ```
   adjusted_impact = rolling_impact × consistency × comeback_bonus
   ```

5. **Bayesian Dampening:**
   ```
   bayes_weight = innings_count / (innings_count + 5)
   dampened_impact = bayes_weight × adjusted_impact + (1 - bayes_weight) × prior_median
   ```
   - Players with few innings (e.g., 3) are shrunk toward the population median.
   - Players with many innings (e.g., 100) are almost fully represented by their own data.
   - `min_innings = 5` is the dampening anchor point.

6. **Logistic Normalization (0–100):**
   ```
   percentile = rank_percentile(dampened_impact)
   impact_score = 100 / (1 + exp(-8 × (percentile - 0.5)))
   ```
   - S-shaped curve that spreads out the middle and compresses extremes.

---

### 7.9 Captaincy & Win Bonus (`captaincy.py`)

**Purpose:** Track captaincy records, credit winning players, and reward playoff consistency.

#### Functions

| Function | Description |
|---|---|
| `build_captaincy_data(info_df)` | Maps `(season, team) → captain_name` using a comprehensive IPL captains dictionary covering **2007–2025**. Returns captaincy DataFrame. |
| `compute_captaincy_stats(captaincy_df  )` | Per-captain: total matches, wins, win_rate, playoff_matches, final_appearances. |
| `compute_win_bonus(bat_agg, bowl_agg, info_df)` | Adds `win_bonus` column: all players on the winning team receive credit. Scales with margin of victory. |
| `compute_playoff_team_bonus(info_df)` | Counts how many seasons each player's team reached the playoffs. Players on consistently successful franchises receive a small bonus. |

#### IPL Captains Dictionary

The system includes a **hardcoded comprehensive mapping** of IPL captains for every team and season from 2007 to 2025, covering:
- All franchise changes (e.g., Delhi Daredevils → Delhi Capitals)
- Mid-season captaincy changes
- New franchises (Gujarat Titans, Lucknow Super Giants from 2022)

---

### 7.10 Visualization (`visualization.py`)

**Purpose:** Generate static matplotlib visualizations for reporting.

**Chart Types:**

| Function | Output | Description |
|---|---|---|
| `gauge_chart(player, score)` | `gauge_{player}.png` | Half-circle impact meter showing score on 0–100 scale with color zones (red/yellow/green). |
| `trend_line(player, scores)` | `trend_{player}.png` | Line chart of last 15 innings impact scores with rolling average overlay. |
| `player_comparison(players)` | `player_comparison.png` | Grouped bar chart comparing impact components for top 6 players. |
| `impact_distribution(all_scores)` | `impact_distribution.png` | Histogram of all players' overall impact scores. |
| `allrounder_scatter(df)` | `allrounder_scatter.png` | Scatter plot: batting impact (x) vs bowling impact (y). All-rounders highlighted. |
| `generate_all_visualizations(df)` | Multiple PNGs | Batch generation: gauges for top 10, trends, comparison, distribution, scatter. |

---

## 8. Impact Scoring — Detailed Formulas

### 8.1 Batting Impact Formula

```
raw_bat_impact = Σ(weight_i × percentile_rank(component_i))  for i = 1..7
               × position_multiplier
               × volume_multiplier
```

Where the 7 components are: Total WPA, Presence Impact, Leverage WPA, Runs Contribution, Clutch WPA, Consistency, Dot Efficiency — each with format-specific weights (see Section 7.7).

### 8.2 Bowling Impact Formula

```
raw_bowl_impact = Σ(weight_i × percentile_rank(component_i))  for i = 1..8
                × volume_multiplier
                + phase_difficulty_bonus
```

Where the 8 components are: Total WPA, Wicket WPA, Leverage WPA, Economy WPA, Clutch WPA, Spell Efficiency, Phase Difficulty, Dot Pressure.

### 8.3 Impact Categories

After computing raw impact, players are categorized:

| Category | Range | Description |
|---|---|---|
| **Match Winner** | Top 5% | Elite game-changing performance |
| **High Impact** | Top 20% | Significant contribution |
| **Above Average** | Top 40% | Solid performance |
| **Average** | Middle 30% | Standard contribution |
| **Below Average** | Bottom 30% | Limited impact |

---

## 9. Overall Impact — 11-Factor Composite Score

The final career-level `overall_impact` score (0–90 scale) combines 11 distinct factors:

| # | Factor | Weight | Description |
|---|---|---|---|
| 1 | **Role-Weighted Score** | 0.28 | `bat_weight × bat_contribution + bowl_weight × bowl_contribution`. Bat/bowl weights are proportional to innings played in each role. |
| 2 | **All-Rounder Bonus** | 0.10 | Dual contribution bonus for players with ≥20 innings in both batting and bowling. Rewards versatility. |
| 3 | **Longevity** | 0.12 | `log(total_matches)` normalized. Rewards sustained careers over flash-in-the-pan performers. |
| 4 | **Player of Match Awards** | 0.10 | Absolute count of POM awards, normalized across all players. |
| 5 | **POM Efficiency** | 0.06 | `pom_awards / total_matches`. Awards per opportunity — rewards high conversion rate. |
| 6 | **Clutch Contribution** | 0.06 | `|total_clutch_bat_wpa| + |total_clutch_bowl_wpa|`. Impact in high-pressure moments. |
| 7 | **Volume** | 0.07 | `log(total_runs)` for batters, `log(total_wickets)` for bowlers. Raw production. |
| 8 | **Win Bonus** | 0.08 | Accumulated bonus from being on winning teams. Rewards winning players. |
| 9 | **Captaincy Bonus** | 0.05 | Captain-only: win rate as captain, weighted by matches captained. 0 for non-captains. |
| 10 | **Playoff Bonus** | 0.04 | Count of seasons where the player's team reached playoffs. Team-level consistency. |
| 11 | **Bowling Visibility** | 0.04 | Bowling impact percentage. Boosts bowlers/all-rounders who may be underrated by batting-heavy metrics. |

### Final Score Computation

```
overall_impact_raw = Σ(weight_i × normalized_factor_i)  for i = 1..11

# Bayesian dampening for sample size
bayes_weight = total_innings / (total_innings + 12)
prior = median(all_players_raw)
overall_impact_dampened = bayes_weight × raw + (1 - bayes_weight) × prior

# Power-curve normalization to 0–90
percentile = rank_percentile(dampened)
overall_impact = 90 × percentile ^ 0.7
```

The power curve (`^0.7`) ensures good separation at the top while compressing the lower end.

---

## 10. Streamlit Dashboard (`app.py`)

### 10.1 Overview

A **dark-themed**, modern web dashboard built with Streamlit and Plotly. Features glassmorphism UI elements, animated charts, and comprehensive filtering.

### 10.2 Launch

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 10.3 Dashboard Tabs

#### Tab 1: Overview

| Feature | Description |
|---|---|
| **Top 15 Leaderboard** | Ranked table of highest-impact players with overall_impact scores |
| **Impact Distribution** | Histogram showing the distribution of all player impact scores |
| **Role Distribution** | Pie chart: Batter vs Bowler vs All-Rounder breakdown |
| **Batting vs Bowling Scatter** | Interactive scatter plot: mean batting impact (x) vs mean bowling impact (y) |
| **Top Batters** | Bar chart of top 15 batters by batting_rating |
| **Top Bowlers** | Bar chart of top 15 bowlers by bowling impact |

#### Tab 2: Player Profile

| Feature | Description |
|---|---|
| **Player Selector** | Dropdown to select any player from the dataset |
| **Impact Gauge** | Half-circle gauge meter showing overall impact (0–90) with color zones |
| **KPI Cards** | Glassmorphism stat cards: runs, SR, innings, wickets, economy, etc. |
| **7-Axis Radar Chart** | Spider/radar chart of player strengths: batting, bowling, clutch, consistency, win impact, dot efficiency, awards |
| **Form Trend Line** | Last 15 innings impact scores as a line chart with rolling average |
| **Season Filter** | Filter stats to specific IPL seasons |

#### Tab 3: Head-to-Head Comparison

| Feature | Description |
|---|---|
| **Dual Player Selector** | Pick any two players to compare side by side |
| **Dumbbell Chart** | Side-by-side comparison of key metrics (runs, SR, WPA, clutch, etc.) |
| **Radar Overlay** | Two overlapping radar charts for direct visual comparison |
| **Form Comparison** | Dual trend lines showing recent form for both players |
| **Rivalry Data** | If the two players have batter-vs-bowler data, shows H2H stats |

#### Tab 4: Match Explorer

| Feature | Description |
|---|---|
| **Match Selector** | Browse matches by season, team, or search |
| **Match Label Format** | "Team A vs Team B \| Venue (Season)" |
| **Top Performers Chart** | Bar chart of top-ranked players in the selected match |
| **Detailed Table** | Full match table: all players ranked by impact, with role, key_stat, WPA, clutch |
| **Season/Team Filters** | Filter match list by season or team |

### 10.4 UI Design

- **Color Palette:** Dark background (`#0e1117`), cyan accents (`#58a6ff`), gold highlights (`#e3b341`)
- **Cards:** Glassmorphism effect with `backdrop-filter: blur` and semi-transparent backgrounds
- **Charts:** Plotly with transparent backgrounds, matching the dark theme
- **Data Caching:** Streamlit `@st.cache_data` for fast CSV loading

---

## 11. Output Files Reference

### 11.1 Career Rankings (`all_players_impact_T20.csv`)

| Column | Description |
|---|---|
| `player` | Player name |
| `bat_innings` | Total batting innings played |
| `total_runs` | Career runs scored |
| `total_balls_faced` | Career balls faced |
| `fours`, `sixes` | Career boundaries |
| `avg_bat_sr` | Average batting strike rate |
| `mean_bat_impact` | Average per-match batting impact score |
| `bowl_innings` | Total bowling innings |
| `total_wickets` | Career wickets taken |
| `total_runs_conceded` | Career runs conceded while bowling |
| `total_balls_bowled` | Career balls bowled |
| `avg_economy` | Average bowling economy rate |
| `mean_bowl_impact` | Average per-match bowling impact score |
| `pom_awards` | Player of the Match awards count |
| `matches_as_captain` | Matches captained |
| `wins_as_captain` | Wins as captain |
| `captaincy_score` | Captaincy effectiveness score |
| `total_wins` | Total matches on winning side |
| `win_rate` | Career win rate |
| `batting_rating` | Normalized batting-only rating |
| `overall_impact_raw` | Raw 11-factor composite score |
| `overall_impact_dampened` | After Bayesian dampening |
| `overall_impact` | **Final score (0–90 scale)** |
| `match_format` | T20 / ODI / Test |

### 11.2 Match Rankings (`match_rankings.csv`)

| Column | Description |
|---|---|
| `match_id` | Match identifier |
| `innings` | 1 or 2 |
| `player` | Player name |
| `role` | `batter` or `bowler` |
| `player_team` | Player's team |
| `opposition` | Opposing team |
| `season` | IPL season year |
| `venue` | Match venue |
| `key_stat` | Human-readable stat line (e.g., "45 runs (150.0 SR)" or "3/24 (8.0 econ)") |
| `performance_score` | Raw performance metric |
| `impact_score` | Normalized impact (0–100) |
| `impact_category` | Match Winner / High Impact / Average / etc. |
| `wpa_score` | Total WPA in that match |
| `clutch_score` | Clutch WPA in that match |
| `rank` | Player's rank in that match (1 = best) |

### 11.3 Per-Match Impact Files

**`batting_impact.csv`** — One row per batter per match-innings:
- Raw stats: runs, balls_faced, strike_rate, fours, sixes, dots
- WPA metrics: total_wpa, presence_impact, leverage_wpa, consistency_wpa, clutch_wpa
- Impact chain: running_impact → adjusted_impact → dampened_impact → impact_pctile → impact_score

**`bowling_impact.csv`** — One row per bowler per match-innings:
- Raw stats: wickets, runs_conceded, economy, balls_bowled, dots
- WPA metrics: total_wpa, wicket_wpa, leverage_wpa, spell_efficiency, clutch_wpa
- Same impact chain as batting

### 11.4 Partnership & Rivalry Files

**`batting_partnerships.csv`** — Partnership runs, SR, boundaries, impact for each batting pair.

**`bowling_partnerships.csv`** — Bowling tandem stats: tandem_wickets, tandem_runs, tandem_economy, tandem_dot_pct.

**`batter_vs_bowler_rivalry.csv`** — Career head-to-head:
- `batter`, `bowler_name`, `matches`, `balls`, `runs`, `dismissals`
- `dots`, `fours`, `sixes`, `sr`, `dot_pct`, `boundary_pct`
- `dismissal_rate`, `dominance_index`

### 11.5 Model File

**`cricket_impact/models/win_prob_model.pkl`** — Serialized XGBoost classifier. Auto-generated by the pipeline. Can be force-retrained with `--retrain` flag.

---

## 12. Troubleshooting & FAQ

### Common Issues

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | Run `pip install xgboost`. If installation fails, the pipeline will automatically fall back to Random Forest. |
| `FileNotFoundError: all_players_impact_T20.csv` | Run the pipeline first: `python run_pipeline.py --data_folder ./ipl_male_csv2` before launching the dashboard. |
| `streamlit: command not found` | Install Streamlit: `pip install streamlit`, or use `python -m streamlit run app.py`. |
| Dashboard loads but shows no data | Ensure CSV output files exist in `cricket_impact/` directory. Re-run the pipeline if needed. |
| Pipeline runs out of memory | Reduce dataset size or increase system RAM. The full IPL dataset (~278K rows) requires ~2 GB during processing. |
| Model accuracy is low (AUC < 0.65) | Re-train with `--retrain` flag. Ensure data folder has sufficient matches (100+ recommended). |
| `KeyError` on a column name | Ensure data CSVs follow the Cricsheet format. Check that both `{id}.csv` and `{id}_info.csv` exist for each match. |

### FAQ

**Q: Can I use this for non-IPL data?**
A: Yes. The system supports any Cricsheet-format CSV data (international T20s, ODIs, Tests). Place the data in any folder and pass it via `--data_folder`. The captaincy module's hardcoded dictionary is IPL-specific, but the pipeline will still work — captaincy bonuses will simply be zero for non-IPL matches.

**Q: How long does the pipeline take?**
A: Processing time depends on dataset size and hardware. The full IPL dataset typically completes in a few minutes on modern hardware.

**Q: Can I add new matches?**
A: Yes. Add new CSV files to the data folder and re-run the pipeline. Use `--retrain` if the new data significantly changes the distribution (e.g., adding a full new season).

**Q: Why does my player have a low score despite high traditional stats?**
A: CricImpact values **context** and **impact**. A player may have high runs but scored them in low-pressure situations, against weak opposition, or in losing causes. The WPA-based system captures when and how runs were scored, not just how many.

**Q: What is Bayesian dampening?**
A: Players with very few innings (<10) have their scores adjusted toward the population median. This prevents a player with 2 brilliant innings from ranking above a consistent performer with 100 innings. As innings count increases, the dampening effect diminishes.

---

## 13. Sample Results

### Top 10 IPL T20 Players by Overall Impact

| Rank | Player | Overall Impact | Notable Strength |
|------|--------|---------------|-------------|
| 1 | RG Sharma | 90.0 | Batting + Longevity + Captaincy |
| 2 | V Kohli | 89.9 | Batting consistency + Clutch |
| 3 | SP Narine | 89.8 | All-rounder + Phase difficulty |
| 4 | RA Jadeja | 89.8 | All-rounder balance |
| 5 | MS Dhoni | 89.7 | Clutch + Captaincy + Win bonus |

*Based on analysis of 1,169 IPL matches, 278,205 deliveries, 768 players.*

---

## Appendix A: Quick Reference — All Commands

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib plotly streamlit joblib

# Run the full pipeline
python run_pipeline.py --data_folder ./ipl_male_csv2

# Run with model retraining
python run_pipeline.py --data_folder ./ipl_male_csv2 --retrain

# Run with custom output directory
python run_pipeline.py --data_folder ./ipl_male_csv2 --output_dir ./my_output

# Launch the interactive dashboard
streamlit run app.py

# Verify rankings
python check_rankings.py
```

## Appendix B: Technology Stack Summary

| Layer | Technology | Usage |
|---|---|---|
| **Language** | Python 3.10+ | Core language |
| **Data** | pandas, numpy | DataFrames, numerical ops |
| **ML** | scikit-learn, XGBoost | Win probability models |
| **Static Viz** | matplotlib | Gauge charts, trends, scatter |
| **Interactive Viz** | Plotly | Dashboard charts (radar, bar, scatter, gauge) |
| **Web UI** | Streamlit | Dashboard framework |
| **Serialization** | joblib | Model persistence (.pkl) |
| **Data Source** | Cricsheet.org | Ball-by-ball CSV data |

---

---

## 14. Thought Process & Cricket Logic — Step-by-Step Derivation of the Impact Metric

This section explains the **why** behind every component and design decision. Each subsection maps a technical choice to a real cricket scenario, so evaluators can understand the practical reasoning.

---

### 14.1 Why Win Probability Added (WPA) Instead of Traditional Stats?

**The Problem with Traditional Metrics:**

Traditional cricket statistics — batting average, strike rate, bowling economy — treat all runs and wickets equally. But cricket is fundamentally a game of **context**:

- A six that takes your team from 150/2 to 156/2 in a chase of 190 in the 18th over is far more valuable than a six at 30/0 in the 3rd over.
- A wicket of a set batsman in the death overs is far more impactful than dismissing a tail-ender in the powerplay of a dead match.
- A dot ball in the 19th over when the batting team needs 14 off 8 is devastating; a dot ball in the 1st over is unremarkable.

**Our Solution — WPA:**

Win Probability Added measures the **change in the batting team's chance of winning** after every single delivery. With an ML model predicting P(win) at each ball state, we compute:

```
WPA[ball i] = P(win after ball i) − P(win before ball i)
```

**Cricket Example:**
- Match: RCB vs MI, 2nd innings, RCB chasing 175.
- At ball 18.3: RCB are 160/5, P(win) = 0.42.
- V Kohli hits a six: RCB 166/5, P(win) = 0.68.
- WPA for that delivery = +0.26 — this single ball increased RCB's win probability by 26 percentage points.
- This delivery is worth far more than a six hit at 10/0 in the 2nd over where P(win) might go from 0.48 to 0.50 (WPA = +0.02).

**Why this matters:** It rewards players who perform when it **actually changes the match outcome**, not just when they pile on runs in dead situations.

---

### 14.2 The ML Win Probability Model — Why XGBoost with These 13 Features?

**Thought Process:**

To compute WPA, we first need an accurate model that predicts P(batting team wins) at any delivery. We chose **13 features** that capture the full match state:

| Feature | Cricket Reasoning |
|---|---|
| `innings` (1 or 2) | Batting first vs chasing are fundamentally different situations. Setting targets vs knowing the target completely changes strategy. |
| `over` | The phase of the game (powerplay, middle, death) dictates risk-taking. Over 19 is very different from over 1. |
| `cumulative_runs` | How many runs have been scored tells you whether the batting team is on track. |
| `cumulative_wickets` | Wickets are a limited resource. Being 60/1 vs 60/5 are vastly different. |
| `cumulative_balls` | Combined with runs, this determines the run rate trajectory. |
| `current_rr` | How fast the batting team is scoring right now. |
| `required_rr` | In a chase: how fast they *need* to score. The gap between current and required RR creates pressure. |
| `rr_pressure_gap` | `required_rr − current_rr`. Positive means behind the rate (under pressure). Negative means ahead. This is the single most important feature in chases. |
| `wickets_remaining` | Having 8 wickets in hand vs 2 drastically changes what's possible. |
| `target` | The target score defines the challenge. Chasing 120 vs 220 is an entirely different game. |
| `venue_difficulty` | Some grounds (e.g., Wankhede) are batsman-friendly; others (e.g., Chepauk) favor bowlers. A 160 all-out at Wankhede is poor; at Chepauk it's competitive. |
| `phase_encoded` | 0=Powerplay (fielding restrictions → scoring), 1=Middle (consolidation), 2=Death (slog). Each phase has different dynamics. |
| `is_chasing` | The chasing team has perfect information (they know the target), which changes win probability entirely. |

**Why XGBoost over Logistic Regression?**

We train three models and compare:
1. **Logistic Regression** — Simple, interpretable, but cannot capture non-linear interactions (e.g., "6 wickets down AND need 12 an over AND 2 overs left" is far worse than any single factor alone).
2. **Random Forest** — Handles non-linearity well, robust to outliers. Our fallback model.
3. **XGBoost (300 trees, depth 6)** — Best at capturing complex feature interactions. Cricket match states are inherently non-linear: losing 1 wicket at 150/1 is minor, but losing 1 wicket at 150/8 is potentially match-ending.

**Calibration:** If the model's Brier Score > 0.22, we apply **isotonic calibration** to ensure that when the model says "70% win probability," the team actually wins ~70% of the time historically. This is critical because WPA is only meaningful if the underlying probabilities are well-calibrated.

**Training split:** We split by **matches, not deliveries**, to prevent data leakage. If we split by deliveries, the model would see balls from the same match in both train and test sets, inflating accuracy.

---

### 14.3 Context Features — Why Each One Matters in Real Cricket

#### 14.3.1 Venue Features

**Cricket Logic:** The venue fundamentally shapes what constitutes a good performance.
- At the **Chinnaswamy Stadium** (Bangalore), 180 is a par score in T20s. Scoring 170 there is average.
- At the **MA Chidambaram Stadium** (Chennai), 155 might be an above-par score. Scoring 170 there is excellent.

**Implementation:**
- We compute the **Venue Scoring Index** (0–1): historical average runs per over at each ground, normalized.
- **Venue Difficulty** = `1 − Scoring Index`. A high-scoring ground has low difficulty and vice versa.
- **Chase Success Rate**: Some grounds historically favor chasing (dew factor, pitch behavior). If a venue has 60% chase success, a defending team faces inherently more pressure.

**Practical Example:** A bowler conceding 35 runs in 4 overs at Chinnaswamy (venue_difficulty = 0.3) should be rated differently than one conceding 35 at Chepauk (venue_difficulty = 0.7). The context feature feeds into the ML model, making the win probability predictions venue-aware.

#### 14.3.2 Opposition Strength

**Cricket Logic:** Taking 3 wickets against Mumbai Indians (strong batting lineup) is harder than 3 wickets against a weaker lineup. Similarly, scoring 50 against a bowling attack led by Bumrah and Archer is more impressive than against a weak attack.

**Implementation:**
- **Team Batting Strength:** Average runs per over scored by the opposition's batting lineup across all their matches.
- **Team Bowling Strength:** Average runs per over conceded by the opposition's bowling lineup.
- These feed into the model so that when a player performs against a strong opposition, the win probability shift reflects that.

#### 14.3.3 Ball-Level Cumulative Context

**Cricket Logic:** Every delivery in cricket happens in a specific state: how many runs have been scored, how many wickets have fallen, how many balls remain, what's the target.

**Key derived features:**
- **Current Run Rate:** Are you scoring fast enough? At 6 RPO at the halfway mark chasing 180 in a T20, you're behind.
- **Required Run Rate (chase only):** This is the mathematical reality of what's needed. If RRR = 14 and you're at 8 RPO, the pressure is enormous.
- **RR Pressure Gap:** `Required RR − Current RR`. This is perhaps the single strongest predictor of chase outcomes. When this is positive and large (e.g., +6), the chasing team is in deep trouble.

**Format-aware capping:** We cap required_rr by format to avoid distortion:
- T20: capped at 15 (beyond that, the chase is effectively over)
- ODI: capped at 9 (very few teams sustain 9+ RPO in ODIs)
- Test: set to 0 (most Test innings aren't time-pressure chases)

**Why this matters:** Without these features, a boundary hit at 20/0 chasing 140 (easy situation) would be treated the same as a boundary at 100/5 chasing 200 (desperate situation). The cumulative context ensures every delivery is evaluated in its true match state.

#### 14.3.4 Pressure Index

**Cricket Logic:** Pressure in cricket comes from three sources simultaneously:
1. **Run rate pressure** — needing runs faster than you're getting them
2. **Wicket pressure** — having lost several wickets
3. **Time pressure** — running out of overs

A team at 120/7 needing 60 off 18 balls faces all three pressures simultaneously. A team at 120/1 needing 60 off 60 balls faces none.

**Implementation:**
- **For T20/ODI (chasing):**
  ```
  pressure_index = 0.4 × (RRR / cap) + 0.3 × (wickets_lost / 10) + 0.3 × (1 / overs_remaining)
  ```
- **For Tests:**
  ```
  pressure_index = 0.6 × (wickets_lost / 10) + 0.4 × (1 / overs_remaining)
  ```
  Tests rarely have RRR pressure — the pressure comes from losing wickets and time (sessions/days left).

- **For Innings 1 (batting first):** There's no target to chase, so we use a simpler formula based on wickets lost and overs consumed. The pressure is about not losing too many wickets and setting a competitive total.

**Practical Example:**
- CSK at 145/3 in 17th over chasing 170: RRR = 8.3, Current RR = 8.5. Pressure = moderate (0.35).
- CSK at 145/7 in 17th over chasing 190: RRR = 15.0, Current RR = 8.5. Pressure = extreme (0.85).
- A six by Dhoni in the second situation earns far more WPA (and impact credit) than in the first.

#### 14.3.5 Clutch Indicator

**Cricket Logic:** "Clutch" in cricket means performing in the most critical moments — the last few overs of a chase, the death overs when the game hangs in the balance. Some players consistently perform under this pressure (e.g., MS Dhoni, Jasprit Bumrah in death overs), and this ability should be explicitly captured.

**Implementation:**
- **T20/ODI:** Clutch = last 20% of overs (overs 16–19 in T20) **AND** pressure_index > 0.6 **OR** wickets_remaining ≤ 3 **OR** rr_pressure_gap > 3.
  - The AND/OR logic ensures we don't tag every death over as clutch — only those where the match is genuinely close.
- **Test:** Since Tests don't have a fixed endpoint, we use a different approach: rolling 60-ball (10-over) windows where the absolute WPA swing exceeds 0.15. These are periods where the match could have gone either way — the critical passages of play.

**Why we separate clutch from total WPA:** A player might have a high total WPA but most of it came in low-pressure situations (e.g., piling on runs at 180/2 with the match already won). Clutch WPA isolates the WPA earned specifically when the match outcome was uncertain.

#### 14.3.6 Leverage Index

**Cricket Logic:** Leverage measures how much a single delivery can swing the match. In a close game during death overs, every ball creates massive swings. In a one-sided contest, individual deliveries barely matter.

**Implementation:**
- **Innings 2 (chase):** `leverage_index = (required_rr / current_rr) + (wickets_lost / 10)`, clamped to [0, 3].
  - High RRR relative to current RR → high leverage (every run matters more).
  - More wickets lost → higher leverage (fewer batters left, each ball is more critical).
- **Innings 1:** A heuristic since there's no chase target: `leverage = 0.5 + wicket_fraction + clutch_factor × 0.5`.

**Practical Example:**
- Leverage = 0.5: A routine middle-over delivery in a one-sided match. WPA × 0.5 = half credit.
- Leverage = 2.5: A death-over delivery in a tight chase with 7 wickets down. WPA × 2.5 = amplified credit.
- This means Bumrah's dot ball in the 19th over of a tight match gets 5× the leverage credit of a routine middle-overs dot ball.

#### 14.3.7 Momentum Shift

**Cricket Logic:** Cricket matches often hinge on momentum swings — three quick wickets, or a sudden boundary spree. The momentum_shift feature captures these sudden changes in match trajectory.

**Implementation:** A rolling 12-ball window of the rate of change in required run rate.
- **Rising momentum_shift:** The required run rate is increasing rapidly — the bowling team is building pressure (e.g., 3 dot balls followed by a wicket).
- **Falling momentum_shift:** The batting team is scoring freely, reducing the required rate.

---

### 14.4 WPA Attribution — Why We Credit Both Batters and Bowlers Differently

**The Core Problem:** When a batsman hits a six, both the batsman (who hit it) and the bowler (who served the ball) are involved. When a bowler takes a wicket, both the bowler (who got the wicket) and the batsman (who got out) are involved.

**Our approach:**

- **Batter's WPA:** The raw WPA (positive = good for batting team). A batter who hits a boundary earns positive WPA. A batter who gets out on a dot ball earns negative WPA.
- **Bowler's WPA:** The **negated** raw WPA (positive = good for bowling team). A bowler who takes a wicket earns positive WPA. A bowler who gets hit for a six earns negative WPA.

This ensures a zero-sum attribution: every delivery's win probability change is credited to both the batter and bowler, from their respective perspectives.

**Batter-Specific Metrics:**

| Metric | Cricket Logic |
|---|---|
| **total_wpa** (sum) | Net match impact. Did this batter, overall, advance their team's win probability? A batter who scores 40(30) with mostly useful runs might have +0.15 total WPA. A batter who scores 40(45) consuming balls at the wrong time might have −0.02. |
| **avg_wpa** (mean per ball) | Quality per delivery. A batter facing 10 balls with +0.10 total WPA is more impactful per ball than one facing 50 balls with the same total. |
| **positive_wpa_pct** | What fraction of deliveries had a positive impact? This penalizes batters who dot frequently — each dot ball typically has negative WPA (time wasted in a chase). |
| **leverage_wpa** (Σ WPA × leverage) | Impact in high-stakes moments. Amplifies WPA earned in crunch situations. MS Dhoni's typical innings might have moderate total_wpa but extremely high leverage_wpa because his runs come in the death overs. |
| **consistency_wpa** | `positive_wpa_pct × (1 − volatility)`. Rewards batters who contribute positively on most balls with steady impact, as opposed to feast-or-famine (big hits interspersed with many dots). |
| **presence_impact** | `P(win when batter leaves) − P(win when batter arrives)`. This captures the total shift in match fortunes during the batter's entire stay. A batter who walks in at P(win) = 0.30 and leaves (dismissed or at end of innings) with P(win) = 0.65 has a presence_impact of +0.35 — they transformed the match. |
| **clutch_wpa** | WPA earned specifically in clutch situations (last 20% of overs + high pressure). Some batters score beautiful fifties in easy situations but fail in tight games. Clutch WPA isolates their crunch-time value. |

**Bowler-Specific Metrics:**

| Metric | Cricket Logic |
|---|---|
| **total_wpa** (negated sum) | Net bowling impact. A bowler with +0.20 total WPA reduced the batting team's win probability by 20 percentage points across their spell. |
| **wicket_wpa** | WPA earned specifically on wicket-taking deliveries. Not all wickets are equal — dismissing Kohli at 80/1 in a chase earns far more WPA than dismissing a tail-ender at 180/8 in a dead match. This metric captures the quality and timing of wickets. |
| **economy_wpa** | WPA earned from dot balls. Bowling dots builds pressure; each dot ball in a chase slightly increases the required run rate for the batting team, earning small but cumulative WPA. This rewards containment bowlers who may not take many wickets but strangle the scoring. |
| **spell_efficiency** | `total_wpa / balls_bowled`. Impact per delivery. A bowler who bowls 24 balls (4 overs) with +0.15 WPA is more efficient than one who bowls the same with 0.15 spread across 30 balls. |
| **leverage_wpa** | WPA × leverage_index, from the bowler's perspective. Bumrah bowling the 19th over in a tight match and going for just 6 runs earns enormous leverage_wpa. |

---

### 14.5 Batting Impact — Why 7 Components and These Specific Weights

**Design Philosophy:** A batter's value in cricket cannot be captured by a single number. We identified **7 distinct dimensions** of batting contribution, each representing a different aspect of batting value:

#### Component 1: Total WPA (T20 weight: 0.27)

**What it measures:** The overall win probability added by the batter across all deliveries.

**Why it has the highest weight:** This is the most direct measure of match impact. It answers: "By how much did this batter's innings change the outcome?" Everything else is a refinement of this core metric.

**Cricket Case:** V Kohli scoring 73(49) in a chase might earn +0.25 total WPA because each of his runs came at times when his team needed them. Compare with KL Rahul scoring 73(57) in the same match where most runs came when the chase was already comfortable — he might earn only +0.08 total WPA.

#### Component 2: Presence Impact (T20: 0.13, Test: 0.22)

**What it measures:** The change in win probability from when the batter arrived at the crease to when they left.

**Why it matters:** This captures the **innings-level contribution**, not just ball-by-ball. A batter who steadies a collapse (comes in at 30/3 and leaves at 120/4) has an enormous presence impact even if their individual deliveries weren't flashy.

**Why higher weight in Tests (0.22 vs 0.13):** In Tests, occupying the crease for extended periods is itself valuable. A batsman who bats 200 balls for 60 runs might save a Test match — their presence at the crease is the contribution, and presence_impact captures this.

**Cricket Case:** Pujara scoring 50(175) in a 4th-innings run-chase defense might have a moderate total_wpa but an enormous presence_impact, because while he was batting, the win probability steadily increased from "likely defeat" toward "likely draw/win."

#### Component 3: Leverage WPA (T20: 0.13, Test: 0.10)

**What it measures:** WPA weighted by the leverage index (how much the match outcome was at stake on each delivery).

**Why it matters:** This separates "runs in crunch moments" from "runs in dead time." Two batters may have the same total_wpa, but the one who earned it during the 18th over of a tight chase is more valuable than the one who earned it during the 5th over of a lopsided match.

**Why higher weight in T20 (0.13 vs 0.10):** In T20s, many matches are decided in the death overs — leverage concentrates there. In Tests, match-deciding moments are more spread out.

**Cricket Case:** Dhoni's 10*(6) in a tight finish might have higher leverage_wpa than Dhawan's 50(30) in the powerplay, because every ball Dhoni faced had 3×–4× the leverage of Dhawan's deliveries.

#### Component 4: Runs Contribution (T20: 0.14, Test: 0.20)

**What it measures:** The batter's share of their team's total innings runs (as a percentage).

**Why it matters alongside WPA:** WPA can be noisy for lower-order batters who face very few balls. Runs contribution provides a direct, stable measure of how much the batter contributed to the team's total.

**Why higher weight in Tests (0.20 vs 0.14):** In Tests, individual centuries often define match outcomes. A batter scoring 120 out of a team total of 350 (34% contribution) has carried their team's innings. In T20s, totals are shorter and more distributed.

**Cricket Case:** If AB de Villiers scores 80 out of RCB's 180 (44% contribution), his runs_contribution is 0.44. If Kohli scores 30 out of the same 180 (17%), his contribution is 0.17. De Villiers gets a significantly higher score on this component, regardless of WPA.

#### Component 5: Clutch WPA (T20: 0.09, Test: 0.08)

**What it measures:** WPA earned specifically in high-pressure, match-defining moments.

**Why it's separate from total WPA:** We want to explicitly reward players who perform under pressure. A player who performs well in easy situations but disappears in crunch moments should not be rated as highly as one who is consistently calm under pressure.

**Cricket Case:** R Ashwin bowling the 19th over with the batting side needing 15 to win and going for just 5 runs — all in a clutch zone — earns large clutch_wpa. Compare with a bowler who had a great economy in overs 2–4 but got hit in the death: their total_wpa might be decent, but clutch_wpa would be low or negative.

#### Component 6: Consistency WPA (T20: 0.13, Test: 0.13)

**What it measures:** `positive_wpa_pct × (1 − volatility)`. How reliably the batter contributes positively on each delivery.

**Why it matters:** Cricket rewards players who are **consistently useful**, not just explosive. A batter who hits a six then dots four balls has a volatile impact. A batter who rotates strike with singles and occasional boundaries has consistent positive impact on most deliveries.

**Why same weight across formats (0.13):** Consistency is universally valued in cricket — whether T20 or Test, teams need reliable performers.

**Cricket Case:** Consider two batters, both with 40 runs in a match:
- Batter A: 6, 0, 0, 0, 4, 0, 6, 0, 0, 4, 0, 0, 6, 0, 1, 0, 0, 6, 0, 1, 0, 0, 4, 0, 0, 2 → volatile (many dots interspersed with big hits). High WPA on boundary balls, negative WPA on dots. Positive_wpa_pct ≈ 30%, volatility high → low consistency_wpa.
- Batter B: 1, 2, 0, 1, 4, 1, 2, 1, 0, 4, 1, 0, 2, 1, 1, 4, 1, 2, 0, 4, 1, 2, 0, 1, 2, 2 → steady accumulation. Positive_wpa_pct ≈ 75%, volatility low → high consistency_wpa.

Batter B's innings may look less exciting but is more valuable for the team because they don't waste deliveries.

#### Component 7: Dot Efficiency (T20: 0.11, Test: 0.05)

**What it measures:** `1 − (dot ball percentage faced)`. A batter who faces fewer dots is more efficient.

**Why it matters in T20 (0.11) but less in Tests (0.05):** In T20 cricket, every delivery is precious — there are only 120 balls in an innings. A batter who dots 60% of deliveries is consuming team resources without contributing. In Tests, patient defense (dot balls) is a valid and often necessary strategy.

**Cricket Case:** Suryakumar Yadav might face 25 balls with only 5 dots (20% dot rate) → dot_efficiency = 0.80. Pujara in a Test might face 100 balls with 70 dots (70% dot rate) → dot_efficiency = 0.30. But in Tests, Pujara's defense is valued differently, which is why the weight is only 0.05 in Tests vs 0.11 in T20s.

---

### 14.6 Why Format-Specific Weights?

**The Core Insight:** What makes a player valuable differs fundamentally across formats:

| Aspect | T20 | ODI | Test |
|---|---|---|---|
| **Batting priority** | Strike rate, boundaries | Balance of accumulation + acceleration | Occupation, patience, big scores |
| **Bowling priority** | Death bowling, economy | Consistency across 10 overs | Sustained spells, wearing down batsmen |
| **Match-defining moments** | Death overs (16–19) | Slog overs (40–50) | Sessions 4–5 on day 3–4 |
| **WPA concentration** | Heavily in last 4 overs | Spread but peaks at end | Spread across days |
| **Dot balls** | Very costly | Moderately costly | Often a valid strategy |

**How weights change:**

- **Total WPA weight drops from T20 (0.27) → Test (0.20):** In Tests, WPA swings per ball are smaller (a single delivery rarely shifts P(win) by >10%), so we reduce its weight and boost presence_impact.
- **Presence Impact increases from T20 (0.13) → Test (0.22):** In Tests, the ability to bat for long periods (occupying the crease) is intrinsically valuable. Presence captures this.
- **Runs Contribution increases from T20 (0.14) → Test (0.20):** In Tests, individual big scores (centuries, double-centuries) win matches. The runs_contribution reflects this.
- **Dot Efficiency drops from T20 (0.11) → Test (0.05):** Leaving balls and blocking are legitimate and important Test batting techniques.

---

### 14.7 Position Multiplier — Rewarding Over-Performance vs Expected Contribution

**Cricket Logic:** Not all batting positions are equal. Openers (positions 1–2) are expected to score ~27% of the team's runs. Number 7–8 batters are expected to score only ~13%. If a number 7 batter scores 30% of the team's runs, that's extraordinary. If an opener scores 14%, that's underperforming.

**Expected Run Share by Position:**

| Position | Expected Share | Cricket Reason |
|---|---|---|
| 1 (Opener) | 14% | Faces the new ball, expected to set the platform |
| 2 (Opener) | 13% | Same as above |
| 3 | 13% | Typically the best batter, enters early |
| 4 | 12% | Middle-order anchor |
| 5 | 11% | Middle-order |
| 6 | 9% | Lower middle-order, may not always bat |
| 7 | 7% | All-rounder slot |
| 8 | 6% | Bowling all-rounder |
| 9 | 4% | Tail-ender |
| 10 | 3% | Tail-ender |
| 11 | 2% | Last man |

**Formula:**
```
share_ratio = actual_contribution% / expected_contribution%
position_multiplier = 0.85 + 0.16 × clamp(share_ratio, 0.5, 2.5)
```

- **Ratio = 1.0** (performing as expected): position_mult = 1.01 → essentially neutral.
- **Ratio = 2.0** (scoring double the expected share): position_mult = 1.17 → 17% bonus.
- **Ratio = 0.5** (underperforming): position_mult = 0.93 → 7% penalty.

**Cricket Case:**
- Hardik Pandya (batting #6, expected 9%) scores 50 out of MI's 160 (31% contribution). Ratio = 31/9 = 3.4, clamped to 2.5. Multiplier = 0.85 + 0.16 × 2.5 = 1.25. His raw impact gets a 25% boost because he massively outperformed his position's expectation.
- Rohit Sharma (batting #1, expected 14%) scores 15 out of 160 (9% contribution). Ratio = 9/14 = 0.64. Multiplier = 0.85 + 0.16 × 0.64 = 0.95. A slight penalty because an opener contributing only 9% is below expectation.

**Why this is fair:** It prevents lower-order players from being systematically underrated just because they bat fewer balls, and prevents openers from being overrated just because they bat first.

---

### 14.8 Volume Multiplier — Why Raw Impact Needs Scaling by Contribution Volume

**The Problem:** A batter who faces 3 balls and hits a six might have a high percentage-based impact per ball, but their overall contribution is minimal. A batter who faces 50 balls and scores 70 has contributed far more to the innings.

**Solution — Format-Aware Volume Multiplier:**

| Format | Formula | Cricket Logic |
|---|---|---|
| **T20** | `0.40 + 0.80 × contrib_pct + 0.30 × (balls/60)` | In a 120-ball innings, facing 60+ balls means you batted through half the innings. The 0.40 base ensures even brief cameos get some credit. |
| **ODI** | `0.35 + 0.65 × contrib_pct + 0.40 × (balls/100)` | 100 balls is a significant ODI innings (~33% of the total). Ball-consumption has more weight (0.40 vs 0.30) because occupying the crease matters more in 50-over cricket. |
| **Test** | `0.30 + 0.50 × contrib_pct + 0.50 × (balls/150)` | 150 balls = 25 overs = a substantial Test innings. Balls-faced gets the highest weight (0.50) because **survival** is itself a contribution in Tests. |

**Cricket Case (T20):**
- Player A: 6*(3) — spectacular cameo but tiny volume. contrib = 3%, balls/60 = 0.05. Volume_mult = 0.40 + 0.024 + 0.015 = 0.44. Raw impact × 0.44.
- Player B: 75(50) — major innings. contrib = 45%, balls/60 = 0.83. Volume_mult = 0.40 + 0.36 + 0.25 = 1.01. Raw impact × 1.01 (full credit).

This ensures that substantial innings are rewarded proportionally while brief cameos get proportionally scaled.

---

### 14.9 Bowling Impact — Why 8 Components

The 8 bowling components mirror the batting components but are tailored to a bowler's unique contribution:

#### Component 1: Total WPA (T20: 0.25)
**Same logic as batting.** The net win probability the bowler added for their bowling team across all deliveries.

#### Component 2: Wicket WPA (T20: 0.15, Test: 0.25)
**Why separate from total WPA:** Wickets are the highest-impact events in cricket. A single wicket can be worth +0.10 to +0.30 WPA in a crucial moment. By isolating wicket-taking WPA, we reward bowlers who take wickets at critical times, not just any time.

**Why higher in Tests (0.25):** In Tests, wickets are the currency of victory. You need 20 wickets to win. Each wicket is more valuable because they're harder to come by.

**Cricket Case:** Bumrah dismissing Kohli at 80/2 in a chase of 170 in the 15th over might earn +0.18 wicket_wpa. But dismissing a number 11 at 160/9 might earn only +0.03 wicket_wpa. Same "1 wicket" in traditional stats, but vastly different impact.

#### Component 3: Leverage WPA (T20: 0.12)
Same principle as batting — WPA amplified in high-leverage situations.

#### Component 4: Economy WPA (T20: 0.07, ODI: 0.10)
**What it measures:** WPA earned from bowling dot balls.

**Why it matters:** Dot balls create pressure. In a chase, every dot ball increases the required run rate for the remaining balls. A bowler who bowls 15 dots in 4 overs at the death is building immense pressure even without taking wickets.

**Why higher in ODI (0.10 vs 0.07):** In ODIs, containment bowling (going at 4 RPO in the middle overs) is a key strategy. Bowlers who dry up runs force batsmen into risky shots, leading to wickets at the other end.

#### Component 5: Clutch WPA (T20: 0.08)
Same as batting — WPA in high-pressure situations.

#### Component 6: Spell Efficiency (T20: 0.07, Test: 0.10)
**What it measures:** `total_wpa / balls_bowled` — impact per ball.

**Why higher in Tests (0.10):** In Tests, bowlers who maintain high impact per delivery over long spells (15–20 overs) are extraordinarily valuable. This rewards sustained quality over volume.

#### Component 7: Phase Difficulty (T20: 0.15, Test: 0.08)
**The key innovation:** Not all bowling overs are equal. Death overs (16–19) in T20 are the hardest phase to bowl — batters are set, field restrictions are lifted, and the game is at its most intense.

**Implementation:**
```
death_econ_score = 1 − ((actual_economy − elite_benchmark) / (bad_benchmark − elite_benchmark))
death_wicket_score = min(wicket_rate / 2.0, 1.0)
phase_difficulty = death_fraction × (0.5 × econ_score + 0.5 × wicket_score) 
                 + pp_fraction × (0.5 × pp_econ_score + 0.5 × pp_wicket_score)
```

**Elite vs Bad benchmarks (T20):**
- Death overs: Elite economy = 6 RPO, Bad = 15 RPO.
- Powerplay: Elite = 5 RPO, Bad = 12 RPO.

**Cricket Case:** Bumrah bowling overs 17–20 with an economy of 7.5 and 2 wickets:
- death_econ_score = 1 − (7.5−6)/(15−6) = 0.83 (excellent)
- death_wicket_score = (2/4)/(2.0) = 0.25 per over → capped → ~0.50
- Combined: 0.5 × 0.83 + 0.5 × 0.50 = 0.665 for the death portion.

Compare a bowler who only bowls overs 3–6 in the middle with the same economy — zero death_fraction, so phase_difficulty = 0. This explicitly rewards bowlers trusted with the toughest overs.

**Why 0.15 weight in T20:** Death bowling often decides T20 matches. This significant weight ensures death specialists like Bumrah, Arshdeep Singh, and Harshal Patel are properly valued.

#### Component 8: Dot Pressure (T20: 0.11, Test: 0.19)
**What it measures:** Percentage of dot balls bowled.

**Why highest in Tests (0.19):** In Tests, bowling maidens and dots is the primary pressure mechanism. A bowler who bowls 60% dots across a 15-over spell is building enormous pressure that leads to wickets, even if they come at the other end. In T20s, dots matter but less so (the batting team has ways to compensate).

---

### 14.10 Rolling Normalization — Why Recency-Weighted with Bayesian Dampening

**The Problem:** Raw impact scores from different seasons aren't directly comparable. A player might have played in 2008 (weaker bowling attacks) and 2024 (much stronger bowling). Additionally, recent form should matter more than distant past.

#### 14.10.1 Recency Weights

```python
RECENCY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
```

The most recent match gets weight 1.0, the 10th most recent gets 0.1. Beyond 10 matches, older data is excluded entirely.

**Cricket Logic:** A player's current ability is best represented by their recent performances. KL Rahul's form in his last 10 innings is more relevant than his form 5 seasons ago. But we don't just use the last match (too noisy) — 10 innings gives a balanced window.

#### 14.10.2 Consistency Modifier

```
consistency = max(0, 1 − (std_dev / mean))
adjusted_impact = rolling_impact × (0.8 + 0.2 × consistency) × comeback_bonus
```

- A player with consistently high impact (low std relative to mean) gets a bonus (up to 20%).
- A player with wildly fluctuating impact gets no bonus.

**Cricket Case:** 
- Kohli: Last 10 impacts = [0.45, 0.50, 0.42, 0.48, 0.55, 0.47, 0.43, 0.50, 0.46, 0.52]. Mean = 0.478, std = 0.037. Consistency = 1 − (0.037/0.478) = 0.92. Modifier = 0.8 + 0.2 × 0.92 = 0.984. Near-full credit.
- A Patel: Last 10 impacts = [0.80, 0.05, 0.70, 0.10, 0.75, 0.08, 0.65, 0.12, 0.72, 0.03]. Mean = 0.40, std = 0.33. Consistency = 1 − (0.33/0.40) = 0.175. Modifier = 0.8 + 0.2 × 0.175 = 0.835. A 16.5% reduction for inconsistency.

#### 14.10.3 Comeback Bonus

If a player's average impact in their last 5 innings exceeds their prior 5 innings by 20%, they receive a **10% bonus** (multiplier 1.1).

**Cricket Case:** A returning-from-injury player whose last 5 impacts are [0.50, 0.55, 0.60, 0.58, 0.62] vs prior 5 of [0.30, 0.25, 0.35, 0.28, 0.32]. Recent_avg = 0.57, prior_avg = 0.30. Improvement = 90% > 20%. Comeback_bonus = 1.1.

This rewards players who are trending upward and may be entering peak form.

#### 14.10.4 Bayesian Dampening

**The Problem:** A player with 3 incredible innings might look like the best player in the world. But with only 3 data points, we can't be confident.

**Solution:**
```
bayes_weight = innings_count / (innings_count + 5)
dampened_impact = bayes_weight × observed_impact + (1 − bayes_weight) × league_median
```

| Innings Played | bayes_weight | Reliance on Observed Data |
|---|---|---|
| 1 | 0.17 | 17% (heavily pulled toward median) |
| 3 | 0.38 | 38% |
| 5 | 0.50 | 50% (half observed, half prior) |
| 10 | 0.67 | 67% |
| 50 | 0.91 | 91% (almost entirely observed) |
| 100 | 0.95 | 95% |

**Cricket Case:** 
- Player X: 3 innings, all brilliant. Raw adjusted = 0.85. But dampened = 0.38 × 0.85 + 0.62 × 0.35 = 0.54. They're pulled toward league average.
- Player Y: 100 innings, consistently good. Raw adjusted = 0.70. Dampened = 0.95 × 0.70 + 0.05 × 0.35 = 0.68. Their observed data dominates.

This is statistically principled — it's the same idea behind why we don't crown someone "batsman of the tournament" after 2 matches.

#### 14.10.5 Logistic Normalization (0–100)

```
percentile = rank_percentile(dampened_impact)
impact_score = 100 / (1 + exp(−8 × (percentile − 0.5)))
```

This S-shaped curve:
- **Spreads out the middle** — Players at the 40th vs 60th percentile get clearly different scores.
- **Compresses the extremes** — The very best and very worst are capped, preventing outliers from distorting the scale.
- **Produces intuitive scores** — 50 = median, 80+ = elite, 30− = below average.

---

### 14.11 Overall Impact (0–90) — Why 11 Factors and These Specific Weights

**Design Philosophy:** A player's career-level impact cannot be captured by batting or bowling stats alone. The 11-factor composite captures the **full range of contributions** a cricketer makes:

#### Factor 1: Role-Weighted Performance Score (Weight: 0.28)

```
role_score = bat_weight × bat_contribution + bowl_weight × bowl_contribution
```

Where `bat_weight = bat_innings / total_innings` and `bowl_weight = bowl_innings / total_innings`.

**Cricket Logic:** A pure batter should be judged on batting, a pure bowler on bowling, and an all-rounder on both — weighted by how much they actually contributed in each role.

**Batting contribution** itself is a 5-factor sub-score:
- 30% log(runs) — volume of runs produced
- 20% batting average — quality of run production
- 15% strike rate — tempo of scoring
- 20% batting WPA — match impact
- 15% batting impact score — our composite batting score

**Bowling contribution:**
- 22% log(wickets) — wicket-taking volume
- 15% economy (inverted) — containment ability
- 22% bowling WPA — match impact
- 21% bowling impact score — our composite bowling score
- 20% phase difficulty — death/powerplay performance

**Why 0.28 weight (the highest):** At its core, cricket is about runs and wickets. Performance is the foundation; everything else is a modifier.

#### Factor 2: All-Rounder Bonus (Weight: 0.10)

```
allrounder_bonus = √(bat_contribution × bowl_contribution)   // only if ≥20 bat + ≥20 bowl innings
```

**Cricket Logic:** All-rounders like Jadeja, Narine, and Hardik Pandya contribute in both departments, effectively giving their team an extra "half-player." This is enormously valuable in team selection and match dynamics.

**Why geometric mean:** The geometric mean rewards **balance**. A player with 0.6 batting and 0.6 bowling (balanced all-rounder) gets √(0.36) = 0.60. But a player with 0.9 batting and 0.1 bowling (basically a batter) gets √(0.09) = 0.30. The formula automatically rewards genuine dual contributors.

**Why ≥20 innings threshold:** With fewer innings, the contribution scores are unreliable. We don't want a bowler who batted 3 times to get an all-rounder bonus.

#### Factor 3: Longevity (Weight: 0.12)

```
longevity = log(total_matches) / max(log(total_matches))
```

**Cricket Logic:** Players who sustain high-level performance over many seasons are inherently more valuable than those with a handful of great matches. Sachin Tendulkar playing 450+ Tests or Dhoni playing 350+ ODIs represents sustained excellence. The logarithmic scaling ensures diminishing returns — the difference between 50 and 100 matches matters more than between 200 and 250.

#### Factor 4: Player of Match Awards (Weight: 0.10)

**Cricket Logic:** POM awards are selected by expert judges who watch the match. They capture match-winning performances that statistics alone might miss (e.g., a fielding run-out that changed the game, or a tactical innings that was more about building partnerships).

#### Factor 5: POM Efficiency (Weight: 0.06)

```
pom_rate = pom_awards / total_matches
```

**Cricket Logic:** A player who wins POM 15 times in 50 matches (30% rate) is more consistently match-winning than one who wins 15 times in 200 matches (7.5%). This rate metric complements the absolute count.

#### Factor 6: Clutch Contribution (Weight: 0.06)

```
total_clutch = |total_clutch_bat_wpa| + |total_clutch_bowl_wpa|
```

**Cricket Logic:** Players who perform in high-pressure moments are more valuable in knockout games, finals, and do-or-die scenarios. MS Dhoni's legendary calm under pressure is exactly what clutch contribution captures.

We use absolute values because both positive and negative clutch contributions indicate that the player was involved in match-defining moments (we want to identify who shows up in clutch situations, then rank them by how they performed there).

#### Factor 7: Volume (Weight: 0.07)

```
volume = log(total_runs) for batters, log(total_wickets) for bowlers
```

**Cricket Logic:** Raw volume of production matters. A player with 5,000+ IPL runs or 100+ IPL wickets has proven themselves over a large sample, which lends reliability to their impact scores.

#### Factor 8: Win Bonus (Weight: 0.08)

**Cricket Logic:** Players who are consistently on winning teams contribute to those wins in ways that individual stats may not capture — team leadership, positive dressing room impact, smart cricket decisions, efficient running between wickets, fielding contributions.

**Implementation:** All players on the winning team receive credit, normalized by the total number of wins across their career.

#### Factor 9: Captaincy Bonus (Weight: 0.05)

**Cricket Logic:** Captains carry additional responsibility — field placements, bowling changes, tactical decisions, toss choices. A captain who leads their team to consistent wins deserves additional credit.

**Important:** This is **0 for non-captains**. Only players who actually captained receive this bonus, and it scales with their captaincy win rate and matches captained.

**Cricket Case:** MS Dhoni — captained CSK in ~150+ matches, won ~60%+, multiple IPL titles. His captaincy_score would be near the top, earning him a bonus that properly reflects his leadership contribution.

#### Factor 10: Playoff Bonus (Weight: 0.04)

**Cricket Logic:** Players on teams that consistently reach the playoffs are more likely to have contributed to team success. This acts as a mild team-success bonus.

**Why small weight (0.04):** We don't want to overly penalize great players stuck on weak franchises (e.g., a great player on an underperforming team). It's a light factor, not a dominant one.

#### Factor 11: Bowling Visibility (Weight: 0.04)

**Cricket Logic:** In traditional cricket analysis, bowlers are often undervalued compared to batsmen (batsmen get more POM awards, more highlight reels, more media attention). This factor gives a small boost to players with strong bowling impact to counteract that systemic bias.

**Implementation:** For players who bowl, it's their bowling impact percentile rank. For non-bowlers, it's 0.

---

### 14.12 Final Score: Bayesian Dampening + Power-Curve Scaling (0–90)

**Step 1: Bayesian dampening (sample-size correction)**

```
bayes_weight = total_innings / (total_innings + 12)
prior = median of all players' overall_impact_raw
dampened = bayes_weight × raw + (1 − bayes_weight) × prior
```

Here `min_innings = 12` (higher than the per-match dampening of 5) because career-level assessment requires more data. A player with 12 total innings is only 50% trusted; a player with 100+ innings is 89% trusted.

**Why this matters:** Without this, a player who played 5 IPL matches and happened to win all of them could rank in the top 10. The dampening ensures they're pulled toward the average until they've proven themselves over sufficient sample.

**Step 2: Power-curve normalization to 0–90**

```
percentile = rank_percentile(dampened)
overall_impact = 90 × (percentile ^ 0.7)
```

**Why 0–90 instead of 0–100?** Scoring 90 should feel like the absolute pinnacle — only the greatest IPL players of all time reach that mark. Leaving headroom above 90 (which no current player reaches) makes the scale feel more aspirational.

**Why power curve (^0.7)?**
- It **spreads out the top**: The difference between the #1 and #5 player is visible (e.g., 90.0 vs 89.5).
- It **compresses the bottom**: The difference between the #600 and #700 player is smaller (e.g., 12 vs 8), because differentiating between marginal players is less important.
- It creates a natural distribution where median ≈ 55, top ≈ 90, bottom ≈ 2.

---

### 14.13 Summary of Cricket Scenarios and How CricImpact Handles Them

| Scenario | How CricImpact Captures It |
|---|---|
| Dhoni's 10*(6) in a tight finish winning the match | High leverage_wpa (death overs, close game), high clutch_wpa (last 20% + high pressure), high presence_impact (walked in with P(win)=0.30, left with 1.0). Position multiplier rewards lower-order impact. |
| Kohli's 73*(49) anchoring a chase of 180 | High total_wpa (consistent run scoring keeping RRR manageable), high consistency_wpa (singles and twos with periodic boundaries), high runs_contribution (40%+ of the total). |
| Bumrah's 4-0-20-3 in the death overs of a close match | Massive phase_difficulty_bonus (death overs with elite economy), high wicket_wpa (3 wickets in high-leverage situations), high leverage_wpa, high clutch_wpa (death overs + tight match). |
| A batter scoring 50(55) in a losing cause at a low-scoring venue | Moderate runs_contribution (50 out of maybe 130), but low total_wpa (team lost, so his innings didn't shift P(win) upward enough). Venue_difficulty gives some context adjustment. The system correctly values this lower than a 50 in a winning chase. |
| An all-rounder scoring 30(20) and taking 2/25 | Both bat_contrib and bowl_contrib are non-zero. Geometric mean gives allrounder_bonus. Role-weighted score uses both batting and bowling innings. |
| A player with 5 matches vs a veteran with 100 matches | Bayesian dampening at both per-match level (min_innings=5) and career level (min_innings=12) pulls the 5-match player toward league average. The veteran's observed data dominates. Longevity factor (log(matches)) additionally rewards the veteran. |
| A captain who leads their team to 4 IPL titles | Captaincy bonus (weight 0.05) with high captaincy_score (high win rate + many matches captained). Win bonus (weight 0.08) from being on winning teams. Playoff bonus (weight 0.04) for consistent playoff appearances. |
| A bowler with great economy but few wickets (containment bowler) | Economy_wpa captures the pressure from dot balls. Dot_pressure_norm (weight 0.11 in T20) rewards high dot percentage. Even without wicket_wpa, these components ensure containment bowlers are valued. |
| A player who was mediocre for years but has recently been outstanding | Comeback bonus gives 10% uplift if recent 5 innings avg > prior 5 by 20%. Recency weights (1.0 for most recent, 0.1 for 10th) ensure recent form dominates the rolling average. |

---

*This thought-process document accompanies the CricImpact system for the CricHeroes HackaMined Data Science Hackathon.*
*Every design decision is rooted in cricket domain knowledge and statistical best practices.*
