# Cricket Player Impact Metric

A data-driven system that quantifies cricket player performance using **Win Probability Added (WPA)**: a machine-learning model prices every delivery, and players are credited with how much they moved their side's chance of winning. Built for the CricHeroes HackaMined Data Science Hackathon.

## Overview

The pipeline processes ball-by-ball match data. A **monotonic gradient-boosting win probability model** prices every delivery, each match scored by a model that never saw its result, and players are rated 0–100 on the win probability they added.

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
│   ├── win_prob_model.py            # Monotonic gradient boosting win probability model, 5-fold cross-fitting
│   ├── rating.py                    # Player rating used by career, season and role leaderboards
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

## Running the app

The app has a FastAPI backend (`backend/`) and a React frontend (`frontend/`). Run each in its own terminal.

**1. Backend (port 8000)**

```bash
pip install -r requirements.txt -r backend/requirements.txt
cd backend
python -m uvicorn main:app --port 8000
```

The backend reads the CSVs produced by `run_pipeline.py` (see Usage below), including `player_match_context.csv`, `ball_win_prob.csv` and `model_card.json`.

**2. Frontend (port 5173)**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend proxies `/api` to the backend.

**For demos and interviews, use the production build.** Development mode compiles pages on first visit, so the first load can take 10+ seconds. The production build loads the home page replay in under half a second:

```bash
cd frontend
npm run build
npm run preview        # http://localhost:4173, same /api proxy
```

**Optional: AI match summaries.** Create a `.env` file in the project root with `GROQ_API_KEY=your_key` and restart the backend.

### Frontend pages

| Page | What it shows |
|------|---------------|
| Home | Live ball-by-ball win probability replay of famous finals, how the metric works, model card, runs-vs-impact insight, league charts |
| Leaderboard | Filterable, sortable rankings with role mix, impact tiers and batting-vs-bowling scatter |
| Player Profile | Impact gauge, batting / bowling / results donuts, skill percentile radar, innings trend, partners, rivalries |
| Compare | Two players side by side: radar overlay, stat-by-stat bars, paired donuts, season trend, head-to-head |
| Match Explorer | Any IPL match: win probability replay with turning points, team impact share, player impact bars, WPA vs clutch scatter, AI summary |
| Upload Match | Score a new match from Cricsheet CSVs with the trained win probability model |

## Rating method (v2)

**Win probability model.** `HistGradientBoostingClassifier` on 11 match-state features, with monotonic rules (a wicket can never raise the batting side's chance, extra runs can never lower it). The run rate is smoothed toward a par of 8 an over (as if 12 overs had already been bowled at par): the raw run rate swings wildly in the first few overs, which made early dot balls look like costly cliffs. Over number and phase are not used, since balls bowled already carries that information without step jumps. Every match is scored by a model trained on the other 80% of matches (5-fold cross-fitting by match), so no player is credited using a model that already knew the result.

**Crediting.** Each ball's Win Probability Added (WPA) is the change in win probability it caused. The whole match is credited: the first ball is measured from a neutral 50%, the last ball of the chase is measured to the actual result, and the jump at the innings break (when the first-innings total becomes a known target) goes to the first-innings batters in proportion to their runs and against the bowlers in proportion to runs conceded. So the credits in a match add up to the full swing from 50% to the result.

**Player rating** (`cricket_impact/rating.py`), the same formula for career, season and role views:

| Part | Weight |
|---|---|
| Impact per innings, shrunk toward the league average | 45% |
| Career value above a replacement-level player (WAR-style) | 45% |
| Player of the Match rate (shrunk) | 6% |
| Captaincy record, team win rate, share of seasons in the playoffs | 4% (1.33% each) |

* **Centring.** Batting WPA is measured against an average batting innings and bowling WPA against an average bowling spell, so batters and bowlers start from the same baseline in mixed tables.
* **Shrinkage** is estimated from the data (empirical Bayes): per-match impact is very noisy compared with the real spread between players, so each player's average is blended with the league average as if they had about 80 extra average matches. One hot season over 11 matches no longer tops the table.
* **Replacement level** is the average impact of fringe players (under 25 career matches), about -2.2 points per match. Career value = (shrunk impact per match - replacement level) x matches, so a long career at a solid level is rewarded, as in baseball's WAR.
* Impact parts are z-scored and blended 50/50, then the result is rescaled 0–100 by min–max so real gaps stay visible. Strike rate and economy use career totals. Season leaderboards use only that season's innings, judged against career-level shrinkage and replacement level.

**Validation** (`python validate_model.py`, results also in `cricket_impact/model_card.json`):

| Check | v1 Random Forest | v2 |
|---|---|---|
| AUC, both 5-fold cross-fitted | 0.782 | 0.787 |
| Brier score, both 5-fold cross-fitted | 0.188 | 0.186 |
| Early first-innings dot balls moving win probability by more than 3 points | 5.0% | 5.1% |
| Wickets that raise the batting side's win probability | 10.1% | 0.2% |
| Boundaries that lower it | 12.0% | 0.1% |
| Match's top WPA player was Player of the Match | 41.6% | 40.9% |
| Player of the Match among the match's top 3 by WPA | 69.4% | 71.0% |

v1's originally reported AUC of 0.815 came from a single 230-match test split. Both models are credited the same way for the Player of the Match check.

**Known limitations.** Innings length is inferred from overs bowled, and DLS targets are not modelled. Match-level scores remain a rolling average of the last 10 innings. WPA rewards tempo in context, so accumulators who score at around league strike rate rate lower than their run totals suggest.

## Usage

```bash
# Run with default data folder
python run_pipeline.py --data_folder ./ipl_male_csv2

# Compare the improved model with the original one
python validate_model.py --data_folder ./ipl_male_csv2   # old vs new model comparison

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
