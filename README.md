# FIFA World Cup ML Prediction

A machine learning pipeline that predicts FIFA World Cup outcomes using historical match data, Elo ratings, and Monte Carlo simulation. Built to forecast the **2026 World Cup** with full backtesting on 2018 and 2022.

## Results

**2026 World Cup Win Probabilities** (10,000 Monte Carlo simulations):

| Rank | Team | Win Probability | 95% CI |
|------|------|----------------|--------|
| 1 | Spain | 21.4% | [20.6%, 22.2%] |
| 2 | Argentina | 19.5% | [18.7%, 20.3%] |
| 3 | France | 11.5% | [10.9%, 12.2%] |
| 4 | England | 5.8% | [5.4%, 6.3%] |
| 5 | Brazil | 5.6% | [5.2%, 6.1%] |
| 6 | Germany | 5.0% | [4.6%, 5.4%] |
| 7 | Netherlands | 4.7% | [4.3%, 5.1%] |
| 8 | Portugal | 3.9% | [3.5%, 4.3%] |

**Model accuracy on held-out test set (2022 onwards):**

| Model | Accuracy | Log-Loss | RPS |
|-------|----------|----------|-----|
| Logistic Regression | **59.9%** | **0.8794** | **0.1726** |
| Random Forest | 59.8% | 0.8797 | 0.1728 |
| XGBoost | 59.4% | 0.8856 | 0.1739 |
| Naive baseline (always Win) | 47.5% | 18.93 | 0.4102 |

## Project Structure

```
WorldCup_Project/
├── scripts/
│   ├── Merged_data.py               # Phase 1: feature engineering
│   ├── Phase 2 - Train the models.py # Phase 2: model training & evaluation
│   ├── MonteCarlo-2026.py           # Phase 3: 2026 tournament simulation
│   ├── MonteCarlo-2022.py           # Backtest: 2022 tournament simulation
│   ├── backtest_worldcups.py        # Backtest on 2018 & 2022 WCs
│   ├── multi_tournament_backtest.py # Multi-tournament validation
│   ├── walk_forward_cv.py           # Walk-forward cross-validation
│   ├── add_rps.py                   # Ranked Probability Score comparison
│   ├── confidence_intervals.py      # Wilson score CIs for predictions
│   ├── feature_ablation.py          # Feature importance ablation study
│   ├── elo_sensitivity.py           # Elo K-factor sensitivity analysis
│   ├── diebold_mariano.py           # Diebold-Mariano statistical test
│   ├── linkedin_visuals.py          # Publication-quality figures
│   └── run_all_analyses.py          # Runner for all analysis scripts
├── data/
│   ├── features_phase1.csv          # Engineered feature matrix (output of Phase 1)
│   └── current_elo_ratings.csv      # Final Elo ratings for all teams
├── models/
│   ├── best_model.pkl               # Best trained model + preprocessors
│   └── all_models.pkl               # All three trained models
├── outputs/
│   ├── figures/                     # All generated plots
│   └── results/                     # CSV results and analysis outputs
├── InternationalFootball_Results(1872-2026)/
│   ├── results.csv                  # Match results (150+ years)
│   ├── goalscorers.csv
│   ├── shootouts.csv
│   └── former_names.csv             # Country name standardisation map
└── Fifa_worldranking(1992-2024)/
    └── fifa_ranking-2024-06-20.csv  # FIFA ranking points history
```

## Pipeline

### Phase 1 — Feature Engineering (`Merged_data.py`)

Processes ~40,000 international matches (1990–2026) and engineers 26 features per match:

- **Elo ratings** — computed from scratch with K=40, capturing team strength over time with no data leakage (pre-match values only)
- **FIFA ranking points** — point-in-time lookup from official FIFA rankings
- **Rolling form** — 10-game rolling win rate, goals scored/conceded
- **Head-to-head record** — historical win rate between the two teams
- **Days of rest** — days since each team's last match
- **World Cup experience** — number of prior World Cups each team has participated in
- **Tournament type dummies** — World Cup, qualifier, friendly, other competitive

Outputs `data/features_phase1.csv` and `data/current_elo_ratings.csv`.

### Phase 2 — Model Training (`Phase 2 - Train the models.py`)

Trains three classifiers on a temporal train/test split (cutoff: 2022-01-01):

- **Logistic Regression** (with StandardScaler)
- **Random Forest** (500 trees, max depth 10)
- **XGBoost** (500 estimators, multiclass softprob)

All preprocessing (imputation, scaling) is fit on training data only to prevent leakage. The best model by log-loss is saved to `models/best_model.pkl`.

### Phase 3 — Monte Carlo Simulation (`MonteCarlo-2026.py`, `MonteCarlo-2022.py`)

Simulates the full tournament bracket 10,000 times:

1. Precomputes match probabilities for all team pairs
2. Simulates group stage (draws allowed, tiebreaker by points → goal difference → Elo)
3. For 2026's 48-team format: top 2 from each of 12 groups + best 8 third-place teams advance to Round of 32
4. Knockout rounds redistribute draw probability 50/50 between the two teams
5. Aggregates champion counts across all simulations

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `add_rps.py` | Ranked Probability Score — penalises confident wrong predictions |
| `confidence_intervals.py` | Wilson score 95% CIs for all win probabilities |
| `feature_ablation.py` | Measures accuracy drop when each feature group is removed |
| `elo_sensitivity.py` | Tests model stability across different Elo K-factor values |
| `walk_forward_cv.py` | Walk-forward cross-validation on annual windows |
| `diebold_mariano.py` | Statistical test for significant differences between models |
| `backtest_worldcups.py` | Full tournament backtests on WC 2018 and WC 2022 |
| `multi_tournament_backtest.py` | Validation across multiple major tournaments |

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm
```

### Full pipeline

```bash
# Phase 1 — build features (run from project root)
python scripts/Merged_data.py

# Phase 2 — train models
python "scripts/Phase 2 - Train the models.py"

# Phase 3 — 2026 predictions
python scripts/MonteCarlo-2026.py

# Phase 3 — 2022 backtest
python scripts/MonteCarlo-2022.py

# All validation analyses
python scripts/run_all_analyses.py
```

All scripts should be run from the project root directory.

## Data Sources

- **International football results** (1872–2026): match outcomes, scores, tournament names — [Kaggle: martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **FIFA World Rankings** (1992–2024): official ranking points history — [Kaggle: cashncarry/fifia-world-ranking](https://www.kaggle.com/datasets/cashncarry/fifia-world-ranking)

## Key Design Decisions

- **Temporal split** for train/test: all data before 2022 trains the model; 2022 onwards is the held-out test. This mirrors real deployment — you never have future data.
- **No data leakage**: Elo ratings, FIFA points, and rolling form are all computed using only data available at match time.
- **Draws in group stage, not knockout**: the model predicts three outcomes (W/D/L); draws are redistributed proportionally in knockout rounds.
- **48-team format for 2026**: the simulation correctly implements the expanded format (12 groups × 4 teams, best 8 third-place teams advance).
