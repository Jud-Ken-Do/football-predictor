# Football Predictor — System Documentation

## What it does

Predicts the outcome (home win / draw / away win) and most likely scoreline for international football matches, with a focus on FIFA World Cup 2026. Produces per-match probabilities, expected group standings, and full tournament win probabilities via Monte Carlo simulation.

---

## Model Stack

The system layers four models on top of each other:

| Step | Model | What it does |
|---|---|---|
| 1 | **XGBoost** | 3-class classifier over 218 features. Predicts P(home win), P(draw), P(away win). |
| 2 | **Temperature Scaling** | Calibrates XGBoost's overconfident probabilities using a single learned scalar T. |
| 3 | **Bayesian Hierarchical Poisson** | Goal model — estimates expected goals (λ) per team per match. Used for scoreline simulation. |
| 4 | **Ensemble** | Blends XGBoost (60%) + BayesPoisson (40%) via learned weight α. Final output. |

---

## Feature Modules (218 features across 16 modules)

Every feature source is a plug-in. Active modules:

| Module | Features | What it captures |
|---|---|---|
| `elo` | 4 | Classic Elo ratings. K=40 international, K=60 WC. No home advantage at neutral venues. |
| `glicko2` | 6 | Glicko-2 rating system — adds rating deviation φ (how uncertain we are about the rating) and volatility σ. |
| `kalman_strength` | 13 | Extended Kalman Filter — time-varying attack/defence strength per team. Uses RTS backward smoother on training data. Most mathematically sophisticated rating. |
| `form` | ~15 | Rolling points / goals / goal difference over last 5, 10, 20 matches. |
| `h2h` | ~8 | Head-to-head win rate, average goals, results from last 10 meetings. |
| `squad_strength` | 14 | Long-term attack/defence quality estimated from 40-match rolling competitive history. |
| `confederation` | ~6 | UEFA / CONMEBOL / CAF / AFC / CONCACAF relative strength encoding + matchup difference. |
| `tournament_stage` | ~5 | Group vs knockout stage, pressure multiplier, must-win flag. |
| `rankings` | ~4 | FIFA world ranking points (Dato-Futbol dataset, 1992–2024). |
| `squad_wc2026` | 24 | Club tier, average age, top-club ratio from official FIFA WC 2026 squad list. |
| `api_form` | 25 | Last-10-match form from API-Football (pre-cached for all WC 2026 teams). |
| `sofifa_ratings` | 33 | EA FC 26 squad quality: overall, pace, shooting, passing, defending, physicality per squad unit. |
| `venue_wc2026` | 8 | Partial home advantage for Mexico/USA/Canada, altitude penalty (Mexico City = 2,240m), travel burden. |
| `odds` | 8 | Bookmaker closing odds — implied probabilities, overround, favourite direction. Strongest single external signal. |
| `injury` | varies | Pre-match unavailable players (passed at prediction time). |
| `standings` | varies | Club/league standing context (used for club competitions). |

---

## Data Sources

| Source | What it provides | Size |
|---|---|---|
| `martj42/international_results` (GitHub CSV) | 47,000+ international matches 1872–present including friendlies, qualifiers, tournaments | Primary training set |
| `football-data.org` REST API | European league results (PL, Bundesliga, Serie A, La Liga, Ligue 1, CL) | Free tier: 2 seasons |
| `api-sports.io` | Recent match form, pre-match data | Cached for WC 2026 teams |
| `FIFA rankings CSV` (Dato-Futbol) | FIFA world ranking points 1992–2024 | 30+ years |
| `EA FC 26 / SoFIFA` | Squad attribute cards | WC 2026 squads |
| `football-data.co.uk` | Closing bookmaker odds (Pinnacle WC 2018, bet365/Betfair WC 2022) | 889+ matches |

**Match type weights** applied during training:

| Tournament type | Weight |
|---|---|
| Friendly | 0.3× |
| Nations League / minor | 0.7× |
| Qualifiers / continental | 1.0× |
| FIFA World Cup | 1.5× |

---

## Inputs

### Training
- Historical international results (auto-downloaded and cached at `~/.cache/football_predictor/`)
- Any recorded actual WC 2026 results (`data/wc2026_actual_results.json`)

### Prediction
- Team names (home + away)
- Competition context (`WC` for World Cup)
- Optional: `home_unavailable` / `away_unavailable` player lists (for injury module)

---

## Outputs

### `scripts/predict_wc2026.py`
- Per-match win/draw/loss probabilities for all 72 group stage fixtures
- Expected group standings (by expected points)
- Tournament progression probabilities (R32 → R16 → QF → SF → Final → Winner) via 50,000 Monte Carlo simulations
- Saved to `output/wc2026_predictions_YYYY-MM-DD_HH-MM-SS.txt`

### `scripts/generate_submission.py`
- `output/output.csv` — one predicted scoreline per match, format: `match_id, group, team1, team2, score1, score2`
- Scores are optimised to **maximise expected competition points** (not just most likely outcome)

### `scripts/backtest.py`
- RPS, log-loss, Brier score, accuracy vs uniform baseline
- Calibration reliability diagram (`output/backtest_YEAR_calibration.png`)
- SHAP feature importance top-30 (`output/backtest_YEAR_shap.png` + `.csv`)

---

## Commands

```bash
# Install
python3.11 -m pip install -e ".[test]"

# Run full WC 2026 prediction
python3.11 scripts/predict_wc2026.py
python3.11 scripts/predict_wc2026.py --sims 100000   # more MC simulations
python3.11 scripts/predict_wc2026.py --mcmc           # full Bayesian inference (~5 min)

# Record an actual result (updates model state for next run)
python3.11 scripts/update_wc2026.py --result "France vs Norway" --score "2-1" --group I

# Generate competition submission
python3.11 scripts/generate_submission.py

# Backtest on past World Cups
python3.11 scripts/backtest.py --years 2018 2022
```

---

## Live result integration

When a WC 2026 match is played, record it:

```bash
python3.11 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A
```

The next prediction run will automatically:
1. Append the result to the training context → Kalman EKF and BayesPoisson update their state
2. Lock the played match to the actual score in group stage simulation
3. Re-run Monte Carlo only over remaining unplayed fixtures

---

## Configuration

All hyperparameters live in `football_predictor/constants.py`:

| Key constant | Value | What it controls |
|---|---|---|
| `DEFAULT_FEATURE_MODULES` | list of 16 modules | Which features are active |
| `ELO_K_FACTOR_INTERNATIONAL` | 40 | Elo K-factor for non-WC matches |
| `ELO_K_FACTOR_WC` | 60 | Elo K-factor for WC matches |
| `ELO_NEUTRAL_ADVANTAGE` | 0 | No home advantage at neutral WC venues |
| `INTERNATIONAL_FORM_WINDOW_SIZES` | [5, 10, 20] | Rolling form windows (wider than club football) |
| `CONFEDERATION_STRENGTH` | dict | Relative offsets by confederation |

To disable a feature module: comment it out in `DEFAULT_FEATURE_MODULES`. To add one: create a class extending `FeatureModule` and add one line to the registry in `features/__init__.py`.
