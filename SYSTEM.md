# Football Predictor — System Documentation

## What it does

Predicts the outcome (home win / draw / away win) and most likely scoreline for international football matches, with a focus on FIFA World Cup 2026. Produces per-match probabilities, expected group standings, and full tournament win probabilities via Monte Carlo simulation.

---

## Model Stack

| Step | Model | What it does |
|---|---|---|
| 1 | **XGBoost** | 3-class classifier over ~120 training features. Predicts P(home win), P(draw), P(away win). |
| 2 | **Temperature Scaling** | Calibrates XGBoost's overconfident log-probabilities using a single learned scalar T (Guo et al. ICML 2017). |
| 3 | **Bayesian Hierarchical Poisson** | Log-linear goal model estimating λ_h, λ_a. Half-life derived from Kalman EM-tuned process noise q. Includes Dixon-Coles ρ correction for low-score joint probabilities. |
| 4 | **Context-Adaptive Ensemble** | Per-match α = sigmoid(w · [odds_available, kalman_uncertainty, log1p_h2h] + b), learned via NLL minimisation with L2 regularisation. Blends XGBoost and BayesPoisson. Falls back to scalar α when context unavailable. |
| 5 | **WC 2026 Post-Processing** | Venue λ adjustment (partial HA for MEX/USA/CAN, altitude), quality nudge (sofifa + api_form), player absence penalty (market-value-weighted). Applied in `wc_context.py`. |

---

## Feature Modules

### Training features (13 modules, ~120 features — `DEFAULT_FEATURE_MODULES`)

These are used for XGBoost training and prediction. All produce non-zero values for historical international matches.

| Module | Features | What it captures |
|---|---|---|
| `elo` | 4 | Classic Elo ratings. K=40 intl, K=60 WC. Match-importance weighted. No HA at neutral venues. |
| `glicko2` | 6 | Glicko-2: rating + deviation φ + volatility σ. Match-importance weighted. Illinois σ update. |
| `kalman_strength` | 13 | EKF time-varying attack/defence in log-goals space. Match-importance weighted R. EM-tuned process noise q. Forward-only states (no RTS leakage into training). |
| `form` | ~15 | Rolling pts/goals/GD over last 5, 10, 20 matches. |
| `sos` | ~6 | Strength-of-schedule: opponent-quality-adjusted win rate. |
| `h2h` | ~8 | Head-to-head win rate, average goals, results from last 10 meetings. |
| `squad_strength` | 14 | Long-term attack/defence quality from 40-match competitive history. |
| `confederation` | ~6 | Data-derived confederation strength offsets (CONMEBOL=65 > UEFA=50 > AFC=25 > CAF=10 > CONCACAF=−20 > OFC=−40). |
| `tournament_stage` | ~5 | Group vs knockout, pressure multiplier, must-win flag. |
| `rankings` | ~4 | FIFA world ranking points (Dato-Futbol, 1992–2024). |
| `odds` | ~10 | Bookmaker closing odds — implied probabilities, overround, favourite direction, `odds_available` flag. |
| `xg_form` | 13 | Rolling xG/xGA from WorldCup2026.xlsx (UEFA/AFC/CONMEBOL qualifiers) + FBref JSON fallback. |
| `transfermarkt` | 5 | Squad market values from Transfermarkt.com (June 2026). |

Near-duplicate features (|Pearson r| > 0.95) are automatically pruned by `prune_correlated_features()` before training, keeping the higher-variance member of each pair.

### WC 2026 post-processing context (5 modules — `WC_CONTEXT_MODULES`)

Applied as adjustments after the XGBoost + BayesPoisson ensemble, not as training features. Produces meaningful values only for WC 2026 fixtures.

| Module | What it adds |
|---|---|
| `venue_wc2026` | Partial home advantage for MEX/USA/CAN (+6% λ), altitude adjustment (CDMX=2240m), travel burden |
| `sofifa_ratings` | EA FC 26 squad quality: overall, pace, shooting, passing, defending, physic + star_rating, top3_premium, squad_depth |
| `api_form` | Last-10-match form from API-Football (pre-cached for all 48 WC 2026 teams) |
| `squad_wc2026` | Club tier, average age, top-club ratio from official FIFA WC 2026 squad list |
| `injury` | Pre-match injuries/suspensions from API-Football cache + market-value-weighted player absence penalty |

---

## Data Sources

| Source | What it provides |
|---|---|
| `martj42/international_results` (GitHub CSV) | 47,000+ international matches 1872–present. Primary training source. |
| `api-sports.io` | Last-10-match form and injury data, pre-cached for WC 2026 teams |
| `FIFA rankings CSV` (Dato-Futbol) | FIFA world ranking points 1992–2024 |
| `EA FC 26 / SoFIFA` | Squad attribute cards for all 48 WC 2026 teams |
| `football-data.co.uk WorldCup2026.xlsx` | xG for UEFA/AFC/CONMEBOL qualifiers; bookmaker odds for WC 2018/2022 |
| `Transfermarkt.com` | Squad market values scraped June 2026 |

**Match type weights** applied during training:

| Tournament type | Weight |
|---|---|
| Friendly | 0.3× |
| Nations League / minor | 0.7× |
| AFCON / Asian Cup | 0.92× |
| Qualifiers / continental | 1.0× |
| FIFA World Cup | 1.5× |

---

## Outputs

### `scripts/predict_wc2026.py`
- Per-match win/draw/loss probabilities for all 72 group stage fixtures
- Expected group standings (by expected points)
- Tournament progression probabilities (R32 → R16 → QF → SF → Final → Winner) via 50,000 Monte Carlo simulations
- Kalman posterior uncertainty propagated to λ via lognormal resampling per simulation draw
- Saved to `output/wc2026_predictions_YYYY-MM-DD_HH-MM-SS.txt`

### `scripts/generate_submission_v2.py`
- `output/output_v2.csv` — one predicted scoreline per match
- Scores optimised to maximise expected competition points considering third-place advancement across all 12 groups jointly

### `scripts/backtest.py`
- Log-loss, Brier score, accuracy vs uniform baseline with bootstrap 95% CI
- Calibration reliability diagram (`output/backtest_YEAR_calibration.png`)
- SHAP feature importance top-30 (`output/backtest_YEAR_shap.png` + `.csv`)
- Supports WC 2014/2018/2022 and continental tournaments (Copa 2021, Euro 2020, AFCON 2022, Asian Cup 2023)

---

## Commands

```bash
# Install
python3.11 -m pip install -e ".[test]"

# PRIMARY — Full pipeline (fetch → load → features → train → predict → simulate)
python3.11 scripts/pipeline.py
python3.11 scripts/pipeline.py --fetch          # re-fetch live data first
python3.11 scripts/pipeline.py --sims 100000
python3.11 scripts/pipeline.py --backtest       # also run WC 2014+2018+2022 backtest
python3.11 scripts/pipeline.py --match "Brazil vs Morocco"
python3.11 scripts/pipeline.py --mcmc           # full MCMC posterior (~5 min extra)
python3.11 scripts/pipeline.py --tune           # Optuna hyperparameter search (60 trials)

# Direct prediction
python3.11 scripts/predict_wc2026.py
python3.11 scripts/predict_wc2026.py --match "Spain vs England"

# Record an actual WC result (updates Kalman state for next run)
python3.11 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A

# Backtest on past World Cups
python3.11 scripts/backtest.py --years 2014 2018 2022
python3.11 scripts/backtest.py --continental    # also Copa/Euro/AFCON/Asian Cup

# Derive data-driven confederation strength offsets
python3.11 scripts/calibrate_confederations.py
python3.11 scripts/calibrate_confederations.py --from-year 2000 --plot

# Generate competition submission
python3.11 scripts/generate_submission_v2.py

# Refresh live data caches
python3.11 scripts/fetch_api_form.py
python3.11 scripts/fetch_transfermarkt.py --force
python3.11 scripts/fetch_wc2026_injuries.py

# Run tests
python3.11 -m pytest tests/ -v
```

---

## Live Result Integration

When a WC 2026 match is played:

```bash
python3.11 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A
```

The next prediction run automatically:
1. Appends the result to the training context → Kalman EKF and BayesPoisson update
2. Locks played matches to actual scores in group stage simulation
3. Re-runs Monte Carlo only over remaining unplayed fixtures

---

## Configuration

All hyperparameters live in `football_predictor/constants.py`:

| Constant | Value | What it controls |
|---|---|---|
| `DEFAULT_FEATURE_MODULES` | 13 modules | Features active in XGBoost training |
| `WC_CONTEXT_MODULES` | 5 modules | Post-processing context for WC 2026 only |
| `ELO_K_FACTOR_INTERNATIONAL` | 40 | Elo K-factor (scaled by match_weight) |
| `ELO_K_FACTOR_WC` | 60 | Elo K-factor for WC matches |
| `ELO_NEUTRAL_ADVANTAGE` | 0 | No home advantage at neutral WC venues |
| `CONFEDERATION_STRENGTH` | dict | Data-derived relative strength offsets |
| `INTERNATIONAL_FORM_WINDOW_SIZES` | [5, 10, 20] | Rolling form windows |
