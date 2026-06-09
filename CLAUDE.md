# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A modular football match outcome prediction system targeting **FIFA World Cup 2026**. Predicts home win / draw / away win probabilities using a full SOTA scientific pipeline:

**Model stack (in order of application):**
1. **XGBoost** — 3-class classifier over ~190 features from 12 feature modules
2. **Temperature Scaling** — single scalar T calibrates XGBoost log-probabilities (Guo et al. ICML 2017)
3. **Bayesian Hierarchical Poisson** — MAP-estimated log-linear goal model (Baio & Blangiardo 2010), convergence-stable replacement for Dixon-Coles
4. **Ensemble** — learned scalar α blends XGBoost and BayesPoisson on calibration set via NLL minimisation

**Rating systems feeding XGBoost:**
- **Glicko-2** (Glickman 2001) — rating μ + deviation φ + volatility σ, Illinois algorithm
- **Extended Kalman Filter + RTS Smoother** (Koopman & Lit 2015) — time-varying attack/defence state per team; RTS backward pass for training data, causal forward pass for prediction

## Environment

- Python: 3.11 ARM64 via Homebrew (`/opt/homebrew/bin/python3.11`)
- VS Code terminal profile: `zsh-arm64` — always verify with `arch` before installing packages
- API key: `FOOTBALL_DATA_ORG_API_KEY` in `.env` (python-dotenv loaded at import)
- API key: `API_SPORTS_KEY` in `.env` (api-sports.io for recent form data)
- Data cache: `~/.cache/football_predictor/` — `international_results.csv` cached here
- Actual WC 2026 results: `data/wc2026_actual_results.json` (recorded via `scripts/update_wc2026.py`)

## Commands

```bash
# Install
python3.11 -m pip install -e ".[test]"

# Full dev install (includes pymc, shap, matplotlib)
python3.11 -m pip install -e ".[dev]"

# Run WC 2026 predictions
python3.11 scripts/predict_wc2026.py
python3.11 scripts/predict_wc2026.py --sims 100000

# Record an actual WC result (updates Kalman state for next run)
python3.11 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A
python3.11 scripts/update_wc2026.py list    # show recorded results
python3.11 scripts/update_wc2026.py refresh # re-run predictions

# Backtest on past World Cups (WC 2018 + 2022)
python3.11 scripts/backtest.py --years 2018 2022
python3.11 scripts/backtest.py --years 2022 --no-shap

# Run tests
python3.11 -m pytest tests/ -v
```

## Architecture

### Inference pipeline (`scripts/predict_wc2026.py`)

1. `train_model()` — loads historical data, appends any actual WC 2026 results, builds feature matrix, trains XGB + temperature scaling + BayesPoisson + ensemble
2. `predict_group_stage()` — builds feature rows for all 72 fixtures (with venue data), runs ensemble blend, stores BayesPoisson lambdas for goal simulation
3. `merge_actual_results()` — locks played matches to known scores; probabilistic simulation only for remaining fixtures
4. Pre-computes 48×48 pair-probability cache for knockout rounds
5. Monte Carlo simulation (default 50k) for tournament progression probabilities
6. Saves full output to `output/wc2026_predictions_YYYY-MM-DD.txt`

### Feature plugin system (`football_predictor/features/`)

Every feature source extends `FeatureModule` (`base.py`) with:
- `fetch(competition, seasons) → DataFrame`
- `transform(match, data) → dict[str, float]`

Registered in `features/__init__.py::REGISTRY`. Active modules in `constants.DEFAULT_FEATURE_MODULES`.

**Active modules (~190 features total):**

| Module | File | Features | Notes |
|---|---|---|---|
| `elo` | `elo.py` | 4 | Classic Elo; K=40 intl, K=60 WC; neutral=no HA |
| `glicko2` | `glicko2.py` | 6 | Rating + RD + win prob; Illinois σ update |
| `kalman_strength` | `kalman_strength.py` | 13 | EKF forward + RTS smoother; Joseph-form covariance |
| `form` | `form.py` | ~15 | Rolling pts/goals/GD over 5/10/20 matches |
| `h2h` | `h2h.py` | ~8 | H2H win rate, avg goals, last 10 meetings |
| `squad_strength` | `squad_strength.py` | 14 | 40-match rolling attack/defence from competitive history |
| `confederation` | `confederation.py` | ~6 | UEFA/CONMEBOL/CAF/AFC/CONCACAF encoding |
| `tournament_stage` | `tournament_stage.py` | ~5 | Group/knockout, pressure multiplier, must-win flag |
| `rankings` | `rankings.py` | ~4 | FIFA world ranking points |
| `squad_wc2026` | `squad_wc2026.py` | 24 | Club tier, avg age, top-club ratio from FIFA squad list |
| `api_form` | `api_form.py` | 25 | Last-10-match form from API-Football (pre-cached) |
| `sofifa_ratings` | `sofifa_ratings.py` | 33 | EA FC 26: overall, pace, shooting, passing, etc. |
| `venue_wc2026` | `venue_wc2026.py` | 8 | Partial home adv (MEX/USA/CAN), altitude (CDMX=2240m), travel burden |

### Kalman EKF + RTS Smoother (`features/kalman_strength.py`)

State per team: `x = [att, def]` in log-goals space.
- **Time update**: `P += q²·Δt·I` between matches (q=0.15/year)
- **EKF observation**: Jacobian `H = [λ, 0]` or `[0, λ]`; innovation covariance `S = HPH' + λ` (Poisson variance = mean); Joseph-form covariance update for numerical stability
- **RTS backward pass**: `G = P_upd[t] · inv(P_pred[t+1])`; smoothed states used for training features, forward states used at prediction time
- Snapshot structure: `_smoothed_snapshots[date][team]` for historical, `_snapshots[date][team]` for future

### Bayesian Hierarchical Poisson (`models/bayesian_poisson.py`)

Model: `log(λ_h) = μ + att_h + def_a + home_adv * I(not_neutral)`

Two fit modes:
- `fit()` — MAP via L-BFGS-B, Gaussian priors as L2 regularisation, corner constraint, 5000 iter max, always converges
- `fit_mcmc()` — full PyMC 5 posterior (requires `pip install pymc`); hierarchical σ_att, σ_def priors; weighted Potential for time-decay; `predict_proba_mcmc()` averages over posterior samples

### Calibration & Ensemble (`models/calibration.py`, `models/ensemble.py`)

- `TemperatureScaling.fit()` — single scalar T minimised via bounded scalar optimisation on calibration NLL; more robust than isotonic for small calibration sets
- `EnsembleModel.fit()` — scalar α ∈ [0,1] blending `P = α·P_xgb + (1-α)·P_bp` via NLL minimisation; typically learns α≈0.60 (XGB-dominant)

### Data sources (`football_predictor/data/sources/`)

| File | Source | What it provides |
|---|---|---|
| `international_results.py` | martj42/international_results (GitHub CSV) | 47,000+ international matches 1872–present. **Primary training source.** |
| `football_data_org.py` | football-data.org REST API | European league results. Free tier: 2 seasons. |

**Match type weights** (in `_match_weight`):
- Friendly: `0.3×` | Nations League / minor: `0.7×` | Qualifiers / continental: `1.0×` | World Cup: `1.5×`

### WC 2026 data (`football_predictor/data/wc2026.py`)

12 groups × 4 teams = 48 teams. `GROUP_STAGE_SCHEDULE` has all 72 fixtures with `date`, `group`, `home_team`, `away_team`, `venue`. `normalise()` handles name variations between draw and dataset. `TEAM_NAME_MAP` for edge cases.

### Backtest (`scripts/backtest.py`)

Trains on pre-WC competitive data, evaluates on WC group stage matches. Reports log-loss, Brier score, accuracy, ECE vs uniform baseline. Outputs to `output/`:
- `backtest_YEAR_metrics.txt` — numeric summary
- `backtest_YEAR_calibration.png` — reliability diagram (3 panels)
- `backtest_YEAR_shap.png` + `.csv` — SHAP feature importance (top 30)

### Post-match updating (`scripts/update_wc2026.py`)

Records actual results to `data/wc2026_actual_results.json`. Next run of `predict_wc2026.py` automatically:
1. Appends results to training context → Kalman EKF and BayesPoisson update
2. Locks played matches to actual scores in group stage simulation
3. Reruns Monte Carlo only over remaining fixtures

## Key constraints

- Python 3.11 ARM64 only — never run `pip install` without verifying `arch`
- API keys only in `.env` — never hardcode, never commit
- `data/wc2026_actual_results.json` may contain real match data — don't overwrite without confirmation
- BayesPoisson MAP always converges; `fit_mcmc()` can be slow (use `draws=500, tune=250` for testing)
- The `venue_wc2026` module returns zeros for historical training data (no venue column) — this is correct; it only contributes signal for WC 2026 fixtures where venue is known

## Key decisions

- **Replaced Dixon-Coles with BayesPoisson MAP**: DC multiplicative MLE with 200+ teams is ill-conditioned on sparse data and fails to converge. Log-linear BayesPoisson with Gaussian priors (L2 reg) always converges.
- **Temperature Scaling over isotonic**: Calibration set has ~200 WC matches; isotonic overfits. Single scalar T is theoretically justified (Guo 2017) and robust.
- **XGBoost-directed simulation**: Group stage uses XGBoost outcome probs for win/draw/loss direction, BayesPoisson lambdas only for goal magnitude. Avoids DC non-convergence corrupting group standings.
- **RTS smoother for training only**: Using future data to improve past strength estimates is only valid for training (non-causal). Prediction always uses forward-only EKF.
- **Friendly weighting not exclusion**: Include at 0.3× — friendlies carry signal for Elo calibration and form for infrequently-playing teams.
- **Venue effects for WC 2026 only**: `venue_wc2026` silently returns zeros for historical rows (no venue column), contributing only when predicting WC 2026 fixtures. No code path changes needed.

## Git branches

- `main` — working system, competition submission ready
- `improvements` — active development branch for known issues

## What's working

- Full pipeline runs end-to-end in ~60s
- 17 active feature modules, ~218 features
- XGBoost + Temperature Scaling + BayesPoisson + Ensemble stack
- Monte Carlo tournament simulator (50k sims default)
- Group-level score optimiser for competition (maximises expected points incl. advancement bonus)
- `scripts/update_wc2026.py` — live result ingestion, Kalman EKF updates
- `scripts/backtest.py` — WC 2022 evaluation with SHAP + calibration plots
- `scripts/generate_submission.py` — competition output.csv
- `notebooks/tutorial.ipynb` — step-by-step walkthrough

## Known issues (see KNOWN_ISSUES.md for full detail)

1. **WC-specific features are zero in training** — sofifa, squad_wc2026, api_form, venue_wc2026 only have values for WC 2026 targets, never during training. XGBoost can't learn from them.
2. **Kalman RTS smoother leaks future data into training features** — causal states should be used for training
3. **Third-place advancement not optimised in submission** — +3 pts bonus for 8 best third-place teams ignored
4. **Validation too thin** — only WC 2022 (64 matches); need rolling backtest across WC 2018 + 2022
5. **Static ensemble blend** — single α=0.60 for all matches; should adapt to match context
