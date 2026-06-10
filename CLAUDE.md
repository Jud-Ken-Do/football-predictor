# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A modular football match outcome prediction system targeting **FIFA World Cup 2026**. Predicts home win / draw / away win probabilities using a full SOTA scientific pipeline:

**Model stack (in order of application):**
1. **XGBoost** — 3-class classifier over ~120 training features (13 modules); near-duplicate features pruned at |Pearson r| > 0.95
2. **Temperature Scaling** — single scalar T calibrates XGBoost log-probabilities (Guo et al. ICML 2017)
3. **Bayesian Hierarchical Poisson** — MAP log-linear goal model (Baio & Blangiardo 2010); half-life derived from Kalman EM-tuned q; Dixon-Coles ρ correction for low-score cells
4. **Context-Adaptive Ensemble** — per-match α = sigmoid(w · [odds_available, kalman_uncertainty, log1p_h2h] + b), learned via NLL + L2; scalar fallback when context unavailable
5. **WC 2026 Post-Processing** — venue λ adjustment, sofifa+api_form quality nudge, player absence penalty (`models/wc_context.py`)

**Rating systems feeding XGBoost:**
- **Glicko-2** (Glickman 2001) — rating μ + deviation φ + volatility σ, Illinois algorithm; match-importance weighted
- **Extended Kalman Filter** (Koopman & Lit 2015) — time-varying attack/defence state; match-importance weighted R; EM-tuned process noise q; forward-only causal states (no RTS leakage into training features)

## Environment

- Python: 3.11 ARM64 via Homebrew (`/opt/homebrew/bin/python3.11`)
- VS Code terminal profile: `zsh-arm64` — always verify with `arch` before installing packages
- API key: `FOOTBALL_DATA_ORG_API_KEY` in `.env` (python-dotenv loaded at import)
- API key: `API_FOOTBALL_KEY` in `.env` (api-sports.io — note: NOT `API_SPORTS_KEY`)
- Data cache: `~/.cache/football_predictor/` — `international_results.csv` cached here
- Actual WC 2026 results: `data/wc2026_actual_results.json` (recorded via `scripts/update_wc2026.py`)

## Commands

```bash
# Install
python3.11 -m pip install -e ".[test]"

# Full dev install (includes pymc, shap, matplotlib)
python3.11 -m pip install -e ".[dev]"

# PRIMARY: Full pipeline (fetch cache → load → features → train → predict → simulate)
python3.11 scripts/pipeline.py
python3.11 scripts/pipeline.py --fetch          # re-fetch live data first
python3.11 scripts/pipeline.py --sims 100000
python3.11 scripts/pipeline.py --backtest       # also run WC 2014+2018+2022 evaluation
python3.11 scripts/pipeline.py --match "Brazil vs Morocco"
python3.11 scripts/pipeline.py --mcmc           # MCMC posterior (~5 min extra)
python3.11 scripts/pipeline.py --tune           # Optuna XGBoost hyperparameter search (60 trials)

# Direct prediction script
python3.11 scripts/predict_wc2026.py
python3.11 scripts/predict_wc2026.py --match "Spain vs England"

# Record an actual WC result (updates Kalman state for next run)
python3.11 scripts/update_wc2026.py --result "Mexico vs South Africa" --score "2-1" --group A

# Backtest on past World Cups
python3.11 scripts/backtest.py --years 2014 2018 2022
python3.11 scripts/backtest.py --years 2022 --no-shap
python3.11 scripts/backtest.py --continental    # also Copa 2021, Euro 2020, AFCON 2022, Asian Cup 2023

# Derive data-driven confederation strength offsets (prints recommendations; edit constants.py manually)
python3.11 scripts/calibrate_confederations.py
python3.11 scripts/calibrate_confederations.py --from-year 2000 --plot

# Refresh live data caches
python3.11 scripts/fetch_api_form.py
python3.11 scripts/fetch_transfermarkt.py --force
python3.11 scripts/fetch_wc2026_injuries.py

# Run tests
python3.11 -m pytest tests/ -v
```

## Architecture

### Primary entry point (`scripts/pipeline.py`)

Unified 7-step orchestrator with structured print output and per-step timing:
1. FETCH — api_form, transfermarkt, injuries (cache-aware, skipped if fresh)
2. LOAD — historical data from 2010
3. FEAT — build_feature_matrix() across 13 DEFAULT_FEATURE_MODULES + prune_correlated_features(threshold=0.95)
4. TRAIN — XGB (+ optional Optuna tuning) + temperature scaling + BayesPoisson (EM-derived half-life, DC ρ) + context-adaptive ensemble
5. PRED — 72 group stage match probabilities + WC 2026 post-processing (venue, quality, absence)
6. SIM — 50k Monte Carlo simulations; Kalman posterior uncertainty propagated to λ via lognormal resampling
7. OUTPUT — print tables + save to `output/wc2026_predictions_YYYY-MM-DD_HH-MM-SS.txt`

### Inference pipeline (`scripts/predict_wc2026.py`)

1. `train_model()` — loads historical data, appends any actual WC 2026 results, builds + prunes feature matrix, trains full model stack
2. `predict_group_stage()` — builds feature rows for all 72 fixtures, runs context-adaptive ensemble blend, extracts Kalman λ stds for MC resampling, applies WC 2026 post-processing
3. `merge_actual_results()` — locks played matches to known scores
4. Pre-computes 48×48 pair-probability cache for knockout rounds
5. Monte Carlo simulation (default 50k) with lognormal λ resampling from Kalman posterior
6. Saves full output to `output/wc2026_predictions_YYYY-MM-DD.txt`

### Feature plugin system (`football_predictor/features/`)

Every feature source extends `FeatureModule` (`base.py`) with:
- `fetch(competition, seasons) → DataFrame`
- `transform(match, data) → dict[str, float]`

Registered in `features/__init__.py::REGISTRY`. Two sets of active modules in `constants.py`:

**`DEFAULT_FEATURE_MODULES` (13 modules, ~120 features — used in XGBoost training):**

| Module | File | Features | Notes |
|---|---|---|---|
| `elo` | `elo.py` | 4 | Classic Elo; K=40 intl, K=60 WC; match-importance weighted K |
| `glicko2` | `glicko2.py` | 6 | Rating + RD + win prob; Illinois σ update; match-importance weighted |
| `kalman_strength` | `kalman_strength.py` | 13 | EKF forward-only (use_smoothed=False); match-importance weighted R; EM-tuned q |
| `form` | `form.py` | ~15 | Rolling pts/goals/GD over 5/10/20 matches |
| `sos` | `sos.py` | ~6 | Strength-of-schedule: opponent-quality-adjusted win rate |
| `h2h` | `h2h.py` | ~8 | H2H win rate, avg goals, last 10 meetings |
| `squad_strength` | `squad_strength.py` | 14 | 40-match rolling attack/defence from competitive history |
| `confederation` | `confederation.py` | ~6 | Data-derived offsets: CONMEBOL=65, UEFA=50, AFC=25, CAF=10, CONCACAF=−20, OFC=−40 |
| `tournament_stage` | `tournament_stage.py` | ~5 | Group/knockout, pressure multiplier, must-win flag |
| `rankings` | `rankings.py` | ~4 | FIFA world ranking points |
| `odds` | `odds.py` | ~10 | Bookmaker closing odds; `odds_available` flag drives context-adaptive ensemble |
| `xg_form` | `xg_form.py` | 13 | Rolling xG/xGA — Excel (UEFA/AFC/CONMEBOL) + FBref JSON fallback |
| `transfermarkt` | `transfermarkt.py` | 5 | Squad market values from Transfermarkt.com (June 2026) |

**`WC_CONTEXT_MODULES` (5 modules — post-processing only, not in XGBoost):**

| Module | File | Notes |
|---|---|---|
| `venue_wc2026` | `venue_wc2026.py` | Partial HA (MEX/USA/CAN), altitude (CDMX=2240m), travel burden |
| `sofifa_ratings` | `sofifa_ratings.py` | EA FC 26: overall, pace, shooting, passing + star_rating, top3_premium, squad_depth |
| `api_form` | `api_form.py` | Last-10-match form from API-Football (pre-cached) |
| `squad_wc2026` | `squad_wc2026.py` | Club tier, avg age, top-club ratio from FIFA squad list |
| `injury` | `injury.py` | Pre-match injuries/suspensions from API-Football cache |

### Kalman EKF (`features/kalman_strength.py`)

State per team: `x = [att, def]` in log-goals space.
- **Time update**: `P += q²·Δt·I`; q tuned via EM (Shumway-Stoffer M-step), stored in `_LAST_TUNED_Q`
- **EKF observation**: Jacobian `H = [λ, 0]` or `[0, λ]`; `S = HPH' + λ/match_weight`; Joseph-form covariance update
- **Training**: `use_smoothed=False` (default) — forward-only causal states, no RTS leakage
- **RTS smoother**: available via `use_smoothed=True` for diagnostics only
- Exposes `get_last_tuned_q()` classmethod so BayesPoisson can derive consistent half-life

### Bayesian Hierarchical Poisson (`models/bayesian_poisson.py`)

Model: `log(λ_h) = μ + att_h + def_a + home_adv * I(not_neutral)`

- `fit()` — MAP via L-BFGS-B; home prior `N(0.20, 0.15²)`; half-life = `half_life_from_q(q_tuned)` clamped [180, 730] days
- `fit_rho()` — estimates Dixon-Coles ρ from low-scoring residuals; applies `_dc_tau` correction to `{(0,0),(0,1),(1,0),(1,1)}` cells in `predict_proba`
- `fit_mcmc()` — full PyMC 5 posterior (requires `pip install pymc`); `predict_proba_mcmc()` averages over posterior samples

### Calibration & Ensemble (`models/calibration.py`, `models/ensemble.py`)

- `TemperatureScaling.fit()` — single scalar T on calibration NLL
- `EnsembleModel.fit(xgb, bp, y, context_X)` — if `context_X` provided, fits 3-weight logistic α with L2 reg; otherwise scalar α ∈ [0,1]; `context_X` must include `odds_available`, `kalman_home_att_std`, `kalman_home_def_std`, `kalman_away_att_std`, `kalman_away_def_std`, `h2h_n_matches`

### Feature preprocessing (`football_predictor/data/pipeline.py`)

- `build_feature_matrix(matches, modules, context)` — assembles feature rows from all active modules
- `prune_correlated_features(X, threshold=0.95)` — removes the lower-variance member of each pair with |r| > threshold; called immediately after build_feature_matrix in all training scripts

### Data sources (`football_predictor/data/sources/`)

| File | Source | What it provides |
|---|---|---|
| `international_results.py` | martj42/international_results (GitHub CSV) | 47,000+ international matches 1872–present. **Primary training source.** Auto-refreshes if cache > 48h old. |
| `football_data_org.py` | football-data.org REST API | European league results. Free tier: 2 seasons. |
| `football_data_co_uk.py` | football-data.co.uk WorldCup2026.xlsx | xG (339/889 qualifier matches), bookmaker odds (WC 2018/2022) |
| `api_football.py` | api-sports.io v3 | Injuries/suspensions pre-cached for WC 2026 fixtures |

**xG coverage:**
- Has xG: UEFA qualifiers, AFC qualifiers, CONMEBOL qualifiers, some CONCACAF
- Missing xG: CAF (all African teams), Mexico/USA/Canada/Curaçao — FBref Cloudflare-blocked, no free automated source

**Match type weights** (in `_match_weight`):
- Friendly: `0.3×` | Nations League/Gold Cup: `0.7×` | AFCON/Asian Cup: `0.92×` | Qualifiers/continental: `1.0×` | World Cup: `1.5×`

### WC 2026 data (`football_predictor/data/wc2026.py`)

12 groups × 4 teams = 48 teams. `GROUP_STAGE_SCHEDULE` has all 72 fixtures with `date`, `group`, `home_team`, `away_team`, `venue`, `neutral=True`. `normalise()` handles name variations. `TEAM_NAME_MAP` for edge cases.

### WC 2026 post-processing (`football_predictor/models/wc_context.py`)

Applied after ensemble, before output. Three stages:
1. `venue_adjust(lam_h, lam_a, ctx)` — partial HA + altitude multipliers on λ; recomputes Poisson probs
2. `quality_nudge(p_h, p_d, p_a, ctx)` — log-odds shift from sofifa overall diff + api_form pts diff
3. `absence_adjust(p_h, p_d, p_a, home, away)` — market-value-weighted log-odds shift from `data/wc2026_player_injuries.json`

### Backtest (`scripts/backtest.py`)

Trains on pre-tournament competitive data, evaluates on group stage. Reports metrics with bootstrap 95% CI. Outputs to `output/`:
- `backtest_YEAR_metrics.txt` — log-loss [CI], Brier [CI], accuracy vs uniform baseline
- `backtest_YEAR_calibration.png` — reliability diagram (3 panels)
- `backtest_YEAR_shap.png` + `.csv` — SHAP feature importance (top 30)

Supports: WC 2014/2018/2022 (`--years`) and continental tournaments (`--continental`).

### Post-match updating (`scripts/update_wc2026.py`)

Records actual results to `data/wc2026_actual_results.json`. Next run of `predict_wc2026.py` automatically:
1. Appends results to training context → Kalman EKF and BayesPoisson update
2. Locks played matches to actual scores in group stage simulation
3. Reruns Monte Carlo only over remaining fixtures

### Confederation calibration (`scripts/calibrate_confederations.py`)

Runs a bias-free Elo pass (no starting offsets) on competitive matches from 2000+, then:
1. Reports mean Elo per confederation (note: inflated for insular confederations — use H2H table instead)
2. Cross-confederation H2H win rates vs Elo expected → identifies miscalibration
3. Prints recommended `CONFEDERATION_STRENGTH` offsets for manual review

Current calibrated values (from H2H analysis, updated 2026-06-10):
`CONMEBOL=65, UEFA=50, AFC=25, CAF=10, CONCACAF=−20, OFC=−40`

## Key constraints

- Python 3.11 ARM64 only — never run `pip install` without verifying `arch`
- API keys only in `.env` — never hardcode, never commit
- API key in `.env` is `API_FOOTBALL_KEY` (not `API_SPORTS_KEY`)
- `data/wc2026_actual_results.json` may contain real match data — don't overwrite without confirmation
- BayesPoisson MAP always converges; `fit_mcmc()` can be slow (use `draws=500, tune=250` for testing)
- WC 2026 fixture rows always carry `neutral=True` explicitly — Elo `transform()` default of `False` is safe
- xG gap for CAF/CONCACAF is a known data limitation — no free automated source exists

## Key decisions

- **Replaced Dixon-Coles with BayesPoisson MAP**: DC multiplicative MLE with 200+ teams is ill-conditioned on sparse data and fails to converge. Log-linear BayesPoisson with Gaussian priors (L2 reg) always converges.
- **Temperature Scaling over isotonic**: Calibration set has ~200 WC matches; isotonic overfits. Single scalar T is theoretically justified (Guo 2017) and robust.
- **WC context modules separated from training**: `sofifa_ratings`, `squad_wc2026`, `api_form`, `venue_wc2026`, `injury` produce zeros for all historical rows — covariate shift. Applied as post-processing in `wc_context.py` instead.
- **Forward-only Kalman for training**: RTS smoother uses future data — valid for analysis but not for training features. `use_smoothed=False` is the default; XGBoost sees only causal states.
- **Context-adaptive ensemble**: Static α=0.60 replaced by per-match logistic regression on odds availability, Kalman uncertainty, and H2H depth. Scalar fallback retained.
- **Timescale consistency**: BayesPoisson half-life derived from Kalman EM-tuned q via `T = P0/(2q²)`, ensuring both models share the same empirical assumption about team strength drift.
- **Confederation offsets from H2H data**: Mean-Elo approach discarded (CONCACAF/CONMEBOL inflate from insular tournaments). Offsets set from cross-confederation win-rate calibration gaps.
- **Friendly weighting not exclusion**: Include at 0.3× — friendlies carry signal for Elo calibration and form for infrequently-playing teams.

## Git branches

- `main` — working system, competition submission ready
- `improvements` — active development branch for known issues

## What's working

- Full pipeline runs end-to-end (`python3.11 scripts/pipeline.py`)
- 13 training feature modules (~120 features after correlation pruning); 5 WC context modules as post-processing
- XGBoost + Temperature Scaling + BayesPoisson MAP (DC ρ) + Context-Adaptive Ensemble
- Kalman EKF with EM-tuned process noise q; match-importance weighted; forward-only causal states
- Glicko-2 with match-importance weighting (Illinois σ update)
- Monte Carlo tournament simulator with Kalman posterior λ resampling (50k sims, ~10s)
- Optional full MCMC posterior (`--mcmc` flag; non-centered parameterisation, 0 divergences)
- Optional Optuna XGBoost hyperparameter tuning (`--tune` flag, 60 TPE trials)
- `scripts/pipeline.py` — unified orchestrator with structured step-by-step output
- `scripts/predict_wc2026.py --match "X vs Y"` — single match prediction card
- `scripts/update_wc2026.py` — live result ingestion, Kalman EKF updates
- `scripts/backtest.py` — WC 2014/2018/2022 + continental evaluation with bootstrap CI + SHAP
- `scripts/calibrate_confederations.py` — data-driven confederation strength derivation
- `scripts/generate_submission_v2.py` — competition output.csv with third-place advancement

## Known limitations (not bugs, architectural constraints)

1. **xG gap for CAF/CONCACAF** — FBref Cloudflare-blocked; api-sports.io has no xG for these confederations. African and CONCACAF teams rely on form/Elo/Kalman without xG signal.
2. **sofifa ratings are static** — EA FC 26 snapshot; no historical versions available. Correct for WC 2026 prediction but would be leakage if mistakenly added to training features. Architecture prevents this (WC_CONTEXT_MODULES only).
3. **MCMC not default** — MAP is always used; `--mcmc` flag adds ~5 min for full posterior. Impact on point estimates is small; main benefit is uncertainty quantification.
4. **No live odds** — WC 2026 fixtures have `odds_available=0.0` pre-tournament. Ensemble context-adaptive α defaults toward scalar when odds signal is absent.
