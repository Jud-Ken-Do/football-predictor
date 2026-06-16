# Commands & Paths — Field Guide

Every runnable command in this project, categorised, with what contains what,
what each reads/writes, and copy-paste recipes. **Always use `python3.11`** —
bare `python3` is the broken Intel 3.9 build on this Mac (see "Golden rules").

---

## 0. TL;DR — the 5 you actually need

```bash
python3.11 scripts/pipeline.py                       # ① full run: data → train → predict → simulate → output
python3.11 scripts/update_wc2026.py --result "Sweden vs Tunisia" --score "5-1"   # ② record a played match
python3.11 scripts/generate_submission.py            # ③ rebuild the competition output.csv
python3.11 scripts/backtest.py --years 2014 2018 2022 --continental   # ④ honest evaluation
streamlit run app/streamlit_app.py                   # ⑤ the dashboard
```

**Typical loop after a match is played:** ② record → ① pipeline (or ③ for the submission).

---

## Golden rules

- **`python3.11` always.** `python3` = Python 3.9 x86_64 → `incompatible architecture` crash.
- **Two different "outputs"** (a recurring confusion):
  - `output.csv` = **optimised submission** (← `generate_submission.py`)
  - `output_raw.csv` = **raw model prediction** (← `pipeline.py`)
- **One full pipeline, many partial ones.** `pipeline.py` is the orchestrator; almost everything else is one slice of it run standalone.

---

## 1. 🚀 The FULL pipeline — `pipeline.py`

The orchestrator. Runs all 7 steps in order and is the single command to "do everything".

```bash
python3.11 scripts/pipeline.py
```

**7 steps (what it _contains_):** `FETCH → LOAD → FEAT → TRAIN → PRED → SIM → OUTPUT`
It **calls the fetcher scripts** (step 1, via subprocess) and **imports prediction
functions** from `predict_wc2026.py` (train/predict/simulate). Writes
`output_raw.csv` + a timestamped `output/wc2026_predictions_*.txt`.

| Flag | Effect |
|---|---|
| `--fetch` / `--no-fetch` | force / skip the live-data fetch step (default: cache-aware) |
| `--sims N` | Monte Carlo simulations (default 50 000) |
| `--match "A vs B"` | single-match card instead of the whole tournament |
| `--backtest` | also run the WC 2014/18/22 backtest at the end |
| `--tune` | force re-tune XGBoost hyperparameters (Optuna) |
| `--retune` | force re-tune `friendly_weight` (grid search) |
| `--tune-trials N` | Optuna trial count (default 60) |
| `--mcmc` `--mcmc-draws` `--mcmc-tune` | full PyMC posterior (~5 min extra) |
| `--from-year Y` | training data start year (default 2010) |
| `--friendly-weight W` | weight on friendlies |
| `--seed N` | MC seed (default 42) |

> ⚠️ `--tune`/`--retune` are only needed after the **feature set** changes. Normal runs reuse the cached params.

---

## 2. 🔮 Prediction & submission (partial pipelines)

All of these internally call `train_model()` + `predict_group_stage()` from `predict_wc2026.py`.

| Command | What it does | Writes |
|---|---|---|
| `python3.11 scripts/predict_wc2026.py` | Group-stage predictions + 50k MC tournament sim (no fetch orchestration) | `output/wc2026_predictions_*.txt` |
| `python3.11 scripts/predict_wc2026.py --match "Spain vs England"` | Single-match prediction card | — |
| `python3.11 scripts/generate_submission.py` | **The competition submission** — EV-optimised scorelines + cross-group 3rd-place advancement | **`output/output.csv`** |
| `python3.11 scripts/generate_league_submission.py` | Variant submission — maximises expected match points only | `output/league_submission.csv` |
| `python3.11 scripts/raw_scorelines.py` | Raw most-likely scoreline per fixture (no optimisation) | `output/raw_scorelines.csv` |

- `predict_wc2026.py` opts: `--sims --match --mcmc --mcmc-draws --mcmc-tune --seed --quiet`
- `generate_submission.py` opts: `--sims --mcmc --mcmc-draws --mcmc-tune --no-two-pass`
- `generate_league_submission.py` opts: `--name`

---

## 3. 📝 Live result updates — `update_wc2026.py`

Records real WC 2026 results so the model relearns. Writes `data/wc2026_actual_results.json`.

```bash
python3.11 scripts/update_wc2026.py --result "Sweden vs Tunisia" --score "5-1"   # record (shorthand)
python3.11 scripts/update_wc2026.py list                                         # show recorded
python3.11 scripts/update_wc2026.py refresh                                       # record-then-rerun predict
python3.11 scripts/update_wc2026.py clear                                         # wipe recorded results
```

Subcommands: `record` (`--result --score --group --date`) · `list` · `refresh` · `clear`.
The top-level `--result/--score` is a shorthand for `record`.

---

## 4. 📥 Data fetchers (step 1 of the pipeline, run individually)

Each populates one cache file. `pipeline.py --fetch` runs the first three for you.

| Command | Source | Writes |
|---|---|---|
| `python3.11 scripts/fetch_api_form.py` | API-Football | `data/api_form_cache.json` |
| `python3.11 scripts/fetch_transfermarkt.py --force` | Transfermarkt | `data/transfermarkt_wc2026.json` |
| `python3.11 scripts/fetch_wc2026_injuries.py --force` | API-Football | `data/wc2026_injuries_cache.json` |
| `python3.11 scripts/fetch_wc2026_odds.py --force` | API-Football | `data/wc2026_odds_cache.json` |
| `python3.11 scripts/fetch_xg_fbref.py` | FBref (Cloudflare-blocked) | `data/xg_fbref.json` |
| `python3.11 scripts/fetch_injury_history.py` | Transfermarkt | injury history (`--force --team --summary --min-value`) |

All fetchers honour `--force` (ignore cache). `fetch_xg_fbref` adds `--dry-run`.

---

## 5. 📊 Evaluation & experiments

| Command | Purpose | Key opts |
|---|---|---|
| `python3.11 scripts/backtest.py --years 2014 2018 2022` | Backtest on past WCs (log-loss, Brier, SHAP) | `--continental --no-shap --ablation --friendly-weight --tune-friendly-weight` |
| `… --continental` | also Copa 2021 / Euro 2020 / AFCON 2022 / Asian Cup 2023 (**266-match wide eval**) | |
| `… --ablation` | xG-observation ablation (baseline vs full) | |
| `python3.11 scripts/odds_blend_backtest.py` | W17 — market 1X2 blend vs model | `--years --friendly-weight` |
| `python3.11 scripts/roadmap_ablation.py` | R9 (form/SoS shrinkage) + R10 (Kalman μ/HA) ablation | `--continental --fw` |
| `python3.11 scripts/benchmark_kalman.py` | compare Kalman feature modes | `--years` |
| `python3.11 scripts/blend_sweep.py --match "A vs B"` | sweep `_MARKET_BLEND` for one match | `--match --blends --sims --runs` |
| `python3.11 scripts/evaluate_wc2022.py` | quick WC 2022 eval | — |

---

## 6. 🛠 Calibration & data-building (run rarely)

| Command | Purpose |
|---|---|
| `python3.11 scripts/calibrate_confederations.py` | Derive `CONFEDERATION_STRENGTH` offsets (prints; edit `constants.py` by hand). `--from-year --plot` |
| `python3.11 scripts/build_xg_proxy.py` | Build proxy xG from shots for matches missing StatsBomb xG |

---

## 7. 🖥 Frontend & tests

```bash
streamlit run app/streamlit_app.py     # the dashboard (7 tabs: Overview/Bracket/Groups/Match Lab/Model-vs-Market/Submission/Diagnostics)
python3.11 -m pytest tests/ -v         # all tests
```
`app/streamlit_app.py` reads `output.csv` + `output_raw.csv` and runs live predictions.
Tests: `test_features`, `test_models`, `test_bracket`, `test_validation`, `test_xg_integration`.

There is also a packaged entry point from `pyproject.toml`: `football-predict` → `football_predictor/predict.py:main` (library CLI; the `scripts/` above are the day-to-day tools).

---

## 8. 🌳 What contains what (relationship map)

```
pipeline.py  ── THE FULL PIPELINE (orchestrator) ───────────────────────────
  │
  ├─ step FETCH ─► subprocess ─► fetch_api_form.py / fetch_transfermarkt.py / fetch_wc2026_injuries.py
  ├─ step LOAD  ─► fetch_training_data + append_actual_results + attach_calibrated_xg
  ├─ step FEAT  ─► build_feature_matrix (12 DEFAULT_FEATURE_MODULES) + prune
  ├─ step TRAIN ─► XGBoost (+Optuna) + Temperature + BayesPoisson + Ensemble
  ├─ step PRED  ─► predict_group_stage  ← shared with predict_wc2026.py & generate_submission.py
  ├─ step SIM   ─► simulate_tournament (imported from predict_wc2026.py)
  └─ step OUTPUT─► output_raw.csv + wc2026_predictions_*.txt

predict_wc2026.py   = PRED + SIM standalone (no fetch step)        → wc2026_predictions_*.txt
generate_submission.py = PRED + EV-optimise scorelines            → output.csv   (THE submission)
generate_league_submission.py = PRED + expected-points optimise   → league_submission.csv
update_wc2026.py    = writes actual results; `refresh` re-runs predict_wc2026.py
backtest.py         = TRAIN + evaluate on past tournaments (its own training window)

  All four prediction scripts share:  train_model() + predict_group_stage()  (in predict_wc2026.py)
```

**Full vs partial:** `pipeline.py` is the only *full* one. `predict_wc2026.py`,
`generate_submission.py`, and the fetchers are each one *slice* run on its own.

---

## 9. 📁 Files & paths reference

**Inputs / caches** (`data/`)
| File | What | Written by |
|---|---|---|
| `wc2026_actual_results.json` | recorded played-match scores | `update_wc2026.py` |
| `api_form_cache.json` | last-20 form per team | `fetch_api_form.py` |
| `wc2026_odds_cache.json` | live 1X2 / AH / O-U odds | `fetch_wc2026_odds.py` |
| `wc2026_injuries_cache.json`, `wc2026_player_injuries.json` | injuries/suspensions | `fetch_wc2026_injuries.py` |
| `transfermarkt_wc2026.json` | squad market values | `fetch_transfermarkt.py` |
| `xg_fbref.json` | FBref xG cache (empty — Cloudflare) | `fetch_xg_fbref.py` |
| `raw/WorldCup2026.xlsx` | qualifier xG + closing odds | (manual download) |
| `tuned_params.json`, `xgb_tuned_params.json` | cached hyperparameters | `--tune` / `--retune` |
| `~/.cache/football_predictor/` | `international_results.csv` (47k matches), `statsbomb/` xG | auto |

**Outputs** (`output/`)
| File | What | Written by |
|---|---|---|
| `output.csv` | **competition submission** (optimised scorelines) | `generate_submission.py` |
| `output_raw.csv` | raw model prediction (+ `mdl_p_*` un-blended cols) | `pipeline.py` |
| `league_submission.csv` | expected-points submission | `generate_league_submission.py` |
| `raw_scorelines.csv` | raw modal scorelines | `raw_scorelines.py` |
| `wc2026_predictions_*.txt` | full timestamped forecast | `pipeline.py` / `predict_wc2026.py` |
| `backtest_*_metrics.txt` / `*.png` / `*_shap.csv` | backtest results | `backtest.py` |

---

## 10. 🍳 Recipes

```bash
# A match was just played → update everything
python3.11 scripts/update_wc2026.py --result "Brazil vs Morocco" --score "2-1"
python3.11 scripts/pipeline.py                      # full refresh (forecast + output_raw.csv)
python3.11 scripts/generate_submission.py           # rebuild output.csv (the submission)

# Honest evaluation on the wide 266-match set
python3.11 scripts/backtest.py --years 2014 2018 2022 --continental --no-shap

# Re-tune after changing the feature set (slow, ~tens of min)
python3.11 scripts/pipeline.py --tune --retune

# Explore visually
streamlit run app/streamlit_app.py
```
