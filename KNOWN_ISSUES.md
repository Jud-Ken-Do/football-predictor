# Known Issues & Improvements

Working branch: `improvements` (branched from `main` on 2026-06-09)

Issues are ordered by impact. Each has its own sub-branch when work begins.

---

## 1. WC-Specific Features Are Dead Weight in XGBoost [CRITICAL] ✅ FIXED

**What:** Several modules return zeros for ALL historical training matches and only produce real values for WC 2026 prediction targets:
- `sofifa_ratings` — only has EA FC 26 data (one snapshot, no historical versions)
- `squad_wc2026` — WC 2026 squad list only
- `api_form` — pre-cached for WC 2026 teams only
- `venue_wc2026` — WC 2026 venues only
- `injury` — always zero unless manually passed

**Why it's bad:** XGBoost cannot learn a relationship from a feature that never varies during training. At prediction time these features become non-zero — the model is asked to use signals it has never seen. Either XGBoost assigns them zero importance (wasted) or has learned spurious splits on their zero values (noise).

This is a covariate shift / distribution mismatch problem.

**Fix (option B+C, 2026-06-10):**
- Moved all 5 modules from `DEFAULT_FEATURE_MODULES` into `WC_CONTEXT_MODULES` in `constants.py` — XGBoost no longer sees them during training or prediction.
- Created `football_predictor/models/wc_context.py` with a two-stage post-processing layer:
  - **Venue adjustment** (`venue_wc2026`): adjusts BayesPoisson λ_h / λ_a for partial home advantage (MEX/USA/CAN, +6% λ) and altitude acclimatisation, then recomputes Poisson probabilities.
  - **Quality nudge** (`sofifa_ratings` + `api_form`): small log-odds shift (w=0.12) from normalised sofifa overall diff and api_form weighted pts diff. Draw stays proportional.
  - **Player absence penalty**: market-value-weighted log-odds shift from `data/wc2026_player_injuries.json`.
- Applied in `predict_group_stage()` (full venue + quality + absence) and tournament simulation.

**Branch:** `fix/wc-feature-covariate-shift`

---

## 2. Kalman RTS Smoother Leaks Future Information into Training Features [SIGNIFICANT] ✅ FIXED

**What:** The RTS backward pass uses future match results to produce "smoothed" Kalman states. When used as XGBoost training features, a match in 2018 has attack/defence features that implicitly encode 2019-2026 results.

At prediction time only causal forward-pass states are available — which are noisier and less informed. XGBoost learned patterns from smooth, future-aware features but must predict with noisier past-only features.

**Why it's bad:** Artificially inflates training performance metrics. Creates systematic bias — model is overconfident because it was trained on better-than-available estimates.

**Fix (2026-06-10):** `KalmanStrengthFeatures` now defaults to `use_smoothed=False`. The REGISTRY instantiates it with no arguments → forward-only causal states for all XGBoost training features. The RTS smoother is retained for diagnostics and the EM Q-tuning step but not exposed to XGBoost.

**Branch:** `fix/kalman-causal-training`

---

## 3. Third-Place Advancement Not Optimised in Submission [COMPETITION POINTS] ✅ FIXED

**What:** In WC 2026, the 8 best third-place teams advance (67% of all third-place finishers). The group-level score optimiser in `generate_submission.py` only accounts for top-2 advancement per group and ignores third-place advancement bonuses entirely.

**Why it's bad:** Each correctly tipped third-place advancer is worth +3 pts. With 8 of 12 third-place teams advancing, ignoring this means ~24 potential points are unoptimised.

**Fix:** `generate_submission_v2.py` — `estimate_advance_probs()` simulates all 12 groups simultaneously, sorts the 12 third-place teams by pts/gd/gf, and credits the best 8 as advancing. `optimise_group_scores()` uses these cross-group P(advance) values instead of per-group isolation.

**Branch:** `fix/third-place-advancement`

---

## 4. Validation Too Thin — Single WC (64 matches) [SIGNIFICANT] ✅ FIXED

**What:** Model was evaluated on WC 2022 only (64 group stage matches). This is insufficient to distinguish signal from noise in log-loss / Brier metrics.

**Why it's bad:** Log-loss variance on 64 matches is high. Good metrics might be luck. Can't trust hyperparameter choices made based on this evaluation.

**Fix (2026-06-10):** `backtest.py` expanded to:
- **WC 2014, 2018, 2022** (`--years 2014 2018 2022`)
- **4 continental tournaments**: Copa América 2021, UEFA Euro 2020, AFCON 2022, Asian Cup 2023 (`--continental`)
- **Bootstrap 95% CI** on log-loss and Brier score (2000 resamples)
- Aggregate summary table across all tournaments

**Branch:** `fix/rolling-backtest`

---

## 5. Static Ensemble Blend Doesn't Adapt to Match Context [SIGNIFICANT] ✅ FIXED

**What:** A single scalar α=0.60 blends XGBoost and BayesPoisson for all 72 matches regardless of context. The right blend likely varies:
- When odds are available → trust XGBoost more
- When Kalman uncertainty (φ) is high → trust BayesPoisson more
- Rare matchups with little H2H data → trust ratings, not H2H

**Fix (2026-06-10):** `EnsembleModel` now learns a 3-weight logistic regression on context features (`odds_available`, mean Kalman uncertainty, `log1p(h2h_n_matches)`) so α varies per match. L2 regularisation prevents extreme weights. Scalar fallback retained when context unavailable. `pipeline.py`, `predict_wc2026.py`, and `backtest.py` all pass `context_X` at both fit and predict time.

**Branch:** `fix/adaptive-ensemble`

---

## 6. Data Cache Stale — Missing Recent Friendlies [MINOR] ✅ FIXED

**What:** `international_results.csv` is cached from June 6. WC pre-tournament friendlies from June 7-10 (e.g., Morocco vs Norway) are missing.

**Fix:** Added 48h staleness check in `fetch_international_results()`. Cache is re-downloaded automatically if older than 48 hours; `use_cache=True` still avoids redundant downloads within the same day.

**Branch:** inline fix (no separate branch needed).

---

## 7. Live Pre-Match Odds Not Available [MINOR — timing]

**What:** All WC 2026 fixtures have `odds_available=0.0` and flat 0.333 implied probabilities. The odds module is effectively switched off for the actual predictions — the single strongest external signal is missing.

**Why:** Betting markets don't open until close to tournament start (after submission deadline of June 10).

**Status:** Nothing to fix before submission. After June 11, re-running `generate_submission.py` with live odds would improve predictions significantly.

---

## Summary Table

| # | Issue | Priority | Branch |
|---|---|---|---|
| 1 | WC features = zeros in training | Critical ✅ | `fix/wc-feature-covariate-shift` |
| 2 | Kalman smoother leaks future data | Significant ✅ | `fix/kalman-causal-training` |
| 3 | Third-place advancement not optimised | Competition ✅ | `fix/third-place-advancement` |
| 4 | Validation on 64 matches only | Significant ✅ | `fix/rolling-backtest` |
| 5 | Static ensemble blend | Significant ✅ | `fix/adaptive-ensemble` |
| 6 | Stale data cache | Minor ✅ | inline |
| 7 | No live odds | Minor (timing) | n/a |
