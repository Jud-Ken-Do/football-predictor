# Known Issues & Improvements

Working branch: `improvements` (branched from `main` on 2026-06-09)

Issues are ordered by impact. Each has its own sub-branch when work begins.

---

## 1. WC-Specific Features Are Dead Weight in XGBoost [CRITICAL]

**What:** Several modules return zeros for ALL historical training matches and only produce real values for WC 2026 prediction targets:
- `sofifa_ratings` — only has EA FC 26 data (one snapshot, no historical versions)
- `squad_wc2026` — WC 2026 squad list only
- `api_form` — pre-cached for WC 2026 teams only
- `venue_wc2026` — WC 2026 venues only
- `injury` — always zero unless manually passed
- `xg_form` — depends on data coverage

**Why it's bad:** XGBoost cannot learn a relationship from a feature that never varies during training. At prediction time these features become non-zero — the model is asked to use signals it has never seen. Either XGBoost assigns them zero importance (wasted) or has learned spurious splits on their zero values (noise).

This is a covariate shift / distribution mismatch problem.

**Fix options (pick one):**
- A: Get historical versions of these features so they vary during training (ideal but hard for SoFIFA)
- B: Move them outside XGBoost — apply as post-processing multipliers on ensemble probabilities
- C: Train XGBoost without them, add a separate "WC adjustment layer" on top that uses current-state features

**Branch:** `fix/wc-feature-covariate-shift`

---

## 2. Kalman RTS Smoother Leaks Future Information into Training Features [SIGNIFICANT]

**What:** The RTS backward pass uses future match results to produce "smoothed" Kalman states. When used as XGBoost training features, a match in 2018 has attack/defence features that implicitly encode 2019-2026 results.

At prediction time only causal forward-pass states are available — which are noisier and less informed. XGBoost learned patterns from smooth, future-aware features but must predict with noisier past-only features.

**Why it's bad:** Artificially inflates training performance metrics. Creates systematic bias — model is overconfident because it was trained on better-than-available estimates.

**Fix:** Use only causal forward-pass Kalman states when constructing XGBoost training features. Keep the smoother for analysis/diagnostics only.

**Branch:** `fix/kalman-causal-training`

---

## 3. Third-Place Advancement Not Optimised in Submission [COMPETITION POINTS]

**What:** In WC 2026, the 8 best third-place teams advance (67% of all third-place finishers). The group-level score optimiser in `generate_submission.py` only accounts for top-2 advancement per group and ignores third-place advancement bonuses entirely.

**Why it's bad:** Each correctly tipped third-place advancer is worth +3 pts. With 8 of 12 third-place teams advancing, ignoring this means ~24 potential points are unoptimised.

**Fix:** After group-level optimisation, simulate full tournament to identify the predicted 8 best third-place teams, then re-run group optimiser with extended advancement targets.

**Branch:** `fix/third-place-advancement`

---

## 4. Validation Too Thin — Single WC (64 matches) [SIGNIFICANT]

**What:** Model is evaluated on WC 2022 only (64 group stage matches). This is insufficient to distinguish signal from noise in RPS metrics.

**Why it's bad:** RPS variance on 64 matches is high. Good metrics might be luck. Can't trust hyperparameter choices made based on this evaluation.

**Fix:** Rolling backtest — train on pre-2018, evaluate WC 2018; train on pre-2022, evaluate WC 2022. Average metrics across both tournaments.

**Branch:** `fix/rolling-backtest`

---

## 5. Static Ensemble Blend Doesn't Adapt to Match Context [SIGNIFICANT]

**What:** A single scalar α=0.60 blends XGBoost and BayesPoisson for all 72 matches regardless of context. The right blend likely varies:
- When odds are available → trust XGBoost more
- When Kalman uncertainty (φ) is high → trust BayesPoisson more  
- Rare matchups with little H2H data → trust ratings, not H2H

Also: BayesPoisson at 40% dilutes 218 features worth of signal — seems high given what XGBoost has access to. The 40% weight may be compensating for XGBoost overconfidence rather than reflecting true BayesPoisson quality.

**Fix:** Replace scalar α with a small logistic regression that predicts optimal blend weight based on match context features (odds_available, kalman_uncertainty, h2h_sample_size, etc.).

**Branch:** `fix/adaptive-ensemble`

---

## 6. Data Cache Stale — Missing Recent Friendlies [MINOR]

**What:** `international_results.csv` is cached from June 6. WC pre-tournament friendlies from June 7-10 (e.g., Morocco vs Norway) are missing.

**Fix:** Add cache freshness check — if cache is >48h old, re-download before training. One line in `fetch_training_data()`.

**Branch:** can be done inline, no separate branch needed.

---

## 7. Live Pre-Match Odds Not Available [MINOR — timing]

**What:** All WC 2026 fixtures have `odds_available=0.0` and flat 0.333 implied probabilities. The odds module is effectively switched off for the actual predictions — the single strongest external signal is missing.

**Why:** Betting markets don't open until close to tournament start (after submission deadline of June 10).

**Status:** Nothing to fix before submission. After June 11, re-running `generate_submission.py` with live odds would improve predictions significantly.

---

## Summary Table

| # | Issue | Priority | Branch |
|---|---|---|---|
| 1 | WC features = zeros in training | Critical | `fix/wc-feature-covariate-shift` |
| 2 | Kalman smoother leaks future data | Significant | `fix/kalman-causal-training` |
| 3 | Third-place advancement not optimised | Competition | `fix/third-place-advancement` |
| 4 | Validation on 64 matches only | Significant | `fix/rolling-backtest` |
| 5 | Static ensemble blend | Significant | `fix/adaptive-ensemble` |
| 6 | Stale data cache | Minor | inline |
| 7 | No live odds | Minor (timing) | n/a |
