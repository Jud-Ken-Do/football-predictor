# StatsBomb xG Integration Plan

## Context

The model has a documented xG coverage gap (CLAUDE.md limitation #1): no xG for CAF and
CONCACAF teams (FBref Cloudflare-blocked, no free source). African and CONCACAF WC 2026
teams currently rely on form/Elo/Kalman with no chance-quality signal.

The literature is clear that **xG predicts future goals better than goals themselves** —
goals = signal + finishing luck, and teams regress to their xG. The winning approach in a
head-to-head study was "xG post-match → Poisson" (RPS 0.148). Our stack already has the
right machinery: a Kalman EKF with attack/defence state in log-goals space (Koopman-Lit) and
a Bayesian Poisson. The scientifically-supported move is therefore **not** "add another
XGBoost column" but **feed a less-noisy xG observation into the state-space update**.

**StatsBomb open-data** (free, github.com/statsbomb/open-data, attribution required) provides
full event data incl. `shot.statsbomb_xg` + freeze-frames for exactly the gap competitions:

| Competition | comp/season | Fills gap for |
|---|---|---|
| African Cup of Nations 2023 | 1267 / 107 | all 9 CAF WC teams |
| Copa America 2024 | 223 / 282 | Mexico, USA, Canada (CONCACAF) |
| FIFA World Cup 2022 | 43 / 106 | bonus: MAR/SEN/TUN/GHA/USA/MEX/CAN |
| *(bonus)* WC 2018, Euro 2020/2024 | 43/3, 55/43, 55/282 | extra UEFA coverage |

### Key risks already de-risked
- **Penalty shootouts inflate xG**: shootout shots are `period == 5` at ~0.78 xG each.
  Aggregation MUST sum `shot.statsbomb_xg` for **periods ≤ 4 only**. (Verified: Egypt 1–1
  Congo DR → 1.37 / 1.58 in-play xG, not 8.4 / 8.6.)
- **Multi-provider scale mismatch**: we'd mix StatsBomb, football-data.co.uk, and (live)
  FIFA-coords xG — same shot can be 0.12 (one model) vs 0.22 (another); per-match MAE ≈ 1 xG,
  but aggregate correlation > 0.97. → per-source affine calibration + aggregate-only
  consumption (Kalman), never per-match cross-provider mixing.
- **Temporal leakage**: AFCON 2023 (Jan 2024) and Copa 2024 (Jul 2024) postdate every past
  World Cup. They are valid for the **WC 2026 prediction** (all in the past) but MUST NOT enter
  any historical-WC backtest. The backtest validates the *mechanism* using only xG sources
  predating each test tournament; the AFCON/Copa coverage gain for 2026 is a justified
  extrapolation of the proven mechanism, not itself backtestable.

## Approach (build behind a flag, keep only if it beats baseline)

### 1. `football_predictor/data/sources/statsbomb.py`
Fetch+cache event JSON per competition; compute per-match in-play team xG.
- Pattern mirrors `international_results.py` (cache to `~/.cache/football_predictor/statsbomb/`,
  48h staleness, stale-fallback on fetch failure).
- `fetch_statsbomb_xg(comps=DEFAULT_SB_COMPS) -> DataFrame[date, home_team, away_team,
  home_score, away_score, home_xg, away_xg, source]` — team names via `wc2026.normalise()`.
- `build_xg_lookup()` -> `{(home, away, date_str): (xg_home, xg_away)}` (mirrors
  `football_data_co_uk.build_xg_lookup`).
- xG = Σ `shot.statsbomb_xg` over `period ≤ 4`, grouped by `team.name`.

### 2. `football_predictor/models/xg_calibration.py`
Per-source affine calibration `xg' = a + b·xg` fit so post-calibration `mean(xg')=mean(goals)`
and `std` matches, on each source's own overlapping matches. Returns per-source `(a, b)`.
Removes the systematic provider bias the literature documents.

### 3. Unify + attach (`data/pipeline.py` or a helper)
Merge StatsBomb + football-data.co.uk into single calibrated `home_xg`/`away_xg` columns on
training rows (source priority: StatsBomb > football-data where both exist). Causal: a row's
xG is the post-match xG of *that* match (used only as a past observation by the Kalman).

### 4. `features/kalman_strength.py` — blended observation (flagged)
Behind `constants.USE_XG_OBSERVATION` (default off until backtest passes):
- observation `obs = β·goals + (1-β)·xg` when xG present for the row, else raw goals (β tunable,
  start 0.5). Innovation becomes `obs - λ`; everything else (Joseph update, R = λ/weight)
  unchanged.
- add xG sum to `_data_fingerprint()` so EM q re-tunes when xG is toggled.
- forward-only causality preserved (per-match observation, no future data).

### 5. `scripts/backtest.py` — ablation + confederation slice
- `ablation`: `baseline` | `calib_only` | `kalman_xg` | `full`.
- per-confederation log-loss/Brier/accuracy slice (via `confederation.TEAM_CONFEDERATION`).
- leakage guard: StatsBomb xG filtered to `date < test cutoff` (auto-excludes AFCON23/Copa24
  for all past-WC tests).

### 6. Run + decide
WC 2014/2018/2022, baseline vs treatment. **Keep only if** log-loss improves consistently
across folds and outside bootstrap CI noise. Slice by confederation (small CAF/CONCACAF gain
can be washed out in the pooled number). Report honestly; kill if no improvement.

### 7. Tests
Mirror `tests/test_features.py`: statsbomb aggregation excludes shootout; calibration yields
`mean(xg')≈mean(goals)`.

## Verification
- `python3.11 -m pytest tests/ -v`
- `python3.11 scripts/backtest.py --years 2014 2018 2022` baseline vs `--ablation full`
- Compare log-loss [CI] + per-confederation slices; decide keep/kill.
