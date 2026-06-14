# Architecture Review — Findings & Proposed Fixes

**Date:** 2026-06-12 (tournament is live — some fixes are time-critical)
**Scope:** rating systems, probabilistic models, feature/data pipeline, orchestration & simulation scripts. Frontend (`app/`) excluded.
**Last updated:** 2026-06-12 — fix progress tracked inline.

**Status legend:** ✅ FIXED (implemented + verified) · ⬜ open

Findings are ordered by severity. Each has a location, why it matters, and a concrete fix. Items marked **⏰ TIME-CRITICAL** corrupt live-tournament predictions and should be fixed before the next prediction run.

## Fix progress

| Phase | Status |
|---|---|
| Phase 0 (live-critical): C1, C2, C3, C4, C5, C8, B1, B11 (+B3 early) | ✅ all fixed 2026-06-12 |
| Phase 1 (model stack): C6, B2, B4, B5, B8, W1–W3 (+W4, B10, B16 early) | ✅ all fixed 2026-06-12 |
| Phase 2 (eval honesty): C7, B7, B9, B10 | ✅ all fixed 2026-06-12 — re-run backtests to refresh reported metrics |
| Phase 3 (robustness): B6, B12–B17, W4–W20 | ✅ fixed 2026-06-12 (W17 accepted by design; W18 partial — see roadmap R8) |

---

## 1. Critical bugs

### ✅ C1. ⏰ Same-date snapshot fallback collapses ratings to defaults (Elo + Glicko-2)
**FIXED 2026-06-12** — `elo.py` / `glicko2.py` transform now falls back `snapshot → latest rating → default` (Kalman pattern). Regression test: Brazil keeps its rating on a same-day fixture.
**Where:** `football_predictor/features/elo.py:44-46`, `football_predictor/features/glicko2.py:114-116`

Rating snapshots are keyed by date and contain only teams that played *in the context data* on that date. `transform()` falls back to defaults when the date exists but the team is absent:

```python
snap = self._snapshot.get(date_str, self._final)
r_home_raw = snap.get(home, self._DEFAULT_RATING)   # ← silent 1500 / Glicko default
```

WC 2026 has 4–6 matches per day. The moment one result for date D is recorded via `update_wc2026.py`, a snapshot exists for D — and **every not-yet-played fixture on the same day gets Elo 1500 / Glicko default for both teams**. Brazil rated equal to Curaçao, silently, feeding XGBoost and the ensemble context.

`kalman_strength.py:240-241` already has the correct pattern: `snap.get(home) or self._states.get(home) or default`.

**Fix:** mirror the Kalman fallback in Elo and Glicko-2 (`snap.get(home) or self._final.get(home) or default`), or better, keep a per-team sorted `(date, rating)` timeline and bisect for "last rating strictly before date".

---

### ✅ C2. ⏰ Imminent double-counting of WC 2026 results (starts within ~48h)
**FIXED 2026-06-12** — new `append_actual_results()` helper in `predict_wc2026.py` (also wired into `pipeline.py step_load`, which previously never saw recorded results at all): canonicalises recorded names, drops duplicate WC-2026 rows from the martj42 data by normalised team pair, validates 48-team coverage. `update_wc2026.py cmd_record` now validates fixtures against the official schedule, swaps reversed orientation, defaults the date from the schedule, and dedups by pair. `merge_actual_results` accepts both orientations. Recorded result dates corrected 06-12 → 06-11 (official schedule dates; scores untouched). Verified: simulated martj42 refresh with raw-CSV names → each match appears exactly once.
**Where:** `scripts/predict_wc2026.py:151-165`, `football_predictor/data/sources/international_results.py:45-50`, `scripts/update_wc2026.py:93-121`

`train_model()` appends `data/wc2026_actual_results.json` to the training data unconditionally. The martj42 cache auto-refreshes every 48h and **will start containing the same WC 2026 matches** → Elo/Glicko/Kalman/BayesPoisson each see played matches twice, at match weight 1.5, double-stepping every rating system.

Aggravating factors:
- `update_wc2026.py` stores team names verbatim from user input (`"Iran"` vs dataset-canonical `"IR Iran"`) and `train_model` appends without `normalise()` → phantom teams in the rating systems.
- The recorded Mexico–South Africa result has `"date": "2026-06-12"` while the schedule says `2026-06-11`, so a naive `(date, home, away)` dedup would miss it.
- `merge_actual_results` looks up the ordered `(home, away)` tuple — a result recorded with teams swapped silently fails to lock.

**Fix:** dedup `all_data` after concat on (sorted normalised team pair, date ±1 day, tournament = World Cup); normalise names in `cmd_record` and reject teams not in the 48-team list; reconcile recorded dates with the schedule; make `merge_actual_results` check both orientations.

---

### ✅ C3. Knockout bracket mapping (R16/QF) does not match the official FIFA 2026 bracket
**FIXED 2026-06-12** — verified against Wikipedia's official bracket (R32_MATCHES was already correct); `R16_PAIRS = [(1,4),(0,2),(3,5),(6,7),(10,11),(8,9),(13,15),(12,14)]`, `QF_PAIRS = [(0,1),(4,5),(2,3),(6,7)]` with full match-number documentation in comments.
**Where:** `scripts/predict_wc2026.py:120-122`

```python
R16_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
QF_PAIRS  = [(0, 1), (2, 3), (4, 5), (6, 7)]
```

The 16 `R32_MATCHES` (M73–M88) are correct, but R16 pairs sequential winners. The official bracket is: M89 = W74 v W77, M90 = W73 v W75, M91 = W76 v W78, M92 = W79 v W80, M93 = W83 v W84, M94 = W81 v W82, M95 = W86 v W88, M96 = W85 v W87; QFs: M97=(89,90), M98=(93,94), M99=(91,92), M100=(95,96). **Every R16/QF/SF/winner probability is computed on the wrong topology.**

**Fix** (keeping the index-into-`R32_MATCHES` convention, list order M89..M96):

```python
R16_PAIRS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF_PAIRS  = [(0, 1), (4, 5), (2, 3), (6, 7)]
```

Verify against the official FIFA match numbering before shipping.

---

### ✅ C4. `normalise()` maps team names in the wrong direction — 6 WC teams get league-average λ in BayesPoisson
**FIXED 2026-06-12** — `TEAM_NAME_MAP` now maps variants → canonical post-rename names; all 48 draw names are identity. Added `ALL_WC2026_TEAMS` + `validate_team_coverage()` ingress check (runs on every training load, prints loud warning). Verified: all 48 teams resolve with >10 matches in the cache.
**Where:** `football_predictor/data/wc2026.py:42-64` vs `football_predictor/data/sources/international_results.py:66-73`

The data source renames raw CSV names *to* `Czechia`, `Türkiye`, `Cabo Verde`, `South Korea`, `Ivory Coast`, `Bosnia and Herzegovina`. But `TEAM_NAME_MAP` maps the **opposite** way (`"South Korea" → "Korea Republic"`, `"Czechia" → "Czech Republic"`, …). Every BayesPoisson call goes through `normalise()`, and `_lambda` falls back silently to attack/def = 0 for unknown names.

**Result:** South Korea, Czechia, Türkiye, Cabo Verde, Ivory Coast, and Bosnia & Herzegovina get exactly league-average strength in the BP leg of the ensemble, in scoreline simulation, and in knockout pair-probs — affecting ~17 of 72 group fixtures. Confirmed empirically: `Korea Republic: 0` rows in the training cache.

**Fix:** make `TEAM_NAME_MAP` map *to* the post-rename canonical names (mostly identity now). Add a startup assertion that `normalise(team)` for all 48 teams returns a name with > 0 training rows.

---

### ✅ C5. WC 2026 odds keyed under normalised names but queried with raw names
**FIXED 2026-06-12** — `OddsFeatures.transform` now normalises both team names before lookup; key-build and query share one canonicalisation.
**Where:** `football_predictor/data/sources/football_data_co_uk.py:158-167` vs `football_predictor/features/odds.py:74`

The odds lookup is built under `(normalise(home), normalise(away), date)` but `OddsFeatures.transform` queries with raw fixture names. With C4's broken map, fixtures involving the six renamed teams silently get `odds_available=0.0` and flat odds features — and `odds_available` gates the context-adaptive ensemble α, so blending behaviour differs systematically for exactly the teams already hurt by C4.

**Fix:** apply one canonical normalisation on both key-build and query.

---

### ✅ C6. Ensemble α fitted on raw XGB probabilities, applied to temperature-scaled ones
**FIXED 2026-06-12** — `ensemble.fit(temp_cal.transform(xgb_proba_cal), ...)` in `predict_wc2026.py`, `pipeline.py`, and both backtest paths.
**Where:** `scripts/predict_wc2026.py:191-214` vs `:265`; same pattern in `scripts/pipeline.py:172-201` vs `:237` and `scripts/backtest.py:315-335` vs `:339-343`
*(independently found by two reviewers)*

```python
xgb_proba_cal = xgb.predict_proba(cal_X)          # raw
temp_cal.fit(xgb_proba_cal, cal_y)
ensemble.fit(xgb_proba_cal, bp_proba_cal, cal_y)  # α learned against RAW probs
# ...prediction time:
xgb_proba = temp_cal.transform(xgb.predict_proba(X_gs))  # ← scaled probs blended
```

With T > 1 the scaled distribution is flatter than what α was optimised against. The learned per-match α and the context weights are tuned for an input distribution the ensemble never sees in production.

**Fix (one line per script):** `ensemble.fit(temp_cal.transform(xgb_proba_cal), bp_proba_cal, cal_y, context_X=cal_X)`.

---

### ✅ C7. Backtest evaluates WC 2018/2022 with hyperparameters tuned on WC 2018/2022
**FIXED 2026-06-12** — `GradientBoostModel(use_tuned_cache=False)` parameter added; both backtest paths use it. **All previously reported backtest numbers should be re-run.**
**Where:** `football_predictor/models/gradient_boost.py:42-45`, `scripts/pipeline.py:154-170`, `scripts/backtest.py:312, 467`

`GradientBoostModel.__init__` silently auto-loads `data/xgb_tuned_params.json` — parameters selected by Optuna with WC 2018/2022 group stages as validation folds. `backtest.py` then reports log-loss on those same tournaments. All reported backtest metrics (and the friendly-weight grid search, which optimises the same numbers) are optimistically biased.

**Fix:** add `GradientBoostModel(use_tuned_cache=False)` for backtests, or treat WC 2014 + continental tournaments as the only clean evaluation sets.

---

### ✅ C8. WC post-processing can produce negative probabilities → crash or garbage
**FIXED 2026-06-12** — `apply_to_match` clips all three probs to ≥1e-6 before renormalising. Stress-tested with a venue λ-shift on a lopsided fixture (draw prob hit the floor instead of going negative; `absence_adjust` no longer crashes).
**Where:** `football_predictor/models/wc_context.py:300-309`, `:263`
*(independently found by two reviewers)*

```python
p_h = ens_p_h + (bp_h - bp_h_orig)        # unbounded additive delta
total = max(p_h + p_d + p_a, 1e-10)       # renormalise — sign survives!
```

If the ensemble gives a heavy underdog `p_a = 0.04` and the venue+market λ shift moves the Poisson away-win prob down by 0.06, `p_a` goes negative and stays negative after renormalisation. Downstream: `absence_adjust` computes `math.log(p_h / p_a)` → `ValueError: math domain error`, and `generate_submission.py:47` `rng.choice(3, p=[...])` raises on negatives. Lopsided fixtures — which the WC group stage is full of — are exactly the trigger.

**Fix:** clip all three probs to `max(p, 1e-6)` before renormalising. Better long-term: apply venue/market shifts multiplicatively in log-odds space, which can never leave the simplex (see roadmap R1).

---

## 2. Bugs

### ✅ B1. Lognormal λ resampling is not mean-preserving (systematic goal inflation)
**FIXED 2026-06-12** — location parameter is now `log(λ) − σ²/2` in both `predict_wc2026.simulate_goals` and `generate_submission._simulate_match`; resample gates are per-side. Verified numerically: bias +8.4% → 0.0% at σ=0.4.
**Where:** `scripts/predict_wc2026.py:336-338`, `scripts/generate_submission.py:39-44`

`np.random.lognormal(np.log(lam), sigma)` has mean `λ·e^{σ²/2}` — with Kalman log-stds of 0.2–0.5 that inflates every expected goal rate by 2–13% in the Monte Carlo simulator and submission optimizer, biasing toward high scorelines and fewer draws.
**Fix:** `mu = log(λ) − σ²/2`. Also: the resample gate `if lam_h_log_std > 0.02` checks only the home std but controls both λs, and home/away draws should be correlated (shared opponent states).

### ✅ B2. Transfermarkt June-2026 values leak into all historical training rows
**FIXED 2026-06-12** — `transfermarkt` moved from `DEFAULT_FEATURE_MODULES` (now 12 modules) to `WC_CONTEXT_MODULES` (now 7). Next training run no longer sees June-2026 values on historical rows. **Re-wired at prediction time (same day):** `quality_nudge` now includes a transfermarkt component (`tm_value_ratio/1.5`, weight 0.2 vs 0.4/0.4 sofifa/form, only when `tm_available`) so the squad-value signal isn't silently dropped after leaving training.
**Where:** `football_predictor/features/transfermarkt.py:57-59, 70-95`

The docstring claims historical rows get 0, but `_value()` is a pure name lookup with no date condition — a 2012 match gets June-2026 squad values with `tm_available=1`. This is exactly the covariate-shift/leakage class the project excluded sofifa for.
**Fix:** move `transfermarkt` from `DEFAULT_FEATURE_MODULES` to `WC_CONTEXT_MODULES`.

### ✅ B3. Glicko-2 Illinois bracketing broken in the high-surprise branch
**FIXED 2026-06-12** — downward bracket search now runs only in the `Δ² ≤ φ² + v` branch and steps by τ, per Glickman (2012) step 5.2.
**Where:** `football_predictor/features/glicko2.py:71-73`

The `while f(B) < 0: B -= 1.0` downward search must run **only** in the `Δ² ≤ φ² + v` branch (Glickman step 5.2). When `Δ² > φ² + v`, `B = ln(Δ²−φ²−v)` is already a valid bracket endpoint; the unconditional loop walks B past the root, so the regula-falsi iteration is unbracketed exactly after upsets — when volatility should rise.
**Fix:** run the search only in the `else` branch, stepping by `_TAU` per the paper.

### ✅ B4. Glicko-2 RD never grows — inactivity branch is dead code
**FIXED 2026-06-12** — `_inflate_phi()` grows RD per 90-day rating period of inactivity (capped at the unknown-team default), applied both in cache building and at transform-time fallback. Verified: dormant-since-2015 team RD 151.7 vs active team 87.0.
**Where:** `football_predictor/features/glicko2.py:53-54, 148-171`

`_update` handles the empty-results RD-inflation case, but the cache only ever calls `_update` for teams that played. RD only shrinks; a team active in 2015 and dormant since keeps a tiny φ forever. This deletes Glicko-2's entire advantage over Elo, and the `glicko2_*_rd` features are systematically wrong for infrequent teams.
**Fix:** define a rating period (60–90 days) and lazily inflate `φ ← sqrt(φ² + σ²·n_periods)` based on days since last match — in cache building and in `transform()`.

### ✅ B5. Kalman: inconsistent R between gain and Joseph-form update; S ignores opponent uncertainty
**FIXED 2026-06-12** — single `r_eff` used in both the gain and the Joseph term; `extra_obs_var` adds the opponent's λ²·P contribution to S (captured before either side updates).
**Where:** `football_predictor/features/kalman_strength.py:96-124`

(a) The gain uses `R = λ/match_weight` but the Joseph covariance term uses `R = λ` — posterior P understated for friendlies, overstated for WC matches.
(b) The home-goals observation depends on both teams' states, but two independent scalar updates each use `S = λ²·P_own + R` and each absorb the full innovation. Correct shared `S = λ²·(P_h[0,0] + P_a[1,1]) + R`. As written, gains are too large → P shrinks too fast → systematic overconfidence in the `kalman_*_std` features that gate the ensemble α.
**Fix:** compute `r` once and use it in both places; compute the shared S including both variance contributions.

### ✅ B6. MCMC path drops the Dixon-Coles correction; unknown team → index 0
**FIXED 2026-06-12** — `predict_proba_mcmc` applies `_dc_tau` to the four low-score cells per posterior sample; unknown teams fall back to league average (zero att/def) with a warning; posterior subsampling is seeded (reproducible).
**Where:** `football_predictor/models/bayesian_poisson.py:241-243, 421-422, 436-448`

`predict_proba_mcmc` builds plain product Poisson — the fitted ρ is ignored under `--mcmc`, so draw probabilities lose the DC boost. And `idx.get(home, 0)` maps an unknown team to whatever team is alphabetically first (the MAP path correctly falls back to league average).
**Fix:** apply the τ matrix per posterior sample; fall back to zero attack/defence for unknown teams (with a warning).

### ✅ B7. Friendly-weight "auto-tuning" doesn't exist; `--retune` is a no-op; backtest uses a different weight
**FIXED 2026-06-12** — `tune_friendly_weight` now persists to `data/tuned_params.json` (with `friendly_weight_tuned_on` provenance); `pipeline.py --retune` wired (also auto-tunes when no cache); `step_backtest` evaluates the deployed weight.
**Where:** `scripts/pipeline.py:462-463`, `scripts/backtest.py:259, 521-560`

Nothing ever **writes** `data/tuned_params.json` — `tune_friendly_weight()` prints the best weight and discards it; `args.retune` is never read. The cache currently holds `friendly_weight: 0.8` (CLAUDE.md claims 0.7), production trains with 0.8, while `pipeline.py --backtest` runs with the default **0.3** — the backtest never evaluates the deployed configuration.
**Fix:** persist `best_w` to the cache, wire `args.retune`, pass the cached weight into the backtest step.

### ✅ B8. Third-place slot allocation: greedy assignment violates FIFA constraints
**FIXED 2026-06-12** — `match_thirds()` backtracking bipartite matching (most-constrained-first). Verified exhaustively: all 495 possible 8-of-12 third-place combinations produce feasible, constraint-satisfying assignments.
**Where:** `scripts/predict_wc2026.py:425-439`

`assign_third` greedily consumes the first allowed group; when the greedy choice exhausts a later slot's options, the fallback hands it a *disallowed* group — even when a feasible assignment exists. A meaningful fraction of the 50k sims run constraint-violating brackets.
**Fix:** solve the 8-slot bipartite matching with backtracking (trivial at this size), or encode FIFA's official allocation table.

### ✅ B9. Backtest "group stage" windows include knockout matches
**FIXED 2026-06-12** — WC 2022 window ends 12-02; Copa América 2021 ends 06-28.
**Where:** `scripts/backtest.py:58-91`

- 2022 window ends `2022-12-09`; the group stage ended **Dec 2** — all 8 R16 matches and 2 QFs leak into the eval set (different draw dynamics, possible ET scorelines).
- Copa América 2021 window ends Jul 2; group stage ended Jun 28.
**Fix:** `2022 → ("2022-11-20", "2022-12-02")`, Copa end `2021-06-28`.

### ✅ B10. Backtest fits ensemble on in-sample BayesPoisson probabilities
**FIXED 2026-06-12** — both backtest paths now fit BP on `[:cut]`, fit the ensemble on the held-out tail, then refit BP on full data + `fit_rho` (mirrors production).
**Where:** `scripts/backtest.py:326-335`

`bp.fit(train_df)` on the full set, then calibration probs computed on a subset of BP's own training data — the ensemble α overweights BP in backtests relative to production (which correctly fits BP on `[:cut]` first). Together with C7, the backtest measures neither the production configuration nor an unbiased one.
**Fix:** mirror the production order: fit BP on `[:cut]`, fit ensemble on `[cut:]`, refit on full, then `fit_rho`.

### ✅ B11. Historical odds/xG joins broken for South Korea and Ivory Coast
**FIXED 2026-06-12** — removed the two wrong entries from `football_data_co_uk._NAME_MAP` (names stay canonical).
**Where:** `football_predictor/data/sources/football_data_co_uk.py:32-33`

`_NAME_MAP` maps to `"Korea Republic"` / `"Côte d'Ivoire"`, names that never appear in training data — every historical odds row and xG timeline for these teams misses silently.
**Fix:** map to `"South Korea"` / `"Ivory Coast"`.

### ✅ B12. Rankings: renamed teams lose pre-rename history; dangerous substring fallback
**FIXED 2026-06-12** — `_RANKING_ALIASES` unions old+new name timelines (`.isin`); the first-word substring fallback (could match a different country) is deleted.
**Where:** `football_predictor/features/rankings.py:92-109`

The cache has `Czech Republic: 314` rows + `Czechia: 14` (same for Turkey/Türkiye), and the lookup never unions them — pre-2023 matches for these teams fall to the default rank 100 / 1000 pts. The fallback `str.contains(team.split()[0])` can silently match a *different country*.
**Fix:** alias table unioning both name variants; drop the substring fallback.

### ✅ B13. `xg_available=1` even when no xG precedes the match
**FIXED 2026-06-12** — flag now reflects whether rolling means actually exist before the match date; `or`-defaults replaced with explicit `is None` checks (a legitimate 0.0 mean no longer reads as missing).
**Where:** `football_predictor/features/xg_form.py:113-123`

The flag checks timeline existence, not whether entries precede the match date — a 2015 row has `xg_available=1` with all xG features at the imputed default. Also `_rolling_mean(...) or _DEFAULT_XG` treats a legitimate 0.0 as missing.
**Fix:** set the flag from `_rolling_mean(...) is not None` on both sides; use explicit `is None` checks.

### ✅ B14. `market_over25` is not vig-stripped (comment claims it is)
**FIXED 2026-06-12** — fetches the Under 2.5 odd and normalises `p = imp_over/(imp_over+imp_under)`; falls back to a 5% margin haircut when no Under odd exists. (Takes effect on next odds fetch.)
**Where:** `scripts/fetch_wc2026_odds.py:67-72`

`min(raw, 0.97)` caps the implied probability; it doesn't remove the 4–7% two-way overround. P(over 2.5) biased high by ~2–3pp → market-implied λ_total inflated ~0.1–0.15 goals per match with odds.
**Fix:** fetch the Under 2.5 odd too and normalise: `p_over = imp_over / (imp_over + imp_under)`.

### ✅ B15. Kalman EM q cache: stale `_LAST_TUNED_Q` on cache hit; `id()`-keyed caches
**FIXED 2026-06-12** — content fingerprint `(len, min date, max date, total goals)` replaces `id()` keys in both the EM cache and `_ensure_cache`; cache hits refresh `_LAST_TUNED_Q`.
**Where:** `football_predictor/features/kalman_strength.py:197-202, 280-287, 382-395`

The cache-hit path doesn't refresh `_LAST_TUNED_Q`, so `get_last_tuned_q()` is ordering-dependent — BayesPoisson can derive dataset A's half-life from dataset B's q. `_EM_Q_CACHE` keys on `(id(data), len(data))`; `id()` values are recycled after GC, so long sessions can silently reuse the wrong q.
**Fix:** key caches by a content fingerprint (`len, min_date, max_date, goals_sum`); always refresh `_LAST_TUNED_Q` on hits; prefer passing q explicitly over a mutable class global.

### ✅ B16. RTS smoother fallback does element-wise matrix division
**FIXED 2026-06-12** — fallback now uses a regularised matrix inverse.
**Where:** `football_predictor/features/kalman_strength.py:142-144`

`G = P_upd / (P_pred + 1e-8·I)` is element-wise — mathematically meaningless, and the diagonal-only regulariser produces `inf` on zero off-diagonals, poisoning the EM-tuned q.
**Fix:** `G = P_upd @ np.linalg.inv(P_pred + 1e-8 * np.eye(2))`.

### ✅ B17. `output_raw.csv` λ columns not swapped with team orientation
**FIXED 2026-06-12** — λs swap together with scores/probs when the row orientation flips.
**Where:** `scripts/pipeline.py:389-404`

Scores/probs are swapped when the row orientation flips but `lam_home`/`lam_away` stay model-oriented — contradictory columns in the same row.

---

## 3. Weaknesses (model-quality, not crashes)

| # | Where | Issue | Fix |
|---|---|---|---|
| ✅ W1 | `predict_wc2026.py` | **FIXED** — `group_standings` resolves residual ties with H2H mini-table then random lots; third-place pool gets random final tiebreak. (`generate_submission._group_standings` still pending — see W18 note) | — |
| ✅ W2 | `predict_wc2026.py` | **FIXED** — KO draws resolved by `p_h/(p_h+p_a)` | — |
| ✅ W3 | `predict_wc2026.py`, `pipeline.py` | **FIXED** — both orientations averaged + renormalised in both pair caches | — |
| ✅ W4 | `pipeline.py`, `predict_wc2026.py` | **FIXED** — `absence_adjust` now applied to KO pair probabilities in both scripts | — |
| ✅ W5 | `wc_context.py:119-139` | **FIXED** — docstrings corrected (p_draw held fixed, then renormalised); `date.today()` flagged as live-only | — |
| ✅ W6 | `features/h2h.py:21-28` | **FIXED** — symmetric 1.2/1.2 no-history defaults | — |
| ✅ W7 | `features/form.py:116-137` | **FIXED** — pts/gd windows now per-game means + `form_*_n_matches` features | — |
| ✅ W8 | `features/tournament_stage.py:58` | **FIXED** — `tournament_stage` removed from training modules (module file kept) | — |
| ✅ W9 | `data/pipeline.py:105`, `gradient_boost.py:66-69` | **FIXED** — loud warning when predict-time columns are missing before zero-fill | — |
| ✅ W10 | `ensemble.py:39-47` | **FIXED** — `prune_correlated_features(protect=...)` defaults to the 6 ensemble context columns | — |
| ✅ W11 | `glicko2.py` (module) | **FIXED** — HA offset 100/173.72 ≈ 0.576 on the μ scale, non-neutral matches only (updates + win-prob feature) | — |
| ✅ W12 | `kalman_strength.py:310-314` | **FIXED** — snapshots store post-time-update (predictive) P; future fixtures age P by q²Δt to the fixture date | — |
| ✅ W13 | `predict_wc2026.py:161` | **FIXED** — host nations (MEX/USA/CAN) as home side recorded non-neutral in rating updates | — |
| ✅ W14 | `predict_wc2026.py:832`, `pipeline.py:314` | **FIXED** — `--seed` on predict_wc2026.py + pipeline.py; `estimate_advance_probs(seed=)`; blend_sweep monkey-patch removed | — |
| ✅ W15 | `bayesian_poisson.py:130, 199-211` | **FIXED** — analytic Poisson-GLM gradient (fit now ~0.1s vs finite differences), linear predictor clipped before exp, ρ likelihood weighted + positivity-adapted lower bound | — |
| ✅ W16 | `calibration.py:85`, `ensemble.py:90` | **FIXED** — `sample_weight` through TemperatureScaling + EnsembleModel fits; all callers pass match weights | — |
| ✅ W17 | `wc_context` + `odds.py` | **RESOLVED WITH EVIDENCE 2026-06-13** — `scripts/odds_blend_backtest.py` showed the "XGB learns the 1X2 market via features" assumption fails empirically: closing odds alone beat the full stack on WC 2018 (0.9312 vs 0.9884) and WC 2022 (1.0318 vs 1.0843); blending improves log-loss monotonically to w≈0.8 (paired Δ −0.056 [−0.116, +0.005], n=96). Added `market_1x2_blend` as final post-processing stage at `_MARKET_1X2_BLEND=0.70` (deployed below optimum: WC-26 cache holds earlier, softer odds than backtest closing lines). Scorelines remain λ-driven; no-op without odds | — |
| ✅ W18 | `generate_submission.py:304-330` | **PARTIALLY FIXED** — multi-restart (greedy + 7 random) coordinate descent against local optima. Open: conditional second pass for cross-group coupling (roadmap R8); verify third-place bonus rule against competition rules | — |
| ✅ W19 | `international_results.py:50-59` | **FIXED** — stale-cache fallback on fetch failure + duplicate-row guard | — |
| ✅ W20 | `bayesian_poisson.py:123` vs `:329` | **FIXED** — MCMC home-advantage prior aligned to MAP's N(0.20, 0.15²) | — |

---

## 4. Recommended fix order

**Phase 0 — before the next prediction run (live-tournament critical):**
1. C2 (dedup live results — corruption starts when martj42 picks up WC matches)
2. C1 (same-date rating collapse — corrupts all same-day fixtures the moment one result is recorded)
3. C3 (bracket topology — all knockout probabilities wrong)
4. C4 + C5 + B11 (team-name direction — 6 teams at league-average strength)
5. C8 + B1 (negative probs crash; lognormal mean bias in every sim)

**Phase 1 — correctness of the model stack:**
6. C6 (ensemble/temperature train-serve mismatch — one line per script)
7. B5 (Kalman S/R consistency — uncertainty channel feeds the ensemble)
8. B2 (transfermarkt leakage — move to context modules, retrain)
9. B3 + B4 (Glicko volatility + RD growth)
10. B8 + W1 + W2 + W3 (simulation fidelity: third-place matching, tiebreakers, KO draws, pair symmetry)

**Phase 2 — evaluation honesty (so future decisions are evidence-based):**
11. C7 + B9 + B10 + B7 (backtest leakage, window contamination, in-sample BP, friendly-weight persistence). Until these land, treat all current backtest numbers as optimistic.

**Phase 3 — robustness & hygiene:** B6, B12–B17, W4–W20.

---

## 5. Model-improvement roadmap (beyond bug fixes)

**❌ R1 (TESTED & REJECTED 2026-06-13). Log-linear (geometric) ensemble pooling.** Replace `α·p_xgb + (1−α)·p_bp` with `softmax(α·log p_xgb + (1−α)·log p_bp)`. Tested at identical config (friendly_weight=0.8): WC backtest average **worsened 1.0092 → 1.0165**, degrading all three years (2014 0.9578→0.9630, 2018 0.9690→0.9749, 2022 1.1008→1.1115). The geometric pool sharpens the blend (it behaves like a product of experts), and on log-loss the linear mixture's hedging wins consistently. Reverted; arithmetic blend retained. The negative-probability motivation (C8, W5) is already handled by clipping.

**✅ R2 (DONE 2026-06-12). Persistent team strength per tournament simulation.** Draw each team's log-λ perturbation **once per simulation** from the Kalman posterior instead of per match. Correlating a team's performance across its own matches materially fattens the tails of "dark horse deep run" probabilities — the main thing the simulator exists to quantify.

**✅ R3 (DONE 2026-06-13). Out-of-time K-fold stacking for T and α.** The single 20% chronological tail is high-variance for fitting the temperature and 4 context weights. Now pools out-of-time predictions across 4 expanding-window folds (last 40% of data) and fits the calibration layer once on the pooled set — shared `fit_stacked_calibration()` in `models/stacking.py`, used identically by `backtest.py` (WC + continental), `predict_wc2026.py`, and `pipeline.py`, fixing the backtest/production wiring divergence (80/20 vs 85/15). Final XGB + BP are refit on the full training window once T/α are locked (previously XGB only ever saw the first 80–85%). WC backtest average improved **1.0092 → 1.0019** (2014 0.9578→0.9489, 2018 0.9690→0.9821, 2022 1.1008→1.0747). MCMC, when requested, is used only for the final BP fit; folds always use MAP.

**✅ R4 (DONE 2026-06-13, metric-neutral correctness). Joint 4-d Kalman update per match.** Stacks `[att_h, def_h, att_a, def_a]` into one state with a block-diagonal prior; the two Poisson observations (home goals, away goals) are now coupled updates on the joint 4×4 covariance instead of four scalar updates across two per-team 2×2 covariances. This removes the `extra_obs_var` approximation (B5): that hack got the first observation's innovation covariance right but was blind to the cross-covariance it induces for the second — R4 handles both correctly. Marginals are scattered back per team (no global O(n_teams²) cross-team covariance retained); the λ_h/λ_a covariance the roadmap envisaged is ~0 for *unplayed* WC fixtures (no goals observed → no coupling), so the MC resampler is unchanged (R2 already handles cross-match correlation).

**Verdict: kept as a correctness/cleanliness consolidation (like R7), not a metric win.** The per-match correction is real (head-to-head on single matches: up to ~1.2e-2 in the log-goals state, ~1.6e-3 in P, largest on defence states in low-scoring games) but **provably zero-impact on predictions**: WC backtest bit-identical (1.0071 → 1.0071, same EM q=0.1466); per-match proba diff ≤1.77e-7 with odds, and **exactly 0** with odds removed. Root cause is XGBoost's piecewise-constant structure — a 1e-2 feature nudge among 100+ features almost never crosses a tree split threshold. Tested the "odds dilution" hypothesis explicitly (removing the odds feature module): R4's effect went to *zero*, not larger, refuting it. (Side lead, not chased: dropping the XGB odds *feature* slightly improved backtest avg 1.0071 → 0.9995, driven by 2014 — within ±0.1 CI on 144 matches; distinct from the production 0.70 market-1X2 *blend* validated in W17.)

**✅ R5 (DONE 2026-06-12). Margin-of-victory Elo.** Standard World Football Elo multiplies K by a goal-difference factor; a 5–0 and 1–0 currently update identically. Among the best-validated cheap improvements for football Elo — two lines in `_build_cache`.

**R6. Canonical-name registry + ingress validation.** One `canonical(team)` function applied at every data boundary, plus a startup check asserting each of the 48 WC teams has ≥30 training matches and a hit in every lookup table (odds, rankings, transfermarkt, xG). Three of the four worst data bugs in this review are silent-name-miss bugs.

**✅ R7 (DONE 2026-06-12). Shared `dc_outcome_probs(lam_h, lam_a, rho)` helper.** Three places re-expand λ into outcome probs without the Dixon-Coles correction (`wc_context.poisson_proba`, `predict_proba_mcmc`, the MC simulator). One shared helper makes draw handling consistent across MAP, MCMC, post-processing, and simulation.

**R8. Joint submission optimisation.** The third-place pool couples groups: the optimal scoreline in group A depends on tips in groups B–L. After per-group optimisation, re-run `estimate_advance_probs` conditioned on the chosen scorelines and do a second pass.

**R9. Bayesian-shrunk form/SoS.** Replace raw window sums with empirical-Bayes shrinkage toward the population mean weighted by match count — fixes W7 cleanly and helps low-cap teams (Curaçao, Jordan) that the 48-team field is full of.

**R10. Estimate μ and home-advantage inside the Kalman EM loop.** Both are hard-coded (`log(1.3)`, `0.20`); both are estimable in the M-step, removing two hand-set hyperparameters.

---


## Post-fix honest baseline (recorded 2026-06-12, first clean run)

Backtests with `use_tuned_cache=False`, corrected group-stage windows, out-of-sample BP,
weighted T/α on scaled probs. friendly_weight=0.8 (cached; not yet re-tuned honestly).

| Tournament | N | Log-loss | 95% CI | Uniform | Acc | T | α(XGB) |
|---|---|---|---|---|---|---|---|
| WC 2014 | 48 | 0.9597 | [0.861, 1.063] | 1.0986 | 0.500 | 1.363 | 0.571 |
| WC 2018 | 48 | 0.9785 | [0.865, 1.094] | 1.0986 | 0.583 | 1.170 | 0.508 |
| WC 2022 | 48 | 1.0975 | [0.936, 1.260] | 1.0986 | 0.500 | 1.210 | 0.602 |
| **Average** | 144 | **1.0119** | — | 1.0986 | 0.528 | — | — |

Every roadmap change (R1–R10) must beat this average to be kept.

**Baseline progression** (same config: friendly_weight=0.8, `use_tuned_cache=False`):

| Date | Change | WC 2014 | WC 2018 | WC 2022 | Average |
|---|---|---|---|---|---|
| 2026-06-12 | post-fix baseline | 0.9597 | 0.9785 | 1.0975 | 1.0119 |
| 2026-06-12 | + R5 MOV Elo (R7/R2 metric-neutral) | 0.9578 | 0.9690 | 1.1008 | 1.0092 |
| 2026-06-13 | R1 geometric pooling — **rejected** | 0.9630 | 0.9749 | 1.1115 | 1.0165 |
| 2026-06-13 | + R3 stacked calibration | 0.9489 | 0.9821 | 1.0747 | **1.0019** |
| 2026-06-13 | + R4 joint Kalman update — **metric-neutral, kept as correctness** | 0.9487 | 0.9884 | 1.0843 | 1.0071 |

Current bar (WC-only set): **1.0019**. (The 1.0071 R4 row reflects same-data cache drift since the 1.0019 run, not an R4 regression — R4 is bit-identical to its own-data baseline; see R4 note.)

### 2026-06-14 — wider eval adopted + new signals

The 144-match WC-only set has ±0.1 CIs that can't separate signal from noise (R4 moved it by 0.000; R1/R3 by amounts inside the CI). **New reference: the wider backtest** (`backtest.py --continental`) over **7 tournaments / 266 matches** — WC 2014/18/22 + Copa 2021, Euro 2020, AFCON 2022, Asian Cup 2023 — at friendly_weight=0.8.

| Date | Change | Eval | Avg log-loss |
|---|---|---|---|
| 2026-06-14 | wide-eval **baseline** | 266 matches | **0.9512** |
| 2026-06-14 | **+ `rest` feature — kept** | 266 matches | **0.9481** (WC2014 0.9487→0.9378, WC2018→0.9758, Euro→0.9113, AFCON→1.0567; 4 better / 3 worse, net positive) |
| 2026-06-14 | **+ xG-in-Kalman observation — kept** (`USE_XG_OBSERVATION=True`) | 5 covered folds (ablation) | 4/5 folds improved, 1 wash, none worse; gap confederations gain most on WC2022 (CONCACAF −0.033, CAF/AFC −0.016). Within per-fold CI → consistency-based call; CAF/CONCACAF 2026 gain is an extrapolation (StatsBomb coverage postdates the backtests). See `XG_INTEGRATION_PLAN.md`. |

**Current bar (wide set): ~0.9481** (rest kept; xG enabled on top). Both are modest/within-CI but directionally consistent and don't regress — same standard as R5/R7. Not roadmap items (R1–R10); recorded here for the baseline trail. Re-run with `--tune --retune` after the 2026-06-14 feature changes is in progress (re-optimises XGB params + friendly_weight for the new feature set).

Continental + WC2014 baseline (same day, via `backtest.py --years 2014 --continental`).
⚠️ Run with the backtest CLI default `friendly_weight=0.3`, NOT the deployed 0.8 —
note WC 2014 scores 0.9852 here vs 0.9597 under weight 0.8 above. When benchmarking
future changes against the deployed config, pass `--friendly-weight 0.8` explicitly.

| Tournament | N | Log-loss | 95% CI | Acc |
|---|---|---|---|---|
| WC 2014 (w=0.3) | 48 | 0.9852 | [0.885, 1.087] | 0.542 |
| Copa América 2021 | 20 | 0.8800 | [0.713, 1.044] | 0.600 |
| UEFA Euro 2020 | 36 | 0.9073 | [0.741, 1.081] | 0.611 |
| AFCON 2022 | 36 | 1.0401 | [0.853, 1.226] | 0.417 |
| Asian Cup 2023 | 30 | 0.7350 | [0.577, 0.904] | 0.700 |
| **Average** | 170 | **0.9095** | — | 0.565 |

AFCON 2022 is the weak spot (acc 0.417) — consistent with the documented CAF
data gap (no xG, sparse odds coverage for African teams). Note WC 2022 barely
beats uniform — the upset-heavy tournament (Saudi–Argentina, Japan–Germany/Spain);
published bookmaker closing-odds log-loss for that group stage is ~1.04–1.07, so
the headroom over the market is structurally small there.

Production run same day: Optuna re-tuned on the new feature set (LL 0.9978,
depth 3 / lr 0.0057 / 534 trees), T=0.917, ᾱ=0.70 XGB. Tournament headline:
Argentina 17.7% / Spain 13.8% / Brazil 9.8%.

## 6. What was checked and found correct

- No leakage in form/SoS/H2H/squad-strength rolling features — caches snapshot state before processing each date; xG uses `bisect_left` (strictly before).
- Elo update math, zero-sum symmetry, HA applied to expectation only.
- Glicko-2 core formulas (g, E, v, Δ, φ′, μ′) match Glickman, modulo B3/B4/W11.
- Kalman Jacobian, time-update discretisation, RTS gain (primary path), EM M-step.
- Half-life derivation `T = P0/(2q²)` and the [180, 730] clamp.
- Temperature scaling math (log-prob operation is equivalent to logit operation); chronological 80/20 split is genuinely temporal.
- Dixon-Coles τ formulas for all four cells; corrections sum to zero; truncated 11×11 grid renormalised.
- Home advantage gated on `not_neutral` consistently in BP fit/predict.
- `_match_pts` implements the 5/3/2/0 competition scoring exactly, including the draw-GD case.
- `estimate_advance_probs` mirrors the main simulator's group logic (actual-result locking, best-8 thirds).
- `wc2026_market.py` is correctly wired and leakage-safe (context-only, name-normalised both sides, reversed-fixture AH negation handled).
- Closing odds as training features is deliberate and standard; the 80/20 calibration split is time-ordered.
