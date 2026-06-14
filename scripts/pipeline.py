#!/usr/bin/env python3.11
"""WC 2026 prediction pipeline — single command, structured output.

Orchestrates every step in order with timing and progress prints.

Usage:
    python3.11 scripts/pipeline.py                   # full run (50k sims)
    python3.11 scripts/pipeline.py --fetch           # re-fetch live data first
    python3.11 scripts/pipeline.py --sims 200000     # more sims
    python3.11 scripts/pipeline.py --backtest        # also run WC 2018/2022 backtest
    python3.11 scripts/pipeline.py --match "Brazil vs Morocco"
    python3.11 scripts/pipeline.py --mcmc            # MCMC posterior (~5 min extra)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_W = 66  # print width


# ── Printer helpers ────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print()
    print("╔" + "═" * (_W - 2) + "╗")
    print("║" + title.center(_W - 2) + "║")
    print("╚" + "═" * (_W - 2) + "╝")


def _step(n: int, total: int, tag: str, label: str) -> None:
    prefix = f"[{tag} {n}/{total}]"
    print(f"\n{prefix} {label}")


def _ok(msg: str, elapsed: float | None = None) -> None:
    t = f"  ({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"  ✓ {msg}{t}")


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def _tick(t0: float) -> float:
    return time.time() - t0


# ── Step 1: Fetch live data (optional) ────────────────────────────────────────

_FETCHERS = [
    ("API form (last 20 matches per team)",  "fetch_api_form.py",       ROOT / "data" / "api_form_cache.json"),
    ("Transfermarkt squad values",           "fetch_transfermarkt.py",  ROOT / "data" / "transfermarkt_wc2026.json"),
    ("Squad injuries & suspensions",         "fetch_wc2026_injuries.py", ROOT / "data" / "wc2026_injuries_cache.json"),
]


def step_fetch(force: bool) -> None:
    tag = "FETCH"
    total = len(_FETCHERS)
    for i, (label, script, cache_path) in enumerate(_FETCHERS, 1):
        _step(i, total, tag, label)
        t0 = time.time()
        if cache_path.exists() and not force:
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600
            _ok(f"Cache hit ({cache_path.name}, {age_h:.0f}h old)")
        else:
            script_path = ROOT / "scripts" / script
            if not script_path.exists():
                _warn(f"Script not found: {script} — skipping")
                continue
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    _warn(f"Fetch failed (exit {result.returncode}): {result.stderr.strip()[:120]}")
                else:
                    _ok(f"Fetched → {cache_path.name}", _tick(t0))
            except subprocess.TimeoutExpired:
                _warn("Timed out after 120s")
            except Exception as exc:
                _warn(f"Error: {exc}")


# ── Step 2–5: Load, build, train, predict ─────────────────────────────────────

def step_load(from_year: int = 2010, friendly_weight: float = 0.7) -> tuple:
    from football_predictor.data.sources.international_results import fetch_training_data
    from football_predictor.constants import DEFAULT_FEATURE_MODULES

    t0 = time.time()
    all_data = fetch_training_data(from_year=from_year, friendly_weight=friendly_weight)
    # Recorded WC 2026 results (deduped against the source) so Kalman/Elo/BP
    # update from live tournament matches — same path as predict_wc2026.py.
    from scripts.predict_wc2026 import append_actual_results
    all_data = append_actual_results(all_data, quiet=True)
    # Calibrated xG for the Kalman observation (constants.USE_XG_OBSERVATION).
    from football_predictor import constants as _c
    if _c.USE_XG_OBSERVATION:
        from football_predictor.data.xg_attach import attach_calibrated_xg
        all_data = attach_calibrated_xg(all_data)
    comp_mask = all_data["tournament"].str.lower().str.contains(
        "qualif|world cup|copa|euro|nations|africa|asian|gold cup", na=False
    )
    train_df = all_data[comp_mask].reset_index(drop=True)
    _ok(
        f"{len(all_data):,} total matches | {len(train_df):,} competitive | "
        f"{len(DEFAULT_FEATURE_MODULES)} feature modules",
        _tick(t0),
    )
    return all_data, train_df


def step_features(train_df, all_data) -> tuple:
    import pandas as pd
    from football_predictor.data.pipeline import build_feature_matrix, prune_correlated_features
    from football_predictor.constants import DEFAULT_FEATURE_MODULES

    t0 = time.time()
    X, y = build_feature_matrix(train_df, DEFAULT_FEATURE_MODULES, context=all_data)
    n_raw = X.shape[1]
    X, dropped = prune_correlated_features(X, threshold=0.95)
    sw = train_df["match_weight"].values
    prune_note = f" → {X.shape[1]} after pruning {len(dropped)} correlated" if dropped else ""
    _ok(f"{n_raw} features{prune_note} | {X.shape[0]:,} rows", _tick(t0))
    return X, y, sw


def step_train(X, y, sw, train_df, use_mcmc: bool, mcmc_draws: int, mcmc_tune: int,
               tune_xgb: bool = False, tune_trials: int = 60) -> tuple:
    import pandas as pd
    from football_predictor.models.gradient_boost import GradientBoostModel
    from football_predictor.models.bayesian_poisson import BayesianPoissonModel
    from football_predictor.data.wc2026 import normalise

    cut = int(len(X) * 0.8)
    cal_X, cal_y = X.iloc[cut:], y.iloc[cut:]

    t0 = time.time()
    from football_predictor.models.gradient_boost import _TUNED_PARAMS_PATH
    xgb = GradientBoostModel()
    if tune_xgb or not _TUNED_PARAMS_PATH.exists():
        # Auto-tune on first run (no cache); force re-tune when --tune is passed.
        dates = pd.to_datetime(train_df["date"])
        val_splits = []
        for wc_year in [2018, 2022]:
            wc_mask = (train_df["tournament"] == "FIFA World Cup") & (dates.dt.year == wc_year)
            if wc_mask.sum() >= 16:
                pre_mask = dates < pd.Timestamp(f"{wc_year}-05-01")
                val_splits.append((
                    X[pre_mask], y[pre_mask], sw[pre_mask.values],
                    X[wc_mask], y[wc_mask],
                ))
        if not val_splits:
            val_splits = [(X.iloc[:cut], y.iloc[:cut], sw[:cut], cal_X, cal_y)]
        label = "re-tuning" if tune_xgb else "first run — tuning"
        print(f"  [XGB] {label} ({tune_trials} Optuna trials, {len(val_splits)} WC val folds)...")
        xgb.tune_hyperparameters(val_splits, n_trials=tune_trials)
    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    from football_predictor.models.stacking import fit_stacked_calibration
    q_tuned = KalmanStrengthFeatures.get_last_tuned_q()
    hl = BayesianPoissonModel.half_life_from_q(q_tuned)
    print(f"  BayesPoisson half-life: {hl}d  (derived from Kalman q={q_tuned:.4f}/yr)")

    def _bp_proba_for(model, matches):
        rows = []
        for _, m in matches.iterrows():
            p = model.predict_proba(normalise(m["home_team"]), normalise(m["away_team"]),
                                    neutral=bool(m.get("neutral", True)))
            rows.append(p)
        return pd.DataFrame(rows, columns=["home_win", "draw", "away_win"])

    def _fit_xgb(X_tr, y_tr, sw_tr):
        m = GradientBoostModel()
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        return m

    def _fit_bp(matches_tr):
        # Folds always use MAP — MCMC only for the final model (speed).
        b = BayesianPoissonModel(half_life_days=hl)
        b.fit(matches_tr)
        return b

    # Stacked calibration (R3): T + ensemble α on pooled out-of-time folds
    temp_cal, ensemble = fit_stacked_calibration(
        X, y, train_df.reset_index(drop=True), sw,
        _fit_xgb, _fit_bp,
        lambda b, m: _bp_proba_for(b, m.reset_index(drop=True)),
    )
    print(f"  Stacked calibration done  T={temp_cal.temperature:.3f} ({_tick(t0):.1f}s)")
    print(f"  Ensemble  ᾱ={ensemble.xgb_weight:.2f} XGB + {ensemble.bp_weight:.2f} BP (context-adaptive)")

    # Final models on the full training window now T and α are locked
    t2 = time.time()
    xgb.fit(X, y, sample_weight=sw)
    bp = BayesianPoissonModel(half_life_days=hl)
    if use_mcmc:
        print(f"  MCMC sampling ({mcmc_draws} draws + {mcmc_tune} tune) ...")
        bp.fit_mcmc(train_df, draws=mcmc_draws, tune=mcmc_tune)
    else:
        bp.fit(train_df)
    bp.fit_rho(train_df)
    print(f"  Final XGB + BP fit + DC ρ estimated ({_tick(t2):.1f}s)")

    def _bp_proba(matches):
        return _bp_proba_for(bp, matches)

    return xgb, temp_cal, bp, ensemble, _bp_proba


def step_predict(xgb, temp_cal, bp, ensemble, all_data, train_df, _bp_proba_fn) -> tuple:
    import numpy as np
    import pandas as pd
    from football_predictor.data.pipeline import build_feature_matrix
    from football_predictor.data.wc2026 import GROUPS, GROUP_STAGE_SCHEDULE, normalise
    from football_predictor.constants import DEFAULT_FEATURE_MODULES

    fixtures = [
        dict(f, neutral=True, tournament="FIFA World Cup",
             match_weight=1.5, home_goals=0, away_goals=0)
        for f in GROUP_STAGE_SCHEDULE
    ]
    if "date" not in pd.DataFrame(fixtures).columns:
        for f in fixtures:
            f["date"] = "2026-06-15"

    from football_predictor.models.wc_context import build_wc_context, apply_to_match

    match_df = pd.DataFrame(fixtures)
    t0 = time.time()
    X_gs, _ = build_feature_matrix(match_df, DEFAULT_FEATURE_MODULES, context=all_data)
    xgb_proba = temp_cal.transform(xgb.predict_proba(X_gs))
    ctx_gs = build_wc_context(match_df, all_data)

    results = []
    for i, f in enumerate(fixtures):
        home, away = f["home_team"], f["away_team"]
        bp_p = bp.predict_proba(normalise(home), normalise(away), neutral=True)
        bp_row = pd.DataFrame([bp_p], columns=["home_win", "draw", "away_win"])
        xgb_row = xgb_proba.iloc[[i]].reset_index(drop=True)
        ens = ensemble.predict_proba(xgb_row, bp_row, context_X=X_gs.iloc[[i]].reset_index(drop=True))
        lam_h, lam_a = bp.get_lambdas(normalise(home), normalise(away), neutral=True)
        ctx_row = ctx_gs.iloc[i].to_dict() if i < len(ctx_gs) else {}
        p_h, p_d, p_a, lam_h_adj, lam_a_adj = apply_to_match(
            lam_h, lam_a,
            float(ens["home_win"].iloc[0]),
            float(ens["draw"].iloc[0]),
            float(ens["away_win"].iloc[0]),
            ctx_row,
            apply_venue=True,
            home_team=home,
            away_team=away,
            rho=bp._rho,
        )
        results.append({
            **f,
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "bp_lam_home": lam_h_adj, "bp_lam_away": lam_a_adj,
        })
    _ok(f"72 group stage matches predicted", _tick(t0))
    return results


def step_simulate(match_data: list[dict], n_sims: int, xgb, temp_cal, bp, ensemble, all_data) -> dict:
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from football_predictor.data.pipeline import build_feature_matrix
    from football_predictor.data.wc2026 import GROUPS, normalise
    from football_predictor.constants import DEFAULT_FEATURE_MODULES

    all_teams = [t for grp in GROUPS.values() for t in grp]
    t0 = time.time()
    pair_rows = [
        {"home_team": h, "away_team": a, "date": "2026-07-01",
         "neutral": True, "tournament": "FIFA World Cup",
         "match_weight": 1.5, "home_goals": 0, "away_goals": 0}
        for h in all_teams for a in all_teams if h != a
    ]
    from football_predictor.models.wc_context import build_wc_context, quality_nudge, absence_adjust

    pair_df = pd.DataFrame(pair_rows)
    X_pairs, _ = build_feature_matrix(pair_df, DEFAULT_FEATURE_MODULES, context=all_data)
    ctx_pairs = build_wc_context(pair_df, all_data)
    xgb_pairs = temp_cal.transform(xgb.predict_proba(X_pairs))
    bp_pairs = pd.DataFrame(
        [bp.predict_proba(normalise(r["home_team"]), normalise(r["away_team"]), neutral=True)
         for _, r in pair_df.iterrows()],
        columns=["home_win", "draw", "away_win"],
    )
    ens_pairs = ensemble.predict_proba(xgb_pairs.reset_index(drop=True), bp_pairs, context_X=X_pairs.reset_index(drop=True))
    prob_cache: dict[tuple, tuple] = {}
    for i, row in pair_df.iterrows():
        p = ens_pairs.iloc[i]
        ctx_row = ctx_pairs.iloc[i].to_dict() if i < len(ctx_pairs) else {}
        p_h, p_d, p_a = quality_nudge(
            float(p["home_win"]), float(p["draw"]), float(p["away_win"]), ctx_row
        )
        # Injury penalty applies in knockouts too (matches predict_wc2026.py).
        p_h, p_d, p_a = absence_adjust(p_h, p_d, p_a, row["home_team"], row["away_team"])
        prob_cache[(row["home_team"], row["away_team"])] = (p_h, p_d, p_a)

    # Symmetrise both orientations — XGB features aren't slot-symmetric and
    # all KO venues are neutral (matches predict_wc2026.py).
    for home, away in list(prob_cache.keys()):
        if home < away and (away, home) in prob_cache:
            f, r = prob_cache[(home, away)], prob_cache[(away, home)]
            p_h = (f[0] + r[2]) / 2.0
            p_d = (f[1] + r[1]) / 2.0
            p_a = (f[2] + r[0]) / 2.0
            s = p_h + p_d + p_a
            prob_cache[(home, away)] = (p_h / s, p_d / s, p_a / s)
            prob_cache[(away, home)] = (p_a / s, p_d / s, p_h / s)
    _ok(f"{len(prob_cache):,} pair probabilities pre-computed (symmetrised)", _tick(t0))

    # Monte Carlo
    from scripts.predict_wc2026 import simulate_tournament, build_team_sigmas
    t1 = time.time()
    round_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def ko_predictor(home: str, away: str) -> tuple:
        return prob_cache.get((home, away), (0.4, 0.2, 0.4))

    # Persistent team strength: one deviation per team per simulation (R2)
    team_sigmas = build_team_sigmas(match_data)

    for i in range(n_sims):
        if i > 0 and i % 10_000 == 0:
            elapsed = _tick(t1)
            rate = i / elapsed
            eta = (n_sims - i) / rate
            print(f"  ... {i:,}/{n_sims:,} sims  ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)
        reached = simulate_tournament(match_data, ko_predictor, team_sigmas=team_sigmas)
        for team, rnd in reached.items():
            round_counts[team][rnd] += 1

    cum_rounds = ["r32", "r16", "qf", "sf", "final", "winner"]
    cum_counts: dict[str, dict[str, int]] = {}
    for team, cnts in round_counts.items():
        cum: dict[str, int] = {}
        running = 0
        for r in reversed(cum_rounds):
            running += cnts.get(r, 0)
            cum[r] = running
        cum_counts[team] = cum

    _ok(f"{n_sims:,} simulations done", _tick(t1))
    return cum_counts


# ── Step: Backtest ─────────────────────────────────────────────────────────────

def step_backtest(years: list[int], friendly_weight: float = 0.7) -> None:
    # Evaluate the DEPLOYED configuration — previously this ran with the
    # default weight while production trained with the cached one.
    from scripts.backtest import run_backtest
    for year in years:
        run_backtest(year, include_shap=False, friendly_weight=friendly_weight)


# ── Output ─────────────────────────────────────────────────────────────────────

def _save_raw_predictions(match_data: list[dict]) -> Path:
    """Write output/output_raw.csv — direct model output, no points optimisation.

    Columns (all oriented to the template's team1/team2):
      score1/score2          — displayed score: actual if played, else floor(λ)
      pred_score1/pred_score2— model's most-likely score floor(λ), NEVER the actual
                               (so played matches still expose the genuine prediction)
      sample_score1/2        — one random draw from the joint Poisson (seeded per
                               match): a "realistic-looking" scoreline. Higher
                               variance, lower accuracy than the most-likely score —
                               for display only, not a better prediction.
      p_home/draw/away       — displayed probs: locked 1/0/0 if played
      pred_p_home/draw/away  — model's genuine pre-lock probabilities
      lam_home/lam_away      — model expected goals (the true continuous prediction)
    """
    import numpy as np

    template_path = ROOT / "templates" / "output_template.csv"
    with open(template_path, newline="") as f:
        template_rows = {
            int(r["match_id"]): r
            for r in csv.DictReader(f)
        }

    _TEMPLATE_TO_SCHEDULE = {
        "Turkiye":    "Türkiye",
        "Cape Verde": "Cabo Verde",
        "Iran":       "IR Iran",
        "Curacao":    "Curaçao",
    }

    fieldnames = ["match_id", "group", "team1", "team2",
                  "score1", "score2", "pred_score1", "pred_score2",
                  "sample_score1", "sample_score2",
                  "p_home", "p_draw", "p_away",
                  "pred_p_home", "pred_p_draw", "pred_p_away",
                  # mdl_p_* = un-blended model's-eye probs (before the 0.70 market 1X2 blend)
                  "mdl_p_home", "mdl_p_draw", "mdl_p_away",
                  "lam_home", "lam_away"]

    lookup: dict[frozenset, dict] = {
        frozenset([m["home_team"], m["away_team"]]): m
        for m in match_data
    }

    rows = []
    for mid, tr in sorted(template_rows.items()):
        t1 = _TEMPLATE_TO_SCHEDULE.get(tr["team1"], tr["team1"])
        t2 = _TEMPLATE_TO_SCHEDULE.get(tr["team2"], tr["team2"])
        key = frozenset([t1, t2])
        m = lookup.get(key)
        if m is None:
            rows.append({k: "" for k in fieldnames}
                        | {"match_id": mid, "group": tr["group"],
                           "team1": tr["team1"], "team2": tr["team2"]})
            continue

        lam_h = m.get("bp_lam_home", 1.0)
        lam_a = m.get("bp_lam_away", 1.0)

        # Model's most-likely score — the mode of each team's Poisson, ALWAYS the
        # genuine prediction even for played matches (never the recorded result).
        pred_h, pred_a = math.floor(lam_h), math.floor(lam_a)

        # One realistic random draw from the joint Poisson, seeded per match so
        # the file is reproducible. Shows the spread real football has; not used
        # for scoring and not more accurate than the most-likely score.
        rng = np.random.default_rng(20260000 + mid)
        samp_h, samp_a = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))

        if m.get("played"):
            s_h = int(m["actual_home_goals"])
            s_a = int(m["actual_away_goals"])
        else:
            s_h, s_a = pred_h, pred_a

        p_h, p_d, p_a = m["p_home"], m["p_draw"], m["p_away"]
        pp_h = m.get("pred_p_home", p_h)
        pp_d = m.get("pred_p_draw", p_d)
        pp_a = m.get("pred_p_away", p_a)
        # Un-blended model's-eye probs (fall back to the blended genuine probs
        # for older match dicts without the pre-blend keys).
        mp_h = m.get("p_home_preblend", pp_h)
        mp_d = m.get("p_draw_preblend", pp_d)
        mp_a = m.get("p_away_preblend", pp_a)

        if m["home_team"] != t1:
            # Orient every home/away-paired field to the template's team1/team2.
            s_h, s_a = s_a, s_h
            pred_h, pred_a = pred_a, pred_h
            samp_h, samp_a = samp_a, samp_h
            p_h, p_a = p_a, p_h
            pp_h, pp_a = pp_a, pp_h
            mp_h, mp_a = mp_a, mp_h
            lam_h, lam_a = lam_a, lam_h

        rows.append({
            "match_id": mid, "group": tr["group"],
            "team1": tr["team1"], "team2": tr["team2"],
            "score1": s_h, "score2": s_a,
            "pred_score1": pred_h, "pred_score2": pred_a,
            "sample_score1": samp_h, "sample_score2": samp_a,
            "p_home": f"{p_h:.3f}", "p_draw": f"{p_d:.3f}", "p_away": f"{p_a:.3f}",
            "pred_p_home": f"{pp_h:.3f}", "pred_p_draw": f"{pp_d:.3f}",
            "pred_p_away": f"{pp_a:.3f}",
            "mdl_p_home": f"{mp_h:.3f}", "mdl_p_draw": f"{mp_d:.3f}",
            "mdl_p_away": f"{mp_a:.3f}",
            "lam_home": f"{lam_h:.3f}", "lam_away": f"{lam_a:.3f}",
        })

    out_path = OUTPUT_DIR / "output_raw.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def _save_output(match_data, cum_counts, n_sims) -> Path:
    import io
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf

    from scripts.predict_wc2026 import (
        print_group_predictions,
        print_expected_group_standings,
        print_tournament_table,
    )
    print_group_predictions(match_data)
    print_expected_group_standings(match_data)
    print_tournament_table(cum_counts, n_sims)

    sys.stdout = old_stdout
    output = buf.getvalue()

    # Also print to terminal
    print(output)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = OUTPUT_DIR / f"wc2026_predictions_{ts}.txt"
    out_path.write_text(output)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WC 2026 prediction pipeline")
    p.add_argument("--fetch",    action="store_true", help="Re-fetch live data (API form, TM, injuries)")
    p.add_argument("--no-fetch", action="store_true", help="Skip fetch step entirely")
    p.add_argument("--sims",     type=int, default=50_000)
    p.add_argument("--backtest", action="store_true", help="Also run WC 2014/2018/2022 backtest after predictions")
    p.add_argument("--match",    type=str, default=None, help="Single match card, e.g. 'Brazil vs Morocco'")
    p.add_argument("--mcmc",     action="store_true", help="Use MCMC posterior for BayesPoisson (~5 min extra)")
    p.add_argument("--mcmc-draws", type=int, default=500)
    p.add_argument("--mcmc-tune",  type=int, default=250)
    p.add_argument("--from-year",  type=int, default=2010, help="Earliest year of training data")
    p.add_argument("--friendly-weight", type=float, default=None,
                   help="Override friendly match weight (skips cache + re-tune)")
    p.add_argument("--retune", action="store_true",
                   help="Re-run friendly weight grid search even if a cached value exists")
    p.add_argument("--tune",     action="store_true",
                   help="Tune XGBoost hyperparameters via Optuna before training (requires: pip install optuna)")
    p.add_argument("--tune-trials", type=int, default=60, help="Number of Optuna trials (default 60)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the Monte Carlo simulation (reproducible runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()

    _header("WC 2026 FOOTBALL PREDICTOR — PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  sims={args.sims:,}  |  mcmc={args.mcmc}")

    # ── FRIENDLY WEIGHT ────────────────────────────────────────────────────────
    if args.friendly_weight is not None:
        friendly_weight = args.friendly_weight
        print(f"\n  friendly_weight={friendly_weight} (CLI override)")
    else:
        _tp = ROOT / "data" / "tuned_params.json"
        cached = None
        try:
            cached = json.loads(_tp.read_text()).get("friendly_weight") if _tp.exists() else None
        except Exception:
            cached = None
        if args.retune or cached is None:
            # Grid search (auto-tunes on first run / forced via --retune);
            # tune_friendly_weight persists the result to the cache.
            label = "re-tuning" if args.retune else "first run — tuning"
            print(f"\n  friendly_weight: {label} (grid search over WC backtests, slow)...")
            from scripts.backtest import tune_friendly_weight
            friendly_weight = tune_friendly_weight([2014, 2018, 2022])
        else:
            friendly_weight = cached
            print(f"\n  friendly_weight={friendly_weight} (from cache)")

    # ── FETCH ──────────────────────────────────────────────────────────────────
    if not args.no_fetch:
        print(f"\n{'─'*_W}")
        print("  STEP 1 — FETCH LIVE DATA")
        print(f"{'─'*_W}")
        step_fetch(force=args.fetch)
    else:
        print("\n  STEP 1 — FETCH: skipped (--no-fetch)")

    # ── LOAD ───────────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print("  STEP 2 — LOAD HISTORICAL DATA")
    print(f"{'─'*_W}")
    _step(1, 1, "DATA", f"Loading results from {args.from_year}...")
    all_data, train_df = step_load(from_year=args.from_year, friendly_weight=friendly_weight)

    # ── FEATURES ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print("  STEP 3 — BUILD FEATURE MATRIX")
    print(f"{'─'*_W}")
    _step(1, 1, "FEAT", "Running all feature modules...")
    X, y, sw = step_features(train_df, all_data)

    # ── TRAIN ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print("  STEP 4 — TRAIN MODELS")
    print(f"{'─'*_W}")
    _step(1, 1, "TRAIN", "XGBoost + BayesPoisson + ensemble calibration...")
    xgb, temp_cal, bp, ensemble, _bp_proba_fn = step_train(
        X, y, sw, train_df, args.mcmc, args.mcmc_draws, args.mcmc_tune,
        tune_xgb=args.tune, tune_trials=args.tune_trials,
    )

    # ── SINGLE MATCH ───────────────────────────────────────────────────────────
    if args.match:
        print(f"\n{'─'*_W}")
        print("  MATCH PREDICTION")
        print(f"{'─'*_W}")
        from scripts.predict_wc2026 import predict_single_match
        predict_single_match(args.match, xgb, temp_cal, bp, ensemble, all_data)
        print(f"\nTotal: {_tick(t_total):.0f}s")
        return

    # ── PREDICT ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print("  STEP 5 — GROUP STAGE PREDICTIONS")
    print(f"{'─'*_W}")
    _step(1, 1, "PRED", "Computing match probabilities (72 fixtures)...")
    match_data = step_predict(xgb, temp_cal, bp, ensemble, all_data, train_df, _bp_proba_fn)

    # Merge actual results
    from scripts.predict_wc2026 import load_actual_results, merge_actual_results
    actual = load_actual_results()
    if actual:
        match_data = merge_actual_results(match_data, actual)
        _ok(f"{len(actual)} actual WC 2026 result(s) merged")

    # ── SIMULATE ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print(f"  STEP 6 — MONTE CARLO SIMULATION ({args.sims:,} runs)")
    print(f"{'─'*_W}")
    _step(1, 1, "SIM", f"Running {args.sims:,} tournament simulations (seed={args.seed})...")
    import numpy as _np
    _np.random.seed(args.seed)  # reproducible Monte Carlo (global stream)
    cum_counts = step_simulate(match_data, args.sims, xgb, temp_cal, bp, ensemble, all_data)

    # ── OUTPUT ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print("  STEP 7 — OUTPUT")
    print(f"{'─'*_W}")
    out_path = _save_output(match_data, cum_counts, args.sims)
    _ok(f"Saved → {out_path.name}")
    raw_path = _save_raw_predictions(match_data)
    _ok(f"Saved → {raw_path.name}  (raw Poisson scores, no optimisation)")

    # ── BACKTEST ───────────────────────────────────────────────────────────────
    if args.backtest:
        print(f"\n{'─'*_W}")
        print("  STEP 8 — BACKTEST (WC 2014 + 2018 + 2022)")
        print(f"{'─'*_W}")
        step_backtest([2014, 2018, 2022], friendly_weight=friendly_weight)

    # ── DONE ───────────────────────────────────────────────────────────────────
    elapsed = _tick(t_total)
    print()
    print("╔" + "═" * (_W - 2) + "╗")
    print("║" + f"  DONE — total {elapsed:.0f}s  |  output/{out_path.name}".ljust(_W - 2) + "║")
    print("╚" + "═" * (_W - 2) + "╝")
    print()


if __name__ == "__main__":
    main()
