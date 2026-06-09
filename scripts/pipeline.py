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
import importlib
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
    ("API form (last 10 matches per team)",  "fetch_api_form.py",       ROOT / "data" / "api_form_cache.json"),
    ("Transfermarkt squad values",           "fetch_transfermarkt.py",  ROOT / "data" / "transfermarkt_wc2026.json"),
    ("Squad injuries & suspensions",         "fetch_wc2026_injuries.py", ROOT / "data" / "injuries_cache.json"),
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

def step_load(from_year: int = 2010) -> tuple:
    from football_predictor.data.sources.international_results import fetch_training_data
    from football_predictor.constants import DEFAULT_FEATURE_MODULES

    t0 = time.time()
    all_data = fetch_training_data(from_year=from_year)
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
    from football_predictor.data.pipeline import build_feature_matrix
    from football_predictor.constants import DEFAULT_FEATURE_MODULES

    t0 = time.time()
    X, y = build_feature_matrix(train_df, DEFAULT_FEATURE_MODULES, context=all_data)
    sw = train_df["match_weight"].values
    _ok(f"{X.shape[1]} features | {X.shape[0]:,} rows", _tick(t0))
    return X, y, sw


def step_train(X, y, sw, train_df, use_mcmc: bool, mcmc_draws: int, mcmc_tune: int) -> tuple:
    import pandas as pd
    from football_predictor.models.gradient_boost import GradientBoostModel
    from football_predictor.models.bayesian_poisson import BayesianPoissonModel
    from football_predictor.models.calibration import TemperatureScaling
    from football_predictor.models.ensemble import EnsembleModel
    from football_predictor.data.wc2026 import normalise

    cut = int(len(X) * 0.8)
    cal_X, cal_y = X.iloc[cut:], y.iloc[cut:]

    t0 = time.time()
    xgb = GradientBoostModel()
    xgb.fit(X.iloc[:cut], y.iloc[:cut], sample_weight=sw[:cut])
    xgb_proba_cal = xgb.predict_proba(cal_X)
    temp_cal = TemperatureScaling()
    temp_cal.fit(xgb_proba_cal, cal_y)
    print(f"  XGB done  T={temp_cal.temperature:.3f} ({_tick(t0):.1f}s)")

    t1 = time.time()
    bp = BayesianPoissonModel()
    if use_mcmc:
        print(f"  MCMC sampling ({mcmc_draws} draws + {mcmc_tune} tune) ...")
        bp.fit_mcmc(train_df.iloc[:cut], draws=mcmc_draws, tune=mcmc_tune)
    else:
        bp.fit(train_df.iloc[:cut])
    print(f"  BayesPoisson done ({_tick(t1):.1f}s)")

    # Ensemble: calibrate on held-out 20%
    def _bp_proba(matches):
        rows = []
        for _, m in matches.iterrows():
            p = bp.predict_proba(normalise(m["home_team"]), normalise(m["away_team"]),
                                 neutral=bool(m.get("neutral", True)))
            rows.append(p)
        return pd.DataFrame(rows, columns=["home_win", "draw", "away_win"])

    bp_proba_cal = _bp_proba(train_df.iloc[cut:])
    ensemble = EnsembleModel()
    ensemble.fit(xgb_proba_cal, bp_proba_cal, cal_y)
    print(f"  Ensemble  α={ensemble.xgb_weight:.2f} XGB + {ensemble.bp_weight:.2f} BP")

    # Refit BP on full data now ensemble weight is locked
    t2 = time.time()
    if use_mcmc:
        bp.fit_mcmc(train_df, draws=mcmc_draws, tune=mcmc_tune)
    else:
        bp.fit(train_df)
    print(f"  BP refit on full data ({_tick(t2):.1f}s)")

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

    match_df = pd.DataFrame(fixtures)
    t0 = time.time()
    X_gs, _ = build_feature_matrix(match_df, DEFAULT_FEATURE_MODULES, context=all_data)
    xgb_proba = temp_cal.transform(xgb.predict_proba(X_gs))

    results = []
    for i, f in enumerate(fixtures):
        home, away = f["home_team"], f["away_team"]
        bp_p = bp.predict_proba(normalise(home), normalise(away), neutral=True)
        bp_row = pd.DataFrame([bp_p], columns=["home_win", "draw", "away_win"])
        xgb_row = xgb_proba.iloc[[i]].reset_index(drop=True)
        ens = ensemble.predict_proba(xgb_row, bp_row)
        lam_h, lam_a = bp.get_lambdas(normalise(home), normalise(away), neutral=True)
        results.append({
            **f,
            "p_home": float(ens["home_win"].iloc[0]),
            "p_draw": float(ens["draw"].iloc[0]),
            "p_away": float(ens["away_win"].iloc[0]),
            "bp_lam_home": lam_h,
            "bp_lam_away": lam_a,
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
    pair_df = pd.DataFrame(pair_rows)
    X_pairs, _ = build_feature_matrix(pair_df, DEFAULT_FEATURE_MODULES, context=all_data)
    xgb_pairs = temp_cal.transform(xgb.predict_proba(X_pairs))
    bp_pairs = pd.DataFrame(
        [bp.predict_proba(normalise(r["home_team"]), normalise(r["away_team"]), neutral=True)
         for _, r in pair_df.iterrows()],
        columns=["home_win", "draw", "away_win"],
    )
    ens_pairs = ensemble.predict_proba(xgb_pairs.reset_index(drop=True), bp_pairs)
    prob_cache: dict[tuple, tuple] = {}
    for i, row in pair_df.iterrows():
        p = ens_pairs.iloc[i]
        prob_cache[(row["home_team"], row["away_team"])] = (
            float(p["home_win"]), float(p["draw"]), float(p["away_win"])
        )
    _ok(f"{len(prob_cache):,} pair probabilities pre-computed", _tick(t0))

    # Monte Carlo
    from scripts.predict_wc2026 import simulate_tournament
    t1 = time.time()
    round_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def ko_predictor(home: str, away: str) -> tuple:
        return prob_cache.get((home, away), (0.4, 0.2, 0.4))

    for i in range(n_sims):
        if i > 0 and i % 10_000 == 0:
            elapsed = _tick(t1)
            rate = i / elapsed
            eta = (n_sims - i) / rate
            print(f"  ... {i:,}/{n_sims:,} sims  ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)
        reached = simulate_tournament(match_data, ko_predictor)
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

def step_backtest(years: list[int]) -> None:
    from scripts.backtest import run_backtest
    for year in years:
        run_backtest(year, include_shap=False)


# ── Output ─────────────────────────────────────────────────────────────────────

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
    p.add_argument("--backtest", action="store_true", help="Also run WC 2018/2022 backtest after predictions")
    p.add_argument("--match",    type=str, default=None, help="Single match card, e.g. 'Brazil vs Morocco'")
    p.add_argument("--mcmc",     action="store_true", help="Use MCMC posterior for BayesPoisson (~5 min extra)")
    p.add_argument("--mcmc-draws", type=int, default=500)
    p.add_argument("--mcmc-tune",  type=int, default=250)
    p.add_argument("--from-year",  type=int, default=2010, help="Earliest year of training data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()

    _header("WC 2026 FOOTBALL PREDICTOR — PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  sims={args.sims:,}  |  mcmc={args.mcmc}")

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
    all_data, train_df = step_load(from_year=args.from_year)

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
        X, y, sw, train_df, args.mcmc, args.mcmc_draws, args.mcmc_tune
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
    _step(1, 1, "SIM", f"Running {args.sims:,} tournament simulations...")
    cum_counts = step_simulate(match_data, args.sims, xgb, temp_cal, bp, ensemble, all_data)

    # ── OUTPUT ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print("  STEP 7 — OUTPUT")
    print(f"{'─'*_W}")
    out_path = _save_output(match_data, cum_counts, args.sims)
    _ok(f"Saved → {out_path.name}")

    # ── BACKTEST ───────────────────────────────────────────────────────────────
    if args.backtest:
        print(f"\n{'─'*_W}")
        print("  STEP 8 — BACKTEST (WC 2018 + 2022)")
        print(f"{'─'*_W}")
        step_backtest([2018, 2022])

    # ── DONE ───────────────────────────────────────────────────────────────────
    elapsed = _tick(t_total)
    print()
    print("╔" + "═" * (_W - 2) + "╗")
    print("║" + f"  DONE — total {elapsed:.0f}s  |  output/{out_path.name}".ljust(_W - 2) + "║")
    print("╚" + "═" * (_W - 2) + "╝")
    print()


if __name__ == "__main__":
    main()
