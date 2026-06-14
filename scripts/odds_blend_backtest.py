"""Backtest blending market 1X2 probabilities into the model's final probabilities.

Resolves W17 (ARCHITECTURE_REVIEW.md) with evidence instead of judgment: trains
the honest backtest stack per year (WC 2018/2022 — the years with closing 1X2
odds), then post-hoc blends bookmaker-implied probabilities at weight w:

  linear:    p = (1-w)·p_model + w·p_market
  log-odds:  p ∝ p_model^(1-w) · p_market^w   (geometric pooling)

Matches without odds keep the model probabilities — the same fallback the
deployed blend would use. Reports per-year and average log-loss across the
sweep, plus a paired bootstrap CI on the best weight vs the model baseline.

Usage:
    python3.11 scripts/odds_blend_backtest.py
    python3.11 scripts/odds_blend_backtest.py --years 2018 2022 --friendly-weight 0.8
"""
import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest import run_backtest, _log_loss
from football_predictor.data.sources.football_data_co_uk import build_odds_lookup
from football_predictor.data.wc2026 import normalise

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
_EPS = 1e-10


def _tuned_friendly_weight() -> float:
    tp = ROOT / "data" / "tuned_params.json"
    try:
        return float(json.loads(tp.read_text()).get("friendly_weight", 0.3))
    except Exception:
        return 0.3


def market_probs(test_df: pd.DataFrame) -> np.ndarray:
    """(n, 3) market-implied [home, draw, away]; NaN rows where no odds found."""
    lookup = build_odds_lookup()
    out = np.full((len(test_df), 3), np.nan)
    for i, (_, r) in enumerate(test_df.iterrows()):
        home = normalise(str(r["home_team"]))
        away = normalise(str(r["away_team"]))
        date_str = str(pd.Timestamp(r["date"]).date())
        entry = lookup.get((home, away, date_str))
        if entry is None:
            rev = lookup.get((away, home, date_str))
            if rev:
                entry = {"ph": rev["pa"], "pd": rev["pd"], "pa": rev["ph"]}
        if entry is not None:
            out[i] = [entry["ph"], entry["pd"], entry["pa"]]
    return out


def blend(p_model: np.ndarray, p_market: np.ndarray, w: float, mode: str) -> np.ndarray:
    """Blend model and market probs at weight w; NaN market rows keep the model."""
    covered = ~np.isnan(p_market[:, 0])
    out = p_model.copy()
    if w == 0.0 or not covered.any():
        return out
    pm = p_model[covered]
    pk = p_market[covered]
    if mode == "linear":
        mixed = (1 - w) * pm + w * pk
    else:  # log-odds (geometric pooling)
        mixed = np.exp((1 - w) * np.log(np.maximum(pm, _EPS))
                       + w * np.log(np.maximum(pk, _EPS)))
    out[covered] = mixed / mixed.sum(axis=1, keepdims=True)
    return out


def paired_bootstrap_diff(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
                          n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Mean log-loss(a) − log-loss(b) with bootstrap 95% CI. Negative = a better."""
    la = -np.log(np.maximum(p_a[np.arange(len(y)), y], _EPS))
    lb = -np.log(np.maximum(p_b[np.arange(len(y)), y], _EPS))
    d = la - lb
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2022])
    parser.add_argument("--friendly-weight", type=float, default=None,
                        help="Default: tuned value from data/tuned_params.json")
    args = parser.parse_args()

    fw = args.friendly_weight if args.friendly_weight is not None else _tuned_friendly_weight()
    print(f"friendly_weight = {fw}")

    results = {}  # year -> (p_model, p_market, y_true)
    for year in args.years:
        m = run_backtest(year, include_shap=False, label="oddsblend", friendly_weight=fw)
        if not m:
            continue
        p_market = market_probs(m["_test_df"])
        results[year] = (m["_proba"], p_market, m["_y_true"])

    print(f"\n{'═'*72}")
    print("  MARKET 1X2 BLEND SWEEP — log-loss by blend weight w")
    print(f"{'═'*72}")

    for year, (p_model, p_market, y) in results.items():
        n_cov = int((~np.isnan(p_market[:, 0])).sum())
        print(f"\n  WC {year}  ({len(y)} matches, odds coverage {n_cov}/{len(y)})")
        cov = ~np.isnan(p_market[:, 0])
        if cov.any():
            ll_model_cov = _log_loss(p_model[cov], y[cov])
            pk = p_market[cov] / p_market[cov].sum(axis=1, keepdims=True)
            ll_market_cov = _log_loss(pk, y[cov])
            print(f"  covered subset: model {ll_model_cov:.4f}  vs  market-only {ll_market_cov:.4f}")
        print(f"  {'w':>6}  {'linear':>8}  {'log-odds':>8}")
        for w in WEIGHTS:
            ll_lin = _log_loss(blend(p_model, p_market, w, "linear"), y)
            ll_geo = _log_loss(blend(p_model, p_market, w, "geo"), y)
            print(f"  {w:>6.2f}  {ll_lin:>8.4f}  {ll_geo:>8.4f}")

    # Average across years + best weight
    print(f"\n  AVERAGE across {sorted(results)}")
    print(f"  {'w':>6}  {'linear':>8}  {'log-odds':>8}")
    best = (None, None, float("inf"))
    for w in WEIGHTS:
        avgs = {}
        for mode in ("linear", "geo"):
            lls = [_log_loss(blend(pm, pk, w, mode), y) for pm, pk, y in results.values()]
            avgs[mode] = float(np.mean(lls))
            if avgs[mode] < best[2]:
                best = (w, mode, avgs[mode])
        print(f"  {w:>6.2f}  {avgs['linear']:>8.4f}  {avgs['geo']:>8.4f}")

    w_best, mode_best, ll_best = best
    print(f"\n  Best: w={w_best:.2f} ({mode_best})  avg log-loss {ll_best:.4f}")

    # Paired bootstrap: best blend vs model-only, pooled across years
    p_a = np.vstack([blend(pm, pk, w_best, mode_best) for pm, pk, _ in results.values()])
    p_b = np.vstack([pm for pm, _, _ in results.values()])
    y_all = np.concatenate([y for _, _, y in results.values()])
    diff, lo, hi = paired_bootstrap_diff(p_a, p_b, y_all)
    sig = "significant" if hi < 0 else "NOT significant"
    print(f"  Paired Δlog-loss (blend − model): {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}] 95% CI → {sig}")


if __name__ == "__main__":
    main()
