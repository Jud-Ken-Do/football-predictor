#!/usr/bin/env python3.11
"""Benchmark: compare three Kalman feature modes on WC backtest.

  smoothed  — RTS states for training, forward for prediction (original)
  forward   — forward-only states for both (covariate-shift fix)
  em_tuned  — EM-tuned Q, forward-only states for both (Option A)

Usage:
    python3.11 scripts/benchmark_kalman.py
    python3.11 scripts/benchmark_kalman.py --years 2022
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from football_predictor.features import REGISTRY
from football_predictor.features.kalman_strength import KalmanStrengthFeatures
from scripts.backtest import run_backtest

MODES = [
    ("smoothed", {"use_smoothed": True,  "tune_q": False}),
    ("forward",  {"use_smoothed": False, "tune_q": False}),
    ("em_tuned", {"use_smoothed": False, "tune_q": True}),
]


def _patch_kalman(**kwargs) -> None:
    REGISTRY["kalman_strength"] = functools.partial(KalmanStrengthFeatures, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2022])
    args = parser.parse_args()

    results: dict[str, list[dict]] = {label: [] for label, _ in MODES}

    for label, kwargs in MODES:
        print(f"\n{'='*60}")
        print(f"  MODE: {label.upper()}")
        print(f"{'='*60}")
        _patch_kalman(**kwargs)
        for year in args.years:
            m = run_backtest(year, include_shap=False, label=label)
            if m:
                m["mode"] = label
                results[label].append(m)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Year':<6} {'Mode':<12} {'LogLoss':<10} {'Brier':<10} {'Acc':<8} {'ECE':<8}")
    print("  " + "-" * 54)

    for label, _ in MODES:
        for m in results[label]:
            print(
                f"  {m['year']:<6} {label:<12} {m['log_loss']:<10.4f} "
                f"{m['brier_score']:<10.4f} {m['accuracy']:<8.3f} {m['ece']:<8.4f}"
            )

    # ── Delta vs smoothed baseline ────────────────────────────────────────────
    print()
    baseline = {m["year"]: m for m in results["smoothed"]}
    for label, _ in MODES[1:]:
        for m in results[label]:
            base = baseline.get(m["year"])
            if not base:
                continue
            delta = m["log_loss"] - base["log_loss"]
            sign = "+" if delta > 0 else ""
            direction = "WORSE" if delta > 0 else "BETTER"
            print(
                f"  WC {m['year']} [{label}]: log-loss {sign}{delta:.4f} "
                f"vs smoothed ({direction})"
            )


if __name__ == "__main__":
    main()
