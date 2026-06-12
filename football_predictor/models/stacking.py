"""Out-of-time K-fold stacked calibration (roadmap R3).

The temperature T and the ensemble's 4 context weights were previously fitted
on a single chronological tail (15-20% of the training data) — a high-variance
estimate for 5 calibration parameters. Instead, pool genuinely out-of-time
predictions across several expanding-window folds and fit the calibration
layer once on the pooled set.

Fold scheme over the chronologically ordered training set:

    fold k:  train [0, b_k)  →  predict [b_k, b_{k+1})

with boundaries b_k spaced evenly between ``first_frac`` and 1.0. Every pooled
prediction is made by a model that has seen only earlier matches, mirroring
deployment, and the calibration set grows from ~20% to ~(1 - first_frac) of
the data.

After calling this, callers should refit the final XGBoost and BayesPoisson
models on the full training window (standard stacking practice — the
calibration layer transfers to the slightly-better final models).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from football_predictor.models.calibration import TemperatureScaling
from football_predictor.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


def fit_stacked_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    matches: pd.DataFrame,
    sample_weight: np.ndarray,
    fit_xgb: Callable[[pd.DataFrame, pd.Series, np.ndarray], object],
    fit_bp: Callable[[pd.DataFrame], object],
    bp_proba: Callable[[object, pd.DataFrame], pd.DataFrame],
    n_folds: int = 4,
    first_frac: float = 0.6,
    quiet: bool = False,
) -> tuple[TemperatureScaling, EnsembleModel]:
    """Fit T and ensemble α on pooled out-of-time fold predictions.

    Args:
        X, y:          Feature matrix and labels, chronologically ordered.
        matches:       Raw match rows positionally aligned with X (for BP).
        sample_weight: Per-row match-type weights.
        fit_xgb:       (X_tr, y_tr, sw_tr) → fitted model with predict_proba.
        fit_bp:        (matches_tr) → fitted BayesianPoissonModel.
        bp_proba:      (bp, matches) → DataFrame[home_win, draw, away_win].
        n_folds:       Number of expanding-window folds.
        first_frac:    Fraction of data the first fold trains on; the pooled
                       calibration set is the remaining (1 - first_frac).

    Returns:
        (temp_cal, ensemble) fitted on the pooled out-of-time predictions.
    """
    n = len(X)
    sw = np.asarray(sample_weight, dtype=float)
    step = (1.0 - first_frac) / n_folds
    bounds = [int(round(n * (first_frac + i * step))) for i in range(n_folds + 1)]

    xgb_parts: list[pd.DataFrame] = []
    bp_parts: list[pd.DataFrame] = []
    idx_parts: list[np.ndarray] = []
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        xgb_k = fit_xgb(X.iloc[:lo], y.iloc[:lo], sw[:lo])
        xgb_parts.append(xgb_k.predict_proba(X.iloc[lo:hi]).reset_index(drop=True))
        bp_k = fit_bp(matches.iloc[:lo])
        bp_parts.append(bp_proba(bp_k, matches.iloc[lo:hi]).reset_index(drop=True))
        idx_parts.append(np.arange(lo, hi))
        if not quiet:
            print(f"    fold {k + 1}/{n_folds}: trained on {lo:,} rows → {hi - lo:,} OOS predictions")

    idx = np.concatenate(idx_parts)
    xgb_oos = pd.concat(xgb_parts, ignore_index=True)
    bp_oos = pd.concat(bp_parts, ignore_index=True)
    y_oos = y.iloc[idx].reset_index(drop=True)
    sw_oos = sw[idx]
    ctx_oos = X.iloc[idx].reset_index(drop=True)

    temp_cal = TemperatureScaling()
    temp_cal.fit(xgb_oos, y_oos, sample_weight=sw_oos)

    ensemble = EnsembleModel()
    # α fitted on temperature-scaled XGB probs — the distribution it blends
    # at predict time.
    ensemble.fit(temp_cal.transform(xgb_oos), bp_oos, y_oos,
                 context_X=ctx_oos, sample_weight=sw_oos)

    logger.info(
        "Stacked calibration: %d pooled OOS rows across %d folds | T=%.3f ᾱ=%.3f",
        len(idx), n_folds, temp_cal.temperature, ensemble.xgb_weight,
    )
    return temp_cal, ensemble
