"""Ensemble model: blend XGBoost + BayesianPoisson probabilities.

Learns the optimal convex combination via log-loss minimisation on the
calibration set. When context features are supplied, α is per-match
(sigmoid of a small logistic regression on 3 context signals); otherwise
falls back to a learned scalar.

Context signals:
  odds_available     — 1.0 if bookmaker odds exist; model trusts XGB more
  kalman_uncertainty — mean √P across all four Kalman state dimensions;
                       high uncertainty → trust BayesPoisson structural prior
  log1p_h2h          — log(1 + h2h_n_matches); thin H2H → XGB features noisier

Reference:
    Groll, A. et al. (2018). Prediction of the FIFA World Cup 2018 — A random
    forest approach with an emphasis on estimated team ability parameters.
    Journal of Quantitative Analysis in Sports, 15(2), 135-154.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_KALMAN_STD_COLS = [
    "kalman_home_att_std", "kalman_home_def_std",
    "kalman_away_att_std", "kalman_away_def_std",
]
_H2H_COL = "h2h_n_matches"
_ODDS_COL = "odds_available"


def _extract_context(X: pd.DataFrame) -> np.ndarray | None:
    """Return (n, 3) context array [odds_available, kalman_unc, log1p_h2h], or None."""
    missing = [c for c in _KALMAN_STD_COLS + [_H2H_COL, _ODDS_COL] if c not in X.columns]
    if missing:
        return None
    odds = X[_ODDS_COL].values.astype(float)
    kalman_unc = X[_KALMAN_STD_COLS].values.mean(axis=1)
    log1p_h2h = np.log1p(X[_H2H_COL].values.astype(float))
    return np.column_stack([odds, kalman_unc, log1p_h2h])


class EnsembleModel:
    """Convex blend of XGBoost and BayesianPoisson probabilities.

    When fit() receives a context DataFrame (X), learns per-match weights:
        α_i = sigmoid(w · ctx_i + b)    where ctx_i are 3 normalised signals

    Without context, falls back to a learned scalar α.
    """

    def __init__(self) -> None:
        self._alpha: float = 0.6        # scalar fallback
        self._w: np.ndarray | None = None    # shape (4,): [w1, w2, w3, bias]
        self._ctx_mean: np.ndarray | None = None
        self._ctx_std: np.ndarray | None = None

    def _alpha_vec(self, ctx_raw: np.ndarray) -> np.ndarray:
        """Per-match α from raw context (n, 3) → (n,)."""
        ctx = (ctx_raw - self._ctx_mean) / self._ctx_std  # type: ignore[operator]
        assert self._w is not None
        logit = ctx @ self._w[:3] + self._w[3]
        return 1.0 / (1.0 + np.exp(-logit))

    def fit(
        self,
        xgb_proba: pd.DataFrame,
        bp_proba: pd.DataFrame,
        y_true: pd.Series,
        context_X: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "EnsembleModel":
        """Fit the blend weight(s) by NLL minimisation.

        Args:
            sample_weight: Optional per-row weights (e.g. match-type weights);
                           when provided, minimises weighted NLL
                           -(w * log p_true).sum() / w.sum().
        """
        xgb = xgb_proba.values
        bp = bp_proba.values
        y = y_true.values
        w = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
        idx = np.arange(len(y))

        def _nll(blended: np.ndarray) -> float:
            log_p_true = np.log(np.maximum(blended[idx, y], 1e-10))
            if w is not None:
                return float(-(w * log_p_true).sum() / w.sum())
            return float(-log_p_true.sum())

        ctx_raw = _extract_context(context_X) if context_X is not None else None

        if ctx_raw is not None:
            self._ctx_mean = ctx_raw.mean(axis=0)
            self._ctx_std = np.where(ctx_raw.std(axis=0) < 1e-8, 1.0, ctx_raw.std(axis=0))
            ctx = (ctx_raw - self._ctx_mean) / self._ctx_std

            def nll_ctx(params: np.ndarray) -> float:
                logit = ctx @ params[:3] + params[3]
                a = 1.0 / (1.0 + np.exp(-logit))  # (n,)
                blended = a[:, None] * xgb + (1 - a[:, None]) * bp
                return _nll(blended)

            # L2 regularisation on w (not bias) to prevent extreme weights
            def nll_reg(params: np.ndarray) -> float:
                return nll_ctx(params) + 0.5 * float(np.dot(params[:3], params[:3]))

            res = minimize(nll_reg, x0=np.zeros(4), method="L-BFGS-B")
            self._w = res.x.astype(float)

            # Report effective α range
            a_vals = self._alpha_vec(ctx_raw)
            self._alpha = float(a_vals.mean())
            logger.info(
                "Ensemble context weights: %s  bias=%.3f | α range [%.2f, %.2f]  mean=%.2f",
                np.round(self._w[:3], 3).tolist(), self._w[3],
                float(a_vals.min()), float(a_vals.max()), self._alpha,
            )
        else:
            def nll(params: np.ndarray) -> float:
                a = float(np.clip(params[0], 1e-4, 1 - 1e-4))
                blended = a * xgb + (1 - a) * bp
                return _nll(blended)

            res = minimize(nll, x0=[0.6], bounds=[(0.0, 1.0)], method="L-BFGS-B")
            self._alpha = float(np.clip(res.x[0], 0.0, 1.0))
            logger.info(
                "Ensemble weights: XGB=%.3f  BayesPoisson=%.3f", self._alpha, 1.0 - self._alpha
            )

        return self

    def predict_proba(
        self,
        xgb_proba: pd.DataFrame,
        bp_proba: pd.DataFrame,
        context_X: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        xgb = xgb_proba.values
        bp = bp_proba.values

        ctx_raw = _extract_context(context_X) if context_X is not None else None

        if ctx_raw is not None and self._w is not None:
            a = self._alpha_vec(ctx_raw)[:, None]  # (n, 1)
        else:
            a = self._alpha  # scalar broadcast

        blended = a * xgb + (1.0 - a) * bp
        blended = np.maximum(blended, 1e-10)
        blended = blended / blended.sum(axis=1, keepdims=True)
        return pd.DataFrame(blended, columns=xgb_proba.columns, index=xgb_proba.index)

    @property
    def xgb_weight(self) -> float:
        return self._alpha

    @property
    def bp_weight(self) -> float:
        return 1.0 - self._alpha
