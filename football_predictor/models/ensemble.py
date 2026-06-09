"""Ensemble model: blend XGBoost + BayesianPoisson probabilities.

Learns the optimal convex combination via log-loss minimisation on the
calibration set. A weighted average outperforms either model alone when
the two models have complementary failure modes (XGB overfits to features
that are absent at WC time; BayesPoisson ignores feature context but has
well-calibrated structural priors).

Reference:
    Groll, A. et al. (2018). Prediction of the FIFA World Cup 2018 — A random
    forest approach with an emphasis on estimated team ability parameters.
    Journal of Quantitative Analysis in Sports, 15(2), 135-154.
    (Ensemble of statistical + ML models consistently outperforms single models.)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football_predictor.constants import OUTCOMES

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Convex blend of XGBoost and BayesianPoisson probabilities.

    Learns a single scalar α ∈ [0, 1] such that:
        P = α * P_xgb + (1 - α) * P_bp

    minimises NLL on the calibration set.
    """

    def __init__(self) -> None:
        self._alpha: float = 0.6  # default: trust XGB more

    def fit(
        self,
        xgb_proba: pd.DataFrame,
        bp_proba: pd.DataFrame,
        y_true: pd.Series,
    ) -> "EnsembleModel":
        xgb = xgb_proba.values
        bp = bp_proba.values
        y = y_true.values

        def nll(params: np.ndarray) -> float:
            a = float(np.clip(params[0], 1e-4, 1 - 1e-4))
            blended = a * xgb + (1 - a) * bp
            blended = np.maximum(blended, 1e-10)
            total = 0.0
            for i in range(len(OUTCOMES)):
                mask = y == i
                if mask.any():
                    total -= np.log(blended[mask, i]).sum()
            return total

        res = minimize(nll, x0=[0.6], bounds=[(0.0, 1.0)], method="L-BFGS-B")
        self._alpha = float(np.clip(res.x[0], 0.0, 1.0))
        logger.info(
            "Ensemble weights: XGB=%.3f  BayesPoisson=%.3f", self._alpha, 1.0 - self._alpha
        )
        return self

    def predict_proba(self, xgb_proba: pd.DataFrame, bp_proba: pd.DataFrame) -> pd.DataFrame:
        blended = self._alpha * xgb_proba.values + (1.0 - self._alpha) * bp_proba.values
        blended = np.maximum(blended, 1e-10)
        blended = blended / blended.sum(axis=1, keepdims=True)
        return pd.DataFrame(blended, columns=xgb_proba.columns, index=xgb_proba.index)

    @property
    def xgb_weight(self) -> float:
        return self._alpha

    @property
    def bp_weight(self) -> float:
        return 1.0 - self._alpha
