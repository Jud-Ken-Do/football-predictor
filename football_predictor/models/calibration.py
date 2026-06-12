from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from football_predictor.constants import CALIBRATION_METHOD, OUTCOMES


class CalibrationLayer:
    """Post-hoc probability calibration via isotonic regression.

    Research shows calibration-optimised models yield 69.86% higher
    simulated returns than accuracy-optimised ones (Petretta et al., 2025).

    Applies independent isotonic regression to each outcome class, then
    renormalises so probabilities sum to 1.
    """

    def __init__(self, method: str = CALIBRATION_METHOD):
        self.method = method
        self._calibrators: dict[str, IsotonicRegression] = {}

    def fit(self, proba: pd.DataFrame, y_true: pd.Series) -> "CalibrationLayer":
        """Fit one calibrator per outcome class.

        Args:
            proba:  DataFrame with columns matching OUTCOMES (home_win, draw, away_win).
            y_true: Integer series — 0 = home win, 1 = draw, 2 = away win.
        """
        for i, outcome in enumerate(OUTCOMES):
            binary_y = (y_true == i).astype(int).values
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(proba[outcome].values, binary_y)
            self._calibrators[outcome] = cal
        return self

    def transform(self, proba: pd.DataFrame) -> pd.DataFrame:
        """Apply calibration and renormalise.

        Args:
            proba: Uncalibrated probability DataFrame (home_win, draw, away_win).

        Returns:
            Calibrated and renormalised probability DataFrame.
        """
        calibrated = pd.DataFrame(index=proba.index)
        for outcome in OUTCOMES:
            raw = proba[outcome].values
            calibrated[outcome] = self._calibrators[outcome].predict(raw)

        row_sums = calibrated.sum(axis=1).clip(lower=1e-8)
        return calibrated.div(row_sums, axis=0)

    def fit_transform(self, proba: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
        return self.fit(proba, y_true).transform(proba)


class TemperatureScaling:
    """Temperature scaling calibration (Guo et al., ICML 2017).

    Learns a single scalar T that divides log-probabilities before softmax:
        P_calibrated = softmax(log P / T)

    T > 1 softens (reduces overconfidence), T < 1 sharpens.

    Advantages over per-class isotonic regression on small calibration sets:
      - Only 1 parameter to fit vs 3N for isotonic → far less overfitting
      - Preserves ranking of probabilities (monotone transformation)
      - Strong theoretical grounding for overconfident neural/ensemble models

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017).
        On calibration of modern neural networks. ICML 2017.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(
        self,
        proba: pd.DataFrame,
        y_true: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "TemperatureScaling":
        """Fit T by minimising NLL on the calibration set.

        Args:
            proba:         Uncalibrated probability DataFrame.
            y_true:        Integer outcome series (0/1/2).
            sample_weight: Optional per-row weights (e.g. match-type weights);
                           when provided, minimises weighted NLL
                           -(w * log p_true).sum() / w.sum().
        """
        log_p = np.log(np.maximum(proba.values, 1e-10))
        y = y_true.values
        w = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)

        def nll(T: float) -> float:
            scaled = log_p / T
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_s = np.exp(scaled)
            p_cal = exp_s / exp_s.sum(axis=1, keepdims=True)
            log_p_true = np.log(np.maximum(p_cal[np.arange(len(y)), y], 1e-10))
            if w is not None:
                return float(-(w * log_p_true).sum() / w.sum())
            return float(-log_p_true.sum())

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, proba: pd.DataFrame) -> pd.DataFrame:
        log_p = np.log(np.maximum(proba.values, 1e-10)) / self.temperature
        log_p -= log_p.max(axis=1, keepdims=True)
        exp_p = np.exp(log_p)
        calibrated = exp_p / exp_p.sum(axis=1, keepdims=True)
        return pd.DataFrame(calibrated, columns=proba.columns, index=proba.index)

    def fit_transform(self, proba: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
        return self.fit(proba, y_true).transform(proba)
