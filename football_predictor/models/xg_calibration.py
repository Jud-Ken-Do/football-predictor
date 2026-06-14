"""Per-source xG calibration.

Different xG providers disagree: the same shot can be 0.12 on one model and 0.22
on another, and per-match MAE between providers is ≈ 1 xG (they only converge in
aggregate, correlation > 0.97 over ~15+ games). We combine StatsBomb,
football-data.co.uk and (live) FIFA-coordinate xG, so before any of it is
consumed we map each source onto a common scale: actual goals.

For each source we fit an affine map ``xg' = a + b · xg`` that makes the source
an **unbiased** estimator of goals — ``mean(xg') == mean(goals)`` on that
source's own matches — while **preserving xG's lower variance**, which is the
entire reason to prefer xG over goals (it is the less-noisy signal). Concretely
we mean-match with unit slope (``b = 1``, ``a = mean(goals) − mean(xg)``).

We deliberately do NOT std-match xG up to goals' variance: that would re-inject
the finishing-luck noise we are trying to remove (the providers' raw xG std is
already ~0.6× goals std, exactly as expected). A diagnostic-only ``match_std``
mode is retained. Calibration sets are small, so the fit is kept simple (cf. the
project's temperature-vs-isotonic decision).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AffineCalibration:
    """xg' = a + b * xg."""
    a: float
    b: float

    def apply(self, xg: float | np.ndarray) -> float | np.ndarray:
        return self.a + self.b * xg


_IDENTITY = AffineCalibration(0.0, 1.0)


def fit_affine(
    xg: np.ndarray, goals: np.ndarray, match_std: bool = False
) -> AffineCalibration:
    """Calibrate xg onto goals.

    Default (``match_std=False``): mean-match with unit slope — removes the
    provider's systematic bias while preserving xG's lower variance.
    ``match_std=True``: also equalise std (diagnostic only; re-inflates noise).

    Falls back to identity if there is too little data.
    """
    xg = np.asarray(xg, dtype=float)
    goals = np.asarray(goals, dtype=float)
    mask = np.isfinite(xg) & np.isfinite(goals)
    xg, goals = xg[mask], goals[mask]
    if len(xg) < 10:
        return _IDENTITY
    if match_std:
        sx = xg.std()
        b = float(goals.std() / sx) if sx > 1e-6 else 1.0
    else:
        b = 1.0
    a = float(goals.mean() - b * xg.mean())
    return AffineCalibration(a, b)


def fit_source_calibrations(
    df: pd.DataFrame, match_std: bool = False
) -> dict[str, AffineCalibration]:
    """Fit one AffineCalibration per ``source`` in a long per-team-match frame.

    Expects columns: ``source``, ``xg``, ``goals`` (one row per team per match —
    i.e. both home and away perspectives stacked). Returns {source: calibration}.
    """
    out: dict[str, AffineCalibration] = {}
    for source, g in df.groupby("source"):
        out[source] = fit_affine(g["xg"].to_numpy(), g["goals"].to_numpy(), match_std)
    return out


def to_team_match_long(df: pd.DataFrame) -> pd.DataFrame:
    """Stack a home/away xG frame into one row per team-match for calibration fit.

    Input columns: source, home_xg, away_xg, home_score, away_score.
    Output columns: source, xg, goals.
    """
    home = pd.DataFrame({
        "source": df["source"], "xg": df["home_xg"], "goals": df["home_score"],
    })
    away = pd.DataFrame({
        "source": df["source"], "xg": df["away_xg"], "goals": df["away_score"],
    })
    return pd.concat([home, away], ignore_index=True).dropna(subset=["xg", "goals"])
