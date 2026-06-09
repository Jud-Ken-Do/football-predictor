from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor.constants import OUTCOMES


def ranked_probability_score(proba: pd.DataFrame, y_true: pd.Series) -> float:
    """Ranked Probability Score (RPS) — lower is better.

    The standard evaluation metric for ordered 3-outcome football prediction.
    Penalises confident wrong predictions more than diffuse ones.
    Lower bound 0 (perfect), upper bound 1.
    """
    n = len(y_true)
    rps_total = 0.0
    for i, (_, row) in enumerate(proba.iterrows()):
        outcome = int(y_true.iloc[i])
        cumulative_pred = np.cumsum([row["home_win"], row["draw"], row["away_win"]])
        cumulative_true = np.cumsum([int(outcome == 0), int(outcome == 1), int(outcome == 2)])
        rps_total += np.sum((cumulative_pred - cumulative_true) ** 2) / 2.0
    return rps_total / n


def brier_score(proba: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """Per-class Brier scores.

    Returns a dict with one score per outcome plus an overall mean.
    """
    scores = {}
    for i, outcome in enumerate(OUTCOMES):
        binary_y = (y_true == i).astype(float).values
        scores[outcome] = float(np.mean((proba[outcome].values - binary_y) ** 2))
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def log_loss(proba: pd.DataFrame, y_true: pd.Series, eps: float = 1e-7) -> float:
    """Multi-class log loss."""
    n = len(y_true)
    total = 0.0
    for i, (_, row) in enumerate(proba.iterrows()):
        outcome = int(y_true.iloc[i])
        p = max(row.iloc[outcome], eps)
        total += np.log(p)
    return -total / n


def accuracy(proba: pd.DataFrame, y_true: pd.Series) -> float:
    """Fraction of matches where the highest-probability outcome was correct."""
    predicted = proba.values.argmax(axis=1)
    return float(np.mean(predicted == y_true.values))


def roi_simulation(
    proba: pd.DataFrame,
    y_true: pd.Series,
    odds: pd.DataFrame,
    stake: float = 1.0,
    min_edge: float = 0.02,
) -> dict[str, float]:
    """Simulate betting returns based on model probabilities vs. market odds.

    Only bets when (model_prob - implied_prob) > min_edge — i.e., when the
    model thinks the market is mispriced by at least `min_edge`.

    Args:
        proba:    Model probabilities (home_win, draw, away_win).
        y_true:   Actual outcomes (0/1/2).
        odds:     Decimal odds DataFrame with same columns as proba.
        stake:    Flat stake per bet.
        min_edge: Minimum edge over market implied probability to bet.

    Returns:
        dict with total_staked, total_return, roi (as fraction), n_bets.
    """
    total_staked = total_return = n_bets = 0.0

    for i, (_, row_p) in enumerate(proba.iterrows()):
        outcome = int(y_true.iloc[i])
        row_o = odds.iloc[i]

        for j, col in enumerate(OUTCOMES):
            if row_o[col] <= 1.0:
                continue
            implied = 1.0 / row_o[col]
            edge = row_p[col] - implied
            if edge >= min_edge:
                n_bets += 1
                total_staked += stake
                if outcome == j:
                    total_return += stake * row_o[col]

    roi = (total_return - total_staked) / total_staked if total_staked > 0 else 0.0
    return {
        "total_staked": total_staked,
        "total_return": total_return,
        "roi": roi,
        "n_bets": n_bets,
    }


def summary(proba: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """All scalar metrics in one call."""
    return {
        "rps": ranked_probability_score(proba, y_true),
        "log_loss": log_loss(proba, y_true),
        "accuracy": accuracy(proba, y_true),
        **{f"brier_{k}": v for k, v in brier_score(proba, y_true).items()},
    }
