from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from football_predictor.constants import DC_TIME_DECAY_HALF_LIFE_DAYS, DC_RHO

logger = logging.getLogger(__name__)

_MAX_GOALS = 10  # upper bound for scoreline enumeration


class DixonColesModel:
    """Dixon-Coles (1997) double-Poisson model with time-decay weighting.

    Fits per-team attack and defence strengths plus a home advantage term.
    Adds the low-score correction factor rho that inflates 0-0, 1-0, 0-1,
    and 1-1 scorelines — the main contribution of the original paper.

    Reference:
        Dixon, M. J., & Coles, S. G. (1997). Modelling association football
        scores and inefficiencies in the football betting market.
        Applied Statistics, 46(2), 265-280.
    """

    def __init__(self, half_life_days: int = DC_TIME_DECAY_HALF_LIFE_DAYS):
        self.half_life_days = half_life_days
        self._params: Optional[dict] = None
        self._teams: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """Fit attack/defence strengths from historical match results.

        Args:
            matches: DataFrame with columns date, home_team, away_team,
                     home_goals, away_goals. Sorted ascending by date.
        """
        matches = matches.copy()
        matches["date"] = pd.to_datetime(matches["date"])
        reference_date = matches["date"].max()

        # Combine time-decay with match-type weight (friendly=0.3×, WC=1.5×)
        match_type_weight = matches.get("match_weight", pd.Series(1.0, index=matches.index))
        days_ago = (reference_date - matches["date"]).dt.days.values.astype(float)
        weights = np.exp(-math.log(2) * days_ago / self.half_life_days) * match_type_weight.values

        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self._teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # Pre-build integer index arrays and goal arrays — passed to vectorised loss
        hi = np.array([idx[t] for t in matches["home_team"]], dtype=np.int32)
        ai = np.array([idx[t] for t in matches["away_team"]], dtype=np.int32)
        hg = matches["home_goals"].values.astype(np.int32)
        ag = matches["away_goals"].values.astype(np.int32)

        x0 = np.concatenate([np.ones(n), np.ones(n), [0.1], [DC_RHO]])

        bounds = (
            [(0.01, 10.0)] * n
            + [(0.01, 10.0)] * n
            + [(-1.0, 1.0)]
            + [(-0.5, 0.5)]
        )

        result = minimize(
            fun=_neg_log_likelihood_vec,
            x0=x0,
            args=(hi, ai, hg, ag, weights, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "maxfun": 50000, "ftol": 1e-6, "gtol": 1e-5},
        )

        if not result.success:
            logger.warning("Dixon-Coles optimisation did not converge: %s", result.message)

        attack = result.x[:n]
        defence = result.x[n : 2 * n]
        home_adv = result.x[2 * n]
        rho = result.x[2 * n + 1]

        self._params = {
            "attack": dict(zip(teams, attack)),
            "defence": dict(zip(teams, defence)),
            "home_adv": home_adv,
            "rho": rho,
        }
        return self

    def predict_proba(self, home_team: str, away_team: str, neutral: bool = False) -> dict[str, float]:
        """Predict W/D/L probabilities by enumerating scorelines.

        Args:
            home_team: First team (nominal home — both away at neutral venues).
            away_team: Second team.
            neutral:   True for World Cup / neutral venue. Suppresses home_adv.

        Returns:
            dict with keys "home_win", "draw", "away_win".
        """
        if self._params is None:
            raise RuntimeError("Model must be fit() before predict_proba().")

        lam_h = self._lambda(home_team, away_team, home=not neutral)
        lam_a = self._lambda(away_team, home_team, home=False)
        rho = self._params["rho"]

        p_home = p_draw = p_away = 0.0
        for h in range(_MAX_GOALS + 1):
            for a in range(_MAX_GOALS + 1):
                p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a) * self._tau(h, a, lam_h, lam_a, rho)
                if h > a:
                    p_home += p
                elif h == a:
                    p_draw += p
                else:
                    p_away += p

        total = p_home + p_draw + p_away
        return {
            "home_win": p_home / total,
            "draw": p_draw / total,
            "away_win": p_away / total,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _lambda(self, attacking: str, defending: str, home: bool) -> float:
        p = self._params
        atk = p["attack"].get(attacking, 1.0)
        dfc = p["defence"].get(defending, 1.0)
        ha = p["home_adv"] if home else 0.0
        return max(atk * dfc * math.exp(ha), 1e-6)

    @staticmethod
    def _tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
        if h == 0 and a == 0:
            return 1.0 - lam_h * lam_a * rho
        if h == 1 and a == 0:
            return 1.0 + lam_a * rho
        if h == 0 and a == 1:
            return 1.0 + lam_h * rho
        if h == 1 and a == 1:
            return 1.0 - rho
        return 1.0

    # _neg_log_likelihood replaced by module-level _neg_log_likelihood_vec


def _neg_log_likelihood_vec(
    x: np.ndarray,
    hi: np.ndarray,
    ai: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n: int,
) -> float:
    """Fully vectorised Dixon-Coles negative log-likelihood. No Python loops."""
    attack = x[:n]
    defence = x[n: 2 * n]
    home_adv = x[2 * n]
    rho = x[2 * n + 1]

    lam_h = np.maximum(attack[hi] * defence[ai] * math.exp(home_adv), 1e-6)
    lam_a = np.maximum(attack[ai] * defence[hi], 1e-6)

    # Tau correction for low-scoring matches (vectorised masking)
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m11] = 1.0 - rho

    if np.any(tau <= 0):
        return 1e10

    log_tau = np.log(np.maximum(tau, 1e-10))
    log_p = poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a) + log_tau
    return -float(np.dot(weights, log_p))
