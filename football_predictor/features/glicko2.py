"""Glicko-2 rating system (Glickman 2001) as a feature module.

Improvements over basic Elo that matter for international football:
  - Rating Deviation φ tracks uncertainty — sparse-match teams get wider φ,
    meaning the model knows how confident it is in each rating
  - Volatility σ tracks inconsistency of performance over time
  - Better suited to national teams (~10 matches/year vs 38 for clubs)
  - Ratings converge faster for active teams, stay uncertain for inactive ones

References:
    Glickman, M.E. (2001). Dynamic paired comparison models with stochastic
    variances. Journal of Applied Statistics, 28(6), 673-689.
    http://www.glicko.net/glicko/glicko2.pdf
"""
from __future__ import annotations

import math

import pandas as pd

from football_predictor.features.base import FeatureModule

# Glicko-2 ↔ Elo-scale conversion
_SCALE = 173.7178          # μ_glicko2 * _SCALE + 1500 = Elo-scale rating
_TAU = 0.5                 # system constant: controls volatility change (0.3–1.2)
_DEFAULT_MU = 0.0          # (1500 - 1500) / 173.7178 = 0
_DEFAULT_PHI = 200.0 / _SCALE   # 200 Elo-pts initial RD → Glicko-2 scale
_DEFAULT_SIGMA = 0.06      # initial volatility
_EPSILON = 1e-6            # Illinois algorithm convergence threshold


def _default_state() -> dict[str, float]:
    return {"mu": _DEFAULT_MU, "phi": _DEFAULT_PHI, "sigma": _DEFAULT_SIGMA}


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _e(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _update(state: dict[str, float], results: list[tuple[float, float, float]]) -> dict[str, float]:
    """One Glicko-2 update after a batch of results.

    results: list of (opponent_mu, opponent_phi, score) where score ∈ {0, 0.5, 1}.
    """
    mu, phi, sigma = state["mu"], state["phi"], state["sigma"]

    if not results:
        # Inactive period: φ grows by volatility (uncertainty increases)
        return {"mu": mu, "phi": math.sqrt(phi * phi + sigma * sigma), "sigma": sigma}

    v_inv = sum(_g(pj) ** 2 * _e(mu, mj, pj) * (1.0 - _e(mu, mj, pj)) for mj, pj, _ in results)
    v = 1.0 / max(v_inv, 1e-10)

    delta = v * sum(_g(pj) * (s - _e(mu, mj, pj)) for mj, pj, s in results)

    # Illinois algorithm to update σ (Glickman 2006 implementation note)
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        d2 = delta * delta
        p2 = phi * phi
        return (ex * (d2 - p2 - v - ex)) / (2.0 * (p2 + v + ex) ** 2) - (x - a) / (_TAU * _TAU)

    A = a
    B = math.log(delta * delta - phi * phi - v) if delta * delta > phi * phi + v else a - 1.0
    while f(B) < 0:
        B -= 1.0

    fA, fB = f(A), f(B)
    for _ in range(100):
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fB * fC < 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
        if abs(B - A) < _EPSILON:
            break

    sigma_new = math.exp(A / 2.0)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * sum(_g(pj) * (s - _e(mu, mj, pj)) for mj, pj, s in results)

    return {"mu": mu_new, "phi": phi_new, "sigma": sigma_new}


class Glicko2Features(FeatureModule):
    """Glicko-2 ratings: μ (rating), φ (deviation/uncertainty), win probability."""

    name = "glicko2"

    def __init__(self) -> None:
        self._ratings: dict[str, dict[str, float]] = {}
        self._snapshots: dict[str, dict[str, dict[str, float]]] = {}
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home, away = match["home_team"], match["away_team"]

        snap = self._snapshots.get(date_str, self._ratings)
        h = snap.get(home, _default_state())
        a = snap.get(away, _default_state())

        r_h = h["mu"] * _SCALE + 1500.0
        r_a = a["mu"] * _SCALE + 1500.0
        phi_h = h["phi"] * _SCALE
        phi_a = a["phi"] * _SCALE

        win_prob = _e(h["mu"], a["mu"], a["phi"])

        return {
            "glicko2_home_rating": r_h,
            "glicko2_home_rd": phi_h,
            "glicko2_away_rating": r_a,
            "glicko2_away_rd": phi_a,
            "glicko2_diff": r_h - r_a,
            "glicko2_home_win_prob": win_prob,
        }

    def _ensure_cache(self, data: pd.DataFrame) -> None:
        if self._data_id == id(data):
            return
        self._data_id = id(data)
        self._build_cache(data)

    def _build_cache(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        ratings: dict[str, dict[str, float]] = {}
        self._snapshots = {}

        for date, group in df.groupby("date", sort=True):
            date_str = str(date.date())
            all_teams = set(group["home_team"]) | set(group["away_team"])

            self._snapshots[date_str] = {
                t: dict(ratings.get(t, _default_state())) for t in all_teams
            }

            # Collect all matches per team this date-batch, then batch-update
            team_results: dict[str, list[tuple[float, float, float]]] = {}
            for _, row in group.iterrows():
                h, a = row["home_team"], row["away_team"]
                hg, ag = float(row["home_goals"]), float(row["away_goals"])
                s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

                h_state = ratings.get(h, _default_state())
                a_state = ratings.get(a, _default_state())

                team_results.setdefault(h, []).append((a_state["mu"], a_state["phi"], s_h))
                team_results.setdefault(a, []).append((h_state["mu"], h_state["phi"], 1.0 - s_h))

            for team, res in team_results.items():
                ratings[team] = _update(ratings.get(team, _default_state()), res)

        self._ratings = ratings

    def feature_names(self) -> list[str]:
        return [
            "glicko2_home_rating", "glicko2_home_rd",
            "glicko2_away_rating", "glicko2_away_rd",
            "glicko2_diff", "glicko2_home_win_prob",
        ]
