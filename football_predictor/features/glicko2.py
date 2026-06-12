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


_RATING_PERIOD_DAYS = 90.0  # one international window ≈ a Glicko rating period

# Home advantage on the μ scale (≈ +100 Elo points, matching elo.py).
# Glicko-2 has no native HA term; without it every non-neutral result is
# mis-attributed to rating strength. Injected by shifting the OPPONENT's μ
# in each result tuple (equivalent to boosting one's own μ inside _update).
# Applied ONLY when the match is not neutral.
_HOME_ADV_MU = 100.0 / _SCALE  # ≈ 0.576


def _default_state() -> dict[str, float]:
    return {"mu": _DEFAULT_MU, "phi": _DEFAULT_PHI, "sigma": _DEFAULT_SIGMA}


def _inflate_phi(state: dict[str, float], gap_days: float) -> dict[str, float]:
    """Grow RD for inactive rating periods: φ ← sqrt(φ² + σ²·n_inactive).

    Without this, RD only ever shrinks — a team active in 2015 and dormant
    since keeps a tiny φ forever, deleting Glicko-2's main advantage over Elo.
    The playing period itself is excluded (φ* = sqrt(φ²+σ²) inside _update
    already covers it); φ is capped at the unknown-team default.
    """
    extra_periods = max(0.0, gap_days / _RATING_PERIOD_DAYS - 1.0)
    if extra_periods <= 0.0:
        return state
    phi = math.sqrt(state["phi"] ** 2 + state["sigma"] ** 2 * extra_periods)
    return {**state, "phi": min(phi, _DEFAULT_PHI)}


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _e(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _update(state: dict[str, float], results: list[tuple[float, float, float, float]]) -> dict[str, float]:
    """One Glicko-2 update after a batch of results.

    results: list of (opponent_mu, opponent_phi, score, match_weight).
    Weight scales information contribution: WC (1.5×) → larger update; friendly (0.3×) → smaller.
    """
    mu, phi, sigma = state["mu"], state["phi"], state["sigma"]

    if not results:
        # Inactive period: φ grows by volatility (uncertainty increases)
        return {"mu": mu, "phi": math.sqrt(phi * phi + sigma * sigma), "sigma": sigma}

    v_inv = sum(w * _g(pj) ** 2 * _e(mu, mj, pj) * (1.0 - _e(mu, mj, pj)) for mj, pj, _, w in results)
    v = 1.0 / max(v_inv, 1e-10)

    delta = v * sum(w * _g(pj) * (s - _e(mu, mj, pj)) for mj, pj, s, w in results)

    # Illinois algorithm to update σ (Glickman 2006 implementation note)
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        d2 = delta * delta
        p2 = phi * phi
        return (ex * (d2 - p2 - v - ex)) / (2.0 * (p2 + v + ex) ** 2) - (x - a) / (_TAU * _TAU)

    # Bracket per Glickman (2012) step 5.2: when Δ² > φ² + v, ln(Δ²−φ²−v) is
    # already a valid endpoint; the downward search applies ONLY to the other
    # branch (and steps by τ, not 1.0). Running the search unconditionally
    # walks B past the root, un-bracketing the iteration exactly after upsets.
    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1.0
        while f(a - k * _TAU) < 0:
            k += 1.0
        B = a - k * _TAU

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
    mu_new = mu + phi_new * phi_new * sum(w * _g(pj) * (s - _e(mu, mj, pj)) for mj, pj, s, w in results)

    return {"mu": mu_new, "phi": phi_new, "sigma": sigma_new}


class Glicko2Features(FeatureModule):
    """Glicko-2 ratings: μ (rating), φ (deviation/uncertainty), win probability."""

    name = "glicko2"

    def __init__(self) -> None:
        self._ratings: dict[str, dict[str, float]] = {}
        self._snapshots: dict[str, dict[str, dict[str, float]]] = {}
        self._last_played: dict[str, pd.Timestamp] = {}
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home, away = match["home_team"], match["away_team"]

        # Per-date snapshots only contain teams that PLAYED on that date.
        # A team absent from an existing snapshot (e.g. predicting a fixture
        # on a day other matches were already recorded) must fall back to its
        # latest rating — NOT to the default, which would silently reset it.
        # The fallback rating gets RD inflated for time since the team's last
        # match (rating periods), so stale ratings carry honest uncertainty.
        snap = self._snapshots.get(date_str, {})

        def _state_for(team: str) -> dict[str, float]:
            s = snap.get(team)
            if s is not None:
                return s
            s = self._ratings.get(team)
            if s is None:
                return _default_state()
            last = self._last_played.get(team)
            if last is not None:
                gap = (pd.Timestamp(match["date"]) - last).days
                s = _inflate_phi(s, float(gap))
            return s

        h = _state_for(home)
        a = _state_for(away)

        r_h = h["mu"] * _SCALE + 1500.0
        r_a = a["mu"] * _SCALE + 1500.0
        phi_h = h["phi"] * _SCALE
        phi_a = a["phi"] * _SCALE

        # Home advantage in the expected-score: boost home μ for non-neutral
        # matches only (WC 2026 fixture rows always carry neutral=True).
        neutral = bool(match.get("neutral", False))
        win_prob = _e(h["mu"] + (0.0 if neutral else _HOME_ADV_MU), a["mu"], a["phi"])

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
        last_played: dict[str, pd.Timestamp] = {}
        self._snapshots = {}

        for date, group in df.groupby("date", sort=True):
            date_str = str(date.date())
            all_teams = set(group["home_team"]) | set(group["away_team"])

            # Inactivity: grow RD for rating periods without matches BEFORE
            # this date's snapshot/update, so both training features and the
            # subsequent update see honestly widened uncertainty.
            for t in all_teams:
                if t in ratings and t in last_played:
                    gap = (date - last_played[t]).days
                    ratings[t] = _inflate_phi(ratings[t], float(gap))

            self._snapshots[date_str] = {
                t: dict(ratings.get(t, _default_state())) for t in all_teams
            }

            # Collect all matches per team this date-batch, then batch-update
            team_results: dict[str, list[tuple[float, float, float]]] = {}
            for _, row in group.iterrows():
                h, a = row["home_team"], row["away_team"]
                hg, ag = float(row["home_goals"]), float(row["away_goals"])
                s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                mw = float(row.get("match_weight", 1.0))

                h_state = ratings.get(h, _default_state())
                a_state = ratings.get(a, _default_state())

                # Home advantage: shift the opponent's μ in the result tuple
                # (≡ boosting one's own μ inside _update). Home team faces an
                # effectively weaker opponent (−HA); away team faces an
                # effectively stronger one (+HA). Neutral matches unchanged.
                ha = 0.0 if bool(row.get("neutral", False)) else _HOME_ADV_MU

                team_results.setdefault(h, []).append((a_state["mu"] - ha, a_state["phi"], s_h, mw))
                team_results.setdefault(a, []).append((h_state["mu"] + ha, h_state["phi"], 1.0 - s_h, mw))

            for team, res in team_results.items():
                ratings[team] = _update(ratings.get(team, _default_state()), res)
                last_played[team] = date

        self._ratings = ratings
        self._last_played = last_played

    def feature_names(self) -> list[str]:
        return [
            "glicko2_home_rating", "glicko2_home_rd",
            "glicko2_away_rating", "glicko2_away_rd",
            "glicko2_diff", "glicko2_home_win_prob",
        ]
