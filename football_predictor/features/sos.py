"""Strength of Schedule (SoS) features.

Raw win rate treats a win over Spain the same as a win over Cabo Verde.
SoS adjusts for opponent quality by weighting results by the opponent's
Elo rating at the time of the match.

Method:
  1. Forward Elo pass over all historical matches (self-contained, K=40).
  2. Per team, record (opponent_elo, pts_earned) before each match.
  3. Roll over last 5 / 10 entries for features.

Features (7):
  sos_home_avg_opp_elo5    — avg Elo of last 5 opponents (home team), normalised /1500
  sos_home_avg_opp_elo10   — avg Elo of last 10 opponents
  sos_away_avg_opp_elo5
  sos_away_avg_opp_elo10
  sos_diff_avg_opp_elo10   — home − away avg opponent quality
  sos_home_adj_win_rate10  — Σ(pts · opp_elo) / (3 · Σ(opp_elo))  last 10
  sos_away_adj_win_rate10

References:
    Massey, K. (1997). Statistical models applied to the rating of sports teams.
    Bradley-Terry model generalisation for schedule-adjusted rankings.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from football_predictor.features.base import FeatureModule

_DEFAULT_ELO: float = 1500.0
_K: float = 40.0
_ELO_NORM: float = 1500.0


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _updated(ra: float, rb: float, score_a: float) -> float:
    return ra + _K * (score_a - _expected(ra, rb))


def _pts(home_goals: int, away_goals: int, is_home: bool) -> float:
    if home_goals > away_goals:
        return 3.0 if is_home else 0.0
    if home_goals < away_goals:
        return 0.0 if is_home else 3.0
    return 1.0


def _window_stats(hist: list[tuple[float, float]], n: int) -> tuple[float, float]:
    """Returns (avg_opp_elo, adj_win_rate) over last n entries."""
    window = hist[-n:] if len(hist) >= n else hist
    if not window:
        return _DEFAULT_ELO, 1.0 / 3.0
    elos = [e for e, _ in window]
    pts  = [p for _, p in window]
    avg_elo = sum(elos) / len(elos)
    total_elo = sum(elos)
    adj_wr = sum(p * e for p, e in zip(pts, elos)) / (3.0 * total_elo) if total_elo > 0 else 1.0 / 3.0
    return avg_elo, adj_wr


class StrengthOfScheduleFeatures(FeatureModule):
    """Schedule-quality-adjusted form features."""

    name = "sos"

    def __init__(self) -> None:
        self._snapshots: dict[str, dict[str, list]] = {}
        self._final: dict[str, list] = {}
        self._data_id: Optional[int] = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home, away = match["home_team"], match["away_team"]

        snap = self._snapshots.get(date_str, {})
        h = snap.get(home, self._final.get(home, []))
        a = snap.get(away, self._final.get(away, []))

        h_elo5,  h_wr5  = _window_stats(h, 5)
        h_elo10, h_wr10 = _window_stats(h, 10)
        a_elo5,  a_wr5  = _window_stats(a, 5)
        a_elo10, a_wr10 = _window_stats(a, 10)

        return {
            "sos_home_avg_opp_elo5":   h_elo5  / _ELO_NORM,
            "sos_home_avg_opp_elo10":  h_elo10 / _ELO_NORM,
            "sos_away_avg_opp_elo5":   a_elo5  / _ELO_NORM,
            "sos_away_avg_opp_elo10":  a_elo10 / _ELO_NORM,
            "sos_diff_avg_opp_elo10":  (h_elo10 - a_elo10) / _ELO_NORM,
            "sos_home_adj_win_rate10": h_wr10,
            "sos_away_adj_win_rate10": a_wr10,
        }

    def feature_names(self) -> list[str]:
        return [
            "sos_home_avg_opp_elo5",  "sos_home_avg_opp_elo10",
            "sos_away_avg_opp_elo5",  "sos_away_avg_opp_elo10",
            "sos_diff_avg_opp_elo10",
            "sos_home_adj_win_rate10", "sos_away_adj_win_rate10",
        ]

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _ensure_cache(self, data: pd.DataFrame) -> None:
        if self._data_id == id(data):
            return
        self._data_id = id(data)
        self._build_cache(data)

    def _build_cache(self, data: pd.DataFrame) -> None:
        if data.empty or "date" not in data.columns:
            return
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        elos: dict[str, float] = {}
        hist: dict[str, list[tuple[float, float]]] = {}
        self._snapshots = {}

        for date, group in df.groupby("date", sort=True):
            date_str = str(date.date())
            all_teams = set(group["home_team"]) | set(group["away_team"])

            # Snapshot BEFORE this date's matches (causal)
            self._snapshots[date_str] = {
                t: list(hist.get(t, [])) for t in all_teams
            }

            for _, row in group.iterrows():
                ht, at = row["home_team"], row["away_team"]
                re = elos.get(ht, _DEFAULT_ELO)
                ae = elos.get(at, _DEFAULT_ELO)
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                h_score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0

                hist.setdefault(ht, []).append((ae, _pts(hg, ag, True)))
                hist.setdefault(at, []).append((re, _pts(hg, ag, False)))

                elos[ht] = _updated(re, ae, h_score)
                elos[at] = _updated(ae, re, 1.0 - h_score)

        self._final = {t: list(v) for t, v in hist.items()}
