from __future__ import annotations

import math

import pandas as pd

from football_predictor.features.base import FeatureModule
from football_predictor.constants import (
    ELO_K_FACTOR_INTERNATIONAL as ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_NEUTRAL_ADVANTAGE,
)


class EloFeatures(FeatureModule):
    """Elo rating features computed from match history.

    Pre-computes all ratings in a single chronological pass (O(n)) on the
    first call, then serves lookups in O(1). Previously O(n²) — rebuilt
    from scratch for every match in the dataset.

    Home advantage is applied for non-neutral historical matches when
    building ratings. At prediction time (WC = neutral venue), HA = 0.
    """

    name = "elo"
    _DEFAULT_RATING = 1500.0

    def __init__(self) -> None:
        self._snapshot: dict[str, dict[str, float]] = {}  # date_str -> {team -> rating}
        self._final: dict[str, float] = {}
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home = match["home_team"]
        away = match["away_team"]

        # Per-date snapshots only contain teams that PLAYED on that date.
        # A team absent from an existing snapshot (e.g. predicting a fixture
        # on a day other matches were already recorded) must fall back to its
        # latest rating — NOT to the 1500 default, which would silently rate
        # Brazil equal to Curaçao for every same-day fixture.
        snap = self._snapshot.get(date_str, {})
        r_home_raw = snap.get(home)
        if r_home_raw is None:
            r_home_raw = self._final.get(home, self._DEFAULT_RATING)
        r_away_raw = snap.get(away)
        if r_away_raw is None:
            r_away_raw = self._final.get(away, self._DEFAULT_RATING)

        is_neutral = bool(match.get("neutral", False))
        ha = ELO_NEUTRAL_ADVANTAGE if is_neutral else ELO_HOME_ADVANTAGE
        r_home = r_home_raw + ha

        exp_home = self._expected(r_home, r_away_raw)

        return {
            "elo_home_rating": r_home_raw,
            "elo_away_rating": r_away_raw,
            "elo_diff": r_home - r_away_raw,
            "elo_home_win_prob": exp_home,
            "elo_away_win_prob": 1.0 - exp_home,
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

        ratings: dict[str, float] = {}
        self._snapshot = {}

        for date, group in df.groupby("date", sort=True):
            date_str = str(date.date())
            # Snapshot BEFORE processing this date's matches
            snap = {t: ratings.get(t, self._DEFAULT_RATING)
                    for t in set(group["home_team"]) | set(group["away_team"])}
            self._snapshot[date_str] = snap

            for _, row in group.iterrows():
                home, away = row["home_team"], row["away_team"]
                match_neutral = bool(row.get("neutral", False))
                ha = ELO_NEUTRAL_ADVANTAGE if match_neutral else ELO_HOME_ADVANTAGE

                r_h = ratings.get(home, self._DEFAULT_RATING) + ha
                r_a = ratings.get(away, self._DEFAULT_RATING)
                exp_h = self._expected(r_h, r_a)

                hg, ag = float(row["home_goals"]), float(row["away_goals"])
                s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                s_a = 1.0 - s_h

                mw = float(row.get("match_weight", 1.0))
                ratings[home] = ratings.get(home, self._DEFAULT_RATING) + ELO_K_FACTOR * mw * (s_h - exp_h)
                ratings[away] = ratings.get(away, self._DEFAULT_RATING) + ELO_K_FACTOR * mw * (s_a - (1.0 - exp_h))

        self._final = dict(ratings)

    @staticmethod
    def _expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))

    def feature_names(self) -> list[str]:
        return [
            "elo_home_rating",
            "elo_away_rating",
            "elo_diff",
            "elo_home_win_prob",
            "elo_away_win_prob",
        ]
