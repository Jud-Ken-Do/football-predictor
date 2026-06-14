"""Rest & fixture-congestion features from the match calendar.

For a match on date D, only a team's matches **strictly before** D are counted,
so the features are causal (no leakage of the current or future fixtures). No
external data — derived from the same match history every other module sees.

Signal is strongest inside tournaments (3–4 day turnarounds, occasional extra
rest from earlier kick-off slots) and near-flat between international windows,
where every team is fully rested. Expected to be a weak group-stage signal;
included so XGBoost can pick up fatigue/rust asymmetries where they exist.

FEATURES (8):
  rest_home_days_since_last, rest_away_days_since_last   — days since prev match (capped)
  rest_home_matches_last30,  rest_away_matches_last30    — fixtures in trailing 30 days
  rest_home_matches_last14,  rest_away_matches_last14    — fixtures in trailing 14 days
  rest_days_diff   — home − away days_since_last (positive = home more rested)
  rest_load_diff   — home − away matches_last14 (positive = home more congested)
"""
from __future__ import annotations

import bisect
from collections import defaultdict

import pandas as pd

from football_predictor.features.base import FeatureModule

# days_since_last is capped here: beyond ~a month a team is simply "rested" and
# the exact gap carries little fatigue signal (and avoids a long-tail feature
# dominated by between-window gaps of 60–120+ days).
_MAX_GAP: int = 30


class RestFeatures(FeatureModule):
    """Days since last match + trailing fixture load per side."""

    name = "rest"

    def __init__(self) -> None:
        self._team_ord: dict[str, list[int]] = {}
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    def _ensure(self, data: pd.DataFrame) -> None:
        if self._data_id == id(data):
            return
        self._data_id = id(data)
        acc: dict[str, list[int]] = defaultdict(list)
        dates = pd.to_datetime(data["date"])
        for d, h, a in zip(dates, data["home_team"], data["away_team"]):
            o = d.toordinal()
            acc[h].append(o)
            acc[a].append(o)
        # Sorted ordinal-day lists per team → O(log n) causal lookups.
        self._team_ord = {t: sorted(v) for t, v in acc.items()}

    def _side(self, team: str, qord: int) -> dict[str, float]:
        ords = self._team_ord.get(team)
        if not ords:
            return {"days_since_last": float(_MAX_GAP),
                    "matches_last30": 0.0, "matches_last14": 0.0}
        # bisect_left → count of this team's matches strictly before date D.
        # The current fixture's own date (== qord) is excluded → causal.
        idx = bisect.bisect_left(ords, qord)
        days = float(_MAX_GAP) if idx == 0 else float(min(qord - ords[idx - 1], _MAX_GAP))
        lo30 = bisect.bisect_left(ords, qord - 30)
        lo14 = bisect.bisect_left(ords, qord - 14)
        return {
            "days_since_last": days,
            "matches_last30": float(idx - lo30),
            "matches_last14": float(idx - lo14),
        }

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure(data)
        qord = pd.Timestamp(match["date"]).toordinal()
        h = self._side(match["home_team"], qord)
        a = self._side(match["away_team"], qord)
        out: dict[str, float] = {}
        for k, v in h.items():
            out[f"rest_home_{k}"] = v
        for k, v in a.items():
            out[f"rest_away_{k}"] = v
        out["rest_days_diff"] = h["days_since_last"] - a["days_since_last"]
        out["rest_load_diff"] = h["matches_last14"] - a["matches_last14"]
        return out

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for side in ("home", "away"):
            names += [
                f"rest_{side}_days_since_last",
                f"rest_{side}_matches_last30",
                f"rest_{side}_matches_last14",
            ]
        names += ["rest_days_diff", "rest_load_diff"]
        return names
