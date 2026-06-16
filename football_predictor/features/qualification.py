"""Qualification-dominance features (Tier-0, trainable, free).

"Cruised in vs scraped in": how a team performed in its qualifying campaign —
points-per-game, goal-difference-per-game and win rate over its most recent
*qualifier* matches, looked up as-of each match date (no future leakage).

Distinct from `form` (all match types incl. friendlies) and `sos` (opponent-Elo
weighting): this isolates competitive-qualifier results, which are a cleaner read
of where a team stands against its confederation peers than rotation-heavy
friendlies. Computable entirely from `international_results` — no external data.

Causality: every value for a match on date D uses only that team's qualifier
results strictly before D (bisect on a per-team chronological list), so it is a
real trainable feature and works identically for past matches and future WC
fixtures.
"""
from __future__ import annotations

import bisect
from typing import Optional

import pandas as pd

from football_predictor.features.base import FeatureModule

_WINDOW = 10          # most-recent qualifier matches considered
_DEFAULT_PPG = 1.4    # population-ish prior for teams with no qualifier history


class QualificationFeatures(FeatureModule):
    """Rolling qualifier-only points/GD/win-rate per team (as-of-date)."""

    name = "qualification"

    def __init__(self) -> None:
        # team -> (dates: list[Timestamp], rows: list[(pts, gd, win)]) sorted by date
        self._hist: dict[str, tuple[list, list]] = {}
        self._data_id: Optional[int] = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def _ensure(self, data: pd.DataFrame) -> None:
        if self._data_id == id(data):
            return
        self._data_id = id(data)
        self._build(data)

    def _build(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        q = df[df["tournament"].str.contains("qualif", case=False, na=False)]
        q = q.sort_values("date")

        hist: dict[str, list] = {}
        for _, r in q.iterrows():
            hg, ag = float(r["home_goals"]), float(r["away_goals"])
            d = r["date"]
            for team, gf, ga in [(r["home_team"], hg, ag), (r["away_team"], ag, hg)]:
                pts = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
                win = 1.0 if gf > ga else 0.0
                hist.setdefault(team, []).append((d, pts, gf - ga, win))

        self._hist = {t: ([x[0] for x in rows], [(x[1], x[2], x[3]) for x in rows])
                      for t, rows in hist.items()}

    def _stats(self, team: str, before: pd.Timestamp) -> tuple[float, float, float, int]:
        entry = self._hist.get(team)
        if not entry:
            return _DEFAULT_PPG, 0.0, 1.0 / 3.0, 0
        dates, rows = entry
        # rows strictly before `before`, take the last _WINDOW
        hi = bisect.bisect_left(dates, before)
        window = rows[max(0, hi - _WINDOW):hi]
        if not window:
            return _DEFAULT_PPG, 0.0, 1.0 / 3.0, 0
        n = len(window)
        ppg = sum(w[0] for w in window) / n
        gdpg = sum(w[1] for w in window) / n
        wr = sum(w[2] for w in window) / n
        return ppg, gdpg, wr, n

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure(data)
        d = pd.Timestamp(match["date"])
        h_ppg, h_gd, h_wr, h_n = self._stats(match["home_team"], d)
        a_ppg, a_gd, a_wr, a_n = self._stats(match["away_team"], d)
        return {
            "qual_home_ppg": h_ppg, "qual_home_gdpg": h_gd,
            "qual_home_win_rate": h_wr, "qual_home_n": float(h_n),
            "qual_away_ppg": a_ppg, "qual_away_gdpg": a_gd,
            "qual_away_win_rate": a_wr, "qual_away_n": float(a_n),
            "qual_ppg_diff": h_ppg - a_ppg,
            "qual_gd_diff": h_gd - a_gd,
        }

    def feature_names(self) -> list[str]:
        return [
            "qual_home_ppg", "qual_home_gdpg", "qual_home_win_rate", "qual_home_n",
            "qual_away_ppg", "qual_away_gdpg", "qual_away_win_rate", "qual_away_n",
            "qual_ppg_diff", "qual_gd_diff",
        ]
