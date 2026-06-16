"""All-time World Cup pedigree (Tier-1, trainable).

Traditional WC powers (Brazil, Germany, Argentina, Uruguay) systematically
overperform their ratings at World Cups — a signal repeatedly confirmed in the
forecasting literature (Groll et al.). We capture it *from the data*, not a
hand-typed table: the team's World Cup history (editions played, WC match
win-rate, WC goals-per-game) computed from the full 1930-present `FIFA World Cup`
results, looked up **as-of each match date** (only WC editions strictly before
the match count) so there is no future leakage.

Historic names are aliased to current ones (West Germany→Germany,
Czechoslovakia→Czechia, Zaire→DR Congo, …) so a country's pedigree isn't split.
"""
from __future__ import annotations

import bisect
from typing import Optional

import pandas as pd

from football_predictor.features.base import FeatureModule

# Historic / predecessor names → current canonical name.
_ALIAS = {
    "West Germany": "Germany", "East Germany": "Germany",
    "Czechoslovakia": "Czechia", "Zaire": "DR Congo",
    "Soviet Union": "Russia", "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia", "Dutch East Indies": "Indonesia",
}


class WCPedigreeFeatures(FeatureModule):
    """As-of-date World Cup appearances / win-rate / goals-per-game per team."""

    name = "wc_pedigree"

    def __init__(self) -> None:
        # team -> (sorted dates, rows[(year, win, gf, ga)])
        self._hist: dict[str, tuple[list, list]] = {}
        self._built = False

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def _build(self) -> None:
        # Independent of the training context: pull the FULL WC history.
        try:
            from football_predictor.data.sources.international_results import fetch_international_results
            wc = fetch_international_results(from_year=1930, tournaments=["FIFA World Cup"])
        except Exception:
            self._hist = {}
            self._built = True
            return
        wc = wc.copy()
        wc["date"] = pd.to_datetime(wc["date"])
        wc = wc.sort_values("date")
        hist: dict[str, list] = {}
        for _, r in wc.iterrows():
            d = r["date"]
            yr = d.year
            hg, ag = float(r["home_goals"]), float(r["away_goals"])
            for team, gf, ga in [(r["home_team"], hg, ag), (r["away_team"], ag, hg)]:
                team = _ALIAS.get(team, team)
                win = 1.0 if gf > ga else 0.0
                hist.setdefault(team, []).append((d, yr, win, gf, ga))
        self._hist = {t: ([x[0] for x in rows], [(x[1], x[2], x[3], x[4]) for x in rows])
                      for t, rows in hist.items()}
        self._built = True

    def _stats(self, team: str, before: pd.Timestamp) -> dict[str, float]:
        if not self._built:
            self._build()
        entry = self._hist.get(_ALIAS.get(team, team))
        if not entry:
            return {"apps": 0.0, "matches": 0.0, "win_rate": 0.0, "gpg": 0.0}
        dates, rows = entry
        # PRIOR pedigree only: count WC editions strictly before the current
        # tournament (year < match year). Excluding the in-progress edition keeps
        # this a clean "history" signal (current-WC form is the job of `form` /
        # Kalman) and avoids a backtest inconsistency — the frozen pre-WC context
        # can't see within-tournament matches that a date cut would let in.
        hi = bisect.bisect_left(dates, before)
        cur_year = before.year
        past = [r for r in rows[:hi] if r[0] < cur_year]
        if not past:
            return {"apps": 0.0, "matches": 0.0, "win_rate": 0.0, "gpg": 0.0}
        years = {r[0] for r in past}
        n = len(past)
        return {
            "apps": float(len(years)),
            "matches": float(n),
            "win_rate": sum(r[1] for r in past) / n,
            "gpg": sum(r[2] for r in past) / n,
        }

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        d = pd.Timestamp(match["date"])
        h = self._stats(match["home_team"], d)
        a = self._stats(match["away_team"], d)
        return {
            "wc_ped_home_apps": h["apps"], "wc_ped_home_matches": h["matches"],
            "wc_ped_home_win_rate": h["win_rate"], "wc_ped_home_gpg": h["gpg"],
            "wc_ped_away_apps": a["apps"], "wc_ped_away_matches": a["matches"],
            "wc_ped_away_win_rate": a["win_rate"], "wc_ped_away_gpg": a["gpg"],
            "wc_ped_apps_diff": h["apps"] - a["apps"],
            "wc_ped_win_rate_diff": h["win_rate"] - a["win_rate"],
        }

    def feature_names(self) -> list[str]:
        return [
            "wc_ped_home_apps", "wc_ped_home_matches", "wc_ped_home_win_rate", "wc_ped_home_gpg",
            "wc_ped_away_apps", "wc_ped_away_matches", "wc_ped_away_win_rate", "wc_ped_away_gpg",
            "wc_ped_apps_diff", "wc_ped_win_rate_diff",
        ]
