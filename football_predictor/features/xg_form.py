"""Rolling xG (expected goals) features from WC 2026 qualifier data.

DATA SOURCES (merged, preferring primary over fallback per-team)
────────────────────────────────────────────────────────────────
1. football-data.co.uk WorldCup2026.xlsx  — 339 matches with xG
   Covers: UEFA, AFC, CONMEBOL, some CONCACAF
2. FBref (StatsBomb-powered)              — scraped via fetch_xg_fbref.py
   Covers: CAF (Morocco/Egypt/Senegal/Ghana etc.), CONCACAF (Mexico/USA/Canada)
   Cache: data/xg_fbref.json

WHY xG MATTERS
───────────────
Goals are noisy — a team can win 1-0 against the run of play repeatedly.
xG measures the *quality* of chances created and conceded, which is a more
stable predictor of future performance than raw goal counts.

FEATURES (13)
──────────────
  xg_home_avg5, xg_home_avg10   — home team avg xG per game (last 5/10 matches)
  xga_home_avg5, xga_home_avg10 — home team avg xG against
  xg_away_avg5, xg_away_avg10   — away team avg xG
  xga_away_avg5, xga_away_avg10 — away team avg xG against
  xg_diff_home, xg_diff_away    — xG - xGA (net dominance, last 10)
  xg_ratio_home, xg_ratio_away  — xG / xGA (Pythagorean-like strength)
  xg_available                  — 1.0 if data found for both teams, else 0.0
"""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Optional

import pandas as pd

from football_predictor.features.base import FeatureModule
from football_predictor.data.sources.football_data_co_uk import build_xg_timelines

_FBREF_CACHE = Path(__file__).resolve().parents[2] / "data" / "xg_fbref.json"


def _load_fbref_timelines() -> dict[str, list[tuple]]:
    """Load FBref xG cache (CAF + CONCACAF) into per-team timeline format."""
    if not _FBREF_CACHE.exists():
        return {}
    try:
        with open(_FBREF_CACHE) as f:
            rows = json.load(f)
    except Exception:
        return {}
    timelines: dict[str, list[tuple]] = {}
    for r in sorted(rows, key=lambda x: x["date"]):
        d = pd.Timestamp(r["date"])
        h, a = r["home"], r["away"]
        hxg, axg = float(r["xg_home"]), float(r["xg_away"])
        timelines.setdefault(h, []).append((d, hxg, axg))
        timelines.setdefault(a, []).append((d, axg, hxg))
    return timelines

_DEFAULT_XG = 1.15    # population mean xG per team per match (international)
_DEFAULT_XGA = 1.15


def _rolling_mean(
    timeline: list[tuple],
    before_date: pd.Timestamp,
    window: int,
    idx: int,  # 1 = xg_for, 2 = xg_against
) -> Optional[float]:
    """Mean of last `window` values strictly before `before_date`."""
    lo = bisect_left([t[0] for t in timeline], before_date)
    relevant = timeline[max(0, lo - window): lo]
    if not relevant:
        return None
    return sum(r[idx] for r in relevant) / len(relevant)


class XGFormFeatures(FeatureModule):
    """Rolling xG / xGA per team from WC 2026 qualifier matches."""

    name = "xg_form"

    def __init__(self) -> None:
        self._timelines: Optional[dict] = None

    def _ensure(self) -> None:
        if self._timelines is None:
            # Primary: football-data.co.uk (UEFA/AFC/CONMEBOL/some CONCACAF)
            self._timelines = dict(build_xg_timelines())
            # Secondary: FBref (CAF + missing CONCACAF) — merge, not overwrite
            fbref = _load_fbref_timelines()
            for team, tl in fbref.items():
                if team not in self._timelines:
                    self._timelines[team] = tl
                else:
                    # Merge and sort chronologically, dedup by date
                    merged = sorted(set(self._timelines[team] + tl), key=lambda x: x[0])
                    self._timelines[team] = merged

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure()

        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        target = pd.Timestamp(match["date"])

        tl_h = self._timelines.get(home, [])
        tl_a = self._timelines.get(away, [])

        h5  = _rolling_mean(tl_h, target, 5,  1) or _DEFAULT_XG
        h10 = _rolling_mean(tl_h, target, 10, 1) or _DEFAULT_XG
        ha5  = _rolling_mean(tl_h, target, 5,  2) or _DEFAULT_XGA
        ha10 = _rolling_mean(tl_h, target, 10, 2) or _DEFAULT_XGA

        a5  = _rolling_mean(tl_a, target, 5,  1) or _DEFAULT_XG
        a10 = _rolling_mean(tl_a, target, 10, 1) or _DEFAULT_XG
        aa5  = _rolling_mean(tl_a, target, 5,  2) or _DEFAULT_XGA
        aa10 = _rolling_mean(tl_a, target, 10, 2) or _DEFAULT_XGA

        available = 1.0 if (tl_h and tl_a) else 0.0

        return {
            "xg_home_avg5":    h5,
            "xg_home_avg10":   h10,
            "xga_home_avg5":   ha5,
            "xga_home_avg10":  ha10,
            "xg_away_avg5":    a5,
            "xg_away_avg10":   a10,
            "xga_away_avg5":   aa5,
            "xga_away_avg10":  aa10,
            "xg_diff_home":    h10 - ha10,
            "xg_diff_away":    a10 - aa10,
            "xg_ratio_home":   h10 / max(ha10, 0.1),
            "xg_ratio_away":   a10 / max(aa10, 0.1),
            "xg_available":    available,
        }

    def feature_names(self) -> list[str]:
        return [
            "xg_home_avg5", "xg_home_avg10",
            "xga_home_avg5", "xga_home_avg10",
            "xg_away_avg5", "xg_away_avg10",
            "xga_away_avg5", "xga_away_avg10",
            "xg_diff_home", "xg_diff_away",
            "xg_ratio_home", "xg_ratio_away",
            "xg_available",
        ]
