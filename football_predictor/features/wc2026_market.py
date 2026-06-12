"""WC 2026 market odds as post-processing context features.

Reads data/wc2026_odds_cache.json (fetched by scripts/fetch_wc2026_odds.py).

Features (9):
  market_p_home        — 1X2 implied P(home win)
  market_p_draw        — 1X2 implied P(draw)
  market_p_away        — 1X2 implied P(away win)
  market_over25        — implied P(over 2.5 goals)
  market_btts          — implied P(both teams score)
  market_cs_home       — implied P(home clean sheet)
  market_cs_away       — implied P(away clean sheet)
  market_ah_line       — Asian Handicap line, home perspective (negative = home favoured)
  market_available     — 1.0 if odds found, 0.0 otherwise

These are WC_CONTEXT_MODULES only — zero for all historical rows (covariate shift).
Used by wc_context.market_odds_adjust() to blend market-implied λ with model λ.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from football_predictor.features.base import FeatureModule

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_odds_cache.json"


@lru_cache(maxsize=1)
def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        from football_predictor.data.wc2026 import GROUP_STAGE_SCHEDULE, normalise
        _sched_dates = {
            (normalise(m["home_team"]), normalise(m["away_team"])): str(m["date"])
            for m in GROUP_STAGE_SCHEDULE
        }
        raw = json.loads(_CACHE_PATH.read_text())
        lookup = {}
        for fid, d in raw.items():
            home = normalise(d.get("home_team", ""))
            away = normalise(d.get("away_team", ""))
            if not (home and away):
                continue
            date = _sched_dates.get((home, away)) or _sched_dates.get((away, home))
            if date:
                lookup[(home, away, date)] = d
        return lookup
    except Exception:
        return {}


_FLAT = {
    "market_p_home": 0.0, "market_p_draw": 0.0, "market_p_away": 0.0,
    "market_over25": 0.0, "market_btts": 0.0,
    "market_cs_home": 0.0, "market_cs_away": 0.0,
    "market_ah_line": 0.0, "market_available": 0.0,
}


class WC2026MarketFeatures(FeatureModule):
    """Pre-match bookmaker odds for WC 2026 fixtures (post-processing context only)."""

    name = "wc2026_market"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        lookup = _load_cache()
        if not lookup:
            return _FLAT.copy()

        from football_predictor.data.wc2026 import normalise
        home = normalise(str(match.get("home_team", "")))
        away = normalise(str(match.get("away_team", "")))
        date = str(pd.Timestamp(match["date"]).date())

        d = lookup.get((home, away, date)) or lookup.get((away, home, date))
        if d is None:
            return _FLAT.copy()

        # If reversed match, swap home/away signals
        reversed_match = (away, home, date) in lookup and (home, away, date) not in lookup

        p1 = d.get("p1", 0.0)
        px = d.get("px", 0.0)
        p2 = d.get("p2", 0.0)
        ah = d.get("ah_line", 0.0)

        if reversed_match:
            p1, p2 = p2, p1
            ah = -ah

        return {
            "market_p_home":    round(p1, 4),
            "market_p_draw":    round(px, 4),
            "market_p_away":    round(p2, 4),
            "market_over25":    round(d.get("over25", 0.0), 4),
            "market_btts":      round(d.get("btts", 0.0), 4),
            "market_cs_home":   round(d.get("cs_home", 0.0) if not reversed_match else d.get("cs_away", 0.0), 4),
            "market_cs_away":   round(d.get("cs_away", 0.0) if not reversed_match else d.get("cs_home", 0.0), 4),
            "market_ah_line":   round(ah, 2),
            "market_available": 1.0,
        }

    def feature_names(self) -> list[str]:
        return list(_FLAT.keys())
