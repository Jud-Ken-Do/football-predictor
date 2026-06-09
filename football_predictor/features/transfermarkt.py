"""Transfermarkt squad market value features.

Squad market value is the strongest free non-odds predictor of WC match
outcomes (Groll et al. 2018, Statistical Modelling). Reflects squad depth,
star quality, and economic investment in football.

Data source: transfermarkt.com national team pages.
Cache file:  data/transfermarkt_wc2026.json
Populate:    python3.11 scripts/fetch_transfermarkt.py

Features (5):
  tm_home_value      — log(squad value €M + 1) for home team
  tm_away_value      — log(squad value €M + 1) for away team
  tm_value_ratio     — log(home / away) — positive = home richer squad
  tm_home_value_pct  — home value as % of combined (0-1)
  tm_available       — 1 if data loaded, 0 otherwise (historical training rows)

For historical training rows both tm_home_value and tm_away_value are 0,
so the model learns that 0 = no data rather than a weak team.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from football_predictor.features.base import FeatureModule

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "transfermarkt_wc2026.json"

# Canonical name aliases (our names → keys used in the JSON cache)
_ALIASES: dict[str, str] = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic":       "South Korea",
    "USA":                  "United States",
    "Iran":                 "IR Iran",
    "Ivory Coast":          "Ivory Coast",
    "Congo DR":             "DR Congo",
    "Curacao":              "Curaçao",
}


@lru_cache(maxsize=1)
def _load() -> dict[str, float]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _value(team: str, data: dict[str, float]) -> float:
    key = _ALIASES.get(team, team)
    return float(data.get(key, data.get(team, 0.0)))


class TransfermarktFeatures(FeatureModule):
    """Squad market value features from Transfermarkt."""

    name = "transfermarkt"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        tm_data = _load()
        if not tm_data:
            return {k: 0.0 for k in self.feature_names()}

        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))

        hv = _value(home, tm_data)
        av = _value(away, tm_data)

        if hv == 0.0 and av == 0.0:
            return {k: 0.0 for k in self.feature_names()}

        log_hv = math.log1p(hv)
        log_av = math.log1p(av)
        ratio  = math.log((hv + 1) / (av + 1))
        pct    = hv / (hv + av) if (hv + av) > 0 else 0.5

        return {
            "tm_home_value":     log_hv,
            "tm_away_value":     log_av,
            "tm_value_ratio":    ratio,
            "tm_home_value_pct": pct,
            "tm_available":      1.0,
        }

    def feature_names(self) -> list[str]:
        return [
            "tm_home_value",
            "tm_away_value",
            "tm_value_ratio",
            "tm_home_value_pct",
            "tm_available",
        ]
