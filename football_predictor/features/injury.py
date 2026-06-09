"""Pre-match injury and suspension features via API-Football.

Two data paths:
  1. Live cache (WC 2026): loaded from data/wc2026_injuries_cache.json,
     populated by scripts/fetch_wc2026_injuries.py before each match day.
     Cache maps fixture_id → list of {player_name, team_name, reason, position}.

  2. Manual override: pass `home_unavailable` / `away_unavailable` in the
     match Series as lists of dicts with optional keys:
       {"name": "Mbappé", "position": "Attacker", "reason": "Injury"}

For all historical training rows, both paths return 0 (no injury data
available) — the model learns that 0 = unknown / assume full squad.

To activate: add "injury" to constants.DEFAULT_FEATURE_MODULES.
Refresh cache: python3.11 scripts/fetch_wc2026_injuries.py

Features (9):
  injury_home_n_unavailable    — total unavailable players (injured + suspended)
  injury_away_n_unavailable
  injury_home_n_suspended      — suspended only (often more impactful, known pre-match)
  injury_away_n_suspended
  injury_home_gk_out           — 1 if starting GK is unavailable
  injury_away_gk_out
  injury_home_key_pos_out      — count of attackers + midfielders unavailable
  injury_away_key_pos_out
  injury_squad_diff            — home_unavailable - away_unavailable (signed)
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from football_predictor.features.base import FeatureModule

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_injuries_cache.json"
_FIX_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_fixtures_cache.json"

_KEY_POSITIONS = {"attacker", "midfielder", "forward", "winger"}
_GK_POSITIONS  = {"goalkeeper"}

# API-Football team name → our canonical names
_API_TEAM_MAP: dict[str, str] = {
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Cape Verde Islands":     "Cabo Verde",
    "Czech Republic":         "Czechia",
    "IR Iran":                "IR Iran",
    "Korea Republic":         "Korea Republic",
    "United States":          "United States",
    "Curaçao":                "Curaçao",
    "DR Congo":               "DR Congo",
}


@lru_cache(maxsize=1)
def _load_injury_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _build_fixture_lookup() -> dict[tuple, int]:
    """(home_team, away_team) → fixture_id from fixtures cache."""
    if not _FIX_CACHE_PATH.exists():
        return {}
    try:
        with open(_FIX_CACHE_PATH) as f:
            fixtures = json.load(f)
    except Exception:
        return {}

    lookup: dict[tuple, int] = {}
    for fix in fixtures:
        home = _API_TEAM_MAP.get(fix["teams"]["home"]["name"], fix["teams"]["home"]["name"])
        away = _API_TEAM_MAP.get(fix["teams"]["away"]["name"], fix["teams"]["away"]["name"])
        lookup[(home, away)] = fix["fixture"]["id"]
    return lookup


def _parse_api_entries(entries: list[dict]) -> list[dict]:
    """Normalise API-Football injury entries to a common format."""
    out = []
    for e in entries:
        out.append({
            "name":     e.get("player_name", ""),
            "position": e.get("position", "").lower(),
            "reason":   e.get("reason", "").lower(),
        })
    return out


def _squad_features(players: list[dict]) -> dict[str, float]:
    if not players:
        return {"n": 0.0, "n_susp": 0.0, "gk_out": 0.0, "key_pos": 0.0}

    n_susp  = sum(1 for p in players if "suspension" in p.get("reason", ""))
    gk_out  = float(any(p.get("position", "") in _GK_POSITIONS for p in players))
    key_pos = sum(1 for p in players if p.get("position", "") in _KEY_POSITIONS)

    return {
        "n":        float(len(players)),
        "n_susp":   float(n_susp),
        "gk_out":   gk_out,
        "key_pos":  float(key_pos),
    }


class InjuryFeatures(FeatureModule):
    """Pre-match injury/suspension features from API-Football cache."""

    name = "injury"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))

        # Try manual override first (passed directly in match Series)
        home_players = list(match.get("home_unavailable") or [])
        away_players = list(match.get("away_unavailable") or [])

        # Fall back to API cache
        if not home_players and not away_players:
            home_players, away_players = self._from_cache(home, away)

        h = _squad_features(home_players)
        a = _squad_features(away_players)

        return {
            "injury_home_n_unavailable": h["n"],
            "injury_away_n_unavailable": a["n"],
            "injury_home_n_suspended":   h["n_susp"],
            "injury_away_n_suspended":   a["n_susp"],
            "injury_home_gk_out":        h["gk_out"],
            "injury_away_gk_out":        a["gk_out"],
            "injury_home_key_pos_out":   h["key_pos"],
            "injury_away_key_pos_out":   a["key_pos"],
            "injury_squad_diff":         h["n"] - a["n"],
        }

    def _from_cache(self, home: str, away: str) -> tuple[list, list]:
        fix_lookup = _build_fixture_lookup()
        fix_id = fix_lookup.get((home, away))
        if fix_id is None:
            return [], []

        cache = _load_injury_cache()
        entries = cache.get(str(fix_id), [])
        parsed = _parse_api_entries(entries)

        home_inj = [p for p in parsed if _API_TEAM_MAP.get(p.get("team",""), p.get("team","")) == home]
        away_inj = [p for p in parsed if _API_TEAM_MAP.get(p.get("team",""), p.get("team","")) == away]
        return home_inj, away_inj

    def feature_names(self) -> list[str]:
        return [
            "injury_home_n_unavailable", "injury_away_n_unavailable",
            "injury_home_n_suspended",   "injury_away_n_suspended",
            "injury_home_gk_out",        "injury_away_gk_out",
            "injury_home_key_pos_out",   "injury_away_key_pos_out",
            "injury_squad_diff",
        ]
