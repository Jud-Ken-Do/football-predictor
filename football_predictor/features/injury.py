"""Pre-match injury and suspension features.

Two data layers:

  Layer 1 — Pre-tournament (Transfermarkt injury history):
    Populated by scripts/fetch_injury_history.py before the tournament.
    Captures chronic injuries, recovery timelines, and squad fitness burden.
    Reads from data/wc2026_player_injuries.json.

  Layer 2 — Match-day (API-Football fixture injuries):
    Populated by scripts/fetch_wc2026_injuries.py before each match day.
    Captures confirmed unavailability (injured/suspended) for a specific fixture.
    Reads from data/wc2026_injuries_cache.json.

For all historical training rows, both layers return 0 — the model learns
that 0 = unknown / full squad available.

Features (20 total):

  From Layer 1 (pre-tournament):
    injury_home_ongoing_n         — players with unresolved injury at match date
    injury_home_ongoing_value_m   — combined market value (€M) of injured players
    injury_home_doubtful_n        — returned from injury <14 days before match
    injury_home_chronic_n         — players with 2+ injuries of same type in last 18m
    injury_home_burden_90d        — total squad days out in last 90 days
    injury_away_* (same 5)
    injury_burden_diff            — home_burden - away_burden (signed)

  From Layer 2 (match-day):
    injury_home_n_unavailable     — total unavailable (injured + suspended)
    injury_away_n_unavailable
    injury_home_n_suspended       — suspended only
    injury_away_n_suspended
    injury_home_gk_out            — 1 if GK unavailable
    injury_away_gk_out
    injury_home_key_pos_out       — attackers + midfielders unavailable
    injury_away_key_pos_out
    injury_squad_diff             — signed count difference

Refresh:
    python3.11 scripts/fetch_injury_history.py        # pre-tournament (once)
    python3.11 scripts/fetch_wc2026_injuries.py       # match-day (before each MD)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from football_predictor.features.base import FeatureModule

logger = logging.getLogger(__name__)

_HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_player_injuries.json"
_INJ_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_injuries_cache.json"
_FIX_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_fixtures_cache.json"

_KEY_POSITIONS = {"attacker", "midfielder", "forward", "winger"}
_GK_POSITIONS  = {"goalkeeper"}

_DOUBTFUL_WINDOW_DAYS = 14   # returned < this many days ago → doubtful
_CHRONIC_WINDOW_DAYS  = 548  # 18 months for chronic injury check
_BURDEN_WINDOW_DAYS   = 90   # rolling window for injury burden

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


# ── Cache loaders ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_history() -> dict:
    if not _HISTORY_PATH.exists():
        return {}
    try:
        with open(_HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_injury_cache() -> dict:
    if not _INJ_CACHE_PATH.exists():
        return {}
    try:
        with open(_INJ_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _build_fixture_lookup() -> dict[tuple, int]:
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


# ── Layer 1: pre-tournament features ──────────────────────────────────────────

def _pre_tournament_features(team: str, match_date: str) -> dict[str, float]:
    """Compute injury burden features for a team as of match_date (ISO string)."""
    history = _load_history()
    players = history.get(team, [])

    if not players:
        return {
            "ongoing_n":       0.0,
            "ongoing_value_m": 0.0,
            "doubtful_n":      0.0,
            "chronic_n":       0.0,
            "burden_90d":      0.0,
        }

    ref = date.fromisoformat(match_date)
    doubtful_cutoff = (ref - timedelta(days=_DOUBTFUL_WINDOW_DAYS)).isoformat()
    chronic_cutoff  = (ref - timedelta(days=_CHRONIC_WINDOW_DAYS)).isoformat()
    burden_cutoff   = (ref - timedelta(days=_BURDEN_WINDOW_DAYS)).isoformat()
    ref_str = match_date

    ongoing_n       = 0.0
    ongoing_value_m = 0.0
    doubtful_n      = 0.0
    burden_days     = 0.0

    chronic_n = 0.0
    for p in players:
        injuries = p.get("injuries", [])
        value    = float(p.get("market_value_m", 0.0))

        # ── ongoing: no return date, injury started before match ──────────────
        for inj in injuries:
            frm   = inj.get("from") or ""
            until = inj.get("until")
            if until is None and frm and frm < ref_str:
                ongoing_n       += 1.0
                ongoing_value_m += value
                break  # count each player once

        # ── doubtful: returned within last DOUBTFUL_WINDOW_DAYS ───────────────
        for inj in injuries:
            until = inj.get("until")
            if until and doubtful_cutoff <= until < ref_str:
                doubtful_n += 1.0
                break

        # ── burden: total days this player was out in last 90d ────────────────
        for inj in injuries:
            frm   = inj.get("from") or ""
            until = inj.get("until") or ref_str
            if frm >= burden_cutoff or until >= burden_cutoff:
                overlap_start = max(frm, burden_cutoff)
                overlap_end   = min(until, ref_str)
                if overlap_end > overlap_start:
                    try:
                        d_start = date.fromisoformat(overlap_start)
                        d_end   = date.fromisoformat(overlap_end)
                        burden_days += (d_end - d_start).days
                    except ValueError:
                        pass

        # ── chronic: 2+ injuries of same broad type in last 18m ───────────────
        recent = [i for i in injuries if (i.get("from") or "") >= chronic_cutoff]
        type_counts: dict[str, int] = {}
        for inj in recent:
            t = inj.get("type", "").lower()
            # Bucket into broad types
            bucket = (
                "muscle"    if any(k in t for k in ("muscle", "muscular", "hamstring", "thigh", "calf", "groin")) else
                "ligament"  if any(k in t for k in ("ligament", "acl", "cruciate", "tendon")) else
                "bone"      if any(k in t for k in ("fracture", "broken", "bone")) else
                "knee"      if "knee" in t else
                "ankle"     if "ankle" in t else
                "back"      if "back" in t else
                "other"
            )
            type_counts[bucket] = type_counts.get(bucket, 0) + 1
        if any(v >= 2 for v in type_counts.values()):
            chronic_n += 1.0

    return {
        "ongoing_n":       ongoing_n,
        "ongoing_value_m": ongoing_value_m,
        "doubtful_n":      doubtful_n,
        "chronic_n":       chronic_n,
        "burden_90d":      burden_days,
    }


# ── Layer 2: match-day features ────────────────────────────────────────────────

def _parse_api_entries(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        raw_team = e.get("team_name", "")
        out.append({
            "name":     e.get("player_name", ""),
            "position": e.get("position", "").lower(),
            "reason":   e.get("reason", "").lower(),
            "team":     _API_TEAM_MAP.get(raw_team, raw_team),
        })
    return out


def _squad_features(players: list[dict]) -> dict[str, float]:
    if not players:
        return {"n": 0.0, "n_susp": 0.0, "gk_out": 0.0, "key_pos": 0.0}
    n_susp  = sum(1 for p in players if "suspension" in p.get("reason", ""))
    gk_out  = float(any(p.get("position", "") in _GK_POSITIONS for p in players))
    key_pos = sum(1 for p in players if p.get("position", "") in _KEY_POSITIONS)
    return {"n": float(len(players)), "n_susp": float(n_susp),
            "gk_out": gk_out, "key_pos": float(key_pos)}


def _matchday_players(home: str, away: str) -> tuple[list, list]:
    fix_lookup = _build_fixture_lookup()
    fix_id = fix_lookup.get((home, away))
    if fix_id is None:
        return [], []
    cache = _load_injury_cache()
    entries = cache.get(str(fix_id), [])
    parsed = _parse_api_entries(entries)
    home_inj = [p for p in parsed if p["team"] == home]
    away_inj = [p for p in parsed if p["team"] == away]
    return home_inj, away_inj


# ── Feature module ─────────────────────────────────────────────────────────────

class InjuryFeatures(FeatureModule):
    """Pre-match injury features from Transfermarkt history + API-Football cache."""

    name = "injury"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        match_date = str(pd.Timestamp(match["date"]).date())

        # ── Layer 1: pre-tournament ────────────────────────────────────────────
        h_pre = _pre_tournament_features(home, match_date)
        a_pre = _pre_tournament_features(away, match_date)

        # ── Layer 2: match-day (manual override or API cache) ─────────────────
        _h_raw = match.get("home_unavailable")
        _a_raw = match.get("away_unavailable")
        home_players = list(_h_raw) if isinstance(_h_raw, (list, tuple)) else []
        away_players = list(_a_raw) if isinstance(_a_raw, (list, tuple)) else []
        if not home_players and not away_players:
            home_players, away_players = _matchday_players(home, away)

        h_md = _squad_features(home_players)
        a_md = _squad_features(away_players)

        return {
            # Pre-tournament layer
            "injury_home_ongoing_n":       h_pre["ongoing_n"],
            "injury_home_ongoing_value_m": h_pre["ongoing_value_m"],
            "injury_home_doubtful_n":      h_pre["doubtful_n"],
            "injury_home_chronic_n":       h_pre["chronic_n"],
            "injury_home_burden_90d":      h_pre["burden_90d"],
            "injury_away_ongoing_n":       a_pre["ongoing_n"],
            "injury_away_ongoing_value_m": a_pre["ongoing_value_m"],
            "injury_away_doubtful_n":      a_pre["doubtful_n"],
            "injury_away_chronic_n":       a_pre["chronic_n"],
            "injury_away_burden_90d":      a_pre["burden_90d"],
            "injury_burden_diff":          h_pre["burden_90d"] - a_pre["burden_90d"],
            # Match-day layer
            "injury_home_n_unavailable":   h_md["n"],
            "injury_away_n_unavailable":   a_md["n"],
            "injury_home_n_suspended":     h_md["n_susp"],
            "injury_away_n_suspended":     a_md["n_susp"],
            "injury_home_gk_out":          h_md["gk_out"],
            "injury_away_gk_out":          a_md["gk_out"],
            "injury_home_key_pos_out":     h_md["key_pos"],
            "injury_away_key_pos_out":     a_md["key_pos"],
            "injury_squad_diff":           h_md["n"] - a_md["n"],
        }

    def feature_names(self) -> list[str]:
        return [
            "injury_home_ongoing_n",       "injury_home_ongoing_value_m",
            "injury_home_doubtful_n",       "injury_home_chronic_n",
            "injury_home_burden_90d",
            "injury_away_ongoing_n",        "injury_away_ongoing_value_m",
            "injury_away_doubtful_n",       "injury_away_chronic_n",
            "injury_away_burden_90d",
            "injury_burden_diff",
            "injury_home_n_unavailable",    "injury_away_n_unavailable",
            "injury_home_n_suspended",      "injury_away_n_suspended",
            "injury_home_gk_out",           "injury_away_gk_out",
            "injury_home_key_pos_out",      "injury_away_key_pos_out",
            "injury_squad_diff",
        ]
