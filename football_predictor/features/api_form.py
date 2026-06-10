"""Recent international form from API-Football (/fixtures?team=X&last=10).

Pre-cached for all 48 WC 2026 teams via scripts/fetch_api_form.py.
Returns zeros for teams not in the cache (historical training rows).
Refresh cache before each round: python3.11 scripts/fetch_api_form.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd

from football_predictor.features.base import FeatureModule

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "api_form_cache.json"
)

# Match-type weight applied when computing weighted form score
_MATCH_WEIGHTS: dict[str, float] = {
    "World Cup": 1.5,
    "World Cup - Qualification": 1.0,
    "UEFA Nations League": 0.7,
    "CONCACAF Nations League": 0.7,
    "African Nations Championship": 0.7,
    "AFC Asian Cup": 1.0,
    "Copa America": 1.0,
    "Africa Cup of Nations": 1.0,
    "UEFA Euro": 1.0,
    "Friendlies": 0.3,
}

# Map our canonical WC team names → API-Football team names
_API_NAME_MAP: dict[str, str] = {
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia-Herzegovina":     "Bosnia & Herzegovina",
    "Cabo Verde":             "Cape Verde Islands",
    "Czechia":                "Czech Republic",
    "Czech Republic":         "Czech Republic",
    "DR Congo":               "Congo DR",
    "IR Iran":                "Iran",
    "Ivory Coast":            "Ivory Coast",
    "South Korea":            "Korea Republic",
    "United States":          "USA",
    "Türkiye":                "Türkiye",
}

_FEATURE_KEYS = [
    "win_rate_last5",  "weighted_pts_last5",
    "win_rate_last10", "draw_rate_last10", "loss_rate_last10",
    "goals_scored_avg_last10", "goals_conceded_avg_last10", "gd_avg_last10",
    "clean_sheet_rate_last10", "weighted_pts_last10",
    "win_rate_last15", "weighted_pts_last15",
    "win_rate_last20", "draw_rate_last20", "loss_rate_last20",
    "goals_scored_avg_last20", "goals_conceded_avg_last20", "gd_avg_last20",
    "clean_sheet_rate_last20", "weighted_pts_last20",
    "has_api_form",
]
_ZERO: dict[str, float] = {k: 0.0 for k in _FEATURE_KEYS}


def _weight(competition: str) -> float:
    for k, w in _MATCH_WEIGHTS.items():
        if k.lower() in competition.lower():
            return w
    return 0.5  # unknown competition


def _compute_features(fixtures: list[dict], team_id: int) -> dict[str, float]:
    """Compute form features from a list of fixture dicts (last 20, newest first)."""
    if not fixtures:
        return _ZERO.copy()

    def result(f: dict) -> str:
        home = f["teams"]["home"]
        away = f["teams"]["away"]
        winner = home["winner"] if home["id"] == team_id else away["winner"]
        if winner is True:
            return "W"
        if winner is False:
            return "L"
        return "D"

    def goals_for(f: dict) -> float:
        if f["teams"]["home"]["id"] == team_id:
            return float(f["goals"]["home"] or 0)
        return float(f["goals"]["away"] or 0)

    def goals_against(f: dict) -> float:
        if f["teams"]["home"]["id"] == team_id:
            return float(f["goals"]["away"] or 0)
        return float(f["goals"]["home"] or 0)

    def _window(n: int) -> tuple:
        sliced = fixtures[:n]
        res = [result(f) for f in sliced]
        gf  = [goals_for(f) for f in sliced]
        ga  = [goals_against(f) for f in sliced]
        comp = [f["league"]["name"] for f in sliced]
        return res, gf, ga, comp

    def _stats(res: list, gf: list, ga: list, comp: list) -> dict:
        n = max(len(res), 1)
        pts_w = sum(
            (3 if r == "W" else 1 if r == "D" else 0) * _weight(c)
            for r, c in zip(res, comp)
        )
        return {
            "win_rate":        res.count("W") / n,
            "draw_rate":       res.count("D") / n,
            "loss_rate":       res.count("L") / n,
            "goals_scored":    sum(gf) / n,
            "goals_conceded":  sum(ga) / n,
            "gd":              (sum(gf) - sum(ga)) / n,
            "clean_sheets":    sum(1 for g in ga if g == 0) / n,
            "weighted_pts":    pts_w,
        }

    s5  = _stats(*_window(5))
    s10 = _stats(*_window(10))
    s15 = _stats(*_window(15))
    s20 = _stats(*_window(20))

    return {
        "win_rate_last5":          s5["win_rate"],
        "weighted_pts_last5":      s5["weighted_pts"],
        "win_rate_last10":         s10["win_rate"],
        "draw_rate_last10":        s10["draw_rate"],
        "loss_rate_last10":        s10["loss_rate"],
        "goals_scored_avg_last10": s10["goals_scored"],
        "goals_conceded_avg_last10": s10["goals_conceded"],
        "gd_avg_last10":           s10["gd"],
        "clean_sheet_rate_last10": s10["clean_sheets"],
        "weighted_pts_last10":     s10["weighted_pts"],
        "win_rate_last15":         s15["win_rate"],
        "weighted_pts_last15":     s15["weighted_pts"],
        "win_rate_last20":         s20["win_rate"],
        "draw_rate_last20":        s20["draw_rate"],
        "loss_rate_last20":        s20["loss_rate"],
        "goals_scored_avg_last20": s20["goals_scored"],
        "goals_conceded_avg_last20": s20["goals_conceded"],
        "gd_avg_last20":           s20["gd"],
        "clean_sheet_rate_last20": s20["clean_sheets"],
        "weighted_pts_last20":     s20["weighted_pts"],
        "has_api_form": 1.0,
    }


def _load_cache() -> dict[str, dict[str, float]]:
    path = os.path.abspath(_CACHE_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw: dict = json.load(f)
    result: dict[str, dict[str, float]] = {}
    for team_name, entry in raw.items():
        team_id = entry["team_id"]
        fixtures = entry["fixtures"]
        result[team_name] = _compute_features(fixtures, team_id)
    return result


class APIFormFeatures(FeatureModule):
    """Pre-cached API-Football form for WC 2026 teams (last 10 internationals).

    Refresh with: python3.11 scripts/fetch_api_form.py
    For historical training matches the features return zero (no cache entry).
    """

    name = "api_form"

    def __init__(self) -> None:
        self._cache = _load_cache()

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def _lookup(self, team: str) -> dict[str, float]:
        api_name = _API_NAME_MAP.get(team, team)
        return self._cache.get(api_name, _ZERO)

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        h = self._lookup(match["home_team"])
        a = self._lookup(match["away_team"])

        out: dict[str, float] = {}
        for k in _FEATURE_KEYS:
            out[f"api_form_home_{k}"] = h[k]
            out[f"api_form_away_{k}"] = a[k]
        out["api_form_diff_win_rate"] = h["win_rate_last10"] - a["win_rate_last10"]
        out["api_form_diff_gd_avg"] = h["gd_avg_last10"] - a["gd_avg_last10"]
        out["api_form_diff_weighted_pts"] = h["weighted_pts_last10"] - a["weighted_pts_last10"]
        return out

    def feature_names(self) -> list[str]:
        names = []
        for side in ("home", "away"):
            for k in _FEATURE_KEYS:
                names.append(f"api_form_{side}_{k}")
        names += [
            "api_form_diff_win_rate",
            "api_form_diff_gd_avg",
            "api_form_diff_weighted_pts",
        ]
        return names
