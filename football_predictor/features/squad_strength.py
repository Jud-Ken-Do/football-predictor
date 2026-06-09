from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from football_predictor.features.base import FeatureModule

_LONG_WINDOW = 40
_COMPETITIVE_KEYWORDS = ["world cup", "qualif", "copa", "euro", "gold cup", "nations league",
                          "africa cup", "asian cup", "confederation"]

_NEUTRAL_METRICS = {
    "attack": 1.4,
    "defense": 1.4,
    "clean_sheet": 0.3,
    "all_goals": 1.4,
    "quality": 0.0,
}


def _is_competitive(tournament: str) -> bool:
    t = tournament.lower()
    return any(k in t for k in _COMPETITIVE_KEYWORDS)


class SquadStrengthFeatures(FeatureModule):
    """Long-term squad quality from match history, pre-computed in O(n).

    Uses a 40-match rolling window per team, processed chronologically once.
    Captures attack/defence dimensions from competitive matches only,
    giving independent signal from short-window FormFeatures.
    """

    name = "squad_strength"

    def __init__(self) -> None:
        self._cache: dict[tuple, dict[str, float]] = {}  # (team, date_str) -> metrics
        self._final: dict[str, dict[str, float]] = {}    # team -> most recent metrics
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home = match["home_team"]
        away = match["away_team"]

        h = self._cache.get((home, date_str)) or self._final.get(home) or _NEUTRAL_METRICS
        a = self._cache.get((away, date_str)) or self._final.get(away) or _NEUTRAL_METRICS

        return {
            "squad_home_attack":      h["attack"],
            "squad_home_defense":     h["defense"],
            "squad_home_clean_sheet": h["clean_sheet"],
            "squad_home_all_goals":   h["all_goals"],
            "squad_home_quality":     h["quality"],
            "squad_away_attack":      a["attack"],
            "squad_away_defense":     a["defense"],
            "squad_away_clean_sheet": a["clean_sheet"],
            "squad_away_all_goals":   a["all_goals"],
            "squad_away_quality":     a["quality"],
            "matchup_attack_vs_def":  h["attack"] - a["defense"],
            "matchup_def_vs_attack":  a["attack"] - h["defense"],
            "matchup_quality_diff":   h["quality"] - a["quality"],
            "matchup_clean_sheet_diff": h["clean_sheet"] - a["clean_sheet"],
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

        team_history: dict[str, deque] = {}
        self._cache = {}

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hg = float(row["home_goals"])
            ag = float(row["away_goals"])
            date_str = str(row["date"].date())
            competitive = _is_competitive(str(row.get("tournament", "")))

            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                key = (team, date_str)
                if key not in self._cache:
                    hist = team_history.get(team, deque())
                    self._cache[key] = _metrics_from_history(hist)

            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                if team not in team_history:
                    team_history[team] = deque(maxlen=_LONG_WINDOW)
                team_history[team].append({"gf": gf, "ga": ga, "competitive": competitive})

        self._final = {team: _metrics_from_history(hist) for team, hist in team_history.items()}

    def feature_names(self) -> list[str]:
        dims = ("attack", "defense", "clean_sheet", "all_goals", "quality")
        return (
            [f"squad_home_{d}" for d in dims]
            + [f"squad_away_{d}" for d in dims]
            + ["matchup_attack_vs_def", "matchup_def_vs_attack",
               "matchup_quality_diff", "matchup_clean_sheet_diff"]
        )


def _metrics_from_history(history: deque) -> dict[str, float]:
    if not history:
        return dict(_NEUTRAL_METRICS)

    gf_all, ga_all, clean, gf_comp, ga_comp = [], [], [], [], []
    for m in history:
        gf_all.append(m["gf"])
        ga_all.append(m["ga"])
        clean.append(1.0 if m["ga"] == 0 else 0.0)
        if m["competitive"]:
            gf_comp.append(m["gf"])
            ga_comp.append(m["ga"])

    attack = float(np.mean(gf_comp)) if gf_comp else float(np.mean(gf_all))
    defense = float(np.mean(ga_comp)) if ga_comp else float(np.mean(ga_all))
    return {
        "attack":      attack,
        "defense":     defense,
        "clean_sheet": float(np.mean(clean)),
        "all_goals":   float(np.mean(gf_all)),
        "quality":     attack - defense,
    }
