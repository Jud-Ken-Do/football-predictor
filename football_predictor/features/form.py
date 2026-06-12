from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from football_predictor.features.base import FeatureModule
from football_predictor.constants import INTERNATIONAL_FORM_WINDOW_SIZES as FORM_WINDOW_SIZES

_MAX_WINDOW = max(FORM_WINDOW_SIZES)


class FormFeatures(FeatureModule):
    """Rolling form: pts, goals, GD, win-rate over recent matches.

    Pre-computes all form states in one chronological pass (O(n)),
    then serves lookups in O(1). Previously O(n²).
    """

    name = "form"

    def __init__(self) -> None:
        self._cache: dict[tuple, dict[str, float]] = {}  # (team, date_str) -> features
        self._final: dict[str, dict[str, float]] = {}
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home = match["home_team"]
        away = match["away_team"]

        h = self._cache.get((home, date_str)) or self._final.get(home) or self._empty_form()
        a = self._cache.get((away, date_str)) or self._final.get(away) or self._empty_form()

        result: dict[str, float] = {}
        for k, v in h.items():
            result[f"form_home_{k}"] = v
        for k, v in a.items():
            result[f"form_away_{k}"] = v
        for w in FORM_WINDOW_SIZES:
            result[f"form_pts_diff_last{w}"] = h.get(f"pts_last{w}", 0.0) - a.get(f"pts_last{w}", 0.0)
            result[f"form_gd_diff_last{w}"]  = h.get(f"gd_last{w}", 0.0)  - a.get(f"gd_last{w}", 0.0)
        return result

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

            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                key = (team, date_str)
                if key not in self._cache:
                    self._cache[key] = _form_from_history(team_history.get(team, deque()))

            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                if team not in team_history:
                    team_history[team] = deque(maxlen=_MAX_WINDOW)
                team_history[team].append({"gf": gf, "ga": ga})

        self._final = {t: _form_from_history(h) for t, h in team_history.items()}

    @staticmethod
    def _empty_form() -> dict[str, float]:
        out: dict[str, float] = {}
        for w in FORM_WINDOW_SIZES:
            out[f"pts_last{w}"] = 0.0
            out[f"gd_last{w}"] = 0.0
            out[f"goals_scored_last{w}"] = 0.0
            out[f"goals_conceded_last{w}"] = 0.0
            out[f"win_rate_last{w}"] = 0.0
        out["n_matches"] = 0.0
        return out

    def feature_names(self) -> list[str]:
        names = []
        for side in ("home", "away"):
            for w in FORM_WINDOW_SIZES:
                names += [
                    f"form_{side}_pts_last{w}",
                    f"form_{side}_gd_last{w}",
                    f"form_{side}_goals_scored_last{w}",
                    f"form_{side}_goals_conceded_last{w}",
                    f"form_{side}_win_rate_last{w}",
                ]
            # Number of matches available (capped at the largest window) —
            # lets XGBoost distinguish thin-history form from established form.
            names += [f"form_{side}_n_matches"]
        for w in FORM_WINDOW_SIZES:
            names += [f"form_pts_diff_last{w}", f"form_gd_diff_last{w}"]
        return names


def _form_from_history(history: deque) -> dict[str, float]:
    out: dict[str, float] = {}
    matches = list(history)  # newest at end (appendleft not used — oldest at front, newest at back)
    # We want the most recent w matches, so take from the end
    for w in FORM_WINDOW_SIZES:
        window = matches[-w:] if len(matches) >= w else matches
        if not window:
            out[f"pts_last{w}"] = 0.0
            out[f"gd_last{w}"] = 0.0
            out[f"goals_scored_last{w}"] = 0.0
            out[f"goals_conceded_last{w}"] = 0.0
            out[f"win_rate_last{w}"] = 0.0
            continue
        pts, gd, scored, conceded, wins = [], [], [], [], []
        for m in window:
            gf, ga = m["gf"], m["ga"]
            scored.append(gf)
            conceded.append(ga)
            gd.append(gf - ga)
            if gf > ga:
                pts.append(3); wins.append(1)
            elif gf == ga:
                pts.append(1); wins.append(0)
            else:
                pts.append(0); wins.append(0)
        # PER-GAME means, not sums: the window may hold fewer than w matches
        # (`matches[-w:]` of a short history), and a raw sum conflates form
        # with match count — a 4-match team with 3 wins would score
        # pts_last20=9 vs an established team's ~30. Feature NAMES are kept
        # for downstream compatibility, but pts_last{w} / gd_last{w} are now
        # per-game averages over the available window.
        out[f"pts_last{w}"]          = float(np.mean(pts))
        out[f"gd_last{w}"]           = float(np.mean(gd))
        out[f"goals_scored_last{w}"] = float(np.mean(scored))
        out[f"goals_conceded_last{w}"] = float(np.mean(conceded))
        out[f"win_rate_last{w}"]     = float(np.mean(wins))
    # Expose sample size explicitly so the model can weigh form reliability
    # (capped at the largest window by the deque maxlen).
    out["n_matches"] = float(min(len(matches), _MAX_WINDOW))
    return out
