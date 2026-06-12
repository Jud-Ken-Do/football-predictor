from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from football_predictor.features.base import FeatureModule

_MAX_MATCHES = 10


class H2HFeatures(FeatureModule):
    """Head-to-head historical record between two teams.

    Pre-computes H2H state for every pair in one chronological pass (O(n)),
    then serves lookups in O(1). Previously O(n²).
    """

    name = "h2h"

    # No-history defaults must be SYMMETRIC: most WC pairings have no H2H and
    # are played at neutral venues, so an asymmetric default (1.3 vs 1.1)
    # would inject a phantom home-advantage signal into the features.
    _DEFAULTS = {
        "h2h_home_win_rate": 0.33,
        "h2h_away_win_rate": 0.33,
        "h2h_draw_rate": 0.33,
        "h2h_avg_goals_home": 1.2,
        "h2h_avg_goals_away": 1.2,
        "h2h_n_matches": 0.0,
    }

    def __init__(self) -> None:
        # (team_a, team_b, date_str) -> features dict  (team_a < team_b alphabetically)
        self._cache: dict[tuple, dict[str, float]] = {}
        self._data_id: int | None = None

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        date_str = str(pd.Timestamp(match["date"]).date())
        home = match["home_team"]
        away = match["away_team"]
        key = (*sorted([home, away]), date_str)

        cached = self._cache.get(key)
        if cached is None:
            return dict(self._DEFAULTS)

        # Rotate perspective if teams are stored opposite to home/away in this match
        t1 = min(home, away)
        if t1 == home:
            return cached
        # Flip win rates
        flipped = dict(cached)
        flipped["h2h_home_win_rate"] = cached["h2h_away_win_rate"]
        flipped["h2h_away_win_rate"] = cached["h2h_home_win_rate"]
        flipped["h2h_avg_goals_home"] = cached["h2h_avg_goals_away"]
        flipped["h2h_avg_goals_away"] = cached["h2h_avg_goals_home"]
        return flipped

    def _ensure_cache(self, data: pd.DataFrame) -> None:
        if self._data_id == id(data):
            return
        self._data_id = id(data)
        self._build_cache(data)

    def _build_cache(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # pair -> deque of (gf_t1, ga_t1) from t1's perspective (t1 = alphabetically first)
        pair_history: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=_MAX_MATCHES))
        self._cache = {}

        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            hg, ag = float(row["home_goals"]), float(row["away_goals"])
            date_str = str(row["date"].date())

            t1, t2 = sorted([home, away])
            key = (t1, t2, date_str)

            if key not in self._cache:
                self._cache[key] = self._compute(pair_history[(t1, t2)], t1_is_home=(t1 == home))

            # Update history from t1's perspective
            gf_t1 = hg if home == t1 else ag
            ga_t1 = ag if home == t1 else hg
            pair_history[(t1, t2)].append((gf_t1, ga_t1))

    @staticmethod
    def _compute(history: deque, t1_is_home: bool) -> dict[str, float]:
        if not history:
            # Symmetric defaults — see _DEFAULTS comment (no home bias for
            # never-met pairs on neutral venues).
            return dict(H2HFeatures._DEFAULTS)

        t1_wins = t2_wins = draws = 0
        goals_t1: list[float] = []
        goals_t2: list[float] = []

        for gf, ga in history:
            goals_t1.append(gf)
            goals_t2.append(ga)
            if gf > ga:
                t1_wins += 1
            elif gf == ga:
                draws += 1
            else:
                t2_wins += 1

        n = len(history)
        # Store from t1's perspective; transform() flips if needed
        return {
            "h2h_home_win_rate": t1_wins / n,
            "h2h_away_win_rate": t2_wins / n,
            "h2h_draw_rate": draws / n,
            "h2h_avg_goals_home": sum(goals_t1) / n,
            "h2h_avg_goals_away": sum(goals_t2) / n,
            "h2h_n_matches": float(n),
        }

    def feature_names(self) -> list[str]:
        return [
            "h2h_home_win_rate",
            "h2h_away_win_rate",
            "h2h_draw_rate",
            "h2h_avg_goals_home",
            "h2h_avg_goals_away",
            "h2h_n_matches",
        ]
