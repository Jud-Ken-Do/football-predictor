from __future__ import annotations

import pandas as pd

from football_predictor.features.base import FeatureModule


class StandingsFeatures(FeatureModule):
    """League table position features computed from match history.

    Captures structural quality: table rank, points total, goal difference.
    Works without any external API — derived entirely from results data.
    """

    name = "standings"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        date = pd.Timestamp(match["date"])
        standings = self._build_standings(date, data)

        home = match["home_team"]
        away = match["away_team"]
        n_teams = max(len(standings), 1)

        h = standings.get(home, {"pts": 0, "gd": 0, "rank": n_teams})
        a = standings.get(away, {"pts": 0, "gd": 0, "rank": n_teams})

        return {
            "standings_home_pts": float(h["pts"]),
            "standings_away_pts": float(a["pts"]),
            "standings_pts_diff": float(h["pts"] - a["pts"]),
            "standings_home_gd": float(h["gd"]),
            "standings_away_gd": float(a["gd"]),
            "standings_gd_diff": float(h["gd"] - a["gd"]),
            "standings_home_rank": float(h["rank"]),
            "standings_away_rank": float(a["rank"]),
            "standings_rank_diff": float(a["rank"] - h["rank"]),  # positive = home ranked higher
        }

    def _build_standings(self, before: pd.Timestamp, data: pd.DataFrame) -> dict[str, dict]:
        table: dict[str, dict] = {}
        past = data[pd.to_datetime(data["date"]) < before]

        for _, row in past.iterrows():
            for team, gf, ga in [
                (row["home_team"], row["home_goals"], row["away_goals"]),
                (row["away_team"], row["away_goals"], row["home_goals"]),
            ]:
                if team not in table:
                    table[team] = {"pts": 0, "gd": 0}
                table[team]["gd"] += gf - ga
                if gf > ga:
                    table[team]["pts"] += 3
                elif gf == ga:
                    table[team]["pts"] += 1

        sorted_teams = sorted(table, key=lambda t: (table[t]["pts"], table[t]["gd"]), reverse=True)
        for rank, team in enumerate(sorted_teams, start=1):
            table[team]["rank"] = rank

        return table

    def feature_names(self) -> list[str]:
        return [
            "standings_home_pts",
            "standings_away_pts",
            "standings_pts_diff",
            "standings_home_gd",
            "standings_away_gd",
            "standings_gd_diff",
            "standings_home_rank",
            "standings_away_rank",
            "standings_rank_diff",
        ]
