from __future__ import annotations

import logging
import pathlib
from typing import Optional

import pandas as pd
import requests

from football_predictor.features.base import FeatureModule

logger = logging.getLogger(__name__)

# FIFA rankings from Dato-Futbol/fifa-ranking (1992–2024, scraped from official FIFA site)
# Columns: team, total_points, date, id, id_num, team_short
_RANKINGS_URL = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
_CACHE_PATH = pathlib.Path.home() / ".cache" / "football_predictor" / "fifa_rankings.csv"

# The FIFA rankings cache contains BOTH old and new name variants for renamed
# teams (e.g. "Czech Republic" 314 rows + "Czechia" 14 rows). A single-name
# lookup truncates the timeline at the rename date, so the as-of-date query
# must UNION all variants of the same country.
_RANKING_ALIASES: dict[str, list[str]] = {
    "Czechia": ["Czechia", "Czech Republic"],
    "Türkiye": ["Türkiye", "Turkey"],
    "Cabo Verde": ["Cabo Verde", "Cape Verde", "Cape Verde Islands"],
    "IR Iran": ["IR Iran", "Iran"],
    "South Korea": ["South Korea", "Korea Republic"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire"],
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina"],
    "United States": ["United States", "USA"],
    "DR Congo": ["DR Congo", "Congo DR"],
    "Curaçao": ["Curaçao", "Curacao"],
}

# Flat lookup: ANY variant of a renamed team → full alias group, so the union
# works regardless of which variant the caller uses as the team name.
_ALIAS_LOOKUP: dict[str, list[str]] = {
    name: group for group in _RANKING_ALIASES.values() for name in group
}


_LOAD_FAILED = object()  # sentinel: tried to load, failed — don't retry


class RankingsFeatures(FeatureModule):
    """FIFA World Rankings features.

    The single most important quality signal for international football —
    confirmed as top predictor in Groll et al. (2024) EURO prediction paper
    and Constantinou & Fenton (2013) pi-ratings study.

    Uses the Dato-Futbol/fifa-ranking dataset: historical monthly FIFA rankings
    scraped from the official FIFA website, 1992–2024, covering 200+ national teams.
    Source: https://github.com/Dato-Futbol/fifa-ranking

    NOTE: martj42/fifa_ranking repo does not exist (404). This module returns
    neutral defaults until a working data source is found. Elo ratings in
    elo.py carry the same signal and are active by default.

    Features produced:
    - rankings_home_rank, rankings_away_rank
    - rankings_rank_diff  (positive = home team ranked higher)
    - rankings_home_points, rankings_away_points
    - rankings_points_diff
    """

    name = "rankings"

    def __init__(self) -> None:
        self._rankings_cache: object = None  # None = not loaded, _LOAD_FAILED = gave up

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        if self._rankings_cache is None:
            result = self._load_rankings()
            self._rankings_cache = result if result is not None else _LOAD_FAILED
        if self._rankings_cache is _LOAD_FAILED:
            return self._unknown()
        rankings = self._rankings_cache
        assert isinstance(rankings, pd.DataFrame)

        date = pd.Timestamp(match["date"])
        home = match["home_team"]
        away = match["away_team"]

        h = self._rank_at(home, date, rankings)
        a = self._rank_at(away, date, rankings)

        return {
            "rankings_home_rank": h["rank"],
            "rankings_away_rank": a["rank"],
            "rankings_rank_diff": a["rank"] - h["rank"],   # positive = home ranked higher
            "rankings_home_points": h["points"],
            "rankings_away_points": a["points"],
            "rankings_points_diff": h["points"] - a["points"],
        }

    def _rank_at(self, team: str, date: pd.Timestamp, rankings: pd.DataFrame) -> dict:
        # Union the timelines of all name variants for renamed teams; the
        # latest row <= date may come from either the old or the new name.
        aliases = _ALIAS_LOOKUP.get(team, [team])
        team_rows = rankings[
            rankings["team"].isin(aliases) & (rankings["rank_date"] <= date)
        ].sort_values("rank_date", ascending=False)

        if team_rows.empty:
            # No substring fallback here: matching on the first word of the
            # team name can silently return a DIFFERENT country (e.g.
            # "Congo" matching "Congo DR"). Unknown teams get the neutral
            # default instead.
            return {"rank": 100.0, "points": 1000.0}

        row = team_rows.iloc[0]
        return {"rank": float(row.get("rank", 100)), "points": float(row["total_points"])}

    def _load_rankings(self) -> Optional[pd.DataFrame]:
        if not _CACHE_PATH.exists():
            try:
                resp = requests.get(_RANKINGS_URL, timeout=30)
                resp.raise_for_status()
                _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CACHE_PATH.write_text(resp.text, encoding="utf-8")
            except Exception as e:
                logger.warning("Could not download FIFA rankings: %s", e)
                return None
        df = pd.read_csv(_CACHE_PATH)
        # Dato-Futbol uses "date" column; rename to rank_date for internal use
        df = df.rename(columns={"date": "rank_date"})
        df["rank_date"] = pd.to_datetime(df["rank_date"])
        # Dato-Futbol doesn't have an explicit rank column — derive from points per date
        if "rank" not in df.columns:
            df["rank"] = df.groupby("rank_date")["total_points"].rank(ascending=False, method="min")
        return df

    @staticmethod
    def _unknown() -> dict[str, float]:
        return {
            "rankings_home_rank": 50.0,
            "rankings_away_rank": 50.0,
            "rankings_rank_diff": 0.0,
            "rankings_home_points": 1000.0,
            "rankings_away_points": 1000.0,
            "rankings_points_diff": 0.0,
        }

    def feature_names(self) -> list[str]:
        return [
            "rankings_home_rank",
            "rankings_away_rank",
            "rankings_rank_diff",
            "rankings_home_points",
            "rankings_away_points",
            "rankings_points_diff",
        ]
