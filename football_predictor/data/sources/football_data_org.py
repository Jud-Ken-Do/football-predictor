from __future__ import annotations

import os
import logging
import pathlib
from typing import Optional

import pandas as pd
import requests

from football_predictor.constants import FOOTBALL_DATA_ORG_BASE_URL, SUPPORTED_COMPETITIONS

logger = logging.getLogger(__name__)

_CACHE_DIR = pathlib.Path.home() / ".cache" / "football_predictor" / "football_data_org"


def _cache_path(competition: str, season: str) -> pathlib.Path:
    return _CACHE_DIR / f"{competition}_{season}.parquet"


def fetch_matches(
    competition: str,
    seasons: list[str],
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical match results from football-data.org.

    Args:
        competition: Competition code, e.g. "PL" for Premier League.
                     Must be in constants.SUPPORTED_COMPETITIONS.
        seasons:     List of season start years as strings, e.g. ["2022", "2023"].
        api_key:     API key for football-data.org. If None, reads from the
                     FOOTBALL_DATA_ORG_API_KEY environment variable.
        use_cache:   If True, skip the API call when a local parquet cache exists.

    Returns:
        DataFrame with columns:
            date, season, competition, home_team, away_team,
            home_goals, away_goals, status
        Rows with status != "FINISHED" are excluded.
    """
    if competition not in SUPPORTED_COMPETITIONS:
        raise ValueError(f"Unsupported competition '{competition}'. Choose from {list(SUPPORTED_COMPETITIONS)}")

    key = api_key or os.environ.get("FOOTBALL_DATA_ORG_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "No API key found. Set FOOTBALL_DATA_ORG_API_KEY or pass api_key= directly. "
            "Register for free at https://www.football-data.org/"
        )

    frames: list[pd.DataFrame] = []
    for season in seasons:
        cache = _cache_path(competition, season)
        if use_cache and cache.exists():
            logger.info("Loading %s %s from cache.", competition, season)
            frames.append(pd.read_parquet(cache))
            continue

        url = f"{FOOTBALL_DATA_ORG_BASE_URL}/competitions/{competition}/matches"
        params = {"season": season}
        headers = {"X-Auth-Token": key}

        logger.info("Fetching %s season %s from football-data.org…", competition, season)
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        raw = response.json().get("matches", [])
        df = _parse_matches(raw, competition, season)

        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined.sort_values("date").reset_index(drop=True)


def _parse_matches(raw: list[dict], competition: str, season: str) -> pd.DataFrame:
    rows = []
    for m in raw:
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "date": m["utcDate"][:10],
                "season": season,
                "competition": competition,
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "status": m["status"],
                "match_id": m.get("id"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    return df.dropna(subset=["home_goals", "away_goals"])
