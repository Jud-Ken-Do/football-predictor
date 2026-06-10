from __future__ import annotations

import logging
import pathlib
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# martj42/international_results — 47,000+ matches from 1872 to present, updated regularly
_CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_CACHE_PATH = pathlib.Path.home() / ".cache" / "football_predictor" / "international_results.csv"


def fetch_international_results(
    from_year: int = 2010,
    tournaments: Optional[list[str]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical international match results from martj42/international_results.

    This is the primary data source for World Cup prediction. It covers all
    international fixtures: World Cup, qualifiers, Copa América, Euros,
    AFCON, friendlies, etc. — which is essential for building team form and
    Elo ratings since national teams play only ~10 matches per year.

    Args:
        from_year:   Only return matches from this year onwards. Use a wider
                     window (e.g. 2010) than you'd use for club football since
                     international teams play far less frequently.
        tournaments: If set, filter to specific tournament names, e.g.
                     ["FIFA World Cup", "FIFA World Cup qualification"].
                     None returns all tournaments.
        use_cache:   Cache the raw CSV locally to avoid re-downloading.

    Returns:
        DataFrame with columns:
            date, home_team, away_team, home_goals, away_goals,
            tournament, city, country, neutral
        Sorted ascending by date.
    """
    _CACHE_MAX_AGE_S = 48 * 3600
    cache_stale = (
        _CACHE_PATH.exists()
        and (time.time() - _CACHE_PATH.stat().st_mtime) > _CACHE_MAX_AGE_S
    )
    if use_cache and _CACHE_PATH.exists() and not cache_stale:
        logger.info("Loading international results from cache.")
        df = pd.read_csv(_CACHE_PATH)
    else:
        logger.info("Downloading international results from GitHub...")
        resp = requests.get(_CSV_URL, timeout=30)
        resp.raise_for_status()
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(resp.text, encoding="utf-8")
        df = pd.read_csv(_CACHE_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df = df.dropna(subset=["home_goals", "away_goals"])

    # Normalise team names from martj42 CSV to WC 2026 canonical names
    _NAME_MAP = {
        "Cape Verde":     "Cabo Verde",
        "Czech Republic": "Czechia",
        "Iran":           "IR Iran",
        "Turkey":         "Türkiye",
    }
    df["home_team"] = df["home_team"].replace(_NAME_MAP)
    df["away_team"] = df["away_team"].replace(_NAME_MAP)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df = df[df["date"].dt.year >= from_year]

    if tournaments:
        df = df[df["tournament"].isin(tournaments)]

    return df.sort_values("date").reset_index(drop=True)


def fetch_world_cup_matches(use_cache: bool = True) -> pd.DataFrame:
    """Return only FIFA World Cup final tournament matches (not qualifiers)."""
    return fetch_international_results(
        from_year=1990,
        tournaments=["FIFA World Cup"],
        use_cache=use_cache,
    )


def fetch_training_data(from_year: int = 2014, use_cache: bool = True, friendly_weight: float = 0.7) -> pd.DataFrame:
    """Return all international matches for model training, with match-type weights.

    Includes everything: WC, qualifiers, continental tournaments, and friendlies.
    Friendlies are included but down-weighted since squads are rotated and outcomes
    are less contested. The friendly_weight parameter controls how much — 0.7 chosen
    by grid search over WC 2018/2022 backtests.

    Match type weights:
        Friendly:                  friendly_weight (tunable, default 0.7)
        Arab Cup / minor:          0.6
        Nations League / Gold Cup: 0.7
        AFCON / Asian Cup:         0.92
        Major qualifier:           1.0
        World Cup:                 1.5
    """
    df = fetch_international_results(from_year=from_year, use_cache=use_cache)
    df["match_weight"] = df["tournament"].apply(lambda t: _match_weight(t, friendly_weight))

    # Remove awarded results — scorelines decided off the pitch carry no form signal.
    # Known case: AFCON 2025 final Morocco 3-0 Senegal (awd.) — Senegal fielded
    # an ineligible player; the result was awarded, not played.
    awarded = (
        (df["home_team"] == "Morocco") & (df["away_team"] == "Senegal") &
        (df["home_goals"] == 3) & (df["away_goals"] == 0) &
        (df["tournament"] == "African Cup of Nations") &
        (df["date"].astype(str).str.startswith("2026-01-18"))
    )
    df = df[~awarded].reset_index(drop=True)

    return df.reset_index(drop=True)


def _match_weight(tournament: str, friendly_weight: float = 0.3) -> float:
    t = tournament.lower()
    if "friendly" in t:
        return friendly_weight
    if "world cup" in t and "qualif" not in t:
        return 1.5
    if any(k in t for k in ["arab cup", "cosafa", "cecafa", "wafu", "aff", "uncaf"]):
        return 0.6
    if any(k in t for k in ["nations league", "gold cup"]):
        return 0.7
    if any(k in t for k in ["african cup", "asian cup"]):
        return 0.92
    return 1.0  # qualifiers, copa america, euro
