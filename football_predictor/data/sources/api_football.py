"""API-Football (api-sports.io) data source.

Fetches pre-match injury and suspension reports for WC 2026 fixtures.
Free tier: 100 requests/day — sufficient for pre-match injury fetching.

Usage:
    python3.11 scripts/fetch_wc2026_injuries.py   # run before each match day

Environment:
    API_FOOTBALL_KEY=<your_key>  in .env
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE = "https://v3.football.api-sports.io"
_KEY  = os.getenv("API_FOOTBALL_KEY", "")
_HEADERS = {"x-apisports-key": _KEY}

_CACHE_DIR  = Path(__file__).resolve().parents[3] / "data"
_INJ_CACHE  = _CACHE_DIR / "wc2026_injuries_cache.json"
_FIX_CACHE  = _CACHE_DIR / "wc2026_fixtures_cache.json"

# api-sports.io league ID for FIFA World Cup
_WC2026_LEAGUE_ID = 1
_WC2026_SEASON    = 2026


# ── Low-level request ─────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict:
    if not _KEY:
        raise RuntimeError("API_FOOTBALL_KEY not set in .env")
    r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"API error: {data['errors']}")
    return data


# ── Fixtures ──────────────────────────────────────────────────────────────────

def fetch_wc2026_fixtures(force: bool = False) -> list[dict]:
    """Fetch all WC 2026 fixture IDs, cached locally."""
    if _FIX_CACHE.exists() and not force:
        with open(_FIX_CACHE) as f:
            return json.load(f)

    logger.info("Fetching WC 2026 fixtures from API-Football...")
    data = _get("fixtures", {"league": _WC2026_LEAGUE_ID, "season": _WC2026_SEASON})
    fixtures = data.get("response", [])

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_FIX_CACHE, "w") as f:
        json.dump(fixtures, f, indent=2)

    logger.info("Cached %d fixtures → %s", len(fixtures), _FIX_CACHE)
    return fixtures


# ── Injuries ──────────────────────────────────────────────────────────────────

def fetch_injuries_for_fixture(fixture_id: int) -> list[dict]:
    """Fetch injury/suspension list for one fixture."""
    data = _get("injuries", {"fixture": fixture_id})
    return data.get("response", [])


def _parse_injury_entry(entry: dict) -> dict:
    """Normalise one injury entry to a flat dict."""
    player = entry.get("player", {})
    team   = entry.get("team", {})
    return {
        "player_name": player.get("name", ""),
        "player_id":   player.get("id"),
        "team_name":   team.get("name", ""),
        "team_id":     team.get("id"),
        "reason":      entry.get("type", ""),    # "Injury" | "Suspension"
        "position":    player.get("type", ""),   # "Goalkeeper" | "Defender" | ...
    }


def fetch_and_cache_all_injuries(force: bool = False, delay: float = 0.8) -> dict:
    """
    Fetch injury data for every WC 2026 fixture and write to cache.
    Respects the free-tier rate limit (100 req/day) — call once per match day.

    Returns: {fixture_id_str: [injury_entry, ...]}
    """
    existing: dict = {}
    if _INJ_CACHE.exists() and not force:
        with open(_INJ_CACHE) as f:
            existing = json.load(f)

    fixtures = fetch_wc2026_fixtures()
    new_fetches = 0

    for fix in fixtures:
        fid = str(fix["fixture"]["id"])
        # Only trust the cache when it has actual entries. An EMPTY cached
        # list usually means the fixture was fetched days before kickoff,
        # when API-Football had no injury data yet — those must be re-checked
        # on every run or the injury features stay a silent no-op for the
        # whole tournament. (All 72 fixtures were cached empty pre-kickoff.)
        if fid in existing and existing[fid] and not force:
            continue

        status = fix["fixture"]["status"]["short"]
        # Only fetch for upcoming (NS) or recently finished (FT) fixtures
        if status not in ("NS", "FT", "AET", "PEN", "1H", "2H", "HT"):
            continue

        try:
            raw = fetch_injuries_for_fixture(int(fid))
            existing[fid] = [_parse_injury_entry(e) for e in raw]
            new_fetches += 1
            logger.info("  Fixture %s: %d injury entries", fid, len(existing[fid]))
        except Exception as exc:
            logger.warning("  Fixture %s failed: %s", fid, exc)

        time.sleep(delay)

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_INJ_CACHE, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Injury cache updated: %d fixtures total, %d newly fetched → %s",
                len(existing), new_fetches, _INJ_CACHE)
    return existing


# ── Public accessor ───────────────────────────────────────────────────────────

def load_injury_cache() -> dict:
    """Load cached injury data. Returns {} if cache doesn't exist."""
    if not _INJ_CACHE.exists():
        return {}
    with open(_INJ_CACHE) as f:
        return json.load(f)


def get_team_injuries(fixture_id: int, team_name: str) -> list[dict]:
    """Return injury list for one team in one fixture (from cache)."""
    cache = load_injury_cache()
    entries = cache.get(str(fixture_id), [])
    return [e for e in entries if e.get("team_name", "").lower() == team_name.lower()]
