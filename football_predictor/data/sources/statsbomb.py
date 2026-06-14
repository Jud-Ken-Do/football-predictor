"""Loader for StatsBomb open-data event xG.

StatsBomb publish full event data — including ``shot.statsbomb_xg`` and shot
freeze-frames — free at https://github.com/statsbomb/open-data.

ATTRIBUTION (required by the StatsBomb open-data user agreement):
    Data provided by StatsBomb — https://statsbomb.com/

This module aggregates that event data into per-match
team xG for the competitions that fill our CAF / CONCACAF xG coverage gap
(CLAUDE.md known-limitation #1):

    African Cup of Nations 2023  (comp 1267, season 107) — all 9 CAF WC teams
    Copa America 2024            (comp  223, season 282) — Mexico, USA, Canada
    FIFA World Cup 2022          (comp   43, season 106) — bonus gap teams

Per-match team xG = Σ ``shot.statsbomb_xg`` over **periods ≤ 4 only**. Period 5
is the penalty shootout — its ~0.78-xG penalties would otherwise inflate a 1–1
draw to ~8 xG/side (verified: Egypt 1–1 Congo DR → 1.37 / 1.58 in-play xG, not
8.4 / 8.6).

Aggregated per-match xG is cached to
``~/.cache/football_predictor/statsbomb/agg_<comp>_<season>.json``. Historical
tournament data never changes, so the aggregate is cached indefinitely; the
(large) raw event files are downloaded once at build time and not retained.
"""
from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd
import requests

from football_predictor.data.wc2026 import normalise

logger = logging.getLogger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_CACHE_DIR = pathlib.Path.home() / ".cache" / "football_predictor" / "statsbomb"

# (competition_id, season_id, human label).
# The first three fill the CAF/CONCACAF gap for WC 2026 prediction. The last
# three predate WC 2022 and so give the WC 2022 *backtest* real pre-tournament
# xG coverage — without them every backtestable WC trains on zero xG (all the
# gap comps + football-data qualifier xG postdate Nov 2022), making the mechanism
# untestable. The per-backtest date filter selects the valid subset automatically.
DEFAULT_SB_COMPS: list[tuple[int, int, str]] = [
    (1267, 107, "African Cup of Nations 2023"),  # CAF — for WC 2026
    (223, 282, "Copa America 2024"),             # CONCACAF guests — for WC 2026
    (43, 106, "FIFA World Cup 2022"),            # gap teams — for WC 2026
    (55, 43, "UEFA Euro 2020"),                  # predates WC 2022 → backtestable
    (43, 3, "FIFA World Cup 2018"),              # predates WC 2022 → backtestable
    (55, 282, "UEFA Euro 2024"),                 # UEFA coverage for WC 2026
]

# StatsBomb team-name variants → our canonical names (then run through
# wc2026.normalise() for anything TEAM_NAME_MAP already covers).
_SB_NAME_MAP: dict[str, str] = {
    "Cape Verde Islands": "Cabo Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
}


def _norm(name: str) -> str:
    return normalise(_SB_NAME_MAP.get(str(name).strip(), str(name).strip()))


def _get_json(url: str, timeout: int = 60) -> object:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _aggregate_match_xg(events: list[dict]) -> dict[str, float]:
    """Sum in-play (period ≤ 4) shot xG per team name for one match."""
    totals: dict[str, float] = {}
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        if ev.get("period", 1) > 4:  # exclude penalty shootout
            continue
        team = ev.get("team", {}).get("name")
        if not team:
            continue
        totals[team] = totals.get(team, 0.0) + float(
            ev.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
        )
    return totals


def _build_comp_aggregate(comp_id: int, season_id: int, label: str) -> list[dict]:
    """Download the season's matches + events once, return per-match xG rows."""
    matches = _get_json(f"{_RAW_BASE}/matches/{comp_id}/{season_id}.json")
    rows: list[dict] = []
    for i, m in enumerate(matches):
        mid = m["match_id"]
        try:
            events = _get_json(f"{_RAW_BASE}/events/{mid}.json")
        except Exception as exc:  # one bad file shouldn't sink the whole comp
            logger.warning("StatsBomb events fetch failed for match %s (%s)", mid, exc)
            continue
        xg = _aggregate_match_xg(events)
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        rows.append({
            "date": m["match_date"],
            "home_team": _norm(home),
            "away_team": _norm(away),
            "home_score": int(m["home_score"]),
            "away_score": int(m["away_score"]),
            "home_xg": round(xg.get(home, 0.0), 4),
            "away_xg": round(xg.get(away, 0.0), 4),
            "source": "statsbomb",
        })
        if (i + 1) % 16 == 0:
            logger.info("  [StatsBomb] %s: %d/%d matches", label, i + 1, len(matches))
    return rows


def _comp_rows(comp_id: int, season_id: int, label: str, use_cache: bool) -> list[dict]:
    cache_path = _CACHE_DIR / f"agg_{comp_id}_{season_id}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())
    logger.info("Building StatsBomb xG aggregate for %s ...", label)
    rows = _build_comp_aggregate(comp_id, season_id, label)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    return rows


def fetch_statsbomb_xg(
    comps: list[tuple[int, int, str]] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return per-match in-play team xG across the configured competitions.

    Columns: date (Timestamp), home_team, away_team, home_score, away_score,
    home_xg, away_xg, source. Team names normalised to WC 2026 canonical names.
    """
    comps = comps or DEFAULT_SB_COMPS
    all_rows: list[dict] = []
    for comp_id, season_id, label in comps:
        try:
            all_rows.extend(_comp_rows(comp_id, season_id, label, use_cache))
        except Exception as exc:
            logger.warning("StatsBomb comp %s unavailable (%s) — skipping", label, exc)

    if not all_rows:
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_score",
                     "away_score", "home_xg", "away_xg", "source"]
        )

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_xg_lookup(use_cache: bool = True) -> dict[tuple, tuple[float, float]]:
    """(home, away, date_str) → (xg_home, xg_away) for StatsBomb matches."""
    df = fetch_statsbomb_xg(use_cache=use_cache)
    return {
        (r.home_team, r.away_team, str(r.date.date())): (float(r.home_xg), float(r.away_xg))
        for r in df.itertuples()
    }
