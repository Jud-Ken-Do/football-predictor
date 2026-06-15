"""Attach calibrated per-match xG to training match rows.

Unifies the available xG providers into a single calibrated pair of columns
(``home_xg``, ``away_xg``) on a match DataFrame:

    1. StatsBomb open-data       (AFCON 23, Copa 24, WC 22 — fills CAF/CONCACAF)
    2. football-data.co.uk XLSX  (WC 2026 qualifiers — UEFA/AFC/CONMEBOL)

Source priority is StatsBomb > football-data where both cover a match. Each
source is moment-matched onto actual goals first (see ``xg_calibration``) so the
two scales are comparable before they feed the Kalman observation.

Orientation: World Cup / continental matches are at neutral venues, so a source's
"home"/"away" labels need not match our fixture orientation. We therefore match
on the **unordered team pair + date** and re-orient the xG to the row's
``home_team``/``away_team``.

Leakage: this operates only on the rows it is given. Callers that must avoid
look-ahead (the backtest) pass an already date-filtered frame, so both the
attached xG and the fitted calibration use only pre-cutoff matches.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from football_predictor.models.xg_calibration import (
    fit_source_calibrations,
    to_team_match_long,
)

logger = logging.getLogger(__name__)


def _collect_source_matches(use_cache: bool = True) -> dict[tuple, dict]:
    """{(frozenset(pair), date_str): {teamA, teamB, xgA, xgB, source}}.

    Higher-priority sources overwrite lower-priority ones for the same key.
    """
    out: dict[tuple, dict] = {}

    # Lowest priority first; StatsBomb overwrites football-data on collision.
    try:
        from football_predictor.data.sources.football_data_co_uk import build_xg_lookup as fd_xg
        for (home, away, date_str), (hxg, axg) in fd_xg().items():
            key = (frozenset((home, away)), date_str)
            out[key] = {"teamA": home, "teamB": away, "xgA": hxg, "xgB": axg,
                        "source": "football_data"}
    except Exception as exc:
        logger.warning("football-data xG unavailable (%s)", exc)

    try:
        from football_predictor.data.sources.statsbomb import build_xg_lookup as sb_xg
        for (home, away, date_str), (hxg, axg) in sb_xg(use_cache=use_cache).items():
            key = (frozenset((home, away)), date_str)
            out[key] = {"teamA": home, "teamB": away, "xgA": hxg, "xgB": axg,
                        "source": "statsbomb"}
    except Exception as exc:
        logger.warning("StatsBomb xG unavailable (%s)", exc)

    return out


def attach_calibrated_xg(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Return a copy of ``df`` with calibrated ``home_xg``/``away_xg`` columns.

    Rows with no xG from any source get NaN (the Kalman falls back to goals for
    those). Calibration is fit on the matched rows of ``df`` only.
    """
    df = df.copy()
    src = _collect_source_matches(use_cache=use_cache)
    if not src:
        df["home_xg"] = np.nan
        df["away_xg"] = np.nan
        return df

    raw_h = np.full(len(df), np.nan)
    raw_a = np.full(len(df), np.nan)
    sources = np.empty(len(df), dtype=object)

    dates = pd.to_datetime(df["date"])
    for i, (home, away, d) in enumerate(zip(df["home_team"], df["away_team"], dates)):
        # Match on team-pair + date with ±1-day tolerance: football-data.co.uk
        # and martj42 disagree by a day on many fixtures (timezone of the local
        # kickoff vs UTC). Exact-date matching silently dropped ~12% of xG
        # records — e.g. all Haiti/Curaçao CONCACAF qualifiers. Two given teams
        # virtually never play twice within a day, so the pair key keeps this safe.
        pair = frozenset((home, away))
        rec = None
        for delta in (0, 1, -1):
            rec = src.get((pair, str((d + pd.Timedelta(days=delta)).date())))
            if rec is not None:
                break
        if rec is None:
            continue
        if home == rec["teamA"]:
            raw_h[i], raw_a[i] = rec["xgA"], rec["xgB"]
        elif home == rec["teamB"]:
            raw_h[i], raw_a[i] = rec["xgB"], rec["xgA"]
        else:  # pair matched via frozenset but neither equals home (shouldn't happen)
            continue
        sources[i] = rec["source"]

    df["_xg_source"] = sources
    df["_raw_home_xg"] = raw_h
    df["_raw_away_xg"] = raw_a

    # Fit per-source calibration on the matched rows, then apply.
    matched = df.dropna(subset=["_raw_home_xg", "_raw_away_xg"]).rename(
        columns={"_raw_home_xg": "home_xg", "_raw_away_xg": "away_xg",
                 "home_goals": "home_score", "away_goals": "away_score"}
    )
    matched["source"] = matched["_xg_source"]
    cals = fit_source_calibrations(to_team_match_long(matched))

    home_cal = np.full(len(df), np.nan)
    away_cal = np.full(len(df), np.nan)
    for i in range(len(df)):
        s = sources[i]
        if s is None or s not in cals:
            continue
        cal = cals[s]
        home_cal[i] = cal.apply(raw_h[i])
        away_cal[i] = cal.apply(raw_a[i])

    df["home_xg"] = np.clip(home_cal, 0.0, None)
    df["away_xg"] = np.clip(away_cal, 0.0, None)
    df = df.drop(columns=["_xg_source", "_raw_home_xg", "_raw_away_xg"])

    n = int(np.isfinite(home_cal).sum())
    logger.info("Attached calibrated xG to %d/%d matches (sources: %s)",
                n, len(df), {k: round(v.b, 2) for k, v in cals.items()})
    return df
