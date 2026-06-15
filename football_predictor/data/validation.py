"""R6 — ingress validation for the 48 WC 2026 teams.

Three of the four worst bugs in the architecture review were *silent name-miss*
bugs: a team's name spelled differently across sources so a lookup quietly
returned a default (league-average λ, Elo 1500, no xG) with no error. This module
asserts, at the start of a run, that every WC 2026 team is actually covered by
the training data and by every prediction-time lookup — and prints a coverage
matrix so any gap is loud, not silent.

`football_predictor.data.wc2026.normalise()` is the single canonical name
function; this validator resolves each source's own spelling back to the
canonical name via the small per-source maps below (kept in lock-step with the
maps used in the feature modules / dashboard).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from football_predictor.data.wc2026 import ALL_WC2026_TEAMS, normalise

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]

# Canonical WC name → the spelling each source actually uses.
_FC26 = {"South Korea": "Korea Republic", "Ivory Coast": "Côte d'Ivoire",
         "DR Congo": "Congo DR", "IR Iran": "Iran", "Curaçao": "Curacao"}
_APIFORM = {"Czechia": "Czech Republic", "Bosnia and Herzegovina": "Bosnia & Herzegovina",
            "United States": "USA", "IR Iran": "Iran", "Cabo Verde": "Cape Verde Islands",
            "DR Congo": "Congo DR"}
# FIFA-ranking dataset spellings for teams whose canonical name differs.
_RANK = {"IR Iran": "Iran", "Türkiye": "Turkey", "Czechia": "Czech Republic",
         "DR Congo": "Congo DR", "Cabo Verde": "Cape Verde", "United States": "USA",
         "South Korea": "Korea Republic", "Ivory Coast": "Côte d'Ivoire"}

# Sources whose silent miss corrupts the prediction (hard gaps). xG / fc26 /
# injuries gaps are acceptable (e.g. New Zealand has no free xG) → warn only.
_CRITICAL = {"matches", "rankings", "odds"}


def _json_keys(rel: str) -> set[str]:
    p = _ROOT / "data" / rel
    if not p.exists():
        return set()
    try:
        d = json.loads(p.read_text())
        return {k for k in d if not k.startswith("_")}
    except Exception:
        return set()


def _covered(team: str, keys: set[str], name_map: dict[str, str]) -> bool:
    return team in keys or normalise(team) in keys or name_map.get(team, team) in keys


def validate_wc2026_coverage(
    all_data: pd.DataFrame, min_matches: int = 30, quiet: bool = False,
    strict: bool = False, verbose: bool = False,
) -> dict:
    """Check every WC 2026 team against training data + all prediction lookups.

    Returns {team: {source: bool/int}}. Prints a coverage matrix and warns on
    gaps. With strict=True, raises if a critical source (training matches /
    rankings / odds) is missing for any team.
    """
    counts = pd.concat([all_data["home_team"], all_data["away_team"]]).value_counts()

    tm_keys = _json_keys("transfermarkt_wc2026.json")
    inj_keys = _json_keys("wc2026_player_injuries.json")
    form_keys = _json_keys("api_form_cache.json")
    try:
        from football_predictor.features.sofifa_ratings import _load_fc26
        fc26_keys = set(_load_fc26())
    except Exception:
        fc26_keys = set()
    try:
        from football_predictor.data.sources.football_data_co_uk import build_odds_lookup
        odds_keys = {t for (h, a, d) in build_odds_lookup() if str(d) >= "2026-06-01"
                     for t in (h, a)}
    except Exception:
        odds_keys = set()
    try:
        import pathlib
        rp = pathlib.Path.home() / ".cache" / "football_predictor" / "fifa_rankings.csv"
        rank_keys = set(pd.read_csv(rp)["team"].astype(str).unique()) if rp.exists() else set()
    except Exception:
        rank_keys = set()
    try:
        from football_predictor.data.xg_attach import attach_calibrated_xg
        _xg = attach_calibrated_xg(all_data)
        _xg = _xg[_xg["home_xg"].notna() & _xg["away_xg"].notna()]
        xg_teams = set(pd.concat([_xg["home_team"], _xg["away_team"]]).map(normalise))
    except Exception:
        xg_teams = set()

    report: dict[str, dict] = {}
    gaps: dict[str, list[str]] = {}
    for team in sorted(ALL_WC2026_TEAMS):
        n = int(counts.get(normalise(team), 0))
        row = {
            "matches":      n >= min_matches,
            "rankings":     _covered(team, rank_keys, _RANK),
            "odds":         _covered(team, odds_keys, {}),
            "transfermarkt": _covered(team, tm_keys, {}),
            "fc26":         _covered(team, fc26_keys, _FC26),
            "api_form":     _covered(team, form_keys, _APIFORM),
            "injuries":     _covered(team, inj_keys, {}),
            "xG":           normalise(team) in xg_teams,
        }
        row["_n"] = n
        report[team] = row
        for src, ok in row.items():
            if src != "_n" and not ok:
                gaps.setdefault(src, []).append(team)

    srcs = ["matches", "rankings", "odds", "transfermarkt", "fc26", "api_form", "injuries", "xG"]
    if not quiet:
        if verbose:
            print(f"\n  WC 2026 data coverage (48 teams, min_matches={min_matches}):")
            print("  " + " ".join(f"{s[:4]:>5}" for s in srcs) + "   team")
            for team in sorted(ALL_WC2026_TEAMS):
                r = report[team]
                cells = " ".join(f"{'  ✓ ' if r[s] else '  ✗ ':>5}" for s in srcs)
                flag = "" if all(r[s] for s in srcs) else "  ←"
                print(f"  {cells}   {team}{flag}")
        n_full = sum(1 for t in ALL_WC2026_TEAMS if all(report[t][s] for s in srcs))
        print(f"  Ingress check: {n_full}/48 WC teams fully covered (min_matches={min_matches}).")
        for src in srcs:
            g = gaps.get(src, [])
            if g:
                tag = "CRITICAL" if src in _CRITICAL else "ok-ish"
                logger.warning("[%s] %d team(s) missing from '%s': %s", tag, len(g), src, ", ".join(g))
                print(f"  ⚠ [{tag}] missing from {src}: {', '.join(g)}")
        if not any(gaps.get(s) for s in srcs):
            print("  ✓ all 48 teams fully covered.")

    critical_gaps = {s: gaps[s] for s in _CRITICAL if gaps.get(s)}
    if strict and critical_gaps:
        raise ValueError(f"WC 2026 ingress validation failed (critical gaps): {critical_gaps}")
    return report
