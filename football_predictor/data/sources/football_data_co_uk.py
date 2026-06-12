"""Loader for football-data.co.uk World Cup XLSX data.

Provides:
  - Closing bookmaker odds (WC 2018 Pinnacle, WC 2022 bet365/Betfair)
  - xG per match (WC 2026 qualifiers only)
  - Match stats: shots, shots on target, corners (all sheets)

Excel file expected at: data/raw/WorldCup2026.xlsx
Download from: football-data.co.uk → Historical → World Cup XLSX
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_EXCEL_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "WorldCup2026.xlsx"

# ── Name normalisation ────────────────────────────────────────────────────────
# football-data.co.uk names → our internal names (matching international_results)
_NAME_MAP: dict[str, str] = {
    "USA":                        "United States",
    "Iran":                       "IR Iran",
    "Turkey":                     "Türkiye",
    "Bosnia & Herzegovina":       "Bosnia and Herzegovina",
    "D.R. Congo":                 "DR Congo",
    "Cape Verde":                 "Cabo Verde",
    "Curacao":                    "Curaçao",
    "Trinidad & Tobago":          "Trinidad and Tobago",
    "Czech Republic":             "Czechia",
    # "South Korea" and "Ivory Coast" are already canonical — do NOT map them
    # to "Korea Republic"/"Côte d'Ivoire" (names that never appear in training
    # data); doing so silently broke all odds/xG joins for these teams.
    "Northern Ireland":           "Northern Ireland",
    "Congo":                      "Republic of Congo",
    "Central Africa":             "Central African Republic",
}


def _norm(name: str) -> str:
    return _NAME_MAP.get(str(name).strip(), str(name).strip())


def _implied_probs(h_odds: float, d_odds: float, a_odds: float) -> tuple[float, float, float]:
    """Convert closing odds to normalised implied probabilities (remove overround)."""
    try:
        ph, pd_, pa = 1.0 / h_odds, 1.0 / d_odds, 1.0 / a_odds
        total = ph + pd_ + pa
        if total <= 0:
            return 1/3, 1/3, 1/3
        return ph / total, pd_ / total, pa / total
    except (ZeroDivisionError, TypeError):
        return 1/3, 1/3, 1/3


@lru_cache(maxsize=1)
def load_all() -> dict[str, pd.DataFrame]:
    """Load and normalise all sheets. Cached after first call."""
    if not _EXCEL_PATH.exists():
        return {}
    xl = pd.ExcelFile(_EXCEL_PATH)
    sheets = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(_EXCEL_PATH, sheet_name=sheet)
        df["Home"] = df["Home"].map(_norm)
        df["Away"] = df["Away"].map(_norm)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        sheets[sheet] = df
    return sheets


# ── Odds lookup ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_odds_lookup() -> dict[tuple, dict[str, float]]:
    """Build (home, away, date_str) → odds dict for all available matches.

    Sources (merged in order):
      1. football-data.co.uk XLSX — WC 2018/2022 closing odds + qualifiers
      2. data/wc2026_odds_cache.json — live pre-match odds for all 72 WC 2026 fixtures
    """
    import json as _json
    sheets = load_all()
    lookup: dict[tuple, dict[str, float]] = {}

    for sheet_name, df in sheets.items():
        # Identify odds columns available in this sheet
        # Primary bookmaker (best accuracy); underscore and dash variants
        h_col = next((c for c in ["bet365-H", "Pinny-H"] if c in df.columns), None)
        d_col = next((c for c in ["bet365-D", "Pinny-D"] if c in df.columns), None)
        a_col = next((c for c in ["bet365-A", "Pinny-A"] if c in df.columns), None)
        # Market average (consensus); covers both naming conventions
        avg_h = next((c for c in ["H-Avg", "H_Avg"] if c in df.columns), h_col)
        avg_d = next((c for c in ["D-Avg", "D_Avg"] if c in df.columns), d_col)
        avg_a = next((c for c in ["A-Avg", "A_Avg"] if c in df.columns), a_col)
        max_h = next((c for c in ["H-Max", "H_Max"] if c in df.columns), avg_h)
        max_d = next((c for c in ["D-Max", "D_Max"] if c in df.columns), avg_d)
        max_a = next((c for c in ["A-Max", "A_Max"] if c in df.columns), avg_a)

        for _, row in df.iterrows():
            if pd.isna(row["Date"]):
                continue
            key = (row["Home"], row["Away"], str(row["Date"].date()))

            # Primary closing odds (most accurate bookmaker available)
            h_o = row.get(h_col) if h_col else None
            d_o = row.get(d_col) if d_col else None
            a_o = row.get(a_col) if a_col else None

            # Average market odds (consensus)
            avg_h_o = row.get(avg_h) if avg_h else None
            avg_d_o = row.get(avg_d) if avg_d else None
            avg_a_o = row.get(avg_a) if avg_a else None

            # Max odds (best price = market underdog signal)
            max_h_o = row.get(max_h) if max_h else None
            max_d_o = row.get(max_d) if max_d else None
            max_a_o = row.get(max_a) if max_a else None

            # Use average odds as primary if specific bookmaker missing
            best_h = avg_h_o if pd.notna(avg_h_o) else h_o
            best_d = avg_d_o if pd.notna(avg_d_o) else d_o
            best_a = avg_a_o if pd.notna(avg_a_o) else a_o

            try:
                best_h, best_d, best_a = float(best_h), float(best_d), float(best_a)
            except (TypeError, ValueError):
                continue
            if not all(pd.notna(x) and x > 1.0 for x in [best_h, best_d, best_a]):
                continue

            ph, pd_p, pa = _implied_probs(float(best_h), float(best_d), float(best_a))
            overround = 1/float(best_h) + 1/float(best_d) + 1/float(best_a)

            entry: dict[str, float] = {
                "ph": ph, "pd": pd_p, "pa": pa,
                "overround": overround,
                "avg_h": float(best_h), "avg_d": float(best_d), "avg_a": float(best_a),
            }

            if all(pd.notna(x) and x > 1.0 for x in [max_h_o, max_d_o, max_a_o]):
                mh, md_p, ma = _implied_probs(float(max_h_o), float(max_d_o), float(max_a_o))
                entry.update({"max_ph": mh, "max_pd": md_p, "max_pa": ma})

            lookup[key] = entry

    # ── WC 2026 live odds (API-Football cache) ────────────────────────────────
    _wc26_cache = Path(__file__).resolve().parents[3] / "data" / "wc2026_odds_cache.json"
    if _wc26_cache.exists():
        try:
            from football_predictor.data.wc2026 import GROUP_STAGE_SCHEDULE, normalise
            # Build (home, away) → schedule date map from our authoritative schedule
            _sched_dates = {
                (normalise(m["home_team"]), normalise(m["away_team"])): str(m["date"])
                for m in GROUP_STAGE_SCHEDULE
            }
            wc26 = _json.loads(_wc26_cache.read_text())
            for fid, d in wc26.items():
                home = normalise(d.get("home_team", ""))
                away = normalise(d.get("away_team", ""))
                if not (home and away) or "p1" not in d:
                    continue
                # Use our schedule date so the key matches predict_wc2026.py fixture rows
                date = _sched_dates.get((home, away)) or _sched_dates.get((away, home))
                if not date:
                    continue
                lookup[(home, away, date)] = {
                    "ph": d["p1"], "pd": d["px"], "pa": d["p2"],
                    "overround": 1.05,
                    "avg_h": round(1 / max(d["p1"], 0.01), 2),
                    "avg_d": round(1 / max(d["px"], 0.01), 2),
                    "avg_a": round(1 / max(d["p2"], 0.01), 2),
                }
        except Exception:
            pass

    return lookup


# ── xG lookup ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_xg_lookup() -> dict[tuple, tuple[float, float]]:
    """(home, away, date_str) → (xg_home, xg_away) from WC 2026 qualifier data."""
    sheets = load_all()
    lookup: dict[tuple, tuple[float, float]] = {}
    df = sheets.get("WorldCup2026Qualifiers")
    if df is None or "HxG" not in df.columns:
        return lookup
    for _, row in df.iterrows():
        if pd.isna(row.get("HxG")) or pd.isna(row.get("AxG")):
            continue
        key = (row["Home"], row["Away"], str(row["Date"].date()))
        lookup[key] = (float(row["HxG"]), float(row["AxG"]))
    return lookup


# ── Rolling xG builder ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_xg_timelines() -> dict[str, list[tuple]]:
    """Per-team chronological list of (date, xg_for, xg_against)."""
    sheets = load_all()
    df = sheets.get("WorldCup2026Qualifiers")
    if df is None or "HxG" not in df.columns:
        return {}

    timelines: dict[str, list[tuple]] = {}
    df_sorted = df.dropna(subset=["HxG", "AxG"]).sort_values("Date")

    for _, row in df_sorted.iterrows():
        d = row["Date"]
        h, a = row["Home"], row["Away"]
        hxg, axg = float(row["HxG"]), float(row["AxG"])
        timelines.setdefault(h, []).append((d, hxg, axg))
        timelines.setdefault(a, []).append((d, axg, hxg))

    return timelines
