#!/usr/bin/env python3.11
# DEPRECATED — FBref is Cloudflare-blocked as of June 2026; automated scraping
# no longer works. CAF/CONCACAF xG remains a known data gap. If FBref CSV files
# are manually downloaded, drop them into data/xg_fbref.json and xg_form.py will
# pick them up automatically as a fallback source.
"""Scrape xG data from FBref for WC 2026 qualifying matches.

Targets the two confederations missing from WorldCup2026.xlsx:
  - CAF (African): Morocco, Egypt, Senegal, Ghana, Tunisia, etc.
  - CONCACAF (North/Central America): Mexico, USA, Canada, Curaçao

FBref publishes national-team schedule pages with xG columns (from StatsBomb).
Data is merged with the existing Excel xG and saved to data/xg_fbref.json.
xg_form.py is updated to prefer fbref data as a second source.

Usage:
    python3.11 scripts/fetch_xg_fbref.py            # fetch & save
    python3.11 scripts/fetch_xg_fbref.py --dry-run  # just show what would be fetched
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "xg_fbref.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# FBref competition IDs for WC 2026 qualifying rounds
# Each entry: (competition_id, slug, confederation)
# Note: Only the final qualifying round is used (group stage with xG available).
_COMPETITIONS = [
    # CAF — African WC Qualifying (Round 3, final group stage 2025)
    (107, "CAF-World-Cup-Qualifying", "CAF"),
    # CONCACAF — Octagonal / Final Qualifying 2025
    (30,  "CONCACAF-World-Cup-Qualifying", "CONCACAF"),
]

# FBref team name → our canonical name
_FBREF_NAME_MAP: dict[str, str] = {
    # CAF
    "Morocco":              "Morocco",
    "Egypt":                "Egypt",
    "Senegal":              "Senegal",
    "Ghana":                "Ghana",
    "Tunisia":              "Tunisia",
    "South Africa":         "South Africa",
    "Ivory Coast":          "Ivory Coast",
    "DR Congo":             "DR Congo",
    "Congo DR":             "DR Congo",
    "Cameroon":             "Cameroon",
    "Algeria":              "Algeria",
    "Mali":                 "Mali",
    "Burkina Faso":         "Burkina Faso",
    "Nigeria":              "Nigeria",
    "Cabo Verde":           "Cabo Verde",
    "Cape Verde":           "Cabo Verde",
    # CONCACAF
    "United States":        "United States",
    "Mexico":               "Mexico",
    "Canada":               "Canada",
    "Panama":               "Panama",
    "Honduras":             "Honduras",
    "Costa Rica":           "Costa Rica",
    "Jamaica":              "Jamaica",
    "Curaçao":              "Curaçao",
    "Curacao":              "Curaçao",
    "Haiti":                "Haiti",
    "Trinidad and Tobago":  "Trinidad and Tobago",
    "El Salvador":          "El Salvador",
}


def _norm(name: str) -> str:
    s = str(name).strip()
    return _FBREF_NAME_MAP.get(s, s)


def _fetch_schedule(comp_id: int, slug: str) -> pd.DataFrame | None:
    """Fetch a FBref schedule/scores page and parse xG columns."""
    url = f"https://fbref.com/en/comps/{comp_id}/schedule/{slug}-Schedules-and-Scores"
    logger.info("  GET %s", url)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("  Fetch failed: %s", exc)
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # FBref schedule tables have id="sched_all" or similar
    table = soup.find("table", {"id": lambda v: v and "sched" in v})
    if table is None:
        logger.warning("  No schedule table found — page structure may have changed")
        return None

    try:
        dfs = pd.read_html(str(table))
        df = dfs[0]
    except Exception as exc:
        logger.warning("  Failed to parse table: %s", exc)
        return None

    return df


def _extract_xg(df: pd.DataFrame, conf: str) -> list[dict]:
    """Extract (date, home, away, xg_home, xg_away) rows from a schedule DataFrame."""
    # Column names vary by FBref version — handle both
    col_map: dict[str, list[str]] = {
        "date":     ["Date"],
        "home":     ["Home", "Squad"],
        "away":     ["Away", "Opponent"],
        "score":    ["Score"],
        "xg_home":  ["xG", "Home xG", "HxG", "xg"],
        "xg_away":  ["xGA", "Away xG", "AxG", "xga"],
    }

    def find_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    date_col  = find_col(col_map["date"])
    home_col  = find_col(col_map["home"])
    away_col  = find_col(col_map["away"])
    xgh_col   = find_col(col_map["xg_home"])
    xga_col   = find_col(col_map["xg_away"])

    if not all([date_col, home_col, away_col, xgh_col, xga_col]):
        logger.warning("  Missing columns: date=%s home=%s away=%s xgh=%s xga=%s",
                       date_col, home_col, away_col, xgh_col, xga_col)
        logger.warning("  Available: %s", list(df.columns))
        return []

    rows = []
    for _, row in df.iterrows():
        try:
            date_val = pd.to_datetime(row[date_col], errors="coerce")
            if pd.isna(date_val):
                continue
            xgh = pd.to_numeric(row[xgh_col], errors="coerce")
            xga = pd.to_numeric(row[xga_col], errors="coerce")
            if pd.isna(xgh) or pd.isna(xga):
                continue
            home = _norm(str(row[home_col]))
            away = _norm(str(row[away_col]))
            rows.append({
                "date":     str(date_val.date()),
                "home":     home,
                "away":     away,
                "xg_home":  float(xgh),
                "xg_away":  float(xga),
                "conf":     conf,
            })
        except Exception:
            continue
    return rows


def fetch_all(dry_run: bool = False) -> list[dict]:
    all_rows: list[dict] = []

    for comp_id, slug, conf in _COMPETITIONS:
        logger.info("\n[%s] comp_id=%d", conf, comp_id)
        if dry_run:
            logger.info("  (dry-run — skipping actual fetch)")
            continue

        df = _fetch_schedule(comp_id, slug)
        if df is None:
            continue

        rows = _extract_xg(df, conf)
        logger.info("  → %d matches with xG", len(rows))
        all_rows.extend(rows)
        time.sleep(2.0)  # polite rate limit for FBref

    return all_rows


def save(rows: list[dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info("\nSaved %d rows → %s", len(rows), CACHE_PATH)


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("\nNo data fetched.")
        return
    import collections
    by_conf: dict[str, int] = collections.Counter(r["conf"] for r in rows)
    teams: set[str] = set()
    for r in rows:
        teams.add(r["home"])
        teams.add(r["away"])
    print(f"\n{'─'*50}")
    print(f"  xG data fetched: {len(rows)} matches, {len(teams)} teams")
    for conf, cnt in sorted(by_conf.items()):
        print(f"    {conf}: {cnt} matches")
    print(f"\n  Teams covered:")
    for t in sorted(teams):
        print(f"    {t}")
    print(f"{'─'*50}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Show plan without fetching")
    p.add_argument("--force",   action="store_true", help="Re-fetch even if cache exists")
    args = p.parse_args()

    if CACHE_PATH.exists() and not args.force and not args.dry_run:
        logger.info("Cache exists at %s  (use --force to re-fetch)", CACHE_PATH)
        with open(CACHE_PATH) as f:
            rows = json.load(f)
        print_summary(rows)
        return

    rows = fetch_all(dry_run=args.dry_run)

    if not args.dry_run:
        save(rows)

    print_summary(rows)


if __name__ == "__main__":
    main()
