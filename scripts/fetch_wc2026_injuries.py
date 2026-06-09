#!/usr/bin/env python3.11
"""Fetch and cache WC 2026 pre-match injury/suspension data from API-Football.

Run this once before each match day to refresh the injury cache.
The injury feature module reads from data/wc2026_injuries_cache.json.

Usage:
    python3.11 scripts/fetch_wc2026_injuries.py
    python3.11 scripts/fetch_wc2026_injuries.py --force   # re-fetch everything

Free tier: 100 requests/day — enough for all WC 2026 fixtures.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from football_predictor.data.sources.api_football import fetch_and_cache_all_injuries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-fetch all fixtures")
    args = p.parse_args()

    print("Fetching WC 2026 injury data from API-Football...")
    cache = fetch_and_cache_all_injuries(force=args.force)
    n_injured = sum(len(v) for v in cache.values())
    print(f"Done. {len(cache)} fixtures cached, {n_injured} total injury/suspension entries.")
    print(f"Run predict_wc2026.py to use updated data.")


if __name__ == "__main__":
    main()
