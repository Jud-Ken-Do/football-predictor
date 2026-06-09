#!/usr/bin/env python3.11
"""Scrape WC 2026 squad market values from Transfermarkt.

Fetches the total squad market value (€M) for all 48 WC 2026 teams and
caches to data/transfermarkt_wc2026.json.

Usage:
    python3.11 scripts/fetch_transfermarkt.py
    python3.11 scripts/fetch_transfermarkt.py --force   # re-fetch even if cached
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "transfermarkt_wc2026.json"

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

# Transfermarkt national team IDs for all 48 WC 2026 teams
# Format: canonical_name → (tm_id, tm_slug)
_TM_TEAMS: dict[str, tuple[int, str]] = {
    "Argentina":              (3437, "argentinien"),
    "Australia":              (3517, "australien"),
    "Austria":                (3404, "osterreich"),
    "Belgium":                (3382, "belgien"),
    "Bosnia and Herzegovina": (3679, "bosnien-herzegowina"),
    "Brazil":                 (3439, "brasilien"),
    "Cabo Verde":             (8456, "kap-verde"),
    "Canada":                 (3630, "kanada"),
    "Colombia":               (3687, "kolumbien"),
    "Croatia":                (3556, "kroatien"),
    "Curaçao":                (10267, "curacao"),
    "Czechia":                (3376, "tschechien"),
    "DR Congo":               (3495, "dr-kongo"),
    "Ecuador":                (3457, "ecuador"),
    "Egypt":                  (3481, "agypten"),
    "England":                (3412, "england"),
    "France":                 (3377, "frankreich"),
    "Germany":                (3378, "deutschland"),
    "Ghana":                  (3498, "ghana"),
    "Haiti":                  (3644, "haiti"),
    "IR Iran":                (3484, "iran"),
    "Iraq":                   (3485, "irak"),
    "Ivory Coast":            (3497, "elfenbeinküste"),
    "Japan":                  (3519, "japan"),
    "Jordan":                 (3487, "jordanien"),
    "Mexico":                 (3637, "mexiko"),
    "Morocco":                (3480, "marokko"),
    "Netherlands":            (3379, "niederlande"),
    "New Zealand":            (3526, "neuseeland"),
    "Norway":                 (3380, "norwegen"),
    "Panama":                 (3648, "panama"),
    "Paraguay":               (3455, "paraguay"),
    "Portugal":               (3381, "portugal"),
    "Qatar":                  (3483, "katar"),
    "Saudi Arabia":           (3486, "saudi-arabien"),
    "Scotland":               (3411, "schottland"),
    "Senegal":                (3503, "senegal"),
    "South Africa":           (3504, "sudafrika"),
    "South Korea":            (3518, "südkorea"),
    "Spain":                  (3375, "spanien"),
    "Sweden":                 (3386, "schweden"),
    "Switzerland":            (3384, "schweiz"),
    "Tunisia":                (3482, "tunesien"),
    "Türkiye":                (3383, "turkei"),
    "United States":          (3629, "usa"),
    "Uruguay":                (3456, "uruguay"),
    "Uzbekistan":             (10693, "usbekistan"),
    "Algeria":                (3479, "algerien"),
}


def _parse_value(text: str) -> float:
    """Parse '€450.80m' or '€23.45m' or '€1.20bn' → float (millions)."""
    t = text.strip().replace("\xa0", "").replace(",", ".")
    t = t.lstrip("€").strip()
    if not t or t == "-":
        return 0.0
    try:
        if t.endswith("bn"):
            return float(t[:-2]) * 1000
        if t.endswith("m"):
            return float(t[:-1])
        if t.endswith("k"):
            return float(t[:-1]) / 1000
        return float(t)
    except ValueError:
        return 0.0


def fetch_team_value(team_name: str, tm_id: int, tm_slug: str) -> float:
    """Return total squad market value in €M for one team."""
    url = f"https://www.transfermarkt.com/{tm_slug}/startseite/verein/{tm_id}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Squad total value is in the header box
        for tag in soup.select("div.data-header__details li"):
            label = tag.get_text(" ", strip=True).lower()
            if "market value" in label or "squad" in label:
                val_tag = tag.select_one("span.data-header__content")
                if val_tag:
                    return _parse_value(val_tag.get_text(strip=True))

        # Fallback: look for the summary line
        for tag in soup.select("div.data-header__market-value-wrapper a"):
            txt = tag.get_text(strip=True)
            if txt:
                return _parse_value(txt)

    except Exception as e:
        logger.warning("  %s: fetch failed (%s)", team_name, e)
    return 0.0


def fetch_all(force: bool = False) -> dict[str, float]:
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        logger.info("Loaded %d teams from cache (%s)", len(cached), CACHE_PATH)
        return cached

    values: dict[str, float] = {}
    for i, (team, (tm_id, slug)) in enumerate(_TM_TEAMS.items(), 1):
        logger.info("[%d/%d] %s ...", i, len(_TM_TEAMS), team)
        val = fetch_team_value(team, tm_id, slug)
        values[team] = val
        logger.info("  → €%.1f M", val)
        time.sleep(1.5)  # polite rate limit

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(values, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", CACHE_PATH)
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = p.parse_args()

    values = fetch_all(force=args.force)
    print(f"\n{'Team':<30}  {'Value (€M)':>12}")
    print("-" * 45)
    for team, val in sorted(values.items(), key=lambda x: -x[1]):
        print(f"{team:<30}  {val:>12.1f}")


if __name__ == "__main__":
    main()
