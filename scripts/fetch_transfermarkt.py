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
    "Australia":              (3433, "australien"),
    "Austria":                (3383, "osterreich"),
    "Belgium":                (3382, "belgien"),
    "Bosnia and Herzegovina": (3446, "bosnien-herzegowina"),
    "Brazil":                 (3439, "brasilien"),
    "Cabo Verde":             (4311, "kap-verde"),
    "Canada":                 (3510, "kanada"),
    "Colombia":               (3816, "kolumbien"),
    "Croatia":                (3556, "kroatien"),
    "Curaçao":                (32364, "curacao"),
    "Czechia":                (3445, "tschechien"),
    "DR Congo":               (3854, "demokratische-republik-kongo"),
    "Ecuador":                (5750, "ecuador"),
    "Egypt":                  (3672, "agypten"),
    "England":                (3299, "england"),
    "France":                 (3377, "frankreich"),
    "Germany":                (3262, "deutschland"),
    "Ghana":                  (3441, "ghana"),
    "Haiti":                  (14161, "haiti"),
    "IR Iran":                (3582, "iran"),
    "Iraq":                   (3560, "irak"),
    "Ivory Coast":            (3591, "elfenbeinküste"),
    "Japan":                  (3435, "japan"),
    "Jordan":                 (15737, "jordanien"),
    "Mexico":                 (6303, "mexiko"),
    "Morocco":                (3575, "marokko"),
    "Netherlands":            (3379, "niederlande"),
    "New Zealand":            (9171, "neuseeland"),
    "Norway":                 (3440, "norwegen"),
    "Panama":                 (3577, "panama"),
    "Paraguay":               (3581, "paraguay"),
    "Portugal":               (3300, "portugal"),
    "Qatar":                  (14162, "katar"),
    "Saudi Arabia":           (3807, "saudi-arabien"),
    "Scotland":               (3380, "schottland"),
    "Senegal":                (3499, "senegal"),
    "South Africa":           (3806, "sudafrika"),
    "South Korea":            (3589, "südkorea"),
    "Spain":                  (3375, "spanien"),
    "Sweden":                 (3557, "schweden"),
    "Switzerland":            (3384, "schweiz"),
    "Tunisia":                (3670, "tunesien"),
    "Türkiye":                (3381, "turkei"),
    "United States":          (3505, "usa"),
    "Uruguay":                (3449, "uruguay"),
    "Uzbekistan":             (3563, "usbekistan"),
    "Algeria":                (3614, "algerien"),
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

        # Primary: the dedicated market-value box in the page header
        # (contains e.g. "€1.05bn Total market value").
        for tag in soup.select("div.data-header__market-value-wrapper, a.data-header__market-value-wrapper"):
            txt = tag.get_text(" ", strip=True)
            if "€" in txt:
                val = _parse_value(txt.split("Total")[0])
                if val > 0:
                    return val

        # Fallback: header detail rows — but ONLY rows labelled "market value".
        # Never match on "squad": the header also contains "Squad size: 26",
        # which once scraped the 26-player WC squad limit as a €26M value for
        # every team and clobbered the cache.
        for tag in soup.select("div.data-header__details li"):
            label = tag.get_text(" ", strip=True).lower()
            if "market value" in label:
                val_tag = tag.select_one("span.data-header__content")
                if val_tag:
                    return _parse_value(val_tag.get_text(strip=True))

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

    # Sanity guard: refuse to overwrite the cache with implausible data.
    # Real WC-squad totals span ~€10M (Curaçao) to ~€2,500M (Spain). If the
    # page layout changes again and we scrape a constant or tiny values,
    # keep the existing cache rather than clobbering good data.
    nums = [v for v in values.values() if isinstance(v, (int, float))]
    n_zero = sum(1 for v in nums if v <= 0)
    if not nums or max(nums) < 100 or len(set(nums)) <= 3 or n_zero > len(nums) // 4:
        logger.error(
            "Scraped values look WRONG (max=%.1f, distinct=%d, zeros=%d/%d) — "
            "NOT overwriting %s. Fix the parser before re-running.",
            max(nums) if nums else 0.0, len(set(nums)), n_zero, len(nums), CACHE_PATH.name,
        )
        if CACHE_PATH.exists():
            with open(CACHE_PATH) as f:
                return json.load(f)
        raise SystemExit(1)

    # Preserve provenance keys from the existing cache, refresh the date note.
    meta = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                meta = {k: v for k, v in json.load(f).items() if k.startswith("_")}
        except Exception:
            meta = {}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({**meta, **values}, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", CACHE_PATH)
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = p.parse_args()

    values = fetch_all(force=args.force)
    print(f"\n{'Team':<30}  {'Value (€M)':>12}")
    print("-" * 45)
    for team, val in sorted(((k, v) for k, v in values.items() if isinstance(v, (int, float))), key=lambda x: -x[1]):
        print(f"{team:<30}  {val:>12.1f}")


if __name__ == "__main__":
    main()
