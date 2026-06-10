#!/usr/bin/env python3.11
"""Scrape pre-tournament injury history for all WC 2026 squads from Transfermarkt.

Two-step scrape:
  1. Team squad pages → player IDs, positions, market values
  2. Each key player's injury history page (last 2 years)

Only fetches injury history for players valued ≥ MIN_VALUE_M (default €5M) to
keep the request count manageable (~500 requests, ~15 minutes with polite delays).

Output: data/wc2026_player_injuries.json

  {
    "Argentina": [
      {
        "player_id":      385110,
        "slug":           "cristian-romero",
        "name":           "Cristian Romero",
        "position":       "Centre-Back",
        "market_value_m": 65.0,
        "injuries": [
          {
            "from":         "2024-03-15",
            "until":        "2024-04-10",   // null = ongoing
            "days":         26,
            "type":         "Muscular problems",
            "games_missed": 5
          }
        ]
      }
    ]
  }

Usage:
    python3.11 scripts/fetch_injury_history.py
    python3.11 scripts/fetch_injury_history.py --force        # re-fetch everything
    python3.11 scripts/fetch_injury_history.py --min-value 10 # only players ≥ €10M
    python3.11 scripts/fetch_injury_history.py --team Argentina  # one team only
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_PATH = ROOT / "data" / "wc2026_player_injuries.json"

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

# Same team map as fetch_transfermarkt.py
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

# Injury type severity weights (higher = more impactful)
SEVERITY: dict[str, float] = {
    "ligament":    3.0,
    "acl":         3.0,
    "cruciate":    3.0,
    "tendon":      2.5,
    "fracture":    2.5,
    "broken":      2.5,
    "hamstring":   2.0,
    "muscle":      1.5,
    "muscular":    1.5,
    "thigh":       1.5,
    "knock":       0.8,
    "illness":     0.8,
    "groin":       1.5,
    "calf":        1.5,
    "ankle":       2.0,
    "knee":        2.0,
    "back":        1.5,
    "suspension":  0.5,
}


def _get(url: str, delay: float = 1.2) -> BeautifulSoup | None:
    """Fetch URL and return BeautifulSoup, with polite delay."""
    time.sleep(delay)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        logger.warning("  GET failed: %s — %s", url, e)
        return None


def _parse_value(text: str) -> float:
    """Parse '€65.00m' or '€450k' → float millions."""
    t = text.strip().replace("\xa0", "").replace(",", ".").lstrip("€").strip()
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


def _parse_date(text: str) -> str | None:
    """Parse Transfermarkt date strings → ISO format YYYY-MM-DD."""
    text = text.strip()
    if not text or text == "-":
        return None
    for fmt in ("%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def fetch_squad(team_name: str, tm_id: int, tm_slug: str) -> list[dict]:
    """Scrape team squad page → list of player dicts with id, slug, name, position, value."""
    url = f"https://www.transfermarkt.com/{tm_slug}/kader/verein/{tm_id}/saison_id/2025/plus/1"
    soup = _get(url)
    if soup is None:
        return []

    players = []
    seen_ids: set[int] = set()

    for row in soup.select("table.items tbody tr.odd, table.items tbody tr.even"):
        # Player link: /name/profil/spieler/ID
        link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if not link:
            continue

        href = link.get("href", "")
        m = re.search(r"/spieler/(\d+)", href)
        if not m:
            continue

        player_id = int(m.group(1))
        if player_id in seen_ids:
            continue
        seen_ids.add(player_id)

        slug_match = re.match(r"/([^/]+)/profil/spieler/", href)
        slug = slug_match.group(1) if slug_match else ""

        name = link.get_text(strip=True)

        # Position
        pos_td = row.select_one("td.posrela table td")
        position = pos_td.get_text(strip=True) if pos_td else ""

        # Market value (last right-aligned cell)
        val_td = row.select_one("td.rechts.hauptlink")
        value = _parse_value(val_td.get_text(strip=True)) if val_td else 0.0

        players.append({
            "player_id": player_id,
            "slug": slug,
            "name": name,
            "position": position,
            "market_value_m": value,
        })

    return players


def fetch_player_injuries(player_id: int, slug: str, since_days: int = 730) -> list[dict]:
    """Scrape player injury history page → list of injury dicts (last `since_days` days)."""
    url = f"https://www.transfermarkt.com/{slug}/verletzungen/spieler/{player_id}"
    soup = _get(url, delay=1.0)
    if soup is None:
        return []

    cutoff = (date.today() - timedelta(days=since_days)).isoformat()
    injuries = []

    table = soup.select_one("table.items")
    if not table:
        return []

    for row in table.select("tbody tr"):
        cells = row.select("td")
        if len(cells) < 5:
            continue

        texts = [c.get_text(strip=True) for c in cells]

        injury_type = texts[1] if len(texts) > 1 else ""
        from_str   = _parse_date(texts[2]) if len(texts) > 2 else None
        until_str  = _parse_date(texts[3]) if len(texts) > 3 else None

        # Skip if too old
        if from_str and from_str < cutoff:
            continue

        # Days out
        days = 0
        if len(texts) > 4:
            try:
                days = int(re.sub(r"[^\d]", "", texts[4]))
            except (ValueError, IndexError):
                pass

        # Games missed
        games = 0
        if len(texts) > 5:
            try:
                games = int(re.sub(r"[^\d]", "", texts[5]))
            except (ValueError, IndexError):
                pass

        if not injury_type or injury_type.lower() in ("", "-"):
            continue

        injuries.append({
            "from":         from_str,
            "until":        until_str,   # None = ongoing
            "days":         days,
            "type":         injury_type,
            "games_missed": games,
        })

    return injuries


def severity_score(injury_type: str) -> float:
    """Map injury type string to severity weight."""
    low = injury_type.lower()
    for keyword, weight in SEVERITY.items():
        if keyword in low:
            return weight
    return 1.0


def fetch_all(
    force: bool = False,
    min_value_m: float = 5.0,
    team_filter: str | None = None,
    delay: float = 1.2,
) -> dict:
    """Main orchestrator. Returns full injury data dict."""
    existing: dict = {}
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH) as f:
            existing = json.load(f)
        logger.info("Loaded %d teams from cache.", len(existing))

    teams = _TM_TEAMS
    if team_filter:
        teams = {k: v for k, v in teams.items() if k.lower() == team_filter.lower()}
        if not teams:
            logger.error("Team '%s' not found.", team_filter)
            return existing

    for i, (team, (tm_id, slug)) in enumerate(teams.items(), 1):
        if team in existing and not force:
            logger.info("[%d/%d] %s — cached, skipping", i, len(teams), team)
            continue

        logger.info("[%d/%d] %s — fetching squad ...", i, len(teams), team)
        players = fetch_squad(team, tm_id, slug)
        logger.info("  %d players found", len(players))

        if not players:
            existing[team] = []
            continue

        # Only fetch injury history for players above value threshold
        key_players = [p for p in players if p["market_value_m"] >= min_value_m]
        logger.info("  %d key players (≥ €%.0fM)", len(key_players), min_value_m)

        for p in key_players:
            logger.info("    %s (%s, €%.1fM) ...", p["name"], p["position"], p["market_value_m"])
            p["injuries"] = fetch_player_injuries(p["player_id"], p["slug"])
            n_inj = len(p["injuries"])
            if n_inj:
                logger.info("      → %d injury entries", n_inj)

        # Players below threshold get empty injury list
        for p in players:
            if "injuries" not in p:
                p["injuries"] = []

        existing[team] = players

        # Save after each team so partial progress is preserved
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info("Done. Saved to %s", CACHE_PATH)
    return existing


def print_summary(data: dict, as_of: str | None = None) -> None:
    """Print a human-readable injury summary."""
    ref_date = as_of or date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"  Pre-tournament injury summary (as of {ref_date})")
    print(f"{'='*60}")

    for team, players in sorted(data.items()):
        current: list[str] = []
        recent: list[str] = []
        for p in players:
            for inj in p.get("injuries", []):
                until = inj.get("until")
                frm   = inj.get("from") or ""
                if until is None and frm >= "2026-01-01":
                    current.append(f"{p['name']} ({inj['type']})")
                elif until and until >= "2026-04-01" and until <= ref_date:
                    recent.append(f"{p['name']} ({inj['type']}, returned {until})")

        if current or recent:
            print(f"\n  {team}:")
            for c in current:
                print(f"    ❌ ONGOING: {c}")
            for r in recent:
                print(f"    ⚠️  RECENT:  {r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force",     action="store_true", help="Re-fetch all teams")
    p.add_argument("--min-value", type=float, default=5.0, help="Min player value €M (default 5)")
    p.add_argument("--team",      type=str,  default=None, help="Fetch one team only")
    p.add_argument("--summary",   action="store_true", help="Print injury summary from cache")
    args = p.parse_args()

    if args.summary and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        print_summary(data)
        return

    data = fetch_all(force=args.force, min_value_m=args.min_value, team_filter=args.team)
    print_summary(data)


if __name__ == "__main__":
    main()
