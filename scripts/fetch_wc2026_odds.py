#!/usr/bin/env python3
"""Fetch pre-match odds for all 72 WC 2026 fixtures from API-Football.

Markets extracted per fixture:
  - Match Winner (1X2) → implied probabilities
  - Goals Over/Under 2.5 → P(over 2.5 goals)
  - Asian Handicap → main line (home perspective)
  - Both Teams Score → P(BTTS)
  - Clean Sheet Home / Away → P(clean sheet each side)

Output: data/wc2026_odds_cache.json  keyed by fixture_id (str).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_CACHE = ROOT / "data" / "wc2026_odds_cache.json"
_FIXTURES = ROOT / "data" / "wc2026_fixtures_cache.json"
BASE = "https://v3.football.api-sports.io"
KEY = os.getenv("API_FOOTBALL_KEY", "")


def _get(endpoint: str, params: dict) -> dict:
    r = requests.get(f"{BASE}/{endpoint}", headers={"x-apisports-key": KEY},
                     params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _implied(odds: float) -> float:
    return 1.0 / max(odds, 1.01)


def _normalise_1x2(h: float, d: float, a: float) -> tuple[float, float, float]:
    total = h + d + a
    if total < 1e-6:
        return 1/3, 1/3, 1/3
    return h / total, d / total, a / total


def _parse_markets(bookmakers: list) -> dict:
    """Extract key market signals from the first bookmaker with sufficient coverage."""
    out: dict = {}
    for bk in bookmakers:
        bets = {b["name"]: b["values"] for b in bk.get("bets", [])}

        # ── 1X2 ─────────────────────────────────────────────────────────────
        if "Match Winner" in bets and "p1" not in out:
            vals = {v["value"]: _implied(float(v["odd"])) for v in bets["Match Winner"]}
            if all(k in vals for k in ("Home", "Draw", "Away")):
                p1, px, p2 = _normalise_1x2(vals["Home"], vals["Draw"], vals["Away"])
                out.update(p1=round(p1, 4), px=round(px, 4), p2=round(p2, 4))

        # ── Over/Under 2.5 ──────────────────────────────────────────────────
        # Vig-strip with the matching Under odd: p = imp_over/(imp_over+imp_under).
        # The previous min(raw, 0.97) only CAPPED the value — a two-way market
        # carries 4–7% margin, so the implied P(over) was biased ~2–3pp high,
        # inflating the market-implied λ_total by ~0.1–0.15 goals per match.
        if "Goals Over/Under" in bets and "over25" not in out:
            imp_over = imp_under = None
            for v in bets["Goals Over/Under"]:
                if v["value"] == "Over 2.5":
                    imp_over = _implied(float(v["odd"]))
                elif v["value"] == "Under 2.5":
                    imp_under = _implied(float(v["odd"]))
            if imp_over is not None:
                if imp_under is not None:
                    out["over25"] = round(imp_over / (imp_over + imp_under), 4)
                else:
                    # No Under odd available: fall back to a ~5% margin haircut
                    out["over25"] = round(min(imp_over / 1.05, 0.97), 4)

        # ── Asian Handicap (most balanced line, home perspective) ───────────
        # API uses "Home -X" (home gives X goals) and "Away -X" (away gives X
        # goals) as separate bets — they never pair as "Home -X" / "Away +X".
        # Fix: use only "Home X" entries; pick the line whose implied prob is
        # closest to 0.50 (that's the market's fair line from home perspective).
        if "Asian Handicap" in bets and "ah_line" not in out:
            best_line, best_gap = 0.0, float("inf")
            for v in bets["Asian Handicap"]:
                val, odd = v["value"], float(v["odd"])
                if not val.startswith("Home "):
                    continue
                try:
                    line = float(val.replace("Home ", ""))
                    prob = 1.0 / max(odd, 1.01)
                    gap = abs(prob - 0.5)
                    if gap < best_gap:
                        best_gap, best_line = gap, line
                except ValueError:
                    continue
            out["ah_line"] = best_line

        # ── Both Teams Score ─────────────────────────────────────────────────
        if "Both Teams Score" in bets and "btts" not in out:
            for v in bets["Both Teams Score"]:
                if v["value"] == "Yes":
                    out["btts"] = round(_implied(float(v["odd"])), 4)
                    break

        # ── Clean Sheet ──────────────────────────────────────────────────────
        if "Clean Sheet - Home" in bets and "cs_home" not in out:
            for v in bets["Clean Sheet - Home"]:
                if v["value"] == "Yes":
                    out["cs_home"] = round(_implied(float(v["odd"])), 4)
                    break
        if "Clean Sheet - Away" in bets and "cs_away" not in out:
            for v in bets["Clean Sheet - Away"]:
                if v["value"] == "Yes":
                    out["cs_away"] = round(_implied(float(v["odd"])), 4)
                    break

        if len(out) >= 8:
            break  # have everything we need

    return out


def fetch_all(force: bool = False) -> dict:
    if _CACHE.exists() and not force:
        print(f"  Cache hit: {_CACHE.name}")
        return json.loads(_CACHE.read_text())

    if not _FIXTURES.exists():
        print("  ERROR: wc2026_fixtures_cache.json not found — run fetch_api_form.py first")
        return {}

    fixtures = json.loads(_FIXTURES.read_text())
    results: dict = {}
    covered = 0

    print(f"  Fetching odds for {len(fixtures)} fixtures...")
    for i, f in enumerate(fixtures, 1):
        fid = str(f["fixture"]["id"])
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        date = f["fixture"]["date"][:10]

        try:
            resp = _get("odds", {"fixture": fid})
            bookmakers = []
            for item in resp.get("response", []):
                bookmakers.extend(item.get("bookmakers", []))

            markets = _parse_markets(bookmakers)
            if markets:
                markets.update(home_team=home, away_team=away, date=date)
                results[fid] = markets
                covered += 1
                print(f"  [{i:02d}/{len(fixtures)}] {home} vs {away}  "
                      f"p1={markets.get('p1','?')} px={markets.get('px','?')} "
                      f"p2={markets.get('p2','?')}  O/U={markets.get('over25','?')}")
            else:
                print(f"  [{i:02d}/{len(fixtures)}] {home} vs {away}  — no odds")
        except Exception as e:
            print(f"  [{i:02d}/{len(fixtures)}] {home} vs {away}  ERROR: {e}")

        if i < len(fixtures):
            time.sleep(0.5)

    _CACHE.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved {covered}/{len(fixtures)} fixtures → {_CACHE.name}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    fetch_all(force=args.force)
