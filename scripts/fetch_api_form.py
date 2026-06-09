"""Fetch last-10 international fixtures for all 48 WC 2026 teams from API-Football.

Run before each round to refresh form data:
    python3.11 scripts/fetch_api_form.py

Uses ~50 API requests (one per team + status check). Saves to data/api_form_cache.json.
"""
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["API_FOOTBALL_KEY"]
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": KEY}
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "api_form_cache.json")


def get(endpoint: str, params: dict) -> dict:
    resp = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    # Step 1: get all 48 WC 2026 team IDs
    print("Fetching WC 2026 teams...")
    data = get("teams", {"league": 1, "season": 2026})
    teams = data.get("response", [])
    print(f"  {len(teams)} teams")

    cache: dict = {}
    for i, entry in enumerate(sorted(teams, key=lambda x: x["team"]["name"])):
        team = entry["team"]
        team_id = team["id"]
        team_name = team["name"]

        fixtures_data = get("fixtures", {"team": team_id, "last": 10})
        fixtures = fixtures_data.get("response", [])

        cache[team_name] = {
            "team_id": team_id,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fixtures": fixtures,
        }

        results = []
        for f in fixtures[:5]:
            home = f["teams"]["home"]
            away = f["teams"]["away"]
            winner = home["winner"] if home["id"] == team_id else away["winner"]
            results.append("W" if winner is True else "L" if winner is False else "D")
        print(f"  [{i+1:2d}/48] {team_name:30s} last5={' '.join(results[:5])}")

        time.sleep(0.15)  # stay within rate limit

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cache)} teams to {OUT}")

    # Show remaining requests
    status = get("status", {})
    req = status["response"]["requests"]
    print(f"API requests used today: {req['current']}/{req['limit_day']}")


if __name__ == "__main__":
    main()
