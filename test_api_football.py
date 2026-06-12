"""Check odds coverage and what markets we can extract as features."""
import os, requests, json
from dotenv import load_dotenv
load_dotenv()

KEY = os.getenv("API_FOOTBALL_KEY", "")
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": KEY}

def get(endpoint, params={}):
    r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# Get all WC 2026 fixture IDs
fixtures = get("fixtures", {"league": 1, "season": 2026})
fids = [f["fixture"]["id"] for f in fixtures.get("response", [])]
print(f"Total WC 2026 fixtures: {len(fids)}")

# Check odds for first 5 fixtures
covered = 0
markets_seen = set()
samples = []

for fid in fids[:5]:
    odds = get("odds", {"fixture": fid})
    resp = odds.get("response", [])
    if resp and resp[0].get("bookmakers"):
        covered += 1
        bk = resp[0]["bookmakers"][0]
        bet_names = [b["name"] for b in bk.get("bets", [])]
        markets_seen.update(bet_names)

        # Extract implied probabilities from 1X2 and Asian Handicap
        sample = {"fixture": fid}
        for bet in bk.get("bets", []):
            if bet["name"] == "Match Winner":
                vals = {v["value"]: float(v["odd"]) for v in bet["values"]}
                # Convert odds to implied prob (normalize)
                inv = {k: 1/v for k, v in vals.items()}
                total = sum(inv.values())
                sample["p_home"] = round(inv.get("Home", 0) / total, 3)
                sample["p_draw"] = round(inv.get("Draw", 0) / total, 3)
                sample["p_away"] = round(inv.get("Away", 0) / total, 3)
            elif bet["name"] == "Goals Over/Under":
                for v in bet["values"]:
                    if v["value"] == "Over 2.5":
                        sample["odds_over2.5"] = float(v["odd"])
            elif bet["name"] == "Asian Handicap":
                sample["asian_handicap_available"] = True
        samples.append(sample)
    else:
        samples.append({"fixture": fid, "no_odds": True})

print(f"\nOdds coverage (first 5 fixtures): {covered}/5")
print(f"\nAll markets available:")
for m in sorted(markets_seen):
    print(f"  - {m}")

print(f"\nSample implied probabilities:")
for s in samples:
    print(f"  fixture={s.get('fixture')}  p_home={s.get('p_home','?')}  p_draw={s.get('p_draw','?')}  p_away={s.get('p_away','?')}  over2.5={s.get('odds_over2.5','?')}")
