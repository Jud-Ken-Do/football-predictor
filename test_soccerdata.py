"""Test fotmob & Sofascore with proper headers for CAF/CONCACAF xG."""
import requests, json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fotmob.com/",
    "Origin": "https://www.fotmob.com",
}

# fotmob — CAF WC qual (614) and CONCACAF (258)
for league_id, name in [(614, "CAF WC Qual"), (258, "CONCACAF WC Qual")]:
    url = f"https://www.fotmob.com/api/leagues?id={league_id}&tab=overview&type=league&timeZone=UTC"
    print(f"\n=== fotmob {name} (id={league_id}) ===")
    r = requests.get(url, headers=HEADERS, timeout=10)
    print(f"  status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(json.dumps(data, indent=2)[:1000])
    else:
        print(f"  body: {r.text[:200]}")

# Also try a known fotmob match to see if xG is in the response
print("\n=== fotmob match details (test with a known match id) ===")
# Try to get matches for CAF by date
url2 = "https://www.fotmob.com/api/matches?date=20240606"
r2 = requests.get(url2, headers=HEADERS, timeout=10)
print(f"  matches status: {r2.status_code}")
if r2.status_code == 200:
    data2 = r2.json()
    leagues = data2.get("leagues", [])
    for lg in leagues:
        if any(k in str(lg.get("name", "")).lower() for k in ["caf", "africa", "concacaf"]):
            print(f"  Found: {lg.get('name')} — {len(lg.get('matches', []))} matches")
            for m in lg.get("matches", [])[:2]:
                print(f"    match_id={m.get('id')} {m.get('home', {}).get('name')} vs {m.get('away', {}).get('name')}")
