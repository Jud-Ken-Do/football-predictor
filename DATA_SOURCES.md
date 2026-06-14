# Data Sources & the xG Gap

## 1. Where our actual data comes from

| Source | Module | What it provides | Status |
|---|---|---|---|
| **martj42 `international_results`** (GitHub CSV, also on Kaggle) | `data/sources/international_results.py` | 47,000+ international matches 1872–present (date, teams, scores, tournament). **Primary training data.** | ✅ live, auto-refresh if cache >48h |
| **football-data.co.uk** `WorldCup2026.xlsx` | `data/sources/football_data_co_uk.py` | WC-2026 **qualifier xG** (339 matches), bookmaker **closing odds** (WC 2018/2022), shots/corners | ✅ `data/raw/WorldCup2026.xlsx` present |
| **API-Football** (api-sports.io v3) | `data/sources/api_football.py` | Injuries/suspensions, last-10 form, live **market odds** (AH / O-U 2.5 / BTTS / 1X2) for WC 2026 | ✅ cached (`data/api_form_cache.json`, `data/wc2026_odds_cache.json`) |
| **football-data.org** REST API | `data/sources/football_data_org.py` | European league results (free tier: 2 seasons) | ✅ secondary |
| **Dato-Futbol** FIFA rankings | `features/rankings.py` | FIFA world-ranking points 1992–2024 | ✅ |
| **EA FC 26 / sofifa** | `features/sofifa_ratings.py` | Player ratings (overall, pace, shooting, passing, star rating) — **static snapshot**, post-processing only | ✅ |
| **Transfermarkt** | `features/transfermarkt.py` | Squad market values (June-2026 snapshot) — post-processing only | ✅ |
| **Recorded WC 2026 results** | `data/wc2026_actual_results.json` | Actual played-match scores, entered via `scripts/update_wc2026.py` | ✅ manual |

---

## 2. xG coverage — what we have

xG (expected goals) measures *chance quality*, a stronger signal than raw goals on
small samples. We source it from two places (`features/xg_form.py`):

1. **football-data.co.uk `WorldCup2026.xlsx`** — 339 qualifier matches with xG.
   Covers **UEFA / CONMEBOL / AFC** qualifiers and parts of CONCACAF.
2. **FBref (StatsBomb-powered)** — *intended* fallback for CAF + CONCACAF,
   scraped via `scripts/fetch_xg_fbref.py` into `data/xg_fbref.json`.

**Current coverage: 35 of 48 WC teams have xG.**

---

## 3. The gap — 13 of 48 teams have NO xG

Missing (≈27% of the field):

- **CAF (9):** Algeria, Cabo Verde, Egypt, Ghana, Ivory Coast, Morocco, Senegal, South Africa, Tunisia
- **CONCACAF hosts (3):** Canada, Mexico, United States
- **OFC (1):** New Zealand

These teams fall back to `_DEFAULT_XG = 1.15` (population mean) with
`xg_available = 0.0`, i.e. the xG features carry **no team-specific signal** for them.

### Where I checked / why it isn't working

| Check | Finding |
|---|---|
| `data/raw/WorldCup2026.xlsx` (Excel xG) | Present, 339 matches — but its `WorldCup2026Qualifiers` sheet only covers UEFA/CONMEBOL/AFC + some CONCACAF. **CAF and Mexico/USA/Canada are simply not in the sheet.** |
| `data/xg_fbref.json` (FBref fallback) | **File is 2 bytes — effectively empty.** The intended CAF/CONCACAF backfill never populated. |
| `scripts/fetch_xg_fbref.py` (the scraper) | Exists, but FBref sits behind **Cloudflare**, which blocks automated requests → the scrape returns nothing, leaving the cache empty. |
| API-Football (api-sports.io) | Has fixtures/odds/injuries for these teams but **no xG** for CAF/CONCACAF internationals on our tier. |

### Why it's effectively "not available (for free, automated)"

- **FBref** has the data but is **Cloudflare-blocked** for automated scraping.
- **No free automated xG feed** exists for CAF or CONCACAF *international* matches
  (Understat is club-only; football-data.co.uk doesn't carry these confederations).
- The only reliable routes are **paid** (Opta / StatsBomb licences) or **manual**
  (hand-copying from FBref behind the browser, which Cloudflare permits interactively).

### Options to close it

1. **StatsBomb Open Data** (free, GitHub) — has *some* international matches; worth
   scanning for CAF/CONCACAF WC-qualifier coverage.
2. **Manual FBref pull** — open the team pages in a real browser (Cloudflare lets a
   human through), export to CSV, drop into `data/xg_fbref.json` in the existing
   `{home, away, date, xg_home, xg_away}` shape — `fetch_xg_fbref.py` already knows
   how to read it.
3. **Paid Opta/StatsBomb** — full coverage, not free.
4. **Accept the gap** — current behaviour: these 13 teams lean on Elo/Kalman/form
   instead of xG. This is the documented architectural limitation, not a bug.

> Note: closing this gap is *completing an existing feature* for ~1/3 of the field —
> likely higher leverage than adding a brand-new weak signal (see `MISSING_SIGNALS.md`).
