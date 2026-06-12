"""FIFA World Cup 2026 fixture data.

Hosts: USA, Canada, Mexico.
Format: 48 teams, 12 groups of 4.
        Top 2 from each group + 8 best 3rd-place teams advance to Round of 32.

Draw: 5 December 2024, Miami.
Playoff winners (resolved from schedule):
  - European Playoff D Winner: Czechia
  - European Playoff A Winner: Bosnia and Herzegovina
  - European Playoff C Winner: Türkiye
  - European Playoff B Winner: Sweden
  - Intercontinental Playoff 2: Iraq
  - Intercontinental Playoff 1: DR Congo

Note: home/away labels are nominal. All matches at neutral venues.
"""
from __future__ import annotations

import itertools

# ── Official Groups (draw + playoff winners resolved) ──────────────────────────

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ── Team name normalisations ───────────────────────────────────────────────────
# Maps name variants from any source → canonical training-data names.
#
# Canonical = martj42 names AFTER international_results._NAME_MAP renames:
#   Czechia, Türkiye, Cabo Verde, IR Iran (renamed), plus South Korea,
#   Ivory Coast, Bosnia and Herzegovina, DR Congo, United States, Curaçao
#   (unchanged from the raw CSV).
# All 48 WC 2026 draw names are themselves canonical → identity passthrough.

TEAM_NAME_MAP: dict[str, str] = {
    # Pre-rename martj42 / FIFA / football-data.co.uk variants
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Cape Verde": "Cabo Verde",
    "Iran": "IR Iran",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    # API-Football name variants
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cabo Verde",
    "Congo DR": "DR Congo",
    "D.R. Congo": "DR Congo",
    "USA": "United States",
    "Curacao": "Curaçao",
}

# All 48 canonical team names (draw names) — used for ingress validation.
ALL_WC2026_TEAMS: frozenset[str] = frozenset(
    t for teams in GROUPS.values() for t in teams
)


def normalise(team: str) -> str:
    """Map any source's team-name variant to the canonical dataset name."""
    return TEAM_NAME_MAP.get(team, team)


def validate_team_coverage(df, min_matches: int = 10) -> list[str]:
    """Return WC 2026 teams under-represented in a training DataFrame.

    Guards against silent name-mismatch bugs: a team whose normalised name
    doesn't appear in the data would otherwise fall back to default ratings /
    league-average strength without any error.
    """
    import pandas as pd

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    problems = []
    for t in sorted(ALL_WC2026_TEAMS):
        c = int(counts.get(normalise(t), 0))
        if c < min_matches:
            problems.append(f"{t} → '{normalise(t)}' ({c} matches)")
    return problems


# ── Group stage fixtures (72 matches: 6 per group) ────────────────────────────

# Full group stage schedule with dates and venues from the official schedule.
GROUP_STAGE_SCHEDULE: list[dict] = [
    # ── June 11 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-11", "group": "A", "home_team": "Mexico",       "away_team": "South Africa",  "venue": "Mexico City"},
    {"date": "2026-06-11", "group": "A", "home_team": "South Korea",  "away_team": "Czechia",        "venue": "Zapopan"},
    # ── June 12 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-12", "group": "B", "home_team": "Canada",       "away_team": "Bosnia and Herzegovina", "venue": "Toronto"},
    {"date": "2026-06-12", "group": "D", "home_team": "United States","away_team": "Paraguay",       "venue": "Inglewood"},
    # ── June 13 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-13", "group": "B", "home_team": "Qatar",        "away_team": "Switzerland",   "venue": "Santa Clara"},
    {"date": "2026-06-13", "group": "C", "home_team": "Brazil",       "away_team": "Morocco",        "venue": "East Rutherford"},
    {"date": "2026-06-13", "group": "C", "home_team": "Haiti",        "away_team": "Scotland",       "venue": "Foxborough"},
    {"date": "2026-06-13", "group": "D", "home_team": "Australia",    "away_team": "Türkiye",        "venue": "Vancouver"},
    # ── June 14 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-14", "group": "E", "home_team": "Germany",      "away_team": "Curaçao",        "venue": "Houston"},
    {"date": "2026-06-14", "group": "F", "home_team": "Netherlands",  "away_team": "Japan",          "venue": "Arlington"},
    {"date": "2026-06-14", "group": "E", "home_team": "Ivory Coast",  "away_team": "Ecuador",        "venue": "Philadelphia"},
    {"date": "2026-06-14", "group": "F", "home_team": "Sweden",       "away_team": "Tunisia",        "venue": "Guadalajara"},
    # ── June 15 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-15", "group": "H", "home_team": "Spain",        "away_team": "Cabo Verde",     "venue": "Atlanta"},
    {"date": "2026-06-15", "group": "G", "home_team": "Belgium",      "away_team": "Egypt",          "venue": "Seattle"},
    {"date": "2026-06-15", "group": "H", "home_team": "Saudi Arabia", "away_team": "Uruguay",        "venue": "Miami Gardens"},
    {"date": "2026-06-15", "group": "G", "home_team": "IR Iran",      "away_team": "New Zealand",    "venue": "Inglewood"},
    # ── June 16 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-16", "group": "I", "home_team": "France",       "away_team": "Senegal",        "venue": "East Rutherford"},
    {"date": "2026-06-16", "group": "I", "home_team": "Iraq",         "away_team": "Norway",         "venue": "Foxborough"},
    {"date": "2026-06-16", "group": "J", "home_team": "Argentina",    "away_team": "Algeria",        "venue": "Kansas City"},
    {"date": "2026-06-16", "group": "J", "home_team": "Austria",      "away_team": "Jordan",         "venue": "Santa Clara"},
    # ── June 17 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-17", "group": "K", "home_team": "Portugal",     "away_team": "DR Congo",       "venue": "Houston"},
    {"date": "2026-06-17", "group": "L", "home_team": "England",      "away_team": "Croatia",        "venue": "Arlington"},
    {"date": "2026-06-17", "group": "L", "home_team": "Ghana",        "away_team": "Panama",         "venue": "Toronto"},
    {"date": "2026-06-17", "group": "K", "home_team": "Uzbekistan",   "away_team": "Colombia",       "venue": "Mexico City"},
    # ── June 18 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-18", "group": "A", "home_team": "Czechia",      "away_team": "South Africa",   "venue": "Atlanta"},
    {"date": "2026-06-18", "group": "B", "home_team": "Switzerland",  "away_team": "Bosnia and Herzegovina", "venue": "Inglewood"},
    {"date": "2026-06-18", "group": "B", "home_team": "Canada",       "away_team": "Qatar",          "venue": "Vancouver"},
    {"date": "2026-06-18", "group": "A", "home_team": "Mexico",       "away_team": "South Korea",    "venue": "Zapopan"},
    # ── June 19 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-19", "group": "D", "home_team": "United States","away_team": "Australia",      "venue": "Seattle"},
    {"date": "2026-06-19", "group": "C", "home_team": "Scotland",     "away_team": "Morocco",        "venue": "Foxborough"},
    {"date": "2026-06-19", "group": "C", "home_team": "Brazil",       "away_team": "Haiti",          "venue": "Philadelphia"},
    {"date": "2026-06-19", "group": "D", "home_team": "Türkiye",      "away_team": "Paraguay",       "venue": "Santa Clara"},
    # ── June 20 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-20", "group": "F", "home_team": "Netherlands",  "away_team": "Sweden",         "venue": "Houston"},
    {"date": "2026-06-20", "group": "E", "home_team": "Germany",      "away_team": "Ivory Coast",    "venue": "Toronto"},
    {"date": "2026-06-20", "group": "E", "home_team": "Ecuador",      "away_team": "Curaçao",        "venue": "Kansas City"},
    {"date": "2026-06-20", "group": "F", "home_team": "Tunisia",      "away_team": "Japan",          "venue": "Guadalajara"},
    # ── June 21 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-21", "group": "H", "home_team": "Spain",        "away_team": "Saudi Arabia",   "venue": "Atlanta"},
    {"date": "2026-06-21", "group": "G", "home_team": "Belgium",      "away_team": "IR Iran",        "venue": "Inglewood"},
    {"date": "2026-06-21", "group": "H", "home_team": "Uruguay",      "away_team": "Cabo Verde",     "venue": "Miami Gardens"},
    {"date": "2026-06-21", "group": "G", "home_team": "New Zealand",  "away_team": "Egypt",          "venue": "Vancouver"},
    # ── June 22 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-22", "group": "J", "home_team": "Argentina",    "away_team": "Austria",        "venue": "Arlington"},
    {"date": "2026-06-22", "group": "I", "home_team": "France",       "away_team": "Iraq",           "venue": "Philadelphia"},
    {"date": "2026-06-22", "group": "I", "home_team": "Norway",       "away_team": "Senegal",        "venue": "East Rutherford"},
    {"date": "2026-06-22", "group": "J", "home_team": "Jordan",       "away_team": "Algeria",        "venue": "Santa Clara"},
    # ── June 23 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-23", "group": "K", "home_team": "Portugal",     "away_team": "Uzbekistan",     "venue": "Houston"},
    {"date": "2026-06-23", "group": "L", "home_team": "England",      "away_team": "Ghana",          "venue": "Foxborough"},
    {"date": "2026-06-23", "group": "L", "home_team": "Panama",       "away_team": "Croatia",        "venue": "Toronto"},
    {"date": "2026-06-23", "group": "K", "home_team": "Colombia",     "away_team": "DR Congo",       "venue": "Zapopan"},
    # ── June 24 (group stage deciders) ────────────────────────────────────────
    {"date": "2026-06-24", "group": "B", "home_team": "Switzerland",  "away_team": "Canada",         "venue": "Vancouver"},
    {"date": "2026-06-24", "group": "B", "home_team": "Bosnia and Herzegovina", "away_team": "Qatar", "venue": "Seattle"},
    {"date": "2026-06-24", "group": "C", "home_team": "Scotland",     "away_team": "Brazil",         "venue": "Miami Gardens"},
    {"date": "2026-06-24", "group": "C", "home_team": "Morocco",      "away_team": "Haiti",          "venue": "Atlanta"},
    {"date": "2026-06-24", "group": "A", "home_team": "Czechia",      "away_team": "Mexico",         "venue": "Mexico City"},
    {"date": "2026-06-24", "group": "A", "home_team": "South Africa", "away_team": "South Korea",    "venue": "Guadalajara"},
    # ── June 25 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-25", "group": "E", "home_team": "Ecuador",      "away_team": "Germany",        "venue": "East Rutherford"},
    {"date": "2026-06-25", "group": "E", "home_team": "Curaçao",      "away_team": "Ivory Coast",    "venue": "Philadelphia"},
    {"date": "2026-06-25", "group": "F", "home_team": "Japan",        "away_team": "Sweden",         "venue": "Arlington"},
    {"date": "2026-06-25", "group": "F", "home_team": "Tunisia",      "away_team": "Netherlands",    "venue": "Kansas City"},
    {"date": "2026-06-25", "group": "D", "home_team": "Türkiye",      "away_team": "United States",  "venue": "Inglewood"},
    {"date": "2026-06-25", "group": "D", "home_team": "Paraguay",     "away_team": "Australia",      "venue": "Santa Clara"},
    # ── June 26 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-26", "group": "I", "home_team": "Norway",       "away_team": "France",         "venue": "Foxborough"},
    {"date": "2026-06-26", "group": "I", "home_team": "Senegal",      "away_team": "Iraq",           "venue": "Toronto"},
    {"date": "2026-06-26", "group": "H", "home_team": "Cabo Verde",   "away_team": "Saudi Arabia",   "venue": "Houston"},
    {"date": "2026-06-26", "group": "H", "home_team": "Uruguay",      "away_team": "Spain",          "venue": "Zapopan"},
    {"date": "2026-06-26", "group": "G", "home_team": "Egypt",        "away_team": "IR Iran",        "venue": "Seattle"},
    {"date": "2026-06-26", "group": "G", "home_team": "New Zealand",  "away_team": "Belgium",        "venue": "Vancouver"},
    # ── June 27 ───────────────────────────────────────────────────────────────
    {"date": "2026-06-27", "group": "L", "home_team": "Panama",       "away_team": "England",        "venue": "East Rutherford"},
    {"date": "2026-06-27", "group": "L", "home_team": "Croatia",      "away_team": "Ghana",          "venue": "Philadelphia"},
    {"date": "2026-06-27", "group": "K", "home_team": "Colombia",     "away_team": "Portugal",       "venue": "Miami Gardens"},
    {"date": "2026-06-27", "group": "K", "home_team": "DR Congo",     "away_team": "Uzbekistan",     "venue": "Atlanta"},
    {"date": "2026-06-27", "group": "J", "home_team": "Algeria",      "away_team": "Austria",        "venue": "Kansas City"},
    {"date": "2026-06-27", "group": "J", "home_team": "Jordan",       "away_team": "Argentina",      "venue": "Arlington"},
]


def group_stage_fixtures() -> list[dict]:
    """Return all 72 group stage fixtures from the official schedule.

    Each dict has: date, group, home_team, away_team, venue, neutral=True.
    home_team/away_team are draw names — use normalise() to map to dataset names.
    """
    return [dict(f, neutral=True, stage="group_stage") for f in GROUP_STAGE_SCHEDULE]


def get_team_group(team: str) -> str | None:
    """Return the group letter for a team (draw name or dataset name)."""
    for group, teams in GROUPS.items():
        if team in teams or normalise(team) in [normalise(t) for t in teams]:
            return group
    return None


def get_group_opponents(team: str) -> list[str]:
    """Return the other teams in the same group (draw names)."""
    for teams in GROUPS.values():
        if team in teams:
            return [t for t in teams if t != team]
    return []


# ── Tournament metadata ────────────────────────────────────────────────────────

WC_2026_META = {
    "year": 2026,
    "hosts": ["Mexico", "Canada", "United States"],
    "n_teams": 48,
    "n_groups": 12,
    "format": "12 groups of 4; top 2 + 8 best 3rd advance to Round of 32 (32-team knockout)",
    "group_stage_start": "2026-06-11",
    "group_stage_end": "2026-06-27",
    "round_of_32_start": "2026-06-28",
    "semifinals": ["2026-07-14", "2026-07-15"],
    "final_venue": "MetLife Stadium, East Rutherford, New Jersey",
    "final_date": "2026-07-19",
    "venues": {
        "USA": ["East Rutherford/New York", "Inglewood/Los Angeles", "Arlington/Dallas",
                "Santa Clara/San Francisco", "Miami Gardens", "Atlanta", "Seattle",
                "Houston", "Philadelphia", "Kansas City", "Foxborough/Boston"],
        "Canada": ["Toronto", "Vancouver"],
        "Mexico": ["Mexico City", "Guadalajara/Zapopan", "Monterrey"],
    },
}
