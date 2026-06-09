"""Venue-specific features for WC 2026.

Three effects modelled:
  1. Partial home advantage — Mexico / USA / Canada in their own country's stadiums
  2. High-altitude penalty — Mexico City (2240 m), Zapopan/Guadalajara (1567 m),
     Monterrey (530 m) vs sea-level venues.  Altitude significantly increases
     oxygen demand; teams from low-altitude countries lose ≈4–6% aerobic capacity
     above 1500 m (Bohner et al. 2015, Sport. Med.).
  3. Travel burden — confederation distance from the host region.
     UEFA/CAF/AFC teams from the other side of the globe face greater fatigue.

Features produced (8 total)
────────────────────────────
  venue_home_partial_adv   — +0.30 if home team is playing in their own country
  venue_away_partial_adv   — +0.30 if away team is playing in their own country
  venue_home_altitude_adv  — advantage for altitude-acclimatised home team
  venue_away_altitude_adv  — advantage for altitude-acclimatised away team
  venue_altitude_m         — venue altitude in metres (normalised /2500)
  venue_high_altitude      — binary: 1 if altitude > 1400 m
  venue_home_travel_burden — travel burden for home team (0=local, 1=inter-continental)
  venue_away_travel_burden — travel burden for away team
"""
from __future__ import annotations

import pandas as pd

from football_predictor.features.base import FeatureModule

# ── Venue data ────────────────────────────────────────────────────────────────

_VENUES: dict[str, dict] = {
    # Mexico
    "Mexico City":       {"country": "Mexico",  "altitude_m": 2240},
    "Zapopan":           {"country": "Mexico",  "altitude_m": 1567},
    "Guadalajara":       {"country": "Mexico",  "altitude_m": 1567},
    "Monterrey":         {"country": "Mexico",  "altitude_m": 530},
    # USA — canonical names
    "Dallas":            {"country": "USA",     "altitude_m": 185},
    "Los Angeles":       {"country": "USA",     "altitude_m": 71},
    "New York/New Jersey": {"country": "USA",   "altitude_m": 8},
    "New York":          {"country": "USA",     "altitude_m": 8},
    "New Jersey":        {"country": "USA",     "altitude_m": 8},
    "San Francisco Bay Area": {"country": "USA","altitude_m": 29},
    "San Francisco":     {"country": "USA",     "altitude_m": 29},
    "Miami":             {"country": "USA",     "altitude_m": 4},
    "Seattle":           {"country": "USA",     "altitude_m": 6},
    "Boston":            {"country": "USA",     "altitude_m": 5},
    "Atlanta":           {"country": "USA",     "altitude_m": 309},
    "Kansas City":       {"country": "USA",     "altitude_m": 271},
    "Philadelphia":      {"country": "USA",     "altitude_m": 12},
    "Houston":           {"country": "USA",     "altitude_m": 15},
    # USA — stadium/suburb aliases used in GROUP_STAGE_SCHEDULE
    "Inglewood":         {"country": "USA",     "altitude_m": 71},   # SoFi Stadium, LA
    "East Rutherford":   {"country": "USA",     "altitude_m": 8},    # MetLife Stadium, NY/NJ
    "Santa Clara":       {"country": "USA",     "altitude_m": 29},   # Levi's Stadium, SF Bay
    "Foxborough":        {"country": "USA",     "altitude_m": 5},    # Gillette Stadium, Boston
    "Arlington":         {"country": "USA",     "altitude_m": 185},  # AT&T Stadium, Dallas
    "Miami Gardens":     {"country": "USA",     "altitude_m": 4},    # Hard Rock Stadium
    # Canada
    "Toronto":           {"country": "Canada",  "altitude_m": 76},
    "Vancouver":         {"country": "Canada",  "altitude_m": 3},
}

# Confederation → home-country list (teams that get partial home advantage)
_COUNTRY_TEAMS: dict[str, list[str]] = {
    "Mexico": ["Mexico"],
    "USA":    ["United States", "USA"],
    "Canada": ["Canada"],
}

# ── Confederation travel distance from North America ─────────────────────────
# 0.0 = no travel burden (CONCACAF);  1.0 = intercontinental max
_TRAVEL_BURDEN: dict[str, float] = {
    "CONCACAF": 0.0,
    "CONMEBOL": 0.25,
    "UEFA":     0.75,
    "CAF":      0.80,
    "AFC":      0.90,
    "OFC":      1.00,
}

# Team → confederation (WC 2026 participants)
_TEAM_CONF: dict[str, str] = {
    # CONCACAF
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    "El Salvador": "CONCACAF", "Costa Rica": "CONCACAF", "Haiti": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "Curaçao": "CONCACAF",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Chile": "CONMEBOL",
    "Peru": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # UEFA
    "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "Portugal": "UEFA",
    "Switzerland": "UEFA", "Croatia": "UEFA", "Austria": "UEFA",
    "Sweden": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Turkey": "UEFA", "Türkiye": "UEFA", "Czech Republic": "UEFA",
    "Czechia": "UEFA", "Ukraine": "UEFA", "Romania": "UEFA",
    "Hungary": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Albania": "UEFA", "Denmark": "UEFA", "Norway": "UEFA",
    "Finland": "UEFA", "Wales": "UEFA", "Iceland": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bosnia-Herzegovina": "UEFA",
    "North Macedonia": "UEFA", "Montenegro": "UEFA", "Kosovo": "UEFA",
    "Georgia": "UEFA", "Poland": "UEFA", "Greece": "UEFA",
    "Italy": "UEFA", "Netherlands": "UEFA",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF", "Algeria": "CAF",
    "Tunisia": "CAF", "DR Congo": "CAF", "Mali": "CAF", "South Africa": "CAF",
    "Cabo Verde": "CAF", "Tanzania": "CAF", "Zambia": "CAF", "Uganda": "CAF",
    "Burkina Faso": "CAF", "Zimbabwe": "CAF", "Mozambique": "CAF",
    "Comoros": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Korea Republic": "AFC",
    "Australia": "AFC", "Iran": "AFC", "IR Iran": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "UAE": "AFC",
    "Uzbekistan": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "China": "AFC", "Indonesia": "AFC", "Thailand": "AFC",
    "Bahrain": "AFC", "Oman": "AFC",
    # OFC
    "New Zealand": "OFC",
}

# High-altitude acclimatisation advantage threshold (m)
_ALTITUDE_HIGH_M: float = 1400.0


def _partial_home_adv(team: str, venue_country: str) -> float:
    for country, teams in _COUNTRY_TEAMS.items():
        if team in teams and country == venue_country:
            return 0.30
    return 0.0


def _altitude_adv(team: str, altitude_m: float) -> float:
    """Partial advantage for teams from high-altitude countries."""
    if altitude_m < _ALTITUDE_HIGH_M:
        return 0.0
    conf = _TEAM_CONF.get(team, "UEFA")
    # Bolivia (CONMEBOL), Mexico (CONCACAF) play at altitude regularly
    high_alt_teams = {"Bolivia", "Peru", "Colombia", "Ecuador", "Mexico"}
    if team in high_alt_teams:
        return 0.20 * (altitude_m / 2500.0)  # normalised boost
    return 0.0


def _travel(team: str) -> float:
    conf = _TEAM_CONF.get(team, "UEFA")
    return _TRAVEL_BURDEN.get(conf, 0.75)


def _lookup_venue(venue_str: str) -> dict:
    for key, data in _VENUES.items():
        if key.lower() in venue_str.lower() or venue_str.lower() in key.lower():
            return data
    return {"country": "USA", "altitude_m": 50}  # default: neutral US city


class VenueWC2026Features(FeatureModule):
    """WC 2026 venue effects: partial home advantage, altitude, travel."""

    name = "venue_wc2026"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        venue_str = str(match.get("venue", ""))

        v = _lookup_venue(venue_str)
        alt = float(v["altitude_m"])
        vc = v["country"]

        h_home_adv = _partial_home_adv(home, vc)
        a_home_adv = _partial_home_adv(away, vc)
        h_alt = _altitude_adv(home, alt)
        a_alt = _altitude_adv(away, alt)

        return {
            "venue_home_partial_adv":  h_home_adv,
            "venue_away_partial_adv":  a_home_adv,
            "venue_home_altitude_adv": h_alt,
            "venue_away_altitude_adv": a_alt,
            "venue_altitude_m":        alt / 2500.0,    # normalised
            "venue_high_altitude":     1.0 if alt > _ALTITUDE_HIGH_M else 0.0,
            "venue_home_travel_burden": _travel(home),
            "venue_away_travel_burden": _travel(away),
        }

    def feature_names(self) -> list[str]:
        return [
            "venue_home_partial_adv",
            "venue_away_partial_adv",
            "venue_home_altitude_adv",
            "venue_away_altitude_adv",
            "venue_altitude_m",
            "venue_high_altitude",
            "venue_home_travel_burden",
            "venue_away_travel_burden",
        ]
