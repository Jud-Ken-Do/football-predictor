from __future__ import annotations

import pandas as pd

from football_predictor.features.base import FeatureModule
from football_predictor.constants import CONFEDERATION_STRENGTH

# Team → confederation mapping.
# Based on FIFA official confederation membership.
# Expand as needed — unknown teams fall back to "UNKNOWN" (strength 0).
TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA
    "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "Italy": "UEFA",
    "England": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Denmark": "UEFA", "Sweden": "UEFA",
    "Poland": "UEFA", "Serbia": "UEFA", "Ukraine": "UEFA", "Austria": "UEFA",
    "Czech Republic": "UEFA", "Czechia": "UEFA", "Hungary": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Scotland": "UEFA", "Wales": "UEFA", "Turkey": "UEFA", "Türkiye": "UEFA", "Greece": "UEFA",
    "Romania": "UEFA", "Norway": "UEFA", "Finland": "UEFA", "Albania": "UEFA",
    "Georgia": "UEFA", "Montenegro": "UEFA", "North Macedonia": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bosnia-Herzegovina": "UEFA", "Kosovo": "UEFA", "Iceland": "UEFA",
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF", "Ivory Coast": "CAF", "Côte d'Ivoire": "CAF",
    "Algeria": "CAF", "Mali": "CAF", "South Africa": "CAF", "DR Congo": "CAF", "Congo DR": "CAF",
    "Guinea": "CAF", "Zambia": "CAF", "Burkina Faso": "CAF", "Cape Verde": "CAF",
    "Cabo Verde": "CAF",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Peru": "CONMEBOL", "Venezuela": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Cuba": "CONCACAF", "Haiti": "CONCACAF", "Curaçao": "CONCACAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Korea Republic": "AFC", "Iran": "AFC", "IR Iran": "AFC",
    "Saudi Arabia": "AFC", "Australia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "United Arab Emirates": "AFC", "Oman": "AFC",
    "China": "AFC", "Indonesia": "AFC", "Vietnam": "AFC", "Thailand": "AFC",
    "Bahrain": "AFC", "Palestine": "AFC", "Kuwait": "AFC",
    # OFC
    "New Zealand": "OFC",
}


class ConfederationFeatures(FeatureModule):
    """Confederation identity and relative strength features.

    UEFA and CONMEBOL teams historically overperform their Elo ratings
    at World Cups, while AFC and OFC teams underperform. Encoding
    confederation lets the model learn these structural biases.

    Research note: Groll et al. (2024) include confederation as a fixed
    effect in their UEFA EURO tournament prediction model.

    Features produced:
    - home_confederation_strength   (from constants.CONFEDERATION_STRENGTH)
    - away_confederation_strength
    - confederation_strength_diff
    - same_confederation            (1 if both teams from same conf)
    - home_is_uefa, home_is_conmebol, home_is_concacaf, home_is_afc, home_is_caf
    - away_is_uefa, away_is_conmebol, away_is_concacaf, away_is_afc, away_is_caf
    """

    name = "confederation"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        home_conf = self._get_conf(match["home_team"])
        away_conf = self._get_conf(match["away_team"])

        h_strength = float(CONFEDERATION_STRENGTH.get(home_conf, 0))
        a_strength = float(CONFEDERATION_STRENGTH.get(away_conf, 0))

        return {
            "home_confederation_strength": h_strength,
            "away_confederation_strength": a_strength,
            "confederation_strength_diff": h_strength - a_strength,
            "same_confederation": float(home_conf == away_conf),
            "home_is_uefa": float(home_conf == "UEFA"),
            "home_is_conmebol": float(home_conf == "CONMEBOL"),
            "home_is_concacaf": float(home_conf == "CONCACAF"),
            "home_is_afc": float(home_conf == "AFC"),
            "home_is_caf": float(home_conf == "CAF"),
            "away_is_uefa": float(away_conf == "UEFA"),
            "away_is_conmebol": float(away_conf == "CONMEBOL"),
            "away_is_concacaf": float(away_conf == "CONCACAF"),
            "away_is_afc": float(away_conf == "AFC"),
            "away_is_caf": float(away_conf == "CAF"),
        }

    @staticmethod
    def _get_conf(team: str) -> str:
        return TEAM_CONFEDERATION.get(team, "UNKNOWN")

    def feature_names(self) -> list[str]:
        return [
            "home_confederation_strength", "away_confederation_strength",
            "confederation_strength_diff", "same_confederation",
            "home_is_uefa", "home_is_conmebol", "home_is_concacaf", "home_is_afc", "home_is_caf",
            "away_is_uefa", "away_is_conmebol", "away_is_concacaf", "away_is_afc", "away_is_caf",
        ]
