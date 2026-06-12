"""WC 2026 squad features from the official FIFA squad list (markdown format).

Club data is only present for ~12 teams in the source; age is available for ~38.
Returns zeros for teams not in the WC 2026 field (historical training rows).
"""
from __future__ import annotations

import os
import re
from datetime import date

import pandas as pd

from football_predictor.features.base import FeatureModule

_LEAGUE_TIER: dict[str, float] = {
    "ENG": 1.00, "ESP": 1.00, "GER": 1.00, "ITA": 1.00, "FRA": 1.00,
    "NED": 0.75, "POR": 0.75, "BEL": 0.65, "TUR": 0.65, "SCO": 0.60,
    "DEN": 0.60, "SUI": 0.60, "GRE": 0.55, "NOR": 0.55, "RUS": 0.55,
    "CZE": 0.55, "CRO": 0.50, "POL": 0.50, "WAL": 0.45, "ARM": 0.30,
    "CYP": 0.40, "HUN": 0.40,
    "ARG": 0.65, "BRA": 0.60, "MEX": 0.50, "KSA": 0.45, "USA": 0.40,
    "EGY": 0.30, "TUN": 0.30, "QAT": 0.30, "ALG": 0.25, "RSA": 0.25,
    "IRQ": 0.25, "UAE": 0.30, "AUS": 0.25, "UZB": 0.20, "NZL": 0.15,
    "THA": 0.20, "IDN": 0.15,
}
_TOP5 = {"ENG", "ESP", "GER", "ITA", "FRA"}
_EUROPEAN = {"ENG", "ESP", "GER", "ITA", "FRA", "NED", "POR", "BEL", "TUR",
             "SCO", "DEN", "SUI", "GRE", "NOR", "RUS", "CZE", "CRO", "POL",
             "WAL", "ARM", "CYP", "HUN"}

_TOURNAMENT_START = date(2026, 6, 11)

_SQUADS_MD = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "world_cup_2026_squads_fifa_2026-06-08.md"
)

_TO_EXCEL: dict[str, str] = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte D'Ivoire",
    "Côte d'Ivoire": "Côte D'Ivoire",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia And Herzegovina",
    "Bosnia-Herzegovina": "Bosnia And Herzegovina",
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Cape Verde": "Cabo Verde",
}

_DOB_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})$")
_CLUB_RE = re.compile(r"\(([A-Z]{3})\)\s*$")
_PLAYER_RE = re.compile(r"^\d{2}\.\s+(GK|DF|MF|FW)\s+-\s+(.+?)\s+-\s*(.*)$")
_SECTION_RE = re.compile(r"^## (.+?)\s*\(([A-Z]{3})\)")


def _parse_dob(s: str) -> date | None:
    m = _DOB_RE.match(s.strip())
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def _parse_squads(md_path: str) -> dict[str, dict[str, float]]:
    if not os.path.exists(md_path):
        return {}

    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    result: dict[str, dict[str, float]] = {}
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    for section in sections:
        header = _SECTION_RE.match(section.lstrip())
        if not header:
            continue
        team_name = header.group(1).strip()

        ages: list[float] = []
        tiers: list[float] = []
        pct_top5_count = 0
        pct_eu_count = 0
        pos_counts: dict[str, int] = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
        total_players = 0

        for line in section.splitlines():
            m = _PLAYER_RE.match(line.strip())
            if not m:
                continue
            pos, _name, info = m.group(1), m.group(2), m.group(3).strip()
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            total_players += 1

            # Try club (ends with (XXX))
            club_m = _CLUB_RE.search(info)
            if club_m:
                code = club_m.group(1)
                tiers.append(_LEAGUE_TIER.get(code, 0.15))
                if code in _TOP5:
                    pct_top5_count += 1
                if code in _EUROPEAN:
                    pct_eu_count += 1
            # Try DOB
            dob = _parse_dob(info)
            if dob:
                ages.append((_TOURNAMENT_START - dob).days / 365.25)

        if total_players == 0:
            continue

        result[team_name] = {
            "avg_age": float(sum(ages) / len(ages)) if ages else 0.0,
            "has_age_data": 1.0 if ages else 0.0,
            "avg_club_tier": float(sum(tiers) / len(tiers)) if tiers else 0.0,
            "pct_top5": pct_top5_count / total_players if tiers else 0.0,
            "pct_european": pct_eu_count / total_players if tiers else 0.0,
            "has_club_data": 1.0 if tiers else 0.0,
            "num_gk": float(pos_counts.get("GK", 0)),
            "num_df": float(pos_counts.get("DF", 0)),
            "num_mf": float(pos_counts.get("MF", 0)),
            "num_fw": float(pos_counts.get("FW", 0)),
        }

    return result


_FEATURE_KEYS = [
    "avg_age", "has_age_data", "avg_club_tier", "pct_top5",
    "pct_european", "has_club_data", "num_gk", "num_df", "num_mf", "num_fw",
]
_ZERO: dict[str, float] = {k: 0.0 for k in _FEATURE_KEYS}


class SquadWC2026Features(FeatureModule):
    """WC 2026 squad features: age (38 teams) + club tier (12 teams) from FIFA list."""

    name = "squad_wc2026"

    def __init__(self) -> None:
        self._squads = _parse_squads(os.path.abspath(_SQUADS_MD))

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def _lookup(self, team: str) -> dict[str, float]:
        name = _TO_EXCEL.get(team, team)
        return self._squads.get(name, _ZERO)

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        h = self._lookup(match["home_team"])
        a = self._lookup(match["away_team"])

        out: dict[str, float] = {}
        for k in _FEATURE_KEYS:
            out[f"squad26_home_{k}"] = h[k]
            out[f"squad26_away_{k}"] = a[k]
        out["squad26_diff_avg_age"] = h["avg_age"] - a["avg_age"]
        out["squad26_diff_avg_club_tier"] = h["avg_club_tier"] - a["avg_club_tier"]
        out["squad26_diff_pct_top5"] = h["pct_top5"] - a["pct_top5"]
        out["squad26_diff_pct_european"] = h["pct_european"] - a["pct_european"]
        return out

    def feature_names(self) -> list[str]:
        names = []
        for side in ("home", "away"):
            for k in _FEATURE_KEYS:
                names.append(f"squad26_{side}_{k}")
        names += [
            "squad26_diff_avg_age", "squad26_diff_avg_club_tier",
            "squad26_diff_pct_top5", "squad26_diff_pct_european",
        ]
        return names
