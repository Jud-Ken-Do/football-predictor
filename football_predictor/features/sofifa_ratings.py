"""EA FC 26 squad quality features from SoFIFA player ratings.

Uses the top-26 players by overall rating per national team as a proxy
for the actual WC 2026 squad. Covers all 48 WC 2026 teams.
Returns zeros for teams not in the FC 26 database (none expected).

CSV file: FC26_20250921.csv in the project root.
"""
from __future__ import annotations

import os

import pandas as pd

from football_predictor.features.base import FeatureModule

_CSV_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "FC26_20250921.csv"
)

# Map our canonical WC team names → nationality_name in FC26 dataset
_FC26_NAME_MAP: dict[str, str] = {
    "DR Congo": "Congo DR",
    "IR Iran": "Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "United States": "United States",   # matches as-is
    "Czechia": "Czechia",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",  # matches as-is
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Côte d'Ivoire",
    "Cabo Verde": "Cabo Verde",         # matches as-is
}

_ATTR_COLS = ["overall", "potential", "pace", "shooting", "passing",
              "dribbling", "defending", "physic", "international_reputation"]

_POS_MAP = {
    "gk":  lambda pos: "GK" in pos,
    "def": lambda pos: any(p in pos for p in ["CB", "LB", "RB", "LWB", "RWB", "DF"]),
    "mid": lambda pos: any(p in pos for p in ["CM", "CDM", "CAM", "LM", "RM", "DM", "AM", "MF"]),
    "fwd": lambda pos: any(p in pos for p in ["ST", "LW", "RW", "CF", "LF", "RF", "FW"]),
}

_SQUAD_SIZE = 26


def _load_fc26() -> dict[str, dict[str, float]]:
    path = os.path.abspath(_CSV_FILE)
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, low_memory=False)

    # Keep only outfield attrs that are numeric
    for col in _ATTR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result: dict[str, dict[str, float]] = {}

    for nat_name, group in df.groupby("nationality_name"):
        # Top-26 by overall rating — best proxy for actual squad selection
        squad = group.nlargest(_SQUAD_SIZE, "overall")

        feats: dict[str, float] = {}
        for col in _ATTR_COLS:
            feats[f"fc26_{col}"] = float(squad[col].mean(skipna=True))

        # Position-specific averages
        positions = squad["player_positions"].fillna("")
        for pos_key, pred in _POS_MAP.items():
            sub = squad[positions.apply(pred)]
            feats[f"fc26_{pos_key}_overall"] = float(sub["overall"].mean()) if len(sub) > 0 else feats["fc26_overall"]

        result[str(nat_name)] = feats

    return result


def _zero() -> dict[str, float]:
    feats: dict[str, float] = {f"fc26_{c}": 0.0 for c in _ATTR_COLS}
    for pos_key in _POS_MAP:
        feats[f"fc26_{pos_key}_overall"] = 0.0
    return feats


class SoFIFARatingsFeatures(FeatureModule):
    """EA FC 26 per-team quality features (top-26 overall, pace, shooting, etc.)."""

    name = "sofifa_ratings"

    def __init__(self) -> None:
        self._db = _load_fc26()

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def _lookup(self, team: str) -> dict[str, float]:
        fc26_name = _FC26_NAME_MAP.get(team, team)
        return self._db.get(fc26_name, _zero())

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        h = self._lookup(match["home_team"])
        a = self._lookup(match["away_team"])

        out: dict[str, float] = {}
        for k in h:
            out[f"sofifa_home_{k}"] = h[k]
            out[f"sofifa_away_{k}"] = a[k]

        # Key diffs — most predictive for the model
        for attr in ["overall", "pace", "shooting", "passing", "dribbling", "defending", "physic"]:
            out[f"sofifa_diff_{attr}"] = h.get(f"fc26_{attr}", 0.0) - a.get(f"fc26_{attr}", 0.0)

        return out

    def feature_names(self) -> list[str]:
        sample = list(_zero().keys())
        names = []
        for side in ("home", "away"):
            for k in sample:
                names.append(f"sofifa_{side}_{k}")
        for attr in ["overall", "pace", "shooting", "passing", "dribbling", "defending", "physic"]:
            names.append(f"sofifa_diff_{attr}")
        return names
