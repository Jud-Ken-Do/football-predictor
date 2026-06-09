from __future__ import annotations

import pandas as pd

from football_predictor.features.base import FeatureModule


STAGE_ENCODING = {
    "group_stage": 0,
    "round_of_32": 1,       # new in WC 2026 (48-team format)
    "round_of_16": 2,
    "quarter_final": 3,
    "semi_final": 4,
    "third_place": 5,
    "final": 6,
}

# Stakes multiplier — how much "must-win" pressure is on each team per stage.
# Research shows teams in elimination games adopt more conservative tactics,
# reducing total goals and increasing draw frequency in normal time.
STAGE_PRESSURE = {
    "group_stage": 0.5,     # can still progress after a loss
    "round_of_32": 1.0,
    "round_of_16": 1.0,
    "quarter_final": 1.2,
    "semi_final": 1.4,
    "third_place": 0.8,     # psychologically deflated after semi loss
    "final": 1.5,
}


class TournamentStageFeatures(FeatureModule):
    """Tournament stage and knockout pressure features.

    At the World Cup, the stage of the competition materially changes
    team behaviour: group-stage teams may rotate squads, protect leads,
    or tactically manage results (e.g. both teams happy to draw in game 3).
    Knockout teams adopt lower-risk strategies, reducing expected goals.

    Usage: pass `stage` as a key in the match Series when predicting.
    If not present, defaults to group_stage.

    Features produced:
    - stage_encoding        (int 0–6)
    - stage_pressure        (float 0.5–1.5)
    - is_knockout           (0 or 1)
    - is_final              (0 or 1)
    - home_must_win         (0 or 1 — set externally if known)
    - away_must_win         (0 or 1 — set externally if known)
    """

    name = "tournament_stage"

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        stage = str(match.get("stage", "group_stage")).lower().replace(" ", "_").replace("-", "_")
        stage = stage if stage in STAGE_ENCODING else "group_stage"

        encoding = STAGE_ENCODING[stage]
        pressure = STAGE_PRESSURE[stage]
        is_knockout = float(encoding >= STAGE_ENCODING["round_of_32"])
        is_final = float(stage == "final")

        return {
            "stage_encoding": float(encoding),
            "stage_pressure": pressure,
            "is_knockout": is_knockout,
            "is_final": is_final,
            "home_must_win": float(match.get("home_must_win", 0)),
            "away_must_win": float(match.get("away_must_win", 0)),
        }

    def feature_names(self) -> list[str]:
        return [
            "stage_encoding",
            "stage_pressure",
            "is_knockout",
            "is_final",
            "home_must_win",
            "away_must_win",
        ]
