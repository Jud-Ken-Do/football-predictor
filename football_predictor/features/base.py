from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureModule(ABC):
    """Base class for all feature plug-ins.

    Every module must implement two methods:
      - fetch()     — pull raw data, return a normalised DataFrame
      - transform() — convert a single match row into a flat feature dict

    The pipeline calls fetch() once per module at pipeline build time, caches
    the result, then calls transform() per match when assembling feature vectors.
    Adding a new signal means subclassing FeatureModule and registering it in
    features/__init__.py — the core model never needs to change.
    """

    name: str  # subclasses must set this as a class attribute

    @abstractmethod
    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        """Pull and normalise raw data for the given competition and seasons.

        Args:
            competition: Competition code, e.g. "PL", "BL1". See constants.SUPPORTED_COMPETITIONS.
            seasons:     List of season strings, e.g. ["2022", "2023"].

        Returns:
            A DataFrame where each row is one team-match observation with at
            minimum the columns: ["date", "home_team", "away_team"].
        """
        ...

    @abstractmethod
    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        """Compute feature values for a single match.

        Args:
            match: A Series with at minimum "date", "home_team", "away_team".
            data:  The full DataFrame returned by fetch(), used for lookups.

        Returns:
            A flat dict mapping feature name → float value.
            Feature names should be prefixed with the module name, e.g.
            "form_home_pts_last5", "elo_home_rating".
        """
        ...

    def feature_names(self) -> list[str]:
        """Return the list of feature keys this module produces.

        Override this for documentation and schema validation. Default returns
        an empty list (computed lazily from the first transform() call instead).
        """
        return []
