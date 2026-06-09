from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from football_predictor.constants import DEFAULT_FEATURE_MODULES, OUTCOMES
from football_predictor.features import REGISTRY, FeatureModule

logger = logging.getLogger(__name__)


def build_feature_matrix(
    matches: pd.DataFrame,
    active_modules: Optional[list[str]] = None,
    context: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble a feature matrix from all active feature modules.

    Args:
        matches:        DataFrame of matches to compute features FOR.
        active_modules: List of module names to use. Defaults to
                        constants.DEFAULT_FEATURE_MODULES.
        context:        Full historical DataFrame for feature lookups.
                        Pass all_data here when computing test features so
                        modules see the complete match history, not just the
                        test slice. Defaults to matches if not provided.

    Returns:
        X: Feature DataFrame, one row per match.
        y: Integer series — 0 = home win, 1 = draw, 2 = away win.
    """
    active_modules = active_modules or DEFAULT_FEATURE_MODULES
    hist = context if context is not None else matches

    modules: list[FeatureModule] = []
    for name in active_modules:
        if name not in REGISTRY:
            logger.warning("Feature module '%s' not found in registry — skipping.", name)
            continue
        modules.append(REGISTRY[name]())

    rows = []
    for _, match in matches.iterrows():
        feature_row: dict[str, float] = {}
        for module in modules:
            try:
                feature_row.update(module.transform(match, hist))
            except Exception as exc:
                logger.warning("Module %s failed on match %s: %s", module.name, match.get("match_id"), exc)
        rows.append(feature_row)

    X = pd.DataFrame(rows)

    hg = matches["home_goals"].values
    ag = matches["away_goals"].values
    y = pd.Series(
        np.where(hg > ag, 0, np.where(hg == ag, 1, 2)),
        name="outcome",
    )

    return X, y


def split_train_test(
    matches: pd.DataFrame,
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split matches into train / test by season.

    Args:
        matches:      Full match DataFrame with a "season" column.
        test_seasons: Season strings reserved for testing (e.g. ["2023"]).

    Returns:
        train_df, test_df
    """
    test_mask = matches["season"].isin(test_seasons)
    return matches[~test_mask].copy(), matches[test_mask].copy()
