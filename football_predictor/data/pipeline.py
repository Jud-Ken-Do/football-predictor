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


def prune_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove near-duplicate features with |Pearson r| > threshold.

    For each correlated pair, the feature with lower variance is dropped —
    higher variance implies more discriminative power. This reduces feature
    count for XGBoost (faster training, cleaner SHAP) without information loss
    because the kept feature encodes the same signal.

    Returns (pruned_X, list_of_dropped_column_names).
    """
    if X.shape[1] < 2:
        return X, []

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    variances = X.var()

    to_drop: set[str] = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > threshold].tolist()
        for other in correlated:
            if other in to_drop:
                continue
            if variances.get(col, 0.0) >= variances.get(other, 0.0):
                to_drop.add(other)
            else:
                to_drop.add(col)
                break  # col itself is being dropped; stop comparing it

    dropped = sorted(to_drop)
    if dropped:
        logger.info("prune_correlated_features: dropped %d / %d  (threshold=%.2f)",
                    len(dropped), X.shape[1], threshold)
    return X.drop(columns=dropped), dropped


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
