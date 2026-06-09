from __future__ import annotations

import logging
import pathlib
from typing import Optional, Union

import pandas as pd

from football_predictor.constants import DEFAULT_FEATURE_MODULES, ModelType, OUTCOMES
from football_predictor.data.pipeline import build_feature_matrix, split_train_test
from football_predictor.data.sources.football_data_org import fetch_matches
from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.gradient_boost import GradientBoostModel
from football_predictor.models.calibration import CalibrationLayer
from football_predictor.evaluate.metrics import summary

logger = logging.getLogger(__name__)


def predict(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    active_modules: Optional[list[str]] = None,
    model_type: ModelType = ModelType.GRADIENT_BOOST,
) -> dict[str, float]:
    """Predict outcome probabilities for a single upcoming fixture.

    The model is trained on-the-fly from `matches` (no pre-trained weights
    shipped). For production use, train once with train_and_save() and
    reload with load_and_predict().

    Args:
        home_team:      Name of the home team (must appear in `matches`).
        away_team:      Name of the away team (must appear in `matches`).
        matches:        Historical match results DataFrame.
        active_modules: Feature modules to use. Defaults to DEFAULT_FEATURE_MODULES.
        model_type:     Which model backend to use.

    Returns:
        dict with keys "home_win", "draw", "away_win" (probabilities sum to 1).
    """
    active_modules = active_modules or DEFAULT_FEATURE_MODULES

    if model_type == ModelType.DIXON_COLES:
        dc = DixonColesModel()
        dc.fit(matches)
        return dc.predict_proba(home_team, away_team)

    X, y = build_feature_matrix(matches, active_modules)

    model = GradientBoostModel()
    model.fit(X, y)

    calibrator = CalibrationLayer()
    cal_proba = calibrator.fit_transform(model.predict_proba(X), y)

    fixture = pd.Series({"date": matches["date"].max(), "home_team": home_team, "away_team": away_team})
    from football_predictor.features import REGISTRY, FeatureModule

    feature_row: dict[str, float] = {}
    for name in active_modules:
        if name in REGISTRY:
            feature_row.update(REGISTRY[name]().transform(fixture, matches))

    X_fixture = pd.DataFrame([feature_row])[X.columns]
    raw_proba = model.predict_proba(X_fixture)
    calibrated = calibrator.transform(raw_proba)

    return calibrated.iloc[0].to_dict()


def train_and_evaluate(
    competition: str,
    train_seasons: list[str],
    test_seasons: list[str],
    api_key: Optional[str] = None,
    active_modules: Optional[list[str]] = None,
) -> dict[str, float]:
    """Full train-test cycle with evaluation metrics.

    Fetches data, builds features, trains gradient boosting + calibration,
    evaluates on held-out test seasons and returns summary metrics.

    Args:
        competition:    Competition code, e.g. "PL".
        train_seasons:  Season strings for training, e.g. ["2020", "2021", "2022"].
        test_seasons:   Season strings for evaluation, e.g. ["2023"].
        api_key:        football-data.org API key.
        active_modules: Feature modules to activate.

    Returns:
        dict with rps, log_loss, accuracy, brier scores.
    """
    active_modules = active_modules or DEFAULT_FEATURE_MODULES
    all_seasons = train_seasons + test_seasons

    logger.info("Fetching match data for %s seasons %s…", competition, all_seasons)
    matches = fetch_matches(competition, all_seasons, api_key=api_key)

    train_df, test_df = split_train_test(matches, test_seasons)

    logger.info("Building feature matrix (%d train, %d test matches)…", len(train_df), len(test_df))
    X_train, y_train = build_feature_matrix(train_df, active_modules)
    X_test, y_test = build_feature_matrix(test_df, active_modules)

    logger.info("Training gradient boost model…")
    model = GradientBoostModel()
    model.fit(X_train, y_train)

    logger.info("Calibrating…")
    calibrator = CalibrationLayer()
    calibrator.fit(model.predict_proba(X_train), y_train)

    test_proba = calibrator.transform(model.predict_proba(X_test))
    metrics = summary(test_proba, y_test)

    logger.info("Evaluation: %s", metrics)
    return metrics


def predict_and_save(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    output_path: Union[str, pathlib.Path],
    active_modules: Optional[list[str]] = None,
) -> None:
    """Predict and write probabilities to a JSON file."""
    import json

    result = predict(home_team, away_team, matches, active_modules)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"home_team": home_team, "away_team": away_team, "probabilities": result}, f, indent=2)
    logger.info("Prediction saved to %s", output_path)
