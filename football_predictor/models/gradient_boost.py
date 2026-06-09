from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class GradientBoostModel:
    """XGBoost classifier for 3-class match outcome prediction.

    Outputs calibrated probabilities for [home_win, draw, away_win].
    Wrap with CalibrationLayer for production use — raw XGBoost
    probabilities are not well-calibrated on imbalanced football data.

    Swap the backend by replacing XGBClassifier with LGBMClassifier or
    CatBoostClassifier — the rest of the class is backend-agnostic.
    """

    OUTCOME_LABELS = ["home_win", "draw", "away_win"]

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=random_state,
            verbosity=0,
        )
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> "GradientBoostModel":
        self._feature_names = list(X.columns)
        w = getattr(sample_weight, "values", sample_weight) if sample_weight is not None else None
        self.model.fit(X.values, y.values, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns home_win, draw, away_win."""
        proba = self.model.predict_proba(X.values)
        return pd.DataFrame(proba, columns=self.OUTCOME_LABELS, index=X.index)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict[str, float]:
        """Quick cross-validation check during development.

        Returns mean and std of negative log-loss across folds.
        """
        scores = cross_val_score(
            self.model, X.values, y.values, cv=cv, scoring="neg_log_loss"
        )
        return {"mean_neg_log_loss": float(scores.mean()), "std": float(scores.std())}

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def save(self, path: pathlib.Path) -> None:
        self.model.save_model(str(path))
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: pathlib.Path) -> "GradientBoostModel":
        instance = cls()
        instance.model.load_model(str(path))
        return instance
