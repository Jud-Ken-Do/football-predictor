from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TUNED_PARAMS_PATH = _ROOT / "data" / "xgb_tuned_params.json"

_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

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

    def __init__(self, random_state: int = 42):
        params = _DEFAULT_PARAMS.copy()
        if _TUNED_PARAMS_PATH.exists():
            try:
                params = json.loads(_TUNED_PARAMS_PATH.read_text())
                print(f"  [XGB] Loaded tuned params from {_TUNED_PARAMS_PATH.name}")
            except Exception:
                pass
        self.model = XGBClassifier(
            **params,
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
        if self._feature_names and list(X.columns) != self._feature_names:
            X = X.reindex(columns=self._feature_names, fill_value=0.0)
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

    def tune_hyperparameters(
        self,
        val_splits: list[tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series]],
        n_trials: int = 60,
        random_state: int = 42,
    ) -> dict:
        """Bayesian hyperparameter optimisation using Optuna (time-series CV splits).

        Each split is a tuple (X_train, y_train, sample_weight, X_val, y_val) produced
        by a time-ordered train/val cut — NOT random k-fold, which would leak future data.

        After tuning, rebuilds self.model with the best parameters so that the next
        call to fit() uses them automatically.

        Requires: pip install optuna

        Example usage in pipeline:
            splits = [
                (X_pre2018, y_pre2018, sw_pre2018, X_wc2018, y_wc2018),
                (X_pre2022, y_pre2022, sw_pre2022, X_wc2022, y_wc2022),
            ]
            xgb.tune_hyperparameters(splits, n_trials=80)
            xgb.fit(X_full, y_full, sample_weight=sw_full)
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed — skipping tuning. Run: pip install optuna")
            return {}

        def _ll(proba: np.ndarray, y: np.ndarray) -> float:
            return float(-np.mean(np.log(np.maximum(proba[np.arange(len(y)), y], 1e-10))))

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 900),
                "max_depth":         trial.suggest_int("max_depth", 3, 6),
                "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 1, 15),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
                "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            }
            fold_losses = []
            for X_tr, y_tr, sw_tr, X_val, y_val in val_splits:
                m = XGBClassifier(
                    **params,
                    objective="multi:softprob", num_class=3,
                    eval_metric="mlogloss", use_label_encoder=False,
                    random_state=random_state, verbosity=0,
                )
                m.fit(X_tr.values, y_tr.values, sample_weight=sw_tr)
                fold_losses.append(_ll(m.predict_proba(X_val.values), y_val.values))
            return float(np.mean(fold_losses))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info("Tuning complete. Best params: %s  log-loss=%.4f", best, study.best_value)
        print(f"  [XGB tuning] best log-loss={study.best_value:.4f}  params={best}")
        _TUNED_PARAMS_PATH.write_text(json.dumps(best, indent=2))
        print(f"  [XGB tuning] params saved → {_TUNED_PARAMS_PATH.name}")

        self.model = XGBClassifier(
            **best,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=random_state, verbosity=0,
        )
        return best

    def save(self, path: pathlib.Path) -> None:
        self.model.save_model(str(path))
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: pathlib.Path) -> "GradientBoostModel":
        instance = cls()
        instance.model.load_model(str(path))
        return instance
