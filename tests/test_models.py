import numpy as np
import pandas as pd
import pytest

from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.calibration import CalibrationLayer
from football_predictor.evaluate.metrics import ranked_probability_score, accuracy, summary


MATCHES = pd.DataFrame([
    {"date": "2023-08-01", "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
    {"date": "2023-08-08", "home_team": "B", "away_team": "C", "home_goals": 0, "away_goals": 0},
    {"date": "2023-08-15", "home_team": "C", "away_team": "A", "home_goals": 1, "away_goals": 3},
    {"date": "2023-08-22", "home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 1},
    {"date": "2023-08-29", "home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 2},
    {"date": "2023-09-05", "home_team": "C", "away_team": "B", "home_goals": 2, "away_goals": 0},
])


def test_dixon_coles_probabilities_sum_to_one():
    dc = DixonColesModel()
    dc.fit(MATCHES)
    result = dc.predict_proba("A", "B")
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_dixon_coles_keys():
    dc = DixonColesModel()
    dc.fit(MATCHES)
    result = dc.predict_proba("A", "B")
    assert set(result.keys()) == {"home_win", "draw", "away_win"}


def test_dixon_coles_all_probabilities_non_negative():
    dc = DixonColesModel()
    dc.fit(MATCHES)
    result = dc.predict_proba("A", "B")
    assert all(v >= 0 for v in result.values())


def test_calibration_preserves_sum():
    proba = pd.DataFrame([
        {"home_win": 0.5, "draw": 0.3, "away_win": 0.2},
        {"home_win": 0.2, "draw": 0.3, "away_win": 0.5},
        {"home_win": 0.4, "draw": 0.2, "away_win": 0.4},
    ])
    y = pd.Series([0, 2, 1])
    cal = CalibrationLayer()
    cal.fit(proba, y)
    calibrated = cal.transform(proba)
    for _, row in calibrated.iterrows():
        assert abs(row.sum() - 1.0) < 1e-6


def test_rps_perfect_prediction():
    proba = pd.DataFrame([{"home_win": 1.0, "draw": 0.0, "away_win": 0.0}])
    y = pd.Series([0])
    assert ranked_probability_score(proba, y) == pytest.approx(0.0)


def test_rps_worst_prediction():
    proba = pd.DataFrame([{"home_win": 1.0, "draw": 0.0, "away_win": 0.0}])
    y = pd.Series([2])
    assert ranked_probability_score(proba, y) == pytest.approx(1.0)


def test_summary_returns_all_keys():
    proba = pd.DataFrame([
        {"home_win": 0.5, "draw": 0.3, "away_win": 0.2},
        {"home_win": 0.2, "draw": 0.4, "away_win": 0.4},
    ])
    y = pd.Series([0, 1])
    result = summary(proba, y)
    assert "rps" in result
    assert "log_loss" in result
    assert "accuracy" in result
