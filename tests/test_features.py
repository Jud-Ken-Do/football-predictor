import pandas as pd
import pytest

from football_predictor.features.form import FormFeatures
from football_predictor.features.elo import EloFeatures
from football_predictor.features.h2h import H2HFeatures
from football_predictor.features.standings import StandingsFeatures


SAMPLE_MATCHES = pd.DataFrame([
    {"date": "2023-08-12", "home_team": "Arsenal", "away_team": "Nottingham Forest", "home_goals": 2, "away_goals": 1},
    {"date": "2023-08-19", "home_team": "Manchester City", "away_team": "Arsenal", "home_goals": 1, "away_goals": 0},
    {"date": "2023-08-26", "home_team": "Arsenal", "away_team": "Fulham", "home_goals": 2, "away_goals": 2},
    {"date": "2023-09-02", "home_team": "Everton", "away_team": "Arsenal", "home_goals": 0, "away_goals": 1},
    {"date": "2023-09-16", "home_team": "Arsenal", "away_team": "Manchester City", "home_goals": 1, "away_goals": 0},
    {"date": "2023-09-23", "home_team": "Tottenham", "away_team": "Arsenal", "home_goals": 2, "away_goals": 2},
])

FIXTURE = pd.Series({"date": "2023-10-01", "home_team": "Arsenal", "away_team": "Manchester City"})


def test_form_returns_expected_keys():
    module = FormFeatures()
    result = module.transform(FIXTURE, SAMPLE_MATCHES)
    assert "form_home_pts_last5" in result
    assert "form_away_pts_last5" in result
    assert "form_pts_diff_last5" in result


def test_form_values_are_floats():
    module = FormFeatures()
    result = module.transform(FIXTURE, SAMPLE_MATCHES)
    for k, v in result.items():
        assert isinstance(v, float), f"{k} is not a float"


def test_elo_returns_expected_keys():
    module = EloFeatures()
    result = module.transform(FIXTURE, SAMPLE_MATCHES)
    assert "elo_home_rating" in result
    assert "elo_diff" in result
    assert "elo_home_win_prob" in result


def test_elo_probabilities_in_range():
    module = EloFeatures()
    result = module.transform(FIXTURE, SAMPLE_MATCHES)
    assert 0.0 <= result["elo_home_win_prob"] <= 1.0


def test_h2h_no_history_returns_defaults():
    module = H2HFeatures()
    fixture = pd.Series({"date": "2023-10-01", "home_team": "Arsenal", "away_team": "Burnley"})
    result = module.transform(fixture, SAMPLE_MATCHES)
    assert result["h2h_n_matches"] == 0.0
    assert abs(result["h2h_home_win_rate"] - 0.33) < 0.01


def test_standings_reflects_results():
    module = StandingsFeatures()
    result = module.transform(FIXTURE, SAMPLE_MATCHES)
    assert "standings_home_pts" in result
    assert "standings_rank_diff" in result
    assert result["standings_home_pts"] >= 0


def test_form_feature_names_match_output():
    module = FormFeatures()
    result = module.transform(FIXTURE, SAMPLE_MATCHES)
    declared = set(module.feature_names())
    actual = set(result.keys())
    assert declared == actual
