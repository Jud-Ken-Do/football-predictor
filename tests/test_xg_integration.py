"""Tests for the StatsBomb xG source, calibration, and attach layer.

Pure-function tests only — no network. The aggregation, calibration and
orientation logic are the parts most likely to silently corrupt the signal.
"""
import numpy as np
import pandas as pd

from football_predictor.data.sources.statsbomb import _aggregate_match_xg, _norm
from football_predictor.models.xg_calibration import (
    AffineCalibration,
    fit_affine,
    fit_source_calibrations,
    to_team_match_long,
)


# ── StatsBomb aggregation ──────────────────────────────────────────────────────

def _shot(team, xg, period):
    return {"type": {"name": "Shot"}, "period": period,
            "team": {"name": team}, "shot": {"statsbomb_xg": xg}}


def test_aggregate_excludes_penalty_shootout():
    """Period-5 (shootout) shots must NOT count toward match xG."""
    events = [
        _shot("A", 0.3, 1), _shot("A", 0.2, 2), _shot("B", 0.5, 1),
        # shootout — would otherwise add ~0.78 each
        _shot("A", 0.78, 5), _shot("B", 0.78, 5), _shot("A", 0.78, 5),
    ]
    totals = _aggregate_match_xg(events)
    assert totals["A"] == 0.5    # 0.3 + 0.2, shootout excluded
    assert totals["B"] == 0.5


def test_aggregate_includes_extra_time():
    """Periods 3 and 4 (extra time) DO count."""
    events = [_shot("A", 0.1, 1), _shot("A", 0.2, 3), _shot("A", 0.3, 4)]
    assert abs(_aggregate_match_xg(events)["A"] - 0.6) < 1e-9


def test_aggregate_ignores_non_shots():
    events = [{"type": {"name": "Pass"}, "period": 1, "team": {"name": "A"}},
              _shot("A", 0.4, 1)]
    assert _aggregate_match_xg(events)["A"] == 0.4


def test_name_normalisation():
    assert _norm("Cape Verde Islands") == "Cabo Verde"
    assert _norm("Congo DR") == "DR Congo"
    assert _norm("Côte d'Ivoire") == "Ivory Coast"
    assert _norm("Iran") == "IR Iran"


# ── Calibration ────────────────────────────────────────────────────────────────

def test_mean_match_preserves_variance():
    """Default calibration matches the mean but keeps xG's (lower) variance."""
    rng = np.random.default_rng(0)
    goals = rng.poisson(1.4, size=500).astype(float)
    xg = 0.6 * goals + 0.5 + rng.normal(0, 0.2, size=500)  # biased, lower spread
    cal = fit_affine(xg, goals)
    cal_xg = cal.apply(xg)
    assert cal.b == 1.0
    assert abs(cal_xg.mean() - goals.mean()) < 1e-6        # unbiased
    assert cal_xg.std() < goals.std()                       # variance preserved (lower)


def test_match_std_equalises_variance():
    rng = np.random.default_rng(1)
    goals = rng.poisson(1.4, size=500).astype(float)
    xg = 0.6 * goals + 0.5 + rng.normal(0, 0.2, size=500)
    cal = fit_affine(xg, goals, match_std=True)
    cal_xg = cal.apply(xg)
    assert abs(cal_xg.std() - goals.std()) < 1e-6


def test_fit_affine_identity_on_tiny_data():
    cal = fit_affine(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert cal.a == 0.0 and cal.b == 1.0


def test_fit_source_calibrations_per_source():
    df = pd.DataFrame({
        "source": ["x"] * 20 + ["y"] * 20,
        "xg": list(np.linspace(0, 2, 20)) + list(np.linspace(0, 4, 20)),
        "goals": list(np.linspace(0, 2, 20)) + list(np.linspace(0, 2, 20)),
    })
    cals = fit_source_calibrations(df)
    assert set(cals) == {"x", "y"}
    assert all(isinstance(c, AffineCalibration) for c in cals.values())


def test_to_team_match_long_stacks_home_away():
    df = pd.DataFrame({
        "source": ["s"], "home_xg": [1.5], "away_xg": [0.5],
        "home_score": [2], "away_score": [0],
    })
    long = to_team_match_long(df)
    assert len(long) == 2
    assert set(long["xg"]) == {1.5, 0.5}
    assert set(long["goals"]) == {2, 0}
