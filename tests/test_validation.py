"""Tests for R6 ingress validation (data/validation.py)."""
import pandas as pd
import pytest

from football_predictor.data.validation import validate_wc2026_coverage
from football_predictor.data.wc2026 import ALL_WC2026_TEAMS, normalise


def _synthetic_full_coverage(n=40):
    """Training frame where every WC team has >= n matches (vs each other)."""
    teams = [normalise(t) for t in ALL_WC2026_TEAMS]
    rows = []
    d = pd.Timestamp("2020-01-01")
    for i, t in enumerate(teams):
        opp = teams[(i + 1) % len(teams)]
        for k in range(n):
            rows.append({"date": d + pd.Timedelta(days=i * 50 + k),
                         "home_team": t, "away_team": opp,
                         "home_goals": 1, "away_goals": 0})
    return pd.DataFrame(rows)


def test_report_covers_all_48_teams():
    rep = validate_wc2026_coverage(_synthetic_full_coverage(), quiet=True)
    assert len(rep) == 48
    assert all(t in rep for t in ALL_WC2026_TEAMS)


def test_matches_threshold_flags_thin_teams():
    rep = validate_wc2026_coverage(_synthetic_full_coverage(n=40), min_matches=30, quiet=True)
    # every team has 40 matches -> matches check passes for all
    assert all(rep[t]["matches"] for t in ALL_WC2026_TEAMS)
    rep2 = validate_wc2026_coverage(_synthetic_full_coverage(n=5), min_matches=30, quiet=True)
    assert not any(rep2[t]["matches"] for t in ALL_WC2026_TEAMS)


def test_strict_raises_on_critical_gap():
    # No training matches at all -> 'matches' (critical) gap -> strict must raise.
    empty = pd.DataFrame({"home_team": [], "away_team": [], "date": [],
                          "home_goals": [], "away_goals": []})
    with pytest.raises(ValueError):
        validate_wc2026_coverage(empty, quiet=True, strict=True)
