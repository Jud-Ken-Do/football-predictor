"""Bookmaker closing odds as features.

WHY THIS MATTERS
─────────────────
Closing odds are the strongest single predictor in football (Zeileis et al. 2018,
*Journal of Statistical Software*; Constantinou & Fenton 2013). They represent the
aggregate of all publicly available information: form, injuries, team news, tactical
matchups, public sentiment — everything the market has already priced in.

Including odds as features lets the ML model learn *when and how much* to deviate
from the market consensus. Research consistently shows that a model trained against
closing odds as a feature outperforms one trained without them, even if the model
itself is calibrated against those same odds.

FEATURES (8)
─────────────
  odds_home_prob      — market implied P(home win), normalised (overround removed)
  odds_draw_prob      — market implied P(draw)
  odds_away_prob      — market implied P(away win)
  odds_home_adv       — log(P_home / P_away), directional strength ratio
  odds_draw_implied   — raw draw implied prob before normalisation (market draw signal)
  odds_overround      — bookmaker margin (1.0 = fair; typically 1.04–1.08)
  odds_favourite      — +1 home favourite, 0 pick-em, -1 away favourite
  odds_available      — 1.0 if odds found in data, 0.0 for future matches (WC 2026)

DATA COVERAGE
──────────────
  WC 2018 — 64 matches (Pinnacle closing odds)
  WC 2022 — 64 matches (bet365 + Betfair closing odds)
  WC 2026 qualifiers — 889 matches (market average odds)
  WC 2026 fixtures   — live pre-match odds via data/wc2026_odds_cache.json
                       (API-Football, merged in build_odds_lookup) → odds_available=1
                       when the cache has the fixture; flat 0.333 otherwise

References:
    Zeileis, A. et al. (2018). Probabilistic forecasts for the 2018 FIFA World Cup.
    Journal of Statistical Software. arXiv:1806.03208
    Constantinou, A. & Fenton, N. (2013). Determining the level of ability of
    football teams by dynamic ratings based on the relative discrepancies in scores.
    Journal of Quantitative Analysis in Sports.
"""
from __future__ import annotations

import math

import pandas as pd

from football_predictor.features.base import FeatureModule
from football_predictor.data.sources.football_data_co_uk import build_odds_lookup

_FLAT = 1.0 / 3.0  # equal-odds baseline for unknown matches


class OddsFeatures(FeatureModule):
    """Bookmaker closing odds → implied probabilities + derived features."""

    name = "odds"

    def __init__(self) -> None:
        self._lookup = None

    def _ensure_lookup(self) -> None:
        if self._lookup is None:
            self._lookup = build_odds_lookup()

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_lookup()

        from football_predictor.data.wc2026 import normalise

        # Lookup keys are built with normalise() (football_data_co_uk.py) —
        # query with the same canonical names or renamed teams silently miss.
        home = normalise(str(match.get("home_team", "")))
        away = normalise(str(match.get("away_team", "")))
        date_str = str(pd.Timestamp(match["date"]).date())

        entry = self._lookup.get((home, away, date_str))

        if entry is None:
            # Try reversed (some datasets store neutral games either way)
            rev = self._lookup.get((away, home, date_str))
            if rev:
                entry = {
                    "ph": rev["pa"], "pd": rev["pd"], "pa": rev["ph"],
                    "overround": rev["overround"],
                    "avg_h": rev["avg_a"], "avg_d": rev["avg_d"], "avg_a": rev["avg_h"],
                }

        if entry is None:
            return self._flat_features()

        ph, pd_p, pa = entry["ph"], entry["pd"], entry["pa"]
        overround = entry.get("overround", 1.05)

        if ph > pa:
            favourite = 1.0
        elif pa > ph:
            favourite = -1.0
        else:
            favourite = 0.0

        log_adv = math.log(ph / max(pa, 1e-6))

        return {
            "odds_home_prob":    ph,
            "odds_draw_prob":    pd_p,
            "odds_away_prob":    pa,
            "odds_home_adv":     log_adv,
            "odds_draw_implied": 1.0 / max(entry.get("avg_d", 3.0), 1.0),
            "odds_overround":    overround,
            "odds_favourite":    favourite,
            "odds_available":    1.0,
        }

    def _flat_features(self) -> dict[str, float]:
        return {
            "odds_home_prob":    _FLAT,
            "odds_draw_prob":    _FLAT,
            "odds_away_prob":    _FLAT,
            "odds_home_adv":     0.0,
            "odds_draw_implied": _FLAT,
            "odds_overround":    1.0,
            "odds_favourite":    0.0,
            "odds_available":    0.0,
        }

    def feature_names(self) -> list[str]:
        return [
            "odds_home_prob", "odds_draw_prob", "odds_away_prob",
            "odds_home_adv", "odds_draw_implied",
            "odds_overround", "odds_favourite", "odds_available",
        ]
