"""Post-processing adjustments using WC-2026-specific context features.

Player absence layer (new):
  3. absence_adjust(p_h, p_d, p_a, home_team, away_team, wc_date) → adjusted probs
     Reads data/wc2026_player_injuries.json to compute a star-player absence penalty.
     For each team, scores absent players by (market_value_m / squad_total_mv) and
     applies a log-odds shift proportional to the quality fraction unavailable.
     Falls back silently if injury data is missing or incomplete.

Applies signals from WC_CONTEXT_MODULES (squad_wc2026, api_form, sofifa_ratings,
venue_wc2026, injury) that cannot be included in XGBoost training because they
are zero for all historical rows (covariate shift).  Applied in two stages:

  1. venue_adjust(lam_h, lam_a, ctx) → adjusted λ pair
     Modifies BayesPoisson goal-rate estimates for partial home advantage
     (MEX/USA/CAN in their own stadium) and altitude acclimatisation.

  2. quality_nudge(p_h, p_d, p_a, ctx) → adjusted (p_h, p_d, p_a)
     Small log-odds shift from sofifa overall rating diff and api_form
     weighted points diff.  p_draw is held fixed while home/away mass is
     reallocated by the log-odds shift, then all three renormalised.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

_INJURY_FILE = Path(__file__).resolve().parents[2] / "data" / "wc2026_player_injuries.json"
_ABSENCE_WEIGHT: float = 0.20   # max log-odds shift when a team's entire squad is absent

# ── Tuning constants ──────────────────────────────────────────────────────────

# Venue: how much partial home advantage (0–1 scale) maps to log-goal space.
# BP home_adv ≈ 0.20 in log-goals; partial_adv of 0.30 → 6% lambda boost.
_HOME_ADV_LOG: float = 0.20

# Altitude: how much the module's altitude_adv value maps to log-goal space.
_ALT_ADV_LOG: float = 0.15

# Quality: weight on the combined quality diff (already normalised to ≈[-1,1]).
# 0.12 means max quality gap shifts log-odds by ~0.12, probability by ~3pp.
_QUALITY_W: float = 0.12

# Sofifa normalisation: typical max diff in overall rating between WC teams.
_SOFIFA_NORM: float = 20.0

# api_form normalisation: max weighted pts diff over last 10 matches across WC teams.
_API_FORM_NORM: float = 15.0

# Poisson sum cap for probability recompute
_MAX_GOALS: int = 10

# Market odds blend weight: fraction of market-implied λ to mix into model λ.
# 0.20 = trust market for 20% of expected goals; rest stays with our model.
_MARKET_BLEND: float = 0.20

# Market 1X2 blend weight: linear mix of final outcome probs with the market's
# implied 1X2. Backtested in scripts/odds_blend_backtest.py (WC 2018+2022,
# 48/48 odds coverage each): log-loss improves monotonically to w≈0.8 with
# closing odds; 0.70 deployed because the WC 2026 cache holds earlier,
# softer pre-match odds. Affects outcome probs only — scorelines stay on λ.
_MARKET_1X2_BLEND: float = 0.70


# ── Helpers ───────────────────────────────────────────────────────────────────

def poisson_proba(lam_h: float, lam_a: float, rho: float = 0.0) -> tuple[float, float, float]:
    """Recompute (p_home_win, p_draw, p_away_win) from Poisson goal rates.

    Delegates to the shared DC-corrected expansion so the post-processing
    deltas use the same probability model as the BayesPoisson leg of the
    ensemble (previously ρ was dropped here, biasing the draw-cell deltas).
    """
    from football_predictor.models.bayesian_poisson import dc_outcome_probs
    return dc_outcome_probs(lam_h, lam_a, rho=rho, max_goals=_MAX_GOALS)


# ── Public API ────────────────────────────────────────────────────────────────

def build_wc_context(
    match_df: pd.DataFrame,
    all_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return a feature DataFrame (one row per match) from WC_CONTEXT_MODULES."""
    from football_predictor.constants import WC_CONTEXT_MODULES
    from football_predictor.data.pipeline import build_feature_matrix

    ctx, _ = build_feature_matrix(match_df, WC_CONTEXT_MODULES, context=all_data)
    return ctx


def venue_adjust(
    lam_h: float,
    lam_a: float,
    ctx: dict[str, float],
) -> tuple[float, float]:
    """Apply venue-level λ multipliers for partial home advantage and altitude."""
    h_partial = ctx.get("venue_home_partial_adv", 0.0)
    a_partial = ctx.get("venue_away_partial_adv", 0.0)
    h_alt = ctx.get("venue_home_altitude_adv", 0.0)
    a_alt = ctx.get("venue_away_altitude_adv", 0.0)

    lam_h_adj = lam_h * math.exp(_HOME_ADV_LOG * h_partial + _ALT_ADV_LOG * h_alt)
    lam_a_adj = lam_a * math.exp(_HOME_ADV_LOG * a_partial + _ALT_ADV_LOG * a_alt)
    return lam_h_adj, lam_a_adj


def quality_nudge(
    p_h: float,
    p_d: float,
    p_a: float,
    ctx: dict[str, float],
) -> tuple[float, float, float]:
    """Shift home/away win log-odds by a small sofifa + api_form quality signal.

    p_draw is held fixed while home/away mass is reallocated by the log-odds
    shift, then all three renormalised.
    """
    sofifa_diff = ctx.get("sofifa_diff_overall", 0.0) / _SOFIFA_NORM
    form_diff = ctx.get("api_form_diff_weighted_pts", 0.0) / _API_FORM_NORM

    # Transfermarkt squad-value signal — re-wired here after the module was
    # removed from XGBoost training (June-2026 values leaked into historical
    # rows). tm_value_ratio = log(home€/away€); /1.5 norms a ~4.5× value gap
    # to 1.0. Down-weighted vs sofifa/form because market value and sofifa
    # overall are strongly correlated (avoid double-counting squad quality).
    components = [sofifa_diff, form_diff]
    weights = [0.4, 0.4]
    if ctx.get("tm_available", 0.0) >= 1.0:
        tm_diff = max(-1.0, min(1.0, ctx.get("tm_value_ratio", 0.0) / 1.5))
        components.append(tm_diff)
        weights.append(0.2)
    else:
        weights = [0.5, 0.5]
    q = sum(w_i * c for w_i, c in zip(weights, components))  # [-1, 1]

    if abs(q) < 1e-6:
        return p_h, p_d, p_a

    p_h = max(p_h, 0.0)
    p_a = max(p_a, 0.0)
    logit = math.log((p_h + 1e-10) / (p_a + 1e-10)) + _QUALITY_W * q
    ratio = math.exp(logit)
    ha_sum = p_h + p_a
    p_h_new = ratio / (1.0 + ratio) * ha_sum
    p_a_new = 1.0 / (1.0 + ratio) * ha_sum

    total = p_h_new + p_d + p_a_new
    return p_h_new / total, p_d / total, p_a_new / total


def _load_injury_db() -> dict:
    """Load player injury history keyed by team name. Returns {} if file absent."""
    if not _INJURY_FILE.exists():
        return {}
    try:
        return json.loads(_INJURY_FILE.read_text())
    except Exception:
        return {}


_INJURY_DB: dict = {}   # module-level cache, loaded once


def _absent_quality_fraction(team: str) -> float:
    """Fraction of squad market value currently injured/suspended at tournament start.

    Returns a value in [0, 1]: 0 = fully available, 0.3 = 30% of squad value absent.
    Falls back to 0.0 when data is unavailable (most teams at present).
    """
    global _INJURY_DB
    if not _INJURY_DB:
        _INJURY_DB = _load_injury_db()

    players = _INJURY_DB.get(team, [])
    if not players:
        return 0.0

    total_mv = sum(p.get("market_value_m", 0.0) for p in players)
    if total_mv <= 0:
        return 0.0

    # NOTE: using the real wall-clock date is correct for live prediction, but
    # would be wrong in a backtest/replay context (absences should then be
    # evaluated as of the simulated match date, not today).
    today = date.today().isoformat()
    absent_mv = 0.0
    for p in players:
        for inj in p.get("injuries", []):
            until = inj.get("until")
            if until and until >= today:
                absent_mv += p.get("market_value_m", 0.0)
                break  # count each player at most once

    return min(absent_mv / total_mv, 1.0)


def _invert_over25(p_over: float) -> float:
    """Binary search for λ s.t. P(Poisson(λ) > 2.5) = p_over."""
    lo, hi = 0.1, 9.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p_under = math.exp(-mid) * (1 + mid + mid ** 2 / 2)
        if (1 - p_under) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def market_odds_adjust(
    lam_h: float,
    lam_a: float,
    ctx: dict[str, float],
) -> tuple[float, float]:
    """Blend model λ with market-implied λ derived from Asian Handicap + O/U 2.5.

    The market encodes two independent signals:
      - O/U 2.5 implied prob → total expected goals (λ_h + λ_a)
      - Asian Handicap line  → expected goal difference (λ_h − λ_a)

    When AH line is absent (= 0.0), falls back to preserving the model's own
    lam_h/lam_a ratio while anchoring the total to the market O/U level. This
    avoids incorrectly treating all matches as balanced when AH is unavailable.
    """
    if ctx.get("market_available", 0.0) < 0.5:
        return lam_h, lam_a

    p_over25 = ctx.get("market_over25", 0.0)
    ah_line = ctx.get("market_ah_line", 0.0)

    if p_over25 <= 0.05 or p_over25 >= 0.98:
        return lam_h, lam_a

    lam_total_market = _invert_over25(p_over25)

    if abs(ah_line) > 0.01:
        # AH line available: goal_diff from AH (negative line = home favoured)
        goal_diff_market = -ah_line
    else:
        # AH absent: preserve model's directional ratio, scale to market total
        lam_total_model = lam_h + lam_a
        if lam_total_model > 0.01:
            goal_diff_market = (lam_h - lam_a) / lam_total_model * lam_total_market
        else:
            goal_diff_market = 0.0

    lam_h_market = max((lam_total_market + goal_diff_market) / 2, 0.20)
    lam_a_market = max((lam_total_market - goal_diff_market) / 2, 0.20)

    lam_h_adj = (1 - _MARKET_BLEND) * lam_h + _MARKET_BLEND * lam_h_market
    lam_a_adj = (1 - _MARKET_BLEND) * lam_a + _MARKET_BLEND * lam_a_market
    return lam_h_adj, lam_a_adj


def market_1x2_blend(
    p_h: float,
    p_d: float,
    p_a: float,
    ctx: dict[str, float],
) -> tuple[float, float, float]:
    """Linear blend of final outcome probs with the market's 1X2 probabilities.

    Applied last: the market price already aggregates injuries, lineups and
    team news, so it anchors the final number; the model side (1−w) carries
    all earlier adjustments. No-op when the fixture has no cached odds.
    """
    if ctx.get("market_available", 0.0) < 0.5:
        return p_h, p_d, p_a

    mh = ctx.get("market_p_home", 0.0)
    md = ctx.get("market_p_draw", 0.0)
    ma = ctx.get("market_p_away", 0.0)
    tot = mh + md + ma
    if tot < 0.9:  # malformed / partial cache row
        return p_h, p_d, p_a
    mh, md, ma = mh / tot, md / tot, ma / tot

    w = _MARKET_1X2_BLEND
    p_h = (1 - w) * p_h + w * mh
    p_d = (1 - w) * p_d + w * md
    p_a = (1 - w) * p_a + w * ma
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


def absence_adjust(
    p_h: float,
    p_d: float,
    p_a: float,
    home_team: str,
    away_team: str,
) -> tuple[float, float, float]:
    """Shift home/away log-odds by star player absence penalty.

    Each team's unavailable quality fraction (absent market value / total) is
    converted to a log-odds shift: full absence → −ABSENCE_WEIGHT, none → 0.
    p_draw is held fixed while home/away mass is reallocated by the log-odds
    shift, then all three renormalised.
    """
    home_absent = _absent_quality_fraction(home_team)
    away_absent = _absent_quality_fraction(away_team)
    net_shift = _ABSENCE_WEIGHT * (away_absent - home_absent)  # positive → home advantage

    if abs(net_shift) < 1e-6:
        return p_h, p_d, p_a

    logit = math.log((p_h + 1e-10) / (p_a + 1e-10)) + net_shift
    ratio = math.exp(logit)
    ha_sum = p_h + p_a
    p_h_new = ratio / (1.0 + ratio) * ha_sum
    p_a_new = 1.0 / (1.0 + ratio) * ha_sum

    total = p_h_new + p_d + p_a_new
    return p_h_new / total, p_d / total, p_a_new / total


def apply_to_match(
    lam_h: float,
    lam_a: float,
    ens_p_h: float,
    ens_p_d: float,
    ens_p_a: float,
    ctx_row: dict[str, float],
    apply_venue: bool = True,
    home_team: str = "",
    away_team: str = "",
    rho: float = 0.0,
    return_preblend: bool = False,
) -> tuple[float, ...]:
    """Full post-processing pipeline for one match.

    Order: venue λ adjustment → market λ blend (AH/O-U) → quality nudge
    (sofifa+api_form+tm) → player absence → market 1X2 blend.
    rho: the fitted Dixon-Coles correlation (pass bp._rho) so the Poisson
    deltas share the ensemble's probability model.
    Returns (p_h, p_d, p_a, lam_h_adj, lam_a_adj).
    """
    if apply_venue:
        lam_h_adj, lam_a_adj = venue_adjust(lam_h, lam_a, ctx_row)
    else:
        lam_h_adj, lam_a_adj = lam_h, lam_a

    # Market odds blend: AH + O/U → market-implied λ blended with model λ
    lam_h_adj, lam_a_adj = market_odds_adjust(lam_h_adj, lam_a_adj, ctx_row)

    bp_h, bp_d, bp_a = poisson_proba(lam_h_adj, lam_a_adj, rho=rho)
    bp_h_orig, bp_d_orig, bp_a_orig = poisson_proba(lam_h, lam_a, rho=rho)

    p_h = ens_p_h + (bp_h - bp_h_orig)
    p_d = ens_p_d + (bp_d - bp_d_orig)
    p_a = ens_p_a + (bp_a - bp_a_orig)
    # The additive delta is unbounded — for lopsided fixtures it can push a
    # small probability below zero, and renormalisation does NOT fix the sign
    # (downstream: math domain error in absence_adjust, ValueError in
    # rng.choice). Clip to a floor before renormalising.
    p_h, p_d, p_a = max(p_h, 1e-6), max(p_d, 1e-6), max(p_a, 1e-6)
    total = p_h + p_d + p_a
    p_h, p_d, p_a = p_h / total, p_d / total, p_a / total

    p_h, p_d, p_a = quality_nudge(p_h, p_d, p_a, ctx_row)

    if home_team and away_team:
        p_h, p_d, p_a = absence_adjust(p_h, p_d, p_a, home_team, away_team)

    # Pre-1X2-blend probabilities = the model's own view before the market
    # anchor: venue + AH/O-U λ-blend + quality + absence are applied, but NOT
    # the 0.70 market 1X2 blend. Exposed for the un-blended "model's-eye" track
    # (the market is far sharper on favourites; this is the model alone).
    p_h_pre, p_d_pre, p_a_pre = p_h, p_d, p_a

    p_h, p_d, p_a = market_1x2_blend(p_h, p_d, p_a, ctx_row)

    if return_preblend:
        return p_h, p_d, p_a, lam_h_adj, lam_a_adj, p_h_pre, p_d_pre, p_a_pre
    return p_h, p_d, p_a, lam_h_adj, lam_a_adj
