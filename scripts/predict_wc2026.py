"""WC 2026 group stage predictions + Monte Carlo tournament simulation.

Usage:
    python3.11 scripts/predict_wc2026.py
    python3.11 scripts/predict_wc2026.py --sims 100000

Outputs:
  1. Per-match win/draw/loss probabilities for all 72 group stage matches
  2. Predicted group standings (expected points)
  3. Tournament progression probabilities (% per round per team)

Actual results integration:
    Record played matches with:  python3.11 scripts/update_wc2026.py --result "..." --score "X-Y"
    Stored in data/wc2026_actual_results.json and automatically loaded here.
"""
import argparse
import itertools
import json
import time
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from football_predictor.data.sources.international_results import fetch_training_data
from football_predictor.data.pipeline import build_feature_matrix, prune_correlated_features
from football_predictor.data.wc2026 import GROUPS, GROUP_STAGE_SCHEDULE, normalise
from football_predictor.models.gradient_boost import GradientBoostModel
from football_predictor.models.bayesian_poisson import BayesianPoissonModel
from football_predictor.models.calibration import TemperatureScaling
from football_predictor.models.ensemble import EnsembleModel
from football_predictor.constants import DEFAULT_FEATURE_MODULES

_ROOT = Path(__file__).resolve().parent.parent
_ACTUAL_RESULTS_FILE = _ROOT / "data" / "wc2026_actual_results.json"
_TUNED_PARAMS_FILE   = _ROOT / "data" / "tuned_params.json"


def _load_friendly_weight() -> float:
    try:
        return json.loads(_TUNED_PARAMS_FILE.read_text()).get("friendly_weight", 0.7) if _TUNED_PARAMS_FILE.exists() else 0.7
    except Exception:
        return 0.7


def load_actual_results() -> list[dict]:
    """Load actual WC 2026 match results recorded by update_wc2026.py."""
    if not _ACTUAL_RESULTS_FILE.exists():
        return []
    try:
        with open(_ACTUAL_RESULTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_actual_results(all_data: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """Append recorded WC 2026 results to the training data, with dedup.

    - Canonicalises recorded names (update_wc2026.py may store user-typed
      variants like "Iran") so no phantom teams enter the rating systems.
    - Dedups against the martj42 source: its cache refreshes every 48h and
      will itself contain played WC 2026 matches — without this, every
      recorded result is seen TWICE by Elo/Glicko/Kalman/BayesPoisson at
      match weight 1.5.
    - Validates that every WC 2026 team resolves to a name present in the
      training data (a silent miss means default ratings for that team).
    """
    # Only WC group stage matches (group A–L) — friendlies or warmups excluded
    actual = [r for r in load_actual_results()
              if str(r.get("group", "")).upper() not in ("", "FRIENDLY", "WARMUP", "TEST")]
    if actual:
        actual_df = pd.DataFrame(actual)
        actual_df["home_team"] = actual_df["home_team"].astype(str).map(normalise)
        actual_df["away_team"] = actual_df["away_team"].astype(str).map(normalise)
        actual_df["date"] = pd.to_datetime(actual_df["date"])
        actual_df["tournament"] = "FIFA World Cup"
        actual_df["match_weight"] = 1.5
        # Host nations playing at home are NOT neutral for rating updates —
        # a Mexico win in Mexico City updated Elo/Kalman with no HA discount,
        # inflating host ratings. (Hosts listed as the away side keep
        # neutral=True — Elo can't express away-side HA; acceptable loss.)
        _hosts = {"Mexico", "United States", "Canada"}
        actual_df["neutral"] = ~actual_df["home_team"].isin(_hosts)
        for col in ["home_goals", "away_goals"]:
            if col not in actual_df.columns:
                actual_df[col] = 0

        recorded_pairs = {
            frozenset((h, a))
            for h, a in zip(actual_df["home_team"], actual_df["away_team"])
        }
        is_wc26 = (
            all_data["tournament"].astype(str).str.contains("World Cup", na=False)
            & ~all_data["tournament"].astype(str).str.contains("qualif", case=False, na=False)
            & (pd.to_datetime(all_data["date"]) >= pd.Timestamp("2026-06-01"))
        )
        if is_wc26.any():
            dup_mask = all_data.loc[is_wc26].apply(
                lambda r: frozenset((normalise(str(r["home_team"])),
                                     normalise(str(r["away_team"])))) in recorded_pairs,
                axis=1,
            )
            n_dropped = int(dup_mask.sum())
            if n_dropped:
                all_data = all_data.drop(index=dup_mask[dup_mask].index)
                if not quiet:
                    print(f"  - {n_dropped} duplicate WC 2026 rows removed from source data")
        all_data = pd.concat([all_data, actual_df], ignore_index=True)
        if not quiet:
            print(f"  + {len(actual)} actual WC 2026 results added to training context")

    from football_predictor.data.wc2026 import validate_team_coverage
    coverage_problems = validate_team_coverage(all_data, min_matches=10)
    if coverage_problems:
        print("  ⚠️  TEAM NAME COVERAGE PROBLEMS (teams will get default ratings!):")
        for p in coverage_problems:
            print(f"     {p}")

    return all_data


def merge_actual_results(match_data: list[dict], actual: list[dict]) -> list[dict]:
    """Attach actual scores to already-played fixtures; freeze their probabilities."""
    if not actual:
        return match_data

    # Build lookup: (normalised_home, normalised_away) → result dict.
    # Both orientations are stored so a result recorded with teams swapped
    # still locks the fixture (goals swapped accordingly).
    lookup: dict[tuple, dict] = {}
    for r in actual:
        h, a = normalise(r["home_team"]), normalise(r["away_team"])
        lookup[(h, a)] = r
        lookup.setdefault((a, h), {**r, "home_goals": r["away_goals"],
                                   "away_goals": r["home_goals"],
                                   "home_team": r["away_team"],
                                   "away_team": r["home_team"]})

    updated = []
    for m in match_data:
        key = (normalise(m["home_team"]), normalise(m["away_team"]))
        if key in lookup:
            r = lookup[key]
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
            m = dict(m, played=True, actual_home_goals=hg, actual_away_goals=ag)
            # Freeze probabilities to the known outcome (for expected standings display)
            if hg > ag:
                m["p_home"], m["p_draw"], m["p_away"] = 1.0, 0.0, 0.0
            elif hg == ag:
                m["p_home"], m["p_draw"], m["p_away"] = 0.0, 1.0, 0.0
            else:
                m["p_home"], m["p_draw"], m["p_away"] = 0.0, 0.0, 1.0
        else:
            m = dict(m, played=False)
        updated.append(m)

    return updated

# ── R32 bracket: (home_slot, away_slot) pairs → R16 groups ──────────────────
# Based on official FIFA WC 2026 bracket (Wikipedia / FIFA.com)
# Notation: ("1A", "2B") means winner of Group A vs runner-up of Group B
# "3rd" slots filled by best-available 3rd-place team from allowed groups.

R32_MATCHES: list[tuple] = [
    # Pod 1 → feeds R16-1 and R16-2
    ("2A", "2B"),           # M73
    ("1E", "3rd_ABCDF"),    # M74
    ("1F", "2C"),           # M75
    ("1C", "2F"),           # M76
    # Pod 2 → feeds R16-3 and R16-4
    ("1I", "3rd_CDFGH"),    # M77
    ("2E", "2I"),           # M78
    ("1A", "3rd_CEFHI"),    # M79
    ("1L", "3rd_EHIJK"),    # M80
    # Pod 3 → feeds R16-5 and R16-6
    ("1D", "3rd_BEFIJ"),    # M81
    ("1G", "3rd_AEHIJ"),    # M82
    ("2K", "2L"),           # M83
    ("1H", "2J"),           # M84
    # Pod 4 → feeds R16-7 and R16-8
    ("1B", "3rd_EFGIJ"),    # M85
    ("1J", "2H"),           # M86
    ("1K", "3rd_DEIJL"),    # M87
    ("2D", "2G"),           # M88
]

# R16 pairings: indices into R32_MATCHES (0-based; index k = match 73+k).
# Official FIFA bracket (M89–M96):
#   M89 = W74 v W77   M90 = W73 v W75   M91 = W76 v W78   M92 = W79 v W80
#   M93 = W83 v W84   M94 = W81 v W82   M95 = W86 v W88   M96 = W85 v W87
# NOT sequential winners — the bracket crosses pods.
R16_PAIRS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
# QF (M97–M100) as indices into R16 winners (index k = match 89+k):
#   M97 = W89 v W90   M98 = W93 v W94   M99 = W91 v W92   M100 = W95 v W96
QF_PAIRS  = [(0, 1), (4, 5), (2, 3), (6, 7)]
# SF (M101–M102) as indices into QF winners (index k = match 97+k):
#   M101 = W97 v W98   M102 = W99 v W100
SF_PAIRS  = [(0, 1), (2, 3)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=50_000)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--match", type=str, default=None,
                   help="Predict a single match, e.g. --match \"Brazil vs Morocco\"")
    p.add_argument("--mcmc", action="store_true",
                   help="Use full PyMC MCMC posterior for BayesPoisson (slow, ~5min)")
    p.add_argument("--mcmc-draws", type=int, default=500)
    p.add_argument("--mcmc-tune",  type=int, default=250)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the Monte Carlo simulation (reproducible runs)")
    return p.parse_args()


# ── Data & model training ─────────────────────────────────────────────────────

def train_model(
    quiet: bool = False,
    use_mcmc: bool = False,
    mcmc_draws: int = 500,
    mcmc_tune: int = 250,
) -> tuple:
    if not quiet:
        print("Loading historical data...")
    _fw = _load_friendly_weight()
    if not quiet:
        print(f"  friendly_weight={_fw} (from cache)" if _TUNED_PARAMS_FILE.exists() else f"  friendly_weight={_fw}")
    all_data = fetch_training_data(from_year=2010, friendly_weight=_fw)

    # Append actual WC 2026 results so Kalman EKF and BayesPoisson see them
    all_data = append_actual_results(all_data, quiet=quiet)

    comp_mask = all_data["tournament"].str.lower().str.contains(
        "qualif|world cup|copa|euro|nations|africa|asian|gold cup", na=False
    )
    train_df = all_data[comp_mask].reset_index(drop=True)

    if not quiet:
        print(f"  {len(train_df)} competitive matches  |  {len(DEFAULT_FEATURE_MODULES)} modules")
        print("Building features...")

    t0 = time.time()
    X, y = build_feature_matrix(train_df, DEFAULT_FEATURE_MODULES, context=all_data)
    X, _ = prune_correlated_features(X, threshold=0.95)
    sw = train_df["match_weight"].values
    if not quiet:
        print(f"  {X.shape[1]} features in {time.time()-t0:.1f}s")

    cut = int(len(X) * 0.8)
    cal_X, cal_y = X.iloc[cut:], y.iloc[cut:]

    # XGBoost + temperature scaling
    xgb = GradientBoostModel()
    xgb.fit(X.iloc[:cut], y.iloc[:cut], sample_weight=sw[:cut])
    xgb_proba_cal = xgb.predict_proba(cal_X)
    temp_cal = TemperatureScaling()
    # Weighted NLL: WC matches (w=1.5) count more than minor tournaments (0.6)
    temp_cal.fit(xgb_proba_cal, cal_y, sample_weight=sw[cut:])
    if not quiet:
        print(f"  XGB temperature T={temp_cal.temperature:.3f}")

    # Bayesian Hierarchical Poisson — trained on same 80% split as XGB so the
    # held-out 20% is genuinely out-of-sample for both models when fitting ensemble α.
    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    _q = KalmanStrengthFeatures.get_last_tuned_q()
    bp = BayesianPoissonModel(half_life_days=BayesianPoissonModel.half_life_from_q(_q))
    if use_mcmc:
        if not quiet:
            print("Fitting Bayesian Poisson (MCMC — this takes ~5 min)...")
        bp.fit_mcmc(train_df.iloc[:cut], draws=mcmc_draws, tune=mcmc_tune)
    else:
        if not quiet:
            print("Fitting Bayesian Poisson model (MAP)...")
        bp.fit(train_df.iloc[:cut])

    # Ensemble: learn optimal blend on held-out calibration set.
    # α must be fitted on the SAME distribution it blends at predict time —
    # i.e. temperature-scaled XGB probabilities, not raw ones.
    bp_proba_cal = _bp_proba_df(bp, train_df.iloc[cut:])
    ensemble = EnsembleModel()
    ensemble.fit(temp_cal.transform(xgb_proba_cal), bp_proba_cal, cal_y,
                 context_X=cal_X, sample_weight=sw[cut:])

    # Refit BP on full training data now that ensemble weight is fixed —
    # more data improves the goal-rate estimates used in prediction.
    if use_mcmc:
        bp.fit_mcmc(train_df, draws=mcmc_draws, tune=mcmc_tune)
    else:
        bp.fit(train_df)
    bp.fit_rho(train_df)
    if not quiet:
        print(f"  Ensemble: XGB={ensemble.xgb_weight:.2f}  BayesPoisson={ensemble.bp_weight:.2f}")

    return xgb, temp_cal, bp, ensemble, all_data, train_df


def _bp_proba_df(bp: BayesianPoissonModel, matches: pd.DataFrame) -> pd.DataFrame:
    """Compute BayesianPoisson match probabilities for a DataFrame of matches."""
    rows = []
    for _, m in matches.iterrows():
        p = bp.predict_proba(
            normalise(m["home_team"]), normalise(m["away_team"]),
            neutral=bool(m.get("neutral", True))
        )
        rows.append(p)
    return pd.DataFrame(rows, columns=["home_win", "draw", "away_win"])


# ── Fixture predictions ────────────────────────────────────────────────────────

def predict_group_stage(
    xgb: GradientBoostModel,
    temp_cal: TemperatureScaling,
    bp: BayesianPoissonModel,
    ensemble: EnsembleModel,
    all_data: pd.DataFrame,
    quiet: bool = False,
) -> list[dict]:
    fixtures = [
        dict(f, neutral=True, tournament="FIFA World Cup",
             match_weight=1.5, home_goals=0, away_goals=0)
        for f in GROUP_STAGE_SCHEDULE
    ]
    match_df = pd.DataFrame(fixtures)
    # venue column already present from GROUP_STAGE_SCHEDULE; use actual date if available
    if "date" not in match_df.columns:
        match_df["date"] = "2026-06-15"

    from football_predictor.models.wc_context import build_wc_context, apply_to_match

    # XGBoost + temperature scaling (trained on DEFAULT_FEATURE_MODULES only)
    X_gs, _ = build_feature_matrix(match_df, DEFAULT_FEATURE_MODULES, context=all_data)
    xgb_proba = temp_cal.transform(xgb.predict_proba(X_gs))

    # WC-context features (venue, sofifa, api_form, injury — not fed to XGBoost)
    ctx_gs = build_wc_context(match_df, all_data)

    results = []
    for i, f in enumerate(fixtures):
        home, away = f["home_team"], f["away_team"]

        # Bayesian Poisson for this fixture
        bp_p = bp.predict_proba(normalise(home), normalise(away), neutral=True)
        bp_row = pd.DataFrame([bp_p], columns=["home_win", "draw", "away_win"])

        # Ensemble blend
        xgb_row = xgb_proba.iloc[[i]].reset_index(drop=True)
        ens_row = ensemble.predict_proba(xgb_row, bp_row, context_X=X_gs.iloc[[i]].reset_index(drop=True))

        lam_h, lam_a = bp.get_lambdas(normalise(home), normalise(away), neutral=True)
        ctx_row = ctx_gs.iloc[i].to_dict() if i < len(ctx_gs) else {}

        # Apply WC context: venue λ-adjustment + quality log-odds nudge
        p_h, p_d, p_a, lam_h_adj, lam_a_adj = apply_to_match(
            lam_h, lam_a,
            float(ens_row["home_win"].iloc[0]),
            float(ens_row["draw"].iloc[0]),
            float(ens_row["away_win"].iloc[0]),
            ctx_row,
            apply_venue=True,
            home_team=home,
            away_team=away,
        )

        # Kalman uncertainty in log(λ): σ_log(λ_h) ≈ √(σ²_att_h + σ²_def_a)
        # Used in Monte Carlo to resample goal rates per simulation run.
        import math as _math
        h_att_std = float(X_gs.iloc[i].get("kalman_home_att_std", 0.0))
        a_def_std = float(X_gs.iloc[i].get("kalman_away_def_std", 0.0))
        a_att_std = float(X_gs.iloc[i].get("kalman_away_att_std", 0.0))
        h_def_std = float(X_gs.iloc[i].get("kalman_home_def_std", 0.0))
        lam_h_log_std = _math.sqrt(h_att_std ** 2 + a_def_std ** 2)
        lam_a_log_std = _math.sqrt(a_att_std ** 2 + h_def_std ** 2)

        results.append({
            **f,
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "bp_lam_home": lam_h_adj, "bp_lam_away": lam_a_adj,
            "kalman_lam_h_log_std": lam_h_log_std,
            "kalman_lam_a_log_std": lam_a_log_std,
        })
    return results


# ── Monte Carlo core ──────────────────────────────────────────────────────────

def simulate_goals(
    p_home: float,
    p_draw: float,
    p_away: float,
    lam_h: float,
    lam_a: float,
    lam_h_log_std: float = 0.0,
    lam_a_log_std: float = 0.0,
) -> tuple[int, int]:
    """Sample a scoreline using ensemble outcome probs for direction, Poisson for magnitude.

    When Kalman log-λ uncertainty (lam_h_log_std / lam_a_log_std) is provided,
    goal rates are resampled from their posterior each call — propagating model
    uncertainty into the tournament simulation rather than using point estimates.
    The direction (win/draw/loss) still comes from the full ensemble so XGBoost
    signal is preserved; only goal magnitude varies with the resampled λ.
    """
    # Mean-preserving lognormal: E[LogNormal(m, s)] = exp(m + s²/2), so the
    # location must be log(λ) − s²/2 or every goal rate inflates by e^{s²/2}
    # (+2–13% at typical Kalman stds). Gates are per-side: a zero-uncertainty
    # away λ must not be resampled just because the home side has uncertainty.
    if lam_h_log_std > 0.02:
        lam_h = float(np.random.lognormal(
            np.log(max(lam_h, 1e-3)) - 0.5 * lam_h_log_std ** 2, lam_h_log_std))
    if lam_a_log_std > 0.02:
        lam_a = float(np.random.lognormal(
            np.log(max(lam_a, 1e-3)) - 0.5 * lam_a_log_std ** 2, lam_a_log_std))

    r = np.random.random()
    if r < p_home:
        hg = max(1, int(np.random.poisson(lam_h)))
        ag = max(0, min(hg - 1, int(np.random.poisson(lam_a))))
    elif r < p_home + p_draw:
        g = int(np.random.poisson((lam_h + lam_a) / 2))
        hg = ag = g
    else:
        ag = max(1, int(np.random.poisson(lam_a)))
        hg = max(0, min(ag - 1, int(np.random.poisson(lam_h))))
    return hg, ag


def group_standings(teams: list[str], results: list[dict]) -> list[dict]:
    """Compute group table from simulated match results.

    Tiebreakers follow FIFA: points → GD → GF → head-to-head mini-table among
    the tied teams (pts, GD, GF) → drawing of lots (random per simulation).
    The previous (pts, gd, gf)-only sort resolved residual ties by Python's
    stable sort, i.e. by group listing order — a systematic bias favouring
    earlier-listed (often seeded) teams in exactly the tied cases that decide
    third-place qualification.
    """
    table: dict[str, dict] = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0} for t in teams}

    for r in results:
        h, a = r["home_team"], r["away_team"]
        hg, ag = r["home_goals"], r["away_goals"]
        for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            table[team]["gf"] += gf
            table[team]["ga"] += ga
            table[team]["gd"] += gf - ga
            table[team]["played"] += 1
        if hg > ag:
            table[h]["pts"] += 3
        elif hg == ag:
            table[h]["pts"] += 1
            table[a]["pts"] += 1
        else:
            table[a]["pts"] += 3

    def key3(t: dict) -> tuple:
        return (-t["pts"], -t["gd"], -t["gf"])

    rows = sorted(table.values(), key=key3)

    # Resolve residual ties: H2H mini-table among tied teams, then lots.
    out: list[dict] = []
    for _, block_iter in itertools.groupby(rows, key=key3):
        block = list(block_iter)
        if len(block) > 1:
            tied = {t["team"] for t in block}
            mini = {t: [0, 0, 0] for t in tied}  # [pts, gd, gf] among tied teams
            for r in results:
                h, a = r["home_team"], r["away_team"]
                if h in tied and a in tied:
                    hg, ag = r["home_goals"], r["away_goals"]
                    mini[h][1] += hg - ag; mini[h][2] += hg
                    mini[a][1] += ag - hg; mini[a][2] += ag
                    if hg > ag:
                        mini[h][0] += 3
                    elif hg == ag:
                        mini[h][0] += 1; mini[a][0] += 1
                    else:
                        mini[a][0] += 3
            lots = {t["team"]: np.random.random() for t in block}
            block.sort(key=lambda t: (-mini[t["team"]][0], -mini[t["team"]][1],
                                      -mini[t["team"]][2], lots[t["team"]]))
        out.extend(block)
    return out


def simulate_group_stage(match_data: list[dict]) -> dict[str, list[dict]]:
    """Simulate all group stage matches; use actual scores for played ones."""
    group_results: dict[str, list] = defaultdict(list)

    for m in match_data:
        if m.get("played"):
            hg, ag = m["actual_home_goals"], m["actual_away_goals"]
        else:
            hg, ag = simulate_goals(
                m["p_home"], m["p_draw"], m["p_away"],
                m["bp_lam_home"], m["bp_lam_away"],
                m.get("kalman_lam_h_log_std", 0.0),
                m.get("kalman_lam_a_log_std", 0.0),
            )
        group_results[m["group"]].append({
            "home_team": m["home_team"], "away_team": m["away_team"],
            "home_goals": hg, "away_goals": ag,
        })

    standings: dict[str, list[dict]] = {}
    for grp, teams in GROUPS.items():
        standings[grp] = group_standings(teams, group_results[grp])
    return standings


def get_qualifiers(standings: dict[str, list[dict]]) -> dict:
    """Determine the 32 teams that advance from group stage."""
    winners, runners_up = {}, {}
    third_place_pool = []

    for grp, table in standings.items():
        winners[grp] = table[0]["team"]
        runners_up[grp] = table[1]["team"]
        t = table[2]
        third_place_pool.append({
            "team": t["team"], "group": grp,
            "pts": t["pts"], "gd": t["gd"], "gf": t["gf"]
        })

    # Random final tiebreak = drawing of lots (otherwise group-letter order
    # systematically decides which tied third-place team qualifies).
    third_place_pool.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"], np.random.random()))
    best_8_thirds = {t["group"]: t["team"] for t in third_place_pool[:8]}

    return {"1": winners, "2": runners_up, "3": best_8_thirds}


def match_thirds(slot_allowed: list[set[str]], available: set[str]) -> list[str | None]:
    """Assign qualified third-place groups to bracket slots — feasibly.

    Backtracking bipartite matching (8 slots × 8 groups — trivial size).
    Slots are processed most-constrained-first so a feasible assignment is
    found whenever one exists. The previous greedy version consumed groups
    alphabetically and could leave a later slot with no allowed group even
    when a feasible assignment existed, silently violating FIFA constraints.
    """
    n = len(slot_allowed)
    options = [sorted(s & available) for s in slot_allowed]
    order = sorted(range(n), key=lambda i: len(options[i]))

    assignment: list[str | None] = [None] * n

    def bt(k: int, used: frozenset) -> bool:
        if k == len(order):
            return True
        i = order[k]
        for g in options[i]:
            if g not in used:
                assignment[i] = g
                if bt(k + 1, used | {g}):
                    return True
                assignment[i] = None
        return False

    if not bt(0, frozenset()):
        # No perfect matching (cannot happen with the official bracket sets,
        # kept as a safety net): fall back to greedy + dump remaining.
        used: set[str] = set()
        for i in range(n):
            pick = next((g for g in options[i] if g not in used), None)
            if pick is None:
                pick = next((g for g in sorted(available) if g not in used), None)
            assignment[i] = pick
            if pick:
                used.add(pick)
    return assignment


def build_r32(qualifiers: dict) -> list[tuple[str, str]]:
    """Map group stage positions to R32 matchups."""
    ones = qualifiers["1"]    # {group: team}
    twos = qualifiers["2"]
    thirds = qualifiers["3"]  # {group: team} — the 8 qualified third-placers

    # Collect third-place slots in bracket order, then solve the matching.
    slot_pos: list[tuple[int, int]] = []   # (match_idx, side)
    slot_allowed: list[set[str]] = []
    for i, (hs, aw) in enumerate(R32_MATCHES):
        if hs.startswith("3rd"):
            slot_pos.append((i, 0)); slot_allowed.append(set(hs.split("_")[1]))
        if aw.startswith("3rd"):
            slot_pos.append((i, 1)); slot_allowed.append(set(aw.split("_")[1]))
    assigned = match_thirds(slot_allowed, set(thirds.keys()))
    third_by_pos = {pos: (thirds.get(g) if g else None)
                    for pos, g in zip(slot_pos, assigned)}

    bracket = []
    for i, (home_slot, away_slot) in enumerate(R32_MATCHES):
        if home_slot.startswith("1"):
            home = ones[home_slot[1]]
        elif home_slot.startswith("2"):
            home = twos[home_slot[1]]
        else:
            home = third_by_pos.get((i, 0)) or "Unknown"

        if away_slot.startswith("1"):
            away = ones[away_slot[1]]
        elif away_slot.startswith("2"):
            away = twos[away_slot[1]]
        else:
            away = third_by_pos.get((i, 1)) or "Unknown"

        bracket.append((home, away))
    return bracket


def ko_winner(home: str, away: str, predictor) -> str:
    """Simulate a knockout match — 90' draw resolved by ET/penalties.

    ET/pens follow relative strength rather than a fair coin: conditioned on
    the 90' being level, the stronger side still wins more often in extra
    time (penalties are closer to even, so the blend slightly shrinks the
    edge toward 0.5 via the p_h/(p_h+p_a) renormalisation).
    """
    p_h, p_d, p_a = predictor(home, away)
    r = np.random.random()
    if r < p_h:
        return home
    elif r < p_h + p_d:
        p_win = p_h / max(p_h + p_a, 1e-10)
        return home if np.random.random() < p_win else away
    else:
        return away


def simulate_tournament(match_data: list[dict], predictor) -> dict[str, str]:
    """One full tournament simulation. Returns team → furthest round reached."""
    reached: dict[str, str] = {t: "group_stage" for grp in GROUPS.values() for t in grp}

    # Group stage
    standings = simulate_group_stage(match_data)
    qualifiers = get_qualifiers(standings)

    all_qualifiers = (
        list(qualifiers["1"].values()) +
        list(qualifiers["2"].values()) +
        [t for t in qualifiers["3"].values() if t]
    )
    for team in all_qualifiers:
        reached[team] = "r32"

    # Build R32
    r32 = build_r32(qualifiers)

    # Simulate R32 → R16 → QF → SF → Final
    r32_winners = [ko_winner(h, a, predictor) for h, a in r32]
    for w in r32_winners:
        reached[w] = "r16"

    r16_winners = [ko_winner(r32_winners[i], r32_winners[j], predictor) for i, j in R16_PAIRS]
    for w in r16_winners:
        reached[w] = "qf"

    qf_winners = [ko_winner(r16_winners[i], r16_winners[j], predictor) for i, j in QF_PAIRS]
    for w in qf_winners:
        reached[w] = "sf"

    sf_winners = [ko_winner(qf_winners[i], qf_winners[j], predictor) for i, j in SF_PAIRS]
    for w in sf_winners:
        reached[w] = "final"

    champion = ko_winner(sf_winners[0], sf_winners[1], predictor)
    reached[champion] = "winner"

    return reached


# ── Output formatting ─────────────────────────────────────────────────────────

def print_group_predictions(match_data: list[dict]) -> None:
    print("\n" + "═" * 80)
    print("  WC 2026 — GROUP STAGE MATCH PREDICTIONS")
    print("═" * 80)

    current_group = None
    for m in match_data:
        if m["group"] != current_group:
            current_group = m["group"]
            print(f"\n  ── Group {current_group} ──")
        h, a = m["home_team"], m["away_team"]
        ph, pd_, pa = m["p_home"], m["p_draw"], m["p_away"]
        fav = h if ph > pa else a
        fav_p = max(ph, pa)
        print(f"  {m['date']}  {h:24s} vs {a:24s}  "
              f"H {ph:4.0%}  D {pd_:4.0%}  A {pa:4.0%}  "
              f"(fav: {fav} {fav_p:.0%})")


def print_tournament_table(counts: dict[str, dict[str, int]], n_sims: int) -> None:
    rounds = ["r32", "r16", "qf", "sf", "final", "winner"]
    labels = ["R32", "R16", "QF", "SF", "Final", "Winner"]

    print("\n" + "═" * 90)
    print("  WC 2026 — TOURNAMENT PROGRESSION PROBABILITIES")
    print(f"  ({n_sims:,} simulations)")
    print("═" * 90)
    print(f"  {'Team':30s}", end="")
    for l in labels:
        print(f"  {l:>7s}", end="")
    print()
    print("  " + "─" * 86)

    # Sort by winner probability then finalist probability
    teams = sorted(
        counts.keys(),
        key=lambda t: (
            -counts[t].get("winner", 0),
            -counts[t].get("final", 0),
            -counts[t].get("sf", 0),
            -counts[t].get("qf", 0),
        )
    )
    for team in teams:
        c = counts[team]
        print(f"  {team:30s}", end="")
        for r in rounds:
            pct = c.get(r, 0) / n_sims  # cum_counts is already cumulative
            print(f"  {pct:6.1%} ", end="")
        print()


def print_expected_group_standings(match_data: list[dict]) -> None:
    """Expected group points derived directly from XGBoost match probabilities."""
    from collections import defaultdict
    exp_pts: dict[str, float] = defaultdict(float)
    for m in match_data:
        h, a = m["home_team"], m["away_team"]
        exp_pts[h] += 3 * m["p_home"] + m["p_draw"]
        exp_pts[a] += 3 * m["p_away"] + m["p_draw"]

    print("\n" + "═" * 60)
    print("  EXPECTED GROUP STANDINGS (ensemble probabilities)")
    print("═" * 60)
    for grp, teams in GROUPS.items():
        ranked = sorted(teams, key=lambda t: -exp_pts[t])
        print(f"\n  Group {grp}")
        for t in ranked:
            print(f"    {t:30s}  {exp_pts[t]:.2f} pts")


# ── Single-match prediction ───────────────────────────────────────────────────

def _last_5(team: str, all_data: pd.DataFrame) -> list[str]:
    """Return last 5 results for a team as ['W 2-1 vs France', ...] strings."""
    mask = (all_data["home_team"] == team) | (all_data["away_team"] == team)
    recent = all_data[mask].sort_values("date").tail(5)
    out = []
    for _, r in recent.iterrows():
        is_home = r["home_team"] == team
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        gf, ga = (hg, ag) if is_home else (ag, hg)
        opp = r["away_team"] if is_home else r["home_team"]
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        out.append(f"{result} {gf}-{ga} vs {opp}")
    return out


def predict_single_match(
    match_str: str,
    xgb: GradientBoostModel,
    temp_cal: TemperatureScaling,
    bp: BayesianPoissonModel,
    ensemble: EnsembleModel,
    all_data: pd.DataFrame,
) -> None:
    """Print a detailed prediction card for a single match."""
    sep = " vs " if " vs " in match_str else " v "
    parts = match_str.split(sep, 1)
    if len(parts) != 2:
        print(f"  Could not parse match: '{match_str}'. Use format 'Team A vs Team B'")
        return

    home_raw, away_raw = parts[0].strip(), parts[1].strip()

    # Fuzzy match against known WC 2026 teams
    all_teams = [t for grp in GROUPS.values() for t in grp]
    def best_match(q: str) -> str:
        q_low = q.lower()
        exact = [t for t in all_teams if t.lower() == q_low]
        if exact:
            return exact[0]
        partial = [t for t in all_teams if q_low in t.lower() or t.lower() in q_low]
        return partial[0] if partial else q

    home = best_match(home_raw)
    away = best_match(away_raw)

    # Find scheduled date/venue if in group stage
    fixture_info = next(
        (f for f in GROUP_STAGE_SCHEDULE
         if normalise(f["home_team"]) == normalise(home)
         and normalise(f["away_team"]) == normalise(away)),
        None,
    )
    date_str = fixture_info["date"] if fixture_info else "2026-07-01"
    venue    = fixture_info.get("venue", "Neutral") if fixture_info else "Neutral"

    # Build feature row
    row = pd.DataFrame([{
        "home_team": home, "away_team": away, "date": date_str,
        "neutral": True, "tournament": "FIFA World Cup",
        "match_weight": 1.5, "home_goals": 0, "away_goals": 0,
        "venue": venue,
    }])
    X, _ = build_feature_matrix(row, DEFAULT_FEATURE_MODULES, context=all_data)
    xgb_p  = temp_cal.transform(xgb.predict_proba(X))
    bp_p   = bp.predict_proba(normalise(home), normalise(away), neutral=True)
    bp_row = pd.DataFrame([bp_p], columns=["home_win", "draw", "away_win"])
    ens    = ensemble.predict_proba(xgb_p.reset_index(drop=True), bp_row, context_X=X.reset_index(drop=True))

    ens_ph = float(ens["home_win"].iloc[0])
    ens_pd = float(ens["draw"].iloc[0])
    ens_pa = float(ens["away_win"].iloc[0])
    lam_h, lam_a = bp.get_lambdas(normalise(home), normalise(away), neutral=True)

    # Apply same post-processing as predict_group_stage (market blend + quality nudge + absence)
    from football_predictor.models.wc_context import build_wc_context, apply_to_match
    ctx = build_wc_context(row, all_data)
    ctx_row = ctx.iloc[0].to_dict() if len(ctx) > 0 else {}
    ph, pd_, pa, lam_h, lam_a = apply_to_match(
        lam_h, lam_a, ens_ph, ens_pd, ens_pa, ctx_row,
        home_team=home, away_team=away,
    )

    # Most likely scorelines (Poisson mass)
    from scipy.stats import poisson as sp_poisson
    scores = []
    for hg in range(6):
        for ag in range(6):
            p = sp_poisson.pmf(hg, lam_h) * sp_poisson.pmf(ag, lam_a)
            scores.append((hg, ag, p))
    scores.sort(key=lambda x: -x[2])

    # Odds comparison (if available)
    from football_predictor.data.sources.football_data_co_uk import build_odds_lookup
    odds_lookup = build_odds_lookup()
    odds = odds_lookup.get((home, away, date_str)) or odds_lookup.get((home, away, ""))
    has_odds = odds is not None

    # Form
    home_form = _last_5(home, all_data)
    away_form = _last_5(away, all_data)

    W = 62
    print()
    print("═" * W)
    print(f"  {home}  vs  {away}".center(W))
    print(f"  {date_str}  ·  {venue}".center(W))
    print("═" * W)

    print(f"\n  {'RESULT PROBABILITIES':}")
    fav, fav_p = max([(home, ph), ("Draw", pd_), (away, pa)], key=lambda x: x[1])
    print(f"    {'Home win: ':12s} {ph*100:5.1f}%   {home}")
    print(f"    {'Draw: ':12s} {pd_*100:5.1f}%")
    print(f"    {'Away win: ':12s} {pa*100:5.1f}%   {away}")
    print(f"    Favourite: {fav} ({fav_p*100:.1f}%)")

    print(f"\n  {'EXPECTED GOALS (Bayesian Poisson)'}")
    print(f"    {home}: {lam_h:.2f}   {away}: {lam_a:.2f}")

    print(f"\n  {'TOP SCORELINES'}")
    for hg, ag, p in scores[:5]:
        print(f"    {hg}-{ag}   {p*100:.1f}%")

    if has_odds:
        print(f"\n  {'BOOKMAKER ODDS (closing)'}")
        raw_h = 1 / odds["ph"] if odds["ph"] > 0 else 0
        raw_d = 1 / odds["pd"] if odds["pd"] > 0 else 0
        raw_a = 1 / odds["pa"] if odds["pa"] > 0 else 0
        diff_h = ph - odds["ph"]
        diff_a = pa - odds["pa"]
        print(f"    Home:  {odds['ph']*100:.1f}% implied  (model: {ph*100:.1f}%,  edge: {diff_h*100:+.1f}%)")
        print(f"    Draw:  {odds['pd']*100:.1f}% implied  (model: {pd_*100:.1f}%)")
        print(f"    Away:  {odds['pa']*100:.1f}% implied  (model: {pa*100:.1f}%,  edge: {diff_a*100:+.1f}%)")
        best_edge = max(diff_h, diff_a, key=abs)
        if abs(diff_h) >= 0.05:
            tag = "VALUE HOME" if diff_h > 0 else "FADE HOME"
            print(f"    *** {tag}: model {diff_h*100:+.1f}% vs market ***")
        if abs(diff_a) >= 0.05:
            tag = "VALUE AWAY" if diff_a > 0 else "FADE AWAY"
            print(f"    *** {tag}: model {diff_a*100:+.1f}% vs market ***")
    else:
        print(f"\n  BOOKMAKER ODDS  — not available yet (pre-match odds not in dataset)")

    print(f"\n  FORM (last 5)")
    home_col = [f"{r}" for r in home_form]
    away_col = [f"{r}" for r in away_form]
    print(f"    {home}")
    for r in home_col:
        print(f"      {r}")
    print(f"    {away}")
    for r in away_col:
        print(f"      {r}")

    print()
    print("═" * W)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t0 = time.time()

    xgb, temp_cal, bp, ensemble, all_data, train_df = train_model(
        quiet=args.quiet,
        use_mcmc=args.mcmc,
        mcmc_draws=args.mcmc_draws,
        mcmc_tune=args.mcmc_tune,
    )

    if args.match:
        predict_single_match(args.match, xgb, temp_cal, bp, ensemble, all_data)
        return

    match_data = predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=args.quiet)

    # Load and merge actual results (recorded via update_wc2026.py)
    actual = load_actual_results()
    if actual and not args.quiet:
        print(f"\n  Loaded {len(actual)} actual WC 2026 result(s) — locking those matches.")
    match_data = merge_actual_results(match_data, actual)

    # Match predictions table
    print_group_predictions(match_data)

    # Expected group standings (from XGBoost probabilities, not DC simulation)
    print_expected_group_standings(match_data)

    # Pre-compute probabilities for ALL 48×48 team pairs in one batch (fast)
    if not args.quiet:
        print("\nPre-computing all team-pair probabilities...")

    all_teams = [t for grp in GROUPS.values() for t in grp]
    pair_rows = []
    for home in all_teams:
        for away in all_teams:
            if home != away:
                pair_rows.append({
                    "home_team": home, "away_team": away,
                    "date": "2026-07-01", "neutral": True,
                    "tournament": "FIFA World Cup", "match_weight": 1.5,
                    "home_goals": 0, "away_goals": 0,
                })
    from football_predictor.models.wc_context import build_wc_context, quality_nudge, absence_adjust

    pair_df = pd.DataFrame(pair_rows)
    X_pairs, _ = build_feature_matrix(pair_df, DEFAULT_FEATURE_MODULES, context=all_data)
    ctx_pairs = build_wc_context(pair_df, all_data)

    # Ensemble: XGB + temperature scaling + BayesPoisson blend
    xgb_pairs = temp_cal.transform(xgb.predict_proba(X_pairs))
    bp_pairs_rows = []
    for _, row in pair_df.iterrows():
        p = bp.predict_proba(normalise(row["home_team"]), normalise(row["away_team"]), neutral=True)
        bp_pairs_rows.append(p)
    bp_pairs = pd.DataFrame(bp_pairs_rows, columns=["home_win", "draw", "away_win"])
    ens_pairs = ensemble.predict_proba(xgb_pairs.reset_index(drop=True), bp_pairs, context_X=X_pairs.reset_index(drop=True))

    prob_cache: dict[tuple, tuple] = {}
    for i, row in pair_df.iterrows():
        p = ens_pairs.iloc[i]
        ctx_row = ctx_pairs.iloc[i].to_dict() if i < len(ctx_pairs) else {}
        p_h, p_d, p_a = quality_nudge(
            float(p["home_win"]), float(p["draw"]), float(p["away_win"]), ctx_row
        )
        # Injury/suspension penalty applies in knockouts too (group-stage
        # matches get it via apply_to_match; KO pairs previously skipped it).
        p_h, p_d, p_a = absence_adjust(p_h, p_d, p_a, row["home_team"], row["away_team"])
        prob_cache[(row["home_team"], row["away_team"])] = (p_h, p_d, p_a)

    # Symmetrise: XGB features are not slot-symmetric, so P(A beats B | A in
    # the home slot) ≠ 1 − P(draw) − P(B beats A | B in the home slot). Average
    # the two orientations so knockout probabilities don't depend on which
    # arbitrary bracket slot a team lands in (all KO venues are neutral).
    for home, away in list(prob_cache.keys()):
        if home < away and (away, home) in prob_cache:
            f, r = prob_cache[(home, away)], prob_cache[(away, home)]
            p_h = (f[0] + r[2]) / 2.0
            p_d = (f[1] + r[1]) / 2.0
            p_a = (f[2] + r[0]) / 2.0
            s = p_h + p_d + p_a
            prob_cache[(home, away)] = (p_h / s, p_d / s, p_a / s)
            prob_cache[(away, home)] = (p_a / s, p_d / s, p_h / s)

    if not args.quiet:
        print(f"  {len(prob_cache)} pairs cached")
        print(f"\nRunning {args.sims:,} Monte Carlo simulations (seed={args.seed})...")

    # Reproducibility: the simulation uses the global np.random stream
    # (simulate_goals, ko_winner, tiebreak lots) — seed it once here.
    np.random.seed(args.seed)

    def ko_predictor(home: str, away: str) -> tuple[float, float, float]:
        return prob_cache.get((home, away), (0.4, 0.2, 0.4))

    # Monte Carlo
    round_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sim_i in range(args.sims):
        reached = simulate_tournament(match_data, ko_predictor)
        for team, rnd in reached.items():
            round_counts[team][rnd] += 1

    # Build cumulative counts for display
    cum_rounds = ["r32", "r16", "qf", "sf", "final", "winner"]
    cum_counts: dict[str, dict[str, int]] = {}
    for team, cnts in round_counts.items():
        cum: dict[str, int] = {}
        running = 0
        for r in reversed(cum_rounds):
            running += cnts.get(r, 0)
            cum[r] = running
        cum_counts[team] = cum

    print_tournament_table(cum_counts, args.sims)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # ── Save full output to file ──────────────────────────────────────────────
    import io
    import sys
    from datetime import datetime as _datetime

    out_dir = _ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"wc2026_predictions_{_datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    # Capture everything printed so far by re-calling the print functions into a buffer
    buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = buf

    print_group_predictions(match_data)
    print_expected_group_standings(match_data)
    print_tournament_table(cum_counts, args.sims)

    if actual:
        print(f"\nActual results incorporated: {len(actual)} matches played")

    sys.stdout = _old_stdout

    with open(out_path, "w") as fh:
        fh.write(buf.getvalue())

    print(f"\nFull output saved → {out_path}")


if __name__ == "__main__":
    main()
