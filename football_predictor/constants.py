from enum import Enum


# ── Signal processing ──────────────────────────────────────────────────────────

FORM_WINDOW_SIZES = [3, 5, 10]          # rolling windows (in matches) for form features

# ── International / World Cup specific ────────────────────────────────────────

# International teams play ~10 matches/year vs 38 for clubs — use wider windows
INTERNATIONAL_FORM_WINDOW_SIZES = [5, 10, 20]
ELO_K_FACTOR_INTERNATIONAL = 40         # higher K for international (more volatile)
# NOTE: there is no separate WC K-factor constant — the effective World Cup
# K-factor is ELO_K_FACTOR_INTERNATIONAL × 1.5 match weight (see elo.py).
ELO_HOME_ADVANTAGE = 100                # Elo points added for home team (0 for neutral venues)
ELO_NEUTRAL_ADVANTAGE = 0              # no home advantage at World Cup
PI_RATING_DECAY = 0.035                  # pi-rating per-game decay constant

CONFEDERATION_STRENGTH = {              # relative strength offsets by confederation
    # Derived from cross-confederation H2H win rates vs Elo expected (2000–2026,
    # competitive matches only). Mean-Elo approach discarded — CONCACAF/CONMEBOL
    # inflate Elo through insular regional tournaments (Gold Cup / Copa América).
    # True ordering from H2H evidence: CONMEBOL > UEFA > AFC > CAF > CONCACAF > OFC.
    #
    # Key findings that drove changes:
    #   CONMEBOL beats everyone +7–17% over expected    → raise from 40 → 65
    #   CONCACAF wins only 21.4% vs CAF (expected 66%)  → drop from 0 → -20
    #   CAF beats AFC only 22.6% (expected 37.6%)       → AFC raised, CAF unchanged at 10
    #   UEFA vs CONCACAF/CAF: ±1% gap → UEFA value confirmed
    "UEFA": 50,
    "CONMEBOL": 65,
    "CONCACAF": -20,
    "AFC": 25,
    "CAF": 10,
    "OFC": -40,
}

WORLD_CUP_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "FIFA World Cup qualification - CONMEBOL",
    "FIFA World Cup qualification - UEFA",
    "FIFA World Cup qualification - AFC",
    "FIFA World Cup qualification - CAF",
    "FIFA World Cup qualification - CONCACAF",
    "FIFA World Cup qualification - OFC",
]

MAJOR_TOURNAMENTS = WORLD_CUP_TOURNAMENTS + [
    "Copa América",
    "UEFA Euro",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "UEFA Nations League",
    "FIFA Confederations Cup",
]

# ── Dixon-Coles model ──────────────────────────────────────────────────────────

DC_TIME_DECAY_HALF_LIFE_DAYS = 120      # matches older than ~4 months contribute less
DC_RHO = -0.13                          # low-score correction (typical fitted value; re-estimated on fit)

# ── Inference thresholds ───────────────────────────────────────────────────────

DEFAULT_ONSET_THRESHOLD = 0.5           # minimum probability to call a win prediction
DEFAULT_DRAW_BAND = 0.10                # probability mass around 1/3 treated as "draw zone"

# ── Calibration ────────────────────────────────────────────────────────────────

CALIBRATION_METHOD = "isotonic"         # "isotonic" | "sigmoid" (Platt)
CALIBRATION_CV_FOLDS = 5

# ── Evaluation ─────────────────────────────────────────────────────────────────

BACKTEST_WINDOW_SEASONS = 2             # seasons held out for rolling backtest
OUTCOMES = ["home_win", "draw", "away_win"]

# ── Model selection ─────────────────────────────────────────────────────────────

class ModelType(Enum):
    DIXON_COLES = "dixon_coles"
    GRADIENT_BOOST = "gradient_boost"
    BAYESIAN = "bayesian"
    GNN = "gnn"                         # future; requires event-level data


DEFAULT_MODEL = ModelType.GRADIENT_BOOST

# ── R9: empirical-Bayes shrinkage of form / SoS window means ──────────────────
# Rolling form & SoS window stats are per-game means over as few as a handful of
# matches — noisy for low-cap teams (Curaçao, Jordan, …). When True, each window
# mean is shrunk toward the population mean weighted by sample size:
#     shrunk = (n·window_mean + K·pop_mean) / (n + K)
# ❌ TESTED & REJECTED 2026-06-15 (scripts/roadmap_ablation.py): worsened WC
# log-loss on all 3 folds (avg 1.0084→1.0128, +0.0044). Shrinkage flattens the
# form/SoS features XGBoost was already exploiting (it has the n_matches feature
# to gauge reliability itself). Kept behind a flag, default OFF.
USE_FORM_SHRINKAGE = False
FORM_SHRINKAGE_K = 5.0   # prior pseudo-count (in matches)

# ── R10: estimate Kalman baseline μ and home advantage from the data ──────────
# Replaces hard-coded μ=log(1.3) and home_adv=0.20 with moment estimates from
# the goal data (att/def are zero-mean, so no-home-term goals give E[goals]=exp(μ);
# non-neutral home goals give the extra exp(home_adv)). Clamped to sane ranges.
# ✅ KEPT 2026-06-15: improved WC log-loss on all 3 folds (avg 1.0084→1.0036,
# −0.0048), best single config in the ablation. Within the ±0.1 CI but
# directionally consistent and principled (removes two hand-set hyperparameters)
# — same keep-standard as R5/R7. Revert to False to disable.
USE_KALMAN_EM_MU_HA = True

# ── xG observation in Kalman EKF ──────────────────────────────────────────────
# When True, the Kalman filter observes a blend of actual goals and calibrated
# per-match xG (StatsBomb + football-data) instead of raw goals — a lower-noise
# state-space observation (Koopman-Lit). Enabled 2026-06-14 after the xG
# ablation (scripts/backtest.py --ablation --continental). Honest read of the
# evidence: full(xG) lowered log-loss on 4 of 5 covered folds (WC2022 -0.0148,
# AFCON2022 -0.0086, Euro2020 -0.0042, Copa2021 -0.0039) and was a wash on the
# 5th (AsianCup2023 -0.0002); none got worse. Every per-fold gain sits WITHIN
# the wide bootstrap CIs — this is a CONSISTENCY-based decision (4/5 same
# direction, sign-test p≈0.06), NOT a CI-beating one. The supporting signal is
# the per-confederation slice: on WC2022 the gap confederations improve most
# (CONCACAF -0.033, CAF/AFC -0.016), exactly the mechanism's promise, and live
# coverage (AFCON23 + Copa24 + WC22 + qualifiers) is far better than the thin
# (~1%) pre-2022 backtest coverage — so the CAF/CONCACAF gain for WC 2026 is an
# extrapolation of a mechanism tested mostly on UEFA data. Revert to False to disable.
USE_XG_OBSERVATION = True
# obs = XG_OBS_BLEND * goals + (1 - XG_OBS_BLEND) * xg  (per team, when xG present)
XG_OBS_BLEND = 0.5

# ── Feature modules active by default ─────────────────────────────────────────

DEFAULT_FEATURE_MODULES = [
    # ── Rating systems ────────────────────────────────────────────────────────
    "elo",              # basic Elo (kept for backward-compat feature coverage)
    "glicko2",          # Glicko-2: rating + deviation φ (Glickman 2001)
    "kalman_strength",  # EKF time-varying attack/defense + uncertainty (Koopman & Lit 2015)
    # ── Match history ─────────────────────────────────────────────────────────
    "form",             # rolling pts/goals/GD over 5/10/20 matches
    "rest",             # days since last match + trailing fixture congestion (Tier-0)
    "sos",              # strength of schedule: opponent-quality-adjusted win rate
    "h2h",              # head-to-head record
    "squad_strength",   # long-term attack/defence quality from intl results
    # ── Context / structure ───────────────────────────────────────────────────
    "confederation",    # UEFA/CONMEBOL/AFC/CAF/CONCACAF strength encoding
    # "tournament_stage" REMOVED from training (2026-06-12): the martj42
    # training data has no `stage` column, so every historical row got the
    # default "group_stage" — all six features were constants (dead weight,
    # pruned or uninformative). Module file kept for potential future use.
    # ── External data (historically available) ────────────────────────────────
    "rankings",         # FIFA world rankings points (Dato-Futbol, 1992–2024)
    "odds",             # bookmaker closing odds (qualifier history; WC 2026 pre-match TBD)
    "xg_form",          # rolling xG / xGA — excel (UEFA/AFC/CONMEBOL) + FBref (CAF/CONCACAF)
    # "xg",             # expected goals — requires StatsBomb/Opta event data
    # "transfermarkt" REMOVED from training (2026-06-12): its June-2026 squad
    # values were applied to ALL historical rows — a future-derived quality
    # measure leaking into past training data (same covariate-shift class that
    # excluded sofifa). Now a WC context module below.
]

# WC-2026-specific modules that produce zeros for all historical training rows.
# Excluded from DEFAULT_FEATURE_MODULES to prevent covariate shift: XGBoost
# cannot learn a signal that is always 0 during training but non-zero at
# prediction time.  Applied as post-processing adjustments instead — see
# football_predictor/models/wc_context.py.
WC_CONTEXT_MODULES = [
    "squad_wc2026",    # WC 2026 club-tier / age features from official FIFA squad list
    "api_form",        # last-10-match form from API-Football (pre-cached for WC 2026 teams)
    "sofifa_ratings",  # EA FC 26 squad quality: overall, pace, shooting, passing, etc.
    "venue_wc2026",    # partial home advantage (MEX/USA/CAN), altitude, travel burden
    "injury",          # pre-match injuries/suspensions from API-Football cache
    "wc2026_market",   # live pre-match odds: AH line, O/U 2.5, BTTS, clean sheet (API-Football)
    "transfermarkt",   # squad market values (June 2026 snapshot — leakage if trained on)
]

# ── Data sources ───────────────────────────────────────────────────────────────

FOOTBALL_DATA_ORG_BASE_URL = "https://api.football-data.org/v4"
UNDERSTAT_BASE_URL = "https://understat.com"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

SUPPORTED_COMPETITIONS = {
    "PL":  "Premier League",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "PD":  "La Liga",
    "FL1": "Ligue 1",
    "CL":  "UEFA Champions League",
}

AUDIO_SAMPLE_RATE = 22050               # kept as a reminder of where this project started
