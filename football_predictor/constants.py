from enum import Enum


# ── Signal processing ──────────────────────────────────────────────────────────

FORM_WINDOW_SIZES = [3, 5, 10]          # rolling windows (in matches) for form features

# ── International / World Cup specific ────────────────────────────────────────

# International teams play ~10 matches/year vs 38 for clubs — use wider windows
INTERNATIONAL_FORM_WINDOW_SIZES = [5, 10, 20]
ELO_K_FACTOR_INTERNATIONAL = 40         # higher K for international (more volatile)
ELO_K_FACTOR_WC = 60                    # extra weight for World Cup matches
ELO_HOME_ADVANTAGE = 100                # Elo points added for home team (0 for neutral venues)
ELO_NEUTRAL_ADVANTAGE = 0              # no home advantage at World Cup
PI_RATING_DECAY = 0.035                  # pi-rating per-game decay constant

CONFEDERATION_STRENGTH = {              # relative Elo starting offsets by confederation
    "UEFA": 50,
    "CONMEBOL": 40,
    "CONCACAF": 0,
    "AFC": -10,
    "CAF": -10,
    "OFC": -30,
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

# ── Feature modules active by default ─────────────────────────────────────────

DEFAULT_FEATURE_MODULES = [
    # ── Rating systems ────────────────────────────────────────────────────────
    "elo",              # basic Elo (kept for backward-compat feature coverage)
    "glicko2",          # Glicko-2: rating + deviation φ (Glickman 2001)
    "kalman_strength",  # EKF time-varying attack/defense + uncertainty (Koopman & Lit 2015)
    # ── Match history ─────────────────────────────────────────────────────────
    "form",             # rolling pts/goals/GD over 5/10/20 matches
    "h2h",              # head-to-head record
    "squad_strength",   # long-term attack/defence quality from intl results
    # ── Context / structure ───────────────────────────────────────────────────
    "confederation",    # UEFA/CONMEBOL/AFC/CAF/CONCACAF strength encoding
    "tournament_stage", # group stage vs knockout, pressure multiplier
    # ── External data ─────────────────────────────────────────────────────────
    "rankings",         # FIFA world rankings points (Dato-Futbol, 1992–2024)
    "squad_wc2026",     # WC 2026 club-tier / age features from official FIFA squad list
    "api_form",         # last-10-match form from API-Football (pre-cached for WC 2026 teams)
    "sofifa_ratings",   # EA FC 26 squad quality: overall, pace, shooting, passing, etc.
    "venue_wc2026",    # partial home advantage (MEX/USA/CAN), altitude, travel burden
    "injury",          # pre-match injuries/suspensions from API-Football cache
    "odds",            # bookmaker closing odds (qualifier history; WC 2026 pre-match TBD)
    "transfermarkt",   # squad market values from Transfermarkt.com (June 2026)
    "xg_form",         # rolling xG / xGA — excel (UEFA/AFC/CONMEBOL) + FBref (CAF/CONCACAF)
    # "xg",            # expected goals — requires StatsBomb/Opta event data
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
