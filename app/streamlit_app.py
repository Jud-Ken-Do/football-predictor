"""
FIFA World Cup 2026 — Prediction Dashboard
Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import warnings
warnings.filterwarnings("ignore")

import json
from collections import defaultdict
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson as sp_poisson

from football_predictor.data.wc2026 import GROUPS, GROUP_STAGE_SCHEDULE, normalise
from football_predictor.constants import DEFAULT_FEATURE_MODULES
from football_predictor.data.pipeline import build_feature_matrix
from football_predictor.models.wc_context import build_wc_context, apply_to_match

from predict_wc2026 import (
    train_model,
    predict_group_stage,
    load_actual_results,
    merge_actual_results,
    simulate_tournament,
    get_qualifiers,
    build_r32,
    ko_winner,
    R16_PAIRS,
    QF_PAIRS,
    SF_PAIRS,
)

ALL_TEAMS = [t for grp in GROUPS.values() for t in grp]
ROUNDS = ["r32", "r16", "qf", "sf", "final", "winner"]
ROUND_LABELS = ["R32", "R16", "QF", "SF", "Final", "Winner"]

# ── Colour palette ────────────────────────────────────────────────────────────
# 12 group colours — muted jewel tones, ~60 % lightness, ~45 % saturation
# Legible on dark bg without visual vibration
_GROUP_PALETTE = [
    "#d4855a",  # A  warm sienna
    "#6fa882",  # B  celadon
    "#5d8fbf",  # C  steel blue
    "#a66ca0",  # D  dusty plum
    "#c9a247",  # E  antique gold
    "#4e9e96",  # F  patina teal
    "#8878b8",  # G  lavender slate
    "#c47272",  # H  dusty rose
    "#5a9e7a",  # I  sea glass
    "#7088c0",  # J  periwinkle
    "#c48c5a",  # K  warm amber
    "#7a9648",  # L  sage olive
]

# Team A / Team B comparison colours
_COL_A = "#60a5fa"            # blue-400  (softer than #2196F3)
_COL_B = "#f87171"            # red-400   (softer than #F44336)
_COL_A_FILL = "rgba(96,165,250,0.15)"
_COL_B_FILL = "rgba(248,113,113,0.15)"

# Match outcome colours (group fixtures, score grids, etc.)
_COL_HOME = "#4ade80"         # green  — home team
_COL_DRAW = "#a1a1aa"         # zinc   — draw
_COL_AWAY = "#fb923c"         # orange — away team

# ── Flag emojis for all 48 WC teams ───────────────────────────────────────────
_FLAGS: dict[str, str] = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
    "Canada": "🇨🇦", "Switzerland": "🇨🇭", "Qatar": "🇶🇦", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Türkiye": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "IR Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cabo Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "Colombia": "🇨🇴", "Uzbekistan": "🇺🇿", "DR Congo": "🇨🇩",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

def flag(team: str) -> str:
    return f"{_FLAGS.get(team, '')} {team}".strip()

# Template name (from submission CSV) → canonical WC schedule name
_TEMPLATE_TO_SCHEDULE: dict[str, str] = {
    "Turkiye":    "Türkiye",
    "Cape Verde": "Cabo Verde",
    "Iran":       "IR Iran",
    "Curacao":    "Curaçao",
}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hero header ── */
.wc-header {
  background: linear-gradient(160deg, #071a0e 0%, #0f2d1a 45%, #091420 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 30px 40px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.wc-header::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 72% 50%, rgba(26,125,59,0.18) 0%, transparent 62%);
  pointer-events: none;
}
.wc-header h1 {
  margin: 0 0 5px;
  font-size: 1.85rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.02em;
  position: relative;
}
.wc-header p {
  margin: 0;
  font-size: 0.76rem;
  color: rgba(255,255,255,0.36);
  letter-spacing: 0.04em;
  position: relative;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 16px 18px;
  transition: border-color 0.18s, background 0.18s;
}
div[data-testid="metric-container"]:hover {
  background: rgba(26,125,59,0.07);
  border-color: rgba(26,125,59,0.4);
}
div[data-testid="metric-container"] label {
  font-size: 0.67rem !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(255,255,255,0.36) !important;
}
div[data-testid="stMetricValue"] {
  font-size: 1.4rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
}
div[data-testid="stMetricDelta"] {
  font-size: 0.73rem !important;
  color: rgba(255,255,255,0.38) !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px;
  background: rgba(255,255,255,0.02);
  border-radius: 10px;
  padding: 3px;
  border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px;
  padding: 7px 14px;
  font-size: 0.81rem;
  font-weight: 500;
  color: rgba(255,255,255,0.44);
  border: none !important;
  background: transparent !important;
  transition: color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: rgba(255,255,255,0.75); }
.stTabs [aria-selected="true"] {
  background: rgba(26,125,59,0.26) !important;
  color: #dff0e5 !important;
  font-weight: 600 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #070d09;
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(135deg, #155e2c, #1a7d3b);
  border: none;
  border-radius: 9px;
  font-weight: 600;
  font-size: 0.85rem;
  color: #fff;
  transition: opacity 0.18s, transform 0.1s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  opacity: 0.88;
  transform: translateY(-1px);
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid rgba(255,255,255,0.07) !important;
  margin: 20px 0 !important;
}

/* ── Dataframes ── */
div[data-testid="stDataFrame"] {
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Section labels ── */
.section-label {
  font-size: 0.67rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.11em;
  color: rgba(255,255,255,0.32);
  margin-bottom: 10px;
}

/* ── Stat comparison row ── */
.stat-row {
  display: flex; align-items: center;
  padding: 8px 12px;
  border-radius: 7px;
  font-size: 0.86rem;
}
.stat-row:nth-child(odd) { background: rgba(255,255,255,0.024); }
.stat-label { flex: 2; color: rgba(255,255,255,0.4); font-size: 0.81rem; }
.stat-val   { flex: 1; font-weight: 600; text-align: center; }

/* ── H2H bar ── */
.h2h-bar-wrap {
  border-radius: 8px; overflow: hidden;
  height: 38px; display: flex;
  margin-top: 8px;
  border: 1px solid rgba(255,255,255,0.07);
}
.h2h-home {
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.82rem; color: #0e1117;
}
.h2h-draw {
  background: rgba(161,161,170,0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.76rem; color: rgba(255,255,255,0.6);
}
.h2h-away {
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.82rem; color: #0e1117;
}

/* ── Expander ── */
details summary { font-size: 0.84rem; font-weight: 500; }

/* ── Info / warning boxes ── */
div[data-testid="stAlert"] { border-radius: 9px !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="wc-header">
  <h1>⚽ FIFA World Cup 2026</h1>
  <p>XGBoost &nbsp;·&nbsp; Bayesian Poisson &nbsp;·&nbsp; Kalman EKF &nbsp;·&nbsp; Context-Adaptive Ensemble &nbsp;·&nbsp; 50 k Monte Carlo</p>
</div>
""", unsafe_allow_html=True)


# ── Dark-theme Plotly wrapper ──────────────────────────────────────────────────
def _pc(fig, **kw) -> None:
    """Apply dark-theme defaults then render with st.plotly_chart."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, sans-serif",
        font_color="#a1a1aa",
        title_font=dict(size=13, color="#d4d4d8"),
        hoverlabel=dict(
            bgcolor="#161d2e",
            bordercolor="rgba(255,255,255,0.14)",
            font_family="Inter, sans-serif",
            font_color="#f4f4f5",
            font_size=13,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, **kw)

# ── Subprocess streaming helper ───────────────────────────────────────────────
def _stream(cmd: list[str], success_msg: str, on_success=None) -> None:
    import subprocess
    log = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(_ROOT), bufsize=1,
    )
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            lines.append(line)
            log.code("\n".join(lines[-30:]), language=None)
    proc.wait()
    if proc.returncode == 0:
        log.empty()
        if on_success:
            on_success()
        st.success(success_msg)
    else:
        st.error("Failed — see output above.")
        log.code("\n".join(lines[-60:]), language=None)


# ── Helpers for result recording (mirrors update_wc2026.py logic) ──────────────
_RESULTS_FILE = _ROOT / "data" / "wc2026_actual_results.json"

def _load_recorded() -> list[dict]:
    if _RESULTS_FILE.exists():
        return json.loads(_RESULTS_FILE.read_text())
    return []

def _save_recorded(results: list[dict]) -> None:
    _RESULTS_FILE.parent.mkdir(exist_ok=True)
    _RESULTS_FILE.write_text(json.dumps(results, indent=2))

def _record_result(home: str, away: str, hg: int, ag: int) -> str:
    from football_predictor.data.wc2026 import GROUP_STAGE_SCHEDULE as _SCH, normalise as _n
    group = next(
        (m["group"] for m in _SCH
         if _n(m["home_team"]) == _n(home) and _n(m["away_team"]) == _n(away)),
        "",
    )
    today = str(date.today())
    results = _load_recorded()
    results = [r for r in results if not (r["home_team"] == home and r["away_team"] == away)]
    results.append({"date": today, "home_team": home, "away_team": away,
                    "home_goals": hg, "away_goals": ag, "neutral": True, "group": group})
    results.sort(key=lambda r: (r["date"], r["home_team"]))
    _save_recorded(results)
    return group

def _delete_result(home: str, away: str) -> None:
    results = [r for r in _load_recorded()
               if not (r["home_team"] == home and r["away_team"] == away)]
    _save_recorded(results)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    # ── Simulation settings ───────────────────────────────────────────────────
    n_sims = st.select_slider(
        "Monte Carlo simulations",
        options=[5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    st.divider()

    # ── Refresh ───────────────────────────────────────────────────────────────
    _busy = st.session_state.get("running", False)
    if st.button("Refresh predictions", type="primary", use_container_width=True, disabled=_busy):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.predictions_ready = False
        st.session_state.running = True
        st.rerun()
    st.caption("Clears cache and re-trains from scratch (~2 min).")

    st.divider()

    # ── Record a result ───────────────────────────────────────────────────────
    with st.expander("📋 Record a result", expanded=False):
        st.caption("Save an actual WC 2026 match result.")
        _sched_opts = [
            f"{m['home_team']} vs {m['away_team']}"
            for m in GROUP_STAGE_SCHEDULE
        ]
        _sel = st.selectbox("Match", _sched_opts, key="rec_match", label_visibility="collapsed")
        _rec_home, _rec_away = _sel.split(" vs ", 1)
        _c1, _c2 = st.columns(2)
        _hg = _c1.number_input(f"{_rec_home[:10]}", min_value=0, max_value=20, value=0, key="rec_hg", step=1)
        _ag = _c2.number_input(f"{_rec_away[:10]}", min_value=0, max_value=20, value=0, key="rec_ag", step=1)
        if st.button("Save result", use_container_width=True, key="rec_save"):
            _grp = _record_result(_rec_home, _rec_away, int(_hg), int(_ag))
            st.success(f"Saved — Group {_grp}: {_rec_home} **{int(_hg)}–{int(_ag)}** {_rec_away}")
            st.cache_data.clear()

    # ── Recorded results list ─────────────────────────────────────────────────
    _recorded = _load_recorded()
    if _recorded:
        with st.expander(f"✅ Recorded results ({len(_recorded)})", expanded=False):
            for _r in sorted(_recorded, key=lambda x: x["date"], reverse=True):
                _col_r, _col_d = st.columns([4, 1])
                _col_r.markdown(
                    f"**{_r['home_team']} {_r['home_goals']}–{_r['away_goals']} {_r['away_team']}**  \n"
                    f"<span style='font-size:0.75rem;color:rgba(255,255,255,0.4)'>{_r['date']} · Grp {_r.get('group','?')}</span>",
                    unsafe_allow_html=True,
                )
                if _col_d.button("✕", key=f"del_{_r['home_team']}_{_r['away_team']}", help="Delete"):
                    _delete_result(_r["home_team"], _r["away_team"])
                    st.cache_data.clear()
                    st.rerun()

    st.divider()

    # ── Pipeline options ──────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline options", expanded=False):
        st.caption("These run in the background. Refresh after they complete.")

        if st.button("🔄 Re-fetch live data", use_container_width=True, key="btn_fetch",
                     help="Re-fetch injuries, form, squad values, odds from APIs (~1 min)"):
            _stream(
                ["python3.11", str(_ROOT / "scripts" / "pipeline.py"), "--fetch"],
                "Live data updated. Click Refresh predictions.",
            )

        if st.button("🎯 Re-tune XGBoost (Optuna)", use_container_width=True, key="btn_tune",
                     help="Run Optuna hyperparameter search (~3 min). Updates xgb_tuned_params.json."):
            _stream(
                ["python3.11", str(_ROOT / "scripts" / "pipeline.py"), "--tune"],
                "Tuning complete. Click Refresh predictions.",
            )

        if st.button("🔬 Refit with MCMC", use_container_width=True, key="btn_mcmc",
                     help="Use MCMC posterior for Bayesian Poisson instead of MAP (~5 min)."):
            st.info("MCMC cannot run inside the app — run in terminal:")
            st.code("python3.11 scripts/predict_wc2026.py --mcmc", language="bash")

        if st.button("📋 Regenerate submission", use_container_width=True, key="btn_sub",
                     help="Re-run the scoreline optimizer and update output.csv (~2 min)."):
            _stream(
                ["python3.11", str(_ROOT / "scripts" / "generate_submission.py")],
                "output.csv updated.",
                on_success=st.cache_data.clear,
            )

    st.divider()

    # ── Model stack info ──────────────────────────────────────────────────────
    with st.expander("🧠 Model stack", expanded=False):
        st.markdown(
            "1. XGBoost · 11 modules · ~110 features  \n"
            "2. Temperature Scaling (Guo 2017)  \n"
            "3. Bayesian Hierarchical Poisson MAP  \n"
            "4. Context-Adaptive Ensemble  \n"
            "5. WC 2026 post-processing  "
        )


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_models():
    return train_model(quiet=True)


@st.cache_data(show_spinner=False)
def _get_match_data():
    xgb, temp_cal, bp, ensemble, all_data, _ = _load_models()
    match_data = predict_group_stage(xgb, temp_cal, bp, ensemble, all_data, quiet=True)
    actual = load_actual_results()
    return merge_actual_results(match_data, actual), all_data


@st.cache_data(show_spinner=False)
def _get_pair_probs():
    xgb, temp_cal, bp, ensemble, all_data, _ = _load_models()
    pair_rows = [
        {
            "home_team": h, "away_team": a,
            "date": "2026-07-01", "neutral": True,
            "tournament": "FIFA World Cup", "match_weight": 1.5,
            "home_goals": 0, "away_goals": 0,
        }
        for h in ALL_TEAMS for a in ALL_TEAMS if h != a
    ]
    pair_df = pd.DataFrame(pair_rows).reset_index(drop=True)
    X_pairs, _ = build_feature_matrix(pair_df, DEFAULT_FEATURE_MODULES, context=all_data)
    ctx_pairs = build_wc_context(pair_df, all_data)

    xgb_p = temp_cal.transform(xgb.predict_proba(X_pairs))
    bp_rows = [
        bp.predict_proba(normalise(r.home_team), normalise(r.away_team), neutral=True)
        for r in pair_df.itertuples()
    ]
    bp_p = pd.DataFrame(bp_rows, columns=["home_win", "draw", "away_win"])
    ens_p = ensemble.predict_proba(
        xgb_p.reset_index(drop=True),
        bp_p,
        context_X=X_pairs.reset_index(drop=True),
    )

    prob_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
    lam_cache: dict[tuple[str, str], tuple[float, float]] = {}
    for i in range(len(pair_df)):
        row = pair_df.iloc[i]
        ctx_row = ctx_pairs.iloc[i].to_dict() if i < len(ctx_pairs) else {}
        lam_h, lam_a = bp.get_lambdas(normalise(row["home_team"]), normalise(row["away_team"]), neutral=True)
        p_h, p_d, p_a, lam_h_adj, lam_a_adj = apply_to_match(
            lam_h, lam_a,
            float(ens_p.iloc[i]["home_win"]),
            float(ens_p.iloc[i]["draw"]),
            float(ens_p.iloc[i]["away_win"]),
            ctx_row,
            home_team=row["home_team"],
            away_team=row["away_team"],
            rho=bp._rho,
        )
        prob_cache[(row["home_team"], row["away_team"])] = (p_h, p_d, p_a)
        lam_cache[(row["home_team"], row["away_team"])] = (lam_h_adj, lam_a_adj)
    return prob_cache, lam_cache


@st.cache_data(show_spinner=False)
def _get_mc_counts(n_sims: int):
    sys.path.insert(0, str(_ROOT / "scripts"))
    from generate_submission import _simulate_match as _sim_m

    match_data, _ = _get_match_data()
    pair_cache, _ = _get_pair_probs()

    rng = np.random.default_rng(42)
    group_order = sorted(GROUPS.keys())

    # ── Vectorized group stage: simulate all n_sims at once per group ──────────
    group_sim: dict[str, dict] = {}
    for grp in group_order:
        grp_teams = list(GROUPS[grp])
        n_teams = len(grp_teams)
        matches = [m for m in match_data if m["group"] == grp]
        n_matches = len(matches)
        match_pairs = [
            (grp_teams.index(m["home_team"]), grp_teams.index(m["away_team"]))
            for m in matches
        ]

        sim_h = np.zeros((n_sims, n_matches), dtype=np.int32)
        sim_a = np.zeros((n_sims, n_matches), dtype=np.int32)
        for i, m in enumerate(matches):
            if m.get("played"):
                sim_h[:, i] = int(m["actual_home_goals"])
                sim_a[:, i] = int(m["actual_away_goals"])
            else:
                sim_h[:, i], sim_a[:, i] = _sim_m(
                    m["bp_lam_home"], m["bp_lam_away"],
                    m["p_home"], m["p_draw"], m["p_away"],
                    n_sims, rng,
                    m.get("kalman_lam_h_log_std", 0.0),
                    m.get("kalman_lam_a_log_std", 0.0),
                )

        pts = np.zeros((n_sims, n_teams), dtype=np.int32)
        gd  = np.zeros((n_sims, n_teams), dtype=np.int32)
        gf  = np.zeros((n_sims, n_teams), dtype=np.int32)
        for m_idx, (hi, ai) in enumerate(match_pairs):
            h = sim_h[:, m_idx]; a = sim_a[:, m_idx]
            hw = h > a; dw = h == a; aw = a > h
            pts[:, hi] += 3 * hw + dw;  pts[:, ai] += 3 * aw + dw
            gd[:, hi]  += h - a;        gd[:, ai]  += a - h
            gf[:, hi]  += h;            gf[:, ai]  += a

        key = pts * 1_000_000 + (gd + 200) * 1_000 + gf
        ranks = np.argsort(-key, axis=1)   # (n_sims, n_teams)
        group_sim[grp] = {"teams": grp_teams, "ranks": ranks, "pts": pts, "gd": gd, "gf": gf}

    # ── KO stage: iterate over sims (group results already vectorized) ─────────
    def ko_predictor(home: str, away: str) -> tuple[float, float, float]:
        return pair_cache.get((home, away), (0.4, 0.2, 0.4))

    # reach_counts[team][round] = number of sims where that round is the FURTHEST reached
    reach_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for s in range(n_sims):
        standings: dict[str, list[dict]] = {}
        for grp in group_order:
            gs = group_sim[grp]
            r = gs["ranks"][s]
            standings[grp] = [
                {"team": gs["teams"][r[i]], "pts": int(gs["pts"][s, r[i]]),
                 "gd": int(gs["gd"][s, r[i]]), "gf": int(gs["gf"][s, r[i]])}
                for i in range(4)
            ]

        # Track furthest round reached (same logic as simulate_tournament)
        reached: dict[str, str] = {t: "group_stage" for grp in GROUPS.values() for t in grp}

        qualifiers = get_qualifiers(standings)
        for team in (list(qualifiers["1"].values()) + list(qualifiers["2"].values()) +
                     [t for t in qualifiers["3"].values() if t]):
            reached[team] = "r32"

        r32 = build_r32(qualifiers)
        r32w = [ko_winner(h, a, ko_predictor) for h, a in r32]
        for w in r32w: reached[w] = "r16"

        r16w = [ko_winner(r32w[i], r32w[j], ko_predictor) for i, j in R16_PAIRS]
        for w in r16w: reached[w] = "qf"

        qfw = [ko_winner(r16w[i], r16w[j], ko_predictor) for i, j in QF_PAIRS]
        for w in qfw: reached[w] = "sf"

        sfw = [ko_winner(qfw[i], qfw[j], ko_predictor) for i, j in SF_PAIRS]
        for w in sfw: reached[w] = "final"

        reached[ko_winner(sfw[0], sfw[1], ko_predictor)] = "winner"

        for team, rnd in reached.items():
            reach_counts[team][rnd] += 1

    # Cumulative: P(reach at least round R) = sum of counts for R and beyond
    cum_counts: dict[str, dict[str, float]] = {}
    for team in (t for grp in GROUPS.values() for t in grp):
        cnts = reach_counts.get(team, {})
        cum: dict[str, float] = {}
        running = 0
        for r in reversed(ROUNDS):
            running += cnts.get(r, 0)
            cum[r] = running / n_sims
        cum_counts[team] = cum
    return cum_counts


@st.cache_data
def _load_submission() -> dict[tuple[str, str], tuple[int, int]]:
    """Load optimised predicted scorelines from output.csv."""
    path = _ROOT / "output" / "output.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in df.iterrows():
        h = _TEMPLATE_TO_SCHEDULE.get(str(row["team1"]), str(row["team1"]))
        a = _TEMPLATE_TO_SCHEDULE.get(str(row["team2"]), str(row["team2"]))
        result[(h, a)] = (int(row["score1"]), int(row["score2"]))
    return result


# ── Session state ──────────────────────────────────────────────────────────────
if "predictions_ready" not in st.session_state:
    st.session_state.predictions_ready = False
if "running" not in st.session_state:
    st.session_state.running = False

if not st.session_state.predictions_ready:
    if not st.session_state.running:
        # ── Landing screen ─────────────────────────────────────────────────────
        st.markdown("""
        <div style="max-width:520px;margin:60px auto;text-align:center">
          <div style="font-size:3rem;margin-bottom:12px">⚽</div>
          <h2 style="margin-bottom:8px">Ready to predict</h2>
          <p style="color:rgba(255,255,255,0.5);margin-bottom:28px">
            Trains XGBoost + Bayesian Poisson + Ensemble on 15 years of international football,
            then simulates 10,000 tournament runs.<br><br>
            <strong style="color:rgba(255,255,255,0.7)">~2 minutes on first load,
            instant on subsequent visits.</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if st.button("Run predictions", type="primary", use_container_width=True):
                st.session_state.running = True
                st.rerun()
        st.stop()

    # ── Pipeline progress screen ────────────────────────────────────────────────
    import time
    import threading

    STEPS = [
        ("⚙️", "Training XGBoost + Bayesian Poisson + Ensemble", _load_models),
        ("📊", "Computing group stage predictions",               _get_match_data),
        ("🔮", "Pre-computing 48×48 team-pair matrix",            _get_pair_probs),
        ("🎲", f"Running {n_sims:,} Monte Carlo simulations",     lambda: _get_mc_counts(n_sims)),
    ]

    st.markdown(
        "<h3 style='margin-bottom:24px'>Running prediction pipeline</h3>",
        unsafe_allow_html=True,
    )
    pb         = st.progress(0.0)
    active_row = st.empty()
    done_rows  = st.empty()

    done = []
    for i, (icon, label, fn) in enumerate(STEPS):
        pb.progress(i / len(STEPS), text=f"Step {i + 1} of {len(STEPS)}")

        # Run the step in a background thread so the main thread can tick a timer
        finished  = threading.Event()
        caught    = [None]

        def _run(f=fn, ev=finished, exc=caught):
            try:
                f()
            except Exception as e:
                exc[0] = e
            finally:
                ev.set()

        threading.Thread(target=_run, daemon=True).start()

        t0 = time.time()
        while not finished.wait(timeout=0.4):
            elapsed = int(time.time() - t0)
            active_row.markdown(
                f"<p style='color:rgba(255,255,255,0.55);margin:12px 0 0'>"
                f"⏳ &nbsp; {icon} {label} &nbsp;&nbsp;"
                f"<code style='color:rgba(255,255,255,0.3)'>{elapsed}s</code></p>",
                unsafe_allow_html=True,
            )

        if caught[0]:
            st.session_state.running = False
            raise caught[0]

        elapsed = time.time() - t0
        done.append(f"✅ &nbsp; {icon} {label} &nbsp;&nbsp; `{elapsed:.1f}s`")
        done_rows.markdown("  \n".join(done))

    pb.progress(1.0, text="Complete!")
    active_row.empty()

    st.session_state.running = False
    st.session_state.predictions_ready = True

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("View predictions →", type="primary", use_container_width=True):
            st.rerun()
    st.stop()

try:
    match_data, all_data = _get_match_data()
    pair_probs, lam_probs = _get_pair_probs()
    mc_counts = _get_mc_counts(n_sims)
except Exception as exc:
    st.error(f"Failed to load predictions: {exc}")
    st.exception(exc)
    st.stop()

n_played = sum(1 for m in match_data if m.get("played"))
if n_played:
    st.info(f"{n_played} match{'es' if n_played > 1 else ''} played — results locked in.")

# ── Summary DataFrame ──────────────────────────────────────────────────────────
summary_rows = []
for team in ALL_TEAMS:
    c = mc_counts.get(team, {})
    grp = next(g for g, ts in GROUPS.items() if team in ts)
    summary_rows.append({
        "Team": team,
        "Group": grp,
        "R32": c.get("r32", 0),
        "R16": c.get("r16", 0),
        "QF": c.get("qf", 0),
        "SF": c.get("sf", 0),
        "Final": c.get("final", 0),
        "Winner": c.get("winner", 0),
    })
summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values("Winner", ascending=False)
    .reset_index(drop=True)
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_groups, tab_predictor, tab_bracket, tab_teams, \
tab_model, tab_data, tab_backtest, tab_submission = st.tabs([
    "🏆 Overview", "🗂 Groups", "⚡ Match", "🏅 Bracket", "🌍 Teams",
    "🤖 Model", "📊 Data", "📈 Backtest", "📋 Submission",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Tournament Favorites")

    # Top-6 metric cards
    top6 = summary_df.head(6)
    cols = st.columns(6)
    for i, (_, row) in enumerate(top6.iterrows()):
        with cols[i]:
            st.metric(
                label=flag(row['Team']),
                value=f"{row['Winner']:.1%}",
                delta=f"Final {row['Final']:.1%}",
            )

    st.divider()

    view_col, _ = st.columns([2, 5])
    with view_col:
        view = st.radio("View", ["Treemap", "Bar chart", "Table"],
                        horizontal=True, label_visibility="collapsed")

    if view == "Treemap":
        tree_df = summary_df.copy()
        tree_df["label"] = tree_df.apply(
            lambda r: f"{_FLAGS.get(r['Team'], '')} {r['Team']}<br>{r['Winner']:.1%}", axis=1
        )
        fig_tree = px.treemap(
            tree_df,
            path=[px.Constant("WC 2026"), "Group", "label"],
            values="Winner",
            color="Winner",
            color_continuous_scale=[(0, "#0d2b17"), (0.4, "#1a5c30"), (0.7, "#1a7d3b"), (1, "#4ade80")],
            range_color=[0, tree_df["Winner"].max()],
            custom_data=["Group", "Winner", "Final", "SF"],
        )
        fig_tree.update_traces(
            texttemplate="%{label}",
            textfont_size=13,
            marker_line_width=2,
            marker_line_color="#0e1117",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Group: %{customdata[0]}<br>"
                "Win: <b>%{customdata[1]:.1%}</b><br>"
                "Final: %{customdata[2]:.1%}<br>"
                "Semi: %{customdata[3]:.1%}"
                "<extra></extra>"
            ),
        )
        fig_tree.update_layout(
            height=620,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title="Win %",
                tickformat=".0%",
                thickness=10,
                len=0.5,
                x=1.01,
            ),
        )
        _pc(fig_tree, use_container_width=True)

    elif view == "Bar chart":
        top20 = summary_df.head(20).copy()
        top20["Team"] = top20["Team"].apply(flag)
        fig_bar = px.bar(
            top20,
            x="Winner", y="Team", orientation="h",
            color="Winner",
            color_continuous_scale=[(0, "#0d4f26"), (1, "#1a7d3b")],
            text=top20["Winner"].map(lambda x: f"{x:.1%}"),
            title="Top 20 — Win Probability",
        )
        fig_bar.update_layout(
            height=560, xaxis_tickformat=".0%",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=0, r=60, t=40, b=0),
        )
        fig_bar.update_traces(textposition="outside")
        _pc(fig_bar, use_container_width=True)

    else:
        display = summary_df.copy()
        display["Team"] = display["Team"].apply(flag)
        for col in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
            display[col] = display[col].map(lambda x: f"{x:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True, height=560)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GROUPS
# ══════════════════════════════════════════════════════════════════════════════
with tab_groups:
    st.subheader("Group Stage")
    _submission_scores = _load_submission()

    # Group selector as tab row
    grp_tabs = st.tabs(sorted(GROUPS.keys()))
    for grp_tab, grp in zip(grp_tabs, sorted(GROUPS.keys())):
        with grp_tab:
            group_matches = [m for m in match_data if m["group"] == grp]

            # Fixtures with stacked bars
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown(f"**Group {grp} Fixtures**")
                for m in group_matches:
                    played = m.get("played", False)
                    h, a = m["home_team"], m["away_team"]
                    ph, pd_, pa = m["p_home"], m["p_draw"], m["p_away"]
                    pred = _submission_scores.get((h, a))

                    if played:
                        hg, ag = int(m["actual_home_goals"]), int(m["actual_away_goals"])
                        result_col = _COL_HOME if hg > ag else (_COL_DRAW if hg == ag else _COL_AWAY)
                        pred_str = f"&nbsp;&nbsp;<span style='font-size:0.73rem;color:rgba(255,255,255,0.3)'>predicted {pred[0]}–{pred[1]}</span>" if pred else ""
                        st.markdown(
                            f"<div style='padding:8px 0 4px'>"
                            f"<span style='font-size:0.82rem;color:rgba(255,255,255,0.45)'>{flag(h)} vs {flag(a)}</span>"
                            f"&nbsp;&nbsp;"
                            f"<span style='font-size:1.1rem;font-weight:700;color:{result_col}'>{hg}–{ag}</span>"
                            f"{pred_str}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        pred_str = f"&nbsp;&nbsp;<span style='color:rgba(255,255,255,0.35)'>📋 {pred[0]}–{pred[1]}</span>" if pred else ""
                        # HTML probability bar — no Plotly, no "undefined" tick label
                        h_col = _COL_HOME if ph >= pa else "rgba(74,222,128,0.35)"
                        a_col = _COL_AWAY if pa > ph else "rgba(251,146,60,0.35)"
                        st.markdown(
                            f"<div style='margin-bottom:2px;font-size:0.8rem;color:rgba(255,255,255,0.5)'>"
                            f"{flag(h)} vs {flag(a)}{pred_str}</div>"
                            f"<div style='display:flex;height:32px;border-radius:6px;overflow:hidden;"
                            f"border:1px solid rgba(255,255,255,0.07);margin-bottom:10px'>"
                            f"<div style='width:{ph*100:.1f}%;background:{h_col};display:flex;"
                            f"align-items:center;justify-content:center;font-size:0.78rem;"
                            f"font-weight:600;color:#0e1117'>{flag(h)}&nbsp;{ph:.0%}</div>"
                            f"<div style='width:{pd_*100:.1f}%;background:rgba(161,161,170,0.25);display:flex;"
                            f"align-items:center;justify-content:center;font-size:0.72rem;"
                            f"color:rgba(255,255,255,0.55)'>D&nbsp;{pd_:.0%}</div>"
                            f"<div style='width:{pa*100:.1f}%;background:{a_col};display:flex;"
                            f"align-items:center;justify-content:center;font-size:0.78rem;"
                            f"font-weight:600;color:#0e1117'>{pa:.0%}&nbsp;{flag(a)}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            with col_right:
                st.markdown(f"**Expected Standings — Group {grp}**")
                exp_pts: dict[str, float] = defaultdict(float)
                for m in group_matches:
                    exp_pts[m["home_team"]] += 3 * m["p_home"] + m["p_draw"]
                    exp_pts[m["away_team"]] += 3 * m["p_away"] + m["p_draw"]

                teams_grp = GROUPS[grp]
                standings_rows = []
                for rank, t in enumerate(
                    sorted(teams_grp, key=lambda t: -exp_pts[t]), start=1
                ):
                    c = mc_counts.get(t, {})
                    standings_rows.append({
                        "#": rank,
                        "Team": t,
                        "Exp pts": f"{exp_pts[t]:.1f}",
                        "R32 %": f"{c.get('r32', 0):.0%}",
                        "Win %": f"{c.get('winner', 0):.1%}",
                    })
                st.dataframe(
                    pd.DataFrame(standings_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                # Win probability mini-chart for the group
                grp_win_data = pd.DataFrame([
                    {"Team": t, "Win %": mc_counts.get(t, {}).get("winner", 0)}
                    for t in teams_grp
                ]).sort_values("Win %", ascending=True)
                fig_grp = px.bar(
                    grp_win_data, x="Win %", y="Team",
                    orientation="h",
                    color_discrete_sequence=["#1a7d3b"],
                    title="Win probability",
                )
                fig_grp.update_layout(
                    height=180, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_tickformat=".1%", showlegend=False,
                )
                _pc(fig_grp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predictor:
    st.subheader("Match Predictor")
    st.caption("Pick any two WC 2026 teams for a detailed prediction.")

    col_h, col_a = st.columns(2)
    with col_h:
        home_team = st.selectbox(
            "Home team", ALL_TEAMS,
            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0,
            key="pred_home",
        )
    with col_a:
        away_opts = [t for t in ALL_TEAMS if t != home_team]
        away_default = away_opts.index("France") if "France" in away_opts else 0
        away_team = st.selectbox("Away team", away_opts, index=away_default, key="pred_away")

    probs = pair_probs.get((home_team, away_team))
    if probs is None:
        st.warning("No prediction available for this pair.")
    else:
        p_h, p_d, p_a = probs

        # Fixture info if scheduled
        fixture = next(
            (f for f in GROUP_STAGE_SCHEDULE
             if f["home_team"] == home_team and f["away_team"] == away_team),
            None,
        )
        if fixture:
            st.caption(f"Scheduled: {fixture['date']}  ·  {fixture.get('venue', 'Neutral')}  ·  Group {fixture['group']}")

        # Outcome probabilities bar
        fig_prob = go.Figure()
        for label, prob, colour in [
            (home_team, p_h, _COL_HOME),
            ("Draw", p_d, _COL_DRAW),
            (away_team, p_a, _COL_AWAY),
        ]:
            fig_prob.add_trace(go.Bar(
                name=label, x=[prob], y=[""],
                orientation="h",
                marker_color=colour,
                text=f"{label}  {prob:.1%}",
                textposition="inside",
                insidetextanchor="middle",
                width=0.6,
            ))
        fig_prob.update_layout(
            barmode="stack",
            height=110,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showticklabels=False, range=[0, 1]),
            yaxis=dict(showticklabels=False),
            showlegend=False,
        )
        _pc(fig_prob, use_container_width=True)

        # xG + favourite callout — use post-processed lambdas (venue + market adjusted)
        lam_h, lam_a = lam_probs.get((home_team, away_team), (1.2, 0.9))

        c1, c2, c3 = st.columns(3)
        fav = home_team if p_h > p_a else (away_team if p_a > p_h else "Draw")
        fav_p = max(p_h, p_a, p_d)
        c1.metric(f"{home_team} xG", f"{lam_h:.2f}")
        c2.metric("Favourite", fav, f"{fav_p:.1%}")
        c3.metric(f"{away_team} xG", f"{lam_a:.2f}")

        st.divider()

        col_score, col_adv = st.columns([1, 1])

        with col_score:
            st.markdown("**Score probability grid**")
            max_g = 5
            z = np.zeros((max_g + 1, max_g + 1))
            for hg in range(max_g + 1):
                for ag in range(max_g + 1):
                    z[hg][ag] = sp_poisson.pmf(hg, lam_h) * sp_poisson.pmf(ag, lam_a)

            fig_score = go.Figure(go.Heatmap(
                z=z,
                x=[str(i) for i in range(max_g + 1)],
                y=[str(i) for i in range(max_g + 1)],
                colorscale=[(0, "#0d1f12"), (0.4, "#1a5c30"), (1, "#4ade80")],
                text=[[f"{z[r][c]:.1%}" for c in range(max_g + 1)] for r in range(max_g + 1)],
                texttemplate="%{text}",
                textfont=dict(size=11),
                showscale=False,
                hovertemplate=f"{home_team} %{{y}} – %{{x}} {away_team}<br><b>%{{text}}</b><extra></extra>",
            ))
            fig_score.update_layout(
                xaxis=dict(title=f"{away_team} goals", tickfont=dict(size=11)),
                yaxis=dict(title=f"{home_team} goals", tickfont=dict(size=11)),
                height=320,
                margin=dict(l=50, r=10, t=10, b=40),
            )
            _pc(fig_score, use_container_width=True)

            # Top 5 scorelines
            scores = sorted(
                [(hg, ag, sp_poisson.pmf(hg, lam_h) * sp_poisson.pmf(ag, lam_a))
                 for hg in range(6) for ag in range(6)],
                key=lambda x: -x[2],
            )[:5]
            st.markdown("**Most likely scorelines**")
            for hg, ag, prob in scores:
                st.write(f"  `{hg}–{ag}`   {prob:.1%}")

        with col_adv:
            st.markdown("**Tournament advancement**")
            adv_rows = [
                {
                    "Round": ROUND_LABELS[i],
                    home_team: mc_counts.get(home_team, {}).get(ROUNDS[i], 0),
                    away_team: mc_counts.get(away_team, {}).get(ROUNDS[i], 0),
                }
                for i in range(len(ROUNDS))
            ]
            adv_df = pd.DataFrame(adv_rows)
            fig_adv = px.line(
                adv_df.melt(id_vars="Round", var_name="Team", value_name="Probability"),
                x="Round", y="Probability", color="Team", markers=True,
                color_discrete_map={home_team: _COL_A, away_team: _COL_B},
            )
            fig_adv.update_layout(
                yaxis_tickformat=".0%",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title="",
            )
            _pc(fig_adv, use_container_width=True)

            # H2H recent form
            st.markdown("**Recent form**")
            for team in [home_team, away_team]:
                mask = (all_data["home_team"] == team) | (all_data["away_team"] == team)
                recent = all_data[mask].sort_values("date").tail(5)
                form_str = ""
                for _, r in recent.iterrows():
                    is_home = r["home_team"] == team
                    hg, ag = int(r["home_goals"]), int(r["away_goals"])
                    gf, ga = (hg, ag) if is_home else (ag, hg)
                    form_str += "W" if gf > ga else ("D" if gf == ga else "L")
                colour = _COL_A if team == home_team else _COL_B
                st.markdown(
                    f"<span style='color:{colour}'><b>{team}</b></span>  "
                    f"`{'  '.join(form_str)}`",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BRACKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_bracket:
    st.subheader("Tournament Bracket — Advancement Probabilities")
    st.caption(f"{n_sims:,} Monte Carlo simulations")

    # Heatmap: teams × rounds
    sorted_teams = summary_df["Team"].tolist()
    z_heat = [
        [mc_counts.get(t, {}).get(r, 0) for r in ROUNDS]
        for t in sorted_teams
    ]
    text_heat = [
        [f"{mc_counts.get(t, {}).get(r, 0):.1%}" for r in ROUNDS]
        for t in sorted_teams
    ]

    _cs = [(0, "#0d1f12"), (0.15, "#0f3a1e"), (0.5, "#1a7d3b"), (1, "#4ade80")]
    fig_heat = go.Figure(go.Heatmap(
        z=z_heat,
        x=ROUND_LABELS,
        y=[flag(t) for t in sorted_teams],
        colorscale=_cs,
        text=text_heat,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Prob", tickformat=".0%", thickness=10, len=0.4),
        hovertemplate="<b>%{y}</b><br>%{x}: <b>%{text}</b><extra></extra>",
        zmin=0, zmax=1,
    ))
    fig_heat.update_layout(
        height=1520,
        xaxis=dict(side="top", tickfont=dict(size=12, color="#d4d4d8")),
        margin=dict(l=180, r=20, t=50, b=10),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    _pc(fig_heat, use_container_width=True)

    # Narrow table view
    with st.expander("Table view", expanded=True):
        table_df = summary_df[["Team", "Group", "R32", "R16", "QF", "SF", "Final", "Winner"]].copy()
        for col in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
            table_df[col] = table_df[col].map(lambda x: f"{x:.1%}")
        st.dataframe(table_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_injuries() -> dict:
    path = _ROOT / "data" / "wc2026_player_injuries.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def _load_transfermarkt() -> dict[str, int]:
    path = _ROOT / "data" / "transfermarkt_wc2026.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, (int, float))}


@st.cache_data
def _load_fc26() -> dict[str, dict]:
    try:
        from football_predictor.features.sofifa_ratings import _load_fc26 as _fc
        return _fc()
    except Exception:
        return {}


@st.cache_data
def _load_api_form() -> dict:
    path = _ROOT / "data" / "api_form_cache.json"
    return json.loads(path.read_text()) if path.exists() else {}


# Canonical WC name → FC26/transfermarkt name (best-effort)
_WC_TO_FC26: dict[str, str] = {
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
    "Czechia": "Czechia",
    "Türkiye": "Türkiye",
    "Cabo Verde": "Cabo Verde",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Curaçao": "Curacao",
    "United States": "United States",
}


def _fc26_key(team: str) -> str:
    return _WC_TO_FC26.get(team, team)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TEAMS
# ══════════════════════════════════════════════════════════════════════════════
with tab_teams:
    injuries_raw  = _load_injuries()
    transfermarkt = _load_transfermarkt()
    fc26_data     = _load_fc26()
    api_form_raw  = _load_api_form()

    WC_START = date(2026, 6, 11)
    RADAR_CATS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physic"]
    RADAR_KEYS = ["fc26_pace", "fc26_shooting", "fc26_passing",
                  "fc26_dribbling", "fc26_defending", "fc26_physic"]

    def _team_fc26(team: str) -> dict:
        return fc26_data.get(_fc26_key(team), {})

    def _team_squad_val(team: str) -> int | None:
        key = next((k for k in transfermarkt if k.lower() == team.lower()), None)
        return transfermarkt.get(key or team)

    def _team_form_rows(team: str) -> list[dict]:
        entry = api_form_raw.get(team, {})
        rows = []
        for fix in sorted(entry.get("fixtures", []),
                          key=lambda f: f["fixture"]["date"], reverse=True)[:10]:
            home_name = fix["teams"]["home"]["name"]
            away_name = fix["teams"]["away"]["name"]
            hg = fix["goals"]["home"] or 0
            ag = fix["goals"]["away"] or 0
            is_home = home_name.lower() == team.lower()
            gf, ga = (hg, ag) if is_home else (ag, hg)
            opp = away_name if is_home else home_name
            rows.append({
                "Date":     fix["fixture"]["date"][:10],
                "H/A":      "H" if is_home else "A",
                "Opponent": opp,
                "Score":    f"{gf}–{ga}",
                "Result":   "W" if gf > ga else ("D" if gf == ga else "L"),
            })
        return rows

    def _team_injury_rows(team: str) -> list[dict]:
        players = injuries_raw.get(team) or injuries_raw.get(normalise(team), [])
        rows = []
        for p in sorted(players, key=lambda x: -(x.get("market_value_m") or 0)):
            name   = p.get("name") or "—"
            pos    = p.get("position") or "—"
            mv     = p.get("market_value_m")
            mv_str = f"€{mv:.0f}M" if mv else "—"
            injuries_list = sorted(
                p.get("injuries", []),
                key=lambda i: i.get("until") or "", reverse=True,
            )
            if injuries_list:
                latest    = injuries_list[0]
                until_str = latest.get("until") or ""
                inj_type  = latest.get("type") or "—"
                games     = latest.get("games_missed") or 0
                try:
                    until_dt = date.fromisoformat(until_str)
                    if until_dt >= WC_START:
                        status = "ACTIVE"
                    elif (WC_START - until_dt).days <= 45:
                        status = "Recent"
                    else:
                        status = "Recovered"
                except ValueError:
                    status = "—"
            else:
                until_str, inj_type, games, status = "—", "—", 0, "Fit"
            rows.append({
                "Player": name, "Pos": pos, "Value": mv_str,
                "Status": status, "Injury": inj_type,
                "Until": until_str, "Games": games,
            })
        return rows

    def _render_form_table(rows: list[dict]) -> None:
        if not rows:
            st.info("No recent form data.")
            return
        df = pd.DataFrame(rows)
        def _cr(val: str) -> str:
            return {
                "W": "background-color:#1a4d2e; color:#4caf50",
                "D": "background-color:#2c2c2c; color:#aaa",
                "L": "background-color:#4d1a1a; color:#f44336",
            }.get(val, "")
        st.dataframe(df.style.map(_cr, subset=["Result"]),
                     use_container_width=True, hide_index=True, height=320)

    def _render_injury_table(rows: list[dict]) -> None:
        if not rows:
            st.info("No injury data available.")
            return
        df = pd.DataFrame(rows)
        def _cs(val: str) -> str:
            return {
                "ACTIVE":    "background-color:#4d1a1a; color:#f44336; font-weight:bold",
                "Recent":    "background-color:#4d3a00; color:#ffb300",
                "Fit":       "background-color:#1a4d2e; color:#4caf50",
                "Recovered": "",
            }.get(val, "")
        st.dataframe(df.style.map(_cs, subset=["Status"]),
                     use_container_width=True, hide_index=True, height=420)
        active = sum(1 for r in rows if r["Status"] == "ACTIVE")
        if active:
            st.warning(f"{active} player(s) with active injury/suspension at WC start.")
        st.caption("Player injury data from API-Football — squad mapping may have inaccuracies for some teams.")

    # ── Team selectors ────────────────────────────────────────────────────────
    sorted_teams = sorted(ALL_TEAMS)
    flagged_teams = [flag(t) for t in sorted_teams]

    def _unflag(s: str) -> str:
        for t in sorted_teams:
            if s.endswith(t):
                return t
        return s

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        team_a_f = st.selectbox("Team A", flagged_teams, key="teams_a",
                                index=sorted_teams.index("Brazil"))
        team_a = _unflag(team_a_f)
    with sel_col2:
        team_b_f = st.selectbox("Team B", flagged_teams, key="teams_b",
                                index=sorted_teams.index("France"))
        team_b = _unflag(team_b_f)

    # ── Stats comparison row ──────────────────────────────────────────────────
    st.divider()
    mc_a  = mc_counts.get(team_a, {})
    mc_b  = mc_counts.get(team_b, {})
    fc_a  = _team_fc26(team_a)
    fc_b  = _team_fc26(team_b)
    grp_a = next(g for g, ts in GROUPS.items() if team_a in ts)
    grp_b = next(g for g, ts in GROUPS.items() if team_b in ts)
    sv_a  = _team_squad_val(team_a)
    sv_b  = _team_squad_val(team_b)

    stat_labels = ["Group", "Win %", "Final %", "SoFIFA overall", "Squad value"]
    stat_a = [
        grp_a,
        f"{mc_a.get('winner', 0):.1%}",
        f"{mc_a.get('final', 0):.1%}",
        f"{fc_a.get('fc26_overall', 0):.0f}" if fc_a else "N/A",
        f"€{sv_a}M" if sv_a else "N/A",
    ]
    stat_b = [
        grp_b,
        f"{mc_b.get('winner', 0):.1%}",
        f"{mc_b.get('final', 0):.1%}",
        f"{fc_b.get('fc26_overall', 0):.0f}" if fc_b else "N/A",
        f"€{sv_b}M" if sv_b else "N/A",
    ]

    # Stat comparison as styled HTML table
    rows_html = "".join(
        f'<div class="stat-row"><span class="stat-label">{lbl}</span>'
        f'<span class="stat-val">{va}</span><span class="stat-val">{vb}</span></div>'
        for lbl, va, vb in zip(stat_labels, stat_a, stat_b)
    )
    st.markdown(
        f'<div class="stat-row" style="padding-bottom:6px">'
        f'<span class="stat-label"></span>'
        f'<span class="stat-val" style="color:{_COL_A}">{flag(team_a)}</span>'
        f'<span class="stat-val" style="color:{_COL_B}">{flag(team_b)}</span>'
        f'</div>{rows_html}',
        unsafe_allow_html=True,
    )

    # H2H bar
    p2 = pair_probs.get((team_a, team_b))
    if p2:
        ph, pd_, pa = p2
        st.markdown(
            f'<div style="margin-top:16px"><p class="section-label">Head-to-head (neutral)</p>'
            f'<div class="h2h-bar-wrap">'
            f'<div class="h2h-home" style="background:{_COL_HOME};width:{ph*100:.1f}%">{ph:.0%}</div>'
            f'<div class="h2h-draw" style="width:{pd_*100:.1f}%">Draw {pd_:.0%}</div>'
            f'<div class="h2h-away" style="background:{_COL_AWAY};width:{pa*100:.1f}%">{pa:.0%}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── SoFIFA radar (overlaid) ───────────────────────────────────────────────
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**SoFIFA Ratings (EA FC 26)**")
        fig_radar = go.Figure()
        for team, fc, colour, fill in [
            (flag(team_a), fc_a, _COL_A, _COL_A_FILL),
            (flag(team_b), fc_b, _COL_B, _COL_B_FILL),
        ]:
            if fc:
                vals = [fc.get(k, 0) for k in RADAR_KEYS]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=RADAR_CATS + [RADAR_CATS[0]],
                    fill="toself",
                    line_color=colour,
                    fillcolor=fill,
                    name=team,
                ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[40, 90])),
            height=360,
            margin=dict(l=30, r=30, t=10, b=10),
            legend=dict(orientation="h", y=-0.05),
        )
        _pc(fig_radar, use_container_width=True)

        # Per-position bar comparison
        pos_keys = ["fc26_gk_overall", "fc26_def_overall", "fc26_mid_overall", "fc26_fwd_overall"]
        pos_labels = ["GK", "DEF", "MID", "FWD"]
        if fc_a or fc_b:
            fa, fb = flag(team_a), flag(team_b)
            pos_df = pd.DataFrame([
                {"Pos": pos, fa: (fc_a or {}).get(k, 0), fb: (fc_b or {}).get(k, 0)}
                for pos, k in zip(pos_labels, pos_keys)
            ])
            fig_pos = px.bar(
                pos_df.melt(id_vars="Pos", var_name="Team", value_name="Rating"),
                x="Pos", y="Rating", color="Team", barmode="group",
                color_discrete_map={fa: _COL_A, fb: _COL_B},
                range_y=[50, 90],
            )
            fig_pos.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10),
                legend_title="", showlegend=True,
            )
            _pc(fig_pos, use_container_width=True)

    # ── Tournament advancement comparison ─────────────────────────────────────
    with right_col:
        st.markdown("**Tournament Advancement**")
        fa, fb = flag(team_a), flag(team_b)
        adv_rows = [
            {
                "Round": ROUND_LABELS[i],
                fa: mc_a.get(ROUNDS[i], 0),
                fb: mc_b.get(ROUNDS[i], 0),
            }
            for i in range(len(ROUNDS))
        ]
        adv_df = pd.DataFrame(adv_rows)
        fig_adv = px.line(
            adv_df.melt(id_vars="Round", var_name="Team", value_name="Probability"),
            x="Round", y="Probability", color="Team", markers=True,
            color_discrete_map={fa: _COL_A, fb: _COL_B},
        )
        fig_adv.update_layout(
            yaxis_tickformat=".0%", height=260,
            margin=dict(l=10, r=10, t=10, b=10), legend_title="",
        )
        _pc(fig_adv, use_container_width=True)

        # Win probability comparison bar
        win_df = pd.DataFrame([
            {"Team": flag(team_a), "Probability": mc_a.get("winner", 0)},
            {"Team": flag(team_b), "Probability": mc_b.get("winner", 0)},
        ])
        fig_win = px.bar(
            win_df, x="Team", y="Probability",
            color="Team",
            color_discrete_map={flag(team_a): _COL_A, flag(team_b): _COL_B},
            text=win_df["Probability"].map(lambda x: f"{x:.1%}"),
        )
        fig_win.update_layout(
            height=220, yaxis_tickformat=".0%",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, title="Win probability",
        )
        _pc(fig_win, use_container_width=True)

    st.divider()

    # ── Recent form (side by side) ────────────────────────────────────────────
    st.markdown("**Recent Form**")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.caption(team_a)
        _render_form_table(_team_form_rows(team_a))
    with fc2:
        st.caption(team_b)
        _render_form_table(_team_form_rows(team_b))

    st.divider()

    # ── Injuries (side by side) ───────────────────────────────────────────────
    st.markdown("**Squad — Injuries & Suspensions**")
    ic1, ic2 = st.columns(2)
    with ic1:
        st.caption(team_a)
        _render_injury_table(_team_injury_rows(team_a))
    with ic2:
        st.caption(team_b)
        _render_injury_table(_team_injury_rows(team_b))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("Model Internals")

    xgb_m, temp_m, bp_m, ens_m, _, train_df_m = _load_models()

    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    from football_predictor.models.bayesian_poisson import BayesianPoissonModel as _BPM

    q  = KalmanStrengthFeatures.get_last_tuned_q()
    hl = _BPM.half_life_from_q(q)
    bp_p = bp_m._params or {}

    # ── Parameter cards ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Fitted Parameters</p>', unsafe_allow_html=True)
    pc = st.columns(8)
    param_items = [
        ("Temp T",        f"{temp_m.temperature:.3f}",        "XGBoost calibration scalar"),
        ("XGB weight",    f"{ens_m.xgb_weight:.2f}",          "Ensemble weight on XGBoost"),
        ("BP weight",     f"{ens_m.bp_weight:.2f}",           "Ensemble weight on BayesPoisson"),
        ("Kalman q",      f"{q:.4f}",                         "EM-tuned process noise (per year)"),
        ("BP half-life",  f"{hl}d",                           "BayesPoisson time decay"),
        ("Home adv",      f"{bp_p.get('home_adv', 0):.3f}",   "BP log-odds home advantage"),
        ("BP μ",          f"{bp_p.get('mu', 0):.3f}",         "BP baseline log goal rate"),
        ("DC ρ",          f"{bp_m._rho:.3f}",                 "Dixon-Coles low-score correction"),
    ]
    for col, (label, val, help_txt) in zip(pc, param_items):
        col.metric(label, val, help=help_txt)

    st.divider()

    left_m, right_m = st.columns(2)

    # ── XGBoost feature importance ────────────────────────────────────────────
    with left_m:
        st.markdown("**XGBoost Feature Importance (gain)**")
        if hasattr(xgb_m.model, "feature_importances_") and xgb_m._feature_names:
            fi_df = (
                pd.DataFrame({
                    "Feature": xgb_m._feature_names,
                    "Importance": xgb_m.model.feature_importances_,
                })
                .sort_values("Importance", ascending=False)
                .head(25)
            )
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=[(0, "#1a3a2a"), (1, "#1a7d3b")],
            )
            fig_fi.update_layout(
                height=620,
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Gain",
                margin=dict(l=0, r=20, t=10, b=10),
            )
            _pc(fig_fi, use_container_width=True)

            # XGB hyperparams
            xgb_hp = json.loads((_ROOT / "data" / "xgb_tuned_params.json").read_text()) \
                if (_ROOT / "data" / "xgb_tuned_params.json").exists() else {}
            if xgb_hp:
                st.markdown("**Tuned hyperparameters**")
                hp_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in xgb_hp.items()])
                st.dataframe(hp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importances not yet available — run the pipeline first.")

    # ── BayesPoisson team ratings ─────────────────────────────────────────────
    with right_m:
        st.markdown("**Bayesian Poisson — Team Attack vs Defence (WC 2026 squads)**")
        if bp_p:
            bp_rows = []
            for team in ALL_TEAMS:
                norm = normalise(team)
                att = bp_p.get(f"att_{norm}", 0.0)
                dff = bp_p.get(f"def_{norm}", 0.0)
                grp = next(g for g, ts in GROUPS.items() if team in ts)
                bp_rows.append({
                    "Team": flag(team), "Group": grp,
                    "Attack": att, "Defence": dff,
                    "Winner %": mc_counts.get(team, {}).get("winner", 0),
                })
            bp_df = pd.DataFrame(bp_rows)

            fig_bp = px.scatter(
                bp_df,
                x="Defence", y="Attack",
                color="Group",
                size="Winner %",
                size_max=30,
                text="Team",
                color_discrete_sequence=_GROUP_PALETTE,
                title="Attack = more goals scored  ·  Defence = fewer conceded (lower = better)",
            )
            fig_bp.update_traces(textposition="top center", textfont_size=10)
            fig_bp.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_bp.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_bp.update_layout(
                height=620,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Defence (lower = stronger defence)",
                yaxis_title="Attack (higher = stronger attack)",
            )
            _pc(fig_bp, use_container_width=True)
        else:
            st.info("BayesPoisson not yet fitted.")

    st.divider()

    # ── MC Convergence animation ──────────────────────────────────────────────
    st.markdown("**Monte Carlo Convergence**")
    st.caption("Shows how win % estimates stabilise as simulation count grows.")

    @st.cache_data(show_spinner=False)
    def _convergence_data():
        match_data_c, _ = _get_match_data()
        pair_cache_c, _ = _get_pair_probs()
        top_teams_c = summary_df.head(10)["Team"].tolist()

        def ko_p(h, a): return pair_cache_c.get((h, a), (0.4, 0.2, 0.4))

        checkpoints = [50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000]
        counts_c: dict[str, int] = defaultdict(int)
        rows_c = []
        sim_i = 0
        for cp in checkpoints:
            while sim_i < cp:
                reached = simulate_tournament(match_data_c, ko_p)
                for tm, rnd in reached.items():
                    if rnd == "winner" and tm in top_teams_c:
                        counts_c[tm] += 1
                sim_i += 1
            for tm in top_teams_c:
                rows_c.append({"Simulations": cp, "Team": flag(tm),
                               "Win %": counts_c[tm] / cp})
        return pd.DataFrame(rows_c)

    if st.button("Run convergence analysis", key="conv_btn"):
        with st.spinner("Running convergence analysis (10k sims)…"):
            conv_df = _convergence_data()
        fig_conv = px.line(
            conv_df, x="Simulations", y="Win %",
            color="Team", markers=True,
            log_x=True,
            title="Win probability convergence — top 10 teams",
            color_discrete_sequence=_GROUP_PALETTE,
        )
        fig_conv.update_layout(
            height=400, yaxis_tickformat=".1%",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        _pc(fig_conv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Data Quality & Coverage")

    _, all_data_d = _get_match_data()

    # ── Training set summary ──────────────────────────────────────────────────
    st.markdown('<p class="section-label">Training Dataset</p>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total matches", f"{len(all_data_d):,}")
    comp_mask = all_data_d["tournament"].str.lower().str.contains(
        "qualif|world cup|copa|euro|nations|africa|asian|gold cup", na=False
    )
    d2.metric("Competitive matches", f"{comp_mask.sum():,}")
    _date_min = pd.to_datetime(all_data_d["date"]).min()
    _date_max = pd.to_datetime(all_data_d["date"]).max()
    d3.metric("Date range", f"{_date_min.year}–{_date_max.year}")
    has_xg = all_data_d.get("xg_home", pd.Series(dtype=float)).notna().sum() \
        if "xg_home" in all_data_d.columns else 0
    d4.metric("Matches with xG", f"{has_xg:,}")

    st.divider()

    left_d, right_d = st.columns(2)

    # ── Competition type breakdown ────────────────────────────────────────────
    with left_d:
        st.markdown("**Match type breakdown**")
        type_counts = all_data_d["tournament"].fillna("Unknown")
        def _simplify(t: str) -> str:
            t = t.lower()
            if "world cup" in t and "qualif" not in t: return "World Cup"
            if "qualif" in t: return "Qualifier"
            if "euro" in t: return "Euro / Nations"
            if "copa" in t or "conmebol" in t: return "Copa América"
            if "africa" in t or "afcon" in t or "chan" in t: return "AFCON"
            if "asian" in t or "afc" in t: return "Asian Cup"
            if "gold" in t or "concacaf" in t: return "CONCACAF"
            if "friendly" in t or "international" in t: return "Friendly"
            return "Other"
        simplified = type_counts.apply(_simplify).value_counts().reset_index()
        simplified.columns = ["Type", "Count"]
        fig_type = px.pie(
            simplified, names="Type", values="Count",
            color_discrete_sequence=_GROUP_PALETTE,
            hole=0.4,
        )
        fig_type.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=10))
        _pc(fig_type, use_container_width=True)

    # ── Per-team coverage ─────────────────────────────────────────────────────
    with right_d:
        st.markdown("**Per-team match coverage (WC 2026 squads)**")
        cov_rows = []
        for team in ALL_TEAMS:
            norm = normalise(team)
            mask = (all_data_d["home_team"] == norm) | (all_data_d["away_team"] == norm)
            team_df = all_data_d[mask]
            comp_df  = team_df[team_df["tournament"].str.lower().str.contains(
                "qualif|world cup|copa|euro|nations|africa|asian|gold cup", na=False)]
            has_inj  = bool(injuries_raw.get(team))
            has_form = bool(api_form_raw.get(team, {}).get("fixtures"))
            cov_rows.append({
                "Team": flag(team),
                "All matches": len(team_df),
                "Competitive": len(comp_df),
                "Injuries": "Yes" if has_inj else "—",
                "API form":  "Yes" if has_form else "—",
            })
        cov_df = pd.DataFrame(cov_rows).sort_values("Competitive", ascending=False)
        st.dataframe(cov_df, use_container_width=True, hide_index=True, height=340)

    st.divider()

    # ── xG coverage heatmap by confederation ─────────────────────────────────
    st.markdown("**xG Data Availability by Confederation**")
    CONF_MAP = {
        "A": "CONCACAF", "B": "CONCACAF/UEFA", "C": "CONMEBOL/CAF",
        "D": "CONCACAF/CONMEBOL", "E": "UEFA/CAF", "F": "UEFA/AFC",
        "G": "UEFA/CAF", "H": "UEFA/CAF/CONMEBOL", "I": "UEFA/CAF",
        "J": "CONMEBOL/UEFA", "K": "UEFA/CONMEBOL", "L": "UEFA/CAF/CONCACAF",
    }
    xg_rows = []
    for team in ALL_TEAMS:
        norm = normalise(team)
        mask = (all_data_d["home_team"] == norm) | (all_data_d["away_team"] == norm)
        team_df = all_data_d[mask]
        grp = next(g for g, ts in GROUPS.items() if team in ts)
        if "xg_home" in team_df.columns:
            total   = len(team_df)
            with_xg = (team_df["xg_home"].notna() | team_df["xg_away"].notna()).sum() \
                if "xg_away" in team_df.columns else team_df["xg_home"].notna().sum()
            pct = with_xg / total if total else 0
        else:
            pct = 0
        xg_rows.append({"Team": flag(team), "Group": grp, "xG coverage": pct})

    xg_df = pd.DataFrame(xg_rows).sort_values(["Group", "xG coverage"], ascending=[True, False])
    fig_xg = px.bar(
        xg_df, x="Team", y="xG coverage", color="Group",
        color_discrete_sequence=_GROUP_PALETTE,
        title="xG data coverage per team (% of historical matches)",
    )
    fig_xg.update_layout(
        height=380, yaxis_tickformat=".0%",
        xaxis_tickangle=-45,
        margin=dict(l=0, r=0, t=40, b=80),
        legend_title="Group",
    )
    _pc(fig_xg, use_container_width=True)

    # ── API form data freshness ───────────────────────────────────────────────
    st.markdown("**API-Football Form Cache Freshness**")
    fresh_rows = []
    for team, entry in api_form_raw.items():
        fetched = entry.get("fetched_at", "")
        n_fix   = len(entry.get("fixtures", []))
        fresh_rows.append({"Team": team, "Fetched at": fetched[:19].replace("T", " "),
                           "Fixtures cached": n_fix})
    if fresh_rows:
        fresh_df = pd.DataFrame(fresh_rows).sort_values("Fetched at", ascending=False)
        st.dataframe(fresh_df, use_container_width=True, hide_index=True, height=260)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — BACKTEST / CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Backtest — Historical World Cup Evaluation")
    st.caption("Trained on pre-tournament data, evaluated on group stage (out-of-sample).")

    # ── Parse metrics files ───────────────────────────────────────────────────
    import re as _re

    def _parse_metrics(path: Path) -> dict:
        if not path.exists():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
        return out

    years = [2014, 2018, 2022]
    metrics_by_year = {y: _parse_metrics(_ROOT / "output" / f"backtest_{y}_metrics.txt") for y in years}

    # ── Metrics comparison table ──────────────────────────────────────────────
    key_metrics = ["log_loss", "log_loss_uniform", "brier_score", "brier_uniform",
                   "accuracy", "ece", "temperature", "ensemble_alpha", "n_matches"]
    labels      = ["Log-loss", "Log-loss (uniform)", "Brier score", "Brier (uniform)",
                   "Accuracy", "ECE", "Temp T", "Ensemble α", "N matches"]

    m_rows = []
    for label, key in zip(labels, key_metrics):
        row = {"Metric": label}
        for y in years:
            v = metrics_by_year[y].get(key)
            row[str(y)] = f"{v:.4f}" if isinstance(v, float) else (str(v) if v else "—")
        m_rows.append(row)

    st.markdown("**Key metrics across tournaments**")
    m_df = pd.DataFrame(m_rows)

    def _highlight_better(row):
        vals = []
        for y in ["2014", "2018", "2022"]:
            try:
                vals.append(float(row[y]))
            except (ValueError, KeyError):
                vals.append(None)
        styles = [""] * len(row)
        valid = [v for v in vals if v is not None]
        if not valid:
            return styles
        is_lower_better = any(k in row["Metric"].lower() for k in ["loss", "brier", "ece"])
        best = min(valid) if is_lower_better else max(valid)
        for i, (col, v) in enumerate(zip(["2014", "2018", "2022"], vals)):
            if v == best:
                styles[list(row.index).index(col)] = "background-color:#1a4d2e; color:#4caf50"
        return styles

    st.dataframe(m_df.style.apply(_highlight_better, axis=1),
                 use_container_width=True, hide_index=True)

    # ── Log-loss and Brier comparison bars ───────────────────────────────────
    st.divider()
    mc1, mc2 = st.columns(2)

    with mc1:
        ll_rows = []
        for y in years:
            m = metrics_by_year[y]
            if "log_loss" in m:
                ll_rows += [
                    {"Year": str(y), "Type": "Model", "Value": m["log_loss"]},
                    {"Year": str(y), "Type": "Uniform baseline", "Value": m.get("log_loss_uniform", 1.099)},
                ]
        if ll_rows:
            fig_ll = px.bar(
                pd.DataFrame(ll_rows), x="Year", y="Value", color="Type",
                barmode="group", title="Log-loss vs Uniform baseline",
                color_discrete_map={"Model": "#1a7d3b", "Uniform baseline": "#555"},
            )
            fig_ll.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                 yaxis_title="Log-loss (lower = better)", legend_title="")
            _pc(fig_ll, use_container_width=True)

    with mc2:
        br_rows = []
        for y in years:
            m = metrics_by_year[y]
            if "brier_score" in m:
                br_rows += [
                    {"Year": str(y), "Type": "Model", "Value": m["brier_score"]},
                    {"Year": str(y), "Type": "Uniform baseline", "Value": m.get("brier_uniform", 0.222)},
                ]
        if br_rows:
            fig_br = px.bar(
                pd.DataFrame(br_rows), x="Year", y="Value", color="Type",
                barmode="group", title="Brier score vs Uniform baseline",
                color_discrete_map={"Model": "#818cf8", "Uniform baseline": "#555"},
            )
            fig_br.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                 yaxis_title="Brier (lower = better)", legend_title="")
            _pc(fig_br, use_container_width=True)

    # ── Calibration plots ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Calibration Reliability Diagrams**")
    cal_cols = st.columns(3)
    for col, year in zip(cal_cols, years):
        png = _ROOT / "output" / f"backtest_{year}_calibration.png"
        with col:
            st.markdown(f"**WC {year}**")
            if png.exists():
                st.image(str(png), use_container_width=True)
            else:
                st.info(f"No calibration plot for {year}.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
with tab_submission:
    st.subheader("Submission")

    _opt_path = _ROOT / "output" / "output.csv"
    _raw_path = _ROOT / "output" / "output_raw.csv"

    if not _opt_path.exists():
        st.info("No submission found — run '📋 Regenerate submission' from the sidebar.")
    else:
        # ── Top bar ───────────────────────────────────────────────────────────
        tb_dl, tb_regen, tb_sims = st.columns([3, 3, 2])
        with tb_dl:
            st.download_button(
                "⬇ Download output.csv",
                data=_opt_path.read_bytes(),
                file_name="output.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with tb_sims:
            regen_sims = st.number_input(
                "Sims", min_value=10_000, max_value=200_000,
                value=30_000, step=10_000, key="sub_regen_sims",
                help="Simulation count passed to --sims",
                label_visibility="collapsed",
            )
            st.caption("sims")
        with tb_regen:
            if st.button("🔄 Regenerate optimised", type="primary",
                         use_container_width=True, key="sub_regen"):
                st.session_state["_sub_regen_sims"] = regen_sims

        st.caption(
            "**Optimised** — `output.csv` scoreline optimizer, updated by Regenerate  ·  "
            "**Raw** — `output_raw.csv` model probabilities, updated by Refresh predictions"
        )

        if "_sub_regen_sims" in st.session_state:
            _sims_val = st.session_state.pop("_sub_regen_sims")
            with st.container(border=True):
                st.markdown("**Regenerating optimised submission…**")
                _stream(
                    ["python3.11", str(_ROOT / "scripts" / "generate_submission.py"),
                     "--sims", str(_sims_val)],
                    "Done — scroll down to see updated numbers.",
                    on_success=st.cache_data.clear,
                )

        st.divider()

        # ── Build merged data ─────────────────────────────────────────────────
        _sub_opt = pd.read_csv(_opt_path)
        _sub_raw = pd.read_csv(_raw_path) if _raw_path.exists() else None

        _actual_map: dict[tuple[str, str], tuple[int, int]] = {}
        for _r in _load_recorded():
            _actual_map[(normalise(_r["home_team"]), normalise(_r["away_team"]))] = (
                int(_r["home_goals"]), int(_r["away_goals"])
            )

        _merged: list[dict] = []
        for _, _row in _sub_opt.iterrows():
            _h = _TEMPLATE_TO_SCHEDULE.get(str(_row["team1"]), str(_row["team1"]))
            _a = _TEMPLATE_TO_SCHEDULE.get(str(_row["team2"]), str(_row["team2"]))
            _actual = _actual_map.get((normalise(_h), normalise(_a)))
            _rr = None
            if _sub_raw is not None:
                _mask = (
                    _sub_raw["team1"].map(lambda x: _TEMPLATE_TO_SCHEDULE.get(str(x), str(x))) == _h
                ) & (
                    _sub_raw["team2"].map(lambda x: _TEMPLATE_TO_SCHEDULE.get(str(x), str(x))) == _a
                )
                _hits = _sub_raw[_mask]
                if not _hits.empty:
                    _rr = _hits.iloc[0]

            _s1, _s2 = int(_row["score1"]), int(_row["score2"])
            _p_h = float(_rr["p_home"]) if _rr is not None else None
            _p_a = float(_rr["p_away"]) if _rr is not None else None
            _upset = False
            if _p_h is not None and _p_a is not None:
                _fav = "home" if _p_h > _p_a else ("away" if _p_a > _p_h else "draw")
                _sub_out = "home" if _s1 > _s2 else ("away" if _s2 > _s1 else "draw")
                _upset = _fav != _sub_out

            _merged.append({
                "group":    _row["group"],
                "h_sched":  _h,
                "a_sched":  _a,
                "opt_s1":   _s1,
                "opt_s2":   _s2,
                "played":   _actual is not None,
                "act_s1":   _actual[0] if _actual else None,
                "act_s2":   _actual[1] if _actual else None,
                "p_home":   _p_h,
                "p_draw":   float(_rr["p_draw"]) if _rr is not None else None,
                "p_away":   _p_a,
                "lam_h":    round(float(_rr["lam_home"]), 2) if _rr is not None else None,
                "lam_a":    round(float(_rr["lam_away"]), 2) if _rr is not None else None,
                "raw_s1":   int(_rr["score1"]) if _rr is not None else None,
                "raw_s2":   int(_rr["score2"]) if _rr is not None else None,
                "upset":    _upset,
            })
        _mdf = pd.DataFrame(_merged)

        # ── Filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns(3)
        _status = fc1.radio("Show", ["All", "Unplayed", "Played"],
                            horizontal=True, key="sub_status")
        _grp_f  = fc2.selectbox("Group", ["All"] + sorted(_mdf["group"].unique().tolist()),
                                key="sub_grp", label_visibility="collapsed")
        _sort   = fc3.selectbox(
            "Sort", ["Group order", "Home win %", "Away win %", "Most uncertain", "Upsets first"],
            key="sub_sort", label_visibility="collapsed",
        )

        _filt = _mdf.copy()
        if _status == "Unplayed":
            _filt = _filt[~_filt["played"]]
        elif _status == "Played":
            _filt = _filt[_filt["played"]]
        if _grp_f != "All":
            _filt = _filt[_filt["group"] == _grp_f]
        if _sort == "Home win %" and _filt["p_home"].notna().any():
            _filt = _filt.sort_values("p_home", ascending=False)
        elif _sort == "Away win %" and _filt["p_away"].notna().any():
            _filt = _filt.sort_values("p_away", ascending=False)
        elif _sort == "Most uncertain" and _filt["p_home"].notna().any():
            _filt = _filt.assign(_unc=(_filt["p_home"] - _filt["p_away"]).abs()).sort_values("_unc")
        elif _sort == "Upsets first":
            _filt = _filt.sort_values("upset", ascending=False)

        st.caption(f"Showing {len(_filt)} of 72 matches"
                   + (f"  ·  ⚠️ {_filt['upset'].sum()} upset submissions" if _filt["upset"].any() else ""))

        # ── Sub-tabs ──────────────────────────────────────────────────────────
        sub_scores, sub_raw_tab, sub_compare = st.tabs([
            "🏆 Optimised", "📊 Raw probabilities", "⚖️ Compare",
        ])

        # ── Optimised scorelines ───────────────────────────────────────────────
        with sub_scores:
            st.caption("From `output.csv` — scoreline optimizer. Use **Regenerate** above to refresh.")
            if _sort == "Group order":
                _gcols = st.columns(3)
                for _gi, _grp in enumerate(sorted(_filt["group"].unique())):
                    _gc = _gcols[_gi % 3]
                    _gdf = _filt[_filt["group"] == _grp]
                    _html = (
                        f"<p style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                        f"letter-spacing:0.1em;color:rgba(255,255,255,0.32);margin:0 0 6px'>"
                        f"Group {_grp}</p>"
                    )
                    for _, _r in _gdf.iterrows():
                        _s1, _s2 = _r["opt_s1"], _r["opt_s2"]
                        _sc = _COL_HOME if _s1 > _s2 else (_COL_DRAW if _s1 == _s2 else _COL_AWAY)
                        _badge = " ⚠️" if _r["upset"] else ""
                        if _r["played"]:
                            _a1, _a2 = int(_r["act_s1"]), int(_r["act_s2"])
                            _hit = (_a1>_a2 and _s1>_s2) or (_a1==_a2 and _s1==_s2) or (_a1<_a2 and _s1<_s2)
                            _act = (f"<div style='text-align:center;font-size:0.7rem;"
                                    f"color:rgba(255,255,255,0.38);margin-top:1px'>"
                                    f"actual {_a1}–{_a2} {'✅' if _hit else '❌'}</div>")
                        else:
                            _act = ""
                        _html += (
                            f"<div style='padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
                            f"<div style='display:flex;align-items:center;font-size:0.82rem'>"
                            f"<span style='flex:3'>{flag(_r['h_sched'])}</span>"
                            f"<span style='flex:2;text-align:center;font-weight:700;color:{_sc}'>"
                            f"{_s1} – {_s2}{_badge}</span>"
                            f"<span style='flex:3;text-align:right'>{flag(_r['a_sched'])}</span>"
                            f"</div>{_act}</div>"
                        )
                    _gc.markdown(
                        f"<div style='background:rgba(255,255,255,0.02);border:1px solid "
                        f"rgba(255,255,255,0.07);border-radius:10px;padding:12px 14px;"
                        f"margin-bottom:12px'>{_html}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                _trows = []
                for _, _r in _filt.iterrows():
                    _s1, _s2 = _r["opt_s1"], _r["opt_s2"]
                    if _r["played"]:
                        _a1, _a2 = int(_r["act_s1"]), int(_r["act_s2"])
                        _hit = (_a1>_a2 and _s1>_s2) or (_a1==_a2 and _s1==_s2) or (_a1<_a2 and _s1<_s2)
                        _act_str = f"{_a1}–{_a2} {'✅' if _hit else '❌'}"
                    else:
                        _act_str = "—"
                    _trows.append({
                        "Grp": _r["group"], "Home": flag(_r["h_sched"]),
                        "Away": flag(_r["a_sched"]),
                        "Prediction": f"{_s1}–{_s2}" + (" ⚠️" if _r["upset"] else ""),
                        "Actual": _act_str,
                        "Home win": _r["p_home"], "Away win": _r["p_away"],
                    })
                _tdf = pd.DataFrame(_trows)
                _pcols = [c for c in ["Home win","Away win"] if _tdf[c].notna().any()]
                _st = _tdf.style.format({c: "{:.1%}" for c in _pcols})
                if "Home win" in _pcols:
                    _st = _st.map(lambda v: f"background:rgba(74,222,128,{v*0.5:.2f})" if pd.notna(v) else "", subset=["Home win"])
                if "Away win" in _pcols:
                    _st = _st.map(lambda v: f"background:rgba(251,146,60,{v*0.5:.2f})" if pd.notna(v) else "", subset=["Away win"])
                st.dataframe(_st, use_container_width=True, hide_index=True)

        # ── Raw probabilities ──────────────────────────────────────────────────
        with sub_raw_tab:
            st.caption("From `output_raw.csv` — direct model output (probabilities + xG). Updated when you click **Refresh predictions**, not Regenerate.")
            if _sub_raw is None:
                st.info("No raw output file — run Refresh predictions to generate it.")
            else:
                _rrows = []
                for _, _r in _filt.iterrows():
                    if _r["p_home"] is None:
                        continue
                    _rrows.append({
                        "Grp": _r["group"], "Home": flag(_r["h_sched"]),
                        "Away": flag(_r["a_sched"]),
                        "Score": f"{_r['opt_s1']}–{_r['opt_s2']}",
                        "Home win": _r["p_home"], "Draw": _r["p_draw"], "Away win": _r["p_away"],
                        "xG H": _r["lam_h"], "xG A": _r["lam_a"],
                    })
                _rdf = pd.DataFrame(_rrows)
                _rst = (
                    _rdf.style
                    .map(lambda v: f"background:rgba(74,222,128,{v*0.55:.2f})",  subset=["Home win"])
                    .map(lambda v: f"background:rgba(161,161,170,{v*0.55:.2f})", subset=["Draw"])
                    .map(lambda v: f"background:rgba(251,146,60,{v*0.55:.2f})",  subset=["Away win"])
                    .format({"Home win":"{:.1%}","Draw":"{:.1%}","Away win":"{:.1%}",
                             "xG H":"{:.2f}","xG A":"{:.2f}"})
                )
                st.dataframe(_rst, use_container_width=True, hide_index=True, height=520)

        # ── Compare optimised vs raw ───────────────────────────────────────────
        with sub_compare:
            st.caption("Optimizer picks (`output.csv`) vs raw Poisson most-likely scores (`output_raw.csv`). Amber rows = the optimizer chose a different scoreline than the model's most likely.")
            if _sub_raw is None:
                st.info("No raw output file — run Refresh predictions to generate it.")
            else:
                _crows = []
                for _, _r in _filt.iterrows():
                    if _r["raw_s1"] is None:
                        continue
                    _opt_s = f"{_r['opt_s1']}–{_r['opt_s2']}"
                    _raw_s = f"{_r['raw_s1']}–{_r['raw_s2']}"
                    _crows.append({
                        "Grp": _r["group"], "Home": flag(_r["h_sched"]),
                        "Away": flag(_r["a_sched"]),
                        "Optimised": _opt_s, "Raw / Poisson": _raw_s,
                        "Δ": "—" if _opt_s == _raw_s else "≠",
                        "Home win": _r["p_home"], "Away win": _r["p_away"],
                    })
                _cdf = pd.DataFrame(_crows)
                _n_diff = (_cdf["Δ"] == "≠").sum()
                st.caption(f"{_n_diff} of {len(_cdf)} matches differ between optimised and raw Poisson")

                def _hl_diff(row):
                    styles = [""] * len(row)
                    if row.get("Δ") == "≠":
                        idx = list(row.index)
                        for col in ["Optimised", "Raw / Poisson"]:
                            if col in idx:
                                styles[idx.index(col)] = "background-color:#4d2800;color:#ffb300"
                    return styles

                _pcols2 = [c for c in ["Home win","Away win"] if _cdf[c].notna().any()]
                _cst = _cdf.style.apply(_hl_diff, axis=1).format({c:"{:.1%}" for c in _pcols2})
                st.dataframe(_cst, use_container_width=True, hide_index=True, height=520)
