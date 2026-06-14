"""Extended Kalman Filter + RTS Smoother for time-varying team strength.

STATE-SPACE MODEL
─────────────────
State per team:   x = [att, def]   (log-goals space)
Evolution:        x_{t+Δ} = x_t + w,   w ~ N(0, q²·Δt·I)   [random walk]
Observation:      goals_h ~ Poisson(exp(μ + att_h + def_a + ha))
                  goals_a ~ Poisson(exp(μ + att_a + def_h))

FORWARD PASS (EKF)
──────────────────
Used at PREDICTION TIME — causally correct, uses only past data.
  1. Time update:    P ← P + q²·Δt·I           (uncertainty grows between matches)
  2. Joint EKF update (roadmap R4): both teams stacked into one state
                     z = [att_h, def_h, att_a, def_a], block-diagonal prior M.
                     Home goals:  H1 = [λ_h, 0, 0, λ_h];  away goals: H2 = [0, λ_a, λ_a, 0]
                     Innovation covariance S = H M H' + λ/weight  (Poisson: Var = mean)
                     Kalman gain K = M H' / S;  z ← z + K·innov;  M ← Joseph-form update
                     Obs 1 couples the two blocks; obs 2 sees that coupling.
                     Marginals scattered back per team (no global cross-team cov).

RTS BACKWARD SMOOTHER (Rauch-Tung-Striebel)
────────────────────────────────────────────
Used for EM Q-tuning and optionally for training features (use_smoothed=True).
  Smoother gain:  G_t = P_upd[t] · P_pred[t+1]⁻¹
  Smoothed state: x̂_t = x_upd[t] + G_t · (x̂_{t+1} − x_pred[t+1])
  Smoothed cov:   P̂_t = P_upd[t] + G_t · (P̂_{t+1} − P_pred[t+1]) · G_t'

EM Q-TUNING
───────────
When tune_q=True, runs EM iterations before building the feature cache:
  M-step: q²_new = (1/2N) · Σ_t [ (||dx_s||² + tr(P_s[t+1]) + tr(P_s[t])
                                    − 2·tr(G_t · P_s[t+1])) / Δt_t ]
  where dx_s = x_smooth[t+1] − x_smooth[t] and the /2 accounts for 2D state.
  Converges in ~5 iterations. Features then use the tuned Q with forward-only EKF.

MODES
─────
  use_smoothed=False, tune_q=False  — forward-only (default, no covariate shift)
  use_smoothed=True,  tune_q=False  — RTS-smoothed training features (original)
  use_smoothed=False, tune_q=True   — EM-tuned Q, forward-only features (Option A)

FEATURES FOR XGBoost (13 total)
────────────────────────────────
  kalman_home_att, kalman_home_def     — strength estimates
  kalman_away_att, kalman_away_def
  kalman_home_att_std, kalman_home_def_std  — uncertainty (√P diagonal)
  kalman_away_att_std, kalman_away_def_std
  kalman_exp_home_goals, kalman_exp_away_goals  — predicted λ
  kalman_goal_diff, kalman_att_diff, kalman_def_diff

References:
    Kalman, R.E. (1960). Trans. ASME–J. Basic Eng., 82(D), 35–45.
    Rauch, H.E., Tung, F., & Striebel, C.T. (1965). AIAA Journal, 3(8), 1445–1450.
    Koopman, S.J. & Lit, R. (2015). JRSS-A, 178(1), 167–186.
    Shumway, R.H. & Stoffer, D.S. (1982). Ann. Stat., 10(2), 423–441.  [EM for SSM]
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from football_predictor.features.base import FeatureModule

# ── Hyperparameters ────────────────────────────────────────────────────────────
_MU: float = math.log(1.3)       # baseline log-goals (≈ 1.3 goals/team/game)
_HOME_ADV: float = 0.20           # log-scale home advantage (≈ +22% goals at home)
_Q_PER_YEAR: float = 0.15         # process noise default (tuned via EM when tune_q=True)
_P0_DIAG: float = 0.16            # initial covariance diagonal (0.4² in log-goals units)
_P_FLOOR: float = 1e-6            # minimum variance to prevent degeneracy
_DAYS_PER_YEAR: float = 365.25


def _default_state() -> dict:
    return {"x": np.zeros(2), "P": _P0_DIAG * np.eye(2)}


def _data_fingerprint(df: pd.DataFrame) -> tuple:
    """Content-based dataset key for caching.

    `id(data)` is NOT a safe cache key: CPython recycles object ids after
    garbage collection, so a long session can silently match a stale entry
    and reuse the wrong tuned q (or skip a needed cache rebuild). The
    fingerprint changes whenever the match set meaningfully changes.
    """
    # match_weight sum included because the friendly weight changes the EKF's
    # effective R (= λ/weight) and therefore the EM-tuned q — without it,
    # retraining with a different friendly weight in the same process would
    # falsely cache-hit on the old q.
    mw_sum = float(df["match_weight"].sum()) if "match_weight" in df.columns else 0.0
    return (
        len(df),
        str(df["date"].min()),
        str(df["date"].max()),
        int(df["home_goals"].sum() + df["away_goals"].sum()),
        round(mw_sum, 3),
    )


def _g_func(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _joint_obs_update(
    z: np.ndarray,
    M: np.ndarray,
    innovation: float,
    h_vec: np.ndarray,
    r_eff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single scalar EKF observation update on a joint (stacked) state.

    Joseph form for numerical stability. Returns (z_new, M_new) rather than
    mutating in place so the two within-match observations compose cleanly.

    r_eff is the measurement variance, R = Poisson_var / match_weight. The
    opponent's contribution to the innovation covariance is NO LONGER passed
    in separately (the old `extra_obs_var` hack): with a joint covariance M
    that spans both teams, `h_vec @ M @ h_vec` already includes both the home
    and away state variances — and, on the second observation, the
    cross-covariance the first observation induced (roadmap R4).
    """
    MH = M @ h_vec
    S = float(h_vec @ MH) + r_eff
    K = MH / S
    z_new = z + K * innovation
    I_KH = np.eye(M.shape[0]) - np.outer(K, h_vec)
    M_new = I_KH @ M @ I_KH.T + r_eff * np.outer(K, K)
    return z_new, M_new


def _ekf_match_update(
    h_state: dict,
    a_state: dict,
    home_goals: int,
    away_goals: int,
    neutral: bool,
    match_weight: float = 1.0,
) -> None:
    """Joint 4-d EKF update for one full match (roadmap R4).

    The two teams' states are stacked into z = [att_h, def_h, att_a, def_a]
    with a block-diagonal prior covariance (teams carry no stored cross-team
    covariance between matches). The two Poisson observations — home goals,
    then away goals — are applied as coupled updates on the joint covariance:

        λ_h = exp(μ + att_h + def_a + ha)   → H1 = [λ_h, 0,   0,   λ_h]
        λ_a = exp(μ + att_a + def_h)        → H2 = [0,   λ_a, λ_a, 0  ]

    The first observation induces cross-covariance between the home and away
    blocks (via each team's own att↔def correlation); the second observation
    then sees that coupling. The previous implementation updated the two
    per-team covariances separately and approximated the opponent's variance
    with an `extra_obs_var` term in S — correct for the first observation but
    blind to the cross-covariance feeding the second. After both updates the
    marginals are scattered back per team (global cross-team covariance is not
    retained — that would mean an O(n_teams²) joint filter).
    """
    ha = 0.0 if neutral else _HOME_ADV

    z = np.concatenate([h_state["x"], a_state["x"]])
    M = np.zeros((4, 4))
    M[0:2, 0:2] = h_state["P"]
    M[2:4, 2:4] = a_state["P"]

    # Observation 1: home goals.
    lam_h = max(math.exp(_MU + z[0] + z[3] + ha), 1e-4)
    r_h = max(lam_h, 1e-4) / max(match_weight, 1e-3)
    z, M = _joint_obs_update(z, M, home_goals - lam_h,
                             np.array([lam_h, 0.0, 0.0, lam_h]), r_h)

    # Observation 2: away goals — relinearised on the post-obs1 state (matches
    # the previous sequential behaviour), now coupled through the joint M.
    lam_a = max(math.exp(_MU + z[2] + z[1]), 1e-4)
    r_a = max(lam_a, 1e-4) / max(match_weight, 1e-3)
    z, M = _joint_obs_update(z, M, away_goals - lam_a,
                             np.array([0.0, lam_a, lam_a, 0.0]), r_a)

    # Scatter marginals back; floor the variance diagonals.
    h_state["x"] = z[0:2].copy()
    a_state["x"] = z[2:4].copy()
    Ph = M[0:2, 0:2].copy()
    Pa = M[2:4, 2:4].copy()
    Ph[0, 0] = max(Ph[0, 0], _P_FLOOR)
    Ph[1, 1] = max(Ph[1, 1], _P_FLOOR)
    Pa[0, 0] = max(Pa[0, 0], _P_FLOOR)
    Pa[1, 1] = max(Pa[1, 1], _P_FLOOR)
    h_state["P"] = Ph
    a_state["P"] = Pa


def _run_rts(team_hist: dict[str, list[dict]]) -> None:
    """In-place RTS backward smoother. Stores smoother gain G for EM."""
    for team, hist in team_hist.items():
        n = len(hist)
        if n == 0:
            continue

        hist[-1]["x_smooth"] = hist[-1]["x_upd"].copy()
        hist[-1]["P_smooth"] = hist[-1]["P_upd"].copy()

        for i in range(n - 2, -1, -1):
            P_upd_i = hist[i]["P_upd"]
            P_pred_i1 = hist[i + 1]["P_pred"]

            try:
                G = P_upd_i @ np.linalg.solve(P_pred_i1.T, np.eye(2)).T
            except np.linalg.LinAlgError:
                # Regularised matrix inverse — element-wise division here is
                # mathematically meaningless and produced inf on zero
                # off-diagonals, poisoning the EM-tuned q.
                G = P_upd_i @ np.linalg.inv(P_pred_i1 + 1e-8 * np.eye(2))

            hist[i]["G"] = G  # stored for EM cross-covariance

            x_diff = hist[i + 1]["x_smooth"] - hist[i + 1]["x_pred"]
            P_diff = hist[i + 1]["P_smooth"] - P_pred_i1

            hist[i]["x_smooth"] = hist[i]["x_upd"] + G @ x_diff
            P_s = P_upd_i + G @ P_diff @ G.T
            P_s[0, 0] = max(P_s[0, 0], _P_FLOOR)
            P_s[1, 1] = max(P_s[1, 1], _P_FLOOR)
            hist[i]["P_smooth"] = P_s


def _em_estimate_q(team_hist: dict[str, list[dict]]) -> float:
    """EM M-step: ML estimate of process noise q from smoothed states.

    Uses the Shumway-Stoffer (1982) update for isotropic random-walk Q:
        q²_hat = (1 / 2N) · Σ_t [ (||dx_s||² + tr(P_s[t+1]) + tr(P_s[t])
                                    − 2·tr(G_t · P_s[t+1])) / Δt_t ]
    The /2 accounts for the 2D state dimension.
    """
    total = 0.0
    n_transitions = 0

    for hist in team_hist.values():
        for i in range(len(hist) - 1):
            dt = (hist[i + 1]["date"] - hist[i]["date"]).days / _DAYS_PER_YEAR
            if dt <= 0:
                continue
            dx = hist[i + 1]["x_smooth"] - hist[i]["x_smooth"]
            G = hist[i].get("G", np.zeros((2, 2)))
            P_cross = G @ hist[i + 1]["P_smooth"]
            innov = (
                np.dot(dx, dx)
                + np.trace(hist[i + 1]["P_smooth"])
                + np.trace(hist[i]["P_smooth"])
                - 2.0 * np.trace(P_cross)
            )
            total += innov / dt
            n_transitions += 1

    if n_transitions == 0:
        return _Q_PER_YEAR
    q_sq = total / (2.0 * n_transitions)
    return float(math.sqrt(max(q_sq, 1e-6)))


class KalmanStrengthFeatures(FeatureModule):
    """EKF with optional RTS smoother and EM-tuned process noise."""

    name = "kalman_strength"

    # Class-level cache so EM runs once per dataset even across multiple instances
    # (build_feature_matrix creates fresh instances for train and test calls).
    # Key: content fingerprint (see _data_fingerprint) — id(data) was unsafe
    # because object ids are recycled after garbage collection.
    _EM_Q_CACHE: dict[tuple, float] = {}
    _LAST_TUNED_Q: float = _Q_PER_YEAR  # updated after each EM run; read by BayesPoisson

    @classmethod
    def get_last_tuned_q(cls) -> float:
        """Return the most recently EM-estimated process noise (q per year).

        Used by BayesianPoissonModel to derive a half-life consistent with
        the Kalman filter's learned assumption about team strength drift rate.
        """
        return cls._LAST_TUNED_Q

    def __init__(self, use_smoothed: bool = False, tune_q: bool = True) -> None:
        self._use_smoothed = use_smoothed
        self._tune_q = tune_q
        self._q_per_year: float = _Q_PER_YEAR
        self._tuned: bool = False
        self._states: dict[str, dict] = {}
        self._snapshots: dict[str, dict[str, dict]] = {}
        self._smoothed_snapshots: dict[str, dict[str, dict]] = {}
        self._last_date: dict[str, pd.Timestamp] = {}  # team -> last match date
        self._data_id: Optional[tuple] = None  # content fingerprint, not id()

    # ── FeatureModule interface ────────────────────────────────────────────────

    def fetch(self, competition: str, seasons: list[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, match: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        self._ensure_cache(data)

        target = pd.Timestamp(match["date"])
        date_str = str(target.date())
        home, away = match["home_team"], match["away_team"]
        is_neutral = bool(match.get("neutral", True))

        if self._use_smoothed and date_str in self._smoothed_snapshots:
            snap = self._smoothed_snapshots[date_str]
        else:
            # Empty default (not self._states) so teams without a snapshot on
            # this date go through the aging fallback below.
            snap = self._snapshots.get(date_str, {})

        def _state_for(team: str) -> dict:
            s = snap.get(team)
            if s is not None:
                return s
            s = self._states.get(team)
            if s is None:
                return _default_state()
            # Future-fixture fallback: the stored final state carries P frozen
            # at the team's LAST match. Apply the time update q²·Δt·I forward
            # to the fixture date so kalman_*_std reflects predictive
            # uncertainty. Copy — never mutate the stored state.
            last = self._last_date.get(team)
            if last is None:
                return {"x": s["x"].copy(), "P": s["P"].copy()}
            dt_years = max((target - last).days, 0) / _DAYS_PER_YEAR
            return {
                "x": s["x"].copy(),
                "P": s["P"] + (self._q_per_year ** 2) * dt_years * np.eye(2),
            }

        h = _state_for(home)
        a = _state_for(away)

        ha = 0.0 if is_neutral else _HOME_ADV
        lam_h = math.exp(_MU + h["x"][0] + a["x"][1] + ha)
        lam_a = math.exp(_MU + a["x"][0] + h["x"][1])

        h_att_std = math.sqrt(max(float(h["P"][0, 0]), _P_FLOOR))
        h_def_std = math.sqrt(max(float(h["P"][1, 1]), _P_FLOOR))
        a_att_std = math.sqrt(max(float(a["P"][0, 0]), _P_FLOOR))
        a_def_std = math.sqrt(max(float(a["P"][1, 1]), _P_FLOOR))

        return {
            "kalman_home_att":         float(h["x"][0]),
            "kalman_home_def":         float(h["x"][1]),
            "kalman_away_att":         float(a["x"][0]),
            "kalman_away_def":         float(a["x"][1]),
            "kalman_home_att_std":     h_att_std,
            "kalman_home_def_std":     h_def_std,
            "kalman_away_att_std":     a_att_std,
            "kalman_away_def_std":     a_def_std,
            "kalman_exp_home_goals":   lam_h,
            "kalman_exp_away_goals":   lam_a,
            "kalman_goal_diff":        lam_h - lam_a,
            "kalman_att_diff":         float(h["x"][0]) - float(a["x"][0]),
            "kalman_def_diff":         float(a["x"][1]) - float(h["x"][1]),
        }

    def feature_names(self) -> list[str]:
        return [
            "kalman_home_att", "kalman_home_def",
            "kalman_away_att", "kalman_away_def",
            "kalman_home_att_std", "kalman_home_def_std",
            "kalman_away_att_std", "kalman_away_def_std",
            "kalman_exp_home_goals", "kalman_exp_away_goals",
            "kalman_goal_diff", "kalman_att_diff", "kalman_def_diff",
        ]

    # ── Cache construction ─────────────────────────────────────────────────────

    def _ensure_cache(self, data: pd.DataFrame) -> None:
        # Content fingerprint instead of id(data): recycled object ids could
        # wrongly skip a rebuild when a different DataFrame reuses an address.
        fingerprint = _data_fingerprint(data)
        if self._data_id == fingerprint:
            return
        self._data_id = fingerprint
        if self._tune_q and not self._tuned:
            self._run_em(data)
            self._tuned = True
        self._build_cache(data)

    def _run_forward_pass(
        self, df: pd.DataFrame, record_hist: bool
    ) -> tuple[dict, dict, dict | None]:
        """Forward EKF pass over a sorted date DataFrame.

        Args:
            df:           Sorted match DataFrame (date as Timestamp).
            record_hist:  If True, build team_hist for RTS/EM. If False, skip it.

        Returns:
            (final_states, snapshots, team_hist)  — team_hist is None if record_hist=False.
        """
        states: dict[str, dict] = {}
        last_date: dict[str, pd.Timestamp] = {}
        snapshots: dict[str, dict[str, dict]] = {}
        team_hist: dict[str, list[dict]] | None = {} if record_hist else None

        for date, group in df.groupby("date", sort=True):
            date_str = str(date.date())
            all_teams = set(group["home_team"]) | set(group["away_team"])

            for team in all_teams:
                if team not in states:
                    states[team] = _default_state()
                    last_date[team] = date

                dt_years = (date - last_date[team]).days / _DAYS_PER_YEAR
                x_pred = states[team]["x"].copy()
                P_pred = states[team]["P"] + (self._q_per_year ** 2) * dt_years * np.eye(2)
                states[team]["P"] = P_pred.copy()
                last_date[team] = date

                if record_hist:
                    team_hist.setdefault(team, []).append({
                        "date": date,
                        "x_pred": x_pred,
                        "P_pred": P_pred.copy(),
                    })

            # Snapshot AFTER the time update (P += q²·Δt·I) so training rows
            # see the PREDICTIVE uncertainty as of this match date — not P
            # frozen at the team's previous match, which understates std for
            # teams returning from long gaps.
            snapshots[date_str] = {
                t: {"x": states[t]["x"].copy(), "P": states[t]["P"].copy()}
                for t in all_teams
            }

            for _, row in group.iterrows():
                _ekf_match_update(
                    states[row["home_team"]], states[row["away_team"]],
                    int(row["home_goals"]), int(row["away_goals"]),
                    bool(row.get("neutral", False)),
                    float(row.get("match_weight", 1.0)),
                )

            if record_hist:
                for team in all_teams:
                    team_hist[team][-1]["x_upd"] = states[team]["x"].copy()
                    team_hist[team][-1]["P_upd"] = states[team]["P"].copy()

        final_states = {
            t: {"x": s["x"].copy(), "P": s["P"].copy()} for t, s in states.items()
        }
        return final_states, snapshots, team_hist

    def _build_cache(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        self._states, self._snapshots, team_hist = self._run_forward_pass(
            df, record_hist=self._use_smoothed
        )

        # Record each team's last match date so transform() can age the
        # frozen final-state P forward to a future fixture date (W12).
        last: dict[str, pd.Timestamp] = {}
        for col in ("home_team", "away_team"):
            for team, d in df.groupby(col)["date"].max().items():
                if team not in last or d > last[team]:
                    last[team] = d
        self._last_date = last

        if self._use_smoothed and team_hist is not None:
            _run_rts(team_hist)
            self._build_smoothed_snapshots(team_hist)

    def _build_with_rts(self, data: pd.DataFrame) -> dict:
        """Forward EKF + RTS smoother. Returns team_hist for EM Q estimation.
        Does not update self._snapshots or self._states."""
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        _, _, team_hist = self._run_forward_pass(df, record_hist=True)
        _run_rts(team_hist)
        return team_hist

    def _run_em(self, data: pd.DataFrame, max_iter: int = 10, tol: float = 1e-4) -> None:
        """EM iterations to estimate Q. Updates self._q_per_year in-place.

        Uses a class-level cache so EM runs at most once per unique dataset
        even when multiple instances are created for the same context data.
        """
        cache_key = _data_fingerprint(data)
        if cache_key in KalmanStrengthFeatures._EM_Q_CACHE:
            self._q_per_year = KalmanStrengthFeatures._EM_Q_CACHE[cache_key]
            # Keep the class-level q in sync on cache HIT too — otherwise
            # get_last_tuned_q() is ordering-dependent and BayesPoisson can
            # derive dataset A's half-life from dataset B's tuned q.
            KalmanStrengthFeatures._LAST_TUNED_Q = self._q_per_year
            return

        for _ in range(max_iter):
            q_old = self._q_per_year
            team_hist = self._build_with_rts(data)
            self._q_per_year = _em_estimate_q(team_hist)
            if abs(self._q_per_year - q_old) < tol:
                break

        KalmanStrengthFeatures._EM_Q_CACHE[cache_key] = self._q_per_year
        KalmanStrengthFeatures._LAST_TUNED_Q = self._q_per_year
        print(f"  [Kalman EM] tuned Q: {self._q_per_year:.4f}/yr  (default: {_Q_PER_YEAR:.4f})")

    def _build_smoothed_snapshots(
        self, team_hist: dict[str, list[dict]]
    ) -> None:
        """Build RTS-smoothed pre-match snapshots for all historical dates."""
        smooth_tl: dict[str, list[tuple]] = {}
        for team, hist in team_hist.items():
            smooth_tl[team] = [
                (
                    e["date"],
                    e.get("x_smooth", e["x_upd"]),
                    e.get("P_smooth", e["P_upd"]),
                )
                for e in hist
            ]

        self._smoothed_snapshots = {}
        for date_str, fwd_snap in self._snapshots.items():
            target_date = pd.Timestamp(date_str)
            s_snap: dict[str, dict] = {}
            for team in fwd_snap:
                tl = smooth_tl.get(team, [])
                lo, hi = 0, len(tl)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tl[mid][0] < target_date:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo > 0:
                    _, xs, Ps = tl[lo - 1]
                    s_snap[team] = {"x": xs, "P": Ps}
                else:
                    s_snap[team] = fwd_snap[team]
            self._smoothed_snapshots[date_str] = s_snap
