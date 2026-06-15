#!/usr/bin/env python3.11
"""Backtest: train on pre-WC data, evaluate on WC group stage.

Usage:
    python3.11 scripts/backtest.py --years 2018 2022
    python3.11 scripts/backtest.py --years 2022 --no-shap

Metrics reported
─────────────────
  Log-loss (cross-entropy) — primary probabilistic accuracy metric
  Brier score (MSE on probabilities) — complementary to log-loss
  Accuracy — directional accuracy (correct win/draw/loss)
  ECE (Expected Calibration Error) — how well probabilities match frequencies
  Baseline: uniform (0.33, 0.33, 0.33) — sanity floor

Outputs saved to ../output/
  backtest_YEAR_metrics.txt   — numeric summary
  backtest_YEAR_calibration.png  — reliability diagram
  backtest_YEAR_shap.png         — SHAP feature importance

References:
    Brier, G.W. (1950). Verification of forecasts. Monthly Weather Review.
    Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
    Lundberg, S. & Lee, S-I. (2017). SHAP. NeurIPS.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from football_predictor.constants import DEFAULT_FEATURE_MODULES, OUTCOMES
from football_predictor.data.sources.international_results import fetch_training_data as load_results
from football_predictor.data.pipeline import build_feature_matrix, prune_correlated_features
from football_predictor.data.wc2026 import normalise
from football_predictor.models import (
    BayesianPoissonModel,
    EnsembleModel,
    GradientBoostModel,
    TemperatureScaling,
)
from football_predictor.models.stacking import fit_stacked_calibration

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTCOME_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

# WC group stage date ranges (group stage ONLY — knockout matches have
# different draw dynamics and may carry extra-time scorelines in the source)
WC_DATE_RANGES = {
    2014: ("2014-06-12", "2014-06-26"),
    2018: ("2018-06-14", "2018-06-28"),
    2022: ("2022-11-20", "2022-12-02"),  # was 12-09: leaked all 8 R16 + 2 QFs
}
WC_CUTOFFS = {
    2014: "2014-06-01",
    2018: "2018-06-01",
    2022: "2022-11-01",
}

# Continental tournament group stage windows (name fragment, date range, training cutoff)
CONTINENTAL_CONFIGS = {
    "Copa América 2021": {
        "name_fragment": "Copa Am",
        "date_range": ("2021-06-13", "2021-06-28"),  # group stage only (was 07-02: leaked QFs)
        "cutoff": "2021-06-01",
    },
    "UEFA Euro 2020": {
        "name_fragment": "UEFA Euro",
        "date_range": ("2021-06-11", "2021-06-23"),  # group stage only
        "cutoff": "2021-06-01",
    },
    "AFCON 2022": {
        "name_fragment": "African Cup",
        "date_range": ("2022-01-09", "2022-01-20"),  # group stage only
        "cutoff": "2022-01-01",
    },
    "Asian Cup 2023": {
        "name_fragment": "AFC Asian Cup",
        "date_range": ("2024-01-12", "2024-01-23"),  # group stage only
        "cutoff": "2024-01-01",
    },
}

_COMP_FILTER = (
    "qualif|world cup|copa am|copa amér|euro|nations|africa|"
    "asian|gold cup|confederations|african|copa do"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _outcome_to_int(row: pd.Series) -> int:
    if row["home_goals"] > row["away_goals"]:
        return 0
    if row["home_goals"] == row["away_goals"]:
        return 1
    return 2


def _bp_proba_df(bp: BayesianPoissonModel, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = bp.predict_proba(normalise(r["home_team"]), normalise(r["away_team"]), bool(r.get("neutral", True)))
        rows.append([p["home_win"], p["draw"], p["away_win"]])
    return pd.DataFrame(rows, columns=OUTCOMES, index=df.index)


def _compute_ece(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (top-class confidence buckets)."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece / len(y_true))


def _log_loss(proba: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-10
    n = len(y_true)
    ll = -sum(np.log(np.maximum(proba[i, y_true[i]], eps)) for i in range(n)) / n
    return float(ll)


def _brier(proba: np.ndarray, y_true: np.ndarray) -> float:
    n_classes = proba.shape[1]
    y_oh = np.zeros_like(proba)
    for i, yi in enumerate(y_true):
        y_oh[i, yi] = 1.0
    return float(np.mean((proba - y_oh) ** 2))


def _accuracy(proba: np.ndarray, y_true: np.ndarray) -> float:
    return float((proba.argmax(axis=1) == y_true).mean())


def _confederation_slice(
    test_df: pd.DataFrame, proba: np.ndarray, y_true: np.ndarray
) -> dict[str, dict]:
    """Per-confederation log-loss / Brier / accuracy.

    A match is attributed to a confederation if *either* team belongs to it, so
    a small CAF/CONCACAF effect (the whole point of the StatsBomb xG data) is not
    washed out in the pooled number. Confederations overlap by design.
    """
    from football_predictor.features.confederation import ConfederationFeatures
    homes = test_df["home_team"].to_numpy()
    aways = test_df["away_team"].to_numpy()
    out: dict[str, dict] = {}
    for conf in ["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF"]:
        mask = np.array([
            ConfederationFeatures._get_conf(h) == conf or ConfederationFeatures._get_conf(a) == conf
            for h, a in zip(homes, aways)
        ])
        if mask.sum() >= 3:
            out[conf] = {
                "n": int(mask.sum()),
                "log_loss": _log_loss(proba[mask], y_true[mask]),
                "brier": _brier(proba[mask], y_true[mask]),
                "accuracy": _accuracy(proba[mask], y_true[mask]),
            }
    return out


def _bootstrap_ci(
    proba: np.ndarray,
    y_true: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for a metric.  Resamples match-level (paired) rows."""
    rng = np.random.default_rng(seed=0)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(proba[idx], y_true[idx]))
    return float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))


# ── Calibration plot (reliability diagram) ───────────────────────────────────

def plot_calibration(proba: np.ndarray, y_true: np.ndarray, year: int, label: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not available — skipping calibration plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, outcome in enumerate(["Home Win", "Draw", "Away Win"]):
        ax = axes[i]
        p_cls = proba[:, i]
        y_bin = (y_true == i).astype(float)
        bins = np.linspace(0, 1, 11)
        bin_acc, bin_conf, bin_cnt = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_cls >= lo) & (p_cls < hi)
            if mask.sum() > 3:
                bin_conf.append(p_cls[mask].mean())
                bin_acc.append(y_bin[mask].mean())
                bin_cnt.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        if bin_conf:
            ax.plot(bin_conf, bin_acc, "o-", color="steelblue", label="Model")
        ax.set_title(f"{outcome} — WC {year}")
        ax.set_xlabel("Predicted prob")
        ax.set_ylabel("Observed freq")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    suffix = f"_{label}" if label else ""
    plt.suptitle(f"Reliability Diagram — WC {year} Group Stage{' (' + label + ')' if label else ''}", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / f"backtest_{year}{suffix}_calibration.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Calibration plot → {out}")


# ── SHAP feature importance ───────────────────────────────────────────────────

def plot_shap(xgb: GradientBoostModel, X_test: pd.DataFrame, year: int, label: str = "") -> None:
    try:
        import shap
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  shap/matplotlib not available — skipping SHAP plot")
        return

    try:
        explainer = shap.TreeExplainer(xgb.model)
        shap_vals = explainer.shap_values(X_test)

        # shap_vals: list of 3 arrays (one per class) or a single 3D array
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2)) if shap_vals.ndim == 3 else np.abs(shap_vals).mean(axis=0)

        feat_imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).head(30)

        fig, ax = plt.subplots(figsize=(9, 10))
        ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="steelblue")
        suffix = f"_{label}" if label else ""
        ax.set_xlabel("Mean |SHAP value| (avg across 3 outcome classes)")
        ax.set_title(f"SHAP Feature Importance — WC {year} backtest{' (' + label + ')' if label else ''}")
        plt.tight_layout()
        out = OUTPUT_DIR / f"backtest_{year}{suffix}_shap.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  SHAP plot → {out}")

        # Also save CSV
        csv_out = OUTPUT_DIR / f"backtest_{year}{suffix}_shap.csv"
        feat_imp.to_csv(csv_out, index=False)
        print(f"  SHAP CSV  → {csv_out}")

    except Exception as exc:
        print(f"  SHAP failed: {exc}")


# ── Main backtest function ────────────────────────────────────────────────────

def run_backtest(
    year: int,
    include_shap: bool = True,
    label: str = "",
    friendly_weight: float = 0.3,
    xg_mode: str | None = None,
) -> dict:
    # Default follows the deployed config so a plain backtest reflects production;
    # the ablation passes explicit "baseline"/"full" to override.
    if xg_mode is None:
        from football_predictor import constants as _c0
        xg_mode = "full" if _c0.USE_XG_OBSERVATION else "baseline"
    print(f"\n{'─'*60}")
    print(f"  Backtest: WC {year} Group Stage  [xg_mode={xg_mode}]")
    print(f"{'─'*60}")

    if year not in WC_DATE_RANGES:
        print(f"  No WC date range defined for {year}. Supported: {list(WC_DATE_RANGES)}")
        return {}

    # ── Load data ─────────────────────────────────────────────────────────────
    print("  Loading historical results...")
    all_data = load_results(from_year=2000, friendly_weight=friendly_weight)
    all_data["date"] = pd.to_datetime(all_data["date"])
    all_data["outcome"] = all_data.apply(_outcome_to_int, axis=1)

    cutoff = WC_CUTOFFS[year]
    wc_start, wc_end = WC_DATE_RANGES[year]

    context = all_data[all_data["date"] < cutoff].copy()
    wc_mask = (
        all_data["tournament"].str.contains("FIFA World Cup", na=False)
        & (all_data["date"] >= wc_start)
        & (all_data["date"] <= wc_end)
    )
    test_df = all_data[wc_mask].copy()

    if len(test_df) == 0:
        print(f"  No WC {year} group stage matches found in dataset.")
        return {}

    print(f"  Training on {len(context)} pre-WC matches, testing on {len(test_df)} WC group stage matches.")

    # ── xG observation ablation ───────────────────────────────────────────────
    # The Kalman reads constants.USE_XG_OBSERVATION at construction (fresh
    # instance per build_feature_matrix call), so we toggle it only around the
    # feature build. xG is attached to `context` (the Kalman observes context
    # matches); the date filter already applied to `context` keeps this causal —
    # only xG sources predating the WC cutoff enter training.
    import football_predictor.constants as _C
    _prev_xg = _C.USE_XG_OBSERVATION
    use_xg = xg_mode != "baseline"
    if use_xg:
        from football_predictor.data.xg_attach import attach_calibrated_xg
        context = attach_calibrated_xg(context)
        n_xg = int(context[["home_xg", "away_xg"]].notna().all(axis=1).sum())
        print(f"  xG attached to {n_xg}/{len(context)} pre-cutoff training matches.")
        if n_xg == 0:
            print(f"  ⚠ No pre-{cutoff} xG coverage — xg_mode='{xg_mode}' is a no-op for WC {year}.")

    # ── Training data: competitive matches only ───────────────────────────────
    comp_mask = context["tournament"].str.contains(_COMP_FILTER, case=False, na=False)
    train_df = context[comp_mask].copy()

    # ── Build feature matrices ────────────────────────────────────────────────
    print("  Building feature matrices...")
    feature_mods = [m for m in DEFAULT_FEATURE_MODULES if m != "venue_wc2026"]

    _C.USE_XG_OBSERVATION = use_xg
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train, y_train = build_feature_matrix(train_df, feature_mods, context=context)
            X_test_raw, y_test = build_feature_matrix(test_df, feature_mods, context=context)
    finally:
        _C.USE_XG_OBSERVATION = _prev_xg

    X_train, _ = prune_correlated_features(X_train, threshold=0.95)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0.0)

    # ── Stacked calibration (R3): T + ensemble α on pooled OOS folds ─────────
    print("  Fitting stacked calibration (T + ensemble α on pooled OOS folds)...")
    sw = train_df["match_weight"].values if "match_weight" in train_df else np.ones(len(X_train))

    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    q_tuned = KalmanStrengthFeatures.get_last_tuned_q()
    hl = BayesianPoissonModel.half_life_from_q(q_tuned)
    print(f"  BayesPoisson half-life: {hl}d  (Kalman q={q_tuned:.4f}/yr)")

    # use_tuned_cache=False: cached hyperparameters were Optuna-selected on
    # WC 2018/2022 — evaluating those tournaments with them is test leakage.
    def _fit_xgb(X_tr, y_tr, sw_tr):
        m = GradientBoostModel(use_tuned_cache=False)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        return m

    def _fit_bp(matches_tr):
        b = BayesianPoissonModel(half_life_days=hl)
        b.fit(matches_tr)
        return b

    temp_cal, ensemble = fit_stacked_calibration(
        X_train, y_train, train_df.reset_index(drop=True), sw,
        _fit_xgb, _fit_bp,
        lambda b, m: _bp_proba_df(b, m.reset_index(drop=True)),
    )

    # Final models refit on the full pre-WC window now T and α are locked.
    print("  Training final XGBoost + BayesianPoisson on full window...")
    xgb = _fit_xgb(X_train, y_train, sw)
    bp = _fit_bp(train_df)
    bp.fit_rho(train_df)

    # ── Predict test set ──────────────────────────────────────────────────────
    print("  Predicting WC group stage...")
    xgb_test = temp_cal.transform(xgb.predict_proba(X_test))
    bp_test = _bp_proba_df(bp, test_df.reset_index(drop=True))
    bp_test.index = test_df.index

    proba_df = ensemble.predict_proba(
        xgb_test.reset_index(drop=True),
        bp_test.reset_index(drop=True),
        context_X=X_test.reset_index(drop=True),
    )
    proba = proba_df.values
    y_true = y_test.values

    # ── Metrics ───────────────────────────────────────────────────────────────
    uniform_proba = np.full_like(proba, 1.0 / 3.0)
    ll = _log_loss(proba, y_true)
    ll_lo, ll_hi = _bootstrap_ci(proba, y_true, _log_loss)
    brier = _brier(proba, y_true)
    brier_lo, brier_hi = _bootstrap_ci(proba, y_true, _brier)
    acc = _accuracy(proba, y_true)
    metrics = {
        "year": year,
        "n_matches": len(test_df),
        "log_loss": ll,
        "log_loss_ci_lo": ll_lo,
        "log_loss_ci_hi": ll_hi,
        "log_loss_uniform": _log_loss(uniform_proba, y_true),
        "brier_score": brier,
        "brier_ci_lo": brier_lo,
        "brier_ci_hi": brier_hi,
        "brier_uniform": _brier(uniform_proba, y_true),
        "accuracy": acc,
        "ece": _compute_ece(proba, y_true),
        "ensemble_alpha": ensemble.xgb_weight,
        "temperature": temp_cal.temperature,
    }

    print(f"\n  Results — WC {year} Group Stage ({metrics['n_matches']} matches):")
    print(f"    Log-loss:       {ll:.4f}  [{ll_lo:.4f}, {ll_hi:.4f}] 95% CI  (uniform: {metrics['log_loss_uniform']:.4f})")
    print(f"    Brier score:    {brier:.4f}  [{brier_lo:.4f}, {brier_hi:.4f}] 95% CI  (uniform: {metrics['brier_uniform']:.4f})")
    print(f"    Accuracy:       {acc:.3f}")
    print(f"    ECE:            {metrics['ece']:.4f}")
    print(f"    Ensemble α:     {metrics['ensemble_alpha']:.3f} XGB + {ensemble.bp_weight:.3f} BP")
    print(f"    Temperature T:  {metrics['temperature']:.3f}")

    # Per-confederation slice — keeps a CAF/CONCACAF effect from washing out.
    conf_metrics = _confederation_slice(test_df, proba, y_true)
    metrics["_conf"] = conf_metrics
    if conf_metrics:
        print("    By confederation (match counted if either team belongs):")
        for conf, cm in conf_metrics.items():
            print(f"      {conf:<9} n={cm['n']:<3} log-loss={cm['log_loss']:.4f}  "
                  f"brier={cm['brier']:.4f}  acc={cm['accuracy']:.3f}")

    # Save metrics text
    suffix = f"_{label}" if label else ""
    metrics_path = OUTPUT_DIR / f"backtest_{year}{suffix}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"WC {year} Group Stage Backtest\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            if k not in ("year",):
                f.write(f"{k:30s}: {v}\n")
    print(f"\n  Metrics → {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_calibration(proba, y_true, year, label=label)
    if include_shap:
        plot_shap(xgb, X_test, year, label=label)

    # Underscore keys: added after the metrics file write so experiment scripts
    # (e.g. odds_blend_backtest.py) get the raw predictions without polluting
    # the saved metrics txt.
    metrics["_proba"] = proba
    metrics["_y_true"] = y_true
    metrics["_test_df"] = test_df

    return metrics


# ── Match-by-match breakdown ──────────────────────────────────────────────────

def print_match_breakdown(test_df: pd.DataFrame, proba: np.ndarray, y_true: np.ndarray) -> None:
    print("\n  Match-by-match predictions:")
    print(f"  {'Date':<12} {'Home':<20} {'Away':<20} {'P(H)':<7} {'P(D)':<7} {'P(A)':<7} {'True':<8} {'Correct'}")
    print("  " + "-" * 90)
    outcome_names = ["HW", "D", "AW"]
    for i, (_, row) in enumerate(test_df.iterrows()):
        ph, pd_p, pa = proba[i]
        true_lbl = outcome_names[y_true[i]]
        pred_lbl = outcome_names[proba[i].argmax()]
        correct = "✓" if pred_lbl == true_lbl else "✗"
        print(f"  {str(row['date'])[:10]:<12} {row['home_team']:<20} {row['away_team']:<20} "
              f"{ph:.3f}  {pd_p:.3f}  {pa:.3f}  {true_lbl:<8} {correct}")


# ── Continental tournament backtest ───────────────────────────────────────────

def run_continental_backtest(name: str, cfg: dict, friendly_weight: float = 0.3,
                             xg_mode: str | None = None) -> dict:
    """Evaluate model on a continental tournament group stage.

    Uses the same train/eval pipeline as WC backtests but with a
    tournament-specific name fragment filter for the evaluation set.
    Provides additional statistical power beyond the 96-match WC sample.
    """
    if xg_mode is None:
        from football_predictor import constants as _c0
        xg_mode = "full" if _c0.USE_XG_OBSERVATION else "baseline"
    print(f"\n{'─'*60}")
    print(f"  Continental backtest: {name}  [xg_mode={xg_mode}]")
    print(f"{'─'*60}")

    all_data = load_results(from_year=2000, friendly_weight=friendly_weight)
    all_data["date"] = pd.to_datetime(all_data["date"])
    all_data["outcome"] = all_data.apply(_outcome_to_int, axis=1)

    cutoff = cfg["cutoff"]
    start, end = cfg["date_range"]
    frag = cfg["name_fragment"]

    context = all_data[all_data["date"] < cutoff].copy()
    eval_mask = (
        all_data["tournament"].str.contains(frag, case=False, na=False)
        & (all_data["date"] >= start)
        & (all_data["date"] <= end)
    )
    test_df = all_data[eval_mask].copy()

    if len(test_df) == 0:
        print(f"  No matches found for '{frag}' in [{start}, {end}].")
        return {}

    print(f"  Training on {len(context)} pre-tournament matches, testing on {len(test_df)} group stage matches.")

    # ── xG observation ablation (see run_backtest for rationale) ───────────────
    import football_predictor.constants as _C
    _prev_xg = _C.USE_XG_OBSERVATION
    use_xg = xg_mode != "baseline"
    if use_xg:
        from football_predictor.data.xg_attach import attach_calibrated_xg
        context = attach_calibrated_xg(context)
        n_xg = int(context[["home_xg", "away_xg"]].notna().all(axis=1).sum())
        print(f"  xG attached to {n_xg}/{len(context)} pre-cutoff training matches.")

    comp_mask = context["tournament"].str.contains(_COMP_FILTER, case=False, na=False)
    train_df = context[comp_mask].copy()

    feature_mods = [m for m in DEFAULT_FEATURE_MODULES if m != "venue_wc2026"]
    _C.USE_XG_OBSERVATION = use_xg
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train, y_train = build_feature_matrix(train_df, feature_mods, context=context)
            X_test_raw, y_test = build_feature_matrix(test_df, feature_mods, context=context)
    finally:
        _C.USE_XG_OBSERVATION = _prev_xg

    X_train, _ = prune_correlated_features(X_train, threshold=0.95)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0.0)
    sw = train_df["match_weight"].values if "match_weight" in train_df else np.ones(len(X_train))

    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    q_tuned = KalmanStrengthFeatures.get_last_tuned_q()
    hl = BayesianPoissonModel.half_life_from_q(q_tuned)

    # use_tuned_cache=False — same leakage rationale as run_backtest.
    def _fit_xgb(X_tr, y_tr, sw_tr):
        m = GradientBoostModel(use_tuned_cache=False)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        return m

    def _fit_bp(matches_tr):
        b = BayesianPoissonModel(half_life_days=hl)
        b.fit(matches_tr)
        return b

    temp_cal, ensemble = fit_stacked_calibration(
        X_train, y_train, train_df.reset_index(drop=True), sw,
        _fit_xgb, _fit_bp,
        lambda b, m: _bp_proba_df(b, m.reset_index(drop=True)),
    )

    # Final models refit on the full pre-tournament window now T and α are locked.
    xgb = _fit_xgb(X_train, y_train, sw)
    bp = _fit_bp(train_df)
    bp.fit_rho(train_df)

    xgb_test = temp_cal.transform(xgb.predict_proba(X_test))
    bp_test = _bp_proba_df(bp, test_df.reset_index(drop=True))
    bp_test.index = test_df.index

    proba_df = ensemble.predict_proba(
        xgb_test.reset_index(drop=True),
        bp_test.reset_index(drop=True),
        context_X=X_test.reset_index(drop=True),
    )
    proba = proba_df.values
    y_true = y_test.values

    uniform_proba = np.full_like(proba, 1.0 / 3.0)
    ll = _log_loss(proba, y_true)
    ll_lo, ll_hi = _bootstrap_ci(proba, y_true, _log_loss)
    metrics = {
        "label": name,
        "n_matches": len(test_df),
        "log_loss": ll,
        "log_loss_ci_lo": ll_lo,
        "log_loss_ci_hi": ll_hi,
        "log_loss_uniform": _log_loss(uniform_proba, y_true),
        "brier_score": _brier(proba, y_true),
        "accuracy": _accuracy(proba, y_true),
        "ece": _compute_ece(proba, y_true),
    }

    metrics["_conf"] = _confederation_slice(test_df, proba, y_true)

    print(f"\n  Results — {name} ({metrics['n_matches']} group-stage matches):")
    print(f"    Log-loss: {ll:.4f}  [{ll_lo:.4f}, {ll_hi:.4f}] 95% CI  (uniform: {metrics['log_loss_uniform']:.4f})")
    print(f"    Brier:    {metrics['brier_score']:.4f}    Accuracy: {metrics['accuracy']:.3f}    ECE: {metrics['ece']:.4f}")
    return metrics


# ── Friendly weight tuning ────────────────────────────────────────────────────

def tune_friendly_weight(years: list[int]) -> float:
    """Grid-search friendly_weight, minimising avg log-loss across WC backtests.

    The selected value is persisted to data/tuned_params.json so
    pipeline.py / predict_wc2026.py / generate_submission.py all pick it up
    (previously the result was printed and discarded — nothing ever wrote
    the cache, so '--retune' was a silent no-op).

    NOTE: weights selected on years that are also the reported evaluation
    make those backtest numbers optimistic — prefer disjoint tuning years
    (e.g. tune on 2014 + continental, report on 2018/2022).
    """
    grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    print(f"\n{'═'*60}")
    print(f"  Tuning friendly_weight — years: {years}")
    print(f"{'═'*60}")
    print(f"  {'weight':>8}  {'avg log-loss':>14}")
    print(f"  {'─'*8}  {'─'*14}")

    best_w, best_ll = 0.3, float("inf")
    results = []
    for w in grid:
        losses = []
        for year in years:
            m = run_backtest(year, include_shap=False, friendly_weight=w)
            if m:
                losses.append(m["log_loss"])
        if losses:
            avg = float(np.mean(losses))
            marker = " ← best" if avg < best_ll else ""
            print(f"  {w:>8.2f}  {avg:>14.4f}{marker}")
            results.append((w, avg))
            if avg < best_ll:
                best_ll, best_w = avg, w

    print(f"\n  Best friendly_weight: {best_w}  (avg log-loss: {best_ll:.4f})")

    # Persist so all entry points train with the tuned value.
    import json as _json
    tp = ROOT / "data" / "tuned_params.json"
    try:
        cur = _json.loads(tp.read_text()) if tp.exists() else {}
    except Exception:
        cur = {}
    cur["friendly_weight"] = best_w
    cur["friendly_weight_tuned_on"] = [str(y) for y in years]
    tp.parent.mkdir(exist_ok=True)
    tp.write_text(_json.dumps(cur, indent=2))
    print(f"  Persisted to {tp}")
    return best_w


# ── xG observation ablation ───────────────────────────────────────────────────

def run_xg_ablation(years: list[int], friendly_weight: float = 0.3,
                    include_shap: bool = False, continental: bool = False) -> None:
    """Baseline (goals) vs full (goals/xG blend) Kalman observation, per fold.

    Decision gate for USE_XG_OBSERVATION: keep only if log-loss improves
    consistently and the improvement is outside bootstrap-CI noise. Folds with no
    pre-cutoff xG coverage are reported as untestable rather than silently shown
    as "no change". With ``continental=True`` the AFCON 2022 / Asian Cup 2024
    folds add CAF/AFC test power — exactly the gap the StatsBomb data fills.
    """
    print(f"\n{'═'*72}")
    print("  xG-OBSERVATION ABLATION  (baseline = goals,  full = goals/xG blend)")
    print(f"{'═'*72}")

    rows = []
    for year in years:
        base = run_backtest(year, include_shap=False, label=f"{year}_baseline",
                            friendly_weight=friendly_weight, xg_mode="baseline")
        full = run_backtest(year, include_shap=include_shap, label=f"{year}_full",
                            friendly_weight=friendly_weight, xg_mode="full")
        if base and full:
            rows.append((year, base, full))

    if continental:
        for name, cfg in CONTINENTAL_CONFIGS.items():
            base = run_continental_backtest(name, cfg, friendly_weight, xg_mode="baseline")
            full = run_continental_backtest(name, cfg, friendly_weight, xg_mode="full")
            if base and full:
                rows.append((name, base, full))

    print(f"\n{'═'*72}")
    print("  SUMMARY — log-loss (lower is better)")
    print(f"  {'Year':<6} {'baseline':>10} {'full(xG)':>10} {'Δ':>9} {'baseline 95% CI':>22} {'verdict'}")
    print("  " + "-" * 78)
    for year, base, full in rows:
        d = full["log_loss"] - base["log_loss"]
        ci = f"[{base['log_loss_ci_lo']:.3f},{base['log_loss_ci_hi']:.3f}]"
        # "real" improvement if full's point estimate falls below baseline CI lower bound
        if abs(d) < 1e-6:
            verdict = "no-op (no xG coverage)"
        elif d < 0 and full["log_loss"] < base["log_loss_ci_lo"]:
            verdict = "IMPROVES (outside CI)"
        elif d < 0:
            verdict = "better (within noise)"
        else:
            verdict = "worse"
        print(f"  {year:<6} {base['log_loss']:>10.4f} {full['log_loss']:>10.4f} "
              f"{d:>+9.4f} {ci:>22}  {verdict}")

    # Per-confederation deltas (where the StatsBomb gap-fill should show up).
    print("\n  Per-confederation log-loss Δ (full − baseline; negative = better):")
    print(f"  {'Year':<6} {'Conf':<9} {'baseline':>10} {'full':>10} {'Δ':>9} {'n':>4}")
    print("  " + "-" * 56)
    for year, base, full in rows:
        bc, fc = base.get("_conf", {}), full.get("_conf", {})
        for conf in fc:
            if conf in bc:
                d = fc[conf]["log_loss"] - bc[conf]["log_loss"]
                print(f"  {year:<6} {conf:<9} {bc[conf]['log_loss']:>10.4f} "
                      f"{fc[conf]['log_loss']:>10.4f} {d:>+9.4f} {fc[conf]['n']:>4}")

    print("\n  DECISION RULE: set constants.USE_XG_OBSERVATION = True only if 'full'")
    print("  improves log-loss consistently and outside CI on years with xG coverage.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest football predictor on past World Cups")
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2022])
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP computation (faster)")
    parser.add_argument("--continental", action="store_true",
                        help="Also backtest on Copa América 2021, Euro 2020, AFCON 2022, Asian Cup 2023")
    parser.add_argument("--tune-friendly-weight", action="store_true",
                        help="Grid-search friendly_weight and report optimal value")
    parser.add_argument("--friendly-weight", type=float, default=0.3,
                        help="Friendly match weight (default 0.3, override or use --tune-friendly-weight)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run baseline vs xG-observation (full) per year and compare")
    args = parser.parse_args()

    if args.tune_friendly_weight:
        tune_friendly_weight(args.years)
        return

    if args.ablation:
        run_xg_ablation(args.years, friendly_weight=args.friendly_weight,
                        include_shap=not args.no_shap, continental=args.continental)
        return

    all_metrics = []
    for year in args.years:
        m = run_backtest(year, include_shap=not args.no_shap, friendly_weight=args.friendly_weight)
        if m:
            all_metrics.append({**m, "label": f"WC {m['year']}"})

    if args.continental:
        for name, cfg in CONTINENTAL_CONFIGS.items():
            m = run_continental_backtest(name, cfg, friendly_weight=args.friendly_weight)
            if m:
                all_metrics.append(m)

    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("  Aggregate evaluation across all backtests:")
        print(f"  {'Tournament':<22} {'N':>4} {'LogLoss':>8} {'95% CI':>16} {'Uniform':>8} {'Acc':>6}")
        print("  " + "-" * 68)
        for m in all_metrics:
            label = m.get("label", str(m.get("year", "?")))
            ci = f"[{m.get('log_loss_ci_lo', 0):.3f},{m.get('log_loss_ci_hi', 0):.3f}]"
            print(f"  {label:<22} {m['n_matches']:>4} {m['log_loss']:>8.4f} {ci:>16} "
                  f"{m['log_loss_uniform']:>8.4f} {m['accuracy']:>6.3f}")

        avg_ll = np.mean([m["log_loss"] for m in all_metrics])
        avg_uni = np.mean([m["log_loss_uniform"] for m in all_metrics])
        total_n = sum(m["n_matches"] for m in all_metrics)
        print(f"\n  Average log-loss: {avg_ll:.4f}  (uniform avg: {avg_uni:.4f})  N={total_n} matches")


if __name__ == "__main__":
    main()
