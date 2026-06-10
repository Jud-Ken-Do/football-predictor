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

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTCOME_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

# WC group stage date ranges
WC_DATE_RANGES = {
    2014: ("2014-06-12", "2014-06-26"),
    2018: ("2018-06-14", "2018-06-28"),
    2022: ("2022-11-20", "2022-12-09"),
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
        "date_range": ("2021-06-13", "2021-07-02"),  # group stage only
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

def run_backtest(year: int, include_shap: bool = True, label: str = "") -> dict:
    print(f"\n{'─'*60}")
    print(f"  Backtest: WC {year} Group Stage")
    print(f"{'─'*60}")

    if year not in WC_DATE_RANGES:
        print(f"  No WC date range defined for {year}. Supported: {list(WC_DATE_RANGES)}")
        return {}

    # ── Load data ─────────────────────────────────────────────────────────────
    print("  Loading historical results...")
    all_data = load_results(from_year=2000)
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

    # ── Training data: competitive matches only ───────────────────────────────
    comp_mask = context["tournament"].str.contains(_COMP_FILTER, case=False, na=False)
    train_df = context[comp_mask].copy()

    # ── Build feature matrices ────────────────────────────────────────────────
    print("  Building feature matrices...")
    feature_mods = [m for m in DEFAULT_FEATURE_MODULES if m != "venue_wc2026"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, y_train = build_feature_matrix(train_df, feature_mods, context=context)
        X_test_raw, y_test = build_feature_matrix(test_df, feature_mods, context=context)

    X_train, _ = prune_correlated_features(X_train, threshold=0.95)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0.0)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    print("  Training XGBoost + temperature scaling...")
    sw = train_df["match_weight"].values if "match_weight" in train_df else np.ones(len(X_train))
    cut = int(len(X_train) * 0.85)

    xgb = GradientBoostModel()
    xgb.fit(X_train.iloc[:cut], y_train.iloc[:cut], sample_weight=sw[:cut])

    xgb_cal = xgb.predict_proba(X_train.iloc[cut:])
    temp_cal = TemperatureScaling()
    temp_cal.fit(xgb_cal, y_train.iloc[cut:])

    # ── Train BayesPoisson ────────────────────────────────────────────────────
    print("  Training BayesianPoisson (MAP)...")
    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    q_tuned = KalmanStrengthFeatures.get_last_tuned_q()
    hl = BayesianPoissonModel.half_life_from_q(q_tuned)
    print(f"  BayesPoisson half-life: {hl}d  (Kalman q={q_tuned:.4f}/yr)")
    bp = BayesianPoissonModel(half_life_days=hl)
    bp.fit(train_df)
    bp.fit_rho(train_df)

    cal_idx = X_train.iloc[cut:].index
    bp_cal_df = _bp_proba_df(bp, train_df.loc[cal_idx].reset_index(drop=True))
    bp_cal_df.index = y_train.iloc[cut:].index

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("  Fitting ensemble (α * XGB + (1-α) * BayesPoisson)...")
    ensemble = EnsembleModel()
    ensemble.fit(xgb_cal, bp_cal_df, y_train.iloc[cut:], context_X=X_train.iloc[cut:].reset_index(drop=True))

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

def run_continental_backtest(name: str, cfg: dict) -> dict:
    """Evaluate model on a continental tournament group stage.

    Uses the same train/eval pipeline as WC backtests but with a
    tournament-specific name fragment filter for the evaluation set.
    Provides additional statistical power beyond the 96-match WC sample.
    """
    print(f"\n{'─'*60}")
    print(f"  Continental backtest: {name}")
    print(f"{'─'*60}")

    all_data = load_results(from_year=2000)
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

    comp_mask = context["tournament"].str.contains(_COMP_FILTER, case=False, na=False)
    train_df = context[comp_mask].copy()

    feature_mods = [m for m in DEFAULT_FEATURE_MODULES if m != "venue_wc2026"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, y_train = build_feature_matrix(train_df, feature_mods, context=context)
        X_test_raw, y_test = build_feature_matrix(test_df, feature_mods, context=context)

    X_train, _ = prune_correlated_features(X_train, threshold=0.95)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0.0)
    sw = train_df["match_weight"].values if "match_weight" in train_df else np.ones(len(X_train))
    cut = int(len(X_train) * 0.85)

    xgb = GradientBoostModel()
    xgb.fit(X_train.iloc[:cut], y_train.iloc[:cut], sample_weight=sw[:cut])
    xgb_cal = xgb.predict_proba(X_train.iloc[cut:])
    temp_cal = TemperatureScaling()
    temp_cal.fit(xgb_cal, y_train.iloc[cut:])

    from football_predictor.features.kalman_strength import KalmanStrengthFeatures
    q_tuned = KalmanStrengthFeatures.get_last_tuned_q()
    hl = BayesianPoissonModel.half_life_from_q(q_tuned)
    bp = BayesianPoissonModel(half_life_days=hl)
    bp.fit(train_df)
    bp.fit_rho(train_df)

    cal_idx = X_train.iloc[cut:].index
    bp_cal_df = _bp_proba_df(bp, train_df.loc[cal_idx].reset_index(drop=True))
    bp_cal_df.index = y_train.iloc[cut:].index

    ensemble = EnsembleModel()
    ensemble.fit(xgb_cal, bp_cal_df, y_train.iloc[cut:], context_X=X_train.iloc[cut:].reset_index(drop=True))

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

    print(f"\n  Results — {name} ({metrics['n_matches']} group-stage matches):")
    print(f"    Log-loss: {ll:.4f}  [{ll_lo:.4f}, {ll_hi:.4f}] 95% CI  (uniform: {metrics['log_loss_uniform']:.4f})")
    print(f"    Brier:    {metrics['brier_score']:.4f}    Accuracy: {metrics['accuracy']:.3f}    ECE: {metrics['ece']:.4f}")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest football predictor on past World Cups")
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP computation (faster)")
    parser.add_argument("--continental", action="store_true",
                        help="Also backtest on Copa América 2021, Euro 2020, AFCON 2022, Asian Cup 2023")
    args = parser.parse_args()

    all_metrics = []
    for year in args.years:
        m = run_backtest(year, include_shap=not args.no_shap)
        if m:
            all_metrics.append({**m, "label": f"WC {m['year']}"})

    if args.continental:
        for name, cfg in CONTINENTAL_CONFIGS.items():
            m = run_continental_backtest(name, cfg)
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
