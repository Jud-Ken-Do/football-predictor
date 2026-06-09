"""Evaluate model on WC 2022. Run: python3.11 scripts/evaluate_wc2022.py"""
import warnings; warnings.filterwarnings("ignore")
import time
import pandas as pd

from football_predictor.data.sources.international_results import fetch_training_data, fetch_world_cup_matches
from football_predictor.data.pipeline import build_feature_matrix
from football_predictor.models.gradient_boost import GradientBoostModel
from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.calibration import CalibrationLayer
from football_predictor.evaluate.metrics import summary
from football_predictor.constants import DEFAULT_FEATURE_MODULES

print("Loading data...")
all_data = fetch_training_data(from_year=2010)
train_mask = pd.to_datetime(all_data["date"]) < pd.Timestamp("2022-11-20")
comp_mask = all_data["tournament"].str.lower().str.contains(
    "qualif|world cup|copa|euro|nations|africa|asian|gold cup", na=False
)
train_df = all_data[train_mask & comp_mask].reset_index(drop=True)
wc_df = fetch_world_cup_matches()
test_df = wc_df[pd.to_datetime(wc_df["date"]).dt.year == 2022].reset_index(drop=True)
print(f"Train: {len(train_df)}  |  Test: {len(test_df)}  |  Modules: {DEFAULT_FEATURE_MODULES}")

t0 = time.time()
print("Building train features...")
X_train, y_train = build_feature_matrix(train_df, DEFAULT_FEATURE_MODULES, context=all_data)
print(f"  done in {time.time()-t0:.1f}s  —  {X_train.shape[1]} features")

print("Building test features...")
t1 = time.time()
X_test, y_test = build_feature_matrix(test_df, DEFAULT_FEATURE_MODULES, context=all_data)
print(f"  done in {time.time()-t1:.1f}s")

print("Training XGBoost...")
sw = train_df["match_weight"].values
# 80/20 split: model trains on 80%, calibration on held-out 20% to avoid overfitting
cut = int(len(X_train) * 0.8)
X_tr, X_cal = X_train.iloc[:cut], X_train.iloc[cut:]
y_tr, y_cal = y_train.iloc[:cut], y_train.iloc[cut:]
sw_tr = sw[:cut]
model = GradientBoostModel()
model.fit(X_tr, y_tr, sample_weight=sw_tr)
calibrator = CalibrationLayer()
calibrator.fit(model.predict_proba(X_cal), y_cal)
test_proba = calibrator.transform(model.predict_proba(X_test))
metrics = summary(test_proba, y_test)
print(f"  trained on {len(X_tr)}, calibrated on {len(X_cal)}")

print("Dixon-Coles baseline...")
dc = DixonColesModel()
dc.fit(train_df)
dc_probas = pd.DataFrame([
    dc.predict_proba(r["home_team"], r["away_team"], neutral=True)
    for _, r in test_df.iterrows()
])
dc_metrics = summary(dc_probas[["home_win", "draw", "away_win"]], y_test)

print()
print("=== WC 2022 Evaluation ===")
for k, v in metrics.items():
    print(f"  {k:35s} {v:.4f}")
delta = dc_metrics["rps"] - metrics["rps"]
print(f"\n  RPS Dixon-Coles: {dc_metrics['rps']:.4f}")
print(f"  RPS XGBoost+cal: {metrics['rps']:.4f}")
print(f"  Improvement:     {delta:+.4f}  ({'BETTER' if delta > 0 else 'WORSE'} than baseline)")

fi = model.feature_importances()
print("\n=== Top 15 features ===")
for feat, imp in fi.head(15).items():  # type: ignore[union-attr]
    print(f"  {imp:.4f}  {feat}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
