"""
Model v3 training (clean, no leakage):
 - Loads data/processed/Model_Input_v3.csv
 - Uses only non-leaky numeric features
 - Trains XGBoost and LightGBM on Flood_Impact_Index_Norm
 - Creates simple ensemble by averaging predictions
 - Evaluates (MAE, RMSE, R2) on Train / Val / Test
 - Saves models, results, and plots (feature importance + actual vs predicted)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- Paths ---
INPUT = "data/processed/Model_Input_v3.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/results"
ANALYSIS_DIR = "data/analysis"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(INPUT)
print(f"Loaded: {INPUT} -> {df.shape}")

target_col = "Flood_Impact_Index_Norm"
if target_col not in df.columns:
    raise KeyError(f"Target {target_col} not found in {INPUT}")

# --- Start from numeric columns, drop target from features ---
numeric_df = df.select_dtypes(include=[np.number]).copy()

if target_col not in numeric_df.columns:
    raise ValueError(f"{target_col} is not numeric in the dataset")

y = numeric_df[target_col].copy()

X = numeric_df.drop(columns=[target_col])

# --- Explicitly drop LEAKAGE / post-impact columns ---
# These either are the raw target or represent realized damage, not early signals.
leak_cols = [
    "Flood_Impact_Index",   # direct parent of the target
    "Human_fatality",       # post-event outcome
    "Human_injured",        # post-event outcome
    "Severity_Score"        # likely composite impact score
]

leak_cols_present = [c for c in leak_cols if c in X.columns]
if leak_cols_present:
    print("Dropping leakage / post-impact columns from features:", leak_cols_present)
    X = X.drop(columns=leak_cols_present)

# --- Optionally drop constant or near-constant columns ---
nunique = X.nunique()
to_drop = nunique[nunique <= 1].index.tolist()
if to_drop:
    X.drop(columns=to_drop, inplace=True)
    print("Dropped constant cols:", to_drop)

print("Final feature shape:", X.shape)
print("Final feature list used for training:")
for col in X.columns:
    print("  -", col)

# --- Train/Val/Test split (70/15/15) ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# --- Helper: evaluate ---
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# --- Model 1: XGBoost ---
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
print("\nTraining XGBoost...")
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_train_pred = xgb.predict(X_train)
xgb_val_pred   = xgb.predict(X_val)
xgb_test_pred  = xgb.predict(X_test)

# --- Model 2: LightGBM ---
lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
print("\nTraining LightGBM...")
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

lgbm_train_pred = lgbm.predict(X_train)
lgbm_val_pred   = lgbm.predict(X_val)
lgbm_test_pred  = lgbm.predict(X_test)

# --- Ensemble (simple average) ---
ens_train_pred = (xgb_train_pred + lgbm_train_pred) / 2.0
ens_val_pred   = (xgb_val_pred + lgbm_val_pred) / 2.0
ens_test_pred  = (xgb_test_pred + lgbm_test_pred) / 2.0

# --- Evaluate and collect results ---
records = []
for name, y_t, y_p in [
    ("XGB Train", y_train, xgb_train_pred),
    ("XGB Val",   y_val,   xgb_val_pred),
    ("XGB Test",  y_test,  xgb_test_pred),

    ("LGBM Train", y_train, lgbm_train_pred),
    ("LGBM Val",   y_val,   lgbm_val_pred),
    ("LGBM Test",  y_test,  lgbm_test_pred),

    ("ENS Train", y_train, ens_train_pred),
    ("ENS Val",   y_val,   ens_val_pred),
    ("ENS Test",  y_test,  ens_test_pred),
]:
    mae, rmse, r2 = evaluate(y_t, y_p)
    print(f"{name:10s} → MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
    records.append({
        "Model": name.split()[0],
        "Split": name.split()[1],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(records)
results_csv = os.path.join(RESULTS_DIR, "xgb1_results_clean.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nSaved results: {results_csv}")

# --- Save models ---
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb2_model.pkl"))
joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm2_model.pkl"))
joblib.dump({"xgb": xgb, "lgbm": lgbm}, os.path.join(MODEL_DIR, "ensemble_2.pkl"))

print("Saved models to models/")

# --- Feature importance (from LGBM & XGB) ---
fi = pd.DataFrame({
    "feature": X.columns,
    "lgbm_importance": lgbm.feature_importances_,
    "xgb_importance": xgb.feature_importances_
})
fi["avg_importance"] = (fi["lgbm_importance"] + fi["xgb_importance"]) / 2.0
fi = fi.sort_values("avg_importance", ascending=False)

fi_csv = os.path.join(ANALYSIS_DIR, "feature_importances_v3_clean.csv")
fi.to_csv(fi_csv, index=False)
print(f"Saved feature importances: {fi_csv}")

# Bar plot of top features
top_n = min(20, fi.shape[0])
plt.figure(figsize=(10,6))
plt.barh(fi["feature"].head(top_n)[::-1], fi["avg_importance"].head(top_n)[::-1])
plt.title("Top feature importances (avg of LGBM & XGB) — CLEAN")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "feature_importances_v3_clean.png"))
plt.close()

# --- Actual vs Predicted (Test set, ensemble) ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, ens_test_pred, s=10, alpha=0.6)
plt.plot([0,1], [0,1], "r--")
plt.xlabel("Actual Flood_Impact_Index_Norm")
plt.ylabel("Predicted (Ensemble)")
plt.title("Actual vs Predicted (Test) - Ensemble (Clean Features)")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "actual_vs_pred_ens_v3_clean.png"))
plt.close()

print("Saved plots to data/analysis/")
print("Training complete (clean, no target leakage).")
