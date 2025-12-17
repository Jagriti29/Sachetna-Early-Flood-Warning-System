#!/usr/bin/env python3
"""
svm_regressor.py

Trains an SVR (Support Vector Regressor) on Flood_Impact_Index_Norm
using Model_Input_v3.csv with:
 - Numeric features only
 - Leaky columns removed
 - Median imputation (train-only)
 - Standard scaling (train-only)
 - 70/15/15 Train/Val/Test split
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Paths ----------------
INPUT = "data/processed/Model_Input_v3.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------- Helper ------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ------------- Load data ---------------
print(f"Loading: {INPUT}")
df = pd.read_csv(INPUT)
print("Shape:", df.shape)

target_col = "Flood_Impact_Index_Norm"
if target_col not in df.columns:
    raise KeyError(f"{target_col} not found in {INPUT}")

# Ensure target is numeric
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

# Numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Leaky / target-like columns to EXCLUDE from features
leak_cols = [
    "Area_Exposure",
    "Flood_Exposure_Score",
    "Flood_Impact_Index",
    "Population_Exposure_Ratio",
    target_col,
]

feature_cols = [c for c in numeric_cols if c not in leak_cols]

if not feature_cols:
    raise ValueError("No numeric features left after dropping leak columns.")

print("Using features:", feature_cols)
print("Number of features:", len(feature_cols))

X = df[feature_cols].copy()
y = df[target_col].copy()

# ------------- Split data --------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train shape:", X_train.shape)
print("Val shape:  ", X_val.shape)
print("Test shape: ", X_test.shape)

# ------------- Impute + Scale ----------
# Drop columns that are entirely NaN (these cause shape mismatches)
cols_to_keep = [c for c in X_train.columns if not X_train[c].isna().all()]
X_train = X_train[cols_to_keep]
X_val = X_val[cols_to_keep]
X_test = X_test[cols_to_keep]

print(f"After dropping all-NaN columns, feature count: {len(cols_to_keep)}")

# Fit imputer ONLY on train
imputer = SimpleImputer(strategy="median")
X_train_imp_arr = imputer.fit_transform(X_train)
X_val_imp_arr   = imputer.transform(X_val)
X_test_imp_arr  = imputer.transform(X_test)

# Back to DataFrame (shapes must match)
X_train_imp = pd.DataFrame(X_train_imp_arr, columns=X_train.columns, index=X_train.index)
X_val_imp   = pd.DataFrame(X_val_imp_arr, columns=X_val.columns, index=X_val.index)
X_test_imp  = pd.DataFrame(X_test_imp_arr, columns=X_test.columns, index=X_test.index)

# Scale for SVR
scaler = StandardScaler()
X_train_scaled_arr = scaler.fit_transform(X_train_imp.values)
X_val_scaled_arr   = scaler.transform(X_val_imp.values)
X_test_scaled_arr  = scaler.transform(X_test_imp.values)

X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train_imp.columns, index=X_train_imp.index)
X_val_scaled   = pd.DataFrame(X_val_scaled_arr, columns=X_val_imp.columns, index=X_val_imp.index)
X_test_scaled  = pd.DataFrame(X_test_scaled_arr, columns=X_test_imp.columns, index=X_test_imp.index)

# Save preprocessors
joblib.dump(imputer, os.path.join(MODEL_DIR, "svr_imputer.pkl"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "svr_scaler.pkl"))

# ------------- Train SVR ----------------
svr = SVR(kernel="rbf", C=1.0, epsilon=0.01)

print("\nTraining SVR...")
svr.fit(X_train_scaled, y_train)

train_pred = svr.predict(X_train_scaled)
val_pred   = svr.predict(X_val_scaled)
test_pred  = svr.predict(X_test_scaled)

train_mae, train_rmse, train_r2 = evaluate(y_train, train_pred)
val_mae,   val_rmse,   val_r2   = evaluate(y_val,   val_pred)
test_mae,  test_rmse,  test_r2  = evaluate(y_test,  test_pred)

print("\nSVR Performance:")
print(f"Train → MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}")
print(f"Val   → MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, R²: {val_r2:.6f}")
print(f"Test  → MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}")

# ------------- Save model + results -----
model_path = os.path.join(MODEL_DIR, "svr_model_v3.pkl")
joblib.dump(svr, model_path)
print("\nSaved SVR model ->", model_path)

results = pd.DataFrame([{
    "Model": "SVR",
    "Split": "Train", "MAE": train_mae, "RMSE": train_rmse, "R2": train_r2
}, {
    "Model": "SVR",
    "Split": "Val", "MAE": val_mae, "RMSE": val_rmse, "R2": val_r2
}, {
    "Model": "SVR",
    "Split": "Test", "MAE": test_mae, "RMSE": test_rmse, "R2": test_r2
}])

results_path = os.path.join(RESULTS_DIR, "svr_results_v3.csv")
results.to_csv(results_path, index=False)
print("Saved SVR results ->", results_path)
