#!/usr/bin/env python3
"""
decision_tree_regressor.py

Trains a DecisionTreeRegressor on Flood_Impact_Index_Norm
using Model_Input_v3.csv with:
 - SAME FEATURES as XGB/LGBM/SVR (11 features)
 - Human_fatality, Human_injured, Flood_Impact_Index removed
 - Median imputation (train-only)
 - 70/15/15 Train/Val/Test split
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
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

# Ensure target numeric
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

# ---- STEP 1: start like XGB/LGBM
# numeric columns only, excluding target
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
y = df[target_col].copy()

# ---- STEP 2: explicitly remove human-impact & raw impact
manual_drop = [
    "Human_fatality",
    "Human_injured",
    "Flood_Impact_Index",
]

for col in manual_drop:
    if col in X.columns:
        X.drop(columns=[col], inplace=True)

# ---- STEP 3: drop constant / near-constant columns (same logic as XGB)
nunique = X.nunique()
to_drop = nunique[nunique <= 1].index.tolist()
if to_drop:
    print("Dropping constant columns:", to_drop)
    X.drop(columns=to_drop, inplace=True)

print("\nFINAL FEATURES USED FOR DT TRAINING:")
print(list(X.columns))
print("Number of features:", len(X.columns))

# ------------- Split data --------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\nSplit shapes:")
print(" Train:", X_train.shape)
print(" Val:  ", X_val.shape)
print(" Test: ", X_test.shape)

# ------------- Impute ------------------
# Drop any column that is entirely NaN in TRAIN
cols_to_keep = [c for c in X_train.columns if not X_train[c].isna().all()]
X_train = X_train[cols_to_keep]
X_val   = X_val[cols_to_keep]
X_test  = X_test[cols_to_keep]

print(f"\nAfter dropping all-NaN columns, feature count: {len(cols_to_keep)}")

# Median imputation (fit on train only)
imputer = SimpleImputer(strategy="median")
X_train_imp_arr = imputer.fit_transform(X_train)
X_val_imp_arr   = imputer.transform(X_val)
X_test_imp_arr  = imputer.transform(X_test)

# Back to DataFrames
X_train_imp = pd.DataFrame(X_train_imp_arr, columns=X_train.columns, index=X_train.index)
X_val_imp   = pd.DataFrame(X_val_imp_arr,   columns=X_val.columns,   index=X_val.index)
X_test_imp  = pd.DataFrame(X_test_imp_arr,  columns=X_test.columns,  index=X_test.index)

# Save imputer
joblib.dump(imputer, os.path.join(MODEL_DIR, "dt_imputer_v3.pkl"))

# ------------- Train Decision Tree -----
dt = DecisionTreeRegressor(max_depth=8, random_state=42)

print("\nTraining DecisionTreeRegressor...")
dt.fit(X_train_imp, y_train)

train_pred = dt.predict(X_train_imp)
val_pred   = dt.predict(X_val_imp)
test_pred  = dt.predict(X_test_imp)

train_mae, train_rmse, train_r2 = evaluate(y_train, train_pred)
val_mae,   val_rmse,   val_r2   = evaluate(y_val,   val_pred)
test_mae,  test_rmse,  test_r2  = evaluate(y_test,  test_pred)

print("\nDecision Tree Performance:")
print(f"Train → MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}")
print(f"Val   → MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, R²: {val_r2:.6f}")
print(f"Test  → MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}")

# ------------- Save model + results ----
model_path = os.path.join(MODEL_DIR, "dt_model_v3.pkl")
joblib.dump(dt, model_path)
print("\nSaved Decision Tree model ->", model_path)

results = pd.DataFrame([
    {"Model": "DT", "Split": "Train", "MAE": train_mae, "RMSE": train_rmse, "R2": train_r2},
    {"Model": "DT", "Split": "Val",   "MAE": val_mae,   "RMSE": val_rmse,   "R2": val_r2},
    {"Model": "DT", "Split": "Test",  "MAE": test_mae,  "RMSE": test_rmse,  "R2": test_r2},
])

results_path = os.path.join(RESULTS_DIR, "dt_results_v3.csv")
results.to_csv(results_path, index=False)
print("Saved Decision Tree results ->", results_path)
