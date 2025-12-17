#!/usr/bin/env python3
"""
svm_regressor_v3.py

Trains an SVR using the EXACT SAME FEATURES as XGB/LGBM,
but additionally removes:
 - Human_fatality
 - Human_injured
 - Flood_Impact_Index

Final features = 11 numeric, non-leaky.
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
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

# ---- STEP 1:  
# Use same starting point as XGB: numeric columns except target
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
y = df[target_col].copy()

# ---- STEP 2:
# Remove human-damage & original impact column (your requirement)
manual_drop = [
    "Human_fatality",
    "Human_injured",
    "Flood_Impact_Index"
]

for col in manual_drop:
    if col in X.columns:
        X.drop(columns=[col], inplace=True)

# ---- STEP 3: drop constant columns (same logic as XGB)
nunique = X.nunique()
to_drop = nunique[nunique <= 1].index.tolist()

if to_drop:
    print("Dropping constant columns (same as XGB):", to_drop)
    X.drop(columns=to_drop, inplace=True)

print("\nFINAL FEATURES USED FOR TRAINING:")
print(list(X.columns))
print("Count:", len(X.columns))

# Expected: 11 features

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

# ------------- Impute + Scale ----------
# Drop columns that are fully NaN
cols_to_keep = [c for c in X_train.columns if not X_train[c].isna().all()]
X_train = X_train[cols_to_keep]
X_val   = X_val[cols_to_keep]
X_test  = X_test[cols_to_keep]

# Fit imputer on TRAIN
imputer = SimpleImputer(strategy="median")
X_train_imp_arr = imputer.fit_transform(X_train)
X_val_imp_arr   = imputer.transform(X_val)
X_test_imp_arr  = imputer.transform(X_test)

# Convert back to DF
X_train_imp = pd.DataFrame(X_train_imp_arr, columns=X_train.columns, index=X_train.index)
X_val_imp   = pd.DataFrame(X_val_imp_arr,   columns=X_val.columns,   index=X_val.index)
X_test_imp  = pd.DataFrame(X_test_imp_arr,  columns=X_test.columns,  index=X_test.index)

# Scale for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled   = scaler.transform(X_val_imp)
X_test_scaled  = scaler.transform(X_test_imp)

joblib.dump(imputer, os.path.join(MODEL_DIR, "svr_imputer_v3.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "svr_scaler_v3.pkl"))

# ------------- Train SVR ----------------
svr = SVR(kernel="rbf", C=1.0, epsilon=0.01)

print("\nTraining SVR...")
svr.fit(X_train_scaled, y_train)

train_pred = svr.predict(X_train_scaled)
val_pred   = svr.predict(X_val_scaled)
test_pred  = svr.predict(X_test_scaled)

# Evaluation
print("\nSVR Results:")
for name, true, pred in [
    ("Train", y_train, train_pred),
    ("Val", y_val, val_pred),
    ("Test", y_test, test_pred)
]:
    mae, rmse, r2 = evaluate(true, pred)
    print(f"{name} → MAE={mae:.5f}, RMSE={rmse:.5f}, R²={r2:.5f}")

# Save model + results
joblib.dump(svr, os.path.join(MODEL_DIR, "svr_model_v3.pkl"))

results = pd.DataFrame({
    "Split": ["Train", "Val", "Test"],
    "MAE": [evaluate(y_train, train_pred)[0], evaluate(y_val, val_pred)[0], evaluate(y_test, test_pred)[0]],
    "RMSE": [evaluate(y_train, train_pred)[1], evaluate(y_val, val_pred)[1], evaluate(y_test, test_pred)[1]],
    "R2": [evaluate(y_train, train_pred)[2], evaluate(y_val, val_pred)[2], evaluate(y_test, test_pred)[2]],
})

results.to_csv(os.path.join(RESULTS_DIR, "svr_results_v3.csv"), index=False)
print("\nSaved results and model.")

