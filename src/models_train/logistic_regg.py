#!/usr/bin/env python3
"""
logistic_regression_classifier.py

Multiclass Logistic Regression for flood impact *classes* (Low/Medium/High)
based on Flood_Impact_Index_Norm from Model_Input_v3.csv

- Target: Flood_Impact_Class  (0 = Low, 1 = Medium, 2 = High)
- Features: SAME 11 FEATURES as XGB/LGBM regression model:
    ['Year',
     'Flood_Frequency',
     'Mean_Duration',
     'Population',
     'Mean_Flood_Duration',
     'Percent_Flooded_Area',
     'Parmanent_Water',
     'Corrected_Percent_Flooded_Area',
     'Population_Exposure_Ratio',
     'Area_Exposure',
     'Flood_Exposure_Score']
- Steps:
    * Load Model_Input_v3.csv
    * Create class labels from Flood_Impact_Index_Norm (3 bins)
    * Use only the 11 agreed features (NO target leakage)
    * Median imputation (train-only)
    * Standard scaling (train-only)
    * 70/15/15 Train / Val / Test split
    * Train multinomial Logistic Regression
    * Report Accuracy & Macro-F1
    * Save model + preprocessors + results
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# ---------------- Paths ----------------
INPUT = "data/processed/Model_Input_v3.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------- Helper ------------------
def print_split_counts(name, y):
    values, counts = np.unique(y, return_counts=True)
    dist = dict(zip(values, counts))
    print(f" {name}: {y.shape[0]} samples | class counts: {dist}")

# ------------- Load data ---------------
print(f"Loading: {INPUT}")
df = pd.read_csv(INPUT)
print("Shape:", df.shape)

target_reg = "Flood_Impact_Index_Norm"
if target_reg not in df.columns:
    raise KeyError(f"{target_reg} not found in {INPUT}")

# Ensure target_reg is numeric
df[target_reg] = pd.to_numeric(df[target_reg], errors="coerce")

# Drop rows where target is missing
df = df.dropna(subset=[target_reg]).reset_index(drop=True)
print("After dropping rows with missing Flood_Impact_Index_Norm:", df.shape)

# ------------- Create 3-class target -------------
# You can adjust strategy: here we use 33% / 66% quantiles like earlier.
q1, q2 = df[target_reg].quantile([0.33, 0.66])

print("\nClass thresholds based on Flood_Impact_Index_Norm:")
print(f"  Low    (class 0): <= {q1:.6f}")
print(f"  Medium (class 1): > {q1:.6f} and <= {q2:.6f}")
print(f"  High   (class 2): > {q2:.6f}")

def assign_class(v):
    if v <= q1:
        return 0
    elif v <= q2:
        return 1
    else:
        return 2

df["Flood_Impact_Class"] = df[target_reg].apply(assign_class).astype(int)

# ------------- Select EXACT 11 features -------------
# These are the same features you used for XGB/LGBM regression training
desired_features = [
    "Year",
    "Flood_Frequency",
    "Mean_Duration",
    "Population",
    "Mean_Flood_Duration",
    "Percent_Flooded_Area",
    "Parmanent_Water",
    "Corrected_Percent_Flooded_Area",
    "Population_Exposure_Ratio",
    "Area_Exposure",
    "Flood_Exposure_Score",
]

# Ensure they exist in the dataframe
missing = [c for c in desired_features if c not in df.columns]
if missing:
    raise KeyError(f"The following expected features are missing in {INPUT}: {missing}")

feature_cols = desired_features  # exactly 11 features
print("\nFINAL FEATURE SET for Logistic Regression (11 features):")
print(feature_cols)
print("Number of features:", len(feature_cols))

X = df[feature_cols].copy()
y = df["Flood_Impact_Class"].copy().astype(int)

# ------------- Split data --------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nSplit shapes & class distribution:")
print_split_counts("Train", y_train)
print_split_counts("Val",   y_val)
print_split_counts("Test",  y_test)

# ------------- Impute + Scale ----------
# Drop columns that are entirely NaN (just in case)
cols_to_keep = [c for c in X_train.columns if not X_train[c].isna().all()]
X_train = X_train[cols_to_keep]
X_val   = X_val[cols_to_keep]
X_test  = X_test[cols_to_keep]

print(f"\nAfter dropping all-NaN columns (if any), feature count: {len(cols_to_keep)}")

# Impute median (fit on TRAIN only)
imputer = SimpleImputer(strategy="median")
X_train_imp_arr = imputer.fit_transform(X_train)
X_val_imp_arr   = imputer.transform(X_val)
X_test_imp_arr  = imputer.transform(X_test)

X_train_imp = pd.DataFrame(X_train_imp_arr, columns=X_train.columns, index=X_train.index)
X_val_imp   = pd.DataFrame(X_val_imp_arr,   columns=X_val.columns,   index=X_val.index)
X_test_imp  = pd.DataFrame(X_test_imp_arr,  columns=X_test.columns,  index=X_test.index)

# Standard scaling (fit on TRAIN only)
scaler = StandardScaler()
X_train_scaled_arr = scaler.fit_transform(X_train_imp.values)
X_val_scaled_arr   = scaler.transform(X_val_imp.values)
X_test_scaled_arr  = scaler.transform(X_test_imp.values)

X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train_imp.columns, index=X_train_imp.index)
X_val_scaled   = pd.DataFrame(X_val_scaled_arr,   columns=X_val_imp.columns,   index=X_val_imp.index)
X_test_scaled  = pd.DataFrame(X_test_scaled_arr,  columns=X_test_imp.columns,  index=X_test_imp.index)

# Save preprocessors (so the app can reuse them later)
joblib.dump(imputer, os.path.join(MODEL_DIR, "logreg_imputer.pkl"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "logreg_scaler.pkl"))

# ------------- Train Logistic Regression -------------
print("\nTraining Logistic Regression (multinomial)...")

logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1,
)

logreg.fit(X_train_scaled, y_train)

train_pred = logreg.predict(X_train_scaled)
val_pred   = logreg.predict(X_val_scaled)
test_pred  = logreg.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred)
val_acc   = accuracy_score(y_val,   val_pred)
test_acc  = accuracy_score(y_test,  test_pred)

train_f1 = f1_score(y_train, train_pred, average="macro")
val_f1   = f1_score(y_val,   val_pred,   average="macro")
test_f1  = f1_score(y_test,  test_pred,  average="macro")

print("\nLogistic Regression Performance:")
print(f"Train → Accuracy: {train_acc:.4f}, Macro-F1: {train_f1:.4f}")
print(f"Val   → Accuracy: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
print(f"Test  → Accuracy: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")

print("\nDetailed classification report (Test set):")
print(classification_report(y_test, test_pred, digits=4))

# ------------- Save model + results -----
model_path = os.path.join(MODEL_DIR, "logistic_regression_classifier_v3.pkl")
joblib.dump(logreg, model_path)
print("\nSaved Logistic Regression model ->", model_path)

results = pd.DataFrame([
    {"Model": "LogReg", "Split": "Train", "Accuracy": train_acc, "MacroF1": train_f1},
    {"Model": "LogReg", "Split": "Val",   "Accuracy": val_acc,   "MacroF1": val_f1},
    {"Model": "LogReg", "Split": "Test",  "Accuracy": test_acc,  "MacroF1": test_f1},
])

results_path = os.path.join(RESULTS_DIR, "logistic_regression_results_v3.csv")
results.to_csv(results_path, index=False)
print("Saved Logistic Regression results ->", results_path)
