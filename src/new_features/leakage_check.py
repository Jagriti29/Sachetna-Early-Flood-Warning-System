"""
Leakage & Dependency Diagnostic
--------------------------------
Performs a sequence of checks to detect data leakage or direct formula
dependencies between features and the target (Flood_Risk_Index).

Outputs (saved to data/analysis/):
 - correlation_with_target.csv
 - high_correlations.csv
 - vif.csv
 - mutual_info.csv
 - shuffle_test_results.csv
 - split_overlap.csv
 - plots: corr_target_bar.png, actual_vs_predicted_shuffled.png
 - textual report: leakage_report.txt

Checks included:
 1) Numeric correlation with target (sort by abs correlation)
 2) High-correlation flags (|r| > 0.9)
 3) VIF (multicollinearity)
 4) Mutual information with target
 5) Train/Val/Test split overlap by Districts and by District-Year
 6) Shuffle-target sanity test (retrain RF on same features with shuffled y)
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import shuffle
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# --- Paths ---
MODEL_PATH = "models/random_forest_baseline.pkl"
MODEL_READY_DIR = "data/processed/model_ready"
REFINED_PATH = "data/processed/Model_Input.csv"  # original features before scaling
ANALYSIS_DIR = "data/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Load model and data ---
print("Loading model and data...")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("Warning: trained model not found at", MODEL_PATH)

train = pd.read_csv(os.path.join(MODEL_READY_DIR, "train.csv"))
val = pd.read_csv(os.path.join(MODEL_READY_DIR, "val.csv"))
test = pd.read_csv(os.path.join(MODEL_READY_DIR, "test.csv"))

# Load the unscaled Model_Input (for careful correlation checks)
raw = pd.read_csv(REFINED_PATH)

# If Districts_encoded column exists in scaled files, map back to names if possible
print("Data shapes:", train.shape, val.shape, test.shape, raw.shape)

# --- 1) Correlation with target (use raw numeric columns where possible) ---
# Align raw features and target: raw has target 'Flood_Risk_Index' and numeric columns
numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
if 'Flood_Risk_Index' not in numeric_cols:
    numeric_cols.append('Flood_Risk_Index')

corr = raw[numeric_cols].corr()['Flood_Risk_Index'].sort_values(key=lambda s: s.abs(), ascending=False)
corr_df = corr.reset_index()
corr_df.columns = ['feature', 'corr_with_target']
corr_df.to_csv(os.path.join(ANALYSIS_DIR, "correlation_with_target.csv"), index=False)

# Save top high correlations
high_corr = corr_df[ corr_df['corr_with_target'].abs() > 0.9 ]
high_corr.to_csv(os.path.join(ANALYSIS_DIR, "high_correlations.csv"), index=False)

# Quick bar plot of top 10 absolute correlations
top10 = corr_df.copy()
top10['abs_corr'] = top10['corr_with_target'].abs()
top10 = top10.sort_values('abs_corr', ascending=False).head(10)
plt.figure(figsize=(8,6))
sns.barplot(x='abs_corr', y='feature', data=top10, palette='mako')
plt.title("Top 10 |correlation| with Flood_Risk_Index (raw data)")
plt.xlabel("Absolute correlation")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "corr_target_bar.png"))
plt.close()

# --- 2) VIF (Variance Inflation Factor) for multicollinearity ---
# Use numeric columns minus the target
vif_df = pd.DataFrame()
X_vif = raw.select_dtypes(include=[np.number]).drop(columns=['Flood_Risk_Index'], errors=False).fillna(0)
X_vif_const = sm.add_constant(X_vif)
vif_list = []
for i in range(X_vif.shape[1]):
    col = X_vif.columns[i]
    try:
        vif_val = variance_inflation_factor(X_vif_const.values, i+1)  # +1 for const
    except Exception as e:
        vif_val = np.nan
    vif_list.append((col, vif_val))
vif_df = pd.DataFrame(vif_list, columns=['feature','VIF']).sort_values('VIF', ascending=False)
vif_df.to_csv(os.path.join(ANALYSIS_DIR, "vif.csv"), index=False)

# --- 3) Mutual Information (non-linear dependency) ---
X_mi = raw.select_dtypes(include=[np.number]).drop(columns=['Flood_Risk_Index'], errors=False).fillna(0)
y_mi = raw['Flood_Risk_Index'].fillna(0)
mi = mutual_info_regression(X_mi, y_mi, random_state=42)
mi_df = pd.DataFrame({'feature': X_mi.columns, 'mutual_info': mi}).sort_values('mutual_info', ascending=False)
mi_df.to_csv(os.path.join(ANALYSIS_DIR, "mutual_info.csv"), index=False)

# --- 4) Split overlap checks (Districts and District-Year) ---
# Load original raw DataFrame's Districts and Year to test overlaps
# The model-ready splits might have encoded features — check using Model_Input original file if present
def load_model_input():
    p = "data/processed/Model_Input.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

model_input = load_model_input()
if model_input is not None and 'Districts' in model_input.columns:
    # Build sets
    train_df = pd.read_csv(os.path.join(MODEL_READY_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(MODEL_READY_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(MODEL_READY_DIR, "test.csv"))

    # If encoded districts exist, fallback to raw district names using mapping if available
    # But we will compute overlap on District-Year if present in model_input
    # We'll attempt to use the refined dataset (Refined_FloodDataset.csv) if available
    refined_path = "data/processed/Refined_FloodDataset.csv"
    if os.path.exists(refined_path):
        refined = pd.read_csv(refined_path)
        # We'll compute the intersection of (Districts, Year) across splits by matching counts
        # Build sets using the final model_input style: District + Year
        pairs = set((row['Districts'], int(row['Year'])) for _, row in refined.iterrows() if pd.notna(row['Districts']))
        # Instead, compute overlap from model_ready if possible:
        def get_pairs_from_scaled(df_scaled):
            # if Districts exists in file, use it; else try Districts_encoded fallback
            if 'Districts' in df_scaled.columns:
                return set((row['Districts'], int(row['Year'])) for _, row in df_scaled.iterrows())
            else:
                # fallback: return empty
                return set()
        pairs_train = get_pairs_from_scaled(train_df)
        pairs_val = get_pairs_from_scaled(val_df)
        pairs_test = get_pairs_from_scaled(test_df)

        overlap_train_val = len(pairs_train.intersection(pairs_val))
        overlap_train_test = len(pairs_train.intersection(pairs_test))
        overlap_val_test = len(pairs_val.intersection(pairs_test))

        overlap_df = pd.DataFrame({
            'pair_type': ['train_vs_val', 'train_vs_test', 'val_vs_test'],
            'overlap_count': [overlap_train_val, overlap_train_test, overlap_val_test]
        })
        overlap_df.to_csv(os.path.join(ANALYSIS_DIR, "split_overlap.csv"), index=False)
    else:
        # Save an empty overlap file if refined not present
        pd.DataFrame([]).to_csv(os.path.join(ANALYSIS_DIR, "split_overlap.csv"), index=False)
else:
    pd.DataFrame([]).to_csv(os.path.join(ANALYSIS_DIR, "split_overlap.csv"), index=False)

# --- 5) Shuffle-target sanity check ---
# We'll train a small RF on the same features with shuffled target (on raw data subset)
# Use a random subset for speed
print("Running shuffle-target sanity check (this may take a moment)...")
work = raw.select_dtypes(include=[np.number]).drop(columns=['Flood_Risk_Index'], errors=False).fillna(0)
y_work = raw['Flood_Risk_Index'].fillna(0)
# limit rows for speed (but keep deterministic)
n = min(2000, work.shape[0])
work_sub = work.sample(n=n, random_state=42)
y_sub = y_work.loc[work_sub.index]

# Split 70/30
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(work_sub, y_sub, test_size=0.3, random_state=42)

# Train on true target
rf_true = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_true.fit(X_tr, y_tr)
y_pred_true = rf_true.predict(X_te)
r2_true = r2_score(y_te, y_pred_true)

# Train on shuffled target
y_tr_shuf = shuffle(y_tr, random_state=42).reset_index(drop=True)
# Need y_tr_shuf length align: reindex
rf_shuf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_shuf.fit(X_tr, y_tr_shuf)
y_pred_shuf = rf_shuf.predict(X_te)
r2_shuf = r2_score(y_te, y_pred_shuf)

shuffle_df = pd.DataFrame({
    'test_r2_true': [r2_true],
    'test_r2_shuffled_target': [r2_shuf]
})
shuffle_df.to_csv(os.path.join(ANALYSIS_DIR, "shuffle_test_results.csv"), index=False)

# Save actual vs predicted for shuffled model for inspection
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_te, y=y_pred_shuf, alpha=0.6)
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--')
plt.xlabel("Actual Flood Risk Index")
plt.ylabel("Predicted (shuffled-target) Flood Risk Index")
plt.title("Actual vs Predicted (Shuffled Target) — Test subset")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "actual_vs_predicted_shuffled.png"))
plt.close()

# --- 6) Write a concise textual report ---
report_lines = []
report_lines.append("Leakage & Dependency Diagnostic Report\n")
report_lines.append("=== Summary ===\n")
report_lines.append(f"Raw data shape: {raw.shape}\n")
report_lines.append(f"Top correlations (abs) saved to: {ANALYSIS_DIR}/correlation_with_target.csv\n")
report_lines.append(f"High correlations (|r|>0.9) saved to: {ANALYSIS_DIR}/high_correlations.csv\n")
report_lines.append(f"VIF table saved to: {ANALYSIS_DIR}/vif.csv\n")
report_lines.append(f"Mutual information saved to: {ANALYSIS_DIR}/mutual_info.csv\n")
report_lines.append(f"Split overlap saved to: {ANALYSIS_DIR}/split_overlap.csv\n")
report_lines.append(f"Shuffle-target test saved to: {ANALYSIS_DIR}/shuffle_test_results.csv\n")
report_lines.append("\n=== Quick Findings (automated) ===\n")

# Automated checks and notes
if not high_corr.empty:
    report_lines.append("WARNING: Features with |r| > 0.9 detected (possible direct dependency):\n")
    for _, rrow in high_corr.iterrows():
        report_lines.append(f" - {rrow['feature']}: corr = {rrow['corr_with_target']:.6f}\n")
else:
    report_lines.append("No features with |r| > 0.9 found.\n")

# VIF flag
high_vif = vif_df[vif_df['VIF'] > 10]
if not high_vif.empty:
    report_lines.append("\nHigh multicollinearity detected (VIF > 10):\n")
    for _, v in high_vif.iterrows():
        report_lines.append(f" - {v['feature']}: VIF = {v['VIF']:.2f}\n")
else:
    report_lines.append("\nNo extreme VIF (>10) detected.\n")

# Shuffle test analysis
rtrue = float(r2_true)
rshuf = float(r2_shuf)
report_lines.append(f"\nShuffle-test R² (true target): {rtrue:.6f}\n")
report_lines.append(f"Shuffle-test R² (shuffled target): {rshuf:.6f}\n")
if rshuf > 0.3:
    report_lines.append("WARNING: Model still gets moderate R² on shuffled target — possible leakage or very strong feature correlations.\n")
else:
    report_lines.append("Shuffle-test passed: R² drops when target is shuffled (expected).\n")

# Save report
with open(os.path.join(ANALYSIS_DIR, "leakage_report.txt"), "w") as f:
    f.writelines(report_lines)

print("\nDiagnostic complete. Results written to", ANALYSIS_DIR)
