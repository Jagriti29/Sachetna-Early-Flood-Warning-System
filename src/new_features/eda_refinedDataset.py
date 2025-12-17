
# Exploratory & Correlation Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Load dataset ---
file_path = "data/processed/Refined_FloodDataset.csv"
df = pd.read_csv(file_path)
print(f"Loaded dataset: {file_path}")
print(f"Shape: {df.shape}\n")

# --- 2. Basic Info ---
print("Column Info:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())

# --- 3. Handle obvious non-numeric columns ---
numeric_df = df.select_dtypes(include=[np.number])

# --- 4. Correlation Matrix ---
corr = numeric_df.corr()
print("\nTop correlations with Flood_Risk_Index (if present):")
if 'Flood_Risk_Index' in corr.columns:
    print(corr['Flood_Risk_Index'].sort_values(ascending=False))
else:
    print("⚠️ Flood_Risk_Index not found — skipping specific correlation list.")

# --- 5. Save correlation matrix ---
os.makedirs("data/analysis", exist_ok=True)
corr.to_csv("data/analysis/correlation_matrix.csv", index=True)
print("\n correlation_matrix.csv saved in data/analysis/")

# --- 6. Heatmap Visualization ---
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Flood Features")
plt.tight_layout()
plt.savefig("data/analysis/correlation_heatmap.png")
plt.close()
print(" correlation_heatmap.png saved in data/analysis/")

# --- 7. Feature Importance Ranking (simple heuristic) ---
if 'Flood_Risk_Index' in corr.columns:
    top_features = corr['Flood_Risk_Index'].abs().sort_values(ascending=False)[1:11]
    print("\nTop 10 correlated features with Flood_Risk_Index:")
    print(top_features)
    top_features.to_csv("data/analysis/top_correlated_features.csv")
    print(" top_correlated_features.csv saved.")
else:
    print("\nFlood_Risk_Index not found for correlation ranking.")

# --- 8. Summary Statistics ---
summary = numeric_df.describe().transpose()
summary.to_csv("data/analysis/feature_summary.csv")
print("feature_summary.csv saved in data/analysis/")

print("\nExploratory analysis complete!")
