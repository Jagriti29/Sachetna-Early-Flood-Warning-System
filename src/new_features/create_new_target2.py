"""
This script:
1. Loads the processed dataset (Model_Input_v2.csv)
2. Computes correlation-based weights for:
   - Mean_Duration (α)
   - Flood_Frequency (β)
   - Population_Exposure_Ratio (γ)
3. Uses these to create a composite target:
   Flood_Impact_Index = Flood_Exposure_Score × (1 + α·Mean_Duration) × (1 + β·Flood_Frequency) × (1 + γ·Population_Exposure_Ratio)
4. Adds a normalized version (0–1) for use in ML/DL models or apps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ===============================
# STEP 1 — Load dataset
# ===============================
# Build paths relative to the repository (one level up from src)
base = Path(__file__).resolve().parent
project_root = base.parent
input_path = project_root / "data" / "processed" / "Model_Input_v2.csv"
if not input_path.exists():
    raise FileNotFoundError(f"File not found: {input_path}")

df = pd.read_csv(input_path)

print(f"Loaded dataset: {input_path}")
print(f"Initial shape: {df.shape}\n")

# ===============================
# STEP 2 — Identify valid columns
# ===============================
core_features = ["Mean_Duration", "Flood_Frequency", "Population_Exposure_Ratio"]
target = "Flood_Exposure_Score"

# Ensure all required columns exist
missing = [col for col in core_features + [target] if col not in df.columns]
if missing:
    raise KeyError(f"Missing columns in dataset: {missing}")

# Drop rows with missing key values
df = df.dropna(subset=core_features + [target])

# ===============================
# STEP 3 — Compute correlations for weight optimization
# ===============================
correlations = df[core_features + [target]].corr()[target].drop(target)
weights = correlations.abs()
weights = weights / weights.sum()  # Normalize so α+β+γ=1

alpha = weights["Mean_Duration"]
beta = weights["Flood_Frequency"]
gamma = weights["Population_Exposure_Ratio"]

print(" Auto-optimized weights (normalized correlations):")
print(f"α (Mean_Duration): {alpha:.4f}")
print(f"β (Flood_Frequency): {beta:.4f}")
print(f"γ (Population_Exposure_Ratio): {gamma:.4f}\n")

# ===============================
# STEP 4 — Compute new Flood Impact Index
# ===============================
df["Flood_Impact_Index"] = (
    df[target] *
    (1 + alpha * df["Mean_Duration"]) *
    (1 + beta * df["Flood_Frequency"]) *
    (1 + gamma * df["Population_Exposure_Ratio"])
)

# Normalized version (0–1)
df["Flood_Impact_Index_Norm"] = (
    (df["Flood_Impact_Index"] - df["Flood_Impact_Index"].min()) /
    (df["Flood_Impact_Index"].max() - df["Flood_Impact_Index"].min())
)

# ===============================
# STEP 5 — Save enhanced dataset
# ===============================
output_path = project_root / "data" / "processed" / "Model_Input_v3.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f" Enhanced dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ===============================
# STEP 6 — Summary Statistics
# ===============================
print("Flood_Impact_Index summary:")
print(df[["Flood_Impact_Index", "Flood_Impact_Index_Norm"]].describe())

# ===============================
# STEP 7 — Visualizations
# ===============================
os.makedirs("data/analysis", exist_ok=True)

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[core_features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Core Predictors vs Flood_Exposure_Score")
plt.tight_layout()
plt.savefig("data/analysis/flood_impact_corr_heatmap.png")
plt.close()


print(" Visualizations saved:")
print(" - data/analysis/flood_impact_corr_heatmap.png")
print(" - data/analysis/flood_impact_distribution.png")
