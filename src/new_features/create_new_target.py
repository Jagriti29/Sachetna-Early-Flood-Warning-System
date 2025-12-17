"""
This script will:

Load your Refined_FloodDataset.csv

Compute Flood_Exposure_Score

Clean + normalize it

Save it as data/processed/Model_Input_v2.csv
Create New Target: Flood_Exposure_Score
---------------------------------------
Combines flood extent (Corrected_Percent_Flooded_Area)
and vulnerability (Population_Exposure_Ratio)
to form a physically meaningful target for ML modeling.
"""

import pandas as pd
import numpy as np
import os

# === Paths ===
input_path = "data/processed/Refined_FloodDataset.csv"
output_path = "data/processed/Model_Input_v2.csv"

# === Load dataset ===
df = pd.read_csv(input_path)
print(f"Loaded dataset: {input_path}")
print(f"Shape: {df.shape}\n")

# === Check columns ===
required_cols = ['Corrected_Percent_Flooded_Area', 'Population_Exposure_Ratio']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise KeyError(f"Missing columns required for new target: {missing}")

# === Compute Flood_Exposure_Score ===
df['Flood_Exposure_Score'] = (
    df['Corrected_Percent_Flooded_Area'].fillna(0) *
    df['Population_Exposure_Ratio'].fillna(0)
)

# === Drop redundant / leakage-prone target ===
if 'Flood_Risk_Index' in df.columns:
    df.drop(columns=['Flood_Risk_Index'], inplace=True)

# === Handle missing / invalid rows ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Flood_Exposure_Score'], inplace=True)

# === Optional normalization (0–1 scaling) ===
if df['Flood_Exposure_Score'].max() > 1:
    df['Flood_Exposure_Score'] = (
        df['Flood_Exposure_Score'] - df['Flood_Exposure_Score'].min()
    ) / (df['Flood_Exposure_Score'].max() - df['Flood_Exposure_Score'].min())

# === Save cleaned dataset ===
os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"New dataset saved as: {output_path}")
print(f"Final shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nPreview:")
print(df.head())
