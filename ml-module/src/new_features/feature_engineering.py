"""
Feature Engineering Script
--------------------------
Input:
    data/processed/Unified_FloodDataset.csv

Output:
    data/processed/Flood_Features_Enhanced.csv

Tasks:
1. Handle missing & inconsistent values
2. Map severity levels to numeric scale (if textual)
3. Remove invalid district names (e.g., "14 Districts", "& Parts Of ...")
4. Create derived flood risk indicators
5. Save enhanced dataset
"""

import pandas as pd
import numpy as np
import re
import os

# === 1. Load dataset ===
input_path = "data/processed/Unified_FloodDataset.csv"
output_path = "data/processed/Flood_Features_Enhanced.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"File not found: {input_path}")

df = pd.read_csv(input_path)
print(f"Loaded dataset: {input_path}")
print("Initial shape:", df.shape)

# === 2. Clean district names ===
def clean_district_name(name):
    if pd.isna(name):
        return np.nan
    name = str(name).strip()
    # Remove entries like "14 Districts", "Parts Of", "&"
    if re.search(r"\d+\s*Districts", name, re.IGNORECASE):
        return np.nan
    if "Parts Of" in name or "&" in name:
        return np.nan
    return name.title().strip()

df['Districts'] = df['Districts'].apply(clean_district_name)
df = df.dropna(subset=['Districts'])

# === 3. Handle missing numeric values ===
numeric_cols = [
    'Flood_Frequency', 'Mean_Duration', 'Mean_Severity', 
    'Human_fatality', 'Human_injured', 'Population',
    'Percent_Flooded_Area', 'Corrected_Percent_Flooded_Area',
    'Population_Exposure_Ratio'
]

for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median(skipna=True)
        df[col].fillna(median_val, inplace=True)

# === 4. Convert Mean_Severity if textual ===
if df['Mean_Severity'].dtype == object:
    severity_map = {
        'Mild': 1, 'Moderate': 2, 'Severe': 3, 
        'Very Severe': 4, 'Extreme': 5
    }
    df['Mean_Severity'] = df['Mean_Severity'].map(severity_map).fillna(0)

# === 5. Derived features ===
df['Flood_Risk_Index'] = (
    df['Flood_Frequency'] * df['Mean_Duration'] * (df['Population_Exposure_Ratio'] + 1)
)

df['Area_Exposure'] = (
    (df['Corrected_Percent_Flooded_Area'] / 100) * df['Population']
)

df['Severity_Score'] = (
    df['Mean_Severity'] * df['Flood_Frequency']
)

# === 6. Basic consistency filtering ===
df = df[df['Flood_Frequency'] > 0]
df = df[df['Mean_Duration'] >= 0]

# === 7. Final cleaning ===
df = df.drop_duplicates(subset=['Districts', 'Year']).reset_index(drop=True)

# === 8. Save enhanced dataset ===
df.to_csv(output_path, index=False)
print(f"\n Enhanced feature dataset saved as: {output_path}")
print("Final shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nPreview:")
print(df.head())
