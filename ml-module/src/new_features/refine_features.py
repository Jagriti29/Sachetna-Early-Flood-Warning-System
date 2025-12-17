"""This will:

Load Refined_FloodDataset.csv

Drop irrelevant or missing-heavy columns

Handle duplicates and type consistency

Save as data/processed/Model_Input.csv
"""
import pandas as pd
import os

# Paths
input_path = "data/processed/Refined_FloodDataset.csv"
output_path = "data/processed/Model_Input.csv"

# Load dataset
df = pd.read_csv(input_path)
print(f"Loaded dataset: {input_path}")
print(f"Initial shape: {df.shape}")

# Drop columns with too many missing or irrelevant info
drop_cols = [
    "Mean_Severity", "AFSI", "Severity_Score",
    "Dist_Name_x", "Dist_Name_y",
    "Parmanent_Water", "Mean_Flood_Duration",
    "Percent_Flooded_Area"  # keep only Corrected version
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

# Drop rows where target is missing
df = df.dropna(subset=["Flood_Risk_Index"])

# Optional: fill missing numeric values with median
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Strip extra spaces from district names
df["Districts"] = df["Districts"].astype(str).str.strip()

# Remove duplicates
df = df.drop_duplicates()

# Save refined dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nRefined dataset saved as: {output_path}")
print(f"Final shape: {df.shape}")
print(f"Columns retained:\n{df.columns.tolist()}")
