"""
Data Preprocessing Script
-------------------------
Merges:
1. IFI_cleaned.csv (cleaned India Flood Inventory)
2. District_FloodImpact.csv (impact metrics)
3. District_FloodedArea.csv (flooded area metrics)

Generates a unified district-year level flood dataset with derived metrics:
- Flood Frequency
- Mean Duration
- Mean Severity
- Average Flood Severity Index (AFSI)
- Population Exposure Ratio
"""

import pandas as pd
import numpy as np

# === 1. Load datasets ===
impact = pd.read_csv("data/raw/IFI Dataset/District_FloodImpact.csv", encoding='utf-8')
flooded = pd.read_csv("data/raw/IFI Dataset/District_FloodedArea.csv", encoding='utf-8')
ifi = pd.read_csv("data/raw/IFI Dataset/India_Flood_Inventory_v3.csv", encoding='utf-8')


# === 2. Parse and extract temporal features ===
ifi['Start Date'] = pd.to_datetime(ifi['Start Date'], errors='coerce')
ifi['End Date'] = pd.to_datetime(ifi['End Date'], errors='coerce')

# Compute duration if missing
ifi['Duration(Days)'] = ifi['Duration(Days)'].fillna(
    (ifi['End Date'] - ifi['Start Date']).dt.days
)

ifi['Year'] = ifi['Start Date'].dt.year
ifi['Month'] = ifi['Start Date'].dt.month

# === 3. Clean and standardize district names ===
for df in [ifi, impact, flooded]:
    df.columns = df.columns.str.strip()

ifi['Districts'] = ifi['Districts'].astype(str).str.strip().str.title()
impact['Dist_Name'] = impact['Dist_Name'].astype(str).str.strip().str.title()
flooded['Dist_Name'] = flooded['Dist_Name'].astype(str).str.strip().str.title()

# Handle minor known inconsistencies (optional)
replacements = {
    'Andaman & Nicobar Islands': 'Andaman And Nicobar Islands',
    'Kachchh': 'Kutch',
}
ifi['Districts'] = ifi['Districts'].replace(replacements)
impact['Dist_Name'] = impact['Dist_Name'].replace(replacements)
flooded['Dist_Name'] = flooded['Dist_Name'].replace(replacements)

# === 4. Compute derived metrics from IFI ===
# Flood frequency (no. of flood events per district per year)
flood_freq = (
    ifi.groupby(['Districts', 'Year'])
    .size()
    .reset_index(name='Flood_Frequency')
)

# Mean flood duration
mean_duration = (
    ifi.groupby(['Districts', 'Year'])['Duration(Days)']
    .mean()
    .reset_index(name='Mean_Duration')
)

# Mean flood severity
mean_severity = (
    ifi.groupby(['Districts', 'Year'])['Severity']
    .mean(numeric_only=True)
    .reset_index(name='Mean_Severity')
)

# === 5. Merge derived metrics ===
district_summary = (
    flood_freq.merge(mean_duration, on=['Districts', 'Year'], how='left')
              .merge(mean_severity, on=['Districts', 'Year'], how='left')
)

# === 6. Derived Metric: Average Flood Severity Index (AFSI) ===
# Combines severity and duration to represent overall intensity
district_summary['AFSI'] = (
    district_summary['Mean_Severity'] * district_summary['Mean_Duration']
)

# === 7. Merge with impact and flooded area data ===
merged = (
    district_summary.merge(impact, left_on='Districts', right_on='Dist_Name', how='left')
                    .merge(flooded, left_on='Districts', right_on='Dist_Name', how='left')
)

# === 8. Derived Metric: Population Exposure Ratio ===
if 'Population' in merged.columns and 'Human_fatality' in merged.columns:
    merged['Population_Exposure_Ratio'] = merged['Human_fatality'] / merged['Population']
    merged['Population_Exposure_Ratio'] = merged['Population_Exposure_Ratio'].clip(0, 1)
else:
    merged['Population_Exposure_Ratio'] = np.nan

# === 9. Final Cleaning ===
merged.drop(columns=['Dist_Name'], inplace=True, errors='ignore')
merged = merged.dropna(subset=['Districts', 'Year']).drop_duplicates(subset=['Districts', 'Year']).reset_index(drop=True)

# === 10. Save unified dataset ===
output_path = "data/processed/Unified_FloodDataset.csv"
merged.to_csv(output_path, index=False)

# === 11. Summary Output ===
print("Unified dataset successfully created!")
print(f"Saved as: {output_path}")
print("Shape:", merged.shape)
print(f"Total Districts: {merged['Districts'].nunique()}, Years: {merged['Year'].nunique()}")
print("\nColumns:")
print(merged.columns.tolist())
print("\nPreview:")
print(merged.head())
