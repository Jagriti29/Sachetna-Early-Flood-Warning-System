'''Refining the Flood_Features_Enhanced.csv dataset by splitting multi-district entries into separate rows for each district.
This ensures each row corresponds to a single district-year combination.
this csv file is generated after feature engineering step.'''
import pandas as pd

# Load your feature-engineered dataset
df = pd.read_csv("data/processed/Flood_Features_Enhanced.csv")

print("Initial shape:", df.shape)

# --- STEP 1: Clean up 'Districts' column ---
df["Districts"] = df["Districts"].astype(str).str.replace(" and ", ", ", regex=False)
df["Districts"] = df["Districts"].str.replace("&", ",", regex=False)
df["Districts"] = df["Districts"].str.replace("  ", " ", regex=False)
df["Districts"] = df["Districts"].str.strip()

# --- STEP 2: Split multi-district rows ---
def expand_districts(row):
    districts = [d.strip() for d in row["Districts"].split(",") if d.strip()]
    rows = []
    for dist in districts:
        new_row = row.copy()
        new_row["Districts"] = dist
        rows.append(new_row)
    return pd.DataFrame(rows)

expanded_df = pd.concat([expand_districts(row) for _, row in df.iterrows()], ignore_index=True)

print("After expansion:", expanded_df.shape)

# --- STEP 3: Drop duplicate district-year entries if any ---
expanded_df.drop_duplicates(subset=["Districts", "Year"], inplace=True)

# --- STEP 4: Save the refined dataset ---
expanded_df.to_csv("data/processed/Refined_FloodDataset.csv", index=False)

print("\nRefined dataset saved as: data/processed/Refined_FloodDataset.csv")
print("Final shape:", expanded_df.shape)
print("Unique Districts:", expanded_df['Districts'].nunique())
