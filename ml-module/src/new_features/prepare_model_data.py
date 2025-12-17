"""Load Model_Input.csv

Encode categorical features (like Districts)

Scale numerical features using StandardScaler

Split the data into train/validation/test sets

Save all as ready-to-train CSVs"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Paths
input_path = "data/processed/Model_Input.csv"
output_dir = "data/processed/model_ready/"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_path)
print(f"Loaded dataset: {input_path}")
print(f"Shape: {df.shape}")

# Encode categorical features
le = LabelEncoder()
df["Districts_encoded"] = le.fit_transform(df["Districts"].astype(str))

# Define features and target
features = [
    "Districts_encoded", "Year", "Flood_Frequency", "Mean_Duration",
    "Human_fatality", "Human_injured", "Population",
    "Corrected_Percent_Flooded_Area", "Population_Exposure_Ratio", "Area_Exposure"
]
target = "Flood_Risk_Index"

X = df[features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to DataFrames for saving
train_df = pd.DataFrame(X_train, columns=features)
train_df[target] = y_train.values

val_df = pd.DataFrame(X_val, columns=features)
val_df[target] = y_val.values

test_df = pd.DataFrame(X_test, columns=features)
test_df[target] = y_test.values

# Save all
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("\n Model-ready data created and saved in data/processed/model_ready/")
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
