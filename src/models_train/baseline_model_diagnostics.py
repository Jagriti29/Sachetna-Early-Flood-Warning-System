"""This script will:

Load the trained Random Forest (models/random_forest_baseline.pkl)

Load the test data

Compute and plot:

Feature importance ranking

Actual vs Predicted Flood_Risk_Index (scatter plot + line fit)

Distribution comparison (histograms)

Check for overfitting / data leakage by comparing metrics across splits"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

# --- Paths ---
model_path = "models/random_forest_baseline.pkl"
test_path = "data/processed/model_ready/test.csv"
train_path = "data/processed/model_ready/train.csv"
val_path = "data/processed/model_ready/val.csv"

# --- Load data ---
train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)

# --- Separate features & target ---
target_col = "Flood_Risk_Index"
X_train, y_train = train.drop(columns=[target_col]), train[target_col]
X_val, y_val = val.drop(columns=[target_col]), val[target_col]
X_test, y_test = test.drop(columns=[target_col]), test[target_col]

# --- Load trained model ---
model = joblib.load(model_path)

# --- Predictions ---
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# --- Compute metrics ---
def compute_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n {label} Performance:")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return mae, rmse, r2

compute_metrics(y_train, y_train_pred, "Train")
compute_metrics(y_val, y_val_pred, "Validation")
compute_metrics(y_test, y_test_pred, "Test")

# --- Feature Importances ---
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
plt.title("Feature Importance (Random Forest Baseline)")
plt.tight_layout()
plt.savefig("data/results/feature_importance.png")
plt.close()

# --- Actual vs Predicted Plot ---
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Flood Risk Index")
plt.ylabel("Predicted Flood Risk Index")
plt.title("Actual vs Predicted (Test Set)")
plt.tight_layout()
plt.savefig("data/results/actual_vs_predicted.png")
plt.close()

# --- Distribution Comparison ---
plt.figure(figsize=(8,5))
sns.kdeplot(y_test, label="Actual", fill=True)
sns.kdeplot(y_test_pred, label="Predicted", fill=True)
plt.title("Distribution of Actual vs Predicted Flood Risk Index")
plt.legend()
plt.tight_layout()
plt.savefig("data/results/distribution_comparison.png")
plt.close()

print("\n Interpretation complete! Plots saved in data/results/")
