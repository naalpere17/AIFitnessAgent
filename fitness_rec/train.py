"""
train.py
========
Phase 1: Train a global XGBoost model on synthetic population data.
Target: TSS (Training Stress Score) — physiologically meaningful intensity proxy.
Run this first. Outputs: global_intensity_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------
# LOAD SYNTHETIC DATASET
# ---------------------------------------------------------------
df = pd.read_csv("synthetic_fitness_dataset.csv")

# ---------------------------------------------------------------
# TARGET: TSS approximation
# TSS = (duration_h × avg_bpm²) / (resting_bpm × max_bpm) × 100
# This is more physiologically meaningful than raw Avg_BPM because
# it captures both intensity AND duration in a single load metric.
# ---------------------------------------------------------------
df["est_max_bpm"] = 220 - df["Age"]
df["tss"] = (
    df["Session_Duration (hours)"]
    * df["Avg_BPM"] ** 2
    / (df["Resting_BPM"] * df["est_max_bpm"])
    * 100
).clip(0, 150)

# ---------------------------------------------------------------
# FEATURES
# Personal state placeholders (acwr, readiness etc.) are set to
# neutral values here — the global model learns population patterns.
# The personal adapter in recommend.py then adjusts for your state.
# ---------------------------------------------------------------
df["acwr"]         = 1.0
df["readiness"]    = 0.5
df["acute_load"]   = df["Calories_Burned"]
df["chronic_load"] = df["Calories_Burned"]
df["bmi"]          = df["BMI"]

FEATURES = [
    "Age", "bmi", "Resting_BPM", "Weight (kg)",
    "Session_Duration (hours)", "Calories_Burned",
    "acwr", "readiness", "acute_load", "chronic_load", "est_max_bpm",
]

df = df[FEATURES + ["tss"]].dropna()
X  = df[FEATURES]
y  = df["tss"]

print(f"Dataset: {len(df)} rows")
print(f"TSS — min: {y.min():.1f}  max: {y.max():.1f}  mean: {y.mean():.1f}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), FEATURES)]
)

global_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ))
])

global_model.fit(X_train, y_train)

val_preds = global_model.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, val_preds):.2f} TSS points")

joblib.dump(global_model, "global_intensity_model.pkl")
joblib.dump(FEATURES, "global_feature_names.pkl")
print("Saved → global_intensity_model.pkl")