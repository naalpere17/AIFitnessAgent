"""
train.py
========
Phase 1: Train a global XGBoost model on synthetic population data.
Target: TSS (Training Stress Score) — physiologically meaningful intensity proxy.
Gender is included as a binary feature (0=male, 1=female) and affects
HRmax estimation via the Tanaka et al. (2001) sex-specific formula.
Run this first. Outputs: global_intensity_model.pkl, global_feature_names.pkl
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
# GENDER ENCODING
# Encode as binary: 0 = male, 1 = female
# Synthetic dataset uses "Male"/"Female" strings in Gender column.
# If your dataset uses different values, adjust the map below.
# ---------------------------------------------------------------
if "Gender" in df.columns:
    df["gender_encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
else:
    # If no gender column, default to 0 (model still trains, just
    # without gender signal — recommend adding it to synthetic data)
    print("Warning: no 'Gender' column found — defaulting to 0 (male)")
    df["gender_encoded"] = 0

# ---------------------------------------------------------------
# GENDER-ADJUSTED HRMAX (Tanaka et al. 2001)
# Male:   208 - (0.7 × age)
# Female: 206 - (0.88 × age)
# This is more accurate than universal 220-age and captures the
# systematic difference in cardiovascular capacity between sexes.
# ---------------------------------------------------------------
df["est_max_bpm"] = np.where(
    df["gender_encoded"] == 1,
    206 - (0.88 * df["Age"]),   # female
    208 - (0.7  * df["Age"])    # male
)

# ---------------------------------------------------------------
# TARGET: TSS using sex-specific HRmax
# ---------------------------------------------------------------
df["tss"] = (
    df["Session_Duration (hours)"]
    * df["Avg_BPM"] ** 2
    / (df["Resting_BPM"] * df["est_max_bpm"])
    * 100
).clip(0, 150)

# ---------------------------------------------------------------
# FEATURES — neutral placeholders for personal state features
# ---------------------------------------------------------------
df["acwr"]         = 1.0
df["readiness"]    = 0.5
df["acute_load"]   = df["Calories_Burned"]
df["chronic_load"] = df["Calories_Burned"]
df["bmi"]          = df["BMI"]

FEATURES = [
    "Age", "gender_encoded", "bmi", "Resting_BPM", "Weight (kg)",
    "Session_Duration (hours)", "Calories_Burned",
    "acwr", "readiness", "acute_load", "chronic_load", "est_max_bpm",
]

df = df[FEATURES + ["tss"]].dropna()
X  = df[FEATURES]
y  = df["tss"]

print(f"Dataset: {len(df)} rows")
print(f"Gender split — male: {(df['gender_encoded']==0).sum()}  "
      f"female: {(df['gender_encoded']==1).sum()}")
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
joblib.dump(FEATURES,     "global_feature_names.pkl")
print("Saved → global_intensity_model.pkl")