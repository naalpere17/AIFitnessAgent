"""
train.py
========
Phase 1: Train a global XGBoost model on synthetic population data.
Exports: train_global_model(verbose=True) -> dict with mae and model path
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

from fitness_rec.config import SYNTHETIC_CSV

FEATURES = [
    "Age", "gender_encoded", "bmi", "Resting_BPM", "Weight (kg)",
    "Session_Duration (hours)", "Calories_Burned",
    "acwr", "readiness", "acute_load", "chronic_load", "est_max_bpm",
]


def train_global_model(verbose: bool = True) -> dict:
    """
    Train the global XGBoost model on synthetic population data.

    Returns
    -------
    dict:
        mae       float — validation MAE in TSS points
        model     str   — path to saved model pickle
        features  str   — path to saved feature names pickle
    """
    df = pd.read_csv(SYNTHETIC_CSV)

    # Gender encoding
    if "Gender" in df.columns:
        df["gender_encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
    else:
        if verbose:
            print("Warning: no 'Gender' column found — defaulting to 0 (male)")
        df["gender_encoded"] = 0

    # Sex-specific HRmax (Tanaka et al. 2001)
    df["est_max_bpm"] = np.where(
        df["gender_encoded"] == 1,
        206 - (0.88 * df["Age"]),
        208 - (0.7  * df["Age"])
    )

    # TSS target
    df["tss"] = (
        df["Session_Duration (hours)"]
        * df["Avg_BPM"] ** 2
        / (df["Resting_BPM"] * df["est_max_bpm"])
        * 100
    ).clip(0, 150)

    # Neutral placeholders for personal state features
    df["acwr"]         = 1.0
    df["readiness"]    = 0.5
    df["acute_load"]   = df["Calories_Burned"]
    df["chronic_load"] = df["Calories_Burned"]
    df["bmi"]          = df["BMI"]

    df = df[FEATURES + ["tss"]].dropna()
    X  = df[FEATURES]
    y  = df["tss"]

    if verbose:
        print(f"Dataset: {len(df)} rows")
        print(f"Gender split — male: {(df['gender_encoded']==0).sum()}  "
              f"female: {(df['gender_encoded']==1).sum()}")
        print(f"TSS — min: {y.min():.1f}  max: {y.max():.1f}  mean: {y.mean():.1f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), FEATURES)]
    )

    model = Pipeline([
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

    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_val, model.predict(X_val))

    if verbose:
        print(f"Validation MAE: {mae:.2f} TSS points")

    joblib.dump(model,    "global_intensity_model.pkl")
    joblib.dump(FEATURES, "global_feature_names.pkl")

    if verbose:
        print("Saved → global_intensity_model.pkl")

    return {
        "mae":      mae,
        "model":    "global_intensity_model.pkl",
        "features": "global_feature_names.pkl",
    }


if __name__ == "__main__":
    train_global_model()
