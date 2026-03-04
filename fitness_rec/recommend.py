"""
Fitness Intensity Recommendation Engine
========================================
Scientifically grounded workout intensity predictor combining:
  - A global XGBoost model trained on population-level fitness data
  - Personal biometric adjustment rules based on HRV, sleep, and ACWR

Scientific basis:
  - ACWR thresholds: Hulin et al. (2016), Gabbett (2016)
  - HRV readiness: Plews et al. (2013)
  - Sleep & recovery: Simpson et al. (2017)
  - Readiness normalization: min-max scaling per component before weighting
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


# ===============================================================
# USER PROFILE — edit these values
# ===============================================================
USER_AGE        = 21       # years
USER_HEIGHT_M   = 1.72     # metres (used to compute BMI properly)
USER_WEIGHT_KG  = 80       # fallback if Apple Health body_mass is missing


# ===============================================================
# GLOBAL MODEL  (population-level dataset)
# ===============================================================

big_df = pd.read_csv("synthetic_fitness_dataset.csv")

# --- Intensity proxy: Training Stress Score (TSS) approximation ---
# TSS ≈ (duration_h * avg_bpm²) / (resting_bpm * max_bpm) * 100
big_df["est_max_bpm"] = 220 - big_df["Age"]
big_df["intensity"] = (
    big_df["Session_Duration (hours)"]
    * big_df["Avg_BPM"] ** 2
    / (big_df["Resting_BPM"] * big_df["est_max_bpm"])
    * 100
).clip(0, 150)

global_features = [
    "Age", "Weight (kg)", "Resting_BPM",
    "Session_Duration (hours)", "Calories_Burned", "BMI", "est_max_bpm",
]

X_big = big_df[global_features]
y_big = big_df["intensity"]

preprocessor_big = ColumnTransformer(
    transformers=[("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), global_features)]
)

global_model = Pipeline([
    ("preprocessing", preprocessor_big),
    ("model", XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ))
])

global_model.fit(X_big, y_big)
joblib.dump(global_model, "global_intensity_model.pkl")
print("Global model trained and saved.")


# ===============================================================
# PERSONAL STATE  (Apple Health data)
# ===============================================================

apple_df = pd.read_csv("output/health_last_60_days.csv")
apple_df["date"] = pd.to_datetime(apple_df["date"])
apple_df = apple_df.sort_values("date").reset_index(drop=True)

apple_df["body_mass_avg"] = apple_df["body_mass_avg"].fillna(USER_WEIGHT_KG)
apple_df["bmi"] = apple_df["body_mass_avg"] / (USER_HEIGHT_M ** 2)

# Sleep: treat <2h as recording error
apple_df.loc[apple_df["sleep_hours"] < 2, "sleep_hours"] = np.nan
apple_df["sleep_hours"] = apple_df["sleep_hours"].fillna(apple_df["sleep_hours"].median())

# Rolling averages
for col in ["sleep_hours", "hrv_avg", "resting_hr_avg", "active_energy"]:
    apple_df[f"{col}_7d"] = apple_df[col].rolling(7, min_periods=4).mean()

# ACWR
apple_df["acute_load"]   = apple_df["active_energy"].rolling(7,  min_periods=4).mean()
apple_df["chronic_load"] = apple_df["active_energy"].rolling(28, min_periods=14).mean()
apple_df["acwr"] = np.where(
    apple_df["chronic_load"] > 0,
    apple_df["acute_load"] / apple_df["chronic_load"],
    np.nan,
)

# ---------------------------------------------------------------
# READINESS SCORE — normalise each component to [0,1] first
# ---------------------------------------------------------------
def minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)

hrv_norm    = minmax_normalize(apple_df["hrv_avg_7d"])
sleep_norm  = minmax_normalize(apple_df["sleep_hours_7d"])
hr_norm_inv = 1 - minmax_normalize(apple_df["resting_hr_avg_7d"])

# Weights: HRV=0.45, Sleep=0.35, Resting HR=0.20
apple_df["readiness"] = (
    0.45 * hrv_norm
    + 0.35 * sleep_norm
    + 0.20 * hr_norm_inv
)

apple_df = apple_df.dropna(subset=["acwr", "readiness"])

if len(apple_df) < 5:
    raise ValueError(
        "Not enough valid rows after cleaning. "
        "Check that your CSV covers at least 28 days with reasonable data."
    )

latest = apple_df.iloc[-1]


# ===============================================================
# INTENSITY CHANGE ENGINE
# ===============================================================

global_model = joblib.load("global_intensity_model.pkl")

KCAL_PER_HOUR  = 420   # moderate activity MET estimate
est_duration_h = float(latest["active_energy"]) / KCAL_PER_HOUR
est_max_bpm    = 220 - USER_AGE

global_input = pd.DataFrame([{
    "Age":                      USER_AGE,
    "Weight (kg)":              float(latest["body_mass_avg"]),
    "Resting_BPM":              float(latest["resting_hr_avg"]),
    "Session_Duration (hours)": est_duration_h,
    "Calories_Burned":          float(latest["active_energy"]),
    "BMI":                      float(latest["bmi"]),
    "est_max_bpm":              est_max_bpm,
}])

baseline_intensity = global_model.predict(global_input)[0]

# ---------------------------------------------------------------
# PERSONAL ADJUSTMENT RULES
# ---------------------------------------------------------------
intensity_change = 0.0
readiness_mean   = apple_df["readiness"].mean()
readiness_std    = apple_df["readiness"].std()

# Readiness: 4-band z-score system
if   latest["readiness"] > readiness_mean + 0.5 * readiness_std:
    intensity_change += 0.08
elif latest["readiness"] > readiness_mean:
    intensity_change += 0.04
elif latest["readiness"] < readiness_mean - readiness_std:
    intensity_change -= 0.12
elif latest["readiness"] < readiness_mean - 0.5 * readiness_std:
    intensity_change -= 0.06

# ACWR: graduated thresholds (Gabbett 2016)
acwr = float(latest["acwr"])
if   acwr > 1.5:  intensity_change -= 0.15
elif acwr > 1.3:  intensity_change -= 0.08
elif acwr < 0.8:  intensity_change += 0.05
# 0.8–1.3 sweet spot → no change

# HRV trend: 3-day acute drop > 8% from 7-day baseline
if len(apple_df) >= 3:
    hrv_recent   = apple_df["hrv_avg"].iloc[-3:].mean()
    hrv_baseline = float(latest["hrv_avg_7d"])
    if pd.notna(hrv_recent) and pd.notna(hrv_baseline) and hrv_baseline > 0:
        if (hrv_baseline - hrv_recent) / hrv_baseline > 0.08:
            intensity_change -= 0.05

# Clamp to ±20%
intensity_change      = max(min(intensity_change, 0.20), -0.20)
recommended_intensity = baseline_intensity * (1 + intensity_change)

# ---------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("  WORKOUT INTENSITY RECOMMENDATION")
print("=" * 50)
print(f"  Date assessed:        {latest['date'].date()}")
print(f"  Readiness score:      {latest['readiness']:.3f}  (0=low, 1=high)")
print(f"  ACWR:                 {acwr:.2f}  (sweet spot 0.8–1.3)")
print(f"  Baseline TSS target:  {baseline_intensity:.1f}")
print(f"  Adjustment:           {intensity_change * 100:+.1f}%")
print(f"  Recommended TSS:      {recommended_intensity:.1f}")

if   recommended_intensity < 40:  label = "Easy / Active Recovery"
elif recommended_intensity < 70:  label = "Moderate Aerobic"
elif recommended_intensity < 100: label = "Threshold / Hard"
else:                              label = "VO2max / Race Effort"

print(f"  Effort zone:          {label}")
print("=" * 50 + "\n")