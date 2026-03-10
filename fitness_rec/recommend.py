"""
recommend.py
============
Phase 2: Train a personal adapter on Apple Health + GPX + RPE log data.

TSS label priority (highest to lowest):
  1. RPE log (rpe_log.csv)     — real perceived effort, most accurate
  2. GPX workout (workouts.csv) — real measured pace/distance
  3. Estimated from active energy — fallback only

Temporal weighting: recent days weighted more heavily using
exponential decay (half-life configurable below).

Run after train.py and parse_gpx.py.
Outputs: personal_adapter.pkl, personal_scaler.pkl, adapter_feature_names.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ===============================================================
# USER PROFILE — must match train.py
# ===============================================================
USER_AGE       = 22
USER_HEIGHT_M  = 1.75
USER_WEIGHT_KG = 70
USER_GENDER    = "male"    # "male" or "female"

# Gender-adjusted max HR formula (Tanaka et al. 2001)
if USER_GENDER == "female":
    EST_MAX_BPM = int(206 - (0.88 * USER_AGE))
else:
    EST_MAX_BPM = int(208 - (0.7 * USER_AGE))

GENDER_ENCODED = 1 if USER_GENDER == "female" else 0
KCAL_PER_HOUR  = 400 if USER_GENDER == "female" else 420
HRV_SEX_ADJUSTMENT = 0.88 if USER_GENDER == "female" else 1.0
HALF_LIFE_DAYS = 60

# ===============================================================
# LOAD DATA & GLOBAL MODEL
# ===============================================================
global_model    = joblib.load("global_intensity_model.pkl")
global_features = joblib.load("global_feature_names.pkl")

apple_df = pd.read_csv("output/health_last_60_days.csv")
apple_df["date"] = pd.to_datetime(apple_df["date"])
apple_df = apple_df.sort_values("date").reset_index(drop=True)

# ===============================================================
# CLEANING
# ===============================================================
apple_df["body_mass_avg"]  = apple_df["body_mass_avg"].fillna(USER_WEIGHT_KG)
apple_df["bmi"]            = apple_df["body_mass_avg"] / (USER_HEIGHT_M ** 2)
apple_df["active_energy"]  = apple_df["active_energy"].fillna(0)
apple_df["resting_hr_avg"] = apple_df["resting_hr_avg"].fillna(
    apple_df["resting_hr_avg"].median()
)
apple_df.loc[apple_df["sleep_hours"] < 2, "sleep_hours"] = np.nan
apple_df["sleep_hours"] = apple_df["sleep_hours"].fillna(apple_df["sleep_hours"].median())

# ===============================================================
# ROLLING FEATURES
# ===============================================================
for col in ["sleep_hours", "hrv_avg", "resting_hr_avg", "active_energy"]:
    apple_df[f"{col}_7d"] = apple_df[col].rolling(7, min_periods=4).mean()

apple_df["acute_load"]   = apple_df["active_energy"].rolling(7,  min_periods=4).mean()
apple_df["chronic_load"] = apple_df["active_energy"].rolling(28, min_periods=14).mean()
apple_df["acwr"]         = np.where(
    apple_df["chronic_load"] > 0,
    apple_df["acute_load"] / apple_df["chronic_load"], np.nan
)

# ===============================================================
# READINESS
# ===============================================================
def minmax_norm(s):
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

apple_df["hrv_avg_7d_adj"] = apple_df["hrv_avg_7d"] / HRV_SEX_ADJUSTMENT

apple_df["readiness"] = (
    0.45 * minmax_norm(apple_df["hrv_avg_7d_adj"])
    + 0.35 * minmax_norm(apple_df["sleep_hours_7d"])
    + 0.20 * (1 - minmax_norm(apple_df["resting_hr_avg_7d"]))
)

for col in ["hrv_avg_7d", "hrv_avg_7d_adj", "sleep_hours_7d", "resting_hr_avg_7d",
            "active_energy_7d", "acute_load", "chronic_load"]:
    apple_df[col] = apple_df[col].fillna(apple_df[col].median())

# ===============================================================
# TSS LABELS — three-tier priority system
# ===============================================================

# Tier 3 (fallback): estimated TSS from active energy
intensity_pct   = 0.55 + (apple_df["readiness"] * 0.20)
est_workout_bpm = (intensity_pct * EST_MAX_BPM).clip(80, 185)
est_duration    = (apple_df["active_energy"] / KCAL_PER_HOUR).clip(0, 1.5)

apple_df["tss_estimated"] = (
    est_duration * est_workout_bpm ** 2
    / (apple_df["resting_hr_avg"] * EST_MAX_BPM) * 100
).clip(0, 150)

apple_df["target_tss"] = apple_df["tss_estimated"]
apple_df["tss_source"] = "estimated"

# Tier 2: real TSS from GPX workouts
if os.path.exists("workouts.csv"):
    workouts_df = pd.read_csv("workouts.csv")
    workouts_df["date"] = pd.to_datetime(workouts_df["date"]).dt.normalize()
    daily_tss = workouts_df.groupby("date")["tss_gpx"].sum().reset_index()
    apple_df  = apple_df.merge(daily_tss, on="date", how="left")
    mask_gpx  = apple_df["tss_gpx"].notna() & (apple_df["tss_gpx"] > 0)
    apple_df.loc[mask_gpx, "target_tss"] = apple_df.loc[mask_gpx, "tss_gpx"]
    apple_df.loc[mask_gpx, "tss_source"] = "gpx"
else:
    apple_df["tss_gpx"] = np.nan

# Tier 1: RPE log — highest priority, overwrites everything
if os.path.exists("rpe_log.csv"):
    rpe_df    = pd.read_csv("rpe_log.csv")
    rpe_df["date"] = pd.to_datetime(rpe_df["date"]).dt.normalize()
    daily_rpe = rpe_df.groupby("date")["tss_rpe"].sum().reset_index()
    apple_df  = apple_df.merge(daily_rpe, on="date", how="left")
    mask_rpe  = apple_df["tss_rpe"].notna() & (apple_df["tss_rpe"] > 0)
    apple_df.loc[mask_rpe, "target_tss"] = apple_df.loc[mask_rpe, "tss_rpe"]
    apple_df.loc[mask_rpe, "tss_source"] = "rpe"
else:
    apple_df["tss_rpe"] = np.nan

# Label source summary
source_counts = apple_df["tss_source"].value_counts()
print(f"Gender: {USER_GENDER}  |  Est. HRmax: {EST_MAX_BPM} bpm  |  "
      f"kcal/h: {KCAL_PER_HOUR}")
print("\nTSS label sources:")
for src in ["rpe", "gpx", "estimated"]:
    count = source_counts.get(src, 0)
    pct   = count / len(apple_df) * 100
    tag   = " ← ground truth" if src == "rpe" else \
            " ← real measured" if src == "gpx" else " ← fallback"
    print(f"  {src:<12} {count:>4} days ({pct:.1f}%){tag}")

apple_df = apple_df.dropna(subset=["acwr", "readiness", "target_tss"])

print(f"\nPersonal dataset: {len(apple_df)} rows")
print(f"Active energy  — min: {apple_df['active_energy'].min():.0f}  "
      f"max: {apple_df['active_energy'].max():.0f}  "
      f"mean: {apple_df['active_energy'].mean():.0f} kcal")
print(f"TSS labels     — min: {apple_df['target_tss'].min():.1f}  "
      f"max: {apple_df['target_tss'].max():.1f}  "
      f"mean: {apple_df['target_tss'].mean():.1f}")

# ===============================================================
# GLOBAL BASELINE PREDICTIONS
# gender_encoded is added as a scalar column to match train.py
# ===============================================================
global_input = pd.DataFrame({
    "Age":                       USER_AGE,
    "gender_encoded":            GENDER_ENCODED,          # scalar → broadcast
    "bmi":                       apple_df["bmi"],
    "Resting_BPM":               apple_df["resting_hr_avg"],
    "Weight (kg)":               apple_df["body_mass_avg"],
    "Session_Duration (hours)":  (apple_df["active_energy"] / KCAL_PER_HOUR).clip(0, 1.5),
    "Calories_Burned":           apple_df["active_energy"],
    "acwr":                      apple_df["acwr"],
    "readiness":                 apple_df["readiness"],
    "acute_load":                apple_df["acute_load"],
    "chronic_load":              apple_df["chronic_load"],
    "est_max_bpm":               EST_MAX_BPM,
})[global_features]   # reorder to match trained feature order exactly

apple_df["global_baseline"] = global_model.predict(global_input)

# ===============================================================
# PERSONAL ADAPTER
# ===============================================================
ADAPTER_FEATURES = [
    "hrv_avg_7d", "sleep_hours_7d", "resting_hr_avg_7d",
    "acwr", "readiness", "acute_load", "chronic_load",
    "global_baseline",
]

X = apple_df[ADAPTER_FEATURES].values
y = apple_df["target_tss"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chronological split — no shuffle (time series)
split    = max(1, int(len(X_scaled) * 0.80))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Temporal weighting — exponential decay
today         = apple_df["date"].max()
days_ago      = (today - apple_df["date"]).dt.days.values
weights       = np.power(0.5, days_ago / HALF_LIFE_DAYS)
weights_train = weights[:split]

# Boost real label days
if apple_df["tss_source"].eq("rpe").any():
    rpe_mask = apple_df["tss_source"].values[:split] == "rpe"
    weights_train[rpe_mask] *= 5.0
    print(f"\nBoosted {rpe_mask.sum()} RPE-logged days (5× weight)")

if apple_df["tss_source"].eq("gpx").any():
    gpx_mask = apple_df["tss_source"].values[:split] == "gpx"
    weights_train[gpx_mask] *= 2.0
    print(f"Boosted {gpx_mask.sum()} GPX-measured days (2× weight)")

adapter = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
adapter.fit(X_train, y_train, sample_weight=weights_train)

preds        = adapter.predict(X_test)
baseline_mae = mean_absolute_error(y_test, apple_df["global_baseline"].values[split:])
adapter_mae  = mean_absolute_error(y_test, preds)
gain         = (baseline_mae - adapter_mae) / baseline_mae * 100

print(f"\nGlobal baseline MAE:  {baseline_mae:.2f} TSS")
print(f"Personal adapter MAE: {adapter_mae:.2f} TSS")
print(f"Personalisation gain: {gain:.1f}%")

print("\nFeature importances:")
for feat, imp in sorted(zip(ADAPTER_FEATURES, adapter.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {feat:<30} {imp:.3f}")

# Save all artifacts including gender config for predict.py
joblib.dump(adapter,          "personal_adapter.pkl")
joblib.dump(scaler,           "personal_scaler.pkl")
joblib.dump(ADAPTER_FEATURES, "adapter_feature_names.pkl")
joblib.dump({
    "gender":        USER_GENDER,
    "gender_encoded": GENDER_ENCODED,
    "est_max_bpm":   EST_MAX_BPM,
    "kcal_per_hour": KCAL_PER_HOUR,
    "hrv_sex_adj":   HRV_SEX_ADJUSTMENT,
}, "user_config.pkl")

print("\nSaved → personal_adapter.pkl, personal_scaler.pkl, user_config.pkl")