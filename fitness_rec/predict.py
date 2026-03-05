import pandas as pd
import joblib
import numpy as np

# ===============================================================
# 1. USER PROFILE
# ===============================================================
USER_AGE       = 22
USER_HEIGHT_M  = 1.75
USER_WEIGHT_KG = 70
USER_BMI       = USER_WEIGHT_KG / (USER_HEIGHT_M ** 2)
EST_MAX_BPM    = 220 - USER_AGE

# ===============================================================
# 2. LOAD MODELS
# ===============================================================
global_model     = joblib.load("global_intensity_model.pkl")
personal_adapter = joblib.load("personal_fitness_adapter.pkl")

# ===============================================================
# 3. PREPARE LATEST DATA
# ===============================================================
apple_df = pd.read_csv("health_last_60_days.csv")
apple_df["body_mass_avg"] = apple_df["body_mass_avg"].fillna(USER_WEIGHT_KG)

# Apply same rolling logic used in training to get the most recent state
for col in ["sleep_hours", "hrv_avg", "resting_hr_avg", "active_energy"]:
    apple_df[f"{col}_7d"] = apple_df[col].rolling(7, min_periods=4).mean()

apple_df["acute_load"]   = apple_df["active_energy"].rolling(7,  min_periods=4).mean()
apple_df["chronic_load"] = apple_df["active_energy"].rolling(28, min_periods=14).mean()
apple_df["acwr"]         = apple_df["acute_load"] / apple_df["chronic_load"]
apple_df["readiness"]    = (
    0.45 * (apple_df["hrv_avg_7d"] / apple_df["hrv_avg_7d"].max()) +
    0.35 * (apple_df["sleep_hours_7d"] / 10) -
    0.20 * (apple_df["resting_hr_avg_7d"] / 100)
).clip(0, 1)

# Get the absolute latest data point
latest = apple_df.iloc[-1]

# ===============================================================
# 4. GENERATE RECOMMENDATION
# ===============================================================

# Step A: Get Global Baseline (Typical intensity for your demographic)
global_input = pd.DataFrame([{
    "Age":                       USER_AGE,
    "BMI":                       USER_BMI,
    "Resting_BPM":               latest["resting_hr_avg"],
    "Weight (kg)":               USER_WEIGHT_KG,
    "Session_Duration (hours)":  1.0,  # Standard 1-hour session target
    "Calories_Burned":           400,  # Baseline burn estimate
    "acwr":                      latest["acwr"],
    "readiness":                 latest["readiness"],
    "acute_load":                latest["acute_load"],
    "chronic_load":              latest["chronic_load"],
    "est_max_bpm":               EST_MAX_BPM
}])
baseline_tss = global_model.predict(global_input)[0]

# Step B: Get Personalized Recommendation (Adjusted for your current recovery)
personal_input = pd.DataFrame([{
    "hrv_avg_7d":        latest["hrv_avg_7d"],
    "sleep_hours_7d":    latest["sleep_hours_7d"],
    "resting_hr_avg_7d": latest["resting_hr_avg_7d"],
    "acwr":              latest["acwr"],
    "readiness":         latest["readiness"],
    "global_baseline_pred": baseline_tss
}])

final_tss = personal_adapter.predict(personal_input)[0]

# ===============================================================
# 5. OUTPUT
# ===============================================================
if   final_tss < 40:  zone = "Zone 1: Easy / Active Recovery"
elif final_tss < 70:  zone = "Zone 2: Moderate Aerobic"
elif final_tss < 100: zone = "Zone 3: Threshold / Hard"
else:                 zone = "Zone 4+: VO2max / Maximum Effort"

print("\n" + "="*40)
print(f" TODAY'S ADAPTIVE WORKOUT PLAN")
print("="*40)
print(f" Readiness Score:  {latest['readiness']:.2f}")
print(f" Fatigue (ACWR):   {latest['acwr']:.2f}")
print(f" Target Intensity: {final_tss:.1f} TSS")
print(f" Training Zone:    {zone}")
print("="*40 + "\n")