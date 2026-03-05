"""
predict.py
==========
Inference: load trained models and predict today's recommended intensity.
Run after train.py, recommend.py.
Also incorporates any RPE logs newer than the Apple Health window.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import date

# ===============================================================
# USER PROFILE — must match train.py and recommend.py
# ===============================================================
USER_AGE       = 22
USER_HEIGHT_M  = 1.75
USER_WEIGHT_KG = 70
EST_MAX_BPM    = 220 - USER_AGE

# ===============================================================
# LOAD MODELS
# ===============================================================
global_model     = joblib.load("global_intensity_model.pkl")
global_features  = joblib.load("global_feature_names.pkl")
adapter          = joblib.load("personal_adapter.pkl")
scaler           = joblib.load("personal_scaler.pkl")
adapter_features = joblib.load("adapter_feature_names.pkl")

# ===============================================================
# LOAD & PREPARE APPLE HEALTH DATA
# ===============================================================
apple_df = pd.read_csv("health_last_60_days.csv")
apple_df["date"] = pd.to_datetime(apple_df["date"])
apple_df = apple_df.sort_values("date").reset_index(drop=True)

apple_df["body_mass_avg"]  = apple_df["body_mass_avg"].fillna(USER_WEIGHT_KG)
apple_df["bmi"]            = apple_df["body_mass_avg"] / (USER_HEIGHT_M ** 2)
apple_df["active_energy"]  = apple_df["active_energy"].fillna(0)
apple_df["resting_hr_avg"] = apple_df["resting_hr_avg"].fillna(
    apple_df["resting_hr_avg"].median()
)
apple_df.loc[apple_df["sleep_hours"] < 2, "sleep_hours"] = np.nan
apple_df["sleep_hours"] = apple_df["sleep_hours"].fillna(apple_df["sleep_hours"].median())

for col in ["sleep_hours", "hrv_avg", "resting_hr_avg", "active_energy"]:
    apple_df[f"{col}_7d"] = apple_df[col].rolling(7, min_periods=4).mean()

apple_df["acute_load"]   = apple_df["active_energy"].rolling(7,  min_periods=4).mean()
apple_df["chronic_load"] = apple_df["active_energy"].rolling(28, min_periods=14).mean()
apple_df["acwr"]         = np.where(
    apple_df["chronic_load"] > 0,
    apple_df["acute_load"] / apple_df["chronic_load"], np.nan
)

def minmax_norm(s):
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

apple_df["readiness"] = (
    0.45 * minmax_norm(apple_df["hrv_avg_7d"])
    + 0.35 * minmax_norm(apple_df["sleep_hours_7d"])
    + 0.20 * (1 - minmax_norm(apple_df["resting_hr_avg_7d"]))
)

for col in ["hrv_avg_7d", "sleep_hours_7d", "resting_hr_avg_7d",
            "active_energy_7d", "acute_load", "chronic_load"]:
    apple_df[col] = apple_df[col].fillna(apple_df[col].median())

# Use latest row as base state
latest = apple_df.iloc[-1].copy()

# ===============================================================
# OVERRIDE WITH TODAY'S RPE LOG IF AVAILABLE
# If you've logged a workout today (even if newer than Apple Health
# window), use today's date and carry forward the last known
# recovery state from Apple Health.
# ===============================================================
today_str = date.today().isoformat()
rpe_today_tss = None

if os.path.exists("rpe_log.csv"):
    rpe_df = pd.read_csv("rpe_log.csv")
    rpe_df["date"] = pd.to_datetime(rpe_df["date"])
    today_logs = rpe_df[rpe_df["date"].dt.date == date.today()]
    if not today_logs.empty:
        rpe_today_tss = today_logs["tss_rpe"].sum()
        print(f"  Found {len(today_logs)} RPE log(s) for today — "
              f"total TSS logged: {rpe_today_tss:.1f}")

# ===============================================================
# STEP 1 — GLOBAL BASELINE
# ===============================================================
global_input = pd.DataFrame([{
    "Age":                       USER_AGE,
    "bmi":                       float(latest["bmi"]),
    "Resting_BPM":               float(latest["resting_hr_avg"]),
    "Weight (kg)":               USER_WEIGHT_KG,
    "Session_Duration (hours)":  min(float(latest["active_energy"]) / 420, 1.5),
    "Calories_Burned":           float(latest["active_energy"]),
    "acwr":                      float(latest["acwr"]),
    "readiness":                 float(latest["readiness"]),
    "acute_load":                float(latest["acute_load"]),
    "chronic_load":              float(latest["chronic_load"]),
    "est_max_bpm":               EST_MAX_BPM,
}])[global_features]

global_baseline = global_model.predict(global_input)[0]

# ===============================================================
# STEP 2 — PERSONAL ADAPTER
# ===============================================================
adapter_input = np.array([[
    float(latest["hrv_avg_7d"]),
    float(latest["sleep_hours_7d"]),
    float(latest["resting_hr_avg_7d"]),
    float(latest["acwr"]),
    float(latest["readiness"]),
    float(latest["acute_load"]),
    float(latest["chronic_load"]),
    global_baseline,
]])

adapter_input_scaled = scaler.transform(adapter_input)
final_tss = max(0.0, adapter.predict(adapter_input_scaled)[0])

# ===============================================================
# OUTPUT
# ===============================================================
if   final_tss < 40:  zone = "Zone 1 — Easy / Active Recovery"
elif final_tss < 70:  zone = "Zone 2 — Moderate Aerobic"
elif final_tss < 100: zone = "Zone 3 — Threshold / Hard"
else:                 zone = "Zone 4+ — VO2max / Maximum Effort"

advice = {
    "Zone 1 — Easy / Active Recovery": "Light movement only. Walk, stretch, or rest. Your body needs it.",
    "Zone 2 — Moderate Aerobic":       "Steady aerobic work. Conversational pace, 45–75 min.",
    "Zone 3 — Threshold / Hard":       "Tempo runs, intervals, or hard lifting. Keep sessions under 60 min.",
    "Zone 4+ — VO2max / Maximum Effort": "Peak effort day. Race pace or max intensity. Recover well after.",
}

# Health data date vs today
health_date  = latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]
data_lag     = (date.today() - health_date).days
lag_note     = f" (Apple Health data is {data_lag} day(s) behind today)" if data_lag > 0 else ""

print("\n" + "=" * 45)
print("   TODAY'S WORKOUT RECOMMENDATION")
print("=" * 45)
print(f"  Today:            {today_str}{lag_note}")
print(f"  Readiness:        {latest['readiness']:.2f} / 1.00")
print(f"  ACWR:             {latest['acwr']:.2f}  (sweet spot 0.8–1.3)")
print(f"  Global baseline:  {global_baseline:.1f} TSS")
print(f"  Recommended TSS:  {final_tss:.1f} TSS")
print(f"  Training zone:    {zone}")
print(f"\n  {advice[zone]}")

if rpe_today_tss is not None:
    remaining = max(0, final_tss - rpe_today_tss)
    print(f"\n  Already logged:   {rpe_today_tss:.1f} TSS today")
    print(f"  Remaining:        {remaining:.1f} TSS")
    if remaining < 5:
        print(f"  → You've hit today's target. Good work.")

print("=" * 45 + "\n")