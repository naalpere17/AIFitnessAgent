import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# User Profile
USER_AGE = 22
USER_HEIGHT_M = 1.75
USER_WEIGHT_KG = 70
EST_MAX_BPM = 220 - USER_AGE

# Load data and global model
global_model = joblib.load("global_intensity_model.pkl")
apple_df = pd.read_csv("health_last_60_days.csv")

# Cleaning & Feature Engineering
apple_df["body_mass_avg"] = apple_df["body_mass_avg"].fillna(USER_WEIGHT_KG)
apple_df["BMI"] = apple_df["body_mass_avg"] / (USER_HEIGHT_M ** 2)
apple_df["resting_hr_avg"] = apple_df["resting_hr_avg"].fillna(apple_df["resting_hr_avg"].median())
apple_df["active_energy"] = apple_df["active_energy"].fillna(0)

# Rolling averages
for col in ["sleep_hours", "hrv_avg", "resting_hr_avg", "active_energy"]:
    apple_df[f"{col}_7d"] = apple_df[col].rolling(7, min_periods=4).mean()

apple_df["acute_load"] = apple_df["active_energy"].rolling(7, min_periods=4).mean()
apple_df["chronic_load"] = apple_df["active_energy"].rolling(28, min_periods=14).mean()
apple_df["acwr"] = apple_df["acute_load"] / apple_df["chronic_load"]

# Readiness and Target TSS
apple_df["readiness"] = (0.45 * (apple_df["hrv_avg_7d"] / apple_df["hrv_avg_7d"].max()) + 
                         0.35 * (apple_df["sleep_hours_7d"] / 10) - 
                         0.20 * (apple_df["resting_hr_avg_7d"] / 100)).clip(0, 1)

apple_df["target_tss"] = ((apple_df["active_energy"] / 420) * (apple_df["resting_hr_avg"] * 1.25)**2 / 
                          (apple_df["resting_hr_avg"] * EST_MAX_BPM) * 100).clip(0, 150)

apple_df = apple_df.dropna(subset=["acwr", "readiness", "target_tss"])

# Generate Global Baseline
personal_as_global = pd.DataFrame({
    "Age": USER_AGE, "BMI": apple_df["BMI"], "Resting_BPM": apple_df["resting_hr_avg"],
    "Weight (kg)": apple_df["body_mass_avg"], "Session_Duration (hours)": apple_df["active_energy"] / 420,
    "Calories_Burned": apple_df["active_energy"], "acwr": apple_df["acwr"], 
    "readiness": apple_df["readiness"], "acute_load": apple_df["acute_load"], 
    "chronic_load": apple_df["chronic_load"], "est_max_bpm": EST_MAX_BPM
})
apple_df["global_baseline_pred"] = global_model.predict(personal_as_global)

# Train Personal Adapter
features = ["hrv_avg_7d", "sleep_hours_7d", "resting_hr_avg_7d", "acwr", "readiness", "global_baseline_pred"]
X_train, X_test, y_train, y_test = train_test_split(apple_df[features], apple_df["target_tss"], test_size=0.2, random_state=42)

personal_adapter = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
personal_adapter.fit(X_train, y_train)

# Report results
preds = personal_adapter.predict(X_test)
print(f"Personal Model MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"Personalization Gain: {((mean_absolute_error(y_test, X_test['global_baseline_pred']) - mean_absolute_error(y_test, preds)) / mean_absolute_error(y_test, X_test['global_baseline_pred']) * 100):.1f}%")

joblib.dump(personal_adapter, "personal_fitness_adapter.pkl")