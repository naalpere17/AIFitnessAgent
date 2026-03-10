import pandas as pd
import joblib
import numpy as np

# Load the models you just trained
global_model = joblib.load("global_intensity_model.pkl")
personal_adapter = joblib.load("personal_fitness_adapter.pkl")

def generate_next_workout_intensity(latest_health_metrics, user_profile):
    """
    Calculates the recommended TSS for the next session.
    latest_health_metrics: Dict containing hrv_avg_7d, sleep_hours_7d, etc.
    user_profile: Dict containing Age, Weight, BMI.
    """
    
    # 1. Prepare input for Global Model
    # We use the user's static profile + a standard 1-hour session guess
    global_input = pd.DataFrame([{
        "Age": user_profile['Age'],
        "Weight (kg)": user_profile['Weight'],
        "Resting_BPM": latest_health_metrics['resting_hr_avg_7d'],
        "Session_Duration (hours)": 1.0, 
        "Calories_Burned": 420, # Baseline estimate
        "BMI": user_profile['BMI'],
        "est_max_bpm": 220 - user_profile['Age']
    }])
    
    # 2. Get the Baseline
    baseline_tss = global_model.predict(global_input)[0]
    
    # 3. Prepare input for Personal Adapter
    # We feed the baseline prediction into the adapter alongside recovery metrics
    adapter_input = pd.DataFrame([{
        "hrv_avg_7d": latest_health_metrics['hrv_avg_7d'],
        "sleep_hours_7d": latest_health_metrics['sleep_hours_7d'],
        "resting_hr_avg_7d": latest_health_metrics['resting_hr_avg_7d'],
        "acwr": latest_health_metrics['acwr'],
        "readiness": latest_health_metrics['readiness'],
        "global_baseline_pred": baseline_tss
    }])
    
    # 4. Generate Final Personalized TSS
    recommended_tss = personal_adapter.predict(adapter_input)[0]
    
    # 5. Map to Workout Zones
    if recommended_tss < 40:
        zone = "Zone 1: Recovery / Light Movement"
    elif recommended_tss < 70:
        zone = "Zone 2: Aerobic Base (Fat Burn)"
    elif recommended_tss < 100:
        zone = "Zone 3: Tempo / Threshold"
    else:
        zone = "Zone 4+: High Intensity / Intervals"
        
    return round(recommended_tss, 1), zone

# --- Example Usage ---
# Pull the last row from your apple_df to get 'latest' metrics
# recommended_tss, zone = generate_next_workout_intensity(latest_metrics, my_profile)