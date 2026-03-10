"""
predict.py
==========
Inference: load trained models and return today's workout recommendation.
Exports: get_recommendation(verbose=True) -> dict with all recommendation fields
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import date

from fitness_rec.config import (
    USER_AGE, USER_HEIGHT_M, USER_WEIGHT_KG,
    HEALTH_CSV, RPE_CSV,
)


def _minmax_norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def _load_health_latest() -> pd.Series:
    """Load Apple Health data and return the most recent row with all features."""
    config       = joblib.load("user_config.pkl")
    kcal_per_h   = config["kcal_per_hour"]
    hrv_sex_adj  = config["hrv_sex_adj"]

    df = pd.read_csv(HEALTH_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["body_mass_avg"]  = df["body_mass_avg"].fillna(USER_WEIGHT_KG)
    df["bmi"]            = df["body_mass_avg"] / (USER_HEIGHT_M ** 2)
    df["active_energy"]  = df["active_energy"].fillna(0)
    df["resting_hr_avg"] = df["resting_hr_avg"].fillna(df["resting_hr_avg"].median())
    df.loc[df["sleep_hours"] < 2, "sleep_hours"] = np.nan
    df["sleep_hours"]    = df["sleep_hours"].fillna(df["sleep_hours"].median())

    for col in ["sleep_hours", "hrv_avg", "resting_hr_avg", "active_energy"]:
        df[f"{col}_7d"] = df[col].rolling(7, min_periods=4).mean()

    df["acute_load"]   = df["active_energy"].rolling(7,  min_periods=4).mean()
    df["chronic_load"] = df["active_energy"].rolling(28, min_periods=14).mean()
    df["acwr"]         = np.where(
        df["chronic_load"] > 0,
        df["acute_load"] / df["chronic_load"], np.nan
    )

    df["hrv_avg_7d_adj"] = df["hrv_avg_7d"] / hrv_sex_adj
    df["readiness"] = (
        0.45 * _minmax_norm(df["hrv_avg_7d_adj"])
        + 0.35 * _minmax_norm(df["sleep_hours_7d"])
        + 0.20 * (1 - _minmax_norm(df["resting_hr_avg_7d"]))
    )

    for col in ["hrv_avg_7d", "hrv_avg_7d_adj", "sleep_hours_7d",
                "resting_hr_avg_7d", "active_energy_7d", "acute_load", "chronic_load"]:
        df[col] = df[col].fillna(df[col].median())

    return df.iloc[-1].copy()


def _get_rpe_today() -> float | None:
    """Return total TSS logged today via RPE, or None if none logged."""
    if not os.path.exists(RPE_CSV):
        return None
    rpe_df = pd.read_csv(RPE_CSV)
    rpe_df["date"] = pd.to_datetime(rpe_df["date"])
    today_logs = rpe_df[rpe_df["date"].dt.date == date.today()]
    if today_logs.empty:
        return None
    return float(today_logs["tss_rpe"].sum())


def _tss_to_zone(tss: float) -> tuple[str, str]:
    """Map TSS to zone label and advice string."""
    zones = {
        "Zone 1 — Easy / Active Recovery":  "Light movement only. Walk, stretch, or rest. Your body needs it.",
        "Zone 2 — Moderate Aerobic":        "Steady aerobic work. Conversational pace, 45–75 min.",
        "Zone 3 — Threshold / Hard":        "Tempo runs, intervals, or hard lifting. Keep sessions under 60 min.",
        "Zone 4+ — VO2max / Maximum Effort": "Peak effort day. Race pace or max intensity. Recover well after.",
    }
    if   tss < 40:  zone = "Zone 1 — Easy / Active Recovery"
    elif tss < 70:  zone = "Zone 2 — Moderate Aerobic"
    elif tss < 100: zone = "Zone 3 — Threshold / Hard"
    else:           zone = "Zone 4+ — VO2max / Maximum Effort"
    return zone, zones[zone]


def get_recommendation(verbose: bool = True) -> dict:
    """
    Run inference and return today's workout recommendation.

    Returns
    -------
    dict:
        today           str   — date string
        data_lag_days   int   — days Apple Health data is behind today
        gender          str   — user gender
        est_max_bpm     int   — sex-specific HRmax
        readiness       float — 0–1 readiness score
        acwr            float — acute:chronic workload ratio
        global_baseline float — population model TSS prediction
        recommended_tss float — final personalised TSS recommendation
        zone            str   — training zone label
        advice          str   — plain-language prescription
        rpe_logged_tss  float | None — TSS already logged today (or None)
        remaining_tss   float | None — TSS remaining today (or None)
    """
    config          = joblib.load("user_config.pkl")
    global_model    = joblib.load("global_intensity_model.pkl")
    global_features = joblib.load("global_feature_names.pkl")
    adapter         = joblib.load("personal_adapter.pkl")
    scaler          = joblib.load("personal_scaler.pkl")

    USER_GENDER   = config["gender"]
    EST_MAX_BPM   = config["est_max_bpm"]
    KCAL_PER_HOUR = config["kcal_per_hour"]
    gender_encoded = config["gender_encoded"]

    latest    = _load_health_latest()
    today_str = date.today().isoformat()

    health_date = latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]
    data_lag    = (date.today() - health_date).days

    # Step 1 — global baseline
    global_input = pd.DataFrame([{
        "Age":                       USER_AGE,
        "gender_encoded":            gender_encoded,
        "bmi":                       float(latest["bmi"]),
        "Resting_BPM":               float(latest["resting_hr_avg"]),
        "Weight (kg)":               USER_WEIGHT_KG,
        "Session_Duration (hours)":  min(float(latest["active_energy"]) / KCAL_PER_HOUR, 1.5),
        "Calories_Burned":           float(latest["active_energy"]),
        "acwr":                      float(latest["acwr"]),
        "readiness":                 float(latest["readiness"]),
        "acute_load":                float(latest["acute_load"]),
        "chronic_load":              float(latest["chronic_load"]),
        "est_max_bpm":               EST_MAX_BPM,
    }])[global_features]

    global_baseline = float(global_model.predict(global_input)[0])

    # Step 2 — personal adapter
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
    final_tss = max(0.0, float(adapter.predict(scaler.transform(adapter_input))[0]))

    zone, advice   = _tss_to_zone(final_tss)
    rpe_logged_tss = _get_rpe_today()
    remaining_tss  = max(0.0, final_tss - rpe_logged_tss) if rpe_logged_tss is not None else None

    result = {
        "today":           today_str,
        "data_lag_days":   data_lag,
        "gender":          USER_GENDER,
        "est_max_bpm":     EST_MAX_BPM,
        "readiness":       round(float(latest["readiness"]), 2),
        "acwr":            round(float(latest["acwr"]), 2),
        "global_baseline": round(global_baseline, 1),
        "recommended_tss": round(final_tss, 1),
        "zone":            zone,
        "advice":          advice,
        "rpe_logged_tss":  round(rpe_logged_tss, 1) if rpe_logged_tss is not None else None,
        "remaining_tss":   round(remaining_tss, 1) if remaining_tss is not None else None,
    }

    if verbose:
        lag_note = f" (Apple Health data is {data_lag} day(s) behind)" if data_lag > 0 else ""
        print("\n" + "=" * 45)
        print("   TODAY'S WORKOUT RECOMMENDATION")
        print("=" * 45)
        print(f"  Today:            {today_str}{lag_note}")
        print(f"  Gender:           {USER_GENDER.capitalize()}  |  HRmax: {EST_MAX_BPM} bpm")
        print(f"  Readiness:        {result['readiness']:.2f} / 1.00")
        print(f"  ACWR:             {result['acwr']:.2f}  (sweet spot 0.8–1.3)")
        print(f"  Global baseline:  {result['global_baseline']:.1f} TSS")
        print(f"  Recommended TSS:  {result['recommended_tss']:.1f} TSS")
        print(f"  Training zone:    {zone}")
        print(f"\n  {advice}")
        if rpe_logged_tss is not None:
            print(f"\n  Already logged:   {rpe_logged_tss:.1f} TSS today")
            print(f"  Remaining:        {remaining_tss:.1f} TSS")
            if remaining_tss < 5:
                print("  → You've hit today's target. Good work.")
        print("=" * 45 + "\n")

    return result


if __name__ == "__main__":
    get_recommendation()
