"""
recommend.py
============
Phase 2: Train a personal adapter on Apple Health + GPX + RPE log data.
Exports: train_personal_adapter(verbose=True) -> dict with mae, gain, importances
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from fitness_rec.config import (
    USER_AGE, USER_HEIGHT_M, USER_WEIGHT_KG, USER_GENDER,
    EST_MAX_BPM, GENDER_ENCODED, KCAL_PER_HOUR, HRV_SEX_ADJUSTMENT,
    HALF_LIFE_DAYS, HEALTH_CSV, WORKOUTS_CSV, RPE_CSV,
)

ADAPTER_FEATURES = [
    "hrv_avg_7d", "sleep_hours_7d", "resting_hr_avg_7d",
    "acwr", "readiness", "acute_load", "chronic_load",
    "global_baseline",
]


def _minmax_norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def _load_and_prepare_health() -> pd.DataFrame:
    """Load Apple Health CSV and compute rolling features + readiness."""
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

    df["hrv_avg_7d_adj"] = df["hrv_avg_7d"] / HRV_SEX_ADJUSTMENT
    df["readiness"] = (
        0.45 * _minmax_norm(df["hrv_avg_7d_adj"])
        + 0.35 * _minmax_norm(df["sleep_hours_7d"])
        + 0.20 * (1 - _minmax_norm(df["resting_hr_avg_7d"]))
    )

    for col in ["hrv_avg_7d", "hrv_avg_7d_adj", "sleep_hours_7d",
                "resting_hr_avg_7d", "active_energy_7d", "acute_load", "chronic_load"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def _attach_tss_labels(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Attach three-tier TSS labels: estimated → GPX → RPE."""
    intensity_pct   = 0.55 + (df["readiness"] * 0.20)
    est_workout_bpm = (intensity_pct * EST_MAX_BPM).clip(80, 185)
    est_duration    = (df["active_energy"] / KCAL_PER_HOUR).clip(0, 1.5)

    df["tss_estimated"] = (
        est_duration * est_workout_bpm ** 2
        / (df["resting_hr_avg"] * EST_MAX_BPM) * 100
    ).clip(0, 150)

    df["target_tss"] = df["tss_estimated"]
    df["tss_source"] = "estimated"

    # Tier 2 — GPX
    if os.path.exists(WORKOUTS_CSV):
        workouts = pd.read_csv(WORKOUTS_CSV)
        workouts["date"] = pd.to_datetime(workouts["date"]).dt.normalize()
        daily_gpx = workouts.groupby("date")["tss_gpx"].sum().reset_index()
        df = df.merge(daily_gpx, on="date", how="left")
        mask = df["tss_gpx"].notna() & (df["tss_gpx"] > 0)
        df.loc[mask, "target_tss"] = df.loc[mask, "tss_gpx"]
        df.loc[mask, "tss_source"] = "gpx"
    else:
        df["tss_gpx"] = np.nan

    # Tier 1 — RPE
    if os.path.exists(RPE_CSV):
        rpe = pd.read_csv(RPE_CSV)
        rpe["date"] = pd.to_datetime(rpe["date"]).dt.normalize()
        daily_rpe = rpe.groupby("date")["tss_rpe"].sum().reset_index()
        df = df.merge(daily_rpe, on="date", how="left")
        mask = df["tss_rpe"].notna() & (df["tss_rpe"] > 0)
        df.loc[mask, "target_tss"] = df.loc[mask, "tss_rpe"]
        df.loc[mask, "tss_source"] = "rpe"
    else:
        df["tss_rpe"] = np.nan

    if verbose:
        source_counts = df["tss_source"].value_counts()
        print(f"Gender: {USER_GENDER}  |  Est. HRmax: {EST_MAX_BPM} bpm  |  "
              f"kcal/h: {KCAL_PER_HOUR}")
        print("\nTSS label sources:")
        for src in ["rpe", "gpx", "estimated"]:
            count = source_counts.get(src, 0)
            pct   = count / len(df) * 100
            tag   = " ← ground truth" if src == "rpe" else \
                    " ← real measured" if src == "gpx" else " ← fallback"
            print(f"  {src:<12} {count:>4} days ({pct:.1f}%){tag}")

    return df


def train_personal_adapter(verbose: bool = True) -> dict:
    """
    Train the personal Gradient Boosting adapter on Apple Health + GPX + RPE data.

    Returns
    -------
    dict:
        baseline_mae    float — global model MAE on test set
        adapter_mae     float — personal adapter MAE on test set
        gain_pct        float — personalisation gain over baseline (%)
        importances     dict  — feature name → importance score
    """
    global_model    = joblib.load("global_intensity_model.pkl")
    global_features = joblib.load("global_feature_names.pkl")

    df = _load_and_prepare_health()
    df = _attach_tss_labels(df, verbose)
    df = df.dropna(subset=["acwr", "readiness", "target_tss"])

    if verbose:
        print(f"\nPersonal dataset: {len(df)} rows")
        print(f"Active energy  — min: {df['active_energy'].min():.0f}  "
              f"max: {df['active_energy'].max():.0f}  "
              f"mean: {df['active_energy'].mean():.0f} kcal")
        print(f"TSS labels     — min: {df['target_tss'].min():.1f}  "
              f"max: {df['target_tss'].max():.1f}  "
              f"mean: {df['target_tss'].mean():.1f}")

    # Global baseline predictions
    global_input = pd.DataFrame({
        "Age":                       USER_AGE,
        "gender_encoded":            GENDER_ENCODED,
        "bmi":                       df["bmi"],
        "Resting_BPM":               df["resting_hr_avg"],
        "Weight (kg)":               df["body_mass_avg"],
        "Session_Duration (hours)":  (df["active_energy"] / KCAL_PER_HOUR).clip(0, 1.5),
        "Calories_Burned":           df["active_energy"],
        "acwr":                      df["acwr"],
        "readiness":                 df["readiness"],
        "acute_load":                df["acute_load"],
        "chronic_load":              df["chronic_load"],
        "est_max_bpm":               EST_MAX_BPM,
    })[global_features]

    df["global_baseline"] = global_model.predict(global_input)

    # Train / test split + scaling
    X        = df[ADAPTER_FEATURES].values
    y        = df["target_tss"].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    split    = max(1, int(len(X_scaled) * 0.80))

    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # Temporal weighting
    today         = df["date"].max()
    days_ago      = (today - df["date"]).dt.days.values
    weights       = np.power(0.5, days_ago / HALF_LIFE_DAYS)
    weights_train = weights[:split].copy()

    if df["tss_source"].eq("rpe").any():
        mask = df["tss_source"].values[:split] == "rpe"
        weights_train[mask] *= 5.0
        if verbose:
            print(f"\nBoosted {mask.sum()} RPE-logged days (5× weight)")

    if df["tss_source"].eq("gpx").any():
        mask = df["tss_source"].values[:split] == "gpx"
        weights_train[mask] *= 2.0
        if verbose:
            print(f"Boosted {mask.sum()} GPX-measured days (2× weight)")

    adapter = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    adapter.fit(X_train, y_train, sample_weight=weights_train)

    preds        = adapter.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, df["global_baseline"].values[split:])
    adapter_mae  = mean_absolute_error(y_test, preds)
    gain         = (baseline_mae - adapter_mae) / baseline_mae * 100
    importances  = dict(zip(ADAPTER_FEATURES, adapter.feature_importances_))

    if verbose:
        print(f"\nGlobal baseline MAE:  {baseline_mae:.2f} TSS")
        print(f"Personal adapter MAE: {adapter_mae:.2f} TSS")
        print(f"Personalisation gain: {gain:.1f}%")
        print("\nFeature importances:")
        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat:<30} {imp:.3f}")

    joblib.dump(adapter,          "personal_adapter.pkl")
    joblib.dump(scaler,           "personal_scaler.pkl")
    joblib.dump(ADAPTER_FEATURES, "adapter_feature_names.pkl")
    joblib.dump({
        "gender":         USER_GENDER,
        "gender_encoded": GENDER_ENCODED,
        "est_max_bpm":    EST_MAX_BPM,
        "kcal_per_hour":  KCAL_PER_HOUR,
        "hrv_sex_adj":    HRV_SEX_ADJUSTMENT,
    }, "user_config.pkl")

    if verbose:
        print("\nSaved → personal_adapter.pkl, personal_scaler.pkl, user_config.pkl")

    return {
        "baseline_mae": baseline_mae,
        "adapter_mae":  adapter_mae,
        "gain_pct":     gain,
        "importances":  importances,
    }


if __name__ == "__main__":
    train_personal_adapter()
