"""
config.py
=========
Shared user profile constants used across all modules.
Edit only this file to update your profile — train, recommend, and predict
all import from here so nothing gets out of sync.
"""

USER_AGE       = 22
USER_HEIGHT_M  = 1.75
USER_WEIGHT_KG = 70
USER_GENDER    = "male"   # "male" or "female"

# Gender-adjusted HRmax (Tanaka et al. 2001)
if USER_GENDER == "female":
    EST_MAX_BPM = int(206 - (0.88 * USER_AGE))
else:
    EST_MAX_BPM = int(208 - (0.7 * USER_AGE))

GENDER_ENCODED     = 1 if USER_GENDER == "female" else 0
KCAL_PER_HOUR      = 400 if USER_GENDER == "female" else 420
HRV_SEX_ADJUSTMENT = 0.88 if USER_GENDER == "female" else 1.0

# Temporal weighting half-life in days.
# 60 = balanced (recommended until you have 30+ RPE logs)
# 30 = recency-focused (better once you have dense recent data)
HALF_LIFE_DAYS = 60

# File paths
HEALTH_CSV     = "output/health_last_60_days.csv"
SYNTHETIC_CSV  = "synthetic_fitness_dataset.csv"
WORKOUTS_CSV   = "workouts.csv"
RPE_CSV        = "rpe_log.csv"