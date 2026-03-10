"""
config.py
=========
Shared user profile constants. These are set at runtime by main.py
based on user input before any other module imports them.
Edit defaults here, or override via main.py at startup.
"""

# Defaults — overwritten by main.py before any module uses them
USER_AGE       = 22
USER_HEIGHT_M  = 1.75
USER_WEIGHT_KG = 70
USER_GENDER    = "male"   # "male" or "female"

# These are derived from the above — call recalculate() after changing them
EST_MAX_BPM        = 192
GENDER_ENCODED     = 0
KCAL_PER_HOUR      = 420
HRV_SEX_ADJUSTMENT = 1.0

HALF_LIFE_DAYS = 60

# File paths
HEALTH_CSV    = "fitness_rec/health_last_60_days.csv"
SYNTHETIC_CSV = "fitness_rec/synthetic_fitness_dataset.csv"
WORKOUTS_CSV  = "fitness_rec/workouts.csv"
RPE_CSV       = "fitness_rec/rpe_log.csv"


def recalculate():
    """
    Recompute all derived constants after USER_AGE, USER_GENDER etc. are set.
    Call this once from main.py after updating the profile variables.
    """
    global EST_MAX_BPM, GENDER_ENCODED, KCAL_PER_HOUR, HRV_SEX_ADJUSTMENT

    if USER_GENDER == "female":
        EST_MAX_BPM    = int(206 - (0.88 * USER_AGE))
        GENDER_ENCODED = 1
        KCAL_PER_HOUR  = 400
        HRV_SEX_ADJUSTMENT = 0.88
    else:
        EST_MAX_BPM    = int(208 - (0.7 * USER_AGE))
        GENDER_ENCODED = 0
        KCAL_PER_HOUR  = 420
        HRV_SEX_ADJUSTMENT = 1.0
