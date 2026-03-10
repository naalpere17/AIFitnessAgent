"""
log_workout.py
==============
CLI tool to log RPE and workout details after each session.
Converts RPE to real TSS using the Session RPE method (Foster et al. 2001):
  TSS = duration_min × RPE_score × 1.67 / 100 × 100
  (scaled to match our 0-150 TSS range)

Usage:
  python log_workout.py

Outputs: rpe_log.csv — one row per logged workout, used by recommend.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, date

LOG_FILE = "fitness_rec/rpe_log.csv"

# RPE → intensity factor mapping (Foster et al. 2001, modified)
# RPE 1-10 scale:
#   1-2  = very easy (recovery)
#   3-4  = easy (Zone 1-2)
#   5-6  = moderate (Zone 2-3)
#   7-8  = hard (Zone 3-4)
#   9-10 = maximal (Zone 4+)
RPE_LABELS = {
    1:  "Very easy — barely moving",
    2:  "Easy — comfortable, could go all day",
    3:  "Light — easy conversation",
    4:  "Moderate — can talk but noticing effort",
    5:  "Somewhat hard — short sentences only",
    6:  "Hard — focused breathing",
    7:  "Very hard — limited to a few words",
    8:  "Very very hard — pushing limits",
    9:  "Near maximal — nearly all out",
    10: "Maximal — absolute limit",
}


def rpe_to_tss(duration_min: float, rpe: int) -> float:
    """
    Session RPE method (Foster et al. 2001).
    Training Load = duration_min × RPE
    Converted to TSS scale (0-150):
      TSS = (Training Load / max_possible_load) × 150
      max_possible_load = 90 min × 10 RPE = 900
    This gives TSS proportional to both duration and intensity.
    """
    training_load = duration_min * rpe
    tss = (training_load / 900) * 150
    return float(np.clip(tss, 0, 150))


def get_input(prompt: str, valid_type, valid_range=None):
    """Keep asking until valid input is received."""
    while True:
        try:
            val = valid_type(input(prompt).strip())
            if valid_range and val not in valid_range:
                print(f"  Please enter a value between {min(valid_range)} and {max(valid_range)}")
                continue
            return val
        except ValueError:
            print(f"  Invalid input, please try again.")


def log_workout():
    print("\n" + "=" * 45)
    print("   LOG TODAY'S WORKOUT")
    print("=" * 45)

    # Date
    today_str = date.today().isoformat()
    date_input = input(f"\nDate [{today_str}]: ").strip()
    workout_date = date_input if date_input else today_str

    # Workout type
    print("\nWorkout type:")
    types = ["Run", "Walk", "Cycle", "Swim", "Strength", "HIIT", "Yoga/Stretch", "Other"]
    for i, t in enumerate(types, 1):
        print(f"  {i}. {t}")
    type_idx   = get_input("Select (1-8): ", int, range(1, 9))
    workout_type = types[type_idx - 1]

    # Duration
    duration_min = get_input("\nDuration (minutes): ", float)

    # RPE
    print("\nRPE — Rate your overall effort (1-10):")
    for rpe, label in RPE_LABELS.items():
        print(f"  {rpe:>2}  {label}")
    rpe = get_input("\nYour RPE (1-10): ", int, range(1, 11))

    # Computed TSS
    tss = rpe_to_tss(duration_min, rpe)

    # Optional notes
    notes = input("\nNotes (optional, press Enter to skip): ").strip()

    # Build record
    record = {
        "date":          workout_date,
        "logged_at":     datetime.now().isoformat(),
        "workout_type":  workout_type,
        "duration_min":  duration_min,
        "rpe":           rpe,
        "tss_rpe":       round(tss, 1),
        "notes":         notes,
    }

    # Append to log
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(LOG_FILE, index=False)

    print("\n" + "=" * 45)
    print(f"  Logged: {workout_type} on {workout_date}")
    print(f"  Duration:  {duration_min:.0f} min")
    print(f"  RPE:       {rpe}/10  — {RPE_LABELS[rpe]}")
    print(f"  TSS:       {tss:.1f}")
    print(f"  Saved → {LOG_FILE}")
    print("=" * 45 + "\n")

    # Show log summary
    print(f"Total logged workouts: {len(df)}")
    if len(df) >= 5:
        print(f"Recent TSS (last 5):   "
              f"{df['tss_rpe'].tail(5).round(1).tolist()}")
        print(f"Avg RPE (all time):    {df['rpe'].mean():.1f}")


if __name__ == "__main__":
    log_workout()
