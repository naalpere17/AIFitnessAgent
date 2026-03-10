"""
parse_gpx.py
============
Parses a folder of Apple Health GPX files and extracts real workout metrics.
Outputs: workouts.csv — one row per workout, joinable to Apple Health data by date.

Metrics extracted:
  - date, start_time, duration_h
  - distance_km
  - elevation_gain_m, elevation_loss_m
  - avg_speed_ms, max_speed_ms
  - avg_pace_min_km
  - tss_estimated — pace-based TSS using Banister's impulse model adapted for running

TSS without HR is computed using pace-based intensity factor:
  IF = avg_pace_threshold / avg_pace_actual
  TSS = (duration_h × 3600 × IF²) / 3600 × 100
  Threshold pace is estimated from your age using Jack Daniels' VDOT tables.

Run: python parse_gpx.py --gpx_dir /path/to/gpx/folder
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------------
USER_AGE       = 22
USER_WEIGHT_KG = 70

# Estimated threshold pace (min/km) for TSS calculation
# Based on Jack Daniels VDOT for a recreational runner aged 22.
# Lower = faster. Adjust this to your actual threshold pace if known.
# 5:00 min/km ≈ 8:00 min/mile — moderate recreational fitness
THRESHOLD_PACE_MIN_KM = 5.0

# Elevation grade penalty: each 1% grade adds this much to effective pace
# Based on Minetti et al. (2002) energy cost of running on slopes
GRADE_PACE_PENALTY = 0.03   # 3% pace penalty per 1% average grade

# ---------------------------------------------------------------
# GPX NAMESPACE
# ---------------------------------------------------------------
NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two GPS points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_gpx_file(filepath: str) -> dict | None:
    """Parse a single GPX file and return a metrics dict."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  [skip] Could not parse: {filepath}")
        return None

    # Collect all track points
    points = []
    for trkpt in root.findall(".//gpx:trkpt", NS):
        try:
            lat  = float(trkpt.attrib["lat"])
            lon  = float(trkpt.attrib["lon"])
            ele  = float(trkpt.find("gpx:ele", NS).text) if trkpt.find("gpx:ele", NS) is not None else None
            time_str = trkpt.find("gpx:time", NS).text

            # Parse speed from extensions if available
            speed = None
            ext = trkpt.find("gpx:extensions", NS)
            if ext is not None:
                spd_el = ext.find("gpx:speed", NS)
                if spd_el is None:
                    # Try without namespace (some Apple exports omit it)
                    spd_el = ext.find("speed")
                if spd_el is not None:
                    speed = float(spd_el.text)

            # Parse UTC time
            t = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            points.append({"lat": lat, "lon": lon, "ele": ele,
                           "time": t, "speed": speed})
        except (KeyError, AttributeError, ValueError):
            continue

    if len(points) < 2:
        return None

    # Sort by time
    points.sort(key=lambda p: p["time"])

    # ---------------------------------------------------------------
    # COMPUTE METRICS
    # ---------------------------------------------------------------
    start_time   = points[0]["time"]
    end_time     = points[-1]["time"]
    duration_s   = (end_time - start_time).total_seconds()
    duration_h   = duration_s / 3600

    if duration_h < 0.05:   # skip spurious <3 min recordings
        return None

    # Distance and elevation
    total_dist_m   = 0.0
    elev_gain_m    = 0.0
    elev_loss_m    = 0.0
    speeds         = []

    for i in range(1, len(points)):
        p0, p1 = points[i-1], points[i]
        d = haversine(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
        total_dist_m += d

        if p0["ele"] is not None and p1["ele"] is not None:
            de = p1["ele"] - p0["ele"]
            if de > 0:
                elev_gain_m += de
            else:
                elev_loss_m += abs(de)

        if p1["speed"] is not None and p1["speed"] > 0:
            speeds.append(p1["speed"])

    dist_km      = total_dist_m / 1000
    avg_speed_ms = np.mean(speeds) if speeds else (total_dist_m / duration_s if duration_s > 0 else 0)
    max_speed_ms = np.max(speeds)  if speeds else avg_speed_ms

    # Pace in min/km
    avg_pace_min_km = (duration_s / 60) / dist_km if dist_km > 0 else None

    # ---------------------------------------------------------------
    # PACE-BASED TSS (no HR required)
    # Accounts for elevation using grade-adjusted pace
    # ---------------------------------------------------------------
    avg_grade_pct = (elev_gain_m / total_dist_m * 100) if total_dist_m > 0 else 0
    grade_penalty = avg_grade_pct * GRADE_PACE_PENALTY

    # Grade-adjusted pace (slower on hills = harder)
    adj_pace = avg_pace_min_km + grade_penalty if avg_pace_min_km else None

    if adj_pace and adj_pace > 0:
        # Intensity Factor: how hard relative to your threshold
        intensity_factor = THRESHOLD_PACE_MIN_KM / adj_pace
        # TSS = duration_h × IF² × 100
        tss = (duration_h * intensity_factor ** 2 * 100)
        tss = float(np.clip(tss, 0, 150))
    else:
        tss = None

    return {
        "date":              start_time.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        "start_time":        start_time.isoformat(),
        "duration_h":        round(duration_h, 3),
        "distance_km":       round(dist_km, 3),
        "elevation_gain_m":  round(elev_gain_m, 1),
        "elevation_loss_m":  round(elev_loss_m, 1),
        "avg_speed_ms":      round(avg_speed_ms, 3),
        "max_speed_ms":      round(max_speed_ms, 3),
        "avg_pace_min_km":   round(avg_pace_min_km, 2) if avg_pace_min_km else None,
        "avg_grade_pct":     round(avg_grade_pct, 2),
        "intensity_factor":  round(THRESHOLD_PACE_MIN_KM / (adj_pace + 1e-9), 3) if adj_pace else None,
        "tss_gpx":           round(tss, 1) if tss else None,
        "source_file":       os.path.basename(filepath),
    }


def parse_folder(gpx_dir: str) -> pd.DataFrame:
    files   = [f for f in os.listdir(gpx_dir) if f.lower().endswith(".gpx")]
    print(f"Found {len(files)} GPX files in {gpx_dir}")

    results = []
    for i, fname in enumerate(sorted(files), 1):
        path   = os.path.join(gpx_dir, fname)
        result = parse_gpx_file(path)
        if result:
            results.append(result)
        if i % 50 == 0:
            print(f"  Parsed {i}/{len(files)}...")

    df = pd.DataFrame(results)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nParsed {len(df)} valid workouts")
    if len(df) > 0:
        print(f"Date range:     {df['date'].min()} → {df['date'].max()}")
        print(f"Distance (km):  min {df['distance_km'].min():.1f}  "
              f"max {df['distance_km'].max():.1f}  "
              f"mean {df['distance_km'].mean():.1f}")
        print(f"Duration (h):   min {df['duration_h'].min():.2f}  "
              f"max {df['duration_h'].max():.2f}  "
              f"mean {df['duration_h'].mean():.2f}")
        if df['tss_gpx'].notna().any():
            print(f"TSS (gpx):      min {df['tss_gpx'].min():.1f}  "
                  f"max {df['tss_gpx'].max():.1f}  "
                  f"mean {df['tss_gpx'].mean():.1f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpx_dir",  required=True,
                        help="Path to folder containing GPX files")
    parser.add_argument("--out",      default="workouts.csv",
                        help="Output CSV path (default: workouts.csv)")
    args = parser.parse_args()

    df = parse_folder(args.gpx_dir)
    df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")