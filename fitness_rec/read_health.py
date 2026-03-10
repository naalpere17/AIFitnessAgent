import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

# ===== CONFIG =====
EXPORT_PATH = "apple_health_export/export.xml"   # put xml in same folder or change path
OUTPUT_PATH = "health_last_60_days.csv"
DAYS_BACK = 500

# ===== DATE WINDOW =====
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS_BACK)

# ===== APPLE HEALTH IDENTIFIERS =====
TYPES = {
    "steps": "HKQuantityTypeIdentifierStepCount",
    "resting_hr": "HKQuantityTypeIdentifierRestingHeartRate",
    "hrv": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "active_energy": "HKQuantityTypeIdentifierActiveEnergyBurned",
    "body_mass": "HKQuantityTypeIdentifierBodyMass",
    "sleep": "HKCategoryTypeIdentifierSleepAnalysis"
}

print("Parsing XML (this may take a minute)...")
tree = ET.parse(EXPORT_PATH)
root = tree.getroot()

data = defaultdict(lambda: {
    "steps": 0,
    "resting_hr": [],
    "hrv": [],
    "active_energy": 0,
    "body_mass": [],
    "sleep_seconds": 0
})

def parse_date(date_str):
    return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")

for record in root.findall("Record"):
    record_type = record.attrib.get("type")
    start = record.attrib.get("startDate")
    end = record.attrib.get("endDate")
    value = record.attrib.get("value")

    if not start:
        continue

    start_dt = parse_date(start)

    if not (start_date <= start_dt <= end_date):
        continue

    day = start_dt.strftime("%Y-%m-%d")

    if record_type == TYPES["steps"]:
        data[day]["steps"] += float(value)

    elif record_type == TYPES["resting_hr"]:
        data[day]["resting_hr"].append(float(value))

    elif record_type == TYPES["hrv"]:
        data[day]["hrv"].append(float(value))

    elif record_type == TYPES["active_energy"]:
        data[day]["active_energy"] += float(value)

    elif record_type == TYPES["body_mass"]:
        data[day]["body_mass"].append(float(value))

    elif record_type == TYPES["sleep"] and start and end:
        end_dt = parse_date(end)
        data[day]["sleep_seconds"] += (end_dt - start_dt).total_seconds()

# ===== BUILD DATAFRAME =====
rows = []

for day, metrics in data.items():
    rows.append({
        "date": day,
        "steps": metrics["steps"],
        "resting_hr_avg": (
            sum(metrics["resting_hr"]) / len(metrics["resting_hr"])
            if metrics["resting_hr"] else None
        ),
        "hrv_avg": (
            sum(metrics["hrv"]) / len(metrics["hrv"])
            if metrics["hrv"] else None
        ),
        "active_energy": metrics["active_energy"],
        "sleep_hours": metrics["sleep_seconds"] / 3600,
        "body_mass_avg": (
            sum(metrics["body_mass"]) / len(metrics["body_mass"])
            if metrics["body_mass"] else None
        )
    })

df = pd.DataFrame(rows)
df = df.sort_values("date")
df.to_csv(OUTPUT_PATH, index=False)

print(f"Done. Saved last {DAYS_BACK} days to {OUTPUT_PATH}")
print(df.tail())