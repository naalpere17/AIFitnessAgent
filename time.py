from icalevents.icalevents import events
from datetime import datetime, timedelta
import pytz

def get_free_slots(ical_url, days_to_search=7, workout_duration_hrs=1):
    # 1. Setup Time Range
    now = datetime.now(pytz.utc)
    end_search = now + timedelta(days=days_to_search)
    
    # 2. Fetch Busy Events
    # icalevents handles recurring events automatically
    try:
        busy_events = events(url=ical_url, start=now, end=end_search)
    except Exception as e:
        return f"Error fetching calendar: {e}"

    # 3. Sort events by start time
    busy_times = []
    for event in busy_events:
        # Normalize all times to UTC for comparison
        start = event.start.astimezone(pytz.utc)
        end = event.end.astimezone(pytz.utc)
        busy_times.append((start, end))
    
    busy_times.sort()

    # 4. Find the Gaps
    free_slots = []
    current_time = now
    workout_delta = timedelta(hours=workout_duration_hrs)

    for busy_start, busy_end in busy_times:
        # Check if the gap between 'current_time' and the next 'busy_start' is long enough
        if busy_start - current_time >= workout_delta:
            free_slots.append({
                "start": current_time,
                "end": busy_start
            })
        
        # Move current_time to the end of this busy block
        if busy_end > current_time:
            current_time = busy_end

    # Check for a final gap after the last event of the search period
    if end_search - current_time >= workout_delta:
        free_slots.append({
            "start": current_time,
            "end": end_search
        })

    return free_slots

# --- Example Usage ---
MY_ICAL_URL = "ADD YOU CALENDER LINK" # Get this from Calendar Settings -> Integrate Calendar
available_slots = get_free_slots(MY_ICAL_URL)

print(f"Found {len(available_slots)} potential workout windows:")
for slot in available_slots:
    local_start = slot['start'].astimezone(pytz.timezone('America/Los_Angeles'))
    print(f"🟢 Free from {local_start.strftime('%a %I:%M %p')} onwards")