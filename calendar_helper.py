from icalevents.icalevents import events
from datetime import datetime, timedelta
import pytz

LOCAL_TZ = pytz.timezone('America/Los_Angeles')

def get_calendar_summary(ical_url, days_to_search=3):
    """Fetches free slots by checking daily 'waking hour' windows."""
    now = datetime.now(LOCAL_TZ)
    workout_delta = timedelta(hours=1)

    # Define 'Waking Hours' for your buddy (e.g., 8 AM to 9 PM)
    START_HOUR = 8
    END_HOUR = 21

    try:
        # Fetching broader range in UTC for processing
        start_search = now.astimezone(pytz.utc)
        end_search = (now + timedelta(days=days_to_search)).astimezone(pytz.utc)
        busy_events = events(url=ical_url, start=start_search, end=end_search)
    except Exception as e:
        return f"Error accessing calendar: {e}"

    all_free_slots = []

    # Check each day individually
    for i in range(days_to_search):
        current_day = (now + timedelta(days=i)).date()

        # Define the window for THIS specific day
        day_start = LOCAL_TZ.localize(datetime.combine(current_day, datetime.min.time().replace(hour=START_HOUR)))
        day_end = LOCAL_TZ.localize(datetime.combine(current_day, datetime.min.time().replace(hour=END_HOUR)))

        # Adjust day_start if we are looking at 'today' and 8 AM has already passed
        if i == 0 and now > day_start:
            day_start = now

        # Filter busy events that happen during this day's window
        day_busy = []
        for e in busy_events:
            e_start = e.start.astimezone(LOCAL_TZ)
            e_end = e.end.astimezone(LOCAL_TZ)
            # If event overlaps with our 8am-9pm window
            if e_start < day_end and e_end > day_start:
                day_busy.append((max(e_start, day_start), min(e_end, day_end)))

        day_busy.sort()

        # Find gaps within the 8am-9pm window
        temp_start = day_start
        for b_start, b_end in day_busy:
            if b_start - temp_start >= workout_delta:
                all_free_slots.append(temp_start)
            temp_start = max(temp_start, b_end)

        # Check for gap after the last meeting of the day
        if day_end - temp_start >= workout_delta:
            all_free_slots.append(temp_start)

    if not all_free_slots:
        return "I checked, but your daytime hours look pretty packed for the next few days!"

    summary = "Here are some specific windows where we could fit in a workout:\n"
    # Take the first 6 slots to give the AI variety
    for slot in all_free_slots[:6]:
        summary += f"- {slot.strftime('%A, %b %d at %I:%M %p')} (ID: {slot.isoformat()})\n"

    return summary

# --- Local Testing Block ---
# This code only runs if you execute this file directly, not when the agent imports it.
if __name__ == "__main__":
    # Test with a placeholder URL
    TEST_URL = "https://calendar.google.com/calendar/ical/achang93%40ucsc.edu/private-8afb01038a9cfac469aafafe81ad8793/basic.ics"
    
    print("--- Testing Calendar Helper ---")
    if "PASTE_YOUR" in TEST_URL:
        print("Please provide a valid iCal URL in the TEST_URL variable to see real results.")
    else:
        result = get_calendar_summary(TEST_URL)
        print(result)
