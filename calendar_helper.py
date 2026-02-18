from icalevents.icalevents import events
from datetime import datetime, timedelta
import pytz

def get_calendar_summary(ical_url, days_to_search=3, workout_duration_hrs=1):
    """
    Fetches free slots from Google Calendar and returns a formatted string 
    summary designed for an AI Agent to read and discuss with a user.
    """
    # 1. Setup Time Range (UTC for internal calculation)
    now = datetime.now(pytz.utc)
    end_search = now + timedelta(days=days_to_search)
    
    # 2. Fetch Busy Events
    try:
        # icalevents handles recurring events and timezones automatically
        busy_events = events(url=ical_url, start=now, end=end_search)
    except Exception as e:
        return f"Error: I couldn't access the calendar. Details: {e}"

    # 3. Sort and Normalize Busy Times
    busy_times = []
    for event in busy_events:
        start = event.start.astimezone(pytz.utc)
        end = event.end.astimezone(pytz.utc)
        busy_times.append((start, end))
    
    busy_times.sort()

    # 4. Identify Gaps (Free Slots)
    free_slots = []
    current_time = now
    workout_delta = timedelta(hours=workout_duration_hrs)
    local_tz = pytz.timezone('America/Los_Angeles')

    for busy_start, busy_end in busy_times:
        # Check if the gap is long enough for a workout
        if busy_start - current_time >= workout_delta:
            free_slots.append(current_time.astimezone(local_tz))
        
        # Advance current_time past the busy block
        if busy_end > current_time:
            current_time = busy_end

    # Check for a final gap after the last event
    if end_search - current_time >= workout_delta:
        free_slots.append(current_time.astimezone(local_tz))

    # 5. Format Output for the AI Agent
    if not free_slots:
        return "I checked the calendar, but there are no 1-hour gaps available in the next few days."

    summary = "Here are the available workout windows I found in your schedule:\n"
    for slot in free_slots[:5]: # Provide top 5 options to prevent context overflow
        summary += f"- {slot.strftime('%A, %b %d at %I:%M %p')}\n"
    
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
