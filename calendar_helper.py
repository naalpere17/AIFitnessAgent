import urllib.parse
from icalevents.icalevents import events
from datetime import datetime, timedelta
import pytz

# --- CONFIGURATION ---
LOCAL_TZ = pytz.timezone('America/Los_Angeles')
BUFFER_MINUTES = 0  # Change this to 5, 10, or 30 depending on your preference

def get_calendar_summary(ical_url, days_to_search=3):
    """Generates 1-hour slots with strict PST normalization and transition buffers."""
    now = datetime.now(LOCAL_TZ)
    workout_duration = timedelta(hours=1)
    buffer_delta = timedelta(minutes=BUFFER_MINUTES)
    
    START_HOUR = 8 
    END_HOUR = 21

    try:
        start_search = (now - timedelta(hours=1)).astimezone(pytz.utc)
        end_search = (now + timedelta(days=days_to_search + 1)).astimezone(pytz.utc)
        busy_events = events(url=ical_url, start=start_search, end=end_search)
    except Exception as e:
        return f"Error accessing calendar: {e}"

    all_free_slots = []

    for i in range(days_to_search):
        current_day = (now + timedelta(days=i)).date()
        day_start = LOCAL_TZ.localize(datetime.combine(current_day, datetime.min.time().replace(hour=START_HOUR)))
        day_end = LOCAL_TZ.localize(datetime.combine(current_day, datetime.min.time().replace(hour=END_HOUR)))

        if i == 0 and now > day_start:
            next_hour_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            day_start = next_hour_start

        # Normalize busy events to PST
        day_busy = []
        for e in busy_events:
            e_start = e.start.astimezone(LOCAL_TZ)
            e_end = e.end.astimezone(LOCAL_TZ)
            if e_start < day_end and e_end > day_start:
                day_busy.append((e_start, e_end))
        
        # Check every hour
        current_slot_start = day_start
        while current_slot_start + workout_duration <= day_end:
            current_slot_end = current_slot_start + workout_duration
            
            # THE BUFFER CHECK:
            # We need to be free from (Start - Buffer) to (End + Buffer)
            required_start = current_slot_start - buffer_delta
            required_end = current_slot_end + buffer_delta
            
            is_free = True
            for b_start, b_end in day_busy:
                # If a busy event overlaps with our buffered window
                if required_start < b_end and required_end > b_start:
                    is_free = False
                    break
            
            if is_free:
                all_free_slots.append(current_slot_start)
            
            current_slot_start += timedelta(hours=1)

    if not all_free_slots:
        return "You're back-to-back! No buffered gaps found in the next few days."

    summary = f"Available 1-hour slots (with {BUFFER_MINUTES}m transition buffers):\n"
    for slot in all_free_slots[:15]:  
        summary += f"- {slot.strftime('%A, %b %d at %I:%M %p')} (ID: {slot.isoformat()})\n"
    
    return summary

def generate_add_to_calendar_link(iso_start, workout_focus):
    """Creates a URL for Google Calendar."""
    try:
        start_dt = datetime.fromisoformat(iso_start)
        end_dt = start_dt + timedelta(hours=1)
        fmt = "%Y%m%dT%H%M%SZ"
        s_str = start_dt.astimezone(pytz.utc).strftime(fmt)
        e_str = end_dt.astimezone(pytz.utc).strftime(fmt)
        
        params = {
            "action": "TEMPLATE",
            "text": f"🏋️ Workout: {workout_focus}",
            "dates": f"{s_str}/{e_str}",
            "details": f"Scheduled by AI Buddy (includes {BUFFER_MINUTES}m buffer).",
            "sf": "true", "output": "xml"
        }
        return "https://www.google.com/calendar/render?" + urllib.parse.urlencode(params)
    except Exception as e:
        return f"Link Error: {e}"