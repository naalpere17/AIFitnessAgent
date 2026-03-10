from exercise_detailer.equipment import Machine, Workout
import json

from schedule.fitness_agent import FitnessAgent
from check_squat_form import run_form_check
from schedule.calendar_helper import get_calendar_summary, generate_add_to_calendar_link
from fitness_rec.train import train_global_model
from fitness_rec.recommend import train_personal_adapter
from fitness_rec.predict import get_recommendation
import fitness_rec.config 


# Default Values
age_default = 22
height_default = 1.72
weight_default = 72
cal_link_default = "https://calendar.google.com/calendar/ical/achang93%40ucsc.edu/private-8afb01038a9cfac469aafafe81ad8793/basic.ics"
age = input("What is your age: ")

if age ==  "":
    age = age_default

height = input("What is your height (m): ")

if height ==  "":
    height = height_default

weight = input("What is your weight (kg): ")

if weight ==  "":
    weight = weight_default

cal_link = input("Input your google calendar link: ")

if cal_link ==  "":
    cal_link = cal_link_default

config.USER_AGE       = age
config.USER_HEIGHT_M  = height
config.USER_WEIGHT_KG = weight
config.USER_GENDER    = gender
config.recalculate()   # recomputes EST_MAX_BPM, GENDER_ENCODED, etc.
print(f"\nProfile set — age: {age}  height: {height}m  "
f"weight: {weight}kg  gender: {gender}  HRmax: {config.EST_MAX_BPM} bpm")

# Save user data to JSON 

exercise1 = Machine(image_path="exercise_detailer/bike.webp")
exercise1.set_exercise_details()
exercise2 = Workout(workout_name="Russian Twist")
exercise2.set_exercise_details()
# RAG data now stored at exercise_detailer/available_exercises.json

print("\n--- Squat Form Checker ---")
video_path = input("Enter path to your squat video (or press Enter to skip): ")

if video_path != "":
    print("\nRunning squat form analysis...")
    run_form_check(video_path)
    print("\nYour squat feedback has been saved to:")
    print("outputs/form_check/squat_feedback.txt\n")

## SCHEDULE AGENT STARTS HERE ###
# Initialize the schedule agent and calendar helper
agent_scheduler = FitnessAgent(model_id="openai/gpt-oss-20b")
print("When do you want to work out?")
user_input = input("You: ")
response = agent_scheduler.generate_response(user_input) 
    # print(f"\nAgent: {response}\n")
## SCHEDULE AGENT ENDS HERE ###

## Recomendation Engine STARTS HERE ###

print("\n[1/3] Training global model...")
train_result = train_global_model(verbose=True)
print(f"      Global MAE: {train_result['mae']:.2f} TSS\n")
print("\n[2/3] Training personal adapter...")
recommend_result = train_personal_adapter(verbose=True)
print(f"      Personalisation gain: {recommend_result['gain_pct']:.1f}%\n")
print("\n[3/3] Getting today's recommendation...")
prediction = get_recommendation(verbose=True)

## Recommendation Engine ENDS HERE ###