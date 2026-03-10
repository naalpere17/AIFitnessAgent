from exercise_detailer.equipment import Machine, Workout
import json

from schedule.fitness_agent import FitnessAgent
from schedule.calendar_helper import CalendarHelper
from check_squat_form import run_form_check

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

# Save user data to JSON 

exercise1 = Machine(image_path="exercise_detailer/images.webp")
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
while True:
    user_input = input("You: ")
        
    # Exit condition
    if user_input.lower() in ["exit", "quit"]: 
        print("Goodbye")
        break
            
    # 3. Pass the input to generate_response
    response = agent_scheduler.generate_response(user_input) 
    print(f"\nAgent: {response}\n")
## SCHEDULE AGENT ENDS HERE ###