from exercise_detailer.equipment import Machine, Workout
import json
import os
import ollama

from schedule.fitness_agent import FitnessAgent
from check_squat_form import run_form_check
from schedule.calendar_helper import get_calendar_summary, generate_add_to_calendar_link
from fitness_rec.train import train_global_model
from fitness_rec.recommend import train_personal_adapter
from fitness_rec.predict import get_recommendation
from fitness_rec.log_workout import log_workout
import fitness_rec.config as config 

use_old_prompt = False
if os.path.exists("previous_prompt.txt"):
    res = input("Would you like to use your fitness data from the last time you ran the code (instead of generating it again)? (yes or no, default 'no'): ")
    use_old_prompt = res == "yes"

if not use_old_prompt:
    # Default Values
    age_default = 22
    height_default = 1.72
    weight_default = 72
    gender_default = "male"
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

    gender = input("What is your gender (male or female)")

    if gender == "":
        gender = gender_default

    demographics = f"""Age: {age}
    Height: {height}
    Weight: {weight}
    Gender: {gender}"""

    cal_link = input("Input your google calendar link: ")

    if cal_link ==  "":
        cal_link = cal_link_default

    config.USER_AGE       = age
    config.USER_HEIGHT_M  = height
    config.USER_WEIGHT_KG = weight
    config.USER_GENDER    = gender
    config.recalculate()   # recomputes EST_MAX_BPM, GENDER_ENCODED, etc.
    profile = f"Profile: — age: {age}  height: {height}m  weight: {weight}kg  gender: {gender}  HRmax: {config.EST_MAX_BPM} bpm"
    print(profile)

    exercise1 = Machine(image_path="exercise_detailer/bike.webp")
    exercise1.set_exercise_details()
    exercise2 = Workout(workout_name="Russia twist")
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

    print("\n[1/2] Training personal adapter...")
    recommend_result = train_personal_adapter(verbose=True)
    print(f"      Personalisation gain: {recommend_result['gain_pct']:.1f}%\n")
    print("\n[2/2] Getting today's recommendation...")
    recommendation = get_recommendation(verbose=True)
    ## Recommendation Engine ENDS HERE ###

    ## User Survey After Work Out
    wo = input("Did you just complete a workout?: yes or no: ")
    if wo == "":
        wo = "yes"
    if wo == "yes":
        log_workout()

    WORKOUT_INFO = "exercise_detailer/available_exercises.json"

    with open(WORKOUT_INFO, 'r', encoding='utf-8') as file:
        data = json.load(file) # Parses into a Python dictionary/list
    workout_context_string = json.dumps(data)

    with open("availability_summary.txt", 'r') as file:
        availability = file.read()

    if os.path.exists("outputs/form_check/squat_feedback.txt"):
        with open("outputs/form_check/squat_feedback.txt", 'r') as file:
            squat_form = file.read()
    else:
        squat_form = "No squat form video provided, ignore this section.\n"

    prompt = f"""You are a helpful workout assistant model. Take iniative with continuing the conversation and motivating the user.
    Your goal is to provide useful information to the user when they ask questions and to provide positive encourgament for them to reach good fitness goals.
    Your knowledge (provided by manually entered data and information from other AI models) about the user, their schedule, and the workouts you are able to recommend are as follows:
    {profile}\n
    Available exercises including details on how to perform exercises and muscle groups worked (in JSON format):
    {workout_context_string}\n
    {availability}
    Design a workout for {recommendation['zone']}.
    My Rediness is {recommendation['readiness']}/1.0 and my ACWR is {recommendation['acwr']}.
    I have {recommendation['remaining_tss']} TSS remaining to spend today.\n
    Details on squat form from provided a provided video:
    {squat_form}
    Initate the conversation by providing a very brief summary about the important things you know about the user's current condition and workout options.
    Then provide a brief recommendation for a workout to the user and ask what they would like to do."""

    with open("previous_prompt.txt", 'w') as file:
        file.write(prompt)

else:
    with open("previous_prompt.txt", 'r') as file:
        prompt = file.read()

MODEL_NAME = "gpt-oss:20b"
messages = [
    {"role": "user", "content": prompt}
]

print("Initializing Workout Assistant... (Waiting for initial response)\n")
print("-" * 50)
try:
    response = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
    assistant_reply = ""
    print("Assistant: ", end="", flush=True)
    
    # Iterate over the chunks as they are generated
    for chunk in response:
        content = chunk['message']['content']
        print(content, end="", flush=True)  # Print chunk immediately
        assistant_reply += content          # Accumulate the full response
        
    print("\n")
    messages.append({"role": "assistant", "content": assistant_reply})

    while True:
        user_input = input("You: ")
        
        # Exit condition
        if user_input.strip().lower() == 'bye':
            print("\nAssistant: Goodbye! Have a great workout! Let me know when you're ready for the next one.")
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        assistant_reply = ""
        print("\nAssistant: ", end="", flush=True)
        
        for chunk in response:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            assistant_reply += content
            
        print("\n")
        messages.append({"role": "assistant", "content": assistant_reply})

    if input("Log this workout? (yes or no): ").strip().lower().startswith("yes"):
        log_workout()

except ollama.ResponseError as e:
    print(f"\nError connecting to Ollama: {e}")
    print(f"Make sure Ollama is running and you have the '{MODEL_NAME}' model installed.")
except KeyboardInterrupt:
    print("\n\nSession terminated by user. Goodbye!")