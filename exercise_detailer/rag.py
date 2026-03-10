import json
import re
import ollama

def normalize_text(text):
    """Strips everything except alphanumeric characters and converts to lowercase."""
    if not text:
        return ""
    return re.sub(r'[^a-z]', '', text.lower())

def find_best_exercise_match(ai_output, exercises_list):
    # 1. Normalize the AI's output (e.g., "Push-Ups" -> "pushups")
    normalized_ai_key = normalize_text(ai_output)
    
    if not normalized_ai_key:
        return None

    # PASS 1: The Exact Normalized Match
    # We do this first so "Running, Treadmill" perfectly matches "Running, Treadmill"
    for exercise in exercises_list:
        db_key = exercise.get('name', '')
        if normalize_text(db_key) == normalized_ai_key:
            return exercise

    # PASS 2: The Comma-Split Fallback
    # If the exact match fails, we check the comma-separated chunks
    for exercise in exercises_list:
        db_key = exercise.get('name', '')
        
        # Split the string: "running, treadmill" -> ["running", " treadmill"]
        comma_parts = db_key.split(',')
        
        for part in comma_parts:
            # Normalize just that chunk (e.g., " treadmill" -> "treadmill")
            if normalize_text(part) == normalized_ai_key:
                print(f"Fallback triggered: Matched AI '{ai_output}' to DB '{db_key}'")
                return exercise

    # If both passes fail
    print(f"Failed to find any match for '{ai_output}'.")
    return None

def get_exercise_details(user_input, db_path='exercises.json', model_name='gemma3:27b'):
    try:
        with open(db_path, 'r',) as file:
            exercises = json.load(file)
    except FileNotFoundError:
        return "Error: Could not find exercises.json. Make sure the file is in your directory!"

    valid_keys = [exercise.get('name') for exercise in exercises if exercise.get('name')]

    prompt = f"""You are a data routing assistant. Your job is to match a user's messy or informal exercise name to the exact corresponding key in my database.

User's input: "{user_input}"

Database keys: 
{valid_keys}

Find the single key from the list that best matches the user's input semantically. 
Respond ONLY with a JSON object containing the exact matched string under the key "matched_key".
If there is absolutely no reasonable match, return an empty string for the value.

CRITICAL INSTRUCTION: You must return the EXACT string from the provided list, character-for-character. 
Do NOT correct spelling, capitalization, or punctuation. 
If the list says "Pushups", you must return "Pushups", NOT "Push-Ups"."""

    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        format='json' # This is crucial: it prevents the LLM from adding conversational text
    )

    # 5. Parse the LLM's JSON response
    try:
        result_json = json.loads(response['message']['content'])
        matched_key = result_json.get('matched_key', '')
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        return None

    final_exercise_data = find_best_exercise_match(matched_key, exercises)
    matched_key = final_exercise_data['name']

    if not matched_key:
        print(f"No match found in the database for '{user_input}'.")
        return None

    if matched_key in valid_keys:
        print(f"Success: The LLM matched '{user_input}' to the exact key '{matched_key}'\n")
    else:
        print(f"Couldn't find a match for '{user_input}'.\n")

    return final_exercise_data