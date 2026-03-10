from exercise_detailer.recognition import identify_workout_machine
from exercise_detailer.rag import get_exercise_details
import os
import json

class Workout:
    def __init__(self, workout_name=None):
        self.workout_name = workout_name

    def set_exercise_details(self, file_path="exercise_detailer/available_exercises.json"):
        details = get_exercise_details(self.workout_name)
        details.pop("id")
        details.pop("images")
        self.muscle_groups = details['primaryMuscles'] + details['secondaryMuscles']
        self.rag_details = details
        
        # 1. Load existing data or initialize an empty list
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    if not isinstance(data, list):
                        data = [] 
            except json.JSONDecodeError:
                data = [] 
        else:
            data = []

        # 2. Remove any existing entry for this specific exercise to prevent duplicates
        current_name = self.rag_details.get('name')
        if current_name:
            data = [exercise for exercise in data if exercise.get('name') != current_name]

        # 3. Append the new exercise object to the list
        data.append(self.rag_details)

        # 4. Write the updated list back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

class Machine(Workout):
    def __init__(self, workout_name=None, image_path=None):
        super().__init__()
        if image_path:
            self.image_path = image_path
            if not self.workout_name:
                self.workout_name = identify_workout_machine(image_path)

    def set_exercise_details(self):
        super().set_exercise_details()
