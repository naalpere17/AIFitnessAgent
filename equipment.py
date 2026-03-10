import recognition
from rag import get_exercise_details

class Workout:
    def __init__(self, workout_name=None):
        self.workout_name = workout_name

    def set_exercise_details(self):
        details = get_exercise_details(self.workout_name)
        details.pop("id")
        details.pop("images")
        self.muscle_groups = details['primaryMuscles'] + details['secondaryMuscles']
        self.rag_details = details

class Machine(Workout):
    def __init__(self, workout_name=None, image_path=None):
        super().__init__()
        if image_path:
            self.image_path = image_path
            if not self.workout_name:
                self.workout_name = recognition.identify_workout_machine(image_path)

    def set_exercise_details(self):
        super().set_exercise_details()