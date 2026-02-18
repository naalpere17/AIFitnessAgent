from recognition import identify_workout_machine

class Workout:
    def __init__(self):
        self.workout_name = None

class Machine(Workout):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.workout_name = identify_workout_machine(image_path)