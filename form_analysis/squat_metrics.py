from form_analysis.pose_utils import calculate_angle

# Mediapipe landmark indexes
HIP = 23
KNEE = 25
ANKLE = 27


def compute_knee_angle(landmarks):
    hip = landmarks[HIP]
    knee = landmarks[KNEE]
    ankle = landmarks[ANKLE]

    return calculate_angle(hip, knee, ankle)