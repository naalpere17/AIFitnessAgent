import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def get_pose_landmarks(frame, pose_model):
    """Return list of (x, y, visibility) in normalized coords, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks.landmark
    return [(lm.x, lm.y, lm.visibility) for lm in lms]


def calculate_angle(a, b, c):
    """Angle ABC in degrees. a,b,c are (x,y) or (x,y,vis)."""
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = float(np.abs(radians * 180.0 / np.pi))
    if angle > 180:
        angle = 360 - angle
    return angle