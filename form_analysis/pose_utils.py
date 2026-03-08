import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def _xy(point):
    """
    Return (x, y) from either:
    - a tuple/list like (x, y) or (x, y, visibility)
    - a MediaPipe landmark object with .x and .y
    """
    if hasattr(point, "x") and hasattr(point, "y"):
        return np.array([point.x, point.y], dtype=np.float32)

    return np.array([point[0], point[1]], dtype=np.float32)


def get_pose_landmarks(frame, pose_model):
    """
    Return list of (x, y, visibility) in normalized coords, or None.

    This keeps the output format compatible with your existing video pipeline.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks.landmark
    return [(lm.x, lm.y, lm.visibility) for lm in lms]


def get_landmark_xy(landmarks, idx):
    """
    Return a landmark's (x, y) as a NumPy array from a landmark list.
    Works whether the landmark is a tuple or MediaPipe object.
    """
    return _xy(landmarks[idx])


def calculate_angle(a, b, c):
    """
    Angle ABC in degrees.
    Accepts:
    - tuples/lists like (x, y) or (x, y, visibility)
    - MediaPipe landmark objects
    """
    a = _xy(a)
    b = _xy(b)
    c = _xy(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = float(np.abs(radians * 180.0 / np.pi))

    if angle > 180:
        angle = 360 - angle

    return angle