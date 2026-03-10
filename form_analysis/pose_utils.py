import cv2
import mediapipe as mp
import numpy as np

# Note: mp.solutions.pose is removed. 
# We now rely on the MediaPipe Tasks API for inference.

def _xy(point):
    """
    Return (x, y) from either:
    - a tuple/list like (x, y) or (x, y, visibility)
    - a MediaPipe landmark object with .x and .y
    """
    if hasattr(point, "x") and hasattr(point, "y"):
        return np.array([point.x, point.y], dtype=np.float32)

    return np.array([point[0], point[1]], dtype=np.float32)


def get_pose_landmarks(frame, pose_landmarker):
    """
    Return list of (x, y, visibility) in normalized coords, or None.

    This keeps the output format compatible with your existing video pipeline,
    updated for the MediaPipe Tasks API.
    """
    # 1. Convert the BGR frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Convert the NumPy array to a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # 3. Process the image using detect() instead of process()
    detection_result = pose_landmarker.detect(mp_image)

    # 4. Check if any poses were detected (Tasks API returns a list of people)
    if not detection_result.pose_landmarks:
        return None

    # 5. Grab the first person detected to match legacy single-person output
    lms = detection_result.pose_landmarks[0]
    
    # The new API includes visibility directly on the landmark objects
    return [(lm.x, lm.y, getattr(lm, 'visibility', 0.0)) for lm in lms]


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