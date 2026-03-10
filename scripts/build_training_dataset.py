import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

from form_analysis.pose_utils import get_pose_landmarks
from form_analysis.squat_metrics import compute_knee_angle

mp_pose = mp.solutions.pose

pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DATA_DIR = "data"
OUT_DIR = "data/features"
LABELS_FILE = os.path.join(DATA_DIR, "squat_labels.csv")

os.makedirs(OUT_DIR, exist_ok=True)

FRAME_STRIDE = 2


def load_labels():
    df = pd.read_csv(LABELS_FILE)
    return dict(zip(df["video"], df["label"]))


def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    seq = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_STRIDE != 0:
            frame_id += 1
            continue

        landmarks = get_pose_landmarks(frame, pose_model)

        if landmarks is None:
            frame_id += 1
            continue

        knee_angle = compute_knee_angle(landmarks)

        seq.append([knee_angle])

        frame_id += 1

    cap.release()

    return np.array(seq)


def main():

    labels = load_labels()

    videos = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".mp4")
    ]

    for video in videos:

        video_path = os.path.join(DATA_DIR, video)

        if video not in labels:
            print(f"Skipping unlabeled video: {video}")
            continue

        label = labels[video]

        print("Processing", video)

        seq = extract_features(video_path)

        if len(seq) < 5:
            print("Too short, skipping")
            continue

        out_path = os.path.join(
            OUT_DIR,
            f"features_{video.replace('.mp4','')}.npz"
        )

        np.savez(
            out_path,
            X=seq,
            y=label,
            video=video
        )

        print("Saved", out_path)


if __name__ == "__main__":
    main()