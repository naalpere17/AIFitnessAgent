import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from form_analysis.pose_utils import get_pose_landmarks
from form_analysis.squat_metrics import compute_knee_angle


# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data"
OUT_DIR = os.path.join("outputs", "squat_day1")
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
DEFAULT_SINGLE_VIDEO = os.path.join(DATA_DIR, "side_angle_squat.mp4")

# Process fewer frames to speed up (set to 1 for every frame)
FRAME_STRIDE = 2

# Visibility threshold for keypoints (0..1). Lower = keep more frames.
VIS_THRESH = 0.50


def analyze_video(video_path: str, pose_model) -> dict:
    """
    Analyze one video, return dict with:
      - frames_total
      - frames_used
      - min_knee_angle
      - avg_knee_angle
      - score
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "video": os.path.basename(video_path),
            "error": "could_not_open",
            "frames_total": 0,
            "frames_used": 0,
            "min_knee_angle": None,
            "avg_knee_angle": None,
            "score": 0.0,
        }

    angles = []
    frames_total = 0
    frames_used = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_total += 1
        if frames_total % FRAME_STRIDE != 0:
            continue

        landmarks = get_pose_landmarks(frame, pose_model)  # expects (x,y,visibility) tuples or None
        if landmarks is None:
            continue

        # Optional: require that hip/knee/ankle visibilities are decent
        # Mediapipe indices: hip 23, knee 25, ankle 27 for left side (you can adjust later)
        hip_vis = landmarks[23][2]
        knee_vis = landmarks[25][2]
        ankle_vis = landmarks[27][2]
        if min(hip_vis, knee_vis, ankle_vis) < VIS_THRESH:
            continue

        angle = compute_knee_angle(landmarks)
        if angle is None or np.isnan(angle):
            continue

        angles.append(float(angle))
        frames_used += 1

    cap.release()

    score = squat_score(angles)

    if len(angles) == 0:
        min_angle = None
        avg_angle = None
    else:
        min_angle = float(np.min(angles))
        avg_angle = float(np.mean(angles))

    return {
        "video": os.path.basename(video_path),
        "frames_total": int(frames_total),
        "frames_used": int(frames_used),
        "min_knee_angle": min_angle,
        "avg_knee_angle": avg_angle,
        "score": float(score),
    }


def squat_score(angles):
    """
    Simple depth-based score based on minimum knee angle.
    Lower angle => deeper squat => higher score.

    Score clamps to [0,1].
    - If depth <= 60°, score ~ 1.0 (very deep)
    - If depth >= 120°, score ~ 0.0 (shallow)
    """
    if not angles:
        return 0.0

    depth = min(angles)
    score = max(0.0, min(1.0, (120.0 - depth) / 60.0))
    return score


def list_videos_in_data(data_dir: str):
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
    return sorted(files)


def main():
    # Decide what to analyze
    videos = list_videos_in_data(DATA_DIR)
    if len(videos) == 0:
        # Fall back to default path (will likely error with helpful message)
        videos = [DEFAULT_SINGLE_VIDEO]

    mp_pose = mp.solutions.pose

    results = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_model:
        for vp in videos:
            print(f"Analyzing: {vp}")
            res = analyze_video(vp, pose_model)
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "per_video.csv"), index=False)

    # Aggregate metric you can report today
    valid = df[df["min_knee_angle"].notna()].copy()
    if len(valid) > 0:
        avg_score = float(valid["score"].mean())
        avg_min_angle = float(valid["min_knee_angle"].mean())
        num_videos_ok = int(len(valid))
    else:
        avg_score = 0.0
        avg_min_angle = None
        num_videos_ok = 0

    metrics = {
        "exercise": "squat",
        "metric_today": "avg_depth_score_over_videos",
        "num_videos_total": int(len(df)),
        "num_videos_ok": num_videos_ok,
        "avg_depth_score": avg_score,
        "avg_min_knee_angle": avg_min_angle,
        "frame_stride": FRAME_STRIDE,
        "visibility_threshold": VIS_THRESH,
        "output_files": {
            "per_video_csv": os.path.join(OUT_DIR, "per_video.csv"),
            "metrics_json": os.path.join(OUT_DIR, "metrics.json"),
        },
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved outputs to:", OUT_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()