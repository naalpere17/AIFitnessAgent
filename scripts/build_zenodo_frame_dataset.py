import os
import sys
from pathlib import Path

# Allow running with: python -m scripts.build_zenodo_frame_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from form_analysis.pose_utils import get_pose_landmarks

DATASET_ROOT = Path("data/Zenodo_Squat_Dataset")
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

OUT_TRAIN_CSV = Path("data/zenodo_train_features.csv")
OUT_TEST_CSV = Path("data/zenodo_test_features.csv")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

mp_pose = mp.solutions.pose


def point_xy(landmarks, idx):
    """
    landmarks[idx] is expected to be a tuple like (x, y, visibility)
    from your existing get_pose_landmarks() helper.
    """
    lm = landmarks[idx]
    return np.array([lm[0], lm[1]], dtype=np.float32)


def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def vector_angle_deg(v1, v2):
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def map_label(class_name: str):
    name = class_name.strip().lower()
    if name == "good":
        return 1
    if name in {"bad back", "bad heel"}:
        return 0
    return None


def extract_features_from_landmarks(landmarks):
    # MediaPipe Pose landmark indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_HEEL, R_HEEL = 29, 30
    L_FOOT, R_FOOT = 31, 32

    ls = point_xy(landmarks, L_SHOULDER)
    rs = point_xy(landmarks, R_SHOULDER)
    lh = point_xy(landmarks, L_HIP)
    rh = point_xy(landmarks, R_HIP)
    lk = point_xy(landmarks, L_KNEE)
    rk = point_xy(landmarks, R_KNEE)
    la = point_xy(landmarks, L_ANKLE)
    ra = point_xy(landmarks, R_ANKLE)
    lheel = point_xy(landmarks, L_HEEL)
    rheel = point_xy(landmarks, R_HEEL)
    lfoot = point_xy(landmarks, L_FOOT)
    rfoot = point_xy(landmarks, R_FOOT)

    left_knee_angle = angle_deg(lh, lk, la)
    right_knee_angle = angle_deg(rh, rk, ra)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

    left_hip_angle = angle_deg(ls, lh, lk)
    right_hip_angle = angle_deg(rs, rh, rk)
    avg_hip_angle = (left_hip_angle + right_hip_angle) / 2.0

    vertical = np.array([0.0, -1.0], dtype=np.float32)
    torso_vec_l = ls - lh
    torso_vec_r = rs - rh
    left_torso_lean = vector_angle_deg(torso_vec_l, vertical)
    right_torso_lean = vector_angle_deg(torso_vec_r, vertical)
    avg_torso_lean = (left_torso_lean + right_torso_lean) / 2.0

    # Simple squat depth proxy: hips lower relative to knees => deeper squat
    hip_y = (lh[1] + rh[1]) / 2.0
    knee_y = (lk[1] + rk[1]) / 2.0
    depth_proxy = float(knee_y - hip_y)

    # Simple foot / heel proxy
    left_heel_foot_dist = float(np.linalg.norm(lheel - lfoot))
    right_heel_foot_dist = float(np.linalg.norm(rheel - rfoot))
    avg_heel_foot_dist = (left_heel_foot_dist + right_heel_foot_dist) / 2.0

    # Left/right symmetry proxies
    knee_angle_diff = abs(left_knee_angle - right_knee_angle)
    hip_angle_diff = abs(left_hip_angle - right_hip_angle)

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "avg_knee_angle": avg_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "avg_hip_angle": avg_hip_angle,
        "left_torso_lean": left_torso_lean,
        "right_torso_lean": right_torso_lean,
        "avg_torso_lean": avg_torso_lean,
        "depth_proxy": depth_proxy,
        "avg_heel_foot_dist": avg_heel_foot_dist,
        "knee_angle_diff": knee_angle_diff,
        "hip_angle_diff": hip_angle_diff,
    }


def process_split(split_dir: Path, split_name: str, pose_model):
    rows = []
    skipped_no_pose = 0
    skipped_bad_image = 0

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {split_dir}")

    print(f"\nProcessing {split_name} split from: {split_dir}")

    for class_dir in sorted(class_dirs):
        label = map_label(class_dir.name)
        if label is None:
            print(f"Skipping unknown class folder: {class_dir.name}")
            continue

        image_paths = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        print(f"  {class_dir.name}: found {len(image_paths)} images")

        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                skipped_bad_image += 1
                continue

            landmarks = get_pose_landmarks(image, pose_model)
            if landmarks is None:
                skipped_no_pose += 1
                continue

            feats = extract_features_from_landmarks(landmarks)

            row = {
                "split": split_name,
                "class_name": class_dir.name,
                "label": label,
                "image_name": img_path.name,
                "image_path": str(img_path),
                **feats,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\nFinished {split_name}:")
    print(f"  usable samples: {len(df)}")
    print(f"  skipped (bad image): {skipped_bad_image}")
    print(f"  skipped (no pose found): {skipped_no_pose}")

    if len(df) > 0:
        print("\nClass counts:")
        print(df["class_name"].value_counts())

    return df


def main():
    pose_model = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    try:
        train_df = process_split(TRAIN_DIR, "train", pose_model)
        test_df = process_split(TEST_DIR, "test", pose_model)
    finally:
        pose_model.close()

    OUT_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_TRAIN_CSV, index=False)
    test_df.to_csv(OUT_TEST_CSV, index=False)

    print(f"\nSaved train features to: {OUT_TRAIN_CSV}")
    print(f"Saved test features to:  {OUT_TEST_CSV}")


if __name__ == "__main__":
    main()