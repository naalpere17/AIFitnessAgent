import os
import sys
from pathlib import Path

# Allow running with: python -m scripts.train_zenodo_frame_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TRAIN_CSV = Path("data/zenodo_train_features.csv")
TEST_CSV = Path("data/zenodo_test_features.csv")
MODEL_OUT = Path("outputs/zenodo_random_forest.joblib")

FEATURE_COLS = [
    "left_knee_angle",
    "right_knee_angle",
    "avg_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "left_torso_lean",
    "right_torso_lean",
    "avg_torso_lean",
    "depth_proxy",
    "avg_heel_foot_dist",
    "knee_angle_diff",
    "hip_angle_diff",
]


def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    train_df = pd.read_csv(TRAIN_CSV).dropna()
    test_df = pd.read_csv(TEST_CSV).dropna()

    print("Train shape:", train_df.shape)
    print("Test shape: ", test_df.shape)

    print("\nTrain label counts:")
    print(train_df["label"].value_counts())

    print("\nTest label counts:")
    print(test_df["label"].value_counts())

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Zenodo Frame Model Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()