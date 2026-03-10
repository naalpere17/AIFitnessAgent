import json
import argparse
import subprocess
import sys
from pathlib import Path


def generate_feedback(results: dict) -> tuple[str, list[str]]:
    """
    Decide whether squat form is good or not based on analysis metrics.
    Adjust these thresholds to match your project.
    """
    avg_depth_score = results.get("avg_depth_score")
    avg_min_knee_angle = results.get("avg_min_knee_angle")

    feedback = []
    good_form = True

    if avg_depth_score is None or avg_min_knee_angle is None:
        return "Unknown", ["Could not find the required squat metrics in the analysis output."]

    # Example thresholds — tune these based on your dataset/results
    if avg_depth_score < 0.65:
        good_form = False
        feedback.append(
            "Try to squat deeper. Your depth score suggests you may not be going low enough."
        )
    else:
        feedback.append("Your squat depth looks solid overall.")

    if avg_min_knee_angle > 95:
        good_form = False
        feedback.append(
            "Bend your knees a bit more during the squat to reach better depth and control."
        )
    elif avg_min_knee_angle < 60:
        feedback.append(
            "You are squatting quite deep. Make sure you keep control and maintain balance."
        )
    else:
        feedback.append("Your knee bend looks to be in a reasonable range.")

    label = "Good Form" if good_form else "Needs Improvement"
    return label, feedback


def run_analyzer(video_path: str, output_dir: str) -> None:
    """
    Run the squat analyzer module on a single video.
    """
    cmd = [
        sys.executable,
        "-m",
        "form_analysis.analyze_squat",
        "--video",
        video_path,
        "--output_dir",
        output_dir,
    ]

    print("Running analyzer...")
    print("Using Python interpreter:", sys.executable)

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Squat analysis failed.")


def write_report(
    video_path: str,
    results: dict,
    label: str,
    feedback: list[str],
    output_txt: Path,
) -> None:
    """
    Write the final form-check report to a text file.
    """
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Squat Form Check Report\n")
        f.write("=======================\n\n")
        f.write(f"Input video: {video_path}\n\n")

        f.write("Analysis Metrics\n")
        f.write("----------------\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

        f.write("\nOverall Result\n")
        f.write("--------------\n")
        f.write(f"{label}\n\n")

        f.write("Feedback\n")
        f.write("--------\n")
        for item in feedback:
            f.write(f"- {item}\n")

    print(f"Saved report to: {output_txt}")


def run_form_check(
    video_path: str,
    output_dir: str = "outputs/form_check",
    report_file: str = "outputs/form_check/squat_feedback.txt",
) -> None:
    """
    Callable function so main.py can run the squat form checker directly.
    """
    output_dir_path = Path(output_dir)
    report_file_path = Path(report_file)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    report_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: run analyzer
    run_analyzer(video_path, str(output_dir_path))

    # Step 2: load JSON produced by analyzer
    json_path = output_dir_path / "metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"No JSON analysis file was found at {json_path}. "
            "Check your analyze_squat script and output path."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Step 3: generate feedback
    label, feedback = generate_feedback(results)

    # Step 4: write final report
    write_report(video_path, results, label, feedback, report_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run squat form checker and save feedback to a text file."
    )
    parser.add_argument("video", help="Path to squat video file")
    parser.add_argument(
        "--output_dir",
        default="outputs/form_check",
        help="Directory where analysis outputs will be saved",
    )
    parser.add_argument(
        "--report_file",
        default="outputs/form_check/squat_feedback.txt",
        help="Path to output text report",
    )
    args = parser.parse_args()

    run_form_check(
        video_path=args.video,
        output_dir=args.output_dir,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    main()