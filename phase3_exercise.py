"""
Phase 3: Rep Counter + Form Feedback
------------------------------------
Counts squats or push-ups and checks each rep's form with the rule engines
in squat_rules.py and pushup_rules.py.

On screen:
  - rep counter: total reps and good reps (reps with no faults)
  - live warnings while you move (e.g. "Hips sagging")
  - the verdict on the last finished rep

USAGE
    Squats on the webcam:
        python phase3_exercise.py --exercise squat
    Push-ups from a recorded clip:
        python phase3_exercise.py --exercise pushup --video pushups.mp4
    Your calibrated thresholds, plus a CSV with one row per rep:
        python phase3_exercise.py --exercise squat --thresholds my_squat.json --reps-csv reps.csv

KEYS (in the video window)
    r      reset the counter
    l      toggle joint-name labels
    space  pause / resume
    q      quit

Film from the SIDE with the whole body in frame. For push-ups, put the camera
at floor height.
"""

import argparse
import csv
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from phase1_pose_overlay import MODEL_PATH, draw_pose, ensure_model
from pushup_rules import PushupAnalyzer, PushupThresholds
from squat_rules import SquatAnalyzer, SquatThresholds

EXERCISES = {
    "squat": (SquatAnalyzer, SquatThresholds),
    "pushup": (PushupAnalyzer, PushupThresholds),
}

GREEN, RED, AMBER, WHITE, GREY = (0, 200, 0), (0, 0, 255), (0, 190, 255), (255, 255, 255), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: rep counter with form checks")
    p.add_argument("--exercise", choices=EXERCISES, required=True)
    p.add_argument("--video", help="path to a video file (default: webcam)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--thresholds", help="JSON file with calibrated thresholds")
    p.add_argument("--reps-csv", help="write one row per finished rep to this CSV file")
    p.add_argument("--save-video", help="write the annotated video to this .mp4 file")
    p.add_argument("--no-display", action="store_true",
                   help="process without opening a window (for batch-processing clips)")
    return p.parse_args()


def text(frame, msg, x, y, colour=WHITE, scale=0.6, thick=2):
    cv2.putText(frame, msg, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, msg, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def draw_hud(frame, exercise, info, messages, fps):
    """Counter panel top-left, live warnings and last-rep verdict below it."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 215), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text(frame, f"{exercise.upper()}   {fps:4.1f} fps", 10, 22, GREY, 0.5, 1)
    text(frame, str(info["reps"]), 10, 85, WHITE, 2.2, 4)
    text(frame, "reps", 95, 60, GREY, 0.6, 1)
    text(frame, f"good: {info['good_reps']}", 95, 85, GREEN, 0.6, 2)

    if not info["tracking"]:
        text(frame, "Can't see body - film from the side", 10, 115, RED, 0.5, 1)
    else:
        state = info["state"]
        if exercise == "pushup" and not info.get("in_plank", True):
            state = "not in plank"
        text(frame, f"state: {state}   side: {info['side']}", 10, 115, GREY, 0.5, 1)
        metrics = "  ".join(f"{k} {v:.0f}" for k, v in info["metrics"].items() if v is not None)
        text(frame, metrics, 10, 137, GREY, 0.45, 1)

    y = 165
    last = info.get("last_rep")
    if last:
        if last["ok"]:
            text(frame, f"Rep {last['rep']}: good form", 10, y, GREEN, 0.55, 2)
        else:
            text(frame, f"Rep {last['rep']}: {', '.join(last['faults'])}", 10, y, RED, 0.55, 2)

    # Live warnings go along the bottom, where they are easy to read mid-rep.
    h = frame.shape[0]
    for i, fault in enumerate(info.get("live_faults", [])):
        text(frame, messages[fault], 10, h - 20 - 30 * i, AMBER, 0.7, 2)


def main():
    args = parse_args()
    ensure_model()

    analyzer_cls, thresholds_cls = EXERCISES[args.exercise]
    thresholds = thresholds_cls.from_json(args.thresholds) if args.thresholds else thresholds_cls()
    analyzer = analyzer_cls(thresholds)

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    from_file = args.video is not None
    cap = cv2.VideoCapture(args.video if from_file else args.camera)
    if not cap.isOpened():
        print(f"Error: could not open {'video ' + args.video if from_file else 'webcam'}.")
        return

    mirrored = not from_file
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    reps_file = reps_writer = None
    if args.reps_csv:
        reps_file = open(args.reps_csv, "w", newline="")
        reps_writer = None   # created on the first rep, once we know the columns

    video_out = None
    show_labels, paused = False, False
    frame_idx, start, last_tick, fps = 0, time.time(), time.time(), 0.0
    display = None
    info = analyzer._not_tracking(None)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if from_file:
                        break
                    continue

                h, w = frame.shape[:2]
                t_s = frame_idx / video_fps if from_file else time.time() - start

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t_s * 1000))

                display = cv2.flip(frame, 1) if mirrored else frame.copy()

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    info = analyzer.update(lms, w, h, t_s)
                    draw_pose(display, lms, show_labels=show_labels, mirrored=mirrored)
                else:
                    info = analyzer._not_tracking(analyzer.side)

                rep = info.get("finished_rep")
                if rep:
                    verdict = "good" if rep["ok"] else "faults: " + ", ".join(rep["faults"])
                    print(f"Rep {rep['rep']} at {rep['time_s']:.1f}s - {verdict}")
                    if reps_file:
                        if reps_writer is None:
                            reps_writer = csv.DictWriter(reps_file, fieldnames=list(rep))
                            reps_writer.writeheader()
                        reps_writer.writerow({**rep, "faults": ";".join(rep["faults"])})

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_tick, 1e-6))
                last_tick = now
                draw_hud(display, args.exercise, info, analyzer.MESSAGES, fps)

                if args.save_video:
                    if video_out is None:
                        video_out = cv2.VideoWriter(args.save_video,
                                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                                    video_fps, (w, h))
                    video_out.write(display)

                frame_idx += 1

            if args.no_display:
                continue

            cv2.imshow("Phase 3: Rep Counter  (r reset, space pause, l labels, q quit)", display)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_labels = not show_labels
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                analyzer = analyzer_cls(thresholds)
                print("Counter reset.")
    finally:
        landmarker.close()
        cap.release()
        if video_out:
            video_out.release()
        if reps_file:
            reps_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"\n{args.exercise}: {analyzer.reps} reps, {analyzer.good_reps} with good form")


if __name__ == "__main__":
    main()
