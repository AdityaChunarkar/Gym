"""
Phase 2: Live Joint Angles
--------------------------
Builds on Phase 1. For every frame it calculates the key joint angles
(knees, hips, elbows, shoulders, torso lean), shows them on screen next
to each joint and in a side panel, and can log them to a CSV file.

The CSV is what you will use to calibrate rules in Phase 3: record a few
good and bad squats, then compare their angles.

USAGE
    Webcam:
        python phase2_angles.py
    Webcam + save angles to CSV:
        python phase2_angles.py --csv my_squats.csv
    A recorded video (for your dataset clips):
        python phase2_angles.py --video clip.mp4 --csv clip_angles.csv
    Also save the annotated video (useful for your demo):
        python phase2_angles.py --video clip.mp4 --save-video annotated.mp4

KEYS (in the video window)
    l      toggle joint-name labels
    space  pause / resume
    q      quit

TIP: for squats and deadlifts, film from the SIDE, full body in frame.
"""

import argparse
import csv
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from angles import ALL_ANGLE_NAMES, JOINT_ANGLES, L_HIP, R_HIP, AngleSmoother, compute_angles
from phase1_pose_overlay import MODEL_PATH, draw_pose, ensure_model

ANGLE_COLOUR = (0, 255, 255)    # yellow (BGR)
PANEL_BG = (0, 0, 0)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: live joint angles")
    p.add_argument("--video", help="path to a video file (default: webcam)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--csv", help="write per-frame angles to this CSV file")
    p.add_argument("--save-video", help="write the annotated video to this .mp4 file")
    p.add_argument("--no-display", action="store_true",
                   help="process without opening a window (for batch-processing clips)")
    return p.parse_args()


def draw_angle_at(frame, text, x, y):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = int(x) - tw // 2, int(y) + th + 14   # just below the joint
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, ANGLE_COLOUR, thick, cv2.LINE_AA)


def draw_angles(frame, landmarks, angles, mirrored):
    """Write each angle's value next to the joint it is measured at."""
    h, w = frame.shape[:2]

    def px(idx):
        x = (1 - landmarks[idx].x) if mirrored else landmarks[idx].x
        return x * w, landmarks[idx].y * h

    for name, (_, joint, _) in JOINT_ANGLES.items():
        if angles.get(name) is not None:
            draw_angle_at(frame, f"{angles[name]:.0f}", *px(joint))

    if angles.get("Torso Lean") is not None:
        (lx, ly), (rx, ry) = px(L_HIP), px(R_HIP)
        draw_angle_at(frame, f"lean {angles['Torso Lean']:.0f}", (lx + rx) / 2, (ly + ry) / 2 + 18)


def draw_panel(frame, angles, fps):
    """Side panel listing all angles (degrees)."""
    x0, y0, line_h = 10, 50, 22
    height = line_h * (len(ALL_ANGLE_NAMES) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 5, y0 - 18), (x0 + 175, y0 - 18 + height), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Angles (deg)   {fps:4.1f} fps", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(ALL_ANGLE_NAMES, start=1):
        value = angles.get(name)
        text = f"{name:<11} {value:6.1f}" if value is not None else f"{name:<11}    --"
        colour = ANGLE_COLOUR if value is not None else (120, 120, 120)
        cv2.putText(frame, text, (x0, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)


def main():
    args = parse_args()
    ensure_model()

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

    # Webcam is shown mirrored (feels natural); recorded videos are shown as-is.
    mirrored = not from_file
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    csv_file = writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "time_s"] + ALL_ANGLE_NAMES)

    video_out = None
    smoother = AngleSmoother(alpha=0.4)
    show_labels, paused = False, False
    frame_idx, start, last_tick, fps = 0, time.time(), time.time(), 0.0
    display = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if from_file:
                        break           # end of video
                    continue

                h, w = frame.shape[:2]
                # Videos: timestamps from the frame number, so processing speed
                # doesn't matter. Webcam: real elapsed time.
                t_s = frame_idx / video_fps if from_file else time.time() - start

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t_s * 1000))

                display = cv2.flip(frame, 1) if mirrored else frame.copy()

                raw = {name: None for name in ALL_ANGLE_NAMES}
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    raw = compute_angles(lms, w, h)
                    shown = smoother.update(raw)
                    draw_pose(display, lms, show_labels=show_labels, mirrored=mirrored)
                    draw_angles(display, lms, shown, mirrored)
                else:
                    shown = smoother.update(raw)
                    cv2.putText(display, "No pose detected - step into frame", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_tick, 1e-6))
                last_tick = now
                draw_panel(display, shown, fps)

                # The CSV stores RAW angles: the on-screen smoothing lags a little,
                # and calibration should see the true values.
                if writer:
                    writer.writerow([frame_idx, f"{t_s:.3f}"] + [
                        "" if raw[n] is None else f"{raw[n]:.2f}" for n in ALL_ANGLE_NAMES])

                if args.save_video:
                    if video_out is None:
                        video_out = cv2.VideoWriter(args.save_video,
                                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                                    video_fps, (w, h))
                    video_out.write(display)

                frame_idx += 1

            if args.no_display:
                continue

            cv2.imshow("Phase 2: Joint Angles  (space pause, l labels, q quit)", display)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_labels = not show_labels
            if key == ord(" "):
                paused = not paused
    finally:
        landmarker.close()
        cap.release()
        if video_out:
            video_out.release()
        if csv_file:
            csv_file.close()
            print(f"Saved angles for {frame_idx} frames to {args.csv}")
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
