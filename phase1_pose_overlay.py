"""
Phase 1: Camera + Labeled Pose Skeleton Overlay
-----------------------------------------------
Opens your webcam, runs MediaPipe's Pose Landmarker on every frame, and
draws a body skeleton with each joint labeled by name (L Shoulder,
R Knee, etc.) on top of the live feed in real time.

Left and right are from the PERSON's point of view, so when you raise
your own left arm, the label on it says "L".

SETUP (run locally on a machine with a webcam):
    pip install -r requirements.txt
    python phase1_pose_overlay.py

The pose model (a few MB) auto-downloads the first time you run this,
then is cached in the same folder for future runs.

Keys in the video window:
    l  - toggle joint labels on/off
    q  - quit
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# --- Model setup -------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

# --- Body parts --------------------------------------------------------
# MediaPipe gives 33 landmarks. These are the ones useful for movement
# analysis, with readable names. Index numbers are fixed by MediaPipe.
BODY_PARTS = {
    0: "Nose",
    11: "L Shoulder", 12: "R Shoulder",
    13: "L Elbow",    14: "R Elbow",
    15: "L Wrist",    16: "R Wrist",
    23: "L Hip",      24: "R Hip",
    25: "L Knee",     26: "R Knee",
    27: "L Ankle",    28: "R Ankle",
    29: "L Heel",     30: "R Heel",
    31: "L Toe",      32: "R Toe",
}

# Drawn as dots but not labeled (too close together on screen).
UNLABELED = {29, 30, 31, 32}

# Lines between joints (landmark index pairs).
POSE_CONNECTIONS = [
    (11, 12),               # shoulders
    (11, 13), (13, 15),     # left arm
    (12, 14), (14, 16),     # right arm
    (11, 23), (12, 24),     # torso sides
    (23, 24),               # hips
    (23, 25), (25, 27),     # left leg
    (24, 26), (26, 28),     # right leg
    (27, 29), (29, 31), (27, 31),   # left foot
    (28, 30), (30, 32), (28, 32),   # right foot
]

# Colours (OpenCV uses BGR order)
LEFT_COLOUR = (255, 140, 0)     # blue-ish  = person's left side
RIGHT_COLOUR = (0, 140, 255)    # orange    = person's right side
CENTRE_COLOUR = (0, 255, 0)     # green     = nose / centre
LINE_COLOUR = (255, 255, 255)   # white skeleton lines

# Joints whose visibility score is below this are not drawn
# (MediaPipe is guessing where they are, e.g. off-screen or hidden).
MIN_VISIBILITY = 0.5


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded and cached.")


def joint_colour(idx):
    name = BODY_PARTS.get(idx, "")
    if name.startswith("L "):
        return LEFT_COLOUR
    if name.startswith("R "):
        return RIGHT_COLOUR
    return CENTRE_COLOUR


def draw_label(frame, text, x, y, colour):
    """Draw text with a dark background box so it stays readable."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = x + 8, y - 8                     # offset so the label sits beside the dot
    h, w = frame.shape[:2]
    x = min(max(x, 0), w - tw - 4)          # keep label inside the frame
    y = min(max(y, th + 4), h - 4)
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_pose(frame, landmarks, show_labels=True, mirrored=True):
    """
    Draw the skeleton and joint labels.

    landmarks come from the UNflipped camera image, so MediaPipe's left/right
    is the person's true left/right. If the displayed frame is mirrored, we
    mirror the x coordinate when drawing instead of flipping the input,
    which keeps the labels correct.
    """
    h, w = frame.shape[:2]

    def to_px(lm):
        x = (1 - lm.x) if mirrored else lm.x
        return int(x * w), int(lm.y * h)

    def visible(idx):
        return getattr(landmarks[idx], "visibility", 1.0) >= MIN_VISIBILITY

    for a, b in POSE_CONNECTIONS:
        if visible(a) and visible(b):
            cv2.line(frame, to_px(landmarks[a]), to_px(landmarks[b]), LINE_COLOUR, 2)

    for idx, name in BODY_PARTS.items():
        if not visible(idx):
            continue
        x, y = to_px(landmarks[idx])
        colour = joint_colour(idx)
        cv2.circle(frame, (x, y), 6, colour, -1)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)
        # Heel/toe dots sit too close together to label without overlap.
        if show_labels and idx not in UNLABELED:
            draw_label(frame, name, x, y, colour)


def draw_legend(frame, show_labels):
    lines = [
        ("Blue = person's LEFT", LEFT_COLOUR),
        ("Orange = person's RIGHT", RIGHT_COLOUR),
        (f"'l' labels {'ON' if show_labels else 'OFF'}   'q' quit", (255, 255, 255)),
    ]
    y = frame.shape[0] - 70
    for text, colour in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y += 22


def main():
    ensure_model()

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)  # 0 = default webcam. Try 1 if you have multiple cameras.
    if not cap.isOpened():
        print("Error: could not open webcam (index 0). "
              "Try changing the index in cv2.VideoCapture() if you have multiple cameras.")
        return

    start_time = time.time()
    show_labels = True

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Detect on the ORIGINAL (unflipped) frame so left/right are correct.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Mirror only the displayed image, so it feels like a mirror.
        display = cv2.flip(frame, 1)

        if result.pose_landmarks:
            draw_pose(display, result.pose_landmarks[0], show_labels, mirrored=True)
            cv2.putText(display, "Pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No pose detected - step into frame", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        draw_legend(display, show_labels)
        cv2.imshow("Phase 1: Labeled Pose Skeleton", display)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('l'):
            show_labels = not show_labels

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
