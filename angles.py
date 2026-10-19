"""
Joint-angle calculations from MediaPipe pose landmarks.

Every angle here is measured in 2D image space (pixels), which works best
when the camera sees the person from the SIDE. Angles are in degrees:
    180 = joint fully straight
     90 = right angle
    small = joint tightly bent
"""

import math

import numpy as np

# MediaPipe landmark index numbers
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Three-point joint angles: name -> (point A, joint B, point C).
# The angle is measured AT the middle point B.
JOINT_ANGLES = {
    "L Knee":     (L_HIP, L_KNEE, L_ANKLE),
    "R Knee":     (R_HIP, R_KNEE, R_ANKLE),
    "L Hip":      (L_SHOULDER, L_HIP, L_KNEE),
    "R Hip":      (R_SHOULDER, R_HIP, R_KNEE),
    "L Elbow":    (L_SHOULDER, L_ELBOW, L_WRIST),
    "R Elbow":    (R_SHOULDER, R_ELBOW, R_WRIST),
    "L Shoulder": (L_ELBOW, L_SHOULDER, L_HIP),   # arm vs torso
    "R Shoulder": (R_ELBOW, R_SHOULDER, R_HIP),
}

ALL_ANGLE_NAMES = list(JOINT_ANGLES) + ["Torso Lean"]


def angle_at(a, b, c):
    """Angle ABC in degrees, measured at point b. Returns 0-180."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return math.nan
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_from_vertical(top, bottom):
    """
    How far the line from `bottom` up to `top` tilts away from straight up.
    0 = perfectly upright, 90 = horizontal, 180 = upside down.
    (Image y grows downward, hence bottom[1] - top[1].)
    """
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]
    return float(np.degrees(math.atan2(abs(dx), dy)))


def to_pixels(landmarks, width, height):
    """Convert normalised (0-1) landmarks to pixel (x, y) tuples.

    Pixels matter: normalised x and y use different scales on a non-square
    frame, which would distort the angles.
    """
    return [(lm.x * width, lm.y * height) for lm in landmarks]


def _visible(landmarks, idx, min_visibility):
    return getattr(landmarks[idx], "visibility", 1.0) >= min_visibility


def compute_angles(landmarks, width, height, min_visibility=0.5):
    """
    Return {angle name: degrees or None}. An angle is None when any of its
    joints is hidden or off-screen, so later phases never judge form from
    guessed joint positions.
    """
    pts = to_pixels(landmarks, width, height)
    out = {}

    for name, (a, b, c) in JOINT_ANGLES.items():
        if all(_visible(landmarks, i, min_visibility) for i in (a, b, c)):
            out[name] = angle_at(pts[a], pts[b], pts[c])
        else:
            out[name] = None

    torso = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if all(_visible(landmarks, i, min_visibility) for i in torso):
        shoulder_mid = np.mean([pts[L_SHOULDER], pts[R_SHOULDER]], axis=0)
        hip_mid = np.mean([pts[L_HIP], pts[R_HIP]], axis=0)
        out["Torso Lean"] = angle_from_vertical(shoulder_mid, hip_mid)
    else:
        out["Torso Lean"] = None

    return out


class AngleSmoother:
    """
    Exponential moving average per angle, to calm frame-to-frame jitter.
    alpha near 1 = responsive but jumpy; near 0 = smooth but laggy.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.state = {}

    def update(self, angles):
        smoothed = {}
        for name, value in angles.items():
            if value is None or math.isnan(value):
                self.state.pop(name, None)      # reset when the joint disappears
                smoothed[name] = None
                continue
            prev = self.state.get(name)
            new = value if prev is None else self.alpha * value + (1 - self.alpha) * prev
            self.state[name] = new
            smoothed[name] = new
        return smoothed
