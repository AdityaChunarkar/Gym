"""
Shared pieces for the exercise rule engines (squat_rules.py, pushup_rules.py).

  - load_thresholds(): read a thresholds dataclass from a JSON file
  - RepAnalyzer: picks the body side the camera sees best, decides whether the
    frame can be trusted at all (joints really visible, camera at the side),
    and keeps the rep counters (total reps and reps with no faults)

Why so many "can we trust this frame?" checks: MediaPipe ALWAYS returns all
33 landmarks, even for body parts that are hidden or out of frame. It just
guesses where they are. Guessed joints jitter, and a jittering knee angle
crosses the "down" and "up" thresholds by itself, which counts fake reps
while you stand still. So a rep only counts when:
  1. every joint the exercise needs is visible and inside the frame,
  2. the camera sees the person from the side (not facing the camera),
  3. the start position was seen first (standing tall / arms locked out),
  4. the body actually travelled (hips/shoulders moved down a real distance),
  5. the rep took a human amount of time (not a 3-frame glitch).
"""

import json
import math
from dataclasses import asdict

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


def load_thresholds(cls, path):
    """Build a thresholds dataclass `cls` from a JSON file, rejecting typos."""
    with open(path) as f:
        data = json.load(f)
    unknown = set(data) - set(asdict(cls()))
    if unknown:
        raise ValueError(f"Unknown threshold names in {path}: {sorted(unknown)}")
    return cls(**data)


def view_ratio(pts):
    """
    Shoulder width divided by torso length, in the image.
    From the SIDE the two shoulders sit almost on top of each other, so this is
    small (about 0.1-0.3). Facing the camera the shoulders are spread out, so
    it is large (about 0.7-1.0). In between = camera at an angle.
    """
    (lx, ly), (rx, ry) = pts[L_SHOULDER], pts[R_SHOULDER]
    sh_mid = ((lx + rx) / 2, (ly + ry) / 2)
    hip_mid = ((pts[L_HIP][0] + pts[R_HIP][0]) / 2, (pts[L_HIP][1] + pts[R_HIP][1]) / 2)
    torso = math.dist(sh_mid, hip_mid)
    if torso < 1e-6:
        return math.inf
    return math.dist((lx, ly), (rx, ry)) / torso


class RepAnalyzer:
    """Base class. Subclasses set SIDES and VIS_KEYS and implement update()."""

    SIDES = {}       # {"L": {joint name: landmark index}, "R": {...}}
    VIS_KEYS = ()    # joints that must be visible for tracking to work

    HINTS = {
        "hidden": "Can't see your whole body - step back so head to feet are in frame",
        "facing": "Turn SIDEWAYS to the camera",
        "start": "Get into the start position to begin counting",
    }

    def __init__(self):
        self.side = None
        self.state = "UP"
        self.ready = False     # becomes True once the start position is seen
        self.reps = 0          # every completed rep
        self.good_reps = 0     # completed reps with no faults
        self.last_rep = None   # dict describing the most recently finished rep
        self.rep_start = 0.0

    def _side_visibility(self, lms):
        return {
            side: sum(getattr(lms[idx[k]], "visibility", 1.0) for k in self.VIS_KEYS)
            / len(self.VIS_KEYS)
            for side, idx in self.SIDES.items()
        }

    def _choose_side(self, scores):
        """Use the side the camera sees best. Only switch between reps, and
        only when the other side is clearly better, so angles don't jump mid-rep."""
        if self.side is None:
            self.side = max(scores, key=scores.get)
        elif self.state == "UP":
            other = "R" if self.side == "L" else "L"
            if scores[other] > scores[self.side] + 0.15:
                self.side = other
        return self.side

    def _visible(self, lms, idx, min_visibility):
        lm = lms[idx]
        in_frame = -0.02 <= lm.x <= 1.02 and -0.02 <= lm.y <= 1.02
        return in_frame and getattr(lm, "visibility", 1.0) >= min_visibility

    def _frame_problem(self, lms, pts, side):
        """Return a HINTS key if this frame can't be trusted, else None.
        Every required joint must be visible on its own; an average would
        let one confident joint hide a guessed one."""
        idx = self.SIDES[side]
        if not all(self._visible(lms, idx[k], self.t.min_visibility) for k in self.VIS_KEYS):
            return "hidden"
        if view_ratio(pts) > self.t.max_view_ratio:
            return "facing"
        return None

    def _abandon(self, side, problem):
        """Tracking was lost: drop any rep in progress and wait for the start
        position again, so a glitch can't finish a rep."""
        self.state = "UP"
        self.ready = False
        return self._not_tracking(side, problem)

    def _record_rep(self, info, faults):
        """Count a finished rep and remember it."""
        self.reps += 1
        if not faults:
            self.good_reps += 1
        self.last_rep = {"rep": self.reps, **info, "faults": faults, "ok": not faults}
        self.state = "UP"
        return self.last_rep

    def _not_tracking(self, side, problem="hidden"):
        return {"tracking": False, "hint": self.HINTS[problem], "side": side,
                "state": self.state, "reps": self.reps, "good_reps": self.good_reps,
                "last_rep": self.last_rep}
