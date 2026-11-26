"""
Push-up rule engine (Phase 3).

CAMERA: side-on (90 degrees), camera on the floor or at most knee height,
about 2 m away, head to feet in frame. A front view does NOT work: the body
line and elbow bend point toward the camera and can't be measured in 2D.

Works from a SIDE view, whole body in frame. For each frame it:
  1. picks the arm/body side the camera sees best,
  2. measures the elbow angle, the body line (shoulder-hip-ankle) and the neck,
  3. counts reps with a state machine (UP -> DOWN -> UP) driven by the elbow,
  4. when a rep finishes, checks it against the thresholds and reports faults.

Reps only count while the body is roughly horizontal (a plank). Bending your
arms while standing or kneeling up won't add push-ups.

Checks
  depth - elbow must bend to depth_angle or less at the bottom
  sag   - hips drop below the shoulder-ankle line (lower back sagging)
  pike  - hips rise above the shoulder-ankle line (bum in the air)
  neck  - ear-shoulder-hip angle: head dropping toward the floor or craning
          up. Skipped when the ear isn't visible.

As with squats, the thresholds are starting points. Calibrate them from your
own recordings and pass a JSON file with --thresholds.
"""

import math
from dataclasses import dataclass

from angles import AngleSmoother, angle_at, angle_from_vertical, to_pixels
from rep_base import RepAnalyzer, load_thresholds

MESSAGES = {
    "depth": "Go lower - bend elbows to 90 degrees",
    "sag": "Hips sagging - squeeze glutes and brace your core",
    "pike": "Hips too high - lower them into a straight line",
    "neck": "Keep your head in line with your body",
}


@dataclass
class PushupThresholds:
    lockout_angle: float = 155.0    # elbow angle above this = arms extended (rep finished)
    descend_angle: float = 140.0    # elbow angle below this = rep has started
    depth_angle: float = 90.0       # at the bottom the elbow must reach this angle or lower
    max_body_bend: float = 20.0     # shoulder-hip-ankle may bend at most this far from straight
    min_neck_angle: float = 150.0   # ear-shoulder-hip angle may not drop below this
    min_plank_angle: float = 50.0   # shoulder-ankle line must tilt at least this far from vertical
    min_visibility: float = 0.5     # joints below this are treated as not seen
    # --- "is this a real rep?" checks (see rep_base.py) ---
    max_view_ratio: float = 0.5     # shoulder width / torso length above this = not side-on
    min_shoulder_drop: float = 0.3  # shoulders must drop at least this many upper-arm lengths
    min_rep_time: float = 0.4       # seconds; anything faster is tracking noise

    @classmethod
    def from_json(cls, path):
        return load_thresholds(cls, path)


def signed_body_bend(shoulder, hip, ankle):
    """
    How far the body bends at the hip, in degrees. 0 = straight line.
    Positive = hips SAG (below the shoulder-ankle line, i.e. lower on screen),
    negative = hips PIKE (above the line). Image y grows downward.
    """
    bend = 180.0 - angle_at(shoulder, hip, ankle)
    dx = ankle[0] - shoulder[0]
    if abs(dx) < 1e-6:
        return bend                    # body vertical: no meaningful direction
    t = (hip[0] - shoulder[0]) / dx
    line_y = shoulder[1] + t * (ankle[1] - shoulder[1])
    return bend if hip[1] > line_y else -bend


class PushupAnalyzer(RepAnalyzer):
    SIDES = {
        "L": {"ear": 7, "shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "ankle": 27},
        "R": {"ear": 8, "shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "ankle": 28},
    }
    VIS_KEYS = ("shoulder", "elbow", "wrist", "hip", "ankle")   # ear is optional
    MESSAGES = MESSAGES

    def __init__(self, thresholds=None):
        super().__init__()
        self.t = thresholds or PushupThresholds()
        self.smoother = AngleSmoother(alpha=0.4)
        self._reset_rep_extremes()

    def _reset_rep_extremes(self):
        self._min_elbow = 180.0
        self._max_sag = 0.0
        self._max_pike = 0.0
        self._min_neck = None
        self._max_shoulder_y = -math.inf

    def update(self, lms, width, height, time_s=0.0):
        """Process one frame. Returns a dict describing what to show."""
        scores = self._side_visibility(lms)
        side = self._choose_side(scores)
        idx = self.SIDES[side]
        pts = to_pixels(lms, width, height)
        problem = self._frame_problem(lms, pts, side)
        if problem:
            return self._abandon(side, problem)

        sh, hip, ank = pts[idx["shoulder"]], pts[idx["hip"]], pts[idx["ankle"]]
        raw = {
            "elbow": angle_at(sh, pts[idx["elbow"]], pts[idx["wrist"]]),
            "bend": signed_body_bend(sh, hip, ank),
            "body_tilt": angle_from_vertical(sh, ank),
            "neck": (angle_at(pts[idx["ear"]], sh, hip)
                     if self._visible(lms, idx["ear"], self.t.min_visibility) else None),
        }
        s = self.smoother.update(raw)
        elbow, bend, tilt, neck = s["elbow"], s["bend"], s["body_tilt"], s["neck"]
        if elbow is None or bend is None or tilt is None:   # joints on top of each other
            return self._abandon(side, "hidden")
        in_plank = tilt >= self.t.min_plank_angle

        finished, hint = None, None
        if self.state == "UP":
            if in_plank and elbow > self.t.lockout_angle:
                # Arms locked out in a plank: (re)measure shoulder height and
                # upper-arm length, so the next rep can prove the body went down.
                self.ready = True
                self._top_shoulder_y = sh[1]
                self._upper_arm = math.dist(sh, pts[idx["elbow"]])
            elif not self.ready:
                hint = self.HINTS["start"]
            if self.ready and in_plank and elbow < self.t.descend_angle:
                self.state = "DOWN"
                self.rep_start = time_s
                self._reset_rep_extremes()
        if self.state == "DOWN":
            if not in_plank:
                self.state = "UP"        # got up mid-rep: abandon it, don't count
                self.ready = False
            else:
                self._max_shoulder_y = max(self._max_shoulder_y, sh[1])
                self._min_elbow = min(self._min_elbow, elbow)
                self._max_sag = max(self._max_sag, bend)
                self._max_pike = max(self._max_pike, -bend)
                if neck is not None:
                    self._min_neck = neck if self._min_neck is None else min(self._min_neck, neck)
                if elbow > self.t.lockout_angle:
                    finished = self._finish_rep(side, time_s)

        live_faults = []
        if in_plank:                      # body line matters for the whole set
            if bend > self.t.max_body_bend:
                live_faults.append("sag")
            elif -bend > self.t.max_body_bend:
                live_faults.append("pike")
            if neck is not None and neck < self.t.min_neck_angle:
                live_faults.append("neck")

        return {"tracking": True, "hint": hint, "side": side, "state": self.state,
                "reps": self.reps, "good_reps": self.good_reps, "in_plank": in_plank,
                "metrics": {"elbow": elbow, "hip bend": bend, "neck": neck},
                "live_faults": live_faults, "finished_rep": finished,
                "last_rep": self.last_rep}

    def _finish_rep(self, side, time_s):
        # Reject "reps" where the elbow angle flickered but the body didn't move.
        drop = (self._max_shoulder_y - self._top_shoulder_y) / max(self._upper_arm, 1e-6)
        if drop < self.t.min_shoulder_drop or time_s - self.rep_start < self.t.min_rep_time:
            self.state = "UP"
            return None

        faults = []
        if self._min_elbow > self.t.depth_angle:
            faults.append("depth")
        if self._max_sag > self.t.max_body_bend:
            faults.append("sag")
        if self._max_pike > self.t.max_body_bend:
            faults.append("pike")
        if self._min_neck is not None and self._min_neck < self.t.min_neck_angle:
            faults.append("neck")
        info = {"time_s": time_s, "side": side, "min_elbow": self._min_elbow,
                "max_sag": self._max_sag, "max_pike": self._max_pike,
                "min_neck": self._min_neck}
        return self._record_rep(info, faults)
