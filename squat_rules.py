"""
Squat rule engine (Phase 3).

Works from a SIDE view. For each frame it:
  1. picks the leg/torso side the camera sees best,
  2. measures the knee angle, torso lean, shin angle and neck angle,
  3. counts reps with a small state machine (UP -> DOWN -> UP),
  4. when a rep finishes, checks it against the thresholds and reports faults.

BACK / SPINE CHECKS - what they can and can't see
  MediaPipe gives no landmarks along the spine, only shoulder and hip. The
  torso is therefore one straight line, and a rounded back looks the same as
  a flat one at the same lean. Real spinal rounding can't be measured here.
  Instead we check three things that go along with a bad back position:
    lean       - torso tilts too far from vertical overall
    torso_shin - torso is much more tilted than the shins. In a good squat the
                 torso stays roughly parallel to the shins. A big gap means the
                 hips shot up first or the chest dropped ("good-morning" squat),
                 which puts the load on the lower back.
    neck       - ear-shoulder-hip angle. Catches the head dropping or cranking
                 up, i.e. the upper spine leaving a neutral line. Skipped when
                 the ear isn't visible.

The numbers in SquatThresholds are sensible starting points, NOT final.
Calibrate them from your own good and bad recordings (record with
phase2_angles.py --csv), then save them in a JSON file and pass --thresholds.
"""

from dataclasses import dataclass

from angles import AngleSmoother, angle_at, angle_from_vertical, to_pixels
from rep_base import RepAnalyzer, load_thresholds

MESSAGES = {
    "depth": "Go deeper - lower your hips",
    "lean": "Chest up - too much forward lean",
    "torso_shin": "Hips rising too fast - keep chest and hips moving together",
    "neck": "Keep your head in line with your back",
}


@dataclass
class SquatThresholds:
    stand_angle: float = 160.0         # knee angle above this = standing (rep finished)
    descend_angle: float = 140.0       # knee angle below this = rep has started
    depth_angle: float = 100.0         # at the bottom the knee must reach this angle or lower
    max_lean: float = 60.0             # torso may not lean further than this from vertical
    max_torso_shin_gap: float = 30.0   # torso lean may exceed shin lean by at most this much
    min_neck_angle: float = 140.0      # ear-shoulder-hip angle may not drop below this
    min_visibility: float = 0.5        # joints below this are treated as not seen

    @classmethod
    def from_json(cls, path):
        return load_thresholds(cls, path)


class SquatAnalyzer(RepAnalyzer):
    # Landmark indices for each side of the body (MediaPipe numbering).
    SIDES = {
        "L": {"ear": 7, "shoulder": 11, "hip": 23, "knee": 25, "ankle": 27},
        "R": {"ear": 8, "shoulder": 12, "hip": 24, "knee": 26, "ankle": 28},
    }
    VIS_KEYS = ("shoulder", "hip", "knee", "ankle")   # ear is optional
    MESSAGES = MESSAGES

    def __init__(self, thresholds=None):
        super().__init__()
        self.t = thresholds or SquatThresholds()
        self.smoother = AngleSmoother(alpha=0.4)
        self._reset_rep_extremes()

    def _reset_rep_extremes(self):
        self._min_knee = 180.0
        self._max_lean = 0.0
        self._max_gap = -180.0
        self._min_neck = None

    def update(self, lms, width, height, time_s=0.0):
        """Process one frame. Returns a dict describing what to show."""
        scores = self._side_visibility(lms)
        side = self._choose_side(scores)
        if scores[side] < self.t.min_visibility:
            return self._not_tracking(side)

        idx = self.SIDES[side]
        pts = to_pixels(lms, width, height)
        raw = {
            "knee": angle_at(pts[idx["hip"]], pts[idx["knee"]], pts[idx["ankle"]]),
            "lean": angle_from_vertical(pts[idx["shoulder"]], pts[idx["hip"]]),
            "shin": angle_from_vertical(pts[idx["knee"]], pts[idx["ankle"]]),
            "neck": (angle_at(pts[idx["ear"]], pts[idx["shoulder"]], pts[idx["hip"]])
                     if self._visible(lms, idx["ear"], self.t.min_visibility) else None),
        }
        s = self.smoother.update(raw)
        knee, lean, shin, neck = s["knee"], s["lean"], s["shin"], s["neck"]
        if knee is None or lean is None or shin is None:   # joints on top of each other
            return self._not_tracking(side)
        gap = lean - shin

        finished = None
        if self.state == "UP":
            if knee < self.t.descend_angle:
                self.state = "DOWN"
                self._reset_rep_extremes()
        if self.state == "DOWN":
            self._min_knee = min(self._min_knee, knee)
            self._max_lean = max(self._max_lean, lean)
            self._max_gap = max(self._max_gap, gap)
            if neck is not None:
                self._min_neck = neck if self._min_neck is None else min(self._min_neck, neck)
            if knee > self.t.stand_angle:
                finished = self._finish_rep(side, time_s)

        live_faults = []
        if self.state == "DOWN":
            if lean > self.t.max_lean:
                live_faults.append("lean")
            if gap > self.t.max_torso_shin_gap:
                live_faults.append("torso_shin")
            if neck is not None and neck < self.t.min_neck_angle:
                live_faults.append("neck")

        return {"tracking": True, "side": side, "state": self.state,
                "reps": self.reps, "good_reps": self.good_reps,
                "metrics": {"knee": knee, "lean": lean, "torso-shin": gap, "neck": neck},
                "live_faults": live_faults, "finished_rep": finished,
                "last_rep": self.last_rep}

    def _finish_rep(self, side, time_s):
        faults = []
        if self._min_knee > self.t.depth_angle:
            faults.append("depth")
        if self._max_lean > self.t.max_lean:
            faults.append("lean")
        if self._max_gap > self.t.max_torso_shin_gap:
            faults.append("torso_shin")
        if self._min_neck is not None and self._min_neck < self.t.min_neck_angle:
            faults.append("neck")
        info = {"time_s": time_s, "side": side, "min_knee": self._min_knee,
                "max_lean": self._max_lean, "max_torso_shin_gap": self._max_gap,
                "min_neck": self._min_neck}
        return self._record_rep(info, faults)
