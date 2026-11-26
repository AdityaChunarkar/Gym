"""
Handstand rule engine (Phase 3, gymnastics).

A handstand is a HOLD, not a rep, so this engine times holds instead of
counting up-down movements:
  - detects when the person is upside-down and stacked vertically,
  - times each hold, and how much of it was in good form,
  - remembers the best hold,
  - when a hold ends, reports the form faults that lasted a real part of it.

CAMERA: side-on (90 degrees), camera at about hip height when standing,
3-4 m away so the whole inverted body (hands to feet) fits in frame.

Checks while holding
  line      - feet not stacked over the shoulders (body tilts off vertical)
  arms      - elbows bent instead of locked
  shoulders - shoulders closed: arms not in line with the torso
              (elbow-shoulder-hip angle). An open shoulder is the
              key gymnastics cue: "push the floor away, ears covered".
  body      - shoulder-hip-ankle not straight: arched (banana) or piked
  knees     - legs bent

LIMITATION: the pose model is trained mostly on upright people. Upside-down
poses are tracked less reliably, especially the face and wrists. Hold
timing survives short dropouts (exit_grace), but expect noisier angles than
for squats. Calibrate thresholds from your own gymnastics clips.
"""

import math
from dataclasses import dataclass

from angles import AngleSmoother, angle_at, angle_from_vertical, to_pixels
from rep_base import RepAnalyzer, load_thresholds

MESSAGES = {
    "line": "Stack feet over hips over shoulders",
    "arms": "Lock your elbows",
    "shoulders": "Open shoulders - push the floor away",
    "body": "Body not straight - squeeze glutes, ribs in",
    "knees": "Straighten and squeeze your legs",
}


@dataclass
class HandstandThresholds:
    max_inverted_tilt: float = 30.0   # shoulder->ankle line within this of vertical = in handstand
    max_line_tilt: float = 12.0       # good form: within this of vertical
    min_elbow: float = 160.0          # elbows locked
    min_shoulder: float = 155.0       # elbow-shoulder-hip: shoulders open
    min_body: float = 160.0           # shoulder-hip-ankle: straight body
    min_knee: float = 165.0           # hip-knee-ankle: straight legs
    min_hold: float = 1.0             # seconds; shorter = a kick-up attempt, not a hold
    exit_grace: float = 0.4           # seconds of lost/upright frames before a hold ends
    fault_share: float = 0.25         # a fault is reported if present for this share of the hold
    min_visibility: float = 0.5
    max_view_ratio: float = 0.5

    @classmethod
    def from_json(cls, path):
        return load_thresholds(cls, path)


class HandstandAnalyzer(RepAnalyzer):
    SIDES = {
        "L": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27},
        "R": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28},
    }
    VIS_KEYS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    MESSAGES = MESSAGES
    HINTS = {**RepAnalyzer.HINTS, "start": "Kick up into a handstand, side-on to the camera"}

    def __init__(self, thresholds=None):
        super().__init__()
        self.t = thresholds or HandstandThresholds()
        self.smoother = AngleSmoother(alpha=0.4)
        self.holding = False
        self.best_hold = 0.0
        self._last_time = None

    # -- hold bookkeeping ------------------------------------------------

    def _start_hold(self, time_s):
        self.holding = True
        self.state = "HOLD"
        self._hold_start = self._last_inverted = time_s
        self._fault_time = {k: 0.0 for k in MESSAGES}
        self._clean_time = 0.0

    def _end_hold(self, side):
        self.holding = False
        self.state = "UP"
        duration = self._last_inverted - self._hold_start
        if duration < self.t.min_hold:
            return None                      # a failed kick-up, not a hold
        faults = [k for k, v in self._fault_time.items()
                  if v > self.t.fault_share * duration]
        self.best_hold = max(self.best_hold, duration)
        info = {"time_s": self._hold_start, "side": side, "hold_s": duration,
                "clean_s": self._clean_time}
        return self._record_rep(info, faults)

    def _info(self, tracking, side, hint=None, metrics=None, live=(), finished=None):
        hold = (self._last_inverted - self._hold_start) if self.holding else 0.0
        return {"tracking": tracking, "hint": hint, "side": side, "state": self.state,
                "reps": self.reps, "good_reps": self.good_reps,
                "hold_s": hold, "best_hold_s": self.best_hold,
                "metrics": metrics or {}, "live_faults": list(live),
                "finished_rep": finished, "last_rep": self.last_rep}

    # -- main entry point ------------------------------------------------

    def update(self, lms, width, height, time_s=0.0):
        dt = 0.0 if self._last_time is None else min(max(time_s - self._last_time, 0.0), 0.2)
        self._last_time = time_s

        scores = self._side_visibility(lms)
        side = self._choose_side(scores)
        idx = self.SIDES[side]
        pts = to_pixels(lms, width, height)
        problem = self._frame_problem(lms, pts, side)

        p = {k: pts[i] for k, i in idx.items()}
        s = None
        if not problem:
            s = self.smoother.update({
                "tilt": angle_from_vertical(p["ankle"], p["shoulder"]),
                "elbow": angle_at(p["shoulder"], p["elbow"], p["wrist"]),
                "shoulder": angle_at(p["elbow"], p["shoulder"], p["hip"]),
                "body": angle_at(p["shoulder"], p["hip"], p["ankle"]),
                "knee": angle_at(p["hip"], p["knee"], p["ankle"]),
            })
            if any(v is None for v in s.values()):
                problem = "hidden"

        inverted = (not problem and s["tilt"] <= self.t.max_inverted_tilt
                    and p["wrist"][1] > p["shoulder"][1])     # hands below shoulders

        finished = None
        if inverted:
            if not self.holding:
                self._start_hold(time_s)
            self._last_inverted = time_s
        elif self.holding and time_s - self._last_inverted > self.t.exit_grace:
            finished = self._end_hold(side)   # came down (or lost for too long)

        if problem:
            return {**self._info(False, side, self.HINTS[problem], finished=finished),
                    "tracking": False}

        live = []
        if self.holding and inverted:
            if s["tilt"] > self.t.max_line_tilt:
                live.append("line")
            if s["elbow"] < self.t.min_elbow:
                live.append("arms")
            if s["shoulder"] < self.t.min_shoulder:
                live.append("shoulders")
            if s["body"] < self.t.min_body:
                live.append("body")
            if s["knee"] < self.t.min_knee:
                live.append("knees")
            for k in live:
                self._fault_time[k] += dt
            if not live:
                self._clean_time += dt

        hint = None if self.holding else self.HINTS["start"]
        metrics = {"tilt": s["tilt"], "elbow": s["elbow"], "shoulder": s["shoulder"],
                   "body": s["body"], "knee": s["knee"]}
        return self._info(True, side, hint, metrics, live, finished)
