"""
Shared pieces for the exercise rule engines (squat_rules.py, pushup_rules.py).

  - load_thresholds(): read a thresholds dataclass from a JSON file
  - RepAnalyzer: picks the body side the camera sees best and keeps the
    rep counters (total reps and reps with no faults)
"""

import json
from dataclasses import asdict


def load_thresholds(cls, path):
    """Build a thresholds dataclass `cls` from a JSON file, rejecting typos."""
    with open(path) as f:
        data = json.load(f)
    unknown = set(data) - set(asdict(cls()))
    if unknown:
        raise ValueError(f"Unknown threshold names in {path}: {sorted(unknown)}")
    return cls(**data)


class RepAnalyzer:
    """Base class. Subclasses set SIDES and VIS_KEYS and implement update()."""

    SIDES = {}       # {"L": {joint name: landmark index}, "R": {...}}
    VIS_KEYS = ()    # joints that must be visible for tracking to work

    def __init__(self):
        self.side = None
        self.state = "UP"
        self.reps = 0          # every completed rep
        self.good_reps = 0     # completed reps with no faults
        self.last_rep = None   # dict describing the most recently finished rep

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
        return getattr(lms[idx], "visibility", 1.0) >= min_visibility

    def _record_rep(self, info, faults):
        """Count a finished rep and remember it."""
        self.reps += 1
        if not faults:
            self.good_reps += 1
        self.last_rep = {"rep": self.reps, **info, "faults": faults, "ok": not faults}
        self.state = "UP"
        return self.last_rep

    def _not_tracking(self, side):
        return {"tracking": False, "side": side, "state": self.state, "reps": self.reps,
                "good_reps": self.good_reps, "last_rep": self.last_rep}
