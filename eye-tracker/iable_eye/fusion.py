"""Fusion pointer (S032): a gaze jump brings the cursor near the target, the head finishes.

Pure state machine around HeadPointer. Rules fixed in fable/prompts/S032_fusion_pointer.md
(DECISIONS cycle 22). No file, no camera, no tool.
"""
import math
from dataclasses import dataclass, fields

import numpy as np

from iable_eye.head_pointer import REST_GATE_V2, HeadPointer


class FusionError(ValueError):
    """Refusal with a short machine code in `.code`: config, value."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


@dataclass(frozen=True)
class FusionConfig:
    a_px: float = 110.0  # gaze ellipse semi-axis x (fixed MVP, S027)
    b_px: float = 140.0  # gaze ellipse semi-axis y
    r_out: float = 1.5  # a far run starts at r >= r_out
    r_in: float = 1.0  # a run breaks at r < r_in
    hold_s: float = 0.25
    cooldown_s: float = 0.5
    head_fast_deg_s: float = 8.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not _is_num(v) or not math.isfinite(v):
                raise FusionError("config", f"{f.name} must be a finite number")
            if f.name in ("a_px", "b_px", "r_out", "r_in") and v <= 0:
                raise FusionError("config", f"{f.name} must be > 0")
            if v < 0:
                raise FusionError("config", f"{f.name} must be >= 0")
        if self.r_in > self.r_out:
            raise FusionError("config", "r_in must be <= r_out")


@dataclass(frozen=True)
class FusionStep:
    x_px: float
    y_px: float
    status: str
    jumped: bool
    r: float


class FusionPointer:
    def __init__(self, screen_w_px, screen_h_px, config=FusionConfig(), pointer_config=REST_GATE_V2, start_xy=None):
        if not isinstance(config, FusionConfig):
            raise FusionError("config", "config must be a FusionConfig")
        self.config = config
        self.head = HeadPointer(screen_w_px, screen_h_px, config=pointer_config, start_xy=start_xy)
        self._t_jump = None  # time of the last jump
        self._run = None  # [start_t, sum_x, sum_y, count] while a run is open

    def step(self, t_s, yaw_deg, pitch_deg, ok, gaze_xy) -> FusionStep:
        c = self.config
        # 1. gaze: None or 2 numbers; non-finite entry = no gaze
        if gaze_xy is not None:
            if isinstance(gaze_xy, (str, bytes)) or not isinstance(gaze_xy, (tuple, list, np.ndarray)) \
                    or len(gaze_xy) != 2 or not all(_is_num(v) for v in gaze_xy):
                raise FusionError("value", "gaze_xy must be None or 2 numbers")
        g = None
        if gaze_xy is not None and all(math.isfinite(v) for v in gaze_xy):
            g = (float(gaze_xy[0]), float(gaze_xy[1]))
        # 2. head step (PointerError passes through)
        h = self.head.step(t_s, yaw_deg, pitch_deg, ok)
        cx, cy = h.x_px, h.y_px
        # 3. normalised ellipse distance
        r = math.nan if g is None else math.sqrt(((cx - g[0]) / c.a_px) ** 2 + ((cy - g[1]) / c.b_px) ** 2)
        # 4. busy head / usable frame
        busy = h.status == "glitch" or (h.status == "move" and h.speed_deg_s >= c.head_fast_deg_s)
        usable = bool(ok) and g is not None and not busy
        # 5. run with hysteresis and cooldown
        if self._run is not None:
            if not usable or r < c.r_in:
                self._run = None
            else:
                self._run[1] += g[0]
                self._run[2] += g[1]
                self._run[3] += 1
        elif usable and r >= c.r_out and (self._t_jump is None or t_s >= self._t_jump + c.cooldown_s):
            self._run = [t_s, g[0], g[1], 1]
        # 6. jump to the clipped mean gaze of the run
        jumped = False
        if self._run is not None and t_s - self._run[0] >= c.hold_s:
            tx = self._run[1] / self._run[3]
            ty = self._run[2] / self._run[3]
            self.head.x = min(max(tx, 0.0), self.head.w - 1.0)
            self.head.y = min(max(ty, 0.0), self.head.h - 1.0)
            self._t_jump = t_s
            self._run = None
            jumped = True
        # 7. result (r from before the jump)
        return FusionStep(self.head.x, self.head.y, h.status, jumped, r)
