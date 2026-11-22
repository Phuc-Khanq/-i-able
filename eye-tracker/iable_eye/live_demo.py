"""S059: the live MVP demo loop. One FaceObs in, one DemoStep out. Pure: no camera, no window, no clock.

Per frame:
    StreamGate (REAL_GATE_V1) -> code, features
    head ok  = code in HEAD_OK (ok, blink, small, pose): head pose is valid with the eyes closed
    gaze     = predict_candidate(model, features, shift), only when code == OK (fusion mode only)
    pointer  = HeadPointer (mode "head") or FusionPointer (mode "fusion"), both DEMO_POINTER
               (REST_GATE_V2 with the live-measured signs, see below)
    click    = ClickRouter (dwell or key; the key always works)
    output   = move + click through GatedOutput, so the AppState pause / consent rules decide
While paused, the pointer and the clicker are not stepped; on resume both start fresh at the last
cursor point (so the cursor never jumps after a pause and a dwell never fires right after resume).

ReachTest (V5) and ClickTest are pure scorers of a cursor path. Numbers only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from iable_eye.candidates import Candidate, predict_candidate
from iable_eye.click import ClickConfig, ClickRouter
from iable_eye.features import DEFAULT_FOCAL_PX
from iable_eye.fusion import FusionConfig, FusionPointer
from iable_eye.head_pointer import REST_GATE_V2, HeadPointer, PointerConfig
from iable_eye.one_euro import OneEuro, OneEuroConfig
from iable_eye.quality import BLINK, OK, POSE, REAL_GATE_V1, SMALL, GateConfig
from iable_eye.recal import fit_dot_recal
from iable_eye.stream_gate import StreamGate

MODES = ("head", "fusion")
CLICKS = ("dwell", "key")
HEAD_OK = frozenset({OK, BLINK, SMALL, POSE})
_YAW, _PITCH = 8, 9  # FEATURE_NAMES indexes
# Measured live on 2026-11-22 (first demo run): with the old assumed signs (+1, -1) the cursor
# went down when the head went up, and left when it went right. So both signs flip (DECISIONS 2026-11-22).
# Second live run: "quite good, can be a bit more stable" -> One Euro min cutoff 1.0 -> 0.6 Hz (steadier,
# a little more lag; --smooth changes it). DECISIONS 2026-11-22.
# S065 (live feedback: polish head stability): the gain used to flip between 10 and 50 px/deg on a noisy one-frame speed;
# now the speed that picks the gain is smoothed (tau 0.3 s): slow-move wobble -26 %, no added lag (recorded runs).
DEMO_POINTER = replace(REST_GATE_V2, sign_x=-1.0, sign_y=1.0, gain_speed_tau_s=0.3,
                       filter=OneEuroConfig(min_cutoff_hz=0.6, beta=0.05))
# S066: a light One Euro filter on the cursor POSITION (output only; the pointer state is untouched): slow-move
# wobble 3.45 -> 2.62 px/frame^2 (-24 %), +6 ms lag on the recorded runs; its cutoff grows with speed, so a
# fusion jump passes within a frame.
CURSOR_SMOOTH_HZ, CURSOR_SMOOTH_BETA = 3.0, 0.01
GAZE_TAU_S = 0.3   # S065: gaze EMA time constant in fusion mode (0 = off); per-frame gaze jitter was about 100 px


class DemoError(ValueError):
    """`.code`: "config", "model", "value"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True)
class DemoConfig:
    mode: str = "head"
    click: str = "dwell"
    pointer: PointerConfig = DEMO_POINTER
    fusion: FusionConfig = FusionConfig()
    gate: GateConfig = REAL_GATE_V1
    click_config: ClickConfig = ClickConfig()
    focal_px: float = DEFAULT_FOCAL_PX
    gaze_tau_s: float = GAZE_TAU_S
    cursor_smooth_hz: float = CURSOR_SMOOTH_HZ     # 0 = off

    def __post_init__(self):
        if self.mode not in MODES:
            raise DemoError("config", f"mode must be one of {MODES}, got {self.mode!r}")
        if not (isinstance(self.gaze_tau_s, (int, float)) and math.isfinite(self.gaze_tau_s) and self.gaze_tau_s >= 0):
            raise DemoError("config", "gaze_tau_s must be a finite number >= 0")
        if not (isinstance(self.cursor_smooth_hz, (int, float)) and math.isfinite(self.cursor_smooth_hz)
                and self.cursor_smooth_hz >= 0):
            raise DemoError("config", "cursor_smooth_hz must be a finite number >= 0")
        if self.click not in CLICKS:
            raise DemoError("config", f"click must be one of {CLICKS}, got {self.click!r}")
        if not isinstance(self.pointer, PointerConfig) or self.pointer.rest_window_s == 0:
            raise DemoError("config", "pointer must be a PointerConfig with rest_window_s > 0 (REST_GATE_V2)")


@dataclass(frozen=True)
class DemoStep:
    t: float
    code: int
    head_ok: bool
    gaze_x: float
    gaze_y: float
    x_px: float
    y_px: float
    status: str
    jumped: bool
    click: object        # ClickEvent or None
    progress: float      # dwell progress 0..1
    paused: bool


class DemoLoop:
    def __init__(self, screen_w: int, screen_h: int, config=DemoConfig(), *, model=None, shift=None,
                 output=None, state=None, affine=None):
        if not isinstance(config, DemoConfig):
            raise DemoError("config", "config must be a DemoConfig")
        if config.mode == "fusion" and not isinstance(model, Candidate):
            raise DemoError("model", "fusion mode needs a trained Candidate model")
        if shift is not None:
            shift = np.asarray(shift, dtype=np.float64)
            if shift.shape != (2,) or not np.all(np.isfinite(shift)):
                raise DemoError("value", "shift must be None or a finite (2,) array")
        self.w, self.h, self.cfg = int(screen_w), int(screen_h), config
        self.model, self.shift = model, shift
        self.affine = None
        if affine is not None:
            self.set_affine(*affine)
        self.output, self.state = output, state
        self._gate = StreamGate(config.gate, config.focal_px)
        self._was_paused = False
        self._g_ema = None       # smoothed gaze (x, y) and its last time
        self._g_t = None
        self._new_pointer((self.w / 2.0, self.h / 2.0))

    def _new_pointer(self, xy) -> None:
        c = self.cfg
        if c.mode == "head":
            self._ptr = HeadPointer(self.w, self.h, config=c.pointer, start_xy=xy)
        else:
            self._ptr = FusionPointer(self.w, self.h, config=c.fusion, pointer_config=c.pointer, start_xy=xy)
        self._clicker = ClickRouter(c.click, c.click_config)
        self._g_ema = None
        self._cur_f = (OneEuro(OneEuroConfig(min_cutoff_hz=c.cursor_smooth_hz, beta=CURSOR_SMOOTH_BETA), dim=2)
                       if c.cursor_smooth_hz > 0 else None)

    def recenter(self) -> None:
        """Put the cursor at the screen centre and start the pointer and clicker fresh (key c)."""
        self._new_pointer(((self.w - 1) / 2.0, (self.h - 1) / 2.0))

    @property
    def xy(self) -> tuple:
        h = self._ptr if self.cfg.mode == "head" else self._ptr.head
        return h.x, h.y

    def set_shift(self, shift) -> None:
        s = np.asarray(shift, dtype=np.float64)
        if s.shape != (2,) or not np.all(np.isfinite(s)):
            raise DemoError("value", "shift must be a finite (2,) array")
        self.shift = s

    def set_affine(self, s, b) -> None:
        """Per-axis live calibration: gaze = s * model_gaze + b (from Calibration.fit)."""
        s, b = np.asarray(s, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if s.shape != (2,) or b.shape != (2,) or not (np.isfinite(s).all() and np.isfinite(b).all()):
            raise DemoError("value", "affine s and b must be finite (2,) arrays")
        self.affine = (s, b)

    def _smooth_gaze(self, t, x, y) -> tuple:
        """EMA over OK frames; starts again after a gap longer than 0.5 s (blink, lost face, pause)."""
        tau = self.cfg.gaze_tau_s
        if tau <= 0:
            return x, y
        if self._g_ema is None or t - self._g_t > 0.5:
            self._g_ema = np.array([x, y], dtype=np.float64)
        else:
            a = 1.0 - math.exp(-(t - self._g_t) / tau)
            self._g_ema = self._g_ema + a * (np.array([x, y]) - self._g_ema)
        self._g_t = t
        return float(self._g_ema[0]), float(self._g_ema[1])

    def gaze_of(self, feats) -> tuple:
        g = predict_candidate(self.model, np.asarray(feats, dtype=np.float64).reshape(1, -1), self.shift)[0]
        if self.affine is not None:
            g = self.affine[0] * g + self.affine[1]
        return float(g[0]), float(g[1])

    def step(self, obs, key_down: bool = False) -> DemoStep:
        code, feats = self._gate.step(obs)           # the gate always runs (its history stays causal)
        t = float(obs.t)
        head_ok = code in HEAD_OK
        paused = bool(self.state is not None and self.state.paused)
        gx = gy = math.nan
        if code == OK and self.model is not None:
            gx, gy = self._smooth_gaze(t, *self.gaze_of(feats))
        if paused:
            self._was_paused = True
            x, y = self.xy
            return DemoStep(t, int(code), head_ok, gx, gy, x, y, "paused", False, None, 0.0, True)
        if self._was_paused:                         # resume: fresh pointer and clicker at the last point
            self._was_paused = False
            self._new_pointer(self.xy)
        yaw, pitch = (float(feats[_YAW]), float(feats[_PITCH])) if head_ok else (0.0, 0.0)
        if self.cfg.mode == "head":
            p = self._ptr.step(t, yaw, pitch, head_ok)
            x, y, status, jumped = p.x_px, p.y_px, p.status, False
        else:
            gaze = (gx, gy) if code == OK and math.isfinite(gx) else None
            f = self._ptr.step(t, yaw, pitch, head_ok, gaze)
            x, y, status, jumped = f.x_px, f.y_px, f.status, f.jumped
        if self._cur_f is not None:                    # smooth what is shown, clicked and sent
            x, y = (float(v) for v in self._cur_f.step(t, [x, y]))
        ev = self._clicker.step(t, x, y, math.nan, math.nan, bool(key_down))
        prog = self._clicker.chosen.progress if self.cfg.click == "dwell" else 0.0
        step = DemoStep(t, int(code), head_ok, gx, gy, x, y, status, jumped, ev, prog, False)
        if self.output is not None:                  # the same 2 calls as output.apply_step; works for an
            if math.isfinite(x) and math.isfinite(y):  # Output and for a GatedOutput (it has no .sent)
                self.output.move(t, x, y)
            if ev is not None:
                self.output.click(t, ev.x_px, ev.y_px)
        return step


# ---------------------------------------------------------------- live gaze calibration (pure)
def calib_targets(w: int, h: int) -> list:
    """9 dots: the centre first, then clockwise from the top-left (10/50/90 % x, 12/50/88 % y)."""
    X, Y = [w * f for f in (0.1, 0.5, 0.9)], [h * f for f in (0.12, 0.5, 0.88)]
    order = [(1, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    return [(X[i], Y[j]) for i, j in order]


@dataclass
class Calibration:
    """Fusion start (S064): each dot shows dot_s seconds; gaze frames in [use_s[0], use_s[1]) after its
    onset count. fit(): per-dot medians -> per-axis scale + offset (recal.fit_dot_recal, scale=True).
    It corrects the squeeze seen live (DECISIONS 2026-11-22): the recordings taught "gaze = head
    direction" in part, and live the head stays still."""
    w: int
    h: int
    dot_s: float = 1.6
    use_s: tuple = (0.5, 1.5)
    min_frames: int = 5
    min_dots: int = 6
    targets: list = field(default_factory=list)
    gaze: dict = field(default_factory=dict)       # dot index -> list of (x, y)
    _t0: float | None = None

    def __post_init__(self):
        self.targets = calib_targets(self.w, self.h)

    def current(self, t: float):
        """(index, (x, y)) of the dot to show at time t, or None when done."""
        if self._t0 is None:
            self._t0 = t
        i = int((t - self._t0) // self.dot_s)
        return None if i >= len(self.targets) else (i, self.targets[i])

    def step(self, t: float, gx: float, gy: float):
        cur = self.current(t)
        if cur is None:
            return None
        i = cur[0]
        dt = t - self._t0 - i * self.dot_s
        if self.use_s[0] <= dt < self.use_s[1] and np.isfinite(gx) and np.isfinite(gy):
            self.gaze.setdefault(i, []).append((gx, gy))
        return cur

    def windows(self) -> list:
        """(t0, t1, x, y) per dot: the label windows for the recorder."""
        return [(self._t0 + i * self.dot_s + self.use_s[0], self._t0 + i * self.dot_s + self.use_s[1], x, y)
                for i, (x, y) in enumerate(self.targets)] if self._t0 is not None else []

    def fit(self):
        """(s, b, n_dots) or None when fewer than min_dots dots have min_frames frames."""
        good = [i for i, g in self.gaze.items() if len(g) >= self.min_frames]
        if len(good) < self.min_dots:
            return None
        P = np.array([np.median(np.array(self.gaze[i]), axis=0) for i in sorted(good)])
        T = np.array([self.targets[i] for i in sorted(good)])
        s, b = fit_dot_recal(P, T, scale=True)
        return s, b, len(good)


# ---------------------------------------------------------------- scoring tests (pure)
def reach_targets(w: int, h: int, margin: int = 40) -> list:
    """V5: 4 corners and 4 edge middles, clockwise from the top-left corner."""
    L, R, T, B = float(margin), float(w - 1 - margin), float(margin), float(h - 1 - margin)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return [(L, T), (cx, T), (R, T), (R, cy), (R, B), (cx, B), (L, B), (L, cy)]


def click_targets(w: int, h: int) -> list:
    """12 targets on a 4 x 3 grid (12.5 % .. 87.5 % of the screen), in a fixed mixed order."""
    xs = [w * f for f in (0.125, 0.375, 0.625, 0.875)]
    ys = [h * f for f in (0.2, 0.5, 0.8)]
    grid = [(x, y) for y in ys for x in xs]
    order = (5, 0, 11, 3, 8, 6, 1, 10, 4, 9, 2, 7)
    return [grid[i] for i in order]


@dataclass
class ReachTest:
    """Each target is shown until the cursor is within radius_px, or max_s passes (then it is missed).
    The next target shows when the cursor is back within radius_px of the screen centre."""
    w: int
    h: int
    radius_px: float = 60.0
    max_s: float = 8.0
    repeats: int = 2
    targets: list = field(default_factory=list)
    times: list = field(default_factory=list)      # seconds per trial, None = missed
    _i: int = 0
    _t0: float | None = None
    _home: bool = True                             # waiting for the cursor at the centre

    def __post_init__(self):
        self.targets = reach_targets(self.w, self.h) * self.repeats

    @property
    def done(self) -> bool:
        return self._i >= len(self.targets)

    def current(self):
        """(kind, (x, y)) with kind "home" or "target", or None when done."""
        if self.done:
            return None
        if self._home:
            return "home", ((self.w - 1) / 2.0, (self.h - 1) / 2.0)
        return "target", self.targets[self._i]

    def step(self, t: float, x: float, y: float) -> None:
        cur = self.current()
        if cur is None:
            return
        kind, (tx, ty) = cur
        near = math.hypot(x - tx, y - ty) <= self.radius_px
        if kind == "home":
            if near:
                self._home, self._t0 = False, t
            return
        if near:
            self.times.append(t - self._t0)
        elif t - self._t0 > self.max_s:
            self.times.append(None)
        else:
            return
        self._i += 1
        self._home = True

    def result(self) -> dict:
        n = len(self.times)
        hit = [s for s in self.times if s is not None]
        return {"test": "reach", "trials": n, "reached": len(hit),
                "reach_frac": (len(hit) / n) if n else None,
                "median_s": float(np.median(hit)) if hit else None,
                "max_s_allowed": self.max_s, "radius_px": self.radius_px,
                "times_s": [None if s is None else round(float(s), 3) for s in self.times]}


@dataclass
class ClickTest:
    """One target at a time. A click inside radius_px is a hit; a click outside is a wrong click
    (the target stays). No click in max_s = a miss. The next target shows at once."""
    w: int
    h: int
    radius_px: float = 40.0
    max_s: float = 15.0
    targets: list = field(default_factory=list)
    rows: list = field(default_factory=list)       # (target index, time s or None, wrong clicks)
    _i: int = 0
    _t0: float | None = None
    _wrong: int = 0

    def __post_init__(self):
        self.targets = click_targets(self.w, self.h)

    @property
    def done(self) -> bool:
        return self._i >= len(self.targets)

    def current(self):
        return None if self.done else self.targets[self._i]

    def step(self, t: float, click) -> None:
        """click: a ClickEvent or None."""
        if self.done:
            return
        if self._t0 is None:
            self._t0 = t
        tx, ty = self.targets[self._i]
        if click is not None:
            if math.hypot(click.x_px - tx, click.y_px - ty) <= self.radius_px:
                self._end(t - self._t0)
                self._t0 = t
            else:
                self._wrong += 1
        elif t - self._t0 > self.max_s:
            self._end(None)
            self._t0 = t

    def _end(self, s) -> None:
        self.rows.append((self._i, s, self._wrong))
        self._i += 1
        self._wrong = 0

    def result(self) -> dict:
        hit = [s for _, s, _ in self.rows if s is not None]
        n = len(self.rows)
        return {"test": "click", "trials": n, "hits": len(hit), "misses": n - len(hit),
                "wrong_clicks": int(sum(w for _, _, w in self.rows)),
                "median_s": float(np.median(hit)) if hit else None,
                "radius_px": self.radius_px, "max_s_allowed": self.max_s,
                "times_s": [None if s is None else round(float(s), 3) for _, s, _ in self.rows]}
