"""Head pointer core: a relative pointer driven by head yaw and pitch (degrees).

Pure state machine. It never reads a clock and never moves the OS cursor.

Gain (smoothstep between a slow and a fast gain):
    u = clip((s - s_slow) / (s_fast - s_slow), 0, 1)
    g = g_slow + (g_fast - g_slow) * (3 u^2 - 2 u^3)

Every speed uses the real time per frame (dt), so the result does not depend
on the frame rate.
"""
import math
from collections import deque
from dataclasses import dataclass, fields

import numpy as np

from .one_euro import OneEuro, OneEuroConfig

STATUSES = ("anchor", "hold", "lost", "glitch", "rest", "move")


class PointerError(ValueError):
    """Refusal with a short machine code in `.code`: config, time, value."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class PointerConfig:
    gain_slow_px_per_deg: float = 10.0
    gain_fast_px_per_deg: float = 50.0
    speed_slow_deg_s: float = 8.0
    speed_fast_deg_s: float = 60.0
    rest_speed_deg_s: float = 1.0  # per axis
    rest_window_s: float = 0.0  # 0 = one-frame speed (old gate)
    move_enter_deg_s: float = 0.0  # 0 = same as rest_speed_deg_s (no hysteresis)
    max_speed_px_s: float = 4000.0  # step cap per frame = max_speed_px_s * dt
    glitch_speed_deg_s: float = 300.0
    blink_hold_s: float = 0.3
    max_gap_s: float = 0.5
    gain_speed_tau_s: float = 0.0  # 0 = the gain uses the one-frame speed (old); > 0 = EMA time constant (S065)
    sign_x: float = 1.0  # assumed: yaw + -> cursor right
    sign_y: float = -1.0  # assumed: pitch + -> cursor up (screen y grows down)
    filter: OneEuroConfig | None = OneEuroConfig(min_cutoff_hz=1.0, beta=0.05)

    def __post_init__(self):
        for f in fields(self):
            if f.name == "filter":
                continue
            v = getattr(self, f.name)
            if not _finite(v):
                raise PointerError("config", f"{f.name} must be a finite number")
            if f.name.startswith("sign_"):
                if v not in (1.0, -1.0):
                    raise PointerError("config", f"{f.name} must be +1 or -1")
            elif f.name in ("rest_window_s", "move_enter_deg_s", "gain_speed_tau_s"):
                if v < 0:
                    raise PointerError("config", f"{f.name} must be >= 0")
            elif v <= 0:
                raise PointerError("config", f"{f.name} must be > 0")
        if self.gain_fast_px_per_deg < self.gain_slow_px_per_deg:
            raise PointerError("config", "gain_fast must be >= gain_slow")
        if self.speed_fast_deg_s <= self.speed_slow_deg_s:
            raise PointerError("config", "speed_fast must be > speed_slow")
        if self.rest_window_s > 2.0:
            raise PointerError("config", "rest_window_s must be <= 2.0")
        if self.move_enter_deg_s > 0 and self.move_enter_deg_s < self.rest_speed_deg_s:
            raise PointerError("config", "move_enter_deg_s must be >= rest_speed_deg_s")
        if self.filter is not None and not isinstance(self.filter, OneEuroConfig):
            raise PointerError("config", "filter must be a OneEuroConfig or None")


REST_GATE_V2 = PointerConfig(rest_window_s=0.25, move_enter_deg_s=2.5)


def smoothstep_gain(speed_deg_s: float, config: PointerConfig) -> float:
    c = config
    u = (speed_deg_s - c.speed_slow_deg_s) / (c.speed_fast_deg_s - c.speed_slow_deg_s)
    u = min(max(u, 0.0), 1.0)
    return c.gain_slow_px_per_deg + (c.gain_fast_px_per_deg - c.gain_slow_px_per_deg) * (3 * u * u - 2 * u ** 3)


@dataclass(frozen=True)
class PointerStep:
    x_px: float
    y_px: float
    dx_px: float
    dy_px: float
    gain_px_per_deg: float
    speed_deg_s: float
    status: str


class HeadPointer:
    def __init__(self, screen_w_px: int, screen_h_px: int, config=PointerConfig(), start_xy=None):
        if not isinstance(config, PointerConfig):
            raise PointerError("config", "config must be a PointerConfig")
        for v in (screen_w_px, screen_h_px):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise PointerError("config", "screen size must be ints >= 1")
        self.config = config
        self.w, self.h = screen_w_px, screen_h_px
        if start_xy is None:
            start_xy = (screen_w_px / 2, screen_h_px / 2)
        self.x, self.y = self._clip(float(start_xy[0]), float(start_xy[1]))
        self._filter = OneEuro(config.filter, dim=2) if config.filter is not None else None
        self._t_last = None  # last step time (any frame)
        self._t_ok = None  # last ok frame time
        self._ref = None  # previous filtered (yaw, pitch)
        self._reanchor = True
        self._hist = deque()  # (t, cur) of ok frames since the last anchor, oldest first
        self._moving = np.array([False, False])
        self._s_ema = None  # smoothed speed for the gain (only when gain_speed_tau_s > 0)

    def _clip(self, x, y):
        return min(max(x, 0.0), self.w - 1.0), min(max(y, 0.0), self.h - 1.0)

    def _out(self, status, dx=0.0, dy=0.0, g=0.0, s=0.0):
        return PointerStep(self.x, self.y, dx, dy, g, s, status)

    def step(self, t_s: float, yaw_deg: float, pitch_deg: float, ok: bool) -> PointerStep:
        c = self.config
        # 1. refusals
        if not _finite(t_s) or (self._t_last is not None and t_s <= self._t_last):
            raise PointerError("time", "t must be finite and rising")
        if ok and not (_finite(yaw_deg) and _finite(pitch_deg)):
            raise PointerError("value", "angles must be finite on an ok frame")
        self._t_last = t_s
        # 2. bad frame: no move
        if not ok:
            self._reanchor = True
            if self._t_ok is not None and t_s - self._t_ok <= c.blink_hold_s:
                return self._out("hold")
            return self._out("lost")
        # 3. anchor
        if self._reanchor or self._t_ok is None or t_s - self._t_ok > c.max_gap_s:
            if self._filter is not None:
                self._filter.reset()
                self._filter.step(t_s, [yaw_deg, pitch_deg])
            self._ref = np.array([yaw_deg, pitch_deg], dtype=np.float64)
            self._t_ok = t_s
            self._reanchor = False
            self._hist = deque([(t_s, self._ref)])
            self._moving[:] = False
            self._s_ema = None
            return self._out("anchor")
        # 4. filter, delta, per-axis speed with the real dt
        dt = t_s - self._t_ok
        raw = np.array([yaw_deg, pitch_deg], dtype=np.float64)
        cur = self._filter.step(t_s, raw) if self._filter is not None else raw
        delta = cur - self._ref
        v = delta / dt
        if c.rest_window_s == 0:
            w = np.abs(v)  # old path: one-frame speed
        else:
            hist = self._hist
            while len(hist) >= 2 and hist[1][0] <= t_s - c.rest_window_s:
                hist.popleft()
            tb, cb = hist[0]
            w = np.abs(cur - cb) / (t_s - tb)
        if c.rest_window_s > 0:
            self._hist.append((t_s, cur))
        self._ref = cur
        self._t_ok = t_s
        # 5. glitch
        if float(np.linalg.norm(v)) > c.glitch_speed_deg_s:
            return self._out("glitch", s=float(np.linalg.norm(v)))
        # 6. rest gate, per axis, with hysteresis (enter at `enter`, stay while w >= rest).
        # Old config (window 0, enter 0): moving = |v| >= rest in both branches,
        # which is exactly the old gate still = |v| < rest.
        enter = c.move_enter_deg_s or c.rest_speed_deg_s
        moving = np.where(self._moving, w >= c.rest_speed_deg_s, w >= enter)
        self._moving = moving
        still = ~moving
        v = np.where(still, 0.0, v)
        delta = np.where(still, 0.0, delta)
        if still.all():
            self._s_ema = None
            return self._out("rest")
        # 7. gain (S065: with gain_speed_tau_s > 0 the speed that picks the gain is smoothed, so one noisy
        # frame does not flip the gain between slow and fast; the step itself still uses delta, no lag added)
        s = float(np.linalg.norm(v))
        if c.gain_speed_tau_s > 0:
            a = 1.0 - math.exp(-dt / c.gain_speed_tau_s)
            self._s_ema = s if self._s_ema is None else self._s_ema + a * (s - self._s_ema)
            s = self._s_ema
        g = smoothstep_gain(s, c)
        d = np.array([c.sign_x * g * delta[0], c.sign_y * g * delta[1]])
        # 8. cap
        cap = c.max_speed_px_s * dt
        n = float(np.linalg.norm(d))
        if n > cap:
            d = d * (cap / n)
        # 9. clip, overflow dropped
        self.x, self.y = self._clip(self.x + d[0], self.y + d[1])
        return self._out("move", float(d[0]), float(d[1]), g, s)


def run_series(t_s, yaw_deg, pitch_deg, ok, screen_w_px, screen_h_px, config=PointerConfig()) -> dict:
    """Run a HeadPointer over arrays; returns {"x", "y", "status"} arrays."""
    p = HeadPointer(screen_w_px, screen_h_px, config)
    xs, ys, st = [], [], []
    for t, a, b, k in zip(t_s, yaw_deg, pitch_deg, ok):
        r = p.step(float(t), float(a), float(b), bool(k))
        xs.append(r.x_px)
        ys.append(r.y_px)
        st.append(r.status)
    return {"x": np.array(xs), "y": np.array(ys), "status": np.array(st)}
