"""S033: click methods as causal state machines. A click is only a returned ClickEvent (no OS call).
Dwell: still within the radius for dwell_s; armed only by a move (not at start, not after a click).
Blink: min(l, r) >= on closes, < off opens; click if blink_min_s <= L <= blink_max_s.
Key: rising edge of a bool with debounce; a key held at start is not a click.
"""
import math
from dataclasses import dataclass, fields

METHODS = ("dwell", "blink", "key")


class ClickError(ValueError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


@dataclass(frozen=True)
class ClickConfig:
    dwell_s: float = 0.6; dwell_radius_px: float = 30.0  # noqa: E702
    blink_on: float = 0.5; blink_off: float = 0.3  # noqa: E702  closed when min(l, r) >= on; open when < off
    blink_min_s: float = 0.6; blink_max_s: float = 2.0  # noqa: E702
    key_debounce_s: float = 0.05; gap_s: float = 0.5  # noqa: E702

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ClickError("config", f"{f.name} must be a finite number")
            if (v < 0) if f.name == "key_debounce_s" else (v <= 0):
                raise ClickError("config", f"{f.name} out of range")
        if not (self.blink_off < self.blink_on <= 1.0 and self.blink_min_s < self.blink_max_s):
            raise ClickError("config", "need blink_off < blink_on <= 1 and blink_min_s < blink_max_s")


@dataclass(frozen=True)
class ClickEvent:
    t_s: float; x_px: float; y_px: float; source: str  # noqa: E702


class _Clock:
    def __init__(self, config):
        if not isinstance(config, ClickConfig):
            raise ClickError("config", "config must be a ClickConfig")
        self.cfg, self.last_t = config, None

    def _tick(self, t_s):
        """Check t (finite, increasing); return True when t - last_t > gap_s."""
        if not isinstance(t_s, (int, float)) or not math.isfinite(t_s) or (
                self.last_t is not None and t_s <= self.last_t):
            raise ClickError("time", "t_s must be finite and increase")
        gap, self.last_t = self.last_t is not None and t_s - self.last_t > self.cfg.gap_s, t_s
        return gap


class DwellClicker(_Clock):
    def __init__(self, config=ClickConfig()):
        super().__init__(config)
        self.anchor, self.start, self.armed = None, None, False

    @property
    def progress(self):
        if not self.armed or self.anchor is None:
            return 0.0
        return min(1.0, (self.last_t - self.start) / self.cfg.dwell_s)

    def step(self, t_s, x_px, y_px):
        gap = self._tick(t_s)
        if gap or not (math.isfinite(x_px) and math.isfinite(y_px)):
            self.anchor = None
            return None
        if self.anchor is None:
            self.anchor, self.start = (x_px, y_px), t_s
            return None
        if math.hypot(x_px - self.anchor[0], y_px - self.anchor[1]) > self.cfg.dwell_radius_px:
            self.anchor, self.start, self.armed = (x_px, y_px), t_s, True
        elif self.armed and t_s - self.start >= self.cfg.dwell_s:
            self.armed = False
            return ClickEvent(t_s, x_px, y_px, "dwell")
        return None


class BlinkClicker(_Clock):
    def __init__(self, config=ClickConfig()):
        super().__init__(config)
        self.closed, self.t_close, self.point = False, None, None

    def step(self, t_s, score_l, score_r, x_px, y_px):
        gap = self._tick(t_s)
        for s in (score_l, score_r):
            if math.isfinite(s) and not 0.0 <= s <= 1.0:
                raise ClickError("value", "score must be in [0, 1]")
        s = min(score_l, score_r)
        if gap or math.isnan(score_l) or math.isnan(score_r):
            self.closed, self.t_close, self.point = False, None, None
            return None
        if not self.closed and s >= self.cfg.blink_on:
            self.closed, self.t_close, self.point = True, t_s, (x_px, y_px)
        elif self.closed and s < self.cfg.blink_off:
            self.closed = False
            length = t_s - self.t_close
            if self.cfg.blink_min_s <= length <= self.cfg.blink_max_s:
                return ClickEvent(t_s, self.point[0], self.point[1], "blink")
        return None


class KeyClicker(_Clock):
    def __init__(self, config=ClickConfig()):
        super().__init__(config)
        self.prev, self.t_last_click = True, None

    def step(self, t_s, key_down, x_px, y_px):
        if not isinstance(key_down, bool):
            raise ClickError("value", "key_down must be a bool")
        self._tick(t_s)
        rising, self.prev = key_down and not self.prev, key_down
        if not rising:
            return None
        if self.t_last_click is not None and t_s - self.t_last_click < self.cfg.key_debounce_s:
            return None
        self.t_last_click = t_s
        return ClickEvent(t_s, x_px, y_px, "key")


class ClickRouter:
    def __init__(self, method, config=ClickConfig()):
        if method not in METHODS:
            raise ClickError("config", f"method must be one of {METHODS}")
        self.method, self.key = method, KeyClicker(config)
        self.chosen = {"dwell": DwellClicker, "blink": BlinkClicker}.get(method, lambda c: None)(config)

    def step(self, t_s, x_px, y_px, score_l, score_r, key_down):
        k = self.key.step(t_s, key_down, x_px, y_px)
        e = None
        if self.method == "dwell":
            e = self.chosen.step(t_s, x_px, y_px)
        elif self.method == "blink":
            e = self.chosen.step(t_s, score_l, score_r, x_px, y_px)
        return k if k is not None else e  # key wins when both click on one frame
