"""S035: one safe output layer. The base class checks and rounds once; backends only send.
The OS cursor backend starts only with three explicit switches (opt_in, env var, Windows)."""
import math
import os
import sys
from collections import deque
from dataclasses import dataclass

from .display import DisplayError, screen_size
from .display import set_dpi_aware as _set_dpi_aware  # module name, tests patch it

class OutputError(ValueError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code  # "screen", "value", "time", "state", "optin"


@dataclass(frozen=True)
class OutputCommand:
    t: float; kind: str; x_px: int; y_px: int  # noqa: E702  kind "move" or "click"


@dataclass(frozen=True)
class OverlayFrame:
    cursor: tuple | None; trail: tuple; flash: bool; progress: float  # noqa: E702


def _finite(v):
    try:
        return not isinstance(v, bool) and math.isfinite(v)
    except TypeError:
        return False


def to_pixel(v: float, n: int) -> int:
    """Round half up (floor(v + 0.5)), then clamp to 0..n-1."""
    if not _finite(v):
        raise OutputError("value", f"coordinate {v!r} is not finite")
    return min(max(math.floor(v + 0.5), 0), n - 1)


class Output:
    def __init__(self, screen_w_px: int, screen_h_px: int):
        for v in (screen_w_px, screen_h_px):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise OutputError("screen", f"screen size {v!r} must be an int >= 1")
        self.w, self.h = screen_w_px, screen_h_px
        self._last_t = None
        self._last_p = None
        self._closed = False
        self.sent = 0  # number of _move + _click calls so far

    def _check(self, t, x, y):
        # every check happens before anything is sent
        if self._closed:
            raise OutputError("state", "output is closed")
        if not _finite(t) or (self._last_t is not None and t < self._last_t):
            raise OutputError("time", f"time {t!r} is not finite or goes back")
        return to_pixel(x, self.w), to_pixel(y, self.h)

    def move(self, t: float, x_px: float, y_px: float) -> bool:
        p = self._check(t, x_px, y_px)
        self._last_t = t
        if p == self._last_p:
            return False
        self._move(t, *p)
        self._last_p = p
        self.sent += 1
        return True

    def click(self, t: float, x_px: float, y_px: float) -> None:
        p = self._check(t, x_px, y_px)
        self._last_t = t
        if p != self._last_p:
            self._move(t, *p)
            self._last_p = p
            self.sent += 1
        self._click(t, *p)
        self.sent += 1

    def close(self) -> None:
        self._closed = True

    def _move(self, t, x: int, y: int) -> None:
        raise NotImplementedError

    def _click(self, t, x: int, y: int) -> None:
        raise NotImplementedError


class NullOutput(Output):
    def __init__(self, screen_w_px, screen_h_px):
        super().__init__(screen_w_px, screen_h_px)
        self.commands = []

    def _move(self, t, x, y):
        self.commands.append(OutputCommand(t, "move", x, y))

    def _click(self, t, x, y):
        self.commands.append(OutputCommand(t, "click", x, y))


class OverlayModel(Output):
    def __init__(self, screen_w_px, screen_h_px, trail_n: int = 8, flash_s: float = 0.3):
        super().__init__(screen_w_px, screen_h_px)
        if not isinstance(trail_n, int) or isinstance(trail_n, bool) or trail_n < 1:
            raise OutputError("value", f"trail_n {trail_n!r} must be an int >= 1")
        if not _finite(flash_s) or flash_s <= 0:
            raise OutputError("value", f"flash_s {flash_s!r} must be finite and > 0")
        self.flash_s = flash_s
        self._cursor = None
        self._trail = deque(maxlen=trail_n)
        self._flash_until = -math.inf
        self._progress = 0.0

    def _move(self, t, x, y):
        self._cursor = (x, y)
        self._trail.append((x, y))

    def _click(self, t, x, y):
        self._flash_until = t + self.flash_s

    def show_progress(self, p: float) -> None:
        if not _finite(p) or not 0 <= p <= 1:
            raise OutputError("value", f"progress {p!r} must be in 0..1")
        self._progress = float(p)

    def frame(self, t: float) -> OverlayFrame:
        return OverlayFrame(self._cursor, tuple(self._trail), t < self._flash_until, self._progress)


class WindowsCursorOutput(Output):
    def __init__(self, screen_w_px, screen_h_px, *, opt_in: bool = False):
        super().__init__(screen_w_px, screen_h_px)
        if not (opt_in is True and os.environ.get("IABLE_OS_CURSOR") == "1" and sys.platform == "win32"):
            raise OutputError("optin", "OS cursor needs opt_in=True, IABLE_OS_CURSOR=1 and Windows")
        self._u = _load_user32()
        self.dpi_mode = _set_dpi_aware()  # S058: DPI first, then read the size (real px)
        try:
            got = screen_size(self._u)
        except DisplayError as e:
            raise OutputError("screen", str(e)) from e
        if got != (self.w, self.h):
            raise OutputError("screen", f"screen is {got[0]} x {got[1]} px (dpi {self.dpi_mode}) "
                                        f"but the cursor map is {self.w} x {self.h} px")

    def _move(self, t, x, y):
        self._u.SetCursorPos(x, y)

    def _click(self, t, x, y):
        self._u.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
        self._u.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def _load_user32():
    import ctypes
    return ctypes.windll.user32


def apply_step(out: Output, step, click=None) -> int:
    """Send one pipeline step (and an optional click from the same frame). Returns commands sent."""
    if click is not None and click.t_s != step.t:
        raise OutputError("time", f"click t {click.t_s!r} is not the step t {step.t!r}")
    before = out.sent
    if _finite(step.x_px) and _finite(step.y_px):
        out.move(step.t, step.x_px, step.y_px)
    if click is not None:
        out.click(step.t, click.x_px, click.y_px)
    return out.sent - before
