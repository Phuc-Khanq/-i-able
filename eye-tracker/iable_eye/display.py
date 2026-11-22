"""display: make the process DPI-aware and read the real screen size (S054).

At 125 % Windows scaling a process that is not DPI-aware sees 1536 x 864
"logical" pixels on a 1920 x 1080 screen. After set_dpi_aware() it sees the
real pixels. Call detect_screen() before any window is made.
Tests pass fake `shcore` / `user32` objects; no real OS call in a test.
"""
from __future__ import annotations

import ctypes
import sys

PER_MONITOR = 2                      # PROCESS_PER_MONITOR_DPI_AWARE
E_ACCESSDENIED = -2147024891         # 0x80070005 as a signed 32-bit int: "already set"
FALLBACK_SIZE = (1920, 1080)


class DisplayError(ValueError):
    """`code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _load(name: str):
    try:
        return getattr(ctypes.windll, name)
    except Exception:
        return None


def set_dpi_aware(shcore=None, user32=None, platform=None) -> str:
    """Returns "per_monitor", "already", "system", "none" or "not_windows"."""
    if (sys.platform if platform is None else platform) != "win32":
        return "not_windows"
    if shcore is None and user32 is None:
        shcore, user32 = _load("shcore"), _load("user32")
    if shcore is not None:
        r = int(shcore.SetProcessDpiAwareness(PER_MONITOR))
        if r == 0:
            return "per_monitor"
        if r == E_ACCESSDENIED:
            return "already"
    if user32 is not None and user32.SetProcessDPIAware():
        return "system"
    return "none"


def screen_size(user32) -> tuple[int, int]:
    w, h = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    if w <= 0 or h <= 0:
        raise DisplayError("screen", f"screen size {w} x {h} is not valid")
    return w, h


def detect_screen(shcore=None, user32=None, platform=None) -> tuple[int, int, str]:
    """(width px, height px, dpi mode). Not Windows -> the fallback 1920 x 1080."""
    plat = sys.platform if platform is None else platform
    if plat == "win32" and shcore is None and user32 is None:
        shcore, user32 = _load("shcore"), _load("user32")
    mode = set_dpi_aware(shcore, user32, plat)
    if mode == "not_windows" or user32 is None:
        return (*FALLBACK_SIZE, mode)
    return (*screen_size(user32), mode)
