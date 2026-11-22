"""The simulated protocol: what the eye looks at, and how the head moves, at each time.

One fixed timeline of 148 s, in 7 blocks. Pure functions: the same inputs give the
same numbers. Nothing is written anywhere. Every number is (simulated).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iable_eye.sim_core import SimError, SimScreen, check_screen

PROTOCOL_VERSION = 1
PROTOCOL_DURATION_S = 148.0
GAZE_LATENCY_S = 0.2      # the eye reaches a new grid dot this long after it appears
DOT_S = 1.5               # each grid dot is shown this long
GRID_COLS, GRID_ROWS = 8, 4
GRID_MARGIN = 0.1         # part of the width/height left empty on each side

# (name, start_s, end_s); the block id is the position in this tuple.
BLOCKS = (("neutral", 0.0, 5.0), ("grid", 5.0, 53.0), ("pursuit", 53.0, 73.0),
          ("rest", 73.0, 83.0), ("statue", 83.0, 103.0), ("range", 103.0, 118.0),
          ("natural", 118.0, 148.0))
_STARTS = np.array([b[1] for b in BLOCKS], dtype=np.float64)
_EPS = 1e-9
N_DOTS = GRID_COLS * GRID_ROWS


def _dot_xy(d: np.ndarray, w: float, h: float):
    """Vectorised grid position of dot numbers d (int array, 0..31). Serpentine rows."""
    j = d // GRID_COLS
    p = d % GRID_COLS
    i = np.where(j % 2 == 0, p, GRID_COLS - 1 - p)
    x = GRID_MARGIN * w + i * (1.0 - 2.0 * GRID_MARGIN) * w / (GRID_COLS - 1)
    y = GRID_MARGIN * h + j * (1.0 - 2.0 * GRID_MARGIN) * h / (GRID_ROWS - 1)
    return x.astype(np.float64), y.astype(np.float64)


def grid_dot_px(d, screen) -> tuple:
    """Dot number d (int, 0..31) -> (x, y) in screen pixels."""
    if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or not 0 <= int(d) < N_DOTS:
        raise SimError("dot", f"d must be an int in 0..{N_DOTS - 1}, got {d!r}")
    check_screen(screen)
    x, y = _dot_xy(np.array([int(d)]), float(screen.width_px), float(screen.height_px))
    return float(x[0]), float(y[0])


@dataclass(frozen=True, eq=False)
class ProtocolFrames:
    gaze_px: np.ndarray      # float64 (N, 2), screen pixels
    block: np.ndarray        # int32 (N,), position in BLOCKS
    dot: np.ndarray          # int32 (N,), grid dot number, -1 outside the grid block
    pose_offset: np.ndarray  # float64 (N, 6): yaw, pitch, roll (deg), tx, ty, tz (mm)


def protocol_at(tau_s, screen) -> ProtocolFrames:
    """Protocol times tau_s (1-D, finite, in [0, 148)) -> what happens at each time."""
    if not isinstance(screen, SimScreen):
        raise SimError("screen", f"expected SimScreen, got {type(screen).__name__}")
    check_screen(screen)
    try:
        tau = np.asarray(tau_s, dtype=np.float64)
    except (TypeError, ValueError):
        raise SimError("protocol", "tau_s must be a 1-D array of real numbers") from None
    if tau.ndim != 1:
        raise SimError("protocol", f"tau_s must be 1-D, got {tau.ndim}-D")
    if not np.all(np.isfinite(tau)):
        raise SimError("protocol", "tau_s must be finite")
    if tau.size and (tau.min() < 0.0 or tau.max() >= PROTOCOL_DURATION_S):
        raise SimError("protocol", f"tau_s must be in [0, {PROTOCOL_DURATION_S})")

    w, h = float(screen.width_px), float(screen.height_px)
    n = tau.size
    block = (np.searchsorted(_STARTS, tau + _EPS, side="right") - 1).astype(np.int32)
    u = tau - _STARTS[block]
    gaze = np.empty((n, 2), dtype=np.float64)
    gaze[:, 0], gaze[:, 1] = w / 2.0, h / 2.0
    dot = np.full(n, -1, dtype=np.int32)
    off = np.zeros((n, 6), dtype=np.float64)

    # grid (1)
    m = block == 1
    if m.any():
        um = u[m]
        d = np.floor(um / DOT_S + _EPS).astype(np.int64)
        d = np.clip(d, 0, N_DOTS - 1)
        late = (um - DOT_S * d) < GAZE_LATENCY_S
        target = np.where(late, d - 1, d)
        x, y = _dot_xy(np.clip(target, 0, N_DOTS - 1), w, h)
        x = np.where(target < 0, w / 2.0, x)
        y = np.where(target < 0, h / 2.0, y)
        gaze[m, 0], gaze[m, 1] = x, y
        dot[m] = d.astype(np.int32)

    # pursuit (2)
    m = block == 2
    if m.any():
        a = 2.0 * np.pi * u[m] / 10.0
        gaze[m, 0] = w / 2.0 + 0.35 * w * np.cos(a)
        gaze[m, 1] = h / 2.0 + 0.35 * h * np.sin(a)

    # statue (4): tx for u < 10, then ty
    m = block == 4
    if m.any():
        um = u[m]
        first = um < 10.0
        off[m, 3] = np.where(first, 30.0 * np.sin(2.0 * np.pi * um / 10.0), 0.0)
        off[m, 4] = np.where(first, 0.0, 30.0 * np.sin(2.0 * np.pi * (um - 10.0) / 10.0))

    # range (5): yaw for u < 7.5, then pitch
    m = block == 5
    if m.any():
        um = u[m]
        first = um < 7.5
        off[m, 0] = np.where(first, 15.0 * np.sin(2.0 * np.pi * um / 7.5), 0.0)
        off[m, 1] = np.where(first, 0.0, 10.0 * np.sin(2.0 * np.pi * (um - 7.5) / 7.5))

    # natural (6): reading-like lines of 6 s
    m = block == 6
    if m.any():
        um = u[m]
        line = np.floor(um / 6.0 + _EPS)
        s = np.maximum(um / 6.0 - line, 0.0)
        gaze[m, 0] = 0.1 * w + 0.8 * w * s
        gaze[m, 1] = 0.2 * h + 0.15 * h * line
        off[m, 0] = 3.0 * (2.0 * s - 1.0)

    return ProtocolFrames(gaze_px=gaze, block=block, dot=dot, pose_offset=off)
