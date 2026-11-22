"""Calibration protocol engine v1: one exact timeline of blocks and dots (S028).

Headless and pure: time is always an argument (no clock), nothing is drawn or written.
Windows are half-open [a, b). Dot times are start + 1.5 n, never a running sum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PROTOCOL_ENGINE_VERSION = 1
DOT_S = 1.5
WIN_START_S, WIN_END_S = 0.4, 1.2   # settled window inside each dot, relative to dot on
GRID_COLS = 8
GRID_MARGIN = 0.1
RECAL_FRACS = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))
RANGE_CUES = ("turn head left", "turn head right", "turn head up", "turn head down")
RANGE_CUE_S = 3.75
PURSUIT_PERIOD_S = 10.0
PURSUIT_AMP = 0.35

# form -> ((name, start_s, end_s, kind, cue), ...); kind: centre | grid<R> | recal | pursuit | range | none
_FORMS = {
    "short": (("neutral", 0.0, 5.0, "centre", "look at the dot"),
              ("grid", 5.0, 53.0, "grid4", "look at each dot")),
    "recal": (("look", 0.0, 3.0, "centre", "look at the dot"),
              ("recal_dots", 3.0, 9.0, "recal", "look at each dot")),
    "session": (("neutral", 0.0, 5.0, "centre", "look at the dot"),
                ("recal_dots", 5.0, 11.0, "recal", "look at each dot"),
                ("grid", 11.0, 71.0, "grid5", "look at each dot"),
                ("pursuit", 71.0, 91.0, "pursuit", "follow the dot"),
                ("rest", 91.0, 101.0, "centre", "rest, look at the dot"),
                ("statue", 101.0, 121.0, "centre", "keep your eyes on the dot, move your head slowly"),
                ("range", 121.0, 136.0, "range", RANGE_CUES[0]),
                ("natural", 136.0, 166.0, "none", "talk or look around")),
}


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Block:
    name: str
    start_s: float
    end_s: float
    cue: str
    cursor_visible: bool


@dataclass(frozen=True)
class Dot:
    index: int
    block: str
    x_px: float
    y_px: float
    on_s: float
    off_s: float
    win_start_s: float
    win_end_s: float
    use: str


@dataclass(frozen=True)
class Timeline:
    form: str
    screen_w_px: int
    screen_h_px: int
    blocks: tuple
    dots: tuple
    duration_s: float


@dataclass(frozen=True)
class State:
    block: str
    dot: int
    x_px: float
    y_px: float
    in_window: bool
    use: str
    cue: str
    cursor_visible: bool
    done: bool


def _check_px(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) <= 0:
        raise ProtocolError("screen", f"{name} must be a positive int, got {v!r}")
    return int(v)


def _check_t(t, name="t_s"):
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating)):
        raise ProtocolError("time", f"{name} must be a real number, got {t!r}")
    t = float(t)
    if not math.isfinite(t) or t < 0.0:
        raise ProtocolError("time", f"{name} must be finite and >= 0, got {t!r}")
    return t


def _grid_xy(n: int, rows: int, w: float, h: float):
    j, p = n // GRID_COLS, n % GRID_COLS
    i = p if j % 2 == 0 else GRID_COLS - 1 - p
    x = GRID_MARGIN * w + i * (1.0 - 2.0 * GRID_MARGIN) * w / (GRID_COLS - 1)
    y = GRID_MARGIN * h + j * (1.0 - 2.0 * GRID_MARGIN) * h / (rows - 1)
    return float(x), float(y)


def build_timeline(form: str, screen_w_px: int, screen_h_px: int) -> Timeline:
    if not isinstance(form, str) or form not in _FORMS:
        raise ProtocolError("form", f"form must be one of {sorted(_FORMS)}, got {form!r}")
    w_i, h_i = _check_px(screen_w_px, "screen_w_px"), _check_px(screen_h_px, "screen_h_px")
    w, h = float(w_i), float(h_i)
    blocks, dots = [], []
    for name, s0, s1, kind, cue in _FORMS[form]:
        blocks.append(Block(name, s0, s1, cue, False))
        if kind.startswith("grid") or kind == "recal":
            rows = int(kind[4:]) if kind != "recal" else 0
            count = GRID_COLS * rows if rows else len(RECAL_FRACS)
            for n in range(count):
                if rows:
                    x, y = _grid_xy(n, rows, w, h)
                else:
                    fx, fy = RECAL_FRACS[n]
                    x, y = fx * w, fy * h
                on = s0 + DOT_S * n
                dots.append(Dot(n, name, x, y, on, on + DOT_S, on + WIN_START_S, on + WIN_END_S,
                                "grid" if rows else "fit"))
    return Timeline(form, w_i, h_i, tuple(blocks), tuple(dots), blocks[-1].end_s)


def _kind(tl: Timeline, name: str) -> str:
    for b in _FORMS[tl.form]:
        if b[0] == name:
            return b[3]
    raise ProtocolError("form", f"unknown block {name!r}")


def state_at(tl: Timeline, t_s: float) -> State:
    t = _check_t(t_s)
    nan = float("nan")
    if t >= tl.duration_s:
        return State("end", -1, nan, nan, False, "", "", False, True)
    b = next(b for b in tl.blocks if b.start_s <= t < b.end_s)
    kind = _kind(tl, b.name)
    w, h = float(tl.screen_w_px), float(tl.screen_h_px)
    if kind.startswith("grid") or kind == "recal":
        bd = [d for d in tl.dots if d.block == b.name]
        n = min(max(int(math.floor((t - b.start_s) / DOT_S)), 0), len(bd) - 1)
        while n > 0 and t < bd[n].on_s:
            n -= 1
        while n < len(bd) - 1 and t >= bd[n].off_s:
            n += 1
        d = bd[n]
        return State(b.name, d.index, d.x_px, d.y_px, d.win_start_s <= t < d.win_end_s,
                     d.use, b.cue, b.cursor_visible, False)
    if kind == "centre":
        return State(b.name, -1, w / 2.0, h / 2.0, False, "", b.cue, b.cursor_visible, False)
    if kind == "pursuit":
        a = 2.0 * np.pi * (t - b.start_s) / PURSUIT_PERIOD_S
        x = w / 2.0 + PURSUIT_AMP * w * np.cos(a)
        y = h / 2.0 + PURSUIT_AMP * h * np.sin(a)
        return State(b.name, -1, float(x), float(y), False, "", b.cue, b.cursor_visible, False)
    cue = b.cue
    if kind == "range":
        k = min(max(int(math.floor((t - b.start_s) / RANGE_CUE_S)), 0), len(RANGE_CUES) - 1)
        cue = RANGE_CUES[k]
    return State(b.name, -1, nan, nan, False, "", cue, b.cursor_visible, False)


def _all_events(tl: Timeline) -> list:
    ev = [(b.start_s, "block_start", -1) for b in tl.blocks]
    for d in tl.dots:
        ev += [(d.on_s, "dot_on", d.index), (d.win_start_s, "window_start", d.index),
               (d.win_end_s, "window_end", d.index), (d.off_s, "dot_off", d.index)]
    ev.append((tl.duration_s, "end", -1))
    order = {"dot_off": 0, "window_end": 1, "end": 2, "block_start": 3, "dot_on": 4, "window_start": 5}
    return sorted(ev, key=lambda e: (e[0], order[e[1]]))


def events_between(tl: Timeline, t0_s: float, t1_s: float) -> list:
    t0, t1 = _check_t(t0_s, "t0_s"), _check_t(t1_s, "t1_s")
    if t1 < t0:
        raise ProtocolError("time", f"t1_s {t1} < t0_s {t0}")
    return [e for e in _all_events(tl) if t0 <= e[0] < t1]


def describe(tl: Timeline) -> list:
    lines = [f"form {tl.form} at {tl.screen_w_px} x {tl.screen_h_px} (engine v{PROTOCOL_ENGINE_VERSION})"]
    for b in tl.blocks:
        n = sum(1 for d in tl.dots if d.block == b.name)
        L = b.end_s - b.start_s
        if n:
            lines.append(f"{b.name}: {n} dots x {DOT_S} s = {n * DOT_S} s  [{b.start_s}, {b.end_s})")
        elif _kind(tl, b.name) == "range":
            lines.append(f"{b.name}: {len(RANGE_CUES)} cues x {RANGE_CUE_S} s = "
                         f"{len(RANGE_CUES) * RANGE_CUE_S} s  [{b.start_s}, {b.end_s})")
        else:
            lines.append(f"{b.name}: {b.end_s} - {b.start_s} = {L} s")
    lines.append(f"total: sum of {len(tl.blocks)} blocks = {sum(b.end_s - b.start_s for b in tl.blocks)} s")
    return lines
