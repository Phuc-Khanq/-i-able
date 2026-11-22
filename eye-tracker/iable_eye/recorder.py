"""S029: the recorder. One dot session (form short, recal or session) through the S010 capture loop.

One clock only: every per-frame label comes from `protocol.state_at(tl, t)` where t is the capture
time of that frame (`res.arrays["t"]`). There is no second timeline.
The drawn canvas (black, one white dot, a grey cue) lives in memory only: it is never kept or written.
ScreenWindow is the only place that shows anything; it is never built in a test or at night.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import session_store as store
from .capture import capture, store_fingerprint
from .fingerprint import session_fingerprint
from .protocol import PROTOCOL_ENGINE_VERSION, build_timeline, state_at

REC_VERSION = 1
FORM_CODES = {"short": 0, "recal": 1, "session": 2}
DOT_RADIUS_PX = 12
CUE_GREY = 160
MIN_FRAMES = 10
CELLS = tuple((light, dist) for light in store.LIGHTS for dist in store.DISTANCES)


class RecorderError(ValueError):
    """`code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def cell_for_session(k: int) -> tuple:
    """(light, distance) for session k: LIGHTS x DISTANCES order, index (k - 1) % 9."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= store.MAX_SESSION_INDEX:
        raise RecorderError("index", f"session index must be an int in 1..{store.MAX_SESSION_INDEX}, got {k!r}")
    return CELLS[(k - 1) % len(CELLS)]


# S057: phase 1 records one cell only (bright x normal). The exam rule (every 5th k) does not change.
FOCUS_CELL = ("bright", "normal")
FOCUS_NAME = "bright/normal"
FOCUS_TARGET_SESSIONS = 10
FOCUS_TARGET_EXAMS = 2


def parse_cell(name: str) -> tuple | None:
    """Name -> cell. 'rotation' -> None; '<light>/<distance>' in CELLS -> (light, distance);
    anything else -> RecorderError('cell')."""
    if name == "rotation":
        return None
    if isinstance(name, str) and name.count("/") == 1:
        cell = tuple(name.split("/"))
        if cell in CELLS:
            return cell
    names = ["rotation"] + [f"{l}/{d}" for l, d in CELLS]
    raise RecorderError("cell", f"cell must be one of {names}, got {name!r}")


def cell_for(k: int, cell: tuple | None = None) -> tuple:
    """The cell for session k: the rotation cell when `cell` is None, else `cell` (must be in CELLS).
    k is always checked first (same rule as cell_for_session)."""
    rot = cell_for_session(k)
    if cell is None:
        return rot
    if tuple(cell) not in CELLS:
        raise RecorderError("cell", f"cell must be one of CELLS, got {cell!r}")
    return tuple(cell)


def render_state(state, screen_w: int, screen_h: int, dot_radius_px: int = DOT_RADIUS_PX) -> np.ndarray:
    """(h, w, 3) uint8 black canvas; a filled white dot at (round(x), round(y)) when x, y are finite;
    the cue in grey CUE_GREY at (40, h - 40). Memory only."""
    w, h = int(screen_w), int(screen_h)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = float(state.x_px), float(state.y_px)
    if np.isfinite(x) and np.isfinite(y):
        cv2.circle(canvas, (int(round(x)), int(round(y))), int(dot_radius_px), (255, 255, 255), -1)
    if state.cue:
        cv2.putText(canvas, state.cue, (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (CUE_GREY, CUE_GREY, CUE_GREY), 2)
    return canvas


def protocol_channels(tl, t) -> dict:
    """Per kept frame i, from state_at(tl, t[i]): gaze_px (N, 2) float64 (NaN if none),
    block int32 (index in tl.blocks; -1 after the end), dot int32, in_window bool."""
    t = np.asarray(t, dtype=np.float64)
    names = {b.name: i for i, b in enumerate(tl.blocks)}
    n = t.shape[0]
    gaze = np.full((n, 2), np.nan, dtype=np.float64)
    block = np.empty(n, dtype=np.int32)
    dot = np.empty(n, dtype=np.int32)
    win = np.zeros(n, dtype=bool)
    for i in range(n):
        s = state_at(tl, float(t[i]))
        gaze[i, 0], gaze[i, 1] = s.x_px, s.y_px
        block[i] = names.get(s.block, -1)
        dot[i] = s.dot
        win[i] = s.in_window
    return {"gaze_px": gaze, "block": block, "dot": dot, "in_window": win}


def record_session(camera, tracker, *, person, source, form, screen_w, screen_h, posture="upright",
                   glasses="off", cam="builtin", data_root, root, clock, now, on_frame=None,
                   cell=None) -> Path | None:
    """Capture one protocol session and write it to the normal store. Returns the path,
    or None when the person stopped with q (then nothing is written)."""
    if not isinstance(form, str) or form not in FORM_CODES:
        raise RecorderError("form", f"form must be one of {sorted(FORM_CODES)}, got {form!r}")
    k = store.next_free_index(person, source, data_root=data_root, root=root)
    light, distance = cell_for(k, cell)
    tl =build_timeline(form, screen_w, screen_h)
    # every word is checked before the camera is read
    meta0 = store.SessionMeta(person=person, session_index=k, source=source, light=light, distance=distance,
                              posture=posture, glasses=glasses, camera=cam,
                              recorded_at="2026-01-01T00:00:00", fingerprint={})
    store.session_path(person, k, source, data_root=data_root, root=root)
    store._check_meta_rest(meta0)
    res = capture(camera, tracker, tl.duration_s, clock=clock, on_frame=on_frame)
    if res.stop_reason == "user":
        return None
    arrays = res.arrays
    n = int(arrays["t"].shape[0])
    if n < MIN_FRAMES:
        raise RecorderError("few", f"only {n} frames were kept (at least {MIN_FRAMES}); nothing written")
    channels = protocol_channels(tl, arrays["t"])
    stamp = now()
    hour = int(stamp.hour)
    fp = store_fingerprint(session_fingerprint(arrays, list(res.frame_stats), hour), hour)
    fp.update(rec_version=float(REC_VERSION), proto_engine=float(PROTOCOL_ENGINE_VERSION),
              proto_form=float(FORM_CODES[form]), screen_w_px=float(tl.screen_w_px),
              screen_h_px=float(tl.screen_h_px), cap_failed=float(res.n_failed),
              cap_skipped=float(res.n_skipped))
    meta = store.SessionMeta(person=person, session_index=k, source=source, light=light, distance=distance,
                             posture=posture, glasses=glasses, camera=cam,
                             recorded_at=stamp.strftime("%Y-%m-%dT%H:%M:%S"), fingerprint=fp)
    return store.write_session(arrays, meta, channels, data_root=data_root, root=root)


def check_window_size(got, want) -> None:
    """Pure: refuse when the real window size `got` is not the dot size `want` (both (w, h))."""
    (gw, gh), (ww, wh) = got, want
    if (int(gw), int(gh)) != (int(ww), int(wh)):
        raise RecorderError("screen", f"window is {gw} x {gh}, but the dots are placed for "
                                      f"{ww} x {wh}; use --screen-w {gw} --screen-h {gh}")


class ScreenWindow:
    """Full-screen cv2 window for the dot canvas. Live mode only; never built in a test or at night."""

    NAME = "iable recorder"

    def __init__(self, screen_w: int, screen_h: int):
        self.w, self.h = int(screen_w), int(screen_h)
        self._open = False

    def _black(self, text=None):
        c = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        if text:
            cv2.putText(c, text, (40, self.h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (CUE_GREY, CUE_GREY, CUE_GREY), 2)
        return c

    def open(self) -> None:
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._open = True
        cv2.imshow(self.NAME, self._black())
        cv2.waitKey(50)
        _, _, rw, rh = cv2.getWindowImageRect(self.NAME)
        try:
            check_window_size((int(rw), int(rh)), (self.w, self.h))
        except RecorderError:
            self.close()
            raise

    def wait_start(self) -> bool:
        """'press space to start, q to quit'. True on space, False on q."""
        canvas = self._black("press space to start, q to quit")
        while True:
            cv2.imshow(self.NAME, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                return True
            if key == ord("q"):
                return False

    def ask(self, lines) -> bool:
        """S056: an instruction page (text only, black). True on space, False on q."""
        c = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(c, str(line), (40, self.h // 3 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (CUE_GREY, CUE_GREY, CUE_GREY), 2)
        while True:
            cv2.imshow(self.NAME, c)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), ord("q")):
                return key == ord(" ")

    def show(self, canvas) -> bool:
        cv2.imshow(self.NAME, canvas)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        if self._open:
            self._open = False
            cv2.destroyWindow(self.NAME)
