"""S010: the environment probe. One 30 s session per cell: 20 s still, 10 s slow head turn.

Probe sessions go through the normal store into a separate store root `<data_root>/probe/`,
so they never take a gaze session index or an exam slot (DECISIONS 2026-11-18).
DotWindow is the only place that shows anything on screen; it draws a black canvas with one
white dot and a cue. It never shows or keeps a camera frame. Never built at night.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from . import session_store as store
from .capture import capture, store_fingerprint
from .fingerprint import session_fingerprint
from .paths import DATA_ROOT, ROOT

PROBE_VERSION = 1
PROBE_DIR = "probe"
PROBE_DURATION_S = 30.0
MIN_FRAMES = 10
# (start_s, end_s, name, cue); block k covers [start, end)
BLOCKS = (
    (0.0, 20.0, "still", "look at the centre dot, keep still"),
    (20.0, 30.0, "turn", "turn your head slowly left and right, keep looking at the dot"),
)


class ProbeError(ValueError):
    """`code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def probe_root(data_root=DATA_ROOT) -> Path:
    return Path(data_root) / PROBE_DIR


def block_at(t: float) -> int:
    return 0 if t < BLOCKS[1][0] else 1


def cue_at(t: float) -> str:
    return BLOCKS[block_at(t)][3]


def probe_path(person, source, data_root=DATA_ROOT, root=ROOT) -> Path:
    """Where the next probe of this person would go. Creates nothing."""
    pr = probe_root(data_root)
    k = store.next_free_index(person, source, data_root=pr, root=root)
    return store.session_path(person, k, source, data_root=pr, root=root)


def run_probe(camera, tracker, *, person, source, light, distance, posture="upright", glasses="off",
              cam="builtin", data_root=DATA_ROOT, root=ROOT, clock=None, now=datetime.now,
              on_frame=None) -> Path:
    """Capture 30 s and write one probe session. Returns its path."""
    pr = probe_root(data_root)
    # every word is checked before the camera is read (a SessionMeta with a dummy fingerprint)
    k = store.next_free_index(person, source, data_root=pr, root=root)
    meta0 = store.SessionMeta(person=person, session_index=k, source=source, light=light, distance=distance,
                              posture=posture, glasses=glasses, camera=cam,
                              recorded_at="2026-01-01T00:00:00", fingerprint={})
    store.session_path(person, k, source, data_root=pr, root=root)
    store._check_meta_rest(meta0)
    kw = {} if clock is None else {"clock": clock}
    res = capture(camera, tracker, PROBE_DURATION_S, on_frame=on_frame, **kw)
    if res.stop_reason == "user":
        raise ProbeError("user", "stopped with q before 30 s; nothing written")
    arrays = res.arrays
    n = int(arrays["t"].shape[0])
    if n < MIN_FRAMES:
        raise ProbeError("few", f"only {n} frames were kept (at least {MIN_FRAMES}); nothing written")
    probe_block = np.where(arrays["t"] < BLOCKS[1][0], 0, 1).astype(np.int32)
    stamp = now()
    hour = int(stamp.hour)
    fp = store_fingerprint(session_fingerprint(arrays, list(res.frame_stats), hour), hour)
    fp.update(probe_version=float(PROBE_VERSION), cap_failed=float(res.n_failed),
              cap_skipped=float(res.n_skipped))
    meta = store.SessionMeta(person=person, session_index=k, source=source, light=light, distance=distance,
                             posture=posture, glasses=glasses, camera=cam,
                             recorded_at=stamp.strftime("%Y-%m-%dT%H:%M:%S"), fingerprint=fp)
    return store.write_session(arrays, meta, {"probe_block": probe_block}, data_root=pr, root=root)


class DotWindow:
    """Full-screen window: black canvas, white centre dot, the cue text. Live mode only.
    show(cue) -> False when the key q was pressed."""

    NAME = "iable probe"

    def __init__(self, screen_w: int, screen_h: int):
        import cv2
        self._cv2 = cv2
        self.w, self.h = int(screen_w), int(screen_h)
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, cue: str) -> bool:
        cv2 = self._cv2
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)   # drawn here, never a camera frame
        cv2.circle(canvas, (self.w // 2, self.h // 2), 12, (255, 255, 255), -1)
        cv2.putText(canvas, cue, (40, self.h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "q = stop", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 1)
        cv2.imshow(self.NAME, canvas)
        del canvas
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def ask(self, lines) -> bool:
        """S056: an instruction page (text only, black). True on space, False on q."""
        cv2 = self._cv2
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(canvas, str(line), (40, self.h // 3 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)
        while True:
            cv2.imshow(self.NAME, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), ord("q")):
                return key == ord(" ")

    def close(self) -> None:
        self._cv2.destroyWindow(self.NAME)
