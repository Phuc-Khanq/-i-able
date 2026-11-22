"""S059: the live MVP demo. Your face moves a cursor; dwell or a key clicks; pause any time.

Modes:  --mode head    (default) the head pointer: turn your head a little, the cursor follows.
        --mode fusion  gaze jumps the cursor near where you look, the head finishes. Needs a model:
                       python eye-tracker/tools/train_demo_model.py --person P001
Output: overlay (default): a full-screen window draws the cursor (nothing outside it moves).
        --os-cursor: the real Windows cursor moves. Needs $env:IABLE_OS_CURSOR='1' (S035 rule).
Tests:  --test reach   8 targets (4 corners, 4 edge middles) x 2; reach each within 8 s (V5).
        --test click   12 targets; click each (dwell 0.6 s, or the key).
Keys (overlay): SPACE click, c recenter, p pause, q quit. Everywhere: Ctrl+Alt+P pause, Ctrl+Alt+Q quit,
        Ctrl+Alt+R recenter, Ctrl+Alt+C click (OS cursor mode).
Frames stay in memory. Writes <data root>/results/<person>_demo_<mode>_<test>_<time>.json and (S064, live
runs) a numbers-only recording <data root>/<person>/demo/d<time>.npz (FaceObs + cursor + labels); --no-record skips it.
--fake: no camera, no window: replays a stored session (for tests and a dry run).

Usage: python eye-tracker/tools/demo.py --person P001 [--mode head|fusion] [--test none|reach|click]
       [--os-cursor] [--click dwell|key] [--gain-slow 10] [--gain-fast 50] [--smooth 1.0]
       [--sign-x -1] [--sign-y 1] [--max-s 600] [--fake]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import session_store as store  # noqa: E402
from iable_eye.app_state import AppState, GatedOutput  # noqa: E402
from iable_eye.capture import ReplayTracker  # noqa: E402
from iable_eye.demo_record import DemoRecorder  # noqa: E402
from iable_eye.fusion import FusionConfig  # noqa: E402
from iable_eye.live_demo import DEMO_POINTER, GAZE_TAU_S, Calibration, ClickTest, DemoConfig, DemoLoop, ReachTest  # noqa: E402
from iable_eye.recal import RecalError  # noqa: E402
from iable_eye.one_euro import OneEuroConfig  # noqa: E402
from iable_eye.model_store import ModelStoreError, load_candidate, model_path  # noqa: E402
from iable_eye.output import NullOutput, OutputError, OverlayModel, WindowsCursorOutput  # noqa: E402
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path  # noqa: E402
from iable_eye.quality import OK  # noqa: E402
from iable_eye.settings import Settings  # noqa: E402

MIN_LOOK = 5             # as run_recal.MIN_FRAMES
FAKE_FPS = 30.0
VK_CTRL, VK_ALT, VK_P, VK_Q, VK_C, VK_R = 0x11, 0x12, 0x50, 0x51, 0x43, 0x52


# ---------------------------------------------------------------- pure helpers (tested)
class Edge:
    """Rising edge of a bool: True once per press."""

    def __init__(self):
        self.prev = True          # a key held at start is not a press

    def __call__(self, down: bool) -> bool:
        hit, self.prev = bool(down) and not self.prev, bool(down)
        return hit


def look_shift(model, gaze_xy, target_xy, min_frames=MIN_LOOK):
    """Same math as candidates.recenter_candidate, from gaze already predicted with the default shift:
    shift = default_shift + (target - median(gaze)). None when there are too few frames."""
    g = np.asarray(gaze_xy, dtype=np.float64).reshape(-1, 2)
    g = g[np.all(np.isfinite(g), axis=1)]
    if len(g) < min_frames:
        return None
    return np.asarray(model.base.default_shift, float) + (np.asarray(target_xy, float) - np.median(g, axis=0))


class Stats:
    def __init__(self):
        self.n = self.head_ok = self.gaze_ok = self.clicks = self.jumps = 0
        self.t_first = self.t_last = None
        self.paused_s = 0.0

    def add(self, step, dt):
        self.n += 1
        self.head_ok += int(step.head_ok)
        self.gaze_ok += int(step.code == OK)
        self.clicks += int(step.click is not None)
        self.jumps += int(step.jumped)
        if step.paused:
            self.paused_s += dt
        self.t_first = step.t if self.t_first is None else self.t_first
        self.t_last = step.t

    def summary(self) -> dict:
        span = (self.t_last - self.t_first) if self.n > 1 else 0.0
        return {"frames": self.n, "duration_s": round(span, 3),
                "fps": round((self.n - 1) / span, 2) if span > 0 else None,
                "head_ok_frac": round(self.head_ok / self.n, 4) if self.n else None,
                "gaze_ok_frac": round(self.gaze_ok / self.n, 4) if self.n else None,
                "clicks": self.clicks, "jumps": self.jumps, "paused_s": round(self.paused_s, 3)}


def run_loop(loop, frames, ui, state, test=None, max_s=600.0, rec=None):
    """frames: iterator of FaceObs. ui.poll() -> set of "click", "pause", "recenter", "quit"; ui.draw(...).
    Ends at quit, test done, max_s, or no more frames. Returns (Stats, stop reason)."""
    stats, prev_t, reason = Stats(), None, "frames"
    for obs in frames:
        ev = ui.poll()
        if "quit" in ev:
            reason = "user"
            break
        if "pause" in ev:
            state.toggle_pause()
        if "recenter" in ev:
            loop.recenter()
        step = loop.step(obs, key_down="click" in ev)
        dt = 0.0 if prev_t is None else step.t - prev_t
        prev_t = step.t
        stats.add(step, dt)
        if rec is not None:
            rec.add(obs, step)
        if test is not None and not step.paused:
            if isinstance(test, ReachTest):
                cur, n0 = test.current(), len(test.times)
                test.step(step.t, step.x_px, step.y_px)
                if rec is not None and len(test.times) > n0 and test.times[-1] is not None:
                    rec.add_label("reach", step.t - 0.3, step.t, *cur[1])
            else:
                cur, n0 = test.current(), len(test.rows)
                test.step(step.t, step.click)
                if rec is not None and len(test.rows) > n0 and test.rows[-1][1] is not None:
                    rec.add_label("click_hit", step.t - loop.cfg.click_config.dwell_s, step.t, *cur)
        ui.draw(step, test, stats)
        if test is not None and test.done:
            reason = "test done"
            break
        if stats.t_first is not None and step.t - stats.t_first >= max_s:
            reason = "max_s"
            break
    return stats, reason


def calib_phase(loop, frames, ui, calib, rec=None):
    """Fusion start (S064): 9 dots, eyes only. Gaze is predicted with the model's default shift.
    Returns "done", "user" or "frames"."""
    for obs in frames:
        if "quit" in ui.poll():
            return "user"
        step = loop.step(obs)
        if rec is not None:
            rec.add(obs, step)
        cur = calib.step(step.t, step.gaze_x, step.gaze_y)
        if cur is None:
            return "done"
        ui.draw_look(cur[1])
    return "frames"


def apply_calibration(loop, calib, centre, log=print):
    """Use the 9-dot fit when it is sane (0.3 <= scale <= 4 per axis); else the centre-dot recenter."""
    try:
        fit = calib.fit()
    except RecalError:
        fit = None
    if fit is not None and np.all((fit[0] >= 0.3) & (fit[0] <= 4.0)):
        loop.set_affine(fit[0], fit[1])
        log(f"calibration: {fit[2]}/9 dots; stretch x {fit[0][0]:.2f}, y {fit[0][1]:.2f} "
            "(1.00 = no squeeze; above 1 = the model was squeezed)")
        return "affine"
    shift = look_shift(loop.model, calib.gaze.get(0, []), centre)
    if shift is not None:
        loop.set_shift(shift)
    log("calibration not used: " + (f"stretch {np.round(fit[0], 2).tolist()} is outside 0.3-4" if fit else "too few good dots")
        + ("; centre recenter used" if shift is not None else "; model default used"))
    return "shift" if shift is not None else "none"


class NullUI:
    """No window, no keys (fake mode and tests). `script`: {frame number: set of events}."""

    def __init__(self, script=None):
        self.script, self.i = dict(script or {}), 0

    def poll(self):
        self.i += 1
        return set(self.script.get(self.i, ()))

    def draw(self, step, test, stats):
        pass

    def draw_look(self, point):
        pass

    def ask(self, lines):
        return True

    def close(self):
        pass


# ---------------------------------------------------------------- live parts (never in a test)
def _async_keys():
    import ctypes
    u = ctypes.windll.user32

    def down(vk):
        return bool(u.GetAsyncKeyState(vk) & 0x8000)
    return down


class LiveUI:
    """Full-screen overlay (or a small status window with --os-cursor). Global hotkeys by polling."""

    NAME = "iable demo"

    def __init__(self, w, h, overlay: bool, mode: str):
        import cv2
        self.cv2, self.w, self.h, self.overlay, self.mode = cv2, w, h, overlay, mode
        self.down = _async_keys() if sys.platform == "win32" else (lambda vk: False)
        self.e_pause, self.e_quit, self.e_click, self.e_center = Edge(), Edge(), Edge(), Edge()
        self._key = -1
        self.fps = 0.0
        self._t = None
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        if overlay:
            cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            cv2.resizeWindow(self.NAME, 560, 170)
            self.canvas = np.zeros((170, 560, 3), dtype=np.uint8)

    def _text(self, s, xy, scale=0.8, color=(200, 200, 200), th=2):
        self.cv2.putText(self.canvas, s, xy, self.cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)

    def ask(self, lines) -> bool:
        cv2 = self.cv2
        self.canvas[:] = 0
        for i, line in enumerate(lines):
            self._text(str(line), (30, 50 + 45 * i), 0.9 if self.overlay else 0.55)
        while True:
            cv2.imshow(self.NAME, self.canvas)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord(" "), ord("q")):
                return k == ord(" ")

    def poll(self):
        ev = set()
        ctrl_alt = self.down(VK_CTRL) and self.down(VK_ALT)
        if self.e_pause(ctrl_alt and self.down(VK_P)) or self._key == ord("p"):
            ev.add("pause")
        if self.e_quit(ctrl_alt and self.down(VK_Q)) or self._key == ord("q"):
            ev.add("quit")
        if self.e_click(ctrl_alt and self.down(VK_C)) or self._key == ord(" "):
            ev.add("click")
        if self.e_center(ctrl_alt and self.down(VK_R)) or self._key == ord("c"):
            ev.add("recenter")
        self._key = -1
        return ev

    def _show(self):
        self.cv2.imshow(self.NAME, self.canvas)
        self._key = self.cv2.waitKey(1) & 0xFF

    def draw_look(self, point):
        cv2 = self.cv2
        self.canvas[:] = 0
        c = (int(point[0]), int(point[1]))
        cv2.circle(self.canvas, c, 14, (255, 255, 255), -1)
        cv2.circle(self.canvas, c, 3, (0, 0, 0), -1)
        self._text("calibration: look at the dot with your EYES, keep your head still",
                   (max(16, self.w // 2 - 480), 40), 0.8)
        self._show()

    def draw(self, step, test, stats):
        pass

    def draw_look(self, point):
        pass

    def ask(self, lines):
        return True

    def close(self):
        pass


# ---------------------------------------------------------------- live parts (never in a test)
def _async_keys():
    import ctypes
    u = ctypes.windll.user32

    def down(vk):
        return bool(u.GetAsyncKeyState(vk) & 0x8000)
    return down


class LiveUI:
    """Full-screen overlay (or a small status window with --os-cursor). Global hotkeys by polling."""

    NAME = "iable demo"

    def __init__(self, w, h, overlay: bool, mode: str):
        import cv2
        self.cv2, self.w, self.h, self.overlay, self.mode = cv2, w, h, overlay, mode
        self.down = _async_keys() if sys.platform == "win32" else (lambda vk: False)
        self.e_pause, self.e_quit, self.e_click, self.e_center = Edge(), Edge(), Edge(), Edge()
        self._key = -1
        self.fps = 0.0
        self._t = None
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        if overlay:
            cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            cv2.resizeWindow(self.NAME, 560, 170)
            self.canvas = np.zeros((170, 560, 3), dtype=np.uint8)

    def _text(self, s, xy, scale=0.8, color=(200, 200, 200), th=2):
        self.cv2.putText(self.canvas, s, xy, self.cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)

    def ask(self, lines) -> bool:
        cv2 = self.cv2
        self.canvas[:] = 0
        for i, line in enumerate(lines):
            self._text(str(line), (30, 50 + 45 * i), 0.9 if self.overlay else 0.55)
        while True:
            cv2.imshow(self.NAME, self.canvas)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord(" "), ord("q")):
                return k == ord(" ")

    def poll(self):
        ev = set()
        ctrl_alt = self.down(VK_CTRL) and self.down(VK_ALT)
        if self.e_pause(ctrl_alt and self.down(VK_P)) or self._key == ord("p"):
            ev.add("pause")
        if self.e_quit(ctrl_alt and self.down(VK_Q)) or self._key == ord("q"):
            ev.add("quit")
        if self.e_click(ctrl_alt and self.down(VK_C)) or self._key == ord(" "):
            ev.add("click")
        if self.e_center(ctrl_alt and self.down(VK_R)) or self._key == ord("c"):
            ev.add("recenter")
        self._key = -1
        return ev

    def _show(self):
        self.cv2.imshow(self.NAME, self.canvas)
        self._key = self.cv2.waitKey(1) & 0xFF

    def draw_look(self, centre):
        cv2 = self.cv2
        self.canvas[:] = 0
        if self.overlay:
            c = (int(centre[0]), int(centre[1]))
            cv2.circle(self.canvas, c, 14, (255, 255, 255), -1)
            self._text("look at the dot", (c[0] - 110, c[1] + 60))
        else:
            self._text("look at the centre of the screen", (15, 80), 0.6)
        self._show()

    def draw(self, step, test, stats):
        cv2 = self.cv2
        if self._t is not None and step.t > self._t:
            self.fps = 0.9 * self.fps + 0.1 / (step.t - self._t) if self.fps else 1.0 / (step.t - self._t)
        self._t = step.t
        self.canvas[:] = 0
        if self.overlay:
            if test is not None and not test.done:
                cur = test.current()
                if isinstance(test, ReachTest):
                    kind, (tx, ty) = cur
                    col = (90, 90, 90) if kind == "home" else (0, 200, 0)
                    r = int(test.radius_px)
                else:
                    tx, ty = cur
                    col, r = (0, 200, 0), int(test.radius_px)
                cv2.circle(self.canvas, (int(tx), int(ty)), r, col, 3)
            if math.isfinite(step.gaze_x) and self.mode == "fusion":
                cv2.circle(self.canvas, (int(step.gaze_x), int(step.gaze_y)), 6, (255, 120, 0), -1)
            p = (int(step.x_px), int(step.y_px))
            cv2.circle(self.canvas, p, 10, (255, 255, 255), -1)
            if step.progress > 0:
                cv2.ellipse(self.canvas, p, (24, 24), -90, 0, 360 * step.progress, (0, 220, 255), 3)
            if step.click is not None:
                cv2.circle(self.canvas, p, 34, (0, 255, 255), 4)
            y0 = 36
        else:
            y0 = 30
        state = "PAUSED (Ctrl+Alt+P)" if step.paused else step.status
        self._text(f"{self.mode} | {self.fps:4.1f} fps | head ok {stats.head_ok / max(stats.n, 1):.2f} | {state}",
                   (16, y0), 0.7 if self.overlay else 0.55)
        keys = ("SPACE click, c recenter, p pause, q quit" if self.overlay
                else "Ctrl+Alt+C click, Ctrl+Alt+R recenter, Ctrl+Alt+P pause, Ctrl+Alt+Q quit")
        self._text(keys, (16, y0 + 34), 0.6 if self.overlay else 0.5, (150, 150, 150), 1)
        if test is not None:
            r = test.result()
            done = r.get("reached", r.get("hits", 0))
            self._text(f"{r['test']} test: {len(r['times_s'])} trials, {done} ok", (16, y0 + 66), 0.6, (150, 150, 150), 1)
        self._show()

    def close(self):
        self.cv2.destroyWindow(self.NAME)


def camera_frames(camera, tracker, clock=time.perf_counter):
    """Live FaceObs stream. Frames stay in memory and are dropped at once."""
    camera.open()
    t0, last = clock(), None
    try:
        while True:
            ok, frame, gray = camera.read()
            try:
                t = float(clock() - t0)
                if not ok or (last is not None and t <= last):
                    continue
                obs = tracker.track(frame, t)
                last = t
            finally:
                del frame, gray
            yield obs
    finally:
        camera.close()


def replay_frames(arrays, fps=FAKE_FPS):
    """Fake stream: a stored session's records, re-stamped at a fixed fps."""
    tr = ReplayTracker(arrays)
    for i in range(tr.n_frames):
        yield tr.track(i, i / fps)


# ---------------------------------------------------------------- main
def _parser():
    ap = argparse.ArgumentParser(description="iable live MVP demo")
    ap.add_argument("--person", required=True)
    ap.add_argument("--mode", choices=("head", "fusion"), default="head")
    ap.add_argument("--test", choices=("none", "reach", "click"), default="none")
    ap.add_argument("--os-cursor", action="store_true")
    ap.add_argument("--click", choices=("dwell", "key"), default=None,
                    help="default: dwell in the overlay, key with --os-cursor")
    ap.add_argument("--gain-slow", type=float, default=DEMO_POINTER.gain_slow_px_per_deg)
    ap.add_argument("--gain-fast", type=float, default=DEMO_POINTER.gain_fast_px_per_deg)
    ap.add_argument("--smooth", type=float, default=DEMO_POINTER.filter.min_cutoff_hz,
                    help="One Euro min cutoff in Hz: lower = steadier but more lag (try 0.5)")
    ap.add_argument("--gaze-smooth", type=float, default=GAZE_TAU_S,
                    help="fusion: gaze smoothing time constant in seconds (0 = off, 0.3 default; higher = steadier, slower)")
    ap.add_argument("--jump-scale", type=float, default=3.0,
                    help="fusion: gaze may only jump the cursor when it is far away; the 110 x 140 px "
                         "ellipse is multiplied by this (3 = jump beyond about 500 px; 1 = old behaviour)")
    ap.add_argument("--sign-x", type=float, choices=(1.0, -1.0), default=DEMO_POINTER.sign_x)
    ap.add_argument("--sign-y", type=float, choices=(1.0, -1.0), default=DEMO_POINTER.sign_y)
    ap.add_argument("--max-s", type=float, default=600.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", choices=store.SOURCES, default="real")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--fake", action="store_true", help="replay a stored session; no camera, no window")
    ap.add_argument("--fake-session", type=int, default=1)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="do not save the numbers-only recording")
    return ap


def results_path(person, mode, test, data_root, stamp) -> Path:
    store._check_person(person)
    return assert_data_path(Path(data_root) / "results" / f"{person}_demo_{mode}_{test}_{stamp}.json",
                            data_root=data_root, root=ROOT)


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    root_d = Path(a.data_root)
    if a.os_cursor and a.test != "none":
        print("REFUSED: --test runs in the overlay only (targets are drawn there); drop --os-cursor")
        return 2
    if a.os_cursor and a.fake:
        print("REFUSED: --fake never moves the OS cursor")
        return 2
    click = a.click or ("key" if a.os_cursor else "dwell")
    try:
        pointer = replace(DEMO_POINTER, gain_slow_px_per_deg=a.gain_slow, gain_fast_px_per_deg=a.gain_fast,
                          sign_x=a.sign_x, sign_y=a.sign_y,
                          filter=OneEuroConfig(min_cutoff_hz=a.smooth, beta=DEMO_POINTER.filter.beta))
        fus = FusionConfig(a_px=FusionConfig().a_px * a.jump_scale, b_px=FusionConfig().b_px * a.jump_scale)
        cfg = DemoConfig(mode=a.mode, click=click, pointer=pointer, fusion=fus, gaze_tau_s=a.gaze_smooth)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    model = None
    if a.mode == "fusion":
        try:
            model, info = load_candidate(model_path(a.person, "kernel_ridge", data_root=root_d, root=ROOT))
        except ModelStoreError as e:
            print(f"REFUSED: {e}. Train it first: python eye-tracker/tools/train_demo_model.py --person {a.person}")
            return 2
        print(f"model: kernel_ridge, trained on sessions {info.get('sessions')} (exams {info.get('exams_skipped')} not used)")

    # screen + output
    if a.fake:
        w, h, dpi = 1920, 1080, "fake"
    else:
        from iable_eye.display import detect_screen
        w, h, dpi = detect_screen()          # DPI-aware first, so 125 % scaling gives real pixels
    print(f"screen: {w} x {h} px (dpi {dpi}); mode {a.mode}; click {click}; "
          f"gain {a.gain_slow}-{a.gain_fast} px/deg; jump x{a.jump_scale:g}; smooth {a.smooth} Hz; signs x {a.sign_x:+.0f} y {a.sign_y:+.0f}; "
          f"output {'OS cursor' if a.os_cursor else 'overlay'}")
    state = AppState(Settings())
    if a.os_cursor:
        try:
            inner = WindowsCursorOutput(w, h, opt_in=True)
        except OutputError as e:
            print(f"REFUSED: {e}. In PowerShell first: $env:IABLE_OS_CURSOR='1'")
            return 2
    else:
        inner = NullOutput(w, h) if a.fake else OverlayModel(w, h)
    out = GatedOutput(state, inner)

    # frames + ui
    tracker = camera = None
    if a.fake:
        ks = store.list_sessions(a.person, a.source, data_root=root_d, root=ROOT)
        if a.fake_session not in ks:
            print(f"REFUSED: no session {a.fake_session} for {a.person} ({a.source}) in {root_d}")
            return 2
        arrays = store.read_session(a.person, a.fake_session, a.source, data_root=root_d, root=ROOT).arrays
        frames, ui = replay_frames(arrays), NullUI()
        print(f"fake: replaying {a.person} s{a.fake_session:04d} ({a.source}) at {FAKE_FPS} fps (no camera, no window)")
    else:
        from iable_eye.capture import CvCamera
        from iable_eye.mp_tracker import MpTracker
        tracker = MpTracker(data_root=root_d)
        camera = CvCamera(a.camera)
        # the calibration dots need the full screen, so fusion always opens the overlay first
        ui = LiveUI(w, h, overlay=not a.os_cursor or a.mode == "fusion", mode=a.mode)
        frames = camera_frames(camera, tracker)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = None
    if not a.fake and not a.no_record and not a.no_write:
        rec = DemoRecorder(a.person, {"mode": a.mode, "test": a.test, "screen": [w, h], "click": click,
                                      "gain_px_per_deg": [a.gain_slow, a.gain_fast], "smooth_hz": a.smooth,
                                      "signs": [a.sign_x, a.sign_y], "stamp": stamp, "dpi": dpi})
    try:
        lines = ["iable demo: the camera tracks your face to move a cursor.",
                 "Frames stay in memory. Face-movement NUMBERS (no images) are saved for training.",
                 "Head mode: turn your head a little; rest = the cursor stays still.",
                 "Dwell click: hold the cursor still for 0.6 s.  Pause: Ctrl+Alt+P.  c = recenter.",
                 "Press SPACE to agree and start, q to quit."]
        if not ui.ask(lines):
            print("stopped: no consent, nothing written")
            return 0
        state.give_consent()                 # in memory only; settings.json is not written
        state.set_cursor_enabled(True)
        state.start_tracking()
        state.start_cursor()
        centre = ((w - 1) / 2.0, (h - 1) / 2.0)
        loop = DemoLoop(w, h, cfg, model=model, output=out, state=state)
        calib_used = None
        if a.mode == "fusion":
            calib = Calibration(w, h)              # its own loop: no output, so no cursor moves yet
            if calib_phase(DemoLoop(w, h, replace(cfg, gaze_tau_s=0.0), model=model), frames, ui, calib, rec) == "user":
                print("stopped with q: nothing written")
                return 0
            calib_used = apply_calibration(loop, calib, centre)
            if rec is not None:
                for t0, t1, x, y in calib.windows():
                    rec.add_label("calib", t0, t1, x, y)
                rec.meta["calibration"] = calib_used
                if loop.affine is not None:
                    rec.meta["affine"] = [loop.affine[0].tolist(), loop.affine[1].tolist()]
            loop.recenter()
            if a.os_cursor:                       # calibration done: shrink to the status window
                ui.close()
                ui = LiveUI(w, h, overlay=False, mode=a.mode)
        test = {"none": None, "reach": ReachTest(w, h), "click": ClickTest(w, h)}[a.test]
        stats, reason = run_loop(loop, frames, ui, state, test, a.max_s, rec)
    finally:
        ui.close()
        if hasattr(frames, "close"):
            frames.close()
        if tracker is not None:
            tracker.close()
    s = stats.summary()
    res = test.result() if test is not None else None
    print(f"stop: {reason}; frames {s['frames']}, fps {s['fps']}, head ok {s['head_ok_frac']}, "
          f"gaze ok {s['gaze_ok_frac']}, clicks {s['clicks']}, jumps {s['jumps']}, paused {s['paused_s']} s"
          f"{' (simulated input)' if a.fake and a.source == 'sim' else ''}")
    if res is not None:
        print("result: " + json.dumps({k: v for k, v in res.items() if k != "times_s"}))
    if a.no_write:
        return 0
    if rec is not None:
        rpath = rec.save(stamp, data_root=root_d, root=ROOT)
        print(f"recorded: {rpath} ({len(rec.obs)} frames, {len(rec.labels)} labels; numbers only)")
    obj = {"person": a.person, "mode": a.mode, "test": a.test, "click": click, "fake": bool(a.fake),
           "source": a.source, "screen": [w, h], "dpi": dpi, "gain_px_per_deg": [a.gain_slow, a.gain_fast],
           "smooth_hz": a.smooth, "signs": [a.sign_x, a.sign_y], "jump_scale": a.jump_scale, "gaze_smooth_s": a.gaze_smooth,
           "os_cursor": bool(a.os_cursor), "stop": reason, "stats": s, "result": res,
           "dropped_while_paused": out.dropped, "demo_version": 2, "calibration": calib_used}
    path = results_path(a.person, a.mode, a.test, root_d, stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
