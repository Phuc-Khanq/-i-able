"""S010: the capture loop. A camera gives frames, a tracker turns each into a FaceObs.

Only numbers survive: each frame and grey image is deleted at the end of its loop turn.
The capture time of a frame is `clock() - t0`, read just after `camera.read()` returns.

FakeCamera and FakeClock make every timing test exact (ticks / rate, no float drift).
CvCamera opens a real device; it is written here but never run at night (no test builds it).
"""
from __future__ import annotations

import abc
import math
import numbers
import time
from dataclasses import dataclass

import numpy as np

from .faceobs import FaceObs, make_faceobs, no_face, stack, validate_stacked
from .fingerprint import FingerprintError, face_box, frame_stats, time_of_day
from .tracker import BaseTracker, TrackerError

MAX_CONSECUTIVE_FAILS = 30


class CaptureError(ValueError):
    """The capture loop refused something. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


# ---------------------------------------------------------------- clock
class FakeClock:
    """A clock that moves only when `sleep` is called. now() = ticks / rate_hz (exact for k / rate)."""

    def __init__(self, rate_hz: float = 1000.0):
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, numbers.Real) or not (
                math.isfinite(rate_hz) and rate_hz > 0):
            raise CaptureError("value", f"rate_hz must be a positive finite number, got {rate_hz!r}")
        self.rate_hz = float(rate_hz)
        self.ticks = 0

    def now(self) -> float:
        return self.ticks / self.rate_hz

    def __call__(self) -> float:
        return self.now()

    def sleep(self, dt: float) -> None:
        self.ticks += max(1, round(dt * self.rate_hz))


# ---------------------------------------------------------------- cameras
class BaseCamera(abc.ABC):
    """open(), read() -> (ok, frame, gray or None), close(). Also a context manager."""

    def __init__(self):
        self.is_open = False

    @abc.abstractmethod
    def _open(self) -> None: ...

    @abc.abstractmethod
    def _read(self) -> tuple: ...

    @abc.abstractmethod
    def _close(self) -> None: ...

    def open(self) -> None:
        if not self.is_open:
            self._open()
            self.is_open = True

    def read(self) -> tuple:
        if not self.is_open:
            raise CaptureError("closed", "read() on a camera that is not open")
        return self._read()

    def close(self) -> None:
        if self.is_open:
            self.is_open = False
            self._close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeCamera(BaseCamera):
    """The 'frame' is the int index 0..n_frames-1; after that every read fails.
    Read r (counting from 1) fails when fail_every > 0 and r % fail_every == 0; a failed
    read uses no frame index. Every read first calls sleep(1 / fps)."""

    def __init__(self, n_frames: int, fps: float = 30.0, *, gray=None, fail_every: int = 0, sleep=time.sleep):
        super().__init__()
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise CaptureError("value", f"n_frames must be an int >= 0, got {n_frames!r}")
        if isinstance(fps, bool) or not isinstance(fps, numbers.Real) or not (math.isfinite(fps) and fps > 0):
            raise CaptureError("value", f"fps must be a positive finite number, got {fps!r}")
        if isinstance(fail_every, bool) or not isinstance(fail_every, int) or fail_every < 0:
            raise CaptureError("value", f"fail_every must be an int >= 0, got {fail_every!r}")
        if gray is not None and (not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8):
            raise CaptureError("gray", "gray must be None or a 2-D uint8 array")
        self.n_frames, self.fps, self.fail_every = n_frames, float(fps), fail_every
        self._gray, self._sleep = gray, sleep
        self.n_reads = 0
        self._next = 0

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def _read(self) -> tuple:
        self._sleep(1.0 / self.fps)
        self.n_reads += 1
        if (self.fail_every and self.n_reads % self.fail_every == 0) or self._next >= self.n_frames:
            return False, None, None
        i = self._next
        self._next += 1
        return True, i, (None if self._gray is None else self._gray.copy())


class CvCamera(BaseCamera):
    """A real webcam through OpenCV (DirectShow). Frame = (h, w, 3) uint8 RGB, gray = (h, w) uint8.
    Never built in a test or at night."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.index, self.width, self.height = int(index), int(width), int(height)
        self._cap = None

    def _open(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise CaptureError("camera", f"camera {self.index} did not open")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def _read(self) -> tuple:
        import cv2
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return False, None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        del bgr
        return True, rgb, gray

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------- tracker
class ReplayTracker(BaseTracker):
    """Frame i -> the stored record i, re-stamped with the capture time t_s."""

    def __init__(self, arrays: dict):
        super().__init__()
        self.n_frames = validate_stacked(arrays)
        self._a = {k: arrays[k] for k in ("face_found", "landmarks", "blendshapes", "head_matrix")}
        self.image_w, self.image_h = int(arrays["image_w"]), int(arrays["image_h"])

    def _track(self, frame, t_s: float) -> FaceObs:
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < self.n_frames:
            raise TrackerError("frame", f"frame must be an int in [0, {self.n_frames}), got {frame!r}")
        a, i = self._a, frame
        if bool(a["face_found"][i]):
            return make_faceobs(t_s, a["landmarks"][i], a["blendshapes"][i], a["head_matrix"][i],
                                self.image_w, self.image_h)
        return no_face(t_s, self.image_w, self.image_h)


# ---------------------------------------------------------------- the loop
@dataclass(frozen=True)
class CaptureResult:
    arrays: dict          # stacked FaceObs arrays (STACK_KEYS)
    frame_stats: tuple    # one FrameStats or None per kept frame
    n_read: int           # every camera.read() call, the dropped last one included
    n_failed: int
    n_skipped: int        # capture time not after the last kept time
    stop_reason: str      # "duration", "camera" or "user"


def _check_duration(d) -> float:
    if isinstance(d, bool) or not isinstance(d, numbers.Real) or not (math.isfinite(d) and d > 0):
        raise CaptureError("value", f"duration_s must be a positive finite number, got {d!r}")
    return float(d)


def _stats(obs: FaceObs, gray):
    if gray is None:
        return None
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.shape != (obs.image_h, obs.image_w):
        shape = getattr(gray, "shape", None)
        raise CaptureError("gray", f"gray must have shape {(obs.image_h, obs.image_w)}, got {shape}")
    if not obs.face_found:
        return None
    try:
        box = face_box(obs.landmarks, obs.image_w, obs.image_h)
    except FingerprintError as e:
        if e.code == "box":   # the face is fully outside the image: no crop, no stats
            return None
        raise
    return frame_stats(gray, box)


def capture(camera, tracker, duration_s, *, clock=time.perf_counter, on_frame=None) -> CaptureResult:
    """Read frames for duration_s seconds of `clock`. Opens the camera if it is closed
    (and then closes it at the end). on_frame(obs, t) returning False stops ("user")."""
    duration_s = _check_duration(duration_s)
    opened_here = not getattr(camera, "is_open", True)
    if opened_here:
        camera.open()
    obs_list, stats_list = [], []
    n_read = n_failed = n_skipped = consecutive = 0
    last_t = None
    stop_reason = None
    try:
        t0 = clock()
        while stop_reason is None:
            ok, frame, gray = camera.read()
            try:
                n_read += 1
                t = float(clock() - t0)
                if t >= duration_s:
                    stop_reason = "duration"            # this read is dropped
                elif not ok:
                    n_failed += 1
                    consecutive += 1
                    if consecutive >= MAX_CONSECUTIVE_FAILS:
                        stop_reason = "camera"
                else:
                    consecutive = 0
                    if last_t is not None and t <= last_t:
                        n_skipped += 1
                    else:
                        obs = tracker.track(frame, t)
                        stats_list.append(_stats(obs, gray))
                        obs_list.append(obs)
                        last_t = t
                        if on_frame is not None and on_frame(obs, t) is False:
                            stop_reason = "user"
            finally:
                del frame, gray
    finally:
        if opened_here:
            camera.close()
    if not obs_list:
        raise CaptureError("empty", f"no frame was kept ({n_read} reads, {n_failed} failed, stop {stop_reason})")
    return CaptureResult(arrays=stack(obs_list), frame_stats=tuple(stats_list), n_read=n_read,
                         n_failed=n_failed, n_skipped=n_skipped, stop_reason=stop_reason)


def store_fingerprint(fp: dict, hour: int) -> dict:
    """A session_fingerprint dict -> what the store accepts: finite numbers only,
    no None, no `time_of_day` (a word); plus "hour" as a float."""
    time_of_day(hour)  # checks hour is an int in 0..23
    out = {}
    for k, v in fp.items():
        if k == "time_of_day" or v is None:
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            continue
        out[k] = float(v)
    out["hour"] = float(hour)
    return out
