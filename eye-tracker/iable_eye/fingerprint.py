"""S009: environment fingerprint of a session: light, distance, frame rate. Numbers only.

Image functions take a grey frame that lives in memory and return 4 Python floats.
Nothing image-like (no frame, no crop) survives a call. No clock: the hour is an argument.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from iable_eye import faceobs
from iable_eye.features import FEATURE_NAMES, session_features

FINGERPRINT_VERSION = 1
FINGERPRINT_KEYS = ("fp_version", "n_frames", "face_frac", "fps", "iris_px_median", "dist_mm_median",
                    "yaw_range_deg", "pitch_range_deg", "roll_range_deg", "brightness_median",
                    "contrast_median", "blur_median", "glare_frac_median", "time_of_day")


class FingerprintError(ValueError):
    """code is one of "shape", "value", "box"."""
    def __init__(self, code, message):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def face_box(landmarks, image_w: int, image_h: int, pad_frac: float = 0.1) -> tuple:
    """478 normalised points -> (x0, y0, x1, y1) Python ints, padded and clipped to the image."""
    if not (_is_int(image_w) and _is_int(image_h)) or image_w <= 0 or image_h <= 0:
        raise FingerprintError("value", f"image size must be positive ints, got {image_w!r} x {image_h!r}")
    if isinstance(pad_frac, bool) or not isinstance(pad_frac, (int, float)) or not 0.0 <= pad_frac <= 1.0:
        raise FingerprintError("value", f"pad_frac must be in [0, 1], got {pad_frac!r}")
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[0] != faceobs.N_LANDMARKS or lm.shape[1] < 2:
        raise FingerprintError("shape", f"landmarks must be ({faceobs.N_LANDMARKS}, 2 or 3), got {lm.shape}")
    xs, ys = lm[:, 0] * image_w, lm[:, 1] * image_h
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise FingerprintError("value", "landmarks must be finite")
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    px, py = pad_frac * (xmax - xmin), pad_frac * (ymax - ymin)
    x0 = min(max(math.floor(xmin - px), 0), image_w)
    x1 = min(max(math.ceil(xmax + px), 0), image_w)
    y0 = min(max(math.floor(ymin - py), 0), image_h)
    y1 = min(max(math.ceil(ymax + py), 0), image_h)
    if x1 <= x0 or y1 <= y0:
        raise FingerprintError("box", f"empty box ({x0}, {y0}, {x1}, {y1})")
    return (int(x0), int(y0), int(x1), int(y1))


@dataclass(frozen=True)
class FrameStats:
    brightness: float
    contrast: float
    blur: float
    glare_frac: float


def frame_stats(gray, box, glare_level: int = 250) -> FrameStats:
    """Mean, population SD, Laplacian variance and glare share of the face crop."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise FingerprintError("shape", "gray must be a 2-D uint8 array")
    if not _is_int(glare_level) or not 0 <= glare_level <= 255:
        raise FingerprintError("value", f"glare_level must be an int in [0, 255], got {glare_level!r}")
    h, w = gray.shape
    if not isinstance(box, (tuple, list)) or len(box) != 4 or not all(_is_int(b) for b in box):
        raise FingerprintError("box", f"box must be 4 ints, got {box!r}")
    x0, y0, x1, y1 = (int(b) for b in box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise FingerprintError("box", f"box {box!r} is empty or outside the {w} x {h} image")
    crop = gray[y0:y1, x0:x1].astype(np.float64)   # crop first, then float64 (no uint8 clipping)
    lap = cv2.Laplacian(crop, cv2.CV_64F, ksize=1)
    return FrameStats(brightness=float(crop.mean()), contrast=float(crop.std()),
                      blur=float(lap.var()), glare_frac=float(np.mean(crop >= glare_level)))


def measured_fps(t) -> float | None:
    """(n - 1) / (t_last - t_first) from the timestamps of the frames that arrived."""
    t = np.asarray(t, dtype=np.float64).ravel()
    if t.size < 2:
        return None
    span = float(t[-1] - t[0])
    if not math.isfinite(span) or span <= 0.0:
        raise FingerprintError("value", f"timestamps must increase, span {span!r}")
    return float((t.size - 1) / span)


def _finite(col):
    col = np.asarray(col, dtype=np.float64)
    return col[np.isfinite(col)]


def _median(col):
    v = _finite(col)
    return float(np.median(v)) if v.size else None


def pose_range_deg(features) -> dict:
    """{"yaw", "pitch", "roll"}: p95 - p5 over the finite values of each column."""
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != len(FEATURE_NAMES):
        raise FingerprintError("shape", f"features must be (n, {len(FEATURE_NAMES)}), got {f.shape}")
    out = {}
    for name in ("yaw", "pitch", "roll"):
        v = _finite(f[:, FEATURE_NAMES.index(f"{name}_deg")])
        out[name] = float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size else None
    return out


def time_of_day(hour: int) -> str:
    if not _is_int(hour) or not 0 <= hour <= 23:
        raise FingerprintError("value", f"hour must be an int in 0..23, got {hour!r}")
    return ("night", "morning", "afternoon", "evening")[int(hour) // 6]


def session_fingerprint(arrays, frame_stats_list=None, hour=None, focal_px=600.0) -> dict:
    """A stacked session -> dict of float / str / None only (no arrays, no numpy scalars)."""
    n = faceobs.validate_stacked(arrays)
    feats = session_features(arrays, focal_px=focal_px)
    pr = pose_range_deg(feats)
    fps = measured_fps(arrays["t"])
    face_frac = float(np.mean(np.asarray(arrays["face_found"], dtype=bool))) if n else None
    img = {"brightness": None, "contrast": None, "blur": None, "glare_frac": None}
    if frame_stats_list is not None:
        stats = [s for s in frame_stats_list if s is not None]
        if not all(isinstance(s, FrameStats) for s in stats):
            raise FingerprintError("value", "frame_stats_list must hold FrameStats or None")
        for k in img:
            img[k] = _median([getattr(s, k) for s in stats])
    out = {
        "fp_version": float(FINGERPRINT_VERSION),
        "n_frames": float(n),
        "face_frac": face_frac,
        "fps": fps,
        "iris_px_median": _median(feats[:, FEATURE_NAMES.index("iris_px")]),
        "dist_mm_median": _median(feats[:, FEATURE_NAMES.index("dist_mm")]),
        "yaw_range_deg": pr["yaw"], "pitch_range_deg": pr["pitch"], "roll_range_deg": pr["roll"],
        "brightness_median": img["brightness"], "contrast_median": img["contrast"],
        "blur_median": img["blur"], "glare_frac_median": img["glare_frac"],
        "time_of_day": None if hour is None else time_of_day(hour),
    }
    assert tuple(out) == FINGERPRINT_KEYS
    return out
