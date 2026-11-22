"""FaceObs: one numbers-only record per camera frame.

A record can hold only 478 landmarks, 52 blendshape scores, one 4x4 head
matrix, a time and the image size. It cannot hold a picture.

Standard library and numpy only. No file input or output here (that is S003).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- constants
SCHEMA_VERSION = 1
N_LANDMARKS = 478  # MediaPipe Face Landmarker: 468 face + 10 iris
N_BLENDSHAPES = 52

LANDMARKS_SHAPE = (N_LANDMARKS, 3)
BLENDSHAPES_SHAPE = (N_BLENDSHAPES,)
HEAD_MATRIX_SHAPE = (4, 4)

XY_RANGE = (-1.0, 2.0)  # normalised x, y (a fraction of the image width / height)
Z_RANGE = (-2.0, 2.0)
BLEND_RANGE = (0.0, 1.0)
IMAGE_SIZE_RANGE = (1, 10000)

_F32 = np.dtype(np.float32)
_F64 = np.dtype(np.float64)
_BOOL = np.dtype(np.bool_)
_I32 = np.dtype(np.int32)

# Bytes of one frame in the stacked form: t, face_found and the three arrays.
BYTES_PER_FRAME = (
    math.prod(LANDMARKS_SHAPE) * _F32.itemsize
    + math.prod(BLENDSHAPES_SHAPE) * _F32.itemsize
    + math.prod(HEAD_MATRIX_SHAPE) * _F32.itemsize
    + _F64.itemsize
    + _BOOL.itemsize
)

STACK_KEYS = ("t", "face_found", "landmarks", "blendshapes", "head_matrix", "image_w", "image_h")
_ARRAY_FIELDS = (
    ("landmarks", LANDMARKS_SHAPE),
    ("blendshapes", BLENDSHAPES_SHAPE),
    ("head_matrix", HEAD_MATRIX_SHAPE),
)


class FaceObsError(ValueError):
    """A record or a stack was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True, eq=False)
class FaceObs:
    """One frame. Build it with make_faceobs() or no_face(), not directly."""

    t: float
    face_found: bool
    landmarks: np.ndarray    # float32 (478, 3), read-only
    blendshapes: np.ndarray  # float32 (52,), read-only
    head_matrix: np.ndarray  # float32 (4, 4), read-only
    image_w: int
    image_h: int


# ---------------------------------------------------------------- single-frame checks
def _check_time(t) -> float:
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, numbers.Real):
        raise FaceObsError("time", f"t must be a real number, got {type(t).__name__}")
    tf = float(t)
    if not math.isfinite(tf) or tf < 0.0:
        raise FaceObsError("time", f"t must be finite and >= 0, got {tf!r}")
    return tf


def _check_image_size(name: str, v) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise FaceObsError("image_size", f"{name} must be an integer, got {type(v).__name__}")
    lo, hi = IMAGE_SIZE_RANGE
    if not lo <= int(v) <= hi:
        raise FaceObsError("image_size", f"{name} must be in [{lo}, {hi}], got {int(v)}")
    return int(v)


def _as_float_array(name: str, x) -> np.ndarray:
    try:
        a = np.asarray(x)
    except ValueError as e:  # ragged nested lists
        raise FaceObsError("shape", f"{name} is not a regular array: {e}") from None
    if a.dtype.kind != "f":
        raise FaceObsError("dtype", f"{name} must have a float dtype, got {a.dtype}")
    return a


def _readonly_f32(a: np.ndarray) -> np.ndarray:
    # np.array copies, so the record never shares memory with the caller.
    with np.errstate(over="ignore"):  # 1e300 -> inf is caught by the "nan" check
        out = np.array(a, dtype=np.float32, copy=True)
    out.setflags(write=False)
    return out


def _check_ranges(lm: np.ndarray, bs: np.ndarray) -> None:
    xy, z = lm[..., :2], lm[..., 2]
    if xy.min() < XY_RANGE[0] or xy.max() > XY_RANGE[1]:
        raise FaceObsError("range", f"landmark x/y outside {XY_RANGE} (pixels instead of normalised?)")
    if z.min() < Z_RANGE[0] or z.max() > Z_RANGE[1]:
        raise FaceObsError("range", f"landmark z outside {Z_RANGE}")
    if bs.min() < BLEND_RANGE[0] or bs.max() > BLEND_RANGE[1]:
        raise FaceObsError("range", f"blendshape outside {BLEND_RANGE}")


def make_faceobs(t, landmarks, blendshapes, head_matrix, image_w, image_h) -> FaceObs:
    """A frame with a face. Validates (order: time, image_size, dtype, shape, nan, range),
    converts to float32, copies, and makes the arrays read-only."""
    tf = _check_time(t)
    w = _check_image_size("image_w", image_w)
    h = _check_image_size("image_h", image_h)
    raw = {"landmarks": landmarks, "blendshapes": blendshapes, "head_matrix": head_matrix}
    arrs = {name: _as_float_array(name, raw[name]) for name, _ in _ARRAY_FIELDS}
    for name, shape in _ARRAY_FIELDS:
        if arrs[name].shape != shape:
            raise FaceObsError("shape", f"{name} must have shape {shape}, got {arrs[name].shape}")
    f32 = {name: _readonly_f32(arrs[name]) for name, _ in _ARRAY_FIELDS}
    for name, _ in _ARRAY_FIELDS:
        if not np.isfinite(f32[name]).all():
            raise FaceObsError("nan", f"{name} has NaN or inf (after the cast to float32)")
    _check_ranges(f32["landmarks"], f32["blendshapes"])
    return FaceObs(tf, True, f32["landmarks"], f32["blendshapes"], f32["head_matrix"], w, h)


def _nan_readonly(shape) -> np.ndarray:
    a = np.full(shape, np.nan, dtype=np.float32)
    a.setflags(write=False)
    return a


def no_face(t, image_w, image_h) -> FaceObs:
    """A frame without a face: the three arrays are all NaN (never zeros)."""
    tf = _check_time(t)
    w = _check_image_size("image_w", image_w)
    h = _check_image_size("image_h", image_h)
    return FaceObs(tf, False, _nan_readonly(LANDMARKS_SHAPE), _nan_readonly(BLENDSHAPES_SHAPE),
                   _nan_readonly(HEAD_MATRIX_SHAPE), w, h)


# ---------------------------------------------------------------- stacking
def _check_time_order(t: np.ndarray) -> None:
    if t.shape[0] > 1 and np.any(t[1:] <= t[:-1]):
        i = int(np.argmax(t[1:] <= t[:-1])) + 1
        raise FaceObsError("time_order", f"times must strictly increase: t[{i}]={t[i]!r} <= t[{i-1}]={t[i-1]!r}")


def stack(obs_list) -> dict:
    """A list of FaceObs -> a dict of stacked arrays (see STACK_KEYS).
    The result is validated with the same checks as unstack()."""
    obs_list = list(obs_list)
    if not obs_list:
        raise FaceObsError("empty", "stack() needs at least one record")
    for i, o in enumerate(obs_list):
        if not isinstance(o, FaceObs):
            raise FaceObsError("type", f"item {i} is {type(o).__name__}, not FaceObs")
    t = np.array([o.t for o in obs_list], dtype=np.float64)
    _check_time_order(t)
    w0, h0 = obs_list[0].image_w, obs_list[0].image_h
    for i, o in enumerate(obs_list):
        if (o.image_w, o.image_h) != (w0, h0):
            raise FaceObsError("mixed_image_size",
                               f"item {i} is {o.image_w}x{o.image_h}, item 0 is {w0}x{h0}")
    arrays = {
        "t": t,
        "face_found": np.array([o.face_found for o in obs_list], dtype=np.bool_),
        "landmarks": np.stack([o.landmarks for o in obs_list]).astype(np.float32, copy=False),
        "blendshapes": np.stack([o.blendshapes for o in obs_list]).astype(np.float32, copy=False),
        "head_matrix": np.stack([o.head_matrix for o in obs_list]).astype(np.float32, copy=False),
        "image_w": np.array(w0, dtype=np.int32),
        "image_h": np.array(h0, dtype=np.int32),
    }
    _validate_stacked(arrays)  # records built by hand (not via make_faceobs) are caught here
    return arrays


_STACK_DTYPES = {"t": _F64, "face_found": _BOOL, "landmarks": _F32, "blendshapes": _F32,
                 "head_matrix": _F32, "image_w": _I32, "image_h": _I32}


def _validate_stacked(arrays) -> int:
    """Check a stacked dict (not trusted: it may come from a file). Returns N."""
    if not isinstance(arrays, dict):
        raise FaceObsError("type", f"expected a dict, got {type(arrays).__name__}")
    keys = set(arrays)
    if keys != set(STACK_KEYS):
        raise FaceObsError("type", f"missing keys {sorted(set(STACK_KEYS) - keys)}, "
                                   f"extra keys {sorted(map(str, keys - set(STACK_KEYS)))}")
    for k in STACK_KEYS:
        if not isinstance(arrays[k], np.ndarray):
            raise FaceObsError("type", f"{k} must be a numpy array, got {type(arrays[k]).__name__}")
        if arrays[k].dtype != _STACK_DTYPES[k]:
            raise FaceObsError("dtype", f"{k} must be {_STACK_DTYPES[k]}, got {arrays[k].dtype}")
    t = arrays["t"]
    if t.ndim != 1:
        raise FaceObsError("shape", f"t must have shape (N,), got {t.shape}")
    n = t.shape[0]
    expected = {"t": (n,), "face_found": (n,), "landmarks": (n, *LANDMARKS_SHAPE),
                "blendshapes": (n, *BLENDSHAPES_SHAPE), "head_matrix": (n, *HEAD_MATRIX_SHAPE),
                "image_w": (), "image_h": ()}
    for k in STACK_KEYS:
        if arrays[k].shape != expected[k]:
            raise FaceObsError("shape", f"{k} must have shape {expected[k]}, got {arrays[k].shape}")
    if n == 0:
        raise FaceObsError("empty", "the stack has no frames")
    if not np.isfinite(t).all() or (t < 0).any():
        raise FaceObsError("time", "every t must be finite and >= 0")
    _check_image_size("image_w", int(arrays["image_w"]))
    _check_image_size("image_h", int(arrays["image_h"]))
    _check_time_order(t)
    ff = arrays["face_found"]
    face = {k: arrays[k][ff] for k, _ in _ARRAY_FIELDS}
    for k, _ in _ARRAY_FIELDS:
        if not np.isfinite(face[k]).all():
            raise FaceObsError("nan", f"a face row of {k} has NaN or inf")
    if ff.any():
        _check_ranges(face["landmarks"], face["blendshapes"])
    for k, _ in _ARRAY_FIELDS:
        if not np.isnan(arrays[k][~ff]).all():
            raise FaceObsError("no_face_not_nan", f"a no-face row of {k} has a number (must be NaN)")
    return n


def validate_stacked(arrays) -> int:
    """Public check of a stacked dict (same checks as unstack). Returns N."""
    return _validate_stacked(arrays)


def unstack(arrays) -> list:
    """Stacked arrays -> a list of FaceObs. Everything is validated again."""
    n = _validate_stacked(arrays)
    w, h = int(arrays["image_w"]), int(arrays["image_h"])
    out = []
    for i in range(n):
        ti = float(arrays["t"][i])  # float64 -> Python float: exact
        if bool(arrays["face_found"][i]):
            out.append(make_faceobs(ti, arrays["landmarks"][i], arrays["blendshapes"][i],
                                    arrays["head_matrix"][i], w, h))
        else:
            out.append(no_face(ti, w, h))
    return out
