"""Pure conversion from a MediaPipe Face Landmarker result to a FaceObs (S008).

No mediapipe import here: the result is only read through its attributes
(`face_landmarks`, `face_blendshapes`, `facial_transformation_matrixes`),
so the tests can build fake results by hand.

Standard library, numpy and the project only. No file is opened here.
"""
from __future__ import annotations

import math

import numpy as np

from .faceobs import (
    IMAGE_SIZE_RANGE,
    N_BLENDSHAPES,
    N_LANDMARKS,
    FaceObsError,
    make_faceobs,
    no_face,
)
from .tracker import TrackerError


def pick_face(face_landmarks) -> int:
    """Index of the face with the largest box area (x span * y span).

    Ties -> the lowest index. No faces -> -1.
    """
    best_i, best_area = -1, -1.0
    for i, pts in enumerate(face_landmarks):
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs else 0.0
        if area > best_area:  # strict ">" keeps the lowest index on a tie
            best_i, best_area = i, area
    return best_i


def video_ms(t_s, last_ms) -> int:
    """Frame time in whole ms for VIDEO mode; always after `last_ms`."""
    try:
        tf = float(t_s)
    except (TypeError, ValueError) as e:
        raise TrackerError("time", f"t_s must be a number, got {t_s!r}") from e
    if not math.isfinite(tf) or tf < 0:
        raise TrackerError("time", f"t_s must be finite and >= 0, got {t_s!r}")
    ms = int(round(tf * 1000.0))
    if last_ms is not None and ms <= last_ms:
        ms = last_ms + 1
    return ms


def check_frame(frame) -> tuple:
    """A frame must be an (h, w, 3) uint8 array with h and w in IMAGE_SIZE_RANGE. Returns (h, w)."""
    if not isinstance(frame, np.ndarray):
        raise TrackerError("frame", f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        raise TrackerError("frame", f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise TrackerError("frame", f"frame shape must be (h, w, 3), got {frame.shape}")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    lo, hi = IMAGE_SIZE_RANGE
    if not (lo <= h <= hi and lo <= w <= hi):
        raise TrackerError("frame", f"frame size {h}x{w} outside {IMAGE_SIZE_RANGE}")
    return h, w


def to_project_convention(landmarks, head_matrix) -> tuple:
    """Convert MediaPipe geometry to the project convention. Today: the identity (copies).

    UNMEASURED: the convention is from memory (DECISIONS line 45); S008 morning checks decide.
    """
    return np.array(landmarks, copy=True), np.array(head_matrix, copy=True)


def rotation_report(head_matrix) -> dict:
    """How close the top-left 3x3 of the head matrix is to a rotation.

    s = mean singular value; R_n = M / s; orth_err = max |R_n^T R_n - I|.
    """
    m = np.asarray(head_matrix, dtype=np.float64)[:3, :3]
    s = float(np.linalg.svd(m, compute_uv=False).mean())
    rn = m / s
    orth_err = float(np.max(np.abs(rn.T @ rn - np.eye(3))))
    return {"scale": s, "det": float(np.linalg.det(m)), "orth_err": orth_err}


def _blendshape_scores(cats) -> np.ndarray:
    cats = list(cats)
    if len(cats) != N_BLENDSHAPES:
        raise TrackerError("blendshapes", f"{len(cats)} categories, need {N_BLENDSHAPES}")
    idx = [c.index for c in cats]
    bs = np.empty(N_BLENDSHAPES, dtype=np.float64)
    if all(isinstance(i, int) and not isinstance(i, bool) for i in idx):
        if sorted(idx) != list(range(N_BLENDSHAPES)):
            raise TrackerError("blendshapes", "the indexes are not a permutation of 0..51")
        for c in cats:
            bs[c.index] = c.score
    elif all(i is None for i in idx):
        for pos, c in enumerate(cats):
            bs[pos] = c.score
    else:
        raise TrackerError("blendshapes", "the indexes mix None and numbers (or are not ints)")
    return bs


def result_to_faceobs(result, t_s, image_w, image_h) -> tuple:
    """(FaceObs, n_faces) from one Face Landmarker result. The largest face is used."""
    faces = list(result.face_landmarks or [])
    n = len(faces)
    if n == 0:
        return no_face(t_s, image_w, image_h), 0
    i = pick_face(faces)

    lm = np.array([[p.x, p.y, p.z] for p in faces[i]], dtype=np.float64)
    if lm.shape != (N_LANDMARKS, 3):
        raise TrackerError("landmarks", f"{len(faces[i])} points")

    all_bs = list(result.face_blendshapes or [])
    if len(all_bs) != n:
        raise TrackerError("blendshapes", f"{len(all_bs)} blendshape lists for {n} faces")
    bs = _blendshape_scores(all_bs[i])

    mats = list(result.facial_transformation_matrixes or [])
    if len(mats) != n:
        raise TrackerError("matrix", f"{len(mats)} matrices for {n} faces")
    hm = np.asarray(mats[i], dtype=np.float64)
    if hm.shape != (4, 4):
        raise TrackerError("matrix", f"matrix shape {hm.shape}, need (4, 4)")

    lm, hm = to_project_convention(lm, hm)
    try:
        obs = make_faceobs(t_s, lm, bs, hm, image_w, image_h)
    except FaceObsError as e:
        raise TrackerError("convert", f"{e.code}: {e}") from e
    return obs, n
