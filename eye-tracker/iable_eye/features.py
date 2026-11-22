"""Eye features that do not change with distance or head roll (S012).

Every length is measured in image pixels first: x_px = x * W, y_px = y * H.
Then two lengths are divided. Both shrink by the same factor f / d when the
person sits farther away, so the ratio stays the same.

Axes: u goes from the right eye's outer corner to the left eye's outer corner
(unit length). n = (-u_y, u_x) points image-down when u = (1, 0).

Per eye: m = midpoint of the two corners, w = corner distance,
c = centroid of the 4 iris ring points (the ring is used as a set).
    h = (c - m) . u / w,   v = (c - m) . n / w,   lid = |upper - lower| / w
iris_px = mean over the 2 eyes of 2 * max_i |r_i - c|.
dist_mm = focal_px * iris_mm / iris_px.

focal_px = 600 is an assumption (ROADMAP section 3); it stays a parameter.
This file never imports the simulator.
"""

import math

import numpy as np

from iable_eye import faceobs
from iable_eye import landmark_ids as L

FEATURE_NAMES = ("h_r", "v_r", "lid_r", "h_l", "v_l", "lid_l",
                 "iris_px", "dist_mm", "yaw_deg", "pitch_deg", "roll_deg")
IRIS_MM = 11.7
DEFAULT_FOCAL_PX = 600.0

_USED = (L.R_EYE_OUTER, L.R_EYE_INNER, L.R_LID_UPPER, L.R_LID_LOWER,
         L.L_EYE_OUTER, L.L_EYE_INNER, L.L_LID_UPPER, L.L_LID_LOWER) + L.R_IRIS[1:] + L.L_IRIS[1:]


class FeatureError(ValueError):
    """A refused input. .code is one of "shape", "nan", "degenerate", "units"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _positive(name: str, v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise FeatureError("units", f"{name} must be a number, got {type(v).__name__}") from None
    if not math.isfinite(f) or f <= 0.0:
        raise FeatureError("units", f"{name} must be finite and > 0, got {v!r}")
    return f


def head_angles(head_matrix) -> tuple:
    """(yaw, pitch, roll) in degrees from the 3x3 rotation part of a 3x3 or 4x4 matrix.

    Order: R = Ry(yaw) Rx(pitch) Rz(roll), the same as sim_core.rotation_matrix.
    Then R[1,2] = -sin(pitch), R[0,2] / R[2,2] = tan(yaw), R[1,0] / R[1,1] = tan(roll).
    """
    m = np.asarray(head_matrix, dtype=np.float64)
    if m.shape not in ((3, 3), (4, 4)):
        raise FeatureError("shape", f"head_matrix must be (3, 3) or (4, 4), got {m.shape}")
    r = m[:3, :3]
    if not np.isfinite(r).all():
        raise FeatureError("nan", "head_matrix has NaN or inf")
    pitch = math.asin(max(-1.0, min(1.0, -float(r[1, 2]))))
    yaw = math.atan2(float(r[0, 2]), float(r[2, 2]))
    roll = math.atan2(float(r[1, 0]), float(r[1, 1]))
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _eye(p, outer, inner, upper, lower, ring, u, nrm):
    """(h, v, lid, iris diameter px) for one eye. p is (478, 2) in px."""
    a, b = p[outer], p[inner]
    w = float(np.linalg.norm(b - a))
    if w < 1.0:
        raise FeatureError("degenerate", f"eye width {w:.3g} px < 1 px")
    m = (a + b) / 2.0
    r = p[list(ring)]
    c = r.mean(axis=0)
    d = c - m
    h = float(d @ u) / w
    v = float(d @ nrm) / w
    lid = float(np.linalg.norm(p[upper] - p[lower])) / w
    diam = 2.0 * float(np.linalg.norm(r - c, axis=1).max())
    return h, v, lid, diam


def frame_features(landmarks, image_w, image_h, head_matrix,
                   focal_px=DEFAULT_FOCAL_PX, iris_mm=IRIS_MM) -> np.ndarray:
    """One frame -> float64 (11,) in the order of FEATURE_NAMES."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != faceobs.LANDMARKS_SHAPE:
        raise FeatureError("shape", f"landmarks must be {faceobs.LANDMARKS_SHAPE}, got {lm.shape}")
    hm = np.asarray(head_matrix, dtype=np.float64)
    if hm.shape != (4, 4):
        raise FeatureError("shape", f"head_matrix must be (4, 4), got {hm.shape}")
    f = _positive("focal_px", focal_px)
    iris = _positive("iris_mm", iris_mm)
    wpx = _positive("image_w", image_w)
    hpx = _positive("image_h", image_h)
    used = lm[list(_USED), :2]
    if not np.isfinite(used).all():
        raise FeatureError("nan", "a used landmark has NaN or inf")
    p = np.full((lm.shape[0], 2), np.nan, dtype=np.float64)
    p[list(_USED)] = used * np.array([wpx, hpx])               # E2: pixels first
    axis = p[L.L_EYE_OUTER] - p[L.R_EYE_OUTER]
    na = float(np.linalg.norm(axis))
    if na < 1.0:
        raise FeatureError("degenerate", f"outer corners {na:.3g} px apart")
    u = axis / na
    nrm = np.array([-u[1], u[0]])
    hr, vr, lr, dr = _eye(p, L.R_EYE_OUTER, L.R_EYE_INNER, L.R_LID_UPPER, L.R_LID_LOWER,
                          L.R_IRIS[1:], u, nrm)
    hl, vl, ll, dl = _eye(p, L.L_EYE_OUTER, L.L_EYE_INNER, L.L_LID_UPPER, L.L_LID_LOWER,
                          L.L_IRIS[1:], u, nrm)
    iris_px = (dr + dl) / 2.0
    if iris_px < 0.5:
        raise FeatureError("degenerate", f"iris_px {iris_px:.3g} < 0.5 px")
    yaw, pitch, roll = head_angles(hm)
    return np.array([hr, vr, lr, hl, vl, ll, iris_px, f * iris / iris_px, yaw, pitch, roll],
                    dtype=np.float64)


def session_features(arrays, focal_px=DEFAULT_FOCAL_PX) -> np.ndarray:
    """A stacked session dict -> float64 (n, 11).
    A row of NaN where face_found is False or the frame is "degenerate"."""
    n = faceobs.validate_stacked(arrays)
    w, h = int(arrays["image_w"]), int(arrays["image_h"])
    out = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for i in range(n):
        if not bool(arrays["face_found"][i]):
            continue
        try:
            out[i] = frame_features(arrays["landmarks"][i], w, h, arrays["head_matrix"][i],
                                    focal_px=focal_px)
        except FeatureError as e:
            if e.code != "degenerate":
                raise
    return out
