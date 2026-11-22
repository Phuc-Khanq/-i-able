"""The simulator core: a 3D person, a pinhole camera, one noise-free FaceObs.

Everything here is pure geometry: the same inputs always give the same bytes.
Nothing is stored between calls and nothing is written anywhere.
Every number this file gives is (simulated) geometry, never a measurement.

Frames (all lengths in mm, angles in degrees at the interface):
- Camera frame: origin at the lens; +x = right in the unmirrored image = the
  person's left; +y = up; +z = out of the back of the camera. The person is at
  z < 0, and the depth of a point is d = -Z.
- Head frame: origin at the midpoint of the two eyeball centres. Facing the
  camera, its axes equal the camera's. Pose: P_C = R P_h + T, R = Ry(yaw) Rx(pitch) Rz(roll).
- Screen: in the plane Z = 0, the lens in the middle, cam_above_top_mm above the top edge.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from iable_eye.faceobs import N_BLENDSHAPES, N_LANDMARKS, FaceObs, _readonly_f32, make_faceobs
from iable_eye import landmark_ids as L


class SimError(ValueError):
    """The simulator refused an input. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


# ---------------------------------------------------------------- inputs
@dataclass(frozen=True)
class SimPerson:
    ipd_mm: float = 63.0
    eye_width_mm: float = 30.0
    eyeball_radius_mm: float = 12.0
    iris_diameter_mm: float = 11.7
    kappa_h_deg: float = 5.0
    kappa_v_deg: float = 1.5
    lid_open: float = 1.0


@dataclass(frozen=True)
class SimCamera:
    image_w_px: int = 640
    image_h_px: int = 480
    focal_px: float = 600.0


@dataclass(frozen=True)
class SimScreen:
    width_px: int = 1920
    height_px: int = 1080
    width_mm: float = 340.0
    cam_above_top_mm: float = 8.0


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tx_mm: float = 0.0
    ty_mm: float = 0.0
    tz_mm: float = -600.0


# (field, low, high), both ends allowed.
_PERSON_RANGES = (("ipd_mm", 54.0, 72.0), ("eye_width_mm", 26.0, 32.0),
                  ("eyeball_radius_mm", 11.0, 13.0), ("iris_diameter_mm", 10.0, 13.0),
                  ("kappa_h_deg", 0.0, 8.0), ("kappa_v_deg", 0.0, 3.0), ("lid_open", 0.0, 1.2))
_CAMERA_INT_RANGES = (("image_w_px", 100, 4000), ("image_h_px", 100, 4000))
_CAMERA_RANGES = (("focal_px", 100.0, 5000.0),)
_SCREEN_INT_RANGES = (("width_px", 100, 10000), ("height_px", 100, 10000))
_SCREEN_RANGES = (("width_mm", 100.0, 2000.0), ("cam_above_top_mm", 0.0, 100.0))
_POSE_RANGES = (("yaw_deg", -60.0, 60.0), ("pitch_deg", -45.0, 45.0), ("roll_deg", -45.0, 45.0),
                ("tx_mm", -500.0, 500.0), ("ty_mm", -500.0, 500.0), ("tz_mm", -2000.0, -200.0))

EYE_RANGE_DEG = 55.0  # largest angle between the optical axis and the head's +z


def _real(code: str, name: str, v) -> float:
    """A finite real number (a bool is refused) -> float, else SimError(code)."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise SimError(code, f"{name} must be a real number, got {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise SimError(code, f"{name} must be finite, got {f!r}")
    return f


def _check_obj(code: str, cls, obj, float_ranges, int_ranges=()) -> None:
    if not isinstance(obj, cls):
        raise SimError(code, f"expected {cls.__name__}, got {type(obj).__name__}")
    for name, lo, hi in int_ranges:
        v = getattr(obj, name)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise SimError(code, f"{name} must be an integer, got {type(v).__name__}")
        if not lo <= int(v) <= hi:
            raise SimError(code, f"{name} must be in [{lo}, {hi}], got {int(v)}")
    for name, lo, hi in float_ranges:
        f = _real(code, name, getattr(obj, name))
        if not lo <= f <= hi:
            raise SimError(code, f"{name} must be in [{lo}, {hi}], got {f!r}")


def check_person(p) -> None:
    _check_obj("person", SimPerson, p, _PERSON_RANGES)


def check_camera(c) -> None:
    _check_obj("camera", SimCamera, c, _CAMERA_RANGES, _CAMERA_INT_RANGES)


def check_screen(s) -> None:
    _check_obj("screen", SimScreen, s, _SCREEN_RANGES, _SCREEN_INT_RANGES)


def check_pose(p) -> None:
    _check_obj("pose", HeadPose, p, _POSE_RANGES)


def check_gaze(g) -> np.ndarray:
    """3 finite real numbers (mm, camera frame) -> float64 (3,), else SimError("gaze")."""
    if isinstance(g, (str, bytes)):
        raise SimError("gaze", "gaze_cam_mm must be 3 numbers, got a string")
    try:
        items = list(g)
    except TypeError:
        raise SimError("gaze", f"gaze_cam_mm must be 3 numbers, got {type(g).__name__}") from None
    if len(items) != 3:
        raise SimError("gaze", f"gaze_cam_mm must have 3 numbers, got {len(items)}")
    return np.array([_real("gaze", f"gaze_cam_mm[{i}]", v) for i, v in enumerate(items)],
                    dtype=np.float64)


# ---------------------------------------------------------------- frames and the camera
def rotation_matrix(yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
    """R = Ry(yaw) Rx(pitch) Rz(roll), float64 (3, 3). No range check (only finite)."""
    psi = math.radians(_real("pose", "yaw_deg", yaw_deg))
    theta = math.radians(_real("pose", "pitch_deg", pitch_deg))
    phi = math.radians(_real("pose", "roll_deg", roll_deg))
    cy, sy = math.cos(psi), math.sin(psi)
    cx, sx = math.cos(theta), math.sin(theta)
    cz, sz = math.cos(phi), math.sin(phi)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return ry @ rx @ rz


def _pose_rt(pose) -> tuple[np.ndarray, np.ndarray]:
    check_pose(pose)
    r = rotation_matrix(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    t = np.array([pose.tx_mm, pose.ty_mm, pose.tz_mm], dtype=np.float64)
    return r, t


def screen_px_to_cam_mm(sx_px, sy_px, screen) -> np.ndarray:
    """A screen point (px, from the top-left as the person sees it) -> camera frame (mm)."""
    check_screen(screen)
    sx = _real("screen", "sx_px", sx_px)
    sy = _real("screen", "sy_px", sy_px)
    p_mm = float(screen.width_mm) / float(screen.width_px)  # mm per px
    x = (screen.width_px / 2.0 - sx) * p_mm
    y = -(float(screen.cam_above_top_mm) + sy * p_mm)
    return np.array([x, y, 0.0], dtype=np.float64)


def _points(code: str, pts) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise SimError(code, f"points must have shape (n, 3), got {a.shape}")
    return a


def to_camera_mm(points_head_mm, pose) -> np.ndarray:
    """Head frame -> camera frame: P_C = R P_h + T. float64 (n, 3)."""
    r, t = _pose_rt(pose)
    ph = _points("pose", points_head_mm)
    return ph @ r.T + t


def project(points_cam_mm, camera, origin_depth_mm) -> np.ndarray:
    """Pinhole projection -> normalised (x_n, y_n, z_n), float64 (n, 3).
    u = cx + f X/d, v = cy - f Y/d, x_n = u/W, y_n = v/H, z_n = f (d - d_O) / (d_O W)."""
    check_camera(camera)
    d_o = _real("camera", "origin_depth_mm", origin_depth_mm)
    if d_o <= 0.0:
        raise SimError("camera", f"origin_depth_mm must be > 0, got {d_o!r}")
    pc = _points("camera", points_cam_mm)
    d = -pc[:, 2]
    if (d <= 0.0).any():
        raise SimError("camera", "a point is not in front of the camera (depth <= 0)")
    w, h, f = float(camera.image_w_px), float(camera.image_h_px), float(camera.focal_px)
    u = w / 2.0 + f * pc[:, 0] / d
    v = h / 2.0 - f * pc[:, 1] / d
    zn = f * (d - d_o) / (d_o * w)
    return np.stack([u / w, v / h, zn], axis=1)


def head_matrix(pose) -> np.ndarray:
    """4x4: upper-left R, last column T / 10 (cm), last row (0, 0, 0, 1). float64."""
    r, t = _pose_rt(pose)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = t / 10.0
    return m


# ---------------------------------------------------------------- the head points
_GOLDEN_ANGLE_RAD = 2.399963229728653
_COS_EYE_RANGE = math.cos(math.radians(EYE_RANGE_DEG))


def _shell_point(i: int) -> tuple[float, float, float]:
    """A placeholder shell point (NOT anatomy): a spiral on an ellipsoid."""
    q = (i + 0.5) / N_LANDMARKS
    th = math.acos(1.0 - q)
    ph = _GOLDEN_ANGLE_RAD * i
    return (70.0 * math.sin(th) * math.cos(ph),
            95.0 * math.sin(th) * math.sin(ph) - 20.0,
            85.0 * math.cos(th) - 50.0)


def _make_shell_mm() -> np.ndarray:
    """(478, 3) float64, read-only: the 455 shell points, made once by the same loop.
    The 23 named rows are 0 here; head_points_mm always overwrites them."""
    pts = np.zeros((N_LANDMARKS, 3), dtype=np.float64)
    named = set(L.NAMED)
    for i in range(N_LANDMARKS):
        if i not in named:
            pts[i] = _shell_point(i)
    pts.setflags(write=False)
    return pts


_SHELL_MM = _make_shell_mm()  # S043b: the shell never changes, so build it once


def _fixed_points(person) -> dict:
    h, w, o = person.ipd_mm / 2.0, person.eye_width_mm / 2.0, person.lid_open
    return {
        L.NOSE_TIP: (0.0, -35.0, 35.0), L.CHIN: (0.0, -95.0, 10.0), L.FOREHEAD: (0.0, 70.0, 15.0),
        L.FACE_RIGHT: (-70.0, -10.0, -50.0), L.FACE_LEFT: (70.0, -10.0, -50.0),
        L.R_EYE_OUTER: (-(h + w), 0.0, 8.0), L.R_EYE_INNER: (-(h - w), 0.0, 8.0),
        L.R_LID_UPPER: (-h, 5.0 * o, 12.0), L.R_LID_LOWER: (-h, -5.0 * o, 12.0),
        L.L_EYE_OUTER: (h + w, 0.0, 8.0), L.L_EYE_INNER: (h - w, 0.0, 8.0),
        L.L_LID_UPPER: (h, 5.0 * o, 12.0), L.L_LID_LOWER: (h, -5.0 * o, 12.0),
    }


def _iris_points(person, r, t, gaze, s: float) -> np.ndarray:
    """5 iris points (head frame) of one eye; s = +1 left eye, -1 right eye."""
    e_h = np.array([s * person.ipd_mm / 2.0, 0.0, 0.0], dtype=np.float64)
    e_c = r @ e_h + t
    los = gaze - e_c
    n = float(np.linalg.norm(los))
    if n == 0.0:
        raise SimError("gaze", "the gaze point is the eyeball centre")
    g = r.T @ (los / n)                                     # line of sight, head frame
    alpha = math.atan2(g[0], g[2])
    beta = math.asin(max(-1.0, min(1.0, float(g[1]))))
    alpha_o = alpha + s * math.radians(person.kappa_h_deg)  # optical axis: to the temple
    beta_o = beta - math.radians(person.kappa_v_deg)        # and down
    o = np.array([math.cos(beta_o) * math.sin(alpha_o), math.sin(beta_o),
                  math.cos(beta_o) * math.cos(alpha_o)], dtype=np.float64)
    if o[2] < _COS_EYE_RANGE:
        side = "left" if s > 0 else "right"
        raise SimError("eye_range", f"the {side} eye would turn more than {EYE_RANGE_DEG} deg")
    c = e_h + person.eyeball_radius_mm * o
    a = np.cross(np.array([0.0, 1.0, 0.0]), o)
    a /= np.linalg.norm(a)
    b = np.cross(o, a)
    rho = person.iris_diameter_mm / 2.0
    return np.stack([c, c + rho * a, c + rho * b, c - rho * a, c - rho * b])


def head_points_mm(person, pose, gaze_cam_mm) -> np.ndarray:
    """All 478 points in the head frame, float64 (478, 3).
    13 fixed face points, 2 x 5 iris points that follow the gaze, and 455
    placeholder shell points (not anatomy)."""
    check_person(person)
    r, t = _pose_rt(pose)
    gaze = check_gaze(gaze_cam_mm)
    pts = _SHELL_MM.copy()
    for i, p in _fixed_points(person).items():
        pts[i] = p
    pts[list(L.L_IRIS)] = _iris_points(person, r, t, gaze, +1.0)
    pts[list(L.R_IRIS)] = _iris_points(person, r, t, gaze, -1.0)
    return pts


# ---------------------------------------------------------------- one frame
def simulate_frame(t, person, camera, pose, gaze_cam_mm) -> FaceObs:
    """One noise-free FaceObs. Checks person, camera, pose, gaze (in this order).
    A FaceObsError from make_faceobs (for example "range") goes up unchanged."""
    check_person(person)
    check_camera(camera)
    check_pose(pose)
    check_gaze(gaze_cam_mm)
    pts_h = head_points_mm(person, pose, gaze_cam_mm)
    pts_c = to_camera_mm(pts_h, pose)
    lm = project(pts_c, camera, -float(pose.tz_mm))  # d_O = -t_z
    blend = np.zeros(N_BLENDSHAPES, dtype=np.float64)
    return make_faceobs(t, lm, blend, head_matrix(pose), camera.image_w_px, camera.image_h_px)


def _simulate_frame_fast(t, person, camera, pose, gaze_cam_mm) -> FaceObs:
    """PRIVATE (S043b): the same numbers as simulate_frame, for simulate_session only.
    The caller checks person and camera once per session. Here: pose and gaze are checked
    once (not 4 times), R and T are made once (not 3 times), and the record is built
    without make_faceobs. faceobs.stack() validates the whole session at the end (E4)."""
    r, tv = _pose_rt(pose)
    gaze = check_gaze(gaze_cam_mm)
    pts_h = _SHELL_MM.copy()                                   # = head_points_mm
    for i, p in _fixed_points(person).items():
        pts_h[i] = p
    pts_h[list(L.L_IRIS)] = _iris_points(person, r, tv, gaze, +1.0)
    pts_h[list(L.R_IRIS)] = _iris_points(person, r, tv, gaze, -1.0)
    pc = pts_h @ r.T + tv                                      # = to_camera_mm
    d_o = -float(pose.tz_mm)                                   # = project, d_O = -t_z
    if d_o <= 0.0:
        raise SimError("camera", f"origin_depth_mm must be > 0, got {d_o!r}")
    d = -pc[:, 2]
    if (d <= 0.0).any():
        raise SimError("camera", "a point is not in front of the camera (depth <= 0)")
    w, h, f = float(camera.image_w_px), float(camera.image_h_px), float(camera.focal_px)
    u = w / 2.0 + f * pc[:, 0] / d
    v = h / 2.0 - f * pc[:, 1] / d
    zn = f * (d - d_o) / (d_o * w)
    lm = np.stack([u / w, v / h, zn], axis=1)
    m = np.eye(4, dtype=np.float64)                            # = head_matrix
    m[:3, :3] = r
    m[:3, 3] = tv / 10.0
    blend = np.zeros(N_BLENDSHAPES, dtype=np.float64)
    return _record_unchecked(t, lm, blend, m, camera.image_w_px, camera.image_h_px)


def _record_unchecked(t, landmarks, blendshapes, head_matrix, image_w, image_h) -> FaceObs:
    """PRIVATE (S043b): make_faceobs without its checks; the same float32 copies, read-only.
    Only for the simulate_session chain: faceobs.stack() validates the whole session (E4)."""
    return FaceObs(float(t), True, _readonly_f32(np.asarray(landmarks)), _readonly_f32(np.asarray(blendshapes)),
                   _readonly_f32(np.asarray(head_matrix)), int(image_w), int(image_h))
