"""Event knobs of the simulator: blinks, glare bursts, posture drift, translation leak.

Every number is (simulated). Events live on the nominal frame grid tau_k = k / fps,
the same index k as the S005 noise.
Randomness comes only from the generators made in make_event_rngs (4 streams,
children 3-6 of the seed; children 0-2 belong to S005's make_rngs).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from iable_eye import landmark_ids as L
from iable_eye.faceobs import FaceObs, make_faceobs
from iable_eye.sim_core import HeadPose, SimError, rotation_matrix
from iable_eye.sim_env import _check_frames, angles_from_matrix


def _is_real(v) -> bool:
    return (not isinstance(v, (bool, np.bool_)) and isinstance(v, numbers.Real)
            and math.isfinite(float(v)))


def _is_int(v) -> bool:
    return not isinstance(v, (bool, np.bool_)) and isinstance(v, int)


# ---------------------------------------------------------------- knobs
@dataclass(frozen=True)
class EventKnobs:
    blink_rate_per_min: float = 0.0     # 0-30; 0 = no blinks
    blink_min_ms: float = 100.0         # 50-500
    blink_max_ms: float = 400.0         # 50-500, and >= blink_min_ms
    glare_rate_per_min: float = 0.0     # 0-20; 0 = no glare
    glare_min_s: float = 0.2            # 0.05-2
    glare_max_s: float = 1.0            # 0.05-2, and >= glare_min_s
    glare_gain: float = 4.0             # 1-8; iris noise SD is multiplied by this in a burst
    drift_deg_per_20s: float = 0.0      # 0-4; SD of each angle's random walk after 20 s
    drift_mm_per_20s: float = 0.0       # 0-40; SD of each translation's random walk after 20 s
    leak_yaw_deg_per_mm: float = 0.0    # -0.05 to 0.05; reported yaw += this * (Tx - T0x)
    leak_pitch_deg_per_mm: float = 0.0  # -0.05 to 0.05; reported pitch += this * (Ty - T0y)


_EVENT_RANGES = (("blink_rate_per_min", 0.0, 30.0), ("blink_min_ms", 50.0, 500.0),
                 ("blink_max_ms", 50.0, 500.0), ("glare_rate_per_min", 0.0, 20.0),
                 ("glare_min_s", 0.05, 2.0), ("glare_max_s", 0.05, 2.0),
                 ("glare_gain", 1.0, 8.0), ("drift_deg_per_20s", 0.0, 4.0),
                 ("drift_mm_per_20s", 0.0, 40.0), ("leak_yaw_deg_per_mm", -0.05, 0.05),
                 ("leak_pitch_deg_per_mm", -0.05, 0.05))


def check_event_knobs(k) -> None:
    if not isinstance(k, EventKnobs):
        raise SimError("events", f"knobs must be an EventKnobs, got {type(k).__name__}")
    for name, lo, hi in _EVENT_RANGES:
        v = getattr(k, name)
        if not _is_real(v):
            raise SimError("events", f"{name} must be a finite real number, got {v!r}")
        if not lo <= float(v) <= hi:
            raise SimError("events", f"{name} must be in [{lo}, {hi}], got {float(v)}")
    if float(k.blink_min_ms) > float(k.blink_max_ms):
        raise SimError("events", "blink_min_ms must be <= blink_max_ms")
    if float(k.glare_min_s) > float(k.glare_max_s):
        raise SimError("events", "glare_min_s must be <= glare_max_s")


def make_event_rngs(seed):
    """-> (rng_blink, rng_glare, rng_glare_noise, rng_drift): children 3-6 of the seed."""
    if not _is_int(seed) or not 0 <= seed < 2 ** 53:
        raise SimError("seed", f"seed must be an int in [0, 2**53), got {seed!r}")
    ss = np.random.SeedSequence(seed).spawn(7)
    return tuple(np.random.default_rng(s) for s in ss[3:])


# ---------------------------------------------------------------- bursts and the walk
def _bursts(total_s, rate_per_min, dmin_s, dmax_s, rng):
    mu = 60.0 / rate_per_min - (dmin_s + dmax_s) / 2.0   # mean gap from a burst's end to the next onset
    onsets, durs, end = [], [], 0.0
    while True:
        onset = end - mu * math.log1p(-rng.random())      # exponential gap, mean mu
        if onset >= total_s:
            break
        dur = dmin_s + (dmax_s - dmin_s) * rng.random()
        onsets.append(onset)
        durs.append(dur)
        end = onset + dur
    return np.array(onsets, dtype=np.float64), np.array(durs, dtype=np.float64)


def random_walk(n_steps, n_series, sd_per_20s, fps, rng) -> np.ndarray:
    """float64 (n_steps, n_series): row 0 is 0; after 20 s the SD is sd_per_20s."""
    if not _is_int(n_steps) or n_steps < 1:
        raise SimError("walk", f"n_steps must be an int >= 1, got {n_steps!r}")
    if not _is_int(n_series) or n_series < 1:
        raise SimError("walk", f"n_series must be an int >= 1, got {n_series!r}")
    if not _is_real(sd_per_20s) or float(sd_per_20s) < 0.0:
        raise SimError("walk", f"sd_per_20s must be a finite real >= 0, got {sd_per_20s!r}")
    if not _is_real(fps) or float(fps) <= 0.0:
        raise SimError("walk", f"fps must be a finite real > 0, got {fps!r}")
    x = np.zeros((n_steps, n_series), dtype=np.float64)
    if n_steps >= 2:
        e = np.asarray(rng.standard_normal((n_steps - 1, n_series)), dtype=np.float64)
        step = float(sd_per_20s) * math.sqrt(1.0 / (20.0 * float(fps)))
        x[1:] = np.cumsum(step * e, axis=0)
    return x


# ---------------------------------------------------------------- the schedule
@dataclass(frozen=True, eq=False)
class EventSchedule:
    n_nominal: int
    fps: float
    closure: np.ndarray            # float64 (N,): blink closure c in [0, 1]
    blink: np.ndarray              # bool (N,): the frame is inside a blink (truth flag)
    blink_onsets_s: np.ndarray     # float64, seconds from tau_0
    blink_durations_s: np.ndarray
    glare_gain: np.ndarray         # float64 (N,): 1.0, or glare_gain inside a burst
    glare: np.ndarray              # bool (N,): the frame is inside a glare burst (truth flag)
    glare_onsets_s: np.ndarray
    glare_durations_s: np.ndarray
    glare_z: object                # None, or float64 (N, 10, 2) standard normal draws
    drift: object                  # None, or float64 (N, 6): yaw, pitch, roll (deg), tx, ty, tz (mm)


def event_schedule(n_nominal, fps, knobs, rngs) -> EventSchedule:
    check_event_knobs(knobs)
    if not _is_int(n_nominal) or not 2 <= n_nominal <= 216000:
        raise SimError("schedule", f"n_nominal must be an int in [2, 216000], got {n_nominal!r}")
    if not _is_real(fps) or not 5.0 <= float(fps) <= 60.0:
        raise SimError("schedule", f"fps must be a finite real in [5, 60], got {fps!r}")
    if not isinstance(rngs, (tuple, list)) or len(rngs) != 4:
        raise SimError("schedule", "rngs must be a tuple or list of 4 generators")
    n, fps = int(n_nominal), float(fps)
    tau = np.arange(n) / fps
    total = n / fps
    empty = np.zeros(0, dtype=np.float64)

    closure = np.zeros(n, dtype=np.float64)
    blink = np.zeros(n, dtype=bool)
    b_on, b_du = empty, empty
    if float(knobs.blink_rate_per_min) > 0.0:
        b_on, b_du = _bursts(total, float(knobs.blink_rate_per_min),
                             float(knobs.blink_min_ms) / 1000.0,
                             float(knobs.blink_max_ms) / 1000.0, rngs[0])
        for o, d in zip(b_on, b_du):
            lo, hi = np.searchsorted(tau, [o, o + d], side="left")
            blink[lo:hi] = True
            closure[lo:hi] = 1.0 - np.abs(2.0 * (tau[lo:hi] - o) / d - 1.0)

    gain = np.ones(n, dtype=np.float64)
    glare = np.zeros(n, dtype=bool)
    g_on, g_du, glare_z = empty, empty, None
    if float(knobs.glare_rate_per_min) > 0.0:
        g_on, g_du = _bursts(total, float(knobs.glare_rate_per_min), float(knobs.glare_min_s),
                             float(knobs.glare_max_s), rngs[1])
        for o, d in zip(g_on, g_du):
            lo, hi = np.searchsorted(tau, [o, o + d], side="left")
            glare[lo:hi] = True
            gain[lo:hi] = float(knobs.glare_gain)
        glare_z = np.asarray(rngs[2].standard_normal((n, 10, 2)), dtype=np.float64)

    drift = None
    if float(knobs.drift_deg_per_20s) > 0.0 or float(knobs.drift_mm_per_20s) > 0.0:
        rot = random_walk(n, 3, float(knobs.drift_deg_per_20s), fps, rngs[3])
        trans = random_walk(n, 3, float(knobs.drift_mm_per_20s), fps, rngs[3])
        drift = np.hstack([rot, trans])

    return EventSchedule(n, fps, closure, blink, b_on, b_du, gain, glare, g_on, g_du, glare_z, drift)


# ---------------------------------------------------------------- effects on one frame
def _check_one_frame(frame) -> None:
    if not isinstance(frame, FaceObs):
        raise SimError("frames", f"frame must be a FaceObs, got {type(frame).__name__}")
    if frame.face_found is not True:
        raise SimError("frames", "frame has no face")


def _toward(lm, ids, target_xy, c) -> None:
    """Move the x, y of landmarks `ids` toward target_xy (shape (2,)) by the fraction c, in place."""
    for n in ids:
        lm[n, :2] = lm[n, :2] + c * (target_xy - lm[n, :2])


def apply_blink(frame, closure, *, _build=make_faceobs):
    _check_one_frame(frame)
    if not _is_real(closure) or not 0.0 <= float(closure) <= 1.0:
        raise SimError("value", f"closure must be a finite real in [0, 1], got {closure!r}")
    c = float(closure)
    if c == 0.0:
        return frame
    orig = np.asarray(frame.landmarks, dtype=np.float64)   # targets are read from here only
    lm = orig.copy()
    for outer, inner, up, low, iris in (
            (L.R_EYE_OUTER, L.R_EYE_INNER, L.R_LID_UPPER, L.R_LID_LOWER, L.R_IRIS),
            (L.L_EYE_OUTER, L.L_EYE_INNER, L.L_LID_UPPER, L.L_LID_LOWER, L.L_IRIS)):
        lid_mid = (orig[up, :2] + orig[low, :2]) / 2.0
        corner_mid = (orig[outer, :2] + orig[inner, :2]) / 2.0
        for n in (up, low):
            lm[n, :2] = orig[n, :2] + c * (lid_mid - orig[n, :2])
        for n in iris:
            lm[n, :2] = orig[n, :2] + c * (corner_mid - orig[n, :2])
    bs = np.asarray(frame.blendshapes, dtype=np.float64).copy()
    bs[L.BS_EYE_BLINK_LEFT] = c
    bs[L.BS_EYE_BLINK_RIGHT] = c
    return _build(frame.t, lm, bs, frame.head_matrix, frame.image_w, frame.image_h)


def apply_glare(frame, gain, z, base_noise_px, *, _build=make_faceobs):
    _check_one_frame(frame)
    if not _is_real(gain) or not 1.0 <= float(gain) <= 8.0:
        raise SimError("value", f"gain must be a finite real in [1, 8], got {gain!r}")
    try:
        zz = np.asarray(z)
        ok = zz.dtype.kind == "f" and zz.shape == (10, 2) and bool(np.isfinite(zz).all())
    except (ValueError, TypeError):
        ok = False
    if not ok:
        raise SimError("value", "z must be a finite float array of shape (10, 2)")
    if not _is_real(base_noise_px) or not 0.0 <= float(base_noise_px) <= 3.0:
        raise SimError("value", f"base_noise_px must be a finite real in [0, 3], got {base_noise_px!r}")
    g, b = float(gain), float(base_noise_px)
    if g == 1.0 or b == 0.0:
        return frame
    a = b * math.sqrt(g * g - 1.0)
    lm = np.asarray(frame.landmarks, dtype=np.float64).copy()
    ids = list(L.R_IRIS + L.L_IRIS)
    lm[ids, 0] += a * zz[:, 0].astype(np.float64) / frame.image_w
    lm[ids, 1] += a * zz[:, 1].astype(np.float64) / frame.image_h
    return _build(frame.t, lm, frame.blendshapes, frame.head_matrix,
                  frame.image_w, frame.image_h)


def _t3(name, v) -> np.ndarray:
    try:
        items = list(v)
    except TypeError:
        raise SimError("value", f"{name} must be 3 numbers") from None
    if len(items) != 3 or not all(_is_real(i) for i in items):
        raise SimError("value", f"{name} must be 3 finite real numbers")
    return np.array([float(i) for i in items], dtype=np.float64)


def apply_leak(frame, t0_mm, knobs, *, _build=make_faceobs):
    check_event_knobs(knobs)
    _check_one_frame(frame)
    t0 = _t3("t0_mm", t0_mm)
    a_yaw, a_pitch = float(knobs.leak_yaw_deg_per_mm), float(knobs.leak_pitch_deg_per_mm)
    if a_yaw == 0.0 and a_pitch == 0.0:
        return frame
    hm = np.asarray(frame.head_matrix, dtype=np.float64).copy()
    dt = 10.0 * hm[:3, 3] - t0          # head matrix translation is in cm; T is in mm
    psi, theta, phi = angles_from_matrix(hm)
    hm[:3, :3] = rotation_matrix(psi + a_yaw * dt[0], theta + a_pitch * dt[1], phi)
    return _build(frame.t, frame.landmarks, frame.blendshapes, hm,
                  frame.image_w, frame.image_h)


def drifted_pose(pose, drift_row) -> HeadPose:
    if not isinstance(pose, HeadPose):
        raise SimError("value", f"pose must be a HeadPose, got {type(pose).__name__}")
    if drift_row is None:
        return pose
    try:
        items = list(drift_row)
    except TypeError:
        raise SimError("value", "drift_row must be 6 numbers") from None
    if len(items) != 6 or not all(_is_real(i) for i in items):
        raise SimError("value", "drift_row must be 6 finite real numbers")
    d = [float(i) for i in items]
    return HeadPose(pose.yaw_deg + d[0], pose.pitch_deg + d[1], pose.roll_deg + d[2],
                    pose.tx_mm + d[3], pose.ty_mm + d[4], pose.tz_mm + d[5])


# ---------------------------------------------------------------- effects on a list of frames
def _check_schedule(sched) -> None:
    if not isinstance(sched, EventSchedule):
        raise SimError("schedule", f"sched must be an EventSchedule, got {type(sched).__name__}")


def apply_events_before_noise(frames, idx, sched, *, _build=make_faceobs):
    """Blinks (they happen in front of the camera, before the tracker noise).
    `_build` is private (S043b), see sim_env.add_noise."""
    _check_schedule(sched)
    ix = _check_frames(frames, idx, sched.n_nominal)
    out = []
    for f, k in zip(frames, ix):
        c = float(sched.closure[k])
        out.append(apply_blink(f, c, _build=_build) if c > 0.0 else f)
    return out


def apply_events_after_noise(frames, idx, sched, knobs, base_noise_px, t0_mm, *, _build=make_faceobs):
    """Glare (extra iris noise), then the translation leak. `_build` is private (S043b)."""
    check_event_knobs(knobs)
    _check_schedule(sched)
    ix = _check_frames(frames, idx, sched.n_nominal)
    if not _is_real(base_noise_px) or not 0.0 <= float(base_noise_px) <= 3.0:
        raise SimError("value", f"base_noise_px must be a finite real in [0, 3], got {base_noise_px!r}")
    _t3("t0_mm", t0_mm)
    out = []
    for f, k in zip(frames, ix):
        if sched.glare_z is not None and sched.glare_gain[k] > 1.0:
            f = apply_glare(f, float(sched.glare_gain[k]), sched.glare_z[k], base_noise_px, _build=_build)
        out.append(apply_leak(f, t0_mm, knobs, _build=_build))
    return out
