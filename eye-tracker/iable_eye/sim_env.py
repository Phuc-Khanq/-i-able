"""Observation knobs of the simulator: frame times, landmark noise, head-pose noise.

Every number is (simulated). Blinks, glare, drift and leak come in S042.
Randomness comes only from the generators made in make_rngs (three streams).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from iable_eye.faceobs import N_LANDMARKS, FaceObs, make_faceobs
from iable_eye.sim_core import SimError, rotation_matrix

TIMELINE_START_S = 1.0
MAX_DURATION_S = 3600.0


@dataclass(frozen=True)
class ObsKnobs:
    fps: float = 30.0               # nominal frame rate, 5-60
    drop_prob: float = 0.0          # chance that a frame is missing, 0-0.3
    jitter_ms: float = 0.0          # timestamp jitter, uniform in +-jitter_ms, 0-15 and < 500/fps
    noise_px: float = 0.0           # landmark noise SD per axis, image px, 0-3
    noise_rho: float = 0.0          # lag-1 correlation of all noise, 0-0.95
    noise_common_frac: float = 0.0  # part of the landmark noise variance shared by all points, 0-1
    pose_noise_deg: float = 0.0     # SD of yaw/pitch/roll noise, degrees, 0-1


_KNOB_RANGES = (("fps", 5.0, 60.0), ("drop_prob", 0.0, 0.3), ("jitter_ms", 0.0, 15.0),
                ("noise_px", 0.0, 3.0), ("noise_rho", 0.0, 0.95),
                ("noise_common_frac", 0.0, 1.0), ("pose_noise_deg", 0.0, 1.0))


def _is_real(v) -> bool:
    return (not isinstance(v, (bool, np.bool_)) and isinstance(v, numbers.Real)
            and math.isfinite(float(v)))


def _is_int(v) -> bool:
    return not isinstance(v, (bool, np.bool_)) and isinstance(v, int)


def check_knobs(k) -> None:
    if not isinstance(k, ObsKnobs):
        raise SimError("knobs", f"knobs must be an ObsKnobs, got {type(k).__name__}")
    for name, lo, hi in _KNOB_RANGES:
        v = getattr(k, name)
        if not _is_real(v):
            raise SimError("knobs", f"{name} must be a finite real number, got {v!r}")
        if not lo <= float(v) <= hi:
            raise SimError("knobs", f"{name} must be in [{lo}, {hi}], got {float(v)}")
    if not float(k.jitter_ms) < 500.0 / float(k.fps):
        raise SimError("knobs", f"jitter_ms must be < 500/fps = {500.0 / float(k.fps):.4f}, "
                                f"got {float(k.jitter_ms)}")


def make_rngs(seed):
    if not _is_int(seed) or not 0 <= seed < 2 ** 53:
        raise SimError("seed", f"seed must be an int in [0, 2**53), got {seed!r}")
    ss = np.random.SeedSequence(seed).spawn(3)
    return tuple(np.random.default_rng(s) for s in ss)


# ---------------------------------------------------------------- frame timeline
def timeline(duration_s, knobs, rng):
    """-> (t_s float64 of the kept frames, idx int64 nominal numbers of the kept frames, N)."""
    check_knobs(knobs)
    if not _is_real(duration_s) or not 0.0 < float(duration_s) <= MAX_DURATION_S:
        raise SimError("timeline", f"duration_s must be in (0, {MAX_DURATION_S}], got {duration_s!r}")
    fps = float(knobs.fps)
    n = int(math.floor(float(duration_s) * fps + 1e-9))
    if n < 2:
        raise SimError("timeline", f"duration {duration_s} s at {fps} fps gives {n} frame(s), need 2")
    u_drop = rng.random(n)
    u_jit = rng.random(n)
    kept = u_drop >= float(knobs.drop_prob)
    offset = (2.0 * u_jit - 1.0) * float(knobs.jitter_ms) / 1000.0
    t_all = TIMELINE_START_S + np.arange(n, dtype=np.float64) / fps + offset
    idx = np.flatnonzero(kept).astype(np.int64)
    t_s = t_all[kept]
    if idx.shape[0] < 2:
        raise SimError("timeline", f"only {idx.shape[0]} frame(s) kept, need 2")
    return t_s, idx, n


# ---------------------------------------------------------------- AR(1) noise
def ar1(n_steps, n_series, sigma, rho, rng) -> np.ndarray:
    """Stationary AR(1): every row has SD sigma, lag-1 correlation rho. float64 (n_steps, n_series)."""
    if not _is_int(n_steps) or n_steps < 1:
        raise SimError("ar1", f"n_steps must be an int >= 1, got {n_steps!r}")
    if not _is_int(n_series) or n_series < 1:
        raise SimError("ar1", f"n_series must be an int >= 1, got {n_series!r}")
    if not _is_real(sigma) or float(sigma) < 0.0:
        raise SimError("ar1", f"sigma must be a finite real >= 0, got {sigma!r}")
    if not _is_real(rho) or not 0.0 <= float(rho) <= 0.95:
        raise SimError("ar1", f"rho must be in [0, 0.95], got {rho!r}")
    sigma, rho = float(sigma), float(rho)
    e = np.asarray(rng.standard_normal((n_steps, n_series)), dtype=np.float64)
    drive = sigma * math.sqrt(1.0 - rho ** 2)
    x = np.empty((n_steps, n_series), dtype=np.float64)
    x[0] = sigma * e[0]
    for k in range(1, n_steps):
        x[k] = rho * x[k - 1] + drive * e[k]
    return x


# ---------------------------------------------------------------- angles
def angles_from_matrix(m):
    """(yaw, pitch, roll) in degrees of R = Ry(yaw) Rx(pitch) Rz(roll). Valid for |pitch| < 90."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    psi = math.atan2(r[0, 2], r[2, 2])
    theta = -math.asin(min(1.0, max(-1.0, float(r[1, 2]))))
    phi = math.atan2(r[1, 0], r[1, 1])
    return math.degrees(psi), math.degrees(theta), math.degrees(phi)


# ---------------------------------------------------------------- noise on frames
def _check_frames(frames, idx, n_nominal) -> np.ndarray:
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise SimError("frames", "frames must be a non-empty list or tuple")
    for i, f in enumerate(frames):
        if not isinstance(f, FaceObs):
            raise SimError("frames", f"frames[{i}] is {type(f).__name__}, not FaceObs")
        if f.face_found is not True:
            raise SimError("frames", f"frames[{i}] has no face")
    if not _is_int(n_nominal) or n_nominal < 1:
        raise SimError("frames", f"n_nominal must be an int >= 1, got {n_nominal!r}")
    try:
        ix = np.asarray(idx)
        if ix.dtype.kind not in "iu":
            raise ValueError("idx must hold integers")
        ix = ix.astype(np.int64)
    except (ValueError, TypeError) as e:
        raise SimError("frames", f"idx is not an integer array: {e}") from None
    if ix.ndim != 1 or ix.shape[0] != len(frames):
        raise SimError("frames", f"idx must be 1-D with length {len(frames)}, got shape {ix.shape}")
    if ix.shape[0] > 1 and np.any(ix[1:] <= ix[:-1]):
        raise SimError("frames", "idx must strictly increase")
    if ix[0] < 0 or ix[-1] >= n_nominal:
        raise SimError("frames", f"idx must be in [0, {n_nominal}), got {int(ix[0])}..{int(ix[-1])}")
    return ix


def add_noise(frames, idx, n_nominal, knobs, rng_landmark, rng_pose, *, _build=make_faceobs):
    """Landmark noise (image px) and head-pose angle noise; indexed by the nominal frame number.
    `_build` is private (S043b): simulate_session passes a builder without the per-frame
    checks, because faceobs.stack() validates the whole session at the end."""
    check_knobs(knobs)
    ix = _check_frames(frames, idx, n_nominal)
    do_lm = float(knobs.noise_px) > 0.0
    do_pose = float(knobs.pose_noise_deg) > 0.0
    if not do_lm and not do_pose:
        return list(frames)
    rho = float(knobs.noise_rho)
    common = own = ang = None
    if do_lm:
        sigma, c = float(knobs.noise_px), float(knobs.noise_common_frac)
        common = ar1(n_nominal, 2, sigma * math.sqrt(c), rho, rng_landmark)
        own = ar1(n_nominal, 2 * N_LANDMARKS, sigma * math.sqrt(1.0 - c), rho,
                  rng_landmark).reshape(n_nominal, N_LANDMARKS, 2)
    if do_pose:
        ang = ar1(n_nominal, 3, float(knobs.pose_noise_deg), rho, rng_pose)
    out = []
    for f, k in zip(frames, ix):
        k = int(k)
        landmarks, head = f.landmarks, f.head_matrix
        if do_lm:
            d = own[k] + common[k]
            landmarks = np.array(f.landmarks, dtype=np.float64)
            landmarks[:, 0] += d[:, 0] / f.image_w
            landmarks[:, 1] += d[:, 1] / f.image_h
        if do_pose:
            psi, theta, phi = angles_from_matrix(f.head_matrix)
            head = np.array(f.head_matrix, dtype=np.float64)
            head[:3, :3] = rotation_matrix(psi + ang[k, 0], theta + ang[k, 1], phi + ang[k, 2])
        out.append(_build(f.t, landmarks, f.blendshapes, head, f.image_w, f.image_h))
    return out
