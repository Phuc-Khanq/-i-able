"""Quality gate (S011): every frame is "ok", or has exactly one reason.

REASONS = ("ok", "no_face", "features", "blink", "small", "pose", "jump").
A frame's code is its index in REASONS. The rules are checked in this order
and the first match wins, so reason_counts always adds up to n.
(S011b: blink moved before small and pose; blink_score default 0.2.)

Rules for frame i (features from features.session_features):
  no_face   face_found[i] is False.
  features  any NaN in the feature row (a degenerate frame).
  blink     only on frames that passed no_face and features.
            detected: max(blendshape 9, blendshape 10) >= blink_score, or
            l_i < lid_frac * median(l_j), l = (lid_r + lid_l) / 2, over earlier
            frames j with t_i - t_j <= lid_window_s that passed no_face, features,
            small and pose, and are not blink (detected or held); only with
            >= lid_min_frames of them (the same baseline set as in S011).
            hold: a frame that passed no_face and features, next to a detected frame
            (1 frame before or after, or within hold_ms) is blink too. Offline
            (causal=False): before and after. causal=True: only after. Not chained.
            For the lid baseline of frame i, only holds caused by detections
            before i are known (one forward pass; no frame looks at its own future).
  small     iris_px < min_iris_px.
  pose      |yaw| > max_yaw_deg or |pitch| > max_pitch_deg.
  jump      median over the 4 eye corners (33, 133, 263, 362) of the move in px
            (x * W, y * H) divided by the real dt = t_i - t_{i-1} > jump_px_per_s.
            Skipped for i = 0, when frame i-1 has no face, or dt > jump_max_gap_s.

There is no "low iris confidence" reason: MediaPipe gives no iris
confidence (S008).

This file never imports the simulator and never reads a truth channel.
"""

import math
from dataclasses import dataclass, fields

import numpy as np

from iable_eye import faceobs
from iable_eye import landmark_ids as L
from iable_eye.features import FEATURE_NAMES, session_features

REASONS = ("ok", "no_face", "features", "blink", "small", "pose", "jump")
OK, NO_FACE, FEATURES, BLINK, SMALL, POSE, JUMP = (REASONS.index(r) for r in (
    "ok", "no_face", "features", "blink", "small", "pose", "jump"))
_F = {n: i for i, n in enumerate(FEATURE_NAMES)}
_CORNERS = (L.R_EYE_OUTER, L.R_EYE_INNER, L.L_EYE_OUTER, L.L_EYE_INNER)


class QualityError(ValueError):
    """A refused input. .code is one of "config", "shape", "codes"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True)
class GateConfig:
    blink_score: float = 0.2
    lid_frac: float = 0.5
    lid_window_s: float = 2.0
    lid_min_frames: int = 5
    hold_ms: float = 60.0
    min_iris_px: float = 5.0
    max_yaw_deg: float = 40.0
    max_pitch_deg: float = 30.0
    jump_px_per_s: float = 2000.0
    jump_max_gap_s: float = 0.5

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise QualityError("config", f"{f.name} must be a number, got {type(v).__name__}")
            if not math.isfinite(float(v)) or v <= 0:
                raise QualityError("config", f"{f.name} must be finite and > 0, got {v!r}")


# S059 (DECISIONS 2026-11-22): the gate for REAL sessions. Real open eyes give a blink
# blendshape of about 0.17-0.23, so the simulator value 0.2 coded 33-81 % of real frames "blink".
# 0.5 = the S033 "eyes closed" level. The simulator keeps GateConfig() (0.2).
REAL_GATE_V1 = GateConfig(blink_score=0.5)


def _held_by(k, t, n, hold_s, before, after):
    """Indices that a detection at frame k holds (not k itself)."""
    out = []
    if after:
        j = k + 1
        while j < n and (j == k + 1 or t[j] - t[k] <= hold_s):
            out.append(j)
            j += 1
    if before:
        j = k - 1
        while j >= 0 and (j == k - 1 or t[k] - t[j] <= hold_s):
            out.append(j)
            j -= 1
    return out


def gate_session(arrays, config=GateConfig(), features=None, causal=False) -> np.ndarray:
    """A stacked session -> int8 (n,) codes (indices into REASONS)."""
    n = faceobs.validate_stacked(arrays)
    if not isinstance(config, GateConfig):
        raise QualityError("config", f"config must be a GateConfig, got {type(config).__name__}")
    c = config
    if features is None:
        feats = session_features(arrays)
    else:
        feats = np.asarray(features, dtype=np.float64)
        if feats.shape != (n, len(FEATURE_NAMES)):
            raise QualityError("shape", f"features must be {(n, len(FEATURE_NAMES))}, got {feats.shape}")
    t = np.asarray(arrays["t"], dtype=np.float64)
    ff = np.asarray(arrays["face_found"], dtype=bool)
    codes = np.zeros(n, dtype=np.int8)

    bad_feat = ff & np.isnan(feats).any(axis=1)
    with np.errstate(invalid="ignore"):
        small = ff & ~bad_feat & (feats[:, _F["iris_px"]] < c.min_iris_px)
        pose = ff & ~bad_feat & ~small & ((np.abs(feats[:, _F["yaw_deg"]]) > c.max_yaw_deg)
                                         | (np.abs(feats[:, _F["pitch_deg"]]) > c.max_pitch_deg))
    codes[~ff] = NO_FACE
    codes[bad_feat] = FEATURES
    codes[small] = SMALL
    codes[pose] = POSE
    passed = codes == OK                     # passed no_face..pose: the lid baseline set (as S011)
    elig = ff & ~bad_feat                    # passed no_face, features: blink is checked here

    # blink: one forward pass
    bs = np.asarray(arrays["blendshapes"], dtype=np.float64)
    bmax = np.fmax(bs[:, L.BS_EYE_BLINK_LEFT], bs[:, L.BS_EYE_BLINK_RIGHT])
    lid = (feats[:, _F["lid_r"]] + feats[:, _F["lid_l"]]) / 2.0
    hold_s = c.hold_ms / 1000.0
    detected = np.zeros(n, dtype=bool)
    held = np.zeros(n, dtype=bool)
    min_frames = int(c.lid_min_frames)
    for i in range(n):
        if not elig[i]:
            continue
        hit = bool(bmax[i] >= c.blink_score)
        if not hit:
            lo = int(np.searchsorted(t, t[i] - c.lid_window_s, side="left"))
            while lo > 0 and t[i] - t[lo - 1] <= c.lid_window_s:
                lo -= 1
            while lo < i and t[i] - t[lo] > c.lid_window_s:
                lo += 1
            m = passed[lo:i] & ~detected[lo:i] & ~held[lo:i]
            if int(m.sum()) >= min_frames:
                hit = bool(lid[i] < c.lid_frac * float(np.median(lid[lo:i][m])))
        if hit:
            detected[i] = True
            for j in _held_by(i, t, n, hold_s, before=not causal, after=True):
                if elig[j]:
                    held[j] = True
    blink = detected | (held & elig)
    codes[blink] = BLINK                     # blink wins over small and pose

    # jump: real dt, only on frames still ok
    lm = np.asarray(arrays["landmarks"], dtype=np.float64)
    scale = np.array([float(arrays["image_w"]), float(arrays["image_h"])])
    for i in range(1, n):
        if codes[i] != OK or not ff[i - 1]:
            continue
        dt = t[i] - t[i - 1]
        if dt > c.jump_max_gap_s:
            continue
        a = lm[i, list(_CORNERS), :2] * scale
        b = lm[i - 1, list(_CORNERS), :2] * scale
        speed = float(np.median(np.linalg.norm(a - b, axis=1))) / dt
        if speed > c.jump_px_per_s:
            codes[i] = JUMP
    return codes


def corner_speeds(arrays) -> np.ndarray:
    """float64 (n,) median corner speed in px/s (NaN where the jump check is skipped by
    i = 0 or no face on i or i-1). For measurement and sanity checks."""
    n = faceobs.validate_stacked(arrays)
    t = np.asarray(arrays["t"], dtype=np.float64)
    ff = np.asarray(arrays["face_found"], dtype=bool)
    lm = np.asarray(arrays["landmarks"], dtype=np.float64)[:, list(_CORNERS), :2]
    lm = lm * np.array([float(arrays["image_w"]), float(arrays["image_h"])])
    out = np.full(n, np.nan)
    if n > 1:
        d = np.median(np.linalg.norm(lm[1:] - lm[:-1], axis=2), axis=1) / np.diff(t)
        ok = ff[1:] & ff[:-1]
        out[1:][ok] = d[ok]
    return out


def reason_counts(codes) -> dict:
    """{reason: count} for all 7 reasons."""
    a = np.asarray(codes)
    if a.ndim != 1 or (a.size and (not np.issubdtype(a.dtype, np.integer)
                                   or a.min() < 0 or a.max() >= len(REASONS))):
        raise QualityError("codes", f"codes must be a 1-D int array in 0..{len(REASONS) - 1}")
    counts = np.bincount(a.astype(np.int64), minlength=len(REASONS)) if a.size else np.zeros(len(REASONS), int)
    return {r: int(counts[i]) for i, r in enumerate(REASONS)}
