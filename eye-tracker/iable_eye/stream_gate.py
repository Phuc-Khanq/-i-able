"""Streaming quality gate (S044): one FaceObs in, (code, feature row) out.

The batch gate is the oracle: for every session,
    [StreamGate(c).step(o)[0] for o in unstack(a)] == gate_session(a, c, causal=True)
and the feature rows equal session_features(a), exactly (no tolerance).

The same math as quality.gate_session, frame by frame (see that docstring for the rules).
Causal: frame i only uses frames 0..i. Holds go only forward (a detection holds the
next frame, and any later frame within hold_ms); they are not chained.
State is bounded: the lid history keeps only frames with t_i - t_j <= lid_window_s.

This file never imports the simulator and never reads a truth channel.
"""

import math
from collections import deque

import numpy as np

from iable_eye import faceobs
from iable_eye import landmark_ids as L
from iable_eye.features import DEFAULT_FOCAL_PX, FEATURE_NAMES, FeatureError, frame_features
from iable_eye.quality import BLINK, FEATURES, JUMP, NO_FACE, OK, POSE, SMALL, GateConfig, _CORNERS, _F

_N_FEAT = len(FEATURE_NAMES)


class StreamGateError(ValueError):
    """A refused input. .code is one of "config", "type", "time", "shape"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class StreamGate:
    def __init__(self, config=GateConfig(), focal_px=DEFAULT_FOCAL_PX):
        if not isinstance(config, GateConfig):
            raise StreamGateError("config", f"config must be a GateConfig, got {type(config).__name__}")
        self.config = config
        self.focal_px = focal_px
        self.reset()

    def reset(self) -> None:
        self._t = None             # previous frame: time, face_found, scaled corners, detected
        self._ff = False
        self._corners = None
        self._det = False
        self._t_det = -math.inf    # time of the last detection
        self._size = None          # (W, H) of the first frame after a reset
        self._hist = deque()       # (t, lid) of frames in the lid baseline set

    def step(self, obs) -> tuple:
        if not isinstance(obs, faceobs.FaceObs):
            raise StreamGateError("type", f"obs must be a FaceObs, got {type(obs).__name__}")
        t = float(obs.t)
        if self._t is not None and not t > self._t:
            raise StreamGateError("time", f"t {t!r} must be > previous t {self._t!r}")
        size = (obs.image_w, obs.image_h)
        if self._size is None:
            self._size = size
        elif size != self._size:
            raise StreamGateError("shape", f"image {size} differs from the first frame {self._size}")
        c = self.config
        ff = bool(obs.face_found)

        # 1. features (as session_features)
        feats = np.full(_N_FEAT, np.nan, dtype=np.float64)
        if ff:
            try:
                feats = frame_features(obs.landmarks, int(obs.image_w), int(obs.image_h),
                                       obs.head_matrix, focal_px=self.focal_px)
            except FeatureError as e:
                if e.code != "degenerate":
                    raise

        # 2. no_face, features, small, pose
        bad = ff and bool(np.isnan(feats).any())
        small = ff and not bad and bool(feats[_F["iris_px"]] < c.min_iris_px)
        pose = ff and not bad and not small and bool(
            (np.abs(feats[_F["yaw_deg"]]) > c.max_yaw_deg) | (np.abs(feats[_F["pitch_deg"]]) > c.max_pitch_deg))
        code = NO_FACE if not ff else FEATURES if bad else SMALL if small else POSE if pose else OK
        passed = code == OK
        elig = ff and not bad

        # 3. held by an earlier detection (the next frame, or within hold_ms)
        held = elig and (self._det or t - self._t_det <= c.hold_ms / 1000.0)

        # 4. detection: blendshape score, else the lid rule over the last lid_window_s
        while self._hist and t - self._hist[0][0] > c.lid_window_s:
            self._hist.popleft()
        detected = False
        lid = (feats[_F["lid_r"]] + feats[_F["lid_l"]]) / 2.0
        if elig:
            bs = np.asarray(obs.blendshapes, dtype=np.float64)
            hit = bool(np.fmax(bs[L.BS_EYE_BLINK_LEFT], bs[L.BS_EYE_BLINK_RIGHT]) >= c.blink_score)
            if not hit and len(self._hist) >= int(c.lid_min_frames):
                med = float(np.median(np.array([h[1] for h in self._hist], dtype=np.float64)))
                hit = bool(lid < c.lid_frac * med)
            detected = hit

        # 5. blink wins over small and pose; 6. the baseline set
        if detected or held:
            code = BLINK
        if passed and not detected and not held:
            self._hist.append((t, lid))

        # 7. jump against the previous frame (real dt)
        scale = np.array([float(obs.image_w), float(obs.image_h)])
        corners = np.asarray(obs.landmarks, dtype=np.float64)[list(_CORNERS), :2] * scale
        if code == OK and self._t is not None and self._ff:
            dt = t - self._t
            if dt <= c.jump_max_gap_s:
                speed = float(np.median(np.linalg.norm(corners - self._corners, axis=1))) / dt
                if speed > c.jump_px_per_s:
                    code = JUMP

        # 8. state for the next frame
        self._t, self._ff, self._corners, self._det = t, ff, corners, detected
        if detected:
            self._t_det = t
        return code, feats
