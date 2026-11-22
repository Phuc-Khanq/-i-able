"""S024: quick re-calibration helpers.

- `recenter_shift`: one look at a known target (e.g. the screen centre) -> a new shift.
  Uses the per-axis median, so one glitch frame cannot pull it (E2).
- `fit_dot_recal` / `apply_recal`: per-axis shift (or scale + shift) from a few dots.

Pure: imports only math, dataclasses, numpy and iable_eye.gaze_baseline (E1).
The look target is an input; this module never sees simulator truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass  # noqa: F401  (allowed import, E1)

import numpy as np

from iable_eye.gaze_baseline import predict


class RecalError(ValueError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


def _xy(a, name):
    try:
        v = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError):
        raise RecalError("shape", f"{name} must be real numbers") from None
    return v


def recenter_shift(model, X_look, target_xy, min_frames=5) -> np.ndarray:
    """shift = default_shift + (target - median(predict(model, X_look)))."""
    t = _xy(target_xy, "target_xy")
    if t.shape != (2,):
        raise RecalError("shape", f"target_xy must be (2,), got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise RecalError("nan", "target_xy must be finite")
    X = _xy(X_look, "X_look")
    if X.ndim != 2:
        raise RecalError("shape", f"X_look must be 2-D, got shape {X.shape}")
    if X.shape[0] < int(min_frames):
        raise RecalError("few", f"need >= {min_frames} look frames, got {X.shape[0]}")
    p = np.median(predict(model, X), axis=0)
    return np.asarray(model.default_shift, dtype=np.float64) + (t - p)


def fit_dot_recal(pred_xy, true_xy, scale=False):
    """(s, b), each (2,), so that true ~ s * pred + b per axis."""
    p = _xy(pred_xy, "pred_xy")
    q = _xy(true_xy, "true_xy")
    if p.ndim != 2 or p.shape[1] != 2 or p.shape != q.shape or p.shape[0] < 1:
        raise RecalError("shape", f"pred and true must both be (n, 2), n >= 1; got {p.shape}, {q.shape}")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise RecalError("nan", "pred and true must be finite")
    if not scale:
        return np.ones(2), (q - p).mean(axis=0)
    s = np.empty(2)
    b = np.empty(2)
    for a in range(2):
        x, y = p[:, a], q[:, a]
        xc = x - x.mean()
        sxx = float(xc @ xc)
        if sxx == 0.0 or not math.isfinite(sxx):
            raise RecalError("degenerate", f"axis {a} of pred has zero spread")
        s[a] = float(xc @ (y - y.mean())) / sxx
        b[a] = y.mean() - s[a] * x.mean()
    return s, b


def apply_recal(pred_xy, s, b) -> np.ndarray:
    return _xy(pred_xy, "pred_xy") * _xy(s, "s") + _xy(b, "b")
