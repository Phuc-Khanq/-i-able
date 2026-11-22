"""S022: the gaze baseline. One shared ridge map from features to screen pixels,
plus one shift (offset) per sitting.

Model, with Z = (X - mu) / s:   y_hat = Z W + shift_g
W is fitted on within-sitting centred data, so a sitting's offset does not leak
into W. A new sitting (test) gets `default_shift` (no re-calibration here, E4).

Pure numpy, closed form. It never sees simulator truth (only X, Y, groups).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class GazeError(ValueError):
    """A fit or predict request was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True, eq=False)
class RidgeMap:
    mean: np.ndarray           # (d,)
    scale: np.ndarray          # (d,) SD with ddof 0; 0 became 1
    lo: np.ndarray             # (d,) train column min (inputs are clipped to [lo, hi])
    hi: np.ndarray             # (d,) train column max
    W: np.ndarray              # (d, 2) in Z units
    shifts: dict               # group -> (2,)
    default_shift: np.ndarray  # (2,) unweighted mean of the shifts
    alpha: float


def _as_X(X, name="X") -> np.ndarray:
    try:
        a = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        raise GazeError("shape", f"{name} must be a real 2-D array") from None
    if a.ndim != 2 or a.shape[1] < 1:
        raise GazeError("shape", f"{name} must be 2-D (n, d) with d >= 1, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise GazeError("nan", f"{name} must be finite")
    return a


def fit_ridge_shift(X, Y, groups, alpha=1.0) -> RidgeMap:
    """Fit W on within-group centred data; shift_g = mean(Y_g) - mean(Z_g) W."""
    X = _as_X(X)
    Y = _as_X(Y, "Y")
    n, d = X.shape
    if Y.shape != (n, 2):
        raise GazeError("shape", f"Y must be ({n}, 2), got {Y.shape}")
    g = np.asarray(groups)
    if g.shape != (n,) or not np.issubdtype(g.dtype, np.integer):
        raise GazeError("shape", f"groups must be int ({n},), got {g.dtype} {g.shape}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise GazeError("alpha", f"alpha must be a finite real >= 0, got {alpha!r}")
    if n < 2:
        raise GazeError("shape", f"need n >= 2 rows, got {n}")
    alpha = float(alpha)

    lo, hi = X.min(axis=0), X.max(axis=0)
    mu = X.mean(axis=0)
    s = X.std(axis=0, ddof=0)
    s = np.where(s == 0.0, 1.0, s)
    Z = (X - mu) / s

    keys = sorted(int(k) for k in np.unique(g))
    Zc = np.empty_like(Z)
    Yc = np.empty_like(Y)
    zbar, ybar = {}, {}
    for k in keys:
        m = g == k
        zbar[k] = Z[m].mean(axis=0)
        ybar[k] = Y[m].mean(axis=0)
        Zc[m] = Z[m] - zbar[k]
        Yc[m] = Y[m] - ybar[k]

    # A column that is 0 after centring (e.g. constant) has W row 0 for any alpha > 0;
    # solving only on the other columns gives the same W and avoids a singular
    # system when alpha = 0 (test H4).
    act = np.flatnonzero(np.any(Zc != 0.0, axis=0))
    W = np.zeros((d, 2), dtype=np.float64)
    if act.size:
        Za = Zc[:, act]
        W[act] = np.linalg.solve(Za.T @ Za + alpha * np.eye(act.size), Za.T @ Yc)
    shifts = {k: ybar[k] - zbar[k] @ W for k in keys}
    default_shift = np.mean(np.stack([shifts[k] for k in keys]), axis=0)
    return RidgeMap(mean=mu, scale=s, lo=lo, hi=hi, W=W, shifts=shifts,
                    default_shift=default_shift, alpha=alpha)


def predict(model, X, shift=None) -> np.ndarray:
    """(n, d) -> (n, 2): clip X to [lo, hi], then Z W + (shift or default_shift)."""
    if not isinstance(model, RidgeMap):
        raise GazeError("model", f"model must be a RidgeMap, got {type(model).__name__}")
    X = _as_X(X)
    if X.shape[1] != model.mean.shape[0]:
        raise GazeError("shape", f"X must have {model.mean.shape[0]} columns, got {X.shape[1]}")
    b = model.default_shift if shift is None else np.asarray(shift, dtype=np.float64)
    if b.shape != (2,) or not np.all(np.isfinite(b)):
        raise GazeError("shift", f"shift must be finite (2,), got shape {b.shape}")
    Z = (np.clip(X, model.lo, model.hi) - model.mean) / model.scale
    return Z @ model.W + b
