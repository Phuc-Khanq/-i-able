"""S025: candidate gaze models = the ridge+shift base plus a residual map f on Z.

    y_hat = Z W + b + f(Z)

- ridge:        f = 0 (the S022 base, unchanged).
- kernel_ridge: f(q) = k(q, A) C, RBF kernel, gamma = 1 / d, lambda = KRR_LAMBDA.
- mlp:          f(q) = RES_SCALE_PX * MLP(q), one tanh layer, lbfgs, fixed seed.

f is fitted on per-(sitting, dot) medians of Z and of the in-sample residual
R = Y - (Z W + shift_g) (own-sitting shift), so the sitting offset is not learned by f.
Hyperparameters are fixed here (DECISIONS 2026-11-19); no tuning.
The model is chosen only by `select_model` with the frozen rule in gates_v1.json.
Imports only numpy, math, dataclasses, sklearn.neural_network, iable_eye.gaze_baseline (X1).
"""
import math
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor

from iable_eye.gaze_baseline import RidgeMap, fit_ridge_shift, predict

KINDS = ("ridge", "kernel_ridge", "mlp")
KRR_LAMBDA = 1.0
MLP_HIDDEN, MLP_ALPHA, MLP_MAX_ITER, MLP_SEED = 16, 1.0, 2000, 0
RES_SCALE_PX = 100.0


class CandidateError(ValueError):
    """Refused request. code: "kind", "shape", "nan", "rule", "few"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True, eq=False)
class Candidate:
    kind: str
    base: RidgeMap
    A: "np.ndarray | None"     # (m, d) dot-median Z rows (kernel_ridge only)
    C: "np.ndarray | None"     # (m, 2) kernel weights (kernel_ridge only)
    gamma: float
    mlp: "object | None"       # fitted MLPRegressor (mlp only)


def _rbf(P, Q, gamma) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    d2 = (P * P).sum(1)[:, None] + (Q * Q).sum(1)[None, :] - 2.0 * P @ Q.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def krr_fit(A, R, gamma, lam) -> np.ndarray:
    """C = solve(K + lam I, R), K_ij = exp(-gamma ||a_i - a_j||^2)."""
    A = np.asarray(A, dtype=np.float64)
    K = _rbf(A, A, gamma)
    return np.linalg.solve(K + lam * np.eye(A.shape[0]), np.asarray(R, dtype=np.float64))


def krr_predict(A, C, gamma, Q) -> np.ndarray:
    return _rbf(Q, A, gamma) @ np.asarray(C, dtype=np.float64)


def dot_rows(Z, R, groups, dots) -> tuple:
    """Per (group, dot) pair, in sorted order: the median Z row and the median R row."""
    Z, R = np.asarray(Z, dtype=np.float64), np.asarray(R, dtype=np.float64)
    g, t = np.asarray(groups), np.asarray(dots)
    pairs = sorted({(int(a), int(b)) for a, b in zip(g, t)})
    A = np.stack([np.median(Z[(g == a) & (t == b)], axis=0) for a, b in pairs])
    Rm = np.stack([np.median(R[(g == a) & (t == b)], axis=0) for a, b in pairs])
    return A, Rm


def _Z(base, X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (np.clip(X, base.lo, base.hi) - base.mean) / base.scale


def fit_candidate(kind, X, Y, groups, dots, alpha=1.0) -> Candidate:
    if kind not in KINDS:
        raise CandidateError("kind", f"kind must be one of {KINDS}, got {kind!r}")
    n = np.asarray(X).shape[0] if np.ndim(X) >= 1 else -1
    if np.asarray(dots).shape != (n,) or np.asarray(groups).shape != (n,):
        raise CandidateError("shape", f"groups and dots must both be ({n},)")
    base = fit_ridge_shift(X, Y, groups, alpha)
    if kind == "ridge":
        return Candidate(kind, base, None, None, 0.0, None)
    Z = _Z(base, X)
    g = np.asarray(groups)
    S = np.stack([base.shifts[int(k)] for k in g])
    R = np.asarray(Y, dtype=np.float64) - (Z @ base.W + S)
    A, Rm = dot_rows(Z, R, g, dots)
    gamma = 1.0 / Z.shape[1]
    if kind == "kernel_ridge":
        return Candidate(kind, base, A, krr_fit(A, Rm, gamma, KRR_LAMBDA), gamma, None)
    net = MLPRegressor(hidden_layer_sizes=(MLP_HIDDEN,), activation="tanh", solver="lbfgs",
                       alpha=MLP_ALPHA, max_iter=MLP_MAX_ITER, random_state=MLP_SEED)
    net.fit(A, Rm / RES_SCALE_PX)
    return Candidate(kind, base, None, None, gamma, net)


def predict_candidate(c, X, shift=None) -> np.ndarray:
    """predict(base, X, shift) + f(Z); ridge: f = 0."""
    if not isinstance(c, Candidate):
        raise CandidateError("kind", f"c must be a Candidate, got {type(c).__name__}")
    p = predict(c.base, X, shift)
    if c.kind == "ridge":
        return p
    Z = _Z(c.base, X)
    if c.kind == "kernel_ridge":
        return p + krr_predict(c.A, c.C, c.gamma, Z)
    return p + RES_SCALE_PX * np.asarray(c.mlp.predict(Z), dtype=np.float64).reshape(-1, 2)


def recenter_candidate(c, X_look, target_xy, min_frames=5) -> np.ndarray:
    """base.default_shift + (target - median(predict_candidate(c, X_look)))."""
    t = np.asarray(target_xy, dtype=np.float64)
    if t.shape != (2,) or not np.all(np.isfinite(t)):
        raise CandidateError("shape", f"target_xy must be finite (2,), got {t.shape}")
    X = np.asarray(X_look, dtype=np.float64)
    if X.ndim != 2:
        raise CandidateError("shape", f"X_look must be 2-D, got {X.shape}")
    if X.shape[0] < int(min_frames):
        raise CandidateError("few", f"need >= {min_frames} look frames, got {X.shape[0]}")
    p = np.median(predict_candidate(c, X), axis=0)
    return np.asarray(c.base.default_shift, dtype=np.float64) + (t - p)


def select_model(medians: dict, rule: dict) -> str:
    """Frozen rule: best = min median; tie set = m - best <= tie_tol_px; first of tie_order."""
    order = list(rule["tie_order"])
    tol = float(rule["tie_tol_px"])
    missing = [k for k in order if k not in medians]
    if missing:
        raise CandidateError("rule", f"medians missing kinds {missing}")
    vals = {k: float(medians[k]) for k in order}
    if not all(math.isfinite(v) for v in vals.values()):
        raise CandidateError("nan", f"medians must be finite, got {vals}")
    best = min(vals.values())
    tie = {k for k, v in vals.items() if v - best <= tol}
    return next(k for k in order if k in tie)
