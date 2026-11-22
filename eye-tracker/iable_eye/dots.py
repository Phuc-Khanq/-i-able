"""S022: one prediction per grid dot, from the frames in its settled window.

Pure numpy. It knows only times, dot numbers and onsets (given by the caller);
it never reads the simulator.
"""
from __future__ import annotations

import numpy as np


def dot_windows(tau_s, dot, onsets_s, window_ms=(400, 1200)) -> list:
    """For each dot d (onset o_d): frame indexes i with dot[i] == d and
    o_d + a <= tau_i < o_d + b (half-open), rising. One entry per dot, even if empty."""
    tau = np.asarray(tau_s, dtype=np.float64)
    dt = np.asarray(dot)
    on = np.asarray(onsets_s, dtype=np.float64)
    if tau.ndim != 1 or dt.shape != tau.shape or on.ndim != 1:
        raise ValueError(f"tau_s and dot must be equal 1-D, onsets 1-D; got {tau.shape}, {dt.shape}, {on.shape}")
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    out = []
    for d, o in enumerate(on):
        m = (dt == d) & (tau >= o + a) & (tau < o + b)
        out.append((d, np.flatnonzero(m)))
    return out


def dot_prediction(pred_xy, ok, idx, n_expected, lost_if_ok_frac_below=0.5):
    """(x, y) median of the ok frames in idx, or None if #ok < frac * n_expected."""
    idx = np.asarray(idx, dtype=np.int64)
    ok = np.asarray(ok, dtype=bool)
    if idx.size == 0:
        return None
    good = idx[ok[idx]]
    if good.size < lost_if_ok_frac_below * n_expected:
        return None
    p = np.asarray(pred_xy, dtype=np.float64)[good]
    return float(np.median(p[:, 0])), float(np.median(p[:, 1]))
