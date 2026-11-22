"""Per-cell metrics and the environment gap V2 (ROADMAP section 4).

Pure functions on small arrays. No file access, no clock.
The only randomness is `numpy.random.default_rng(seed)` in `bootstrap_v2`.

Cell score: m_c = median of the valid dot errors, loss_c = l_c / (n_c + l_c).
V2 = max_c m_c / min_c m_c, defined only when all 9 cells are MEASURED.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from iable_eye.session_store import DISTANCES, LIGHTS


class MetricsError(ValueError):
    """A metrics request was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


CELLS = tuple(f"{l}/{d}" for l in LIGHTS for d in DISTANCES)
MIN_DOTS = 30
MAX_LOSS = 0.20
V1_CELL_PX = 150.0
V1_ALL_PX = 120.0
V2_MAX = 1.5
BOOT_RESAMPLES = 2000


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def dot_errors_px(pred_xy, true_xy) -> np.ndarray:
    """Euclidean error per dot: e = sqrt((x_hat - x)^2 + (y_hat - y)^2)."""
    p = np.asarray(pred_xy, dtype=np.float64)
    t = np.asarray(true_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape != t.shape or p.shape[0] < 1:
        raise MetricsError("shape", f"need equal (n, 2) shapes with n >= 1, got {p.shape} and {t.shape}")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
        raise MetricsError("nan", "points must be finite")
    return np.sqrt(np.sum((p - t) ** 2, axis=1))


def _check_errors(errors) -> np.ndarray:
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim != 1:
        raise MetricsError("shape", f"errors must be 1-D, got shape {e.shape}")
    if not np.all(np.isfinite(e)):
        raise MetricsError("nan", "errors must be finite")
    if np.any(e < 0):
        raise MetricsError("negative", "errors must be >= 0")
    return e


@dataclass(frozen=True)
class CellScore:
    cell: str
    n: int
    lost: int
    median_px: float
    loss: float
    measured: bool


def cell_score(cell, errors, n_lost, min_dots=MIN_DOTS) -> CellScore:
    """Score one cell: median error of valid dots, loss rate, measured flag."""
    if cell not in CELLS:
        raise MetricsError("cell", f"unknown cell {cell!r}")
    e = _check_errors(errors)
    if not _is_int(n_lost) or n_lost < 0:
        raise MetricsError("lost", f"n_lost must be an int >= 0, got {n_lost!r}")
    n = int(e.size)
    lost = int(n_lost)
    if n + lost < 1:
        raise MetricsError("empty", f"cell {cell} has no dots at all")
    median_px = float(np.median(e)) if n > 0 else math.nan
    return CellScore(cell, n, lost, median_px, lost / (n + lost), n >= min_dots)


@dataclass(frozen=True)
class V2Report:
    source: str
    status: str
    v2: float | None
    v2_partial: float | None
    n_measured: int
    worst_cell: str | None
    best_cell: str | None
    overall_median_px: float
    cells: tuple
    reasons: tuple
    gate_ok: bool
    product_pass: bool
    label: str


def _check_cell_keys(d, name: str) -> None:
    if not isinstance(d, dict) or set(d.keys()) != set(CELLS):
        got = sorted(d.keys()) if isinstance(d, dict) else type(d).__name__
        raise MetricsError("cells", f"{name} keys must be exactly the 9 cells, got {got}")


def v2_report(errors_by_cell: dict, lost_by_cell: dict, source, *, min_dots=MIN_DOTS,
              max_loss=MAX_LOSS, v1_cell_px=V1_CELL_PX, v1_all_px=V1_ALL_PX,
              v2_max=V2_MAX) -> V2Report:
    """V2 with the loss rule and the V1 checks in one place."""
    _check_cell_keys(errors_by_cell, "errors_by_cell")
    _check_cell_keys(lost_by_cell, "lost_by_cell")
    if source not in ("real", "sim"):
        raise MetricsError("source", f"source must be 'real' or 'sim', got {source!r}")

    scores = []
    all_errors = []
    for c in CELLS:
        e = _check_errors(errors_by_cell[c])
        lost = lost_by_cell[c]
        if e.size == 0 and _is_int(lost) and lost == 0:
            scores.append(CellScore(c, 0, 0, math.nan, 0.0, False))
        else:
            scores.append(cell_score(c, e, lost, min_dots=min_dots))
        all_errors.append(e)

    measured = [s for s in scores if s.measured]
    n_measured = len(measured)
    worst = best = None
    v2_partial = None
    if measured:
        worst = measured[0]
        best = measured[0]
        for s in measured[1:]:  # strict compare keeps the first in CELLS order on ties
            if s.median_px > worst.median_px:
                worst = s
            if s.median_px < best.median_px:
                best = s
        if n_measured >= 2:
            v2_partial = worst.median_px / best.median_px

    if n_measured == len(CELLS):
        status, v2 = "MEASURED", v2_partial
    else:
        status, v2 = "UNMEASURED", None

    pooled = np.concatenate(all_errors)
    overall = float(np.median(pooled)) if pooled.size > 0 else math.nan

    reasons = set()
    if status != "MEASURED":
        reasons.add("unmeasured")
    if v2 is not None and v2 > v2_max:
        reasons.add("v2")
    if any(s.loss > max_loss for s in scores):
        reasons.add("loss")
    if any(s.median_px > v1_cell_px for s in measured):
        reasons.add("v1_cell")
    if overall > v1_all_px:
        reasons.add("v1_all")
    reasons = tuple(sorted(reasons))
    gate_ok = reasons == ()

    return V2Report(
        source=source,
        status=status,
        v2=v2,
        v2_partial=v2_partial,
        n_measured=n_measured,
        worst_cell=worst.cell if worst else None,
        best_cell=best.cell if best else None,
        overall_median_px=overall,
        cells=tuple(scores),
        reasons=reasons,
        gate_ok=gate_ok,
        product_pass=gate_ok and source == "real",
        label="V2 (simulated)" if source == "sim" else "V2",
    )


def bootstrap_v2(errors_by_cell, seed, n_resamples=BOOT_RESAMPLES) -> tuple[float, float]:
    """95 % bootstrap interval of V2: resample dots inside each cell, fixed seed."""
    _check_cell_keys(errors_by_cell, "errors_by_cell")
    if not _is_int(seed):
        raise MetricsError("seed", f"seed must be an int, got {seed!r}")
    if not _is_int(n_resamples) or n_resamples < 100:
        raise MetricsError("resamples", f"n_resamples must be an int >= 100, got {n_resamples!r}")
    arrays = []
    for c in CELLS:
        e = _check_errors(errors_by_cell[c])
        if e.size < 1:
            raise MetricsError("unmeasured", f"cell {c} has no errors")
        arrays.append(e)
    rng = np.random.default_rng(seed)
    medians = np.empty((int(n_resamples), len(CELLS)), dtype=np.float64)
    for j, e in enumerate(arrays):
        idx = rng.integers(0, e.size, size=(int(n_resamples), e.size))
        medians[:, j] = np.median(e[idx], axis=1)
    ratios = medians.max(axis=1) / medians.min(axis=1)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(low), float(high)


def px_to_deg(e_px, mm_per_px, distance_mm) -> float:
    """Pixel error to visual angle: eps = arctan(e * p / d), in degrees."""
    try:
        e = float(e_px)
        p = float(mm_per_px)
        d = float(distance_mm)
    except (TypeError, ValueError):
        raise MetricsError("units", "e_px, mm_per_px, distance_mm must be numbers") from None
    if not (math.isfinite(e) and math.isfinite(p) and math.isfinite(d)) or p <= 0 or d <= 0 or e < 0:
        raise MetricsError("units", f"need e_px >= 0, mm_per_px > 0, distance_mm > 0; got {e_px}, {mm_per_px}, {distance_mm}")
    return math.degrees(math.atan(e * p / d))
