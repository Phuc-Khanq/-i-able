"""S059: save / load a trained gaze Candidate as plain numpy arrays (no pickle), in the data root only.

File: <data_root>/models/<person>_gaze_<kind>_v<version>.npz. Kinds: ridge, kernel_ridge
(an mlp holds a fitted sklearn object, so it is refused: it would need pickle).
A load gives a Candidate whose predict_candidate output equals the saved one exactly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from iable_eye import session_store as store
from iable_eye.candidates import Candidate
from iable_eye.gaze_baseline import RidgeMap
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path

MODELS_DIR = "models"
STORE_VERSION = 1
SAVE_KINDS = ("ridge", "kernel_ridge")


class ModelStoreError(ValueError):
    """`.code`: "kind", "file", "version"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def model_path(person: str, kind: str, version: int = 1, data_root=DATA_ROOT, root=ROOT) -> Path:
    store._check_person(person)
    if kind not in SAVE_KINDS:
        raise ModelStoreError("kind", f"kind must be one of {SAVE_KINDS}, got {kind!r}")
    return assert_data_path(Path(data_root) / MODELS_DIR / f"{person}_gaze_{kind}_v{int(version)}.npz",
                            data_root=data_root, root=root)


def save_candidate(c: Candidate, path, info: dict | None = None) -> Path:
    """Write the arrays to a .tmp file, then os.replace. `info`: a small JSON-able dict (numbers, words)."""
    if not isinstance(c, Candidate) or c.kind not in SAVE_KINDS:
        raise ModelStoreError("kind", f"only {SAVE_KINDS} Candidates can be saved")
    b = c.base
    keys = sorted(b.shifts)
    arrays = {
        "kind": np.array(c.kind), "store_version": np.array(STORE_VERSION),
        "mean": b.mean, "scale": b.scale, "lo": b.lo, "hi": b.hi, "W": b.W,
        "shift_keys": np.array(keys, dtype=np.int64),
        "shift_vals": np.array([b.shifts[k] for k in keys], dtype=np.float64).reshape(-1, 2),
        "default_shift": b.default_shift, "alpha": np.array(float(b.alpha)),
        "gamma": np.array(float(c.gamma)),
        "info": np.array(json.dumps(info or {}, sort_keys=True)),
    }
    if c.kind == "kernel_ridge":
        arrays["A"], arrays["C"] = c.A, c.C
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_candidate(path) -> tuple[Candidate, dict]:
    """(Candidate, info). Never unpickles."""
    path = Path(path)
    if not path.is_file():
        raise ModelStoreError("file", f"no model file: {path}")
    with np.load(path, allow_pickle=False) as z:
        if int(z["store_version"]) != STORE_VERSION:
            raise ModelStoreError("version", f"store_version {int(z['store_version'])} != {STORE_VERSION}")
        kind = str(z["kind"])
        if kind not in SAVE_KINDS:
            raise ModelStoreError("kind", f"unknown kind {kind!r} in {path}")
        shifts = {int(k): np.array(v, dtype=np.float64) for k, v in zip(z["shift_keys"], z["shift_vals"])}
        base = RidgeMap(mean=z["mean"].copy(), scale=z["scale"].copy(), lo=z["lo"].copy(), hi=z["hi"].copy(),
                        W=z["W"].copy(), shifts=shifts, default_shift=z["default_shift"].copy(),
                        alpha=float(z["alpha"]))
        A = z["A"].copy() if kind == "kernel_ridge" else None
        C = z["C"].copy() if kind == "kernel_ridge" else None
        info = json.loads(str(z["info"]))
        return Candidate(kind, base, A, C, float(z["gamma"]), None), info
