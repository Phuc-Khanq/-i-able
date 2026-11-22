"""S064: record every demo run as numbers (FaceObs + cursor path + labels). No image, ever.

File: <data_root>/<person>/demo/d<YYYYmmdd_HHMMSS>.npz. This folder is NOT the session store
(<person>/sessions), so demo runs never change the session numbers k or the exam schedule.

Labels (one row each: kind, t0, t1, x_px, y_px) say where the person looked during [t0, t1]:
    calib      fusion start: one calibration dot (strong label: "look at the dot")
    click_hit  click test: the dwell before a hit click on the target (the person looked at it)
    reach      reach test: the last 0.3 s before the target was reached (weak label)
Training may use them only through the pre-registered check (DECISIONS 2026-11-22, rule D1).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from iable_eye import faceobs
from iable_eye import session_store as store
from iable_eye.features import session_features
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path
from iable_eye.quality import OK, REAL_GATE_V1, gate_session

DEMO_DIR = "demo"
RECORD_VERSION = 1
LABEL_KINDS = ("calib", "click_hit", "reach")
_CH = ("x_px", "y_px", "gaze_x", "gaze_y", "code", "head_ok", "paused")


class DemoRecordError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def demo_dir(person, data_root=DATA_ROOT, root=ROOT) -> Path:
    store._check_person(person)
    return assert_data_path(Path(data_root) / person / DEMO_DIR, data_root=data_root, root=root)


def list_demos(person, data_root=DATA_ROOT, root=ROOT) -> list:
    d = demo_dir(person, data_root, root)
    return sorted(p for p in d.glob("d*.npz")) if d.is_dir() else []


class DemoRecorder:
    def __init__(self, person: str, meta: dict):
        store._check_person(person)
        self.person, self.meta = person, dict(meta)
        self.obs, self.ch, self.labels = [], {k: [] for k in _CH}, []

    def add(self, obs, step=None) -> None:
        self.obs.append(obs)
        if step is None:
            vals = (np.nan, np.nan, np.nan, np.nan, -1, False, False)
        else:
            vals = (step.x_px, step.y_px, step.gaze_x, step.gaze_y, step.code, step.head_ok, step.paused)
        for k, v in zip(_CH, vals):
            self.ch[k].append(v)

    def add_label(self, kind, t0, t1, x, y) -> None:
        if kind not in LABEL_KINDS:
            raise DemoRecordError("label", f"kind must be one of {LABEL_KINDS}, got {kind!r}")
        if not (np.isfinite([t0, t1, x, y]).all() and t1 > t0):
            raise DemoRecordError("label", "label needs finite t0 < t1, x, y")
        self.labels.append((LABEL_KINDS.index(kind), float(t0), float(t1), float(x), float(y)))

    def save(self, stamp: str, data_root=DATA_ROOT, root=ROOT) -> Path | None:
        """Write once; None when there is no frame. Never overwrites."""
        if not self.obs:
            return None
        d = demo_dir(self.person, data_root, root)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"d{stamp}.npz"
        if path.exists():
            raise DemoRecordError("exists", f"{path} exists")
        arrays = faceobs.stack(self.obs)
        ch = {f"ch_{k}": np.asarray(v, dtype=(np.int16 if k == "code" else bool if k in ("head_ok", "paused")
                                              else np.float32)) for k, v in self.ch.items()}
        lab = np.array(self.labels, dtype=np.float64).reshape(-1, 5)
        meta = dict(self.meta, person=self.person, record_version=RECORD_VERSION, n_frames=len(self.obs),
                    label_kinds=list(LABEL_KINDS))
        tmp = path.with_name(path.stem + ".tmp.npz")
        try:
            np.savez_compressed(tmp, meta=np.array(json.dumps(meta, sort_keys=True)), labels=lab, **arrays, **ch)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path


def read_demo(path) -> dict:
    """-> {"arrays": stacked FaceObs dict, "labels": (n, 5), "meta": dict, "ch": dict}."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in faceobs.STACK_KEYS}
        meta = json.loads(str(z["meta"]))
        ch = {k[3:]: z[k].copy() for k in z.files if k.startswith("ch_")}
        lab = z["labels"].copy()
    faceobs.validate_stacked(arrays)
    return {"arrays": arrays, "labels": lab, "meta": meta, "ch": ch}


def label_rows(demo: dict, kinds=("calib",), gate=REAL_GATE_V1) -> tuple:
    """Training rows from a read_demo dict: (X (n, 11), Y (n, 2) label px, D (n,) label index).
    Frames: gate code OK and t in [t0, t1) of a label of the given kinds."""
    a, lab = demo["arrays"], demo["labels"]
    feats = session_features(a)
    ok = gate_session(a, config=gate, features=feats) == OK
    t = np.asarray(a["t"], dtype=np.float64)
    want = [LABEL_KINDS.index(k) for k in kinds]
    X, Y, D = [], [], []
    for j, (kind, t0, t1, x, y) in enumerate(lab):
        if int(kind) not in want:
            continue
        ix = np.flatnonzero(ok & (t >= t0) & (t < t1))
        X.append(feats[ix])
        Y.append(np.tile([x, y], (len(ix), 1)))
        D.append(np.full(len(ix), j, dtype=np.int64))
    if not X:
        return np.zeros((0, feats.shape[1])), np.zeros((0, 2)), np.zeros(0, np.int64)
    return np.vstack(X), np.vstack(Y), np.concatenate(D)
