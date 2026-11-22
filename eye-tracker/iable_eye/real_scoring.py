"""S047b: load stored sessions for the frozen scoring pipeline, with one timeline per source.

Two paths, chosen by the caller:
- protocol path (`grid=None`): a recorded session (proto_form 2.0 = form "session"). Onsets and
  truth come from `protocol.build_timeline("session", w, h)`, dots of block "grid" only.
- sim path (`grid=(onsets, positions)`): a simulator session (no proto_form). The tool passes the
  simulator grid in, so this module never imports a simulator module.
The returned dict has the keys of `tools.run_gaze_baseline.load_session`, plus `screen` and `dot_px`.
Writes only the scores file (`write_scores`), and only inside the data root.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from iable_eye import session_store as store
from iable_eye.dots import dot_windows
from iable_eye.features import session_features
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path
from iable_eye.protocol import build_timeline
from iable_eye.quality import OK, GateConfig, gate_session
from iable_eye.splits import info_from_meta

CHOSEN_KIND = "kernel_ridge"   # pre-registered (DECISIONS 2026-11-19); never chosen here
RESULTS_DIR = "results"
SCORES_VERSION = 1
GRID_BLOCK = "grid"
SESSION_FORM = "session"
SESSION_FORM_CODE = 2.0        # recorder.FORM_CODES["session"]
SIM_DEFAULT_SCREEN = (1920, 1080)


class ScoreError(ValueError):
    """`code`: "none" (no session), "fps", "screen", "form", "path"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _grid_dots(screen_w: int, screen_h: int) -> list:
    tl = build_timeline(SESSION_FORM, screen_w, screen_h)
    return sorted((d for d in tl.dots if d.block == GRID_BLOCK), key=lambda d: d.index)


def grid_onsets(screen_w: int, screen_h: int) -> list[float]:
    """Onsets (s) of the dots of block "grid" of form "session", in index order."""
    return [float(d.on_s) for d in _grid_dots(screen_w, screen_h)]


def grid_px(screen_w: int, screen_h: int) -> np.ndarray:
    """(40, 2) float64: (x_px, y_px) of the same dots."""
    return np.array([(d.x_px, d.y_px) for d in _grid_dots(screen_w, screen_h)], dtype=np.float64)


def _screen(fp: dict, source: str, allow_default: bool) -> tuple:
    if "screen_w_px" in fp and "screen_h_px" in fp:
        return int(fp["screen_w_px"]), int(fp["screen_h_px"])
    if allow_default and source == "sim":
        return SIM_DEFAULT_SCREEN
    raise ScoreError("screen", "the session has no screen_w_px / screen_h_px")


def load_store_session(person, k, source, window_ms, grid=None, data_root=DATA_ROOT, root=ROOT,
                       gate_config=GateConfig()) -> dict | None:
    """One stored session -> the load_session dict (+ screen, dot_px), or None if its form is not "session".
    gate_config: GateConfig() for the simulator (default, unchanged); REAL_GATE_V1 for real data (S059)."""
    s = store.read_session(person, k, source, data_root=data_root, root=root)
    fp, ch = s.meta.fingerprint, s.channels
    form = fp.get("proto_form")
    if form is not None and form != SESSION_FORM_CODE:
        return None                                                   # rule 1: skipped
    if grid is None:                                                  # rule 2: protocol path
        if form is None or "block" not in ch:
            raise ScoreError("form", f"s{k:04d}: the protocol path needs proto_form 2.0 and a block channel")
        w, h = _screen(fp, source, allow_default=False)
        gi = [b.name for b in build_timeline(SESSION_FORM, w, h).blocks].index(GRID_BLOCK)
        dot = np.where(ch["block"] == gi, ch["dot"], -1)
        onsets, dot_px = grid_onsets(w, h), grid_px(w, h)
    else:                                                             # rule 3: sim path
        if source != "sim" or form is not None:
            raise ScoreError("form", f"s{k:04d}: a sim grid is only for simulator sessions without proto_form")
        w, h = _screen(fp, source, allow_default=True)
        onsets, positions = grid
        dot, dot_px = ch["dot"], np.asarray(positions, dtype=np.float64)
    if "nominal_index" in ch:                                         # rule 4: time and fps
        if "sim_fps" not in fp:
            raise ScoreError("fps", f"s{k:04d}: nominal_index but no sim_fps")
        fps = float(fp["sim_fps"])
        tau = ch["nominal_index"] / fps
    else:
        if "fps" not in fp:
            raise ScoreError("fps", f"s{k:04d}: no fps in the fingerprint")
        fps = float(fp["fps"])
        tau = np.asarray(s.arrays["t"], dtype=np.float64)            # the recorder labelled with state_at(tl, t)
    feats = session_features(s.arrays)
    ok = gate_session(s.arrays, config=gate_config, features=feats) == OK
    if np.isnan(feats[ok]).any():
        raise RuntimeError(f"session {k}: an ok frame has a NaN feature")
    win = dot_windows(tau, dot, onsets, window_ms)
    return {
        "k": k, "info": info_from_meta(s.meta), "cell": f"{s.meta.light}/{s.meta.distance}",
        "feats": feats, "ok": ok, "gaze_px": ch["gaze_px"], "windows": win,
        "n_expected": int(round(0.8 * fps)), "n_frames": int(ok.size), "tau": tau,
        "screen": (w, h), "dot_px": dot_px,
    }


def scores_path(person, data_root=DATA_ROOT, root=ROOT) -> Path:
    """<data_root>/results/<person>_scores.json, checked by the path guard (a DataPathError goes up
    unchanged). Creates nothing."""
    store._check_person(person)
    return assert_data_path(Path(data_root) / RESULTS_DIR / f"{person}_scores.json",
                            data_root=data_root, root=root)


def write_scores(obj: dict, path) -> None:
    """Sorted JSON; a .tmp file, then os.replace (one file, never half written)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, sort_keys=True, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
