"""Evaluation runner on top of `metrics.py`, bound to the frozen gates file (S017).

Pre-registration: the gates file was written before any result. Its SHA-256 is
recorded in a separate file. Every evaluation call re-hashes the raw bytes and
refuses to score if the hash differs. Reads only those two files; writes nothing.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iable_eye import metrics


class GateError(ValueError):
    """An evaluation request was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
GATES_PATH = CONFIG_DIR / "gates_v1.json"
GATES_SHA_PATH = CONFIG_DIR / "gates_v1.sha256"
TARGET_KEYS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
TOP_KEYS = frozenset({"gates_version", "cells", "settled_window_ms", "lost_if_ok_frac_below",
                      "selection_rule", "targets"})
KINDS = ("OPEN", "LIVE")
ORIGINS = ("replay", "live")
_HEX = frozenset("0123456789abcdef")


def sha256_file(path) -> str:
    """SHA-256 of the raw bytes (no text decoding, so line endings count)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH, sha_path=GATES_SHA_PATH) -> dict:
    """Load the gates file only if its hash equals the recorded one."""
    recorded = Path(sha_path).read_text(encoding="ascii").strip()
    if len(recorded) != 64 or not set(recorded) <= _HEX:
        raise GateError("sha_file", f"sha file must hold 64 lowercase hex chars, got {len(recorded)} chars")
    actual = sha256_file(path)
    if actual != recorded:
        raise GateError("hash", f"gates file changed: recorded {recorded[:12]}..., file {actual[:12]}...")
    g = json.loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(g, dict) or set(g.keys()) != TOP_KEYS:
        got = sorted(g.keys()) if isinstance(g, dict) else type(g).__name__
        raise GateError("keys", f"top-level keys must be {sorted(TOP_KEYS)}, got {got}")
    t = g["targets"]
    if not isinstance(t, dict) or set(t.keys()) != set(TARGET_KEYS):
        got = sorted(t.keys()) if isinstance(t, dict) else type(t).__name__
        raise GateError("keys", f"targets must be exactly {list(TARGET_KEYS)}, got {got}")
    for k in TARGET_KEYS:
        if not isinstance(t[k], dict) or t[k].get("kind") not in KINDS:
            raise GateError("kind", f"target {k} kind must be one of {KINDS}")
    if g["cells"] != list(metrics.CELLS):
        raise GateError("cells", f"cells must equal metrics.CELLS, got {g['cells']}")
    out = dict(g)
    out["_sha256"] = actual
    return out


@dataclass(frozen=True)
class DotRow:
    person: str
    session_index: int
    source: str
    cell: str
    is_exam: bool
    error_px: float | None  # None = a lost dot
    origin: str  # "replay" or "live"


def check_rows(rows, target, gates) -> None:
    """Refuse rows that may not be scored for this target."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, DotRow) for r in rows):
        raise GateError("rows", "rows must be a non-empty list of DotRow")
    if target not in TARGET_KEYS:
        raise GateError("target", f"target must be one of {TARGET_KEYS}, got {target!r}")
    sources = {r.source for r in rows}
    if len(sources) != 1:
        raise GateError("mixed_source", f"rows mix sources {sorted(sources)}")
    persons = {r.person for r in rows}
    if len(persons) != 1:
        raise GateError("mixed_person", f"rows mix persons {sorted(persons)}")
    n_train = sum(1 for r in rows if r.is_exam is not True)
    if n_train:
        raise GateError("not_exam", f"{n_train} rows are not exam rows")
    bad = {r.origin for r in rows} - set(ORIGINS)
    if bad:
        raise GateError("origin", f"origin must be one of {ORIGINS}, got {sorted(bad)}")
    if gates["targets"][target]["kind"] == "LIVE" and any(r.origin == "replay" for r in rows):
        raise GateError("live_from_replay", f"{target} is a LIVE target; replayed rows cannot score it")


def evaluate_v2(rows, path=GATES_PATH, sha_path=GATES_SHA_PATH) -> tuple:
    """V2 report, bootstrap interval (or None) and the gates hash used."""
    gates = load_gates(path, sha_path)  # every call: a changed file is caught here
    check_rows(rows, "V2", gates)
    v1 = gates["targets"]["V1"]
    v2 = gates["targets"]["V2"]
    vals = {c: [] for c in metrics.CELLS}
    lost = {c: 0 for c in metrics.CELLS}
    for r in rows:
        if r.cell not in vals:
            raise GateError("cell", f"cell {r.cell!r} is not one of the 9 cells")
        if r.error_px is None:
            lost[r.cell] += 1
        else:
            vals[r.cell].append(float(r.error_px))
    errors = {c: np.asarray(vals[c], dtype=np.float64) for c in metrics.CELLS}
    report = metrics.v2_report(errors, lost, rows[0].source, min_dots=v2["min_dots"],
                               max_loss=v2["max_loss"], v1_cell_px=v1["cell_max_px"],
                               v1_all_px=v1["all_max_px"], v2_max=v2["max_ratio"])
    interval = None
    if report.status == "MEASURED":
        interval = metrics.bootstrap_v2(errors, seed=v2["boot_seed"], n_resamples=v2["boot_resamples"])
    return report, interval, gates["_sha256"]
