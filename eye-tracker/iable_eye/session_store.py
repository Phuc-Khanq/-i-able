"""The session store: one numbers-only .npz file per session, under the data root only.

A file holds the seven stacked FaceObs arrays, one metadata JSON string (fixed words and
numbers only, no free text) and up to 16 small per-frame channels. Never a picture.
A session is never overwritten. The exam flag is computed from the session index.

Standard library, numpy and the project only.
"""
from __future__ import annotations

import json
import math
import numbers
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from iable_eye.faceobs import SCHEMA_VERSION, STACK_KEYS, validate_stacked
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path

# ---------------------------------------------------------------- constants
STORE_VERSION = 1
EXAM_EVERY = 5
MAX_SESSION_INDEX = 9999
MAX_CHANNEL_COLUMNS = 16
MAX_CHANNELS = 16

SOURCES = ("real", "sim")
LIGHTS = ("bright", "dim", "backlit")
DISTANCES = ("near", "normal", "far")
POSTURES = ("upright", "leaning")
GLASSES = ("off", "on")
CAMERAS = ("builtin", "external")

_PERSON_RE = re.compile(r"^[PS][0-9]{3}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")          # fingerprint keys and channel names
_RECORDED_AT_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")
_MAX_FINGERPRINT = 64
_CHANNEL_PREFIX = "ch__"
_CHANNEL_DTYPES = (np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.bool_))
META_KEYS = ("store_version", "schema_version", "person", "session_index", "source", "is_exam",
             "light", "distance", "posture", "glasses", "camera", "recorded_at", "fingerprint",
             "n_frames")


class SessionStoreError(ValueError):
    """A session was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


# ---------------------------------------------------------------- names, places, exam flag
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_person(person) -> str:
    if not isinstance(person, str) or not _PERSON_RE.match(person) or person[1:] == "000":
        raise SessionStoreError("person", f"person must be P001-P999 or S001-S999, got {person!r}")
    return person


def _check_source(person: str, source) -> str:
    if not isinstance(source, str) or source not in SOURCES:
        raise SessionStoreError("source", f"source must be one of {SOURCES}, got {source!r}")
    if (person[0] == "P") != (source == "real"):
        raise SessionStoreError("source", f"person {person} does not match source {source!r}")
    return source


def _check_index(session_index) -> int:
    if not _is_int(session_index) or not 1 <= session_index <= MAX_SESSION_INDEX:
        raise SessionStoreError("index", f"session index must be an integer in [1, {MAX_SESSION_INDEX}], "
                                         f"got {session_index!r}")
    return session_index


def exam_flag(session_index) -> bool:
    """Session k is an exam when k mod 5 == 0 (it is never trained on)."""
    return _check_index(session_index) % EXAM_EVERY == 0


def _person_dir(person: str, source: str, data_root) -> Path:
    base = Path(data_root) / "SIM" if source == "sim" else Path(data_root)
    return base / person / "sessions"


def session_path(person, session_index, source, data_root=DATA_ROOT, root=ROOT) -> Path:
    """The file of one session. Checks person, source, index, then the path guard.
    A DataPathError goes up unchanged. Creates nothing."""
    _check_person(person)
    _check_source(person, source)
    _check_index(session_index)
    path = _person_dir(person, source, data_root) / f"s{session_index:04d}.npz"
    return assert_data_path(path, data_root=data_root, root=root)


# ---------------------------------------------------------------- metadata
@dataclass(frozen=True, kw_only=True)
class SessionMeta:
    """What the caller gives. No free-text field: a name has no place to go.
    There is no is_exam field: the exam flag is computed from session_index."""

    person: str
    session_index: int
    source: str
    light: str
    distance: str
    posture: str = "upright"
    glasses: str = "off"
    camera: str = "builtin"
    recorded_at: str           # like 2026-11-13T07:45:00 (local time); the store never reads the clock
    fingerprint: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Session:
    """One session read back from its file."""

    arrays: dict
    meta: SessionMeta
    channels: dict
    is_exam: bool
    n_frames: int
    path: Path


def _check_word(code: str, name: str, value, allowed) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise SessionStoreError(code, f"{name} must be one of {allowed}, got {value!r}")


def _check_recorded_at(v) -> None:
    if not isinstance(v, str) or not _RECORDED_AT_RE.match(v):
        raise SessionStoreError("recorded_at", f"recorded_at must look like 2026-11-13T07:45:00, got {v!r}")
    try:
        datetime.fromisoformat(v)
    except ValueError:
        raise SessionStoreError("recorded_at", f"recorded_at is not a real date: {v!r}") from None


def _check_fingerprint(fp) -> dict:
    """Returns a copy with plain float values."""
    if not isinstance(fp, dict):
        raise SessionStoreError("fingerprint", f"fingerprint must be a dict, got {type(fp).__name__}")
    if len(fp) > _MAX_FINGERPRINT:
        raise SessionStoreError("fingerprint", f"fingerprint has {len(fp)} entries (at most {_MAX_FINGERPRINT})")
    out = {}
    for k, v in fp.items():
        if not isinstance(k, str) or not _NAME_RE.match(k):
            raise SessionStoreError("fingerprint", f"fingerprint key {k!r} must match {_NAME_RE.pattern}")
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
            raise SessionStoreError("fingerprint", f"fingerprint value of {k!r} must be a finite number")
        out[k] = float(v)
    return out


def _check_meta_rest(meta: SessionMeta) -> dict:
    """Checks 5-8 (cell, tag, recorded_at, fingerprint). Returns the checked fingerprint."""
    _check_word("cell", "light", meta.light, LIGHTS)
    _check_word("cell", "distance", meta.distance, DISTANCES)
    _check_word("tag", "posture", meta.posture, POSTURES)
    _check_word("tag", "glasses", meta.glasses, GLASSES)
    _check_word("tag", "camera", meta.camera, CAMERAS)
    _check_recorded_at(meta.recorded_at)
    return _check_fingerprint(meta.fingerprint)


# ---------------------------------------------------------------- channels
def _check_channels(channels, n: int) -> dict:
    """Per-frame channels: small numeric arrays only (this keeps a picture out)."""
    if channels is None:
        return {}
    if not isinstance(channels, dict):
        raise SessionStoreError("channel_name", f"channels must be a dict, got {type(channels).__name__}")
    if len(channels) > MAX_CHANNELS:
        raise SessionStoreError("channel_name", f"{len(channels)} channels (at most {MAX_CHANNELS})")
    for name, a in channels.items():
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise SessionStoreError("channel_name", f"channel name {name!r} must match {_NAME_RE.pattern}")
        if not isinstance(a, np.ndarray) or a.dtype not in _CHANNEL_DTYPES:
            dt = a.dtype if isinstance(a, np.ndarray) else type(a).__name__
            raise SessionStoreError("channel_dtype", f"channel {name!r}: dtype {dt} not in "
                                                     f"float32, float64, int32, bool")
        ok_shape = (a.ndim == 1 and a.shape[0] == n) or (
            a.ndim == 2 and a.shape[0] == n and 1 <= a.shape[1] <= MAX_CHANNEL_COLUMNS)
        if not ok_shape:
            raise SessionStoreError("channel_shape", f"channel {name!r}: shape {a.shape}, need ({n},) or "
                                                     f"({n}, k) with 1 <= k <= {MAX_CHANNEL_COLUMNS}")
        if a.dtype.kind == "f" and np.isinf(a).any():
            raise SessionStoreError("channel_value", f"channel {name!r} has inf (NaN is allowed)")
    return dict(channels)


# ---------------------------------------------------------------- write
def _meta_dict(meta: SessionMeta, fingerprint: dict, n: int) -> dict:
    return {
        "store_version": STORE_VERSION, "schema_version": SCHEMA_VERSION,
        "person": meta.person, "session_index": meta.session_index, "source": meta.source,
        "is_exam": exam_flag(meta.session_index),
        "light": meta.light, "distance": meta.distance, "posture": meta.posture,
        "glasses": meta.glasses, "camera": meta.camera, "recorded_at": meta.recorded_at,
        "fingerprint": fingerprint, "n_frames": n,
    }


def write_session(arrays, meta: SessionMeta, channels=None, data_root=DATA_ROOT, root=ROOT) -> Path:
    """Write one session to <data_root>/P###/sessions/s####.npz (or SIM/S###/...).
    Everything is checked before anything is created. Never overwrites."""
    if not isinstance(meta, SessionMeta):
        raise SessionStoreError("meta", f"meta must be a SessionMeta, got {type(meta).__name__}")
    path = session_path(meta.person, meta.session_index, meta.source, data_root=data_root, root=root)  # 1-4
    fingerprint = _check_meta_rest(meta)                                                              # 5-8
    n = validate_stacked(arrays)                                                                      # 9
    chans = _check_channels(channels, n)                                                              # 10
    if path.exists():                                                                                 # 11
        raise SessionStoreError("exists", f"{path.name} already exists; a session is never overwritten")
    payload = {k: arrays[k] for k in STACK_KEYS}
    payload["meta_json"] = np.array(json.dumps(_meta_dict(meta, fingerprint, n), sort_keys=True))
    for name, a in chans.items():
        payload[_CHANNEL_PREFIX + name] = a
    path.parent.mkdir(parents=True, exist_ok=True)
    # S010: a unique tmp name, then os.link (an exclusive create: it fails if the file exists,
    # also on Windows, where os.replace would overwrite). The tmp file is always removed.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "xb") as fh:  # a handle: np.savez adds no ".npz" to the name
            np.savez(fh, **payload)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise SessionStoreError("exists", f"{path.name} already exists; a session is never overwritten") \
                from None
    finally:
        tmp.unlink(missing_ok=True)
    return path


def next_free_index(person, source, data_root=DATA_ROOT, root=ROOT) -> int:
    """max(list_sessions) + 1, or 1 for a new person. Never fills a gap (sessions stay in time order)."""
    have = list_sessions(person, source, data_root=data_root, root=root)
    nxt = (max(have) + 1) if have else 1
    return _check_index(nxt)


# ---------------------------------------------------------------- read
def _load_members(path: Path) -> dict:
    """Every member of the .npz, loaded with pickle switched off. Refused: "format"."""
    try:
        with np.load(path, allow_pickle=False) as z:  # 'with' closes the file (Windows locks it)
            if not isinstance(z, np.lib.npyio.NpzFile):
                raise ValueError("not an .npz archive")
            return {name: z[name] for name in z.files}  # an object array raises ValueError here
    except SessionStoreError:
        raise
    except Exception as e:  # BadZipFile, OSError, EOFError, ValueError (pickle), ...
        raise SessionStoreError("format", f"{path.name} is not a readable numbers-only .npz: "
                                          f"{type(e).__name__}") from None


def _parse_meta(m) -> dict:
    if not isinstance(m, np.ndarray) or m.ndim != 0 or m.dtype.kind != "U":
        raise SessionStoreError("meta", "meta_json must be a 0-d string array")
    try:
        d = json.loads(str(m))
    except ValueError:
        raise SessionStoreError("meta", "meta_json is not JSON") from None
    if not isinstance(d, dict) or set(d) != set(META_KEYS):
        raise SessionStoreError("meta", f"meta_json must be an object with exactly the keys {META_KEYS}")
    if not isinstance(d["is_exam"], bool):
        raise SessionStoreError("meta", "is_exam must be a bool")
    if not _is_int(d["n_frames"]):
        raise SessionStoreError("meta", "n_frames must be an integer")
    return d


def read_session(person, session_index, source, data_root=DATA_ROOT, root=ROOT) -> Session:
    """Read one session back, checking everything again (the file is not trusted)."""
    path = session_path(person, session_index, source, data_root=data_root, root=root)  # 1-4
    if not path.is_file():                                                              # 5
        raise SessionStoreError("missing", f"{path.name} does not exist")
    members = _load_members(path)                                                       # 6
    keys = set(members)
    extra = {k for k in keys - set(STACK_KEYS) - {"meta_json"} if not k.startswith(_CHANNEL_PREFIX)}
    if not set(STACK_KEYS) <= keys or "meta_json" not in keys or extra:                 # 7
        raise SessionStoreError("keys", f"bad key set: missing {sorted((set(STACK_KEYS) | {'meta_json'}) - keys)}, "
                                        f"extra {sorted(extra)}")
    d = _parse_meta(members["meta_json"])                                               # 8
    for k, want in (("store_version", STORE_VERSION), ("schema_version", SCHEMA_VERSION)):  # 9
        if not _is_int(d[k]) or d[k] != want:
            raise SessionStoreError("version", f"{k} is {d[k]!r}, this code reads {want}")
    _check_person(d["person"])                                                          # 10
    _check_source(d["person"], d["source"])
    _check_index(d["session_index"])
    try:
        meta = SessionMeta(person=d["person"], session_index=d["session_index"], source=d["source"],
                           light=d["light"], distance=d["distance"], posture=d["posture"],
                           glasses=d["glasses"], camera=d["camera"], recorded_at=d["recorded_at"],
                           fingerprint=d["fingerprint"])
    except TypeError:
        raise SessionStoreError("meta", "meta_json fields do not fit SessionMeta") from None
    _check_meta_rest(meta)
    if (d["person"], d["session_index"], d["source"]) != (person, session_index, source):  # 11
        raise SessionStoreError("mismatch", f"the file says {d['person']} s{d['session_index']:04d} "
                                            f"{d['source']}, but {path.name} was asked for")
    if d["is_exam"] != exam_flag(session_index):                                        # 12
        raise SessionStoreError("exam_flag", f"stored is_exam={d['is_exam']} but session {session_index} "
                                             f"has exam flag {exam_flag(session_index)}")
    arrays = {k: members[k] for k in STACK_KEYS}
    n = validate_stacked(arrays)                                                        # 13
    channels = {k[len(_CHANNEL_PREFIX):]: v for k, v in members.items() if k.startswith(_CHANNEL_PREFIX)}
    _check_channels(channels, n)                                                        # 14
    if d["n_frames"] != n:                                                              # 15
        raise SessionStoreError("mismatch", f"n_frames is {d['n_frames']} but the arrays have {n} frames")
    return Session(arrays=arrays, meta=meta, channels=channels, is_exam=d["is_exam"], n_frames=n, path=path)


def list_sessions(person, source, data_root=DATA_ROOT, root=ROOT) -> list:
    """Sorted indexes of the s####.npz files of one person. Opens no file."""
    _check_person(person)
    _check_source(person, source)
    folder = assert_data_path(_person_dir(person, source, data_root), data_root=data_root, root=root)
    if not folder.is_dir():
        return []
    out = []
    for p in folder.iterdir():
        m = re.match(r"^s([0-9]{4})\.npz$", p.name)
        if m and p.is_file() and 1 <= int(m.group(1)) <= MAX_SESSION_INDEX:
            out.append(int(m.group(1)))
    return sorted(out)
