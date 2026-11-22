"""The 9 frozen simulated cells (light x distance), and a hash that pins them.

A cell is one light word and one distance word. Every number here is a Python float
(only the two version numbers are ints), so the JSON text, and so the hash, is stable.
If you change any number, the hash test fails on purpose: then a new PRESET_VERSION is needed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from iable_eye import session_store as _store
from iable_eye import sim_protocol as _proto
from iable_eye.sim_core import HeadPose, SimError
from iable_eye.sim_env import ObsKnobs
from iable_eye.sim_events import EventKnobs

PRESET_VERSION = 1
LIGHTS = ("bright", "dim", "backlit")
DISTANCES = ("near", "normal", "far")
assert LIGHTS == _store.LIGHTS and DISTANCES == _store.DISTANCES

# cell number c = 3 * light index + distance index
CELL_ORDER = tuple((light, dist) for light in LIGHTS for dist in DISTANCES)

DISTANCE_MM = {"near": 450.0, "normal": 600.0, "far": 800.0}
_LIGHT_OBS = {"bright": (30.0, 0.3), "dim": (10.0, 1.0), "backlit": (20.0, 0.7)}  # fps, noise_px

SITTING_POS_SD_MM = 20.0
SITTING_ANGLE_SD_DEG = 2.0
VIEW_TY_FRAC = 0.1   # eyes sit this part of the distance above the lens

PRESETS_SHA256 = "e139e37328f8140811ab9a79fe13addeed999788dd012297a867da4f6e6176d6"


@dataclass(frozen=True)
class CellPreset:
    light: str
    distance: str
    distance_mm: float
    obs: ObsKnobs
    events: EventKnobs


def _events() -> EventKnobs:
    return EventKnobs(blink_rate_per_min=15.0, blink_min_ms=100.0, blink_max_ms=400.0,
                      glare_rate_per_min=0.0, drift_deg_per_20s=1.5, drift_mm_per_20s=10.0,
                      leak_yaw_deg_per_mm=0.02, leak_pitch_deg_per_mm=0.02)


def preset(light, distance) -> CellPreset:
    if light not in LIGHTS or distance not in DISTANCES:
        raise SimError("cell", f"unknown cell ({light!r}, {distance!r})")
    fps, noise = _LIGHT_OBS[light]
    obs = ObsKnobs(fps=fps, drop_prob=0.02, jitter_ms=3.0, noise_px=noise, noise_rho=0.73,
                   noise_common_frac=0.5, pose_noise_deg=0.1)
    return CellPreset(light=light, distance=distance, distance_mm=DISTANCE_MM[distance],
                      obs=obs, events=_events())


def cell_of_session(k) -> tuple:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 9999:
        raise SimError("index", f"session index must be an int in 1..9999, got {k!r}")
    return CELL_ORDER[(k - 1) % 9]


def base_pose(distance_mm) -> HeadPose:
    if isinstance(distance_mm, bool) or not isinstance(distance_mm, (int, float)):
        raise SimError("pose", f"distance_mm must be a number, got {distance_mm!r}")
    d = float(distance_mm)
    if not 350.0 <= d <= 1000.0:   # NaN fails this too
        raise SimError("pose", f"distance_mm must be in [350, 1000], got {d!r}")
    return HeadPose(0.0, 0.0, 0.0, 0.0, VIEW_TY_FRAC * d, -d)


def presets_obj() -> dict:
    cells = []
    for light, dist in CELL_ORDER:
        p = preset(light, dist)
        cells.append({"light": p.light, "distance": p.distance, "distance_mm": p.distance_mm,
                      "obs": asdict(p.obs), "events": asdict(p.events)})
    return {"preset_version": PRESET_VERSION,
            "protocol_version": _proto.PROTOCOL_VERSION,
            "cells": cells,
            "sitting": {"pos_sd_mm": SITTING_POS_SD_MM, "angle_sd_deg": SITTING_ANGLE_SD_DEG},
            "view_ty_frac": VIEW_TY_FRAC,
            "protocol": {"blocks": [list(b) for b in _proto.BLOCKS],
                         "dot_s": _proto.DOT_S, "latency_s": _proto.GAZE_LATENCY_S,
                         "grid": [float(_proto.GRID_COLS), float(_proto.GRID_ROWS), _proto.GRID_MARGIN]}}


def presets_json() -> str:
    return json.dumps(presets_obj(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def presets_sha256() -> str:
    return hashlib.sha256(presets_json().encode("utf-8")).hexdigest()
