"""One simulated session, built in memory by one fixed chain, and a writer for a sim person.

The chain: cell -> preset -> seed -> timeline and events -> protocol and sitting shift
-> one noise-free frame per kept frame -> blinks -> tracker noise -> glare and leak.
The simulator never reads the clock: `recorded_at` comes from the session index.
Every number is (simulated).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import datetime, timedelta

import numpy as np

from iable_eye import session_store as _store
from iable_eye.faceobs import stack
from iable_eye.paths import ROOT
from iable_eye.sim_core import (SimCamera, SimError, SimPerson, SimScreen, _record_unchecked,
                                _simulate_frame_fast, check_camera, check_person, screen_px_to_cam_mm)
from iable_eye.sim_env import _is_int, _is_real, add_noise, make_rngs, timeline
from iable_eye.sim_events import (apply_events_after_noise, apply_events_before_noise, drifted_pose,
                                  event_schedule, make_event_rngs)
from iable_eye.sim_presets import (CELL_ORDER, PRESET_VERSION, PRESETS_SHA256, SITTING_ANGLE_SD_DEG,
                                   SITTING_POS_SD_MM, CellPreset, base_pose, cell_of_session, preset)
from iable_eye.sim_protocol import PROTOCOL_DURATION_S, PROTOCOL_VERSION, protocol_at

_SIM_PERSON_RE = re.compile(r"^S[0-9]{3}$")
_SHIFT_NAMES = ("yaw_deg", "pitch_deg", "roll_deg", "tx_mm", "ty_mm", "tz_mm")


def session_seed(base_seed, k) -> int:
    if not _is_int(base_seed) or not 0 <= base_seed < 2 ** 53:
        raise SimError("seed", f"base_seed must be an int in [0, 2**53), got {base_seed!r}")
    cell_of_session(k)  # checks k
    return (base_seed * 1_000_003 + k) % 2 ** 53


def sim_recorded_at(k) -> str:
    cell_of_session(k)  # checks k
    return (datetime(2026, 1, 1, 9, 0, 0) + timedelta(days=k - 1)).isoformat(timespec="seconds")


def sitting_shift(seed) -> np.ndarray:
    """(6,) float64: yaw, pitch, roll (deg), tx, ty, tz (mm). Child 7 of spawn(8)."""
    if not _is_int(seed) or not 0 <= seed < 2 ** 53:
        raise SimError("seed", f"seed must be an int in [0, 2**53), got {seed!r}")
    a = np.random.default_rng(np.random.SeedSequence(seed).spawn(8)[7]).standard_normal(6)
    out = np.empty(6, dtype=np.float64)
    out[:3] = a[:3] * SITTING_ANGLE_SD_DEG
    out[3:] = a[3:] * SITTING_POS_SD_MM
    return out


@dataclass(frozen=True, eq=False)
class SimSession:
    arrays: dict
    channels: dict
    meta: _store.SessionMeta


def _check_person_code(person_code) -> str:
    if not isinstance(person_code, str) or not _SIM_PERSON_RE.match(person_code) or person_code == "S000":
        raise SimError("person", f"person_code must look like S001 (not S000), got {person_code!r}")
    return person_code


def _check_duration(duration_s) -> float:
    if not _is_real(duration_s) or not 0.0 < float(duration_s) <= PROTOCOL_DURATION_S:
        raise SimError("duration", f"duration_s must be a real in (0, {PROTOCOL_DURATION_S}], got {duration_s!r}")
    return float(duration_s)


def _fingerprint(p, person, camera, screen, seed, base_seed, cell, duration_s, n, override, shift) -> dict:
    fp = {}
    for obj in (p.obs, p.events, person, camera, screen):
        for f in fields(obj):
            fp["sim_" + f.name] = float(getattr(obj, f.name))
    fp.update({
        "sim_seed": float(seed), "sim_base_seed": float(base_seed),
        "sim_preset_version": float(PRESET_VERSION), "sim_protocol_version": float(PROTOCOL_VERSION),
        "sim_preset_hash52": float(int(PRESETS_SHA256[:13], 16)),
        "sim_cell": float(cell), "sim_duration_s": float(duration_s), "sim_n_nominal": float(n),
        "sim_preset_override": 1.0 if override else 0.0,
    })
    for name, v in zip(_SHIFT_NAMES, shift):
        fp["sim_shift_" + name] = float(v)
    return fp


def simulate_session(session_index, base_seed, *, person_code="S001", duration_s=PROTOCOL_DURATION_S,
                     person=SimPerson(), camera=SimCamera(), screen=SimScreen(),
                     preset_override=None) -> SimSession:
    _check_person_code(person_code)
    duration_s = _check_duration(duration_s)
    if preset_override is not None and not isinstance(preset_override, CellPreset):
        raise SimError("preset", f"preset_override must be a CellPreset, got {type(preset_override).__name__}")
    k = session_index
    # 1. cell, preset, seed
    light, distance = cell_of_session(k)
    p = preset_override or preset(light, distance)
    seed = session_seed(base_seed, k)
    # 2. timeline and events
    rt, rl, rp = make_rngs(seed)
    t_s, idx, n = timeline(duration_s, p.obs, rt)
    sched = event_schedule(n, p.obs.fps, p.events, make_event_rngs(seed))
    # 3. protocol and sitting shift
    proto = protocol_at(np.arange(n) / p.obs.fps, screen)
    shift = sitting_shift(seed)
    shifted = drifted_pose(base_pose(p.distance_mm), shift)
    t0 = (shifted.tx_mm, shifted.ty_mm, shifted.tz_mm)
    # 4. one noise-free frame per kept frame (S043b: private fast path; stack() validates at the end)
    check_person(person)
    check_camera(camera)
    frames, poses = [], []
    for j in range(len(t_s)):
        kk = int(idx[j])
        row = proto.pose_offset[kk] + (sched.drift[kk] if sched.drift is not None else 0.0)
        pose = drifted_pose(shifted, row)
        gaze = screen_px_to_cam_mm(*proto.gaze_px[kk], screen)
        frames.append(_simulate_frame_fast(float(t_s[j]), person, camera, pose, gaze))
        poses.append((pose.yaw_deg, pose.pitch_deg, pose.roll_deg, pose.tx_mm, pose.ty_mm, pose.tz_mm))
    # 5. events and noise
    frames = apply_events_before_noise(frames, idx, sched, _build=_record_unchecked)
    frames = add_noise(frames, idx, n, p.obs, rl, rp, _build=_record_unchecked)
    frames = apply_events_after_noise(frames, idx, sched, p.events, p.obs.noise_px, t0, _build=_record_unchecked)
    arrays = stack(frames)  # validates every frame of the session (E4)
    # 6. truth channels (kept frames only)
    channels = {
        "nominal_index": idx.astype(np.int32),
        "gaze_px": np.asarray(proto.gaze_px[idx], dtype=np.float64),
        "block": proto.block[idx].astype(np.int32),
        "dot": proto.dot[idx].astype(np.int32),
        "blink": sched.blink[idx].astype(bool),
        "closure": sched.closure[idx].astype(np.float64),
        "glare": sched.glare[idx].astype(bool),
        "true_pose": np.asarray(poses, dtype=np.float64).reshape(len(t_s), 6),
    }
    # 7-8. fingerprint and meta
    cell = CELL_ORDER.index((light, distance))
    fp = _fingerprint(p, person, camera, screen, seed, base_seed, cell, duration_s, n,
                      preset_override is not None, shift)
    meta = _store.SessionMeta(person=person_code, session_index=k, source="sim", light=light,
                              distance=distance, recorded_at=sim_recorded_at(k), fingerprint=fp)
    return SimSession(arrays=arrays, channels=channels, meta=meta)


def write_sim_person(data_root, *, n_sessions=45, base_seed=20261116, person_code="S001",
                     duration_s=PROTOCOL_DURATION_S, root=ROOT) -> list:
    """Simulate and write sessions 1..n_sessions. Refuses if the person already has sessions."""
    _check_person_code(person_code)
    _check_duration(duration_s)
    if not _is_int(n_sessions) or not 1 <= n_sessions <= _store.MAX_SESSION_INDEX:
        raise SimError("index", f"n_sessions must be an int in 1..{_store.MAX_SESSION_INDEX}, got {n_sessions!r}")
    session_seed(base_seed, 1)  # checks base_seed
    if _store.list_sessions(person_code, "sim", data_root, root):
        raise SimError("exists", f"{person_code} already has sim sessions in {data_root}")
    paths = []
    for k in range(1, n_sessions + 1):
        s = simulate_session(k, base_seed, person_code=person_code, duration_s=duration_s)
        paths.append(_store.write_session(s.arrays, s.meta, s.channels, data_root=data_root, root=root))
    return paths
