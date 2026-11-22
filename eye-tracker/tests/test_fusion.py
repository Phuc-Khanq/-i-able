"""S032 hand tests F1-F12 for the fusion pointer. Frames at t_i = i / 8 (exact in binary).

Expected values are worked out by hand from the prompt rules, never with the code under test.
"""
import ast
import math
import os

import pytest

from iable_eye.fusion import FusionConfig, FusionError, FusionPointer
from iable_eye.head_pointer import REST_GATE_V2, HeadPointer, PointerConfig, PointerError

SRC = os.path.join(os.path.dirname(__file__), "..", "iable_eye", "fusion.py")
N = 21  # i = 0..20


def run(gaze=lambda i: (960.0, 540.0), yaw=lambda i: 0.0, ok=lambda i: True, label=""):
    fp = FusionPointer(1920, 1080, pointer_config=PointerConfig(filter=None), start_xy=(960, 540))
    out = [fp.step(i / 8, yaw(i), 0.0, ok(i), gaze(i)) for i in range(N)]
    jumps = [i for i, s in enumerate(out) if s.jumped]
    if label:
        print(f"{label} jumps {jumps}")
    return out, jumps


def test_f1_all_still():
    out, jumps = run(label="F1")
    assert jumps == []
    assert all(s.x_px == 960.0 and s.y_px == 540.0 for s in out)


def test_f2_jump_after_hold():
    out, jumps = run(gaze=lambda i: (1300.0, 540.0) if i >= 4 else (960.0, 540.0), label="F2")
    assert jumps == [6]
    assert all(s.x_px == 1300.0 for s in out[6:])
    assert out[4].r == pytest.approx(340 / 110, abs=1e-12)


def test_f3_target_is_mean():
    pts = {4: (1290.0, 530.0), 5: (1300.0, 540.0), 6: (1310.0, 550.0)}
    out, jumps = run(gaze=lambda i: pts.get(i, (1300.0, 540.0) if i > 6 else (960.0, 540.0)), label="F3")
    assert jumps == [6]
    assert out[6].x_px == pytest.approx(1300.0, abs=1e-9)
    assert out[6].y_px == pytest.approx(540.0, abs=1e-9)


def test_f4a_hysteresis_keeps_run():
    gx = {4: 1136.0, 5: 1092.0, 6: 1092.0}
    out, jumps = run(gaze=lambda i: (gx.get(i, 960.0), 540.0), label="F4A")
    assert jumps == [6]
    assert out[6].x_px == pytest.approx(3320 / 3, abs=1e-9)


def test_f4b_break_below_r_in():
    gx = {4: 1136.0, 5: 1059.0}
    out, jumps = run(gaze=lambda i: (gx.get(i, 1136.0 if i >= 6 else 960.0), 540.0), label="F4B")
    assert jumps == [8]
    assert out[8].x_px == 1136.0


def test_f4c_between_thresholds_never_starts():
    out, jumps = run(gaze=lambda i: (1092.0 if i >= 4 else 960.0, 540.0), label="F4C")
    assert jumps == []


def test_f5_fast_head_blocks():
    yaw = lambda i: 2.0 * (i - 3) if 4 <= i <= 9 else (12.0 if i > 9 else 0.0)
    out, jumps = run(gaze=lambda i: (1800.0, 540.0) if i >= 4 else (960.0, 540.0), yaw=yaw, label="F5")
    assert jumps == [12]
    assert out[12].x_px == 1800.0
    # the busy frames really are fast moves (bare head pointer: 2 deg per 1/8 s = 16 deg/s)
    hp = HeadPointer(1920, 1080, config=PointerConfig(filter=None), start_xy=(960, 540))
    st = [hp.step(i / 8, yaw(i), 0.0, True) for i in range(10)]
    assert all(st[i].status == "move" and st[i].speed_deg_s == 16.0 for i in range(4, 10))


def test_f6_cooldown():
    g = lambda i: (1300.0, 540.0) if 4 <= i <= 6 else ((300.0, 540.0) if i >= 7 else (960.0, 540.0))
    out, jumps = run(gaze=g, label="F6")
    assert jumps == [6, 12]
    assert out[-1].x_px == 300.0


def test_f7a_not_ok_frame_breaks_run():
    out, jumps = run(gaze=lambda i: (1300.0, 540.0) if i >= 4 else (960.0, 540.0), ok=lambda i: i != 5,
                     label="F7A")
    assert jumps == [8]


def test_f7b_nan_gaze_breaks_run():
    g = lambda i: (math.nan, 540.0) if i == 5 else ((1300.0, 540.0) if i >= 4 else (960.0, 540.0))
    out, jumps = run(gaze=g, label="F7B")
    assert jumps == [8]
    assert math.isnan(out[5].r)


def test_f8_slow_head_allowed():
    out, jumps = run(gaze=lambda i: (1800.0, 540.0) if i >= 4 else (960.0, 540.0),
                     yaw=lambda i: 0.5 * (i - 3) if i >= 4 else 0.0, label="F8")
    assert jumps == [6]
    assert all(out[i].status == "move" for i in range(4, 7))


def test_f9_head_finishes_from_jump_point():
    out, jumps = run(gaze=lambda i: (1800.0, 540.0) if i >= 4 else (960.0, 540.0),
                     yaw=lambda i: 0.5 * (i - 3) if i >= 4 else 0.0, label="F9")
    assert jumps == [6]
    assert out[7].x_px == 1805.0  # 1800 + 10 px/deg * 0.5 deg
    assert out[20].x_px == 1870.0  # 1800 + 14 frames * 5 px


def test_f10_clip():
    out, jumps = run(gaze=lambda i: (2500.0, -100.0) if i >= 4 else (960.0, 540.0), label="F10")
    # The prompt says jumps [6]. By rules 1-7 (hand): after the clip the cursor is (1919, 0) and the
    # gaze stays off screen, r = sqrt((581/110)^2 + (100/140)^2) ~ 5.33 >= r_out, so a new run opens
    # when the cooldown ends (i = 10, 16) and jumps after the hold (i = 12, 18) to the same point.
    # Flagged in opus/reports/S032_report.md for the Architect.
    assert jumps[0] == 6
    assert jumps == [6, 12, 18]
    assert all((out[i].x_px, out[i].y_px) == (1919.0, 0.0) for i in range(6, N))
    assert out[10].r == pytest.approx(math.hypot(581 / 110, 100 / 140), abs=1e-12)


@pytest.mark.parametrize("kw", [dict(r_in=2.0, r_out=1.5), dict(a_px=0.0), dict(hold_s=-1.0),
                                dict(cooldown_s=math.nan)])
def test_f11_config_refusals(kw):
    with pytest.raises(FusionError) as e:
        FusionConfig(**kw)
    assert e.value.code == "config"


def test_f11_value_and_time_refusals():
    fp = FusionPointer(1920, 1080)
    for bad in [(1.0,), "ab"]:
        with pytest.raises(FusionError) as e:
            fp.step(0.0, 0.0, 0.0, True, bad)
        assert e.value.code == "value"
    fp.step(0.5, 0.0, 0.0, True, None)
    with pytest.raises(PointerError) as e:
        fp.step(0.25, 0.0, 0.0, True, None)
    assert e.value.code == "time"


def test_f12_source_and_default():
    src = open(SRC, encoding="utf-8").read()
    assert "sim_" not in src
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module)
    assert mods <= {"math", "dataclasses", "numpy", "iable_eye.head_pointer"}, mods
    assert FusionPointer(1920, 1080).head.config == REST_GATE_V2
