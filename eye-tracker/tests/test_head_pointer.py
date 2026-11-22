"""S020 hand tests H1-H10 and F1 for head_pointer.py. Expected numbers are from the prompt."""
import math

import numpy as np
import pytest

from iable_eye.head_pointer import HeadPointer, PointerConfig, PointerError, run_series, smoothstep_gain

NF = PointerConfig(filter=None)
G30 = 25.42103  # gain at 30 deg/s, u = 22/52 (hand value)
TOL = 1e-4


def ptr(start=None, cfg=NF):
    return HeadPointer(1920, 1080, cfg, start)


@pytest.mark.parametrize("s,g", [(0, 10), (8, 10), (21, 16.25), (34, 30), (60, 50), (100, 50)])
def test_h1_gain(s, g):
    assert abs(smoothstep_gain(s, NF) - g) < 1e-9


def test_h2_30fps():
    p = ptr()
    assert p.step(0.0, 0.0, 0.0, True).status == "anchor"
    r = p.step(1 / 30, 1.0, 0.0, True)
    print(f"H2 s={r.speed_deg_s:.5f} g={r.gain_px_per_deg:.5f} dx={r.dx_px:.5f} dy={r.dy_px}")
    assert r.status == "move" and abs(r.speed_deg_s - 30) < 1e-9
    assert abs(r.gain_px_per_deg - G30) < TOL and abs(r.dx_px - G30) < TOL and r.dy_px == 0


def _ramp_dx(fps):
    p = ptr((0, 540))
    x0 = p.x
    for k in range(fps + 1):
        t = k / fps
        p.step(t, 30.0 * t, 0.0, True)
    return p.x - x0


def test_h3_frame_rate():
    totals = {f: _ramp_dx(f) for f in (8, 15, 30)}
    print("H3 total dx", {f: round(v, 6) for f, v in totals.items()})
    for f, v in totals.items():
        assert abs(v - 762.6309) < TOL, f"fps {f}: {v}"
    assert max(totals.values()) - min(totals.values()) < 1e-6


def test_h4a_rest():
    p = ptr()
    p.step(0.0, 0.0, 0.0, True)
    for k in range(1, 10):
        r = p.step(k / 30, 0.02 * k, 0.0, True)
        assert r.status == "rest" and r.dx_px == 0
    assert p.x == 960


def test_h4b_rest_per_axis():
    p = ptr()
    p.step(0.0, 0.0, 0.0, True)
    r = p.step(1 / 30, 1.0, 0.02, True)
    print(f"H4b dx={r.dx_px:.5f} dy={r.dy_px}")
    assert abs(r.dx_px - G30) < TOL and r.dy_px == 0


def test_h5_cap():
    p = ptr()
    p.step(0.0, 0.0, 0.0, True)
    r = p.step(1 / 30, 5.0, 0.0, True)
    print(f"H5 g={r.gain_px_per_deg} dx={r.dx_px:.4f}")
    assert r.gain_px_per_deg == 50 and abs(r.dx_px - 133.3333) < TOL


def test_h6_glitch():
    p = ptr()
    rs = [p.step(k / 30, y, 0.0, True) for k, y in enumerate([0, 12, 0, 1])]
    print("H6", [r.status for r in rs], f"dx={rs[-1].dx_px:.5f}")
    assert [r.status for r in rs] == ["anchor", "glitch", "glitch", "move"]
    assert abs(rs[-1].dx_px - G30) < TOL and abs(p.x - (960 + G30)) < TOL


def test_h7_edge():
    p = ptr((1900, 540))
    xs = [p.step(k / 30, y, 0.0, True).x_px for k, y in enumerate([0, 1, 0])]
    print("H7 x", xs)
    assert xs[1] == 1919 and abs(xs[2] - 1893.57897) < TOL


def test_h8_blink():
    p = ptr()
    rs = [p.step(i / 30, y, 0.0, bool(g)) for i, (y, g) in enumerate(zip([0, 0, 3, 5, 5], [1, 1, 0, 0, 1]))]
    print("H8", [r.status for r in rs])
    assert [r.status for r in rs] == ["anchor", "rest", "hold", "hold", "anchor"]
    assert all(r.x_px == 960 for r in rs)
    q = ptr()
    q.step(0.0, 0.0, 0.0, True)
    st = {k: q.step(k / 30, 0.0, 0.0, False).status for k in range(1, 13)}
    assert all(st[k] == "hold" for k in range(1, 9)) and all(st[k] == "lost" for k in range(10, 13))


def test_h9_gap():
    p = ptr()
    p.step(0.0, 0.0, 0.0, True)
    r = p.step(0.6, 1.0, 0.0, True)
    assert r.status == "anchor" and p.x == 960


def test_h10_refusals():
    p = ptr()
    p.step(0.0, 0.0, 0.0, True)
    with pytest.raises(PointerError) as e:
        p.step(0.0, 0.0, 0.0, True)
    assert e.value.code == "time"
    with pytest.raises(PointerError) as e:
        p.step(0.1, math.nan, 0.0, True)
    assert e.value.code == "value"
    assert p.step(0.2, math.nan, 0.0, False).status == "hold"
    bad = [dict(gain_slow_px_per_deg=60.0), dict(speed_fast_deg_s=8.0), dict(rest_speed_deg_s=0.0),
           dict(max_speed_px_s=-1.0), dict(blink_hold_s=0.0), dict(max_gap_s=math.inf),
           dict(sign_x=0.5), dict(sign_y=2.0)]
    for kw in bad:
        with pytest.raises(PointerError) as e:
            PointerConfig(**kw)
        assert e.value.code == "config", kw
    PointerConfig(sign_x=-1.0, sign_y=1.0)


def test_f1_filter_noise():
    """(simulated) white noise SD 0.1 deg, 20 s at 30 fps."""
    rng = np.random.default_rng(20261118)
    n = 600
    t = np.arange(n) / 30
    yaw, pitch = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)
    ok = np.ones(n, bool)
    out = {}
    for name, cfg in (("filter", PointerConfig()), ("none", NF)):
        r = run_series(t, yaw, pitch, ok, 1920, 1080, cfg)
        moving = int(np.sum(r["status"] == "move"))
        rms = float(np.sqrt(np.mean((r["x"] - 960) ** 2 + (r["y"] - 540) ** 2)))
        out[name] = (moving, rms)
    print(f"F1 (simulated) filter: moving {out['filter'][0]} rms {out['filter'][1]:.2f} px; "
          f"none: moving {out['none'][0]} rms {out['none'][1]:.2f} px")
    assert out["filter"][0] < out["none"][0]
