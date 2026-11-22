"""S064: live gaze calibration, demo recordings (numbers only) and rule D1 pieces. Fake input only."""
import math

import numpy as np
import pytest

from iable_eye import session_store as store
from iable_eye.app_state import AppState
from iable_eye.candidates import fit_candidate
from iable_eye.click import ClickEvent
from iable_eye.demo_record import (LABEL_KINDS, DemoRecordError, DemoRecorder, label_rows, list_demos,
                                   read_demo)
from iable_eye.live_demo import DEMO_POINTER, Calibration, ClickTest, DemoConfig, DemoLoop, DemoStep, calib_targets
from iable_eye.paths import ROOT, DataPathError
from iable_eye.quality import REAL_GATE_V1
from iable_eye.settings import Settings
from iable_eye.sim_session import simulate_session
from iable_eye.stream_gate import StreamGate
from tools import check_demo_data as chk
from tools import demo as demo_tool

W, H = 1920, 1080


@pytest.fixture(scope="module")
def sim():
    a = simulate_session(2, 20261116, duration_s=20.0).arrays
    return a, list(demo_tool.replay_frames(a))


@pytest.fixture(scope="module")
def model(sim):
    _, obs = sim
    g = StreamGate(REAL_GATE_V1)
    X = np.array([f for c, f in (g.step(o) for o in obs) if c == 0])
    Y = np.column_stack([960 + 400 * X[:, 0], 540 + 300 * X[:, 1]])
    return fit_candidate("kernel_ridge", X, Y, np.arange(len(X)) % 3, np.arange(len(X)) // 20)


def test_c1_targets_and_timing():
    t = calib_targets(W, H)
    assert t[0] == (960.0, 540.0) and t[1] == (192.0, 129.6) and t[5] == (1728.0, 950.4) and len(t) == 9
    c = Calibration(W, H)
    assert c.current(10.0) == (0, t[0])                      # the first call sets t0 = 10.0
    assert c.current(11.59)[0] == 0 and c.current(11.61)[0] == 1
    assert c.current(10.0 + 9 * 1.6 + 0.01) is None
    assert c.windows()[1] == (10.0 + 1.6 + 0.5, 10.0 + 1.6 + 1.5, 192.0, 129.6)


def _squeezed(calib, k=0.5, fps=30.0):
    """Gaze that moves only k times as far as the eyes: centre + k (target - centre)."""
    t, cx, cy = 0.0, W / 2.0, H / 2.0
    while True:
        cur = calib.current(t)
        if cur is None:
            return
        x, y = cur[1]
        calib.step(t, cx + k * (x - cx), cy + k * (y - cy))
        t += 1 / fps


def test_c2_fit_undoes_a_squeeze(model):
    c = Calibration(W, H)
    _squeezed(c, 0.5)
    s, b, n = c.fit()
    assert n == 9 and np.allclose(s, [2.0, 2.0]) and np.allclose(b, [-960.0, -540.0])
    loop = DemoLoop(W, H, DemoConfig(mode="fusion"), model=model)
    assert demo_tool.apply_calibration(loop, c, (959.5, 539.5), log=lambda *x: None) == "affine"
    assert np.allclose(loop.affine[0], [2.0, 2.0])
    c2 = Calibration(W, H)                                   # only 3 dots seen -> fallback
    for i in range(3):
        c2.gaze[i] = [(500.0, 500.0)] * 10
    c2._t0 = 0.0
    assert demo_tool.apply_calibration(loop, c2, (959.5, 539.5), log=lambda *x: None) == "shift"


def test_c3_affine_changes_gaze_only(sim, model):
    _, obs = sim
    a = DemoLoop(W, H, DemoConfig(mode="fusion"), model=model)
    b = DemoLoop(W, H, DemoConfig(mode="fusion"), model=model, affine=([2.0, 1.0], [-960.0, 0.0]))
    for o in obs[:60]:
        sa, sb = a.step(o), b.step(o)
        if math.isfinite(sa.gaze_x):
            assert sb.gaze_x == pytest.approx(2.0 * sa.gaze_x - 960.0) and sb.gaze_y == pytest.approx(sa.gaze_y)


def test_c4_demo_smoothing_default():
    assert DEMO_POINTER.filter.min_cutoff_hz == 0.6 and DEMO_POINTER.filter.beta == 0.05


# ---------------------------------------------------------------- recorder
def test_r1_round_trip_and_folders(tmp_path, sim):
    a, obs = sim
    rec = DemoRecorder("S001", {"mode": "fusion"})
    for o in obs[:90]:
        rec.add(o)
    rec.add_label("calib", 0.5, 1.5, 960.0, 540.0)
    rec.add_label("click_hit", 2.0, 2.6, 100.0, 200.0)
    with pytest.raises(DemoRecordError, match="label"):
        rec.add_label("blink", 0, 1, 0, 0)
    p = rec.save("20261122_150000", data_root=tmp_path)
    assert p == tmp_path / "S001" / "demo" / "d20261122_150000.npz" and list_demos("S001", tmp_path) == [p]
    d = read_demo(p)
    assert np.array_equal(d["arrays"]["landmarks"], a["landmarks"][:90])
    assert d["labels"].tolist() == [[0, 0.5, 1.5, 960.0, 540.0], [1, 2.0, 2.6, 100.0, 200.0]]
    assert d["meta"]["mode"] == "fusion" and d["meta"]["n_frames"] == 90
    assert store.list_sessions("S001", "sim", data_root=tmp_path) == []      # never in the session store
    with pytest.raises(DemoRecordError, match="exists"):
        rec.save("20261122_150000", data_root=tmp_path)
    assert DemoRecorder("S001", {}).save("x", data_root=tmp_path) is None   # no frame, no file
    with pytest.raises(DataPathError):
        rec.save("20261122_150001", data_root=ROOT)                          # never under the project


def test_r2_label_rows(tmp_path, sim):
    a, obs = sim
    rec = DemoRecorder("S001", {})
    for o in obs[:120]:
        rec.add(o)
    rec.add_label("calib", 1.0, 2.0, 10.0, 20.0)
    rec.add_label("reach", 2.0, 3.0, 30.0, 40.0)
    d = read_demo(rec.save("20261122_150002", data_root=tmp_path))
    X, Y, D = label_rows(d)
    t = np.asarray(d["arrays"]["t"])                              # the replay re-stamps t
    assert 0 < len(X) <= int(((t >= 1.0) & (t < 2.0)).sum()) and set(D.tolist()) == {0}
    assert np.all(Y == [10.0, 20.0])
    X2, _, D2 = label_rows(d, kinds=("calib", "reach"))
    assert set(D2.tolist()) == {0, 1} and len(X2) > len(X)


class _FakeLoop:
    cfg = DemoConfig()

    def __init__(self, clicks):
        self.clicks = clicks

    def step(self, obs, key_down=False):
        c = self.clicks.get(obs.t)
        return DemoStep(obs.t, 0, True, math.nan, math.nan, 0.0, 0.0, "rest", False, c, 0.0, False)


def test_r3_click_hit_label(sim):
    _, obs = sim
    tx, ty = ClickTest(W, H).current()
    t_hit = obs[45].t
    loop = _FakeLoop({obs[10].t: ClickEvent(obs[10].t, 5.0, 5.0, "dwell"),     # wrong click: no label
                      t_hit: ClickEvent(t_hit, tx, ty, "dwell")})
    rec = DemoRecorder("S001", {})
    state = AppState(Settings())
    demo_tool.run_loop(loop, iter(obs[:60]), demo_tool.NullUI(), state, ClickTest(W, H), rec=rec)
    assert len(rec.obs) == 60
    assert rec.labels == [(LABEL_KINDS.index("click_hit"), t_hit - 0.6, t_hit, tx, ty)]


# ---------------------------------------------------------------- rule D1 pieces
def test_d1_score_demo_and_usable(tmp_path, sim, model):
    _, obs = sim
    rec = DemoRecorder("S001", {})
    for o in obs[:600]:
        rec.add(o)
    for i, (x, y) in enumerate(calib_targets(W, H)):
        rec.add_label("calib", 0.5 + 1.6 * i, 1.5 + 1.6 * i, x, y)
    d = read_demo(rec.save("20261122_150003", data_root=tmp_path))
    assert len(chk.usable(d)) == 9
    e = chk.score_demo(model, d)
    assert len(e) == 8 and all(np.isfinite(e))                  # centre dot used for the recenter only
    Xs, Ys, Gs, Ds = chk.demo_train_rows([d, d])
    assert set(Gs[0].tolist()) == {1000} and set(Gs[1].tolist()) == {1001}
    assert Ds[1].min() >= 200000                                 # dot ids never collide across recordings
