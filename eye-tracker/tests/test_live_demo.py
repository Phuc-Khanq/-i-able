"""S059: live demo loop, scorers, model store and the demo / train tools (fake input only; no camera,
no window, no OS cursor). Hand numbers where possible; the simulator gives realistic head motion."""
import json
import math

import numpy as np
import pytest

from iable_eye import session_store as store
from iable_eye.app_state import AppState, GatedOutput
from iable_eye.candidates import fit_candidate, predict_candidate, recenter_candidate
from iable_eye.click import ClickEvent
from iable_eye.fusion import FusionPointer
from iable_eye.head_pointer import REST_GATE_V2, HeadPointer
from iable_eye.live_demo import DEMO_POINTER, ClickTest, DemoConfig, DemoError, DemoLoop, ReachTest, reach_targets
from iable_eye.model_store import ModelStoreError, load_candidate, model_path, save_candidate
from iable_eye.output import NullOutput
from iable_eye.paths import DataPathError
from iable_eye.quality import BLINK, OK, POSE, REAL_GATE_V1, SMALL
from iable_eye.settings import Settings
from iable_eye.sim_session import simulate_session
from iable_eye.stream_gate import StreamGate
from tools import demo as demo_tool
from tools import train_demo_model
from tests.sim_store_fixture import write_sim_sessions

W, H = 1920, 1080
HEAD_CODES = {OK, BLINK, SMALL, POSE}          # written out by hand (DECISIONS 2026-11-22)


@pytest.fixture(scope="module")
def obs_list():
    a = simulate_session(1, 20261116, duration_s=20.0).arrays
    return list(demo_tool.replay_frames(a))


@pytest.fixture(scope="module")
def model(obs_list):
    g = StreamGate(REAL_GATE_V1)
    rows = [g.step(o) for o in obs_list]
    X = np.array([f for c, f in rows if c == OK])
    rng = np.random.default_rng(5)
    Y = np.column_stack([960 + 400 * X[:, 0], 540 + 300 * X[:, 1]]) + rng.normal(0, 5, (len(X), 2))
    return fit_candidate("kernel_ridge", X, Y, np.arange(len(X)) % 3, np.arange(len(X)) // 20)


def ready_state():
    s = AppState(Settings())
    s.give_consent()
    s.set_cursor_enabled(True)
    s.start_tracking()
    s.start_cursor()
    return s


# ---------------------------------------------------------------- the loop
def test_l1_head_mode_equals_hand_chain(obs_list):
    gate, hp = StreamGate(REAL_GATE_V1), HeadPointer(W, H, DEMO_POINTER)
    loop = DemoLoop(W, H, DemoConfig())
    n_blink_moves = 0
    for o in obs_list:
        code, f = gate.step(o)
        ok = code in HEAD_CODES
        p = hp.step(o.t, float(f[8]) if ok else 0.0, float(f[9]) if ok else 0.0, ok)
        s = loop.step(o)
        assert (s.x_px, s.y_px, s.status, s.code, s.head_ok) == (p.x_px, p.y_px, p.status, code, ok)
        n_blink_moves += int(code == BLINK and s.head_ok)
    assert n_blink_moves > 0            # the head keeps working on blink frames (sim has blinks)


def test_l2_fusion_mode_equals_hand_chain(obs_list, model):
    gate = StreamGate(REAL_GATE_V1)
    fp = FusionPointer(W, H, pointer_config=DEMO_POINTER)
    loop = DemoLoop(W, H, DemoConfig(mode="fusion", gaze_tau_s=0.0), model=model)
    for o in obs_list:
        code, f = gate.step(o)
        ok = code in HEAD_CODES
        gaze = tuple(predict_candidate(model, f.reshape(1, -1))[0]) if code == OK else None
        e = fp.step(o.t, float(f[8]) if ok else 0.0, float(f[9]) if ok else 0.0, ok, gaze)
        s = loop.step(o)
        assert (s.x_px, s.y_px, s.jumped) == (e.x_px, e.y_px, e.jumped)
        assert (code == OK) == math.isfinite(s.gaze_x)


def test_l3_pause_sends_nothing_and_resume_does_not_jump(obs_list):
    state, inner = ready_state(), NullOutput(W, H)
    loop = DemoLoop(W, H, DemoConfig(), output=GatedOutput(state, inner), state=state)
    for o in obs_list[:150]:
        loop.step(o)
    sent, xy = len(inner.commands), loop.xy
    state.toggle_pause()
    for o in obs_list[150:300]:
        s = loop.step(o, key_down=True)
        assert s.paused and s.click is None and (s.x_px, s.y_px) == xy and s.status == "paused"
    assert len(inner.commands) == sent
    state.toggle_pause()
    s = loop.step(obs_list[300])
    assert not s.paused and s.status == "anchor" and (s.x_px, s.y_px) == xy   # fresh pointer at the last point


def test_l3b_short_pause_still_starts_fresh(obs_list):
    """A pause shorter than max_gap_s (0.5 s): only the resume reset re-anchors the pointer."""
    state = ready_state()
    loop = DemoLoop(W, H, DemoConfig(), state=state)
    for o in obs_list[:150]:
        loop.step(o)
    state.toggle_pause()
    for o in obs_list[150:153]:          # 3 frames = 0.1 s
        loop.step(o)
    state.toggle_pause()
    assert loop.step(obs_list[153]).status == "anchor"


def test_l4_key_click_rising_edge_and_wins(obs_list):
    loop = DemoLoop(W, H, DemoConfig(click="dwell"))
    clicks = [loop.step(o, key_down=(i in (10, 11))).click for i, o in enumerate(obs_list[:40])]
    got = [(i, c.source) for i, c in enumerate(clicks) if c is not None]
    assert got == [(10, "key")]                   # held on frame 11: no second click


def test_l5_config_refusals(model):
    with pytest.raises(DemoError, match="mode"):
        DemoConfig(mode="eyes")
    with pytest.raises(DemoError, match="click"):
        DemoConfig(click="blink")
    with pytest.raises(DemoError, match="rest_window_s"):
        DemoConfig(pointer=REST_GATE_V2.__class__())
    with pytest.raises(DemoError, match="model"):
        DemoLoop(W, H, DemoConfig(mode="fusion"))


# ---------------------------------------------------------------- scorers (hand numbers)
def test_s1_reach_hand():
    r = ReachTest(W, H, repeats=1)
    c = ((W - 1) / 2.0, (H - 1) / 2.0)
    tgt = reach_targets(W, H)
    assert tgt[0] == (40.0, 40.0) and tgt[4] == (1879.0, 1039.0)
    r.step(0.0, *c)                      # home -> target 0 shown at t0 = 0
    r.step(0.5, 500.0, 500.0)            # not near
    r.step(1.0, 45.0, 50.0)              # within 60 px of (40, 40): 1.0 s
    r.step(1.1, *c)                      # home -> target 1 at t0 = 1.1
    r.step(9.0, *c)                      # 7.9 s: still running
    r.step(9.2, *c)                      # 8.1 s > 8: missed
    assert r.times == [1.0, None]
    res = r.result()
    assert (res["trials"], res["reached"], res["reach_frac"], res["median_s"]) == (2, 1, 0.5, 1.0)


def test_s2_click_hand():
    t = ClickTest(W, H)
    x0, y0 = t.current()
    assert (x0, y0) == (W * 0.375, H * 0.5)
    t.step(0.0, None)
    t.step(0.5, ClickEvent(0.5, 10.0, 10.0, "dwell"))           # wrong click
    t.step(2.0, ClickEvent(2.0, x0 + 30.0, y0, "dwell"))        # hit (30 < 40): 2.0 s, 1 wrong
    t.step(17.5, None)                                          # 15.5 s > 15: miss
    assert t.rows == [(0, 2.0, 1), (1, None, 0)]
    res = t.result()
    assert (res["hits"], res["misses"], res["wrong_clicks"], res["median_s"]) == (1, 1, 1, 2.0)


def test_s3_look_shift_equals_recenter(obs_list, model):
    g = StreamGate(REAL_GATE_V1)
    X = np.array([f for c, f in (g.step(o) for o in obs_list[:120]) if c == OK])
    pred = predict_candidate(model, X)
    a = demo_tool.look_shift(model, pred, (959.5, 539.5))
    b = recenter_candidate(model, X, (959.5, 539.5))
    assert np.allclose(a, b, atol=1e-9)
    assert demo_tool.look_shift(model, pred[:4], (0, 0)) is None


def test_s4_edge():
    e = demo_tool.Edge()
    assert [e(v) for v in (True, True, False, True, True, False, True)] == [False, False, False, True, False,
                                                                           False, True]


# ---------------------------------------------------------------- model store
@pytest.mark.parametrize("kind", ["ridge", "kernel_ridge"])
def test_m1_round_trip_exact(tmp_path, model, kind, obs_list):
    g = StreamGate(REAL_GATE_V1)
    X = np.array([f for c, f in (g.step(o) for o in obs_list[:200]) if c == OK])
    Y = predict_candidate(model, X)
    c = model if kind == "kernel_ridge" else fit_candidate("ridge", X, Y, np.arange(len(X)) % 2,
                                                           np.zeros(len(X), int))
    p = save_candidate(c, model_path("S001", kind, data_root=tmp_path), {"sessions": [1, 2]})
    c2, info = load_candidate(p)
    assert info == {"sessions": [1, 2]} and c2.kind == kind
    assert np.array_equal(predict_candidate(c, X), predict_candidate(c2, X))
    with np.load(p, allow_pickle=False) as z:     # no object arrays inside
        assert all(z[k].dtype != object for k in z.files)
    assert sorted(x.name for x in p.parent.iterdir()) == [p.name]


def test_m2_refusals(tmp_path, model):
    with pytest.raises(ModelStoreError, match="kind"):
        model_path("S001", "mlp", data_root=tmp_path)
    with pytest.raises(ModelStoreError, match="file"):
        load_candidate(tmp_path / "none.npz")
    from iable_eye.paths import ROOT
    with pytest.raises(DataPathError):                 # never under the project folder (rule 2)
        model_path("S001", "ridge", data_root=ROOT)


# ---------------------------------------------------------------- tools (fake data, temp root)
@pytest.fixture(scope="module")
def sim6(tmp_path_factory):
    root = tmp_path_factory.mktemp("s059")
    write_sim_sessions(6, root, duration_s=40.0)
    return root


def test_t1_train_skips_exams(sim6):
    c, info = train_demo_model.train("S001", "sim", sim6, grid=demo_tool_grid())
    assert info["sessions"] == [1, 2, 3, 4, 6] and info["exams_skipped"] == [5]
    assert info["gate_blink_score"] == 0.2 and c.kind == "kernel_ridge"


def demo_tool_grid():
    from tools.run_real_scores import sim_grid
    return sim_grid()


def test_t2_demo_fake_writes_one_json(sim6, capsys):
    before = {p for p in sim6.rglob("*")}
    rc = demo_tool.main(["--person", "S001", "--source", "sim", "--fake", "--data-root", str(sim6),
                         "--test", "reach", "--max-s", "10"])
    assert rc == 0
    new = sorted({p for p in sim6.rglob("*")} - before)
    files = [p for p in new if p.is_file()]
    assert len(files) == 1 and files[0].parent.name == "results" and files[0].suffix == ".json"
    obj = json.loads(files[0].read_text(encoding="utf-8"))
    assert obj["fake"] is True and obj["mode"] == "head" and obj["result"]["test"] == "reach"
    assert obj["stats"]["fps"] == 30.0 and obj["stop"] == "max_s"
    assert "(simulated input)" in capsys.readouterr().out


def test_t3_demo_refusals(sim6):
    base = ["--person", "S001", "--source", "sim", "--data-root", str(sim6), "--no-write"]
    assert demo_tool.main(base + ["--os-cursor", "--test", "reach"]) == 2
    assert demo_tool.main(base + ["--os-cursor", "--fake"]) == 2
    assert demo_tool.main(base + ["--fake", "--mode", "fusion"]) == 2          # no model trained yet
    assert demo_tool.main(base + ["--fake", "--fake-session", "99"]) == 2


def test_t4_run_loop_quit_and_pause(obs_list):
    state = ready_state()
    loop = DemoLoop(W, H, DemoConfig(), output=GatedOutput(state, NullOutput(W, H)), state=state)
    ui = demo_tool.NullUI({5: {"pause"}, 10: {"pause"}, 20: {"quit"}})
    stats, why = demo_tool.run_loop(loop, iter(obs_list), ui, state)
    assert why == "user" and stats.n == 19
    assert stats.paused_s == pytest.approx(5 / 30.0)            # frames 5-9 paused, dt 1/30 s each


def test_l6_live_signs_and_recenter(obs_list):
    """Live test 2026-11-22: head up -> cursor up, head right -> cursor right needs signs (-1, +1)."""
    assert (DEMO_POINTER.sign_x, DEMO_POINTER.sign_y) == (-1.0, 1.0)
    assert DEMO_POINTER.rest_window_s == REST_GATE_V2.rest_window_s
    c = ((W - 1) / 2.0, (H - 1) / 2.0)
    loop = DemoLoop(W, H, DemoConfig())
    for o in obs_list[:300]:
        loop.step(o)
    assert loop.xy != c                                  # the sim head moved the cursor away
    loop.recenter()
    assert loop.xy == c and loop.step(obs_list[300]).status == "anchor"
    state = ready_state()                                # wiring: the "recenter" event of run_loop
    loop2 = DemoLoop(W, H, DemoConfig(), state=state)
    demo_tool.run_loop(loop2, iter(obs_list[:300]), demo_tool.NullUI({300: {"recenter"}}), state)
    assert loop2.xy == c


def test_l7_gaze_ema_hand_and_gap(obs_list, model):
    """EMA: g_k = g_{k-1} + (1 - exp(-dt/tau)) (raw - g_{k-1}); restarts after a gap > 0.5 s."""
    loop = DemoLoop(W, H, DemoConfig(mode="fusion", gaze_tau_s=0.3), model=model)
    a = 1.0 - math.exp(-(1 / 30.0) / 0.3)
    assert loop._smooth_gaze(0.0, 100.0, 200.0) == (100.0, 200.0)           # first value passes
    x, y = loop._smooth_gaze(1 / 30.0, 400.0, 200.0)
    assert x == pytest.approx(100.0 + a * 300.0) and y == pytest.approx(200.0)
    assert loop._smooth_gaze(1 / 30.0 + 0.6, 900.0, 50.0) == (900.0, 50.0)   # gap 0.6 s: restart
    off = DemoLoop(W, H, DemoConfig(mode="fusion", gaze_tau_s=0.0), model=model)
    assert off._smooth_gaze(0.0, 1.0, 2.0) == (1.0, 2.0) and off._smooth_gaze(0.03, 9.0, 9.0) == (9.0, 9.0)
    with pytest.raises(DemoError, match="gaze_tau_s"):
        DemoConfig(gaze_tau_s=-1.0)


def test_l8_gain_speed_smoothing_is_opt_in_and_steadier():
    """Default PointerConfig keeps the old gain; with tau the gain stops flickering on a noisy speed."""
    from dataclasses import replace
    assert REST_GATE_V2.gain_speed_tau_s == 0.0 and DEMO_POINTER.gain_speed_tau_s == 0.3
    rng = np.random.default_rng(3)
    t = np.arange(300) / 30.0
    yaw = 20.0 * t + rng.normal(0, 0.15, t.size)            # steady 20 deg/s turn + angle noise
    def gains(cfg):
        hp = HeadPointer(W, H, cfg)
        return np.array([hp.step(float(ti), float(y), 0.0, True).gain_px_per_deg for ti, y in zip(t, yaw)])[30:]
    g0 = gains(replace(DEMO_POINTER, gain_speed_tau_s=0.0))
    g1 = gains(DEMO_POINTER)
    assert g1.std() < 0.5 * g0.std()
