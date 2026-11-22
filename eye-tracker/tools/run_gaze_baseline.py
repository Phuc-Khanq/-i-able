"""S022: the gaze baseline (ridge map + one shift per sitting) on simulated sessions,
scored by the frozen S017 runner. Prints only; writes nothing. Every number is (simulated).

Only this tool reads the simulator (sessions, the `gaze_px` label, the protocol timing).
The model (`gaze_baseline`) and the dot logic (`dots`) never see a sim module.

Usage: python eye-tracker/tools/run_gaze_baseline.py [--n 45] [--duration S]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import evaluate, metrics  # noqa: E402
from iable_eye.dots import dot_prediction, dot_windows  # noqa: E402
from iable_eye.features import session_features  # noqa: E402
from iable_eye.gaze_baseline import fit_ridge_shift, predict  # noqa: E402
from iable_eye.quality import OK, gate_session  # noqa: E402
from iable_eye.sim_core import SimScreen  # noqa: E402
from iable_eye.sim_protocol import BLOCKS, DOT_S, N_DOTS, PROTOCOL_DURATION_S, grid_dot_px  # noqa: E402
from iable_eye.sim_session import simulate_session  # noqa: E402
from iable_eye.splits import assert_no_leak, forward_splits, info_from_meta  # noqa: E402

PERSON = "S001"
BASE_SEED = 20261116
ALPHA = 1.0          # fixed before any exam number (E2; DECISIONS 2026-11-18)
MIN_TRAIN = 2
_GRID = [b for b in BLOCKS if b[0] == "grid"][0]
ONSETS = [_GRID[1] + DOT_S * d for d in range(N_DOTS)]  # o_d = 5.0 + 1.5 d


def load_session(k, duration_s, window_ms):
    """Simulate session k in memory; keep only what the baseline needs (no landmarks)."""
    s = simulate_session(k, BASE_SEED, person_code=PERSON, duration_s=duration_s)
    feats = session_features(s.arrays)
    ok = gate_session(s.arrays, features=feats) == OK  # S011b DONE: ok = gate code 0
    if np.isnan(feats[ok]).any():
        raise RuntimeError(f"session {k}: an ok frame has a NaN feature")
    fps = float(s.meta.fingerprint["sim_fps"])
    tau = s.channels["nominal_index"] / fps
    dot = s.channels["dot"]
    win = dot_windows(tau, dot, ONSETS, window_ms)
    return {
        "k": k, "info": info_from_meta(s.meta), "cell": f"{s.meta.light}/{s.meta.distance}",
        "feats": feats, "ok": ok, "gaze_px": s.channels["gaze_px"], "windows": win,
        "n_expected": int(round(0.8 * fps)), "n_frames": int(ok.size),
        "tau": tau,  # S024: needed to pick the look window
    }


def train_rows(sess):
    """ok frames inside the settled windows; label gaze_px."""
    ix = np.concatenate([w for _, w in sess["windows"]]) if sess["windows"] else np.array([], int)
    ix = ix[sess["ok"][ix]]
    return sess["feats"][ix], sess["gaze_px"][ix]


def score_session(model, sess, frac, screen, shift=None):
    """One (dot, error_px or None) per grid dot. shift=None: default_shift (no re-cal, E4).
    S024: a re-cal shift may be passed."""
    pred = np.full((sess["n_frames"], 2), np.nan)
    ok = sess["ok"]
    if ok.any():
        pred[ok] = predict(model, sess["feats"][ok], shift)
    out = []
    for d, ix in sess["windows"]:
        p = dot_prediction(pred, ok, ix, sess["n_expected"], frac)
        if p is None:
            out.append((d, None))
            continue
        true = np.array([grid_dot_px(d, screen)])
        out.append((d, float(metrics.dot_errors_px(np.array([p]), true)[0])))
    return out


def run(n=45, duration_s=PROTOCOL_DURATION_S, log=print):
    gates = evaluate.load_gates()
    window_ms = tuple(gates["settled_window_ms"])
    frac = float(gates["lost_if_ok_frac_below"])
    screen = SimScreen()
    sessions = {}
    for k in range(1, n + 1):
        t0 = time.perf_counter()
        sessions[k] = load_session(k, duration_s, window_ms)
        log(f"  session {k:2d} {sessions[k]['cell']:15s} frames {sessions[k]['n_frames']:5d} "
            f"ok {sessions[k]['ok'].mean():.3f}  ({time.perf_counter() - t0:.1f} s)")
    splits = forward_splits([s["info"] for s in sessions.values()], min_train=MIN_TRAIN)
    assert_no_leak(splits)
    exam_rows = []
    nonexam = {c: {"e": [], "lost": 0} for c in metrics.CELLS}
    exam_trains = {}
    for sp in splits:
        Xs, Ys, Gs = [], [], []
        for j in sp.train_indexes:
            X, Y = train_rows(sessions[j])
            Xs.append(X)
            Ys.append(Y)
            Gs.append(np.full(len(X), j, dtype=np.int64))
        model = fit_ridge_shift(np.vstack(Xs), np.vstack(Ys), np.concatenate(Gs), alpha=ALPHA)
        test = sessions[sp.test_index]
        res = score_session(model, test, frac, screen)
        if sp.test_is_exam:
            exam_trains[sp.test_index] = sp.train_indexes
            for _, e in res:
                exam_rows.append(evaluate.DotRow(PERSON, sp.test_index, "sim", test["cell"], True, e, "replay"))
        else:
            for _, e in res:
                if e is None:
                    nonexam[test["cell"]]["lost"] += 1
                else:
                    nonexam[test["cell"]]["e"].append(e)
    return {"gates": gates, "splits": splits, "exam_trains": exam_trains,
            "exam_rows": exam_rows, "nonexam": nonexam, "sessions": sessions}


def _table(title, scores):
    print(title)
    print(f"  {'cell':15s} {'n':>4s} {'lost':>4s} {'median px':>10s} {'loss':>6s}")
    for s in scores:
        med = "nan" if s.median_px != s.median_px else f"{s.median_px:.1f}"
        print(f"  {s.cell:15s} {s.n:4d} {s.lost:4d} {med:>10s} {s.loss:6.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--duration", type=float, default=PROTOCOL_DURATION_S)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    r = run(a.n, a.duration)
    print(f"alpha {ALPHA}  min_train {MIN_TRAIN}  splits {len(r['splits'])}  (simulated)")
    for k, tr in r["exam_trains"].items():
        bad = [j for j in tr if j % 5 == 0]
        print(f"  exam {k:2d} train {list(tr)}  exams in train: {bad}")
    report, interval, sha = evaluate.evaluate_v2(r["exam_rows"])
    print(f"gates sha {sha[:12]}")
    _table("Exam cells (simulated):", report.cells)
    if report.status == "MEASURED":
        print(f"{report.label}: {report.v2:.3f}  worst {report.worst_cell}  best {report.best_cell}  "
              f"interval [{interval[0]:.3f}, {interval[1]:.3f}]")
    else:
        print(f"{report.label}: UNMEASURED ({report.n_measured}/9 cells measured; "
              f"v2_partial {report.v2_partial}); worst {report.worst_cell}; interval None")
    print(f"overall exam median {report.overall_median_px:.1f} px  reasons {report.reasons}  "
          f"gate_ok {report.gate_ok}  product_pass {report.product_pass}")
    ne = [metrics.cell_score(c, np.asarray(v["e"]), v["lost"]) for c, v in r["nonexam"].items()
          if v["e"] or v["lost"]]
    _table("Non-exam forward, simulated (the table later steps may tune on):", ne)
    bn = [s for s in report.cells if s.cell == "bright/normal"][0]
    v = f"{report.v2:.3f}" if report.status == "MEASURED" else "UNMEASURED"
    print(f"runtime {time.perf_counter() - t0:.1f} s")
    print(f"S022 part 4: bright/normal exam median {bn.median_px:.1f} px, V2 (simulated) {v}")


if __name__ == "__main__":
    main()
