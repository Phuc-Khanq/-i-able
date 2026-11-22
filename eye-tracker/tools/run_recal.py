"""S024: one-look recenter vs no re-cal on simulated sessions. Prints only; writes nothing.
Every number is (simulated).

Variant A = S022 (default shift). Variant B = shift from `recal.recenter_shift` on the
gate-ok frames with 1.0 <= tau < 4.8 s (inside the neutral block), target = screen centre.
Fixed before any result (E2): that window, min_frames 5, per-axis median.

Usage: python eye-tracker/tools/run_recal.py [--n 45] [--duration S]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import evaluate, metrics  # noqa: E402
from iable_eye.gaze_baseline import fit_ridge_shift  # noqa: E402
from iable_eye.recal import RecalError, recenter_shift  # noqa: E402
from iable_eye.sim_core import SimScreen  # noqa: E402
from iable_eye.sim_protocol import PROTOCOL_DURATION_S  # noqa: E402
from iable_eye.splits import assert_no_leak, forward_splits  # noqa: E402
from tools.run_gaze_baseline import (ALPHA, MIN_TRAIN, PERSON, _table, load_session,  # noqa: E402
                                     score_session, train_rows)

LOOK_T0, LOOK_T1, MIN_FRAMES = 1.0, 4.8, 5  # E2, fixed


def look_indexes(sess):
    t = sess["tau"]
    return np.flatnonzero(sess["ok"] & (t >= LOOK_T0) & (t < LOOK_T1))


def scored_indexes(sess):
    ws = [w for _, w in sess["windows"]]
    return np.concatenate(ws) if ws else np.array([], int)


def run(n=45, duration_s=PROTOCOL_DURATION_S, log=print):
    gates = evaluate.load_gates()
    window_ms = tuple(gates["settled_window_ms"])
    frac = float(gates["lost_if_ok_frac_below"])
    screen = SimScreen()
    target = np.array([screen.width_px / 2.0, screen.height_px / 2.0])
    sessions = {}
    for k in range(1, n + 1):
        t0 = time.perf_counter()
        sessions[k] = load_session(k, duration_s, window_ms)
        log(f"  session {k:2d} {sessions[k]['cell']:15s} ({time.perf_counter() - t0:.1f} s)")
    splits = forward_splits([s["info"] for s in sessions.values()], min_train=MIN_TRAIN)
    assert_no_leak(splits)
    out = {v: {"exam_rows": [], "nonexam": {c: {"e": [], "lost": 0} for c in metrics.CELLS}}
           for v in ("A", "B")}
    exam_trains, look_n, fallbacks = {}, {}, 0
    tot_look = tot_scored = tot_inter = 0
    for sp in splits:
        Xs, Ys, Gs = [], [], []
        for j in sp.train_indexes:
            X, Y = train_rows(sessions[j])
            Xs.append(X)
            Ys.append(Y)
            Gs.append(np.full(len(X), j, dtype=np.int64))
        model = fit_ridge_shift(np.vstack(Xs), np.vstack(Ys), np.concatenate(Gs), alpha=ALPHA)
        test = sessions[sp.test_index]
        li, si = look_indexes(test), scored_indexes(test)
        inter = np.intersect1d(li, si).size
        assert inter == 0, f"session {sp.test_index}: look frames are also scored"
        tot_look, tot_scored, tot_inter = tot_look + li.size, tot_scored + si.size, tot_inter + inter
        look_n[sp.test_index] = int(li.size)
        try:
            shift = recenter_shift(model, test["feats"][li], target, min_frames=MIN_FRAMES)
        except RecalError as e:
            if e.code != "few":
                raise
            shift, fallbacks = None, fallbacks + 1
        res = {"A": score_session(model, test, frac, screen),
               "B": score_session(model, test, frac, screen, shift=shift)}
        if sp.test_is_exam:
            exam_trains[sp.test_index] = sp.train_indexes
        for v, rr in res.items():
            for _, e in rr:
                if sp.test_is_exam:
                    out[v]["exam_rows"].append(
                        evaluate.DotRow(PERSON, sp.test_index, "sim", test["cell"], True, e, "replay"))
                elif e is None:
                    out[v]["nonexam"][test["cell"]]["lost"] += 1
                else:
                    out[v]["nonexam"][test["cell"]]["e"].append(e)
    return {"gates": gates, "splits": splits, "exam_trains": exam_trains, "out": out,
            "look_n": look_n, "fallbacks": fallbacks,
            "disjoint": (tot_look, tot_scored, tot_inter)}


def _nonexam_scores(ne):
    return [metrics.cell_score(c, np.asarray(v["e"]), v["lost"]) for c, v in ne.items()
            if v["e"] or v["lost"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--duration", type=float, default=PROTOCOL_DURATION_S)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    r = run(a.n, a.duration)
    print(f"alpha {ALPHA}  min_train {MIN_TRAIN}  splits {len(r['splits'])}  (simulated)")
    for k, tr in r["exam_trains"].items():
        print(f"  exam {k:2d} train {list(tr)}  exams in train: {[j for j in tr if j % 5 == 0]}")
    ln = np.array(list(r["look_n"].values()))
    print(f"look frames per test session: min {ln.min()} median {np.median(ln):.0f} max {ln.max()}  "
          f"fallbacks {r['fallbacks']}")
    L, S, I = r["disjoint"]
    print(f"disjoint check: look total {L}, scored-window total {S}, intersection {I}")
    overall = {}
    for v in ("A", "B"):
        report, interval, sha = evaluate.evaluate_v2(r["out"][v]["exam_rows"])
        print(f"--- variant {v} ---  gates sha {sha[:12]}")
        _table(f"Exam cells, variant {v} (simulated):", report.cells)
        vs = f"{report.v2:.3f}" if report.status == "MEASURED" else \
            f"UNMEASURED ({report.n_measured}/9 cells; v2_partial {report.v2_partial})"
        print(f"{report.label} {v}: {vs}")
        ne = _nonexam_scores(r["out"][v]["nonexam"])
        _table(f"Non-exam forward, variant {v} (simulated):", ne)
        all_ne = np.concatenate([np.asarray(x["e"]) for x in r["out"][v]["nonexam"].values()])
        overall[v] = (report.overall_median_px, float(np.median(all_ne)))
        print(f"overall exam median {v} {overall[v][0]:.1f} px  overall non-exam median {overall[v][1]:.1f} px")
    rule = overall["B"][0] < overall["A"][0] and overall["B"][1] < overall["A"][1]
    print(f"E4 rule (B exam < A exam and B non-exam < A non-exam): {rule}")
    print(f"runtime {time.perf_counter() - t0:.1f} s")
    print(f"S024 part 4: A overall {overall['A'][0]:.1f} px, B overall {overall['B'][0]:.1f} px (simulated)")


if __name__ == "__main__":
    main()
