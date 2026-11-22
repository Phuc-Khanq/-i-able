"""S025: candidate gaze models (ridge, kernel_ridge, mlp) on simulated sessions.
Prints only; writes nothing. Every number is (simulated).

Phase 1: non-exam forward splits, every kind, recenter (S024 B). Never scores an exam.
          Wiring proof: ridge must reproduce S024 B non-exam 150.9 px. Then the frozen rule.
Phase 2: exam splits, the chosen kind only, scored once -> evaluate_v2.

Usage: python eye-tracker/tools/run_candidates.py [--n 45] [--duration S]
"""
import argparse
import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import evaluate, metrics  # noqa: E402
from iable_eye.candidates import (KINDS, CandidateError, fit_candidate,  # noqa: E402
                                  predict_candidate, recenter_candidate, select_model)
from iable_eye.dots import dot_prediction  # noqa: E402
from iable_eye.sim_core import SimScreen  # noqa: E402
from iable_eye.sim_protocol import PROTOCOL_DURATION_S, grid_dot_px  # noqa: E402
from iable_eye.splits import assert_no_leak, forward_splits  # noqa: E402
from tools.run_gaze_baseline import (ALPHA, MIN_TRAIN, PERSON, _table, load_session,  # noqa: E402
                                     train_rows)
from tools.run_recal import MIN_FRAMES, look_indexes  # noqa: E402

S024_RIDGE_NONEXAM, S024_RIDGE_EXAM = 150.9, 131.1


def train_rows_dots(sess):
    """Same rows and order as train_rows, plus the dot id of each row."""
    if not sess["windows"]:
        return sess["feats"][[]], sess["gaze_px"][[]], np.array([], int)
    ix = np.concatenate([w for _, w in sess["windows"]])
    dt = np.concatenate([np.full(len(w), d, dtype=np.int64) for d, w in sess["windows"]])
    keep = sess["ok"][ix]
    return sess["feats"][ix[keep]], sess["gaze_px"][ix[keep]], dt[keep]


def score_candidate(c, sess, frac, screen, shift=None):
    """Like run_gaze_baseline.score_session, but with predict_candidate."""
    pred = np.full((sess["n_frames"], 2), np.nan)
    ok = sess["ok"]
    if ok.any():
        pred[ok] = predict_candidate(c, sess["feats"][ok], shift)
    out = []
    for d, ix in sess["windows"]:
        p = dot_prediction(pred, ok, ix, sess["n_expected"], frac)
        if p is None:
            out.append((d, None))
            continue
        out.append((d, float(metrics.dot_errors_px(np.array([p]), np.array([grid_dot_px(d, screen)]))[0])))
    return out


def fit_and_score(kind, sp, sessions, frac, screen, target, stats):
    Xs, Ys, Gs, Ds = [], [], [], []
    for j in sp.train_indexes:
        X, Y, D = train_rows_dots(sessions[j])
        Xs.append(X)
        Ys.append(Y)
        Ds.append(D)
        Gs.append(np.full(len(X), j, dtype=np.int64))
    stats["exam_scored"] += int(sp.test_is_exam)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = fit_candidate(kind, np.vstack(Xs), np.vstack(Ys), np.concatenate(Gs), np.concatenate(Ds), alpha=ALPHA)
    stats["fit_s"][kind] += time.perf_counter() - t0
    stats["conv"][kind] += sum(1 for x in w if x.category.__name__ == "ConvergenceWarning")
    test = sessions[sp.test_index]
    try:
        shift = recenter_candidate(c, test["feats"][look_indexes(test)], target, min_frames=MIN_FRAMES)
    except CandidateError as e:
        if e.code != "few":
            raise
        shift, stats["fallbacks"][kind] = None, stats["fallbacks"][kind] + 1
    return score_candidate(c, test, frac, screen, shift)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--duration", type=float, default=PROTOCOL_DURATION_S)
    a = ap.parse_args(argv)
    t_all = time.perf_counter()
    sha_before = hashlib.sha256(evaluate.GATES_PATH.read_bytes()).hexdigest()
    gates = evaluate.load_gates()
    window_ms, frac = tuple(gates["settled_window_ms"]), float(gates["lost_if_ok_frac_below"])
    screen = SimScreen()
    target = np.array([screen.width_px / 2.0, screen.height_px / 2.0])
    sessions = {k: load_session(k, a.duration, window_ms) for k in range(1, a.n + 1)}
    print(f"loaded {len(sessions)} sessions ({time.perf_counter() - t_all:.1f} s)")
    s1 = sessions[1]
    X1, Y1, _ = train_rows_dots(s1)
    X0, Y0 = train_rows(s1)
    assert np.array_equal(X1, X0) and np.array_equal(Y1, Y0)
    print("train_rows_dots == train_rows (session 1): True")
    splits = forward_splits([s["info"] for s in sessions.values()], min_train=MIN_TRAIN)
    assert_no_leak(splits)
    stats = {f: {k: 0 for k in KINDS} for f in ("fit_s", "conv", "fallbacks")}
    stats["exam_scored"] = 0

    # ---- phase 1: non-exam only ----
    ne = {k: {c: {"e": [], "lost": 0} for c in metrics.CELLS} for k in KINDS}
    for sp in splits:
        if sp.test_is_exam:
            continue
        for kind in KINDS:
            for _, e in fit_and_score(kind, sp, sessions, frac, screen, target, stats):
                cell = ne[kind][sessions[sp.test_index]["cell"]]
                if e is None:
                    cell["lost"] += 1
                else:
                    cell["e"].append(e)
    exam_scored_p1 = stats["exam_scored"]
    assert exam_scored_p1 == 0, "phase 1 scored an exam"
    print(f"phase 1: non-exam splits {sum(not s.test_is_exam for s in splits)}, exam sessions scored {exam_scored_p1}")
    medians = {}
    for kind in KINDS:
        allv = np.concatenate([np.asarray(v["e"]) for v in ne[kind].values()])
        medians[kind] = float(np.median(allv))
        sc = [metrics.cell_score(c, np.asarray(v["e"]), v["lost"]) for c, v in ne[kind].items() if v["e"] or v["lost"]]
        _table(f"Non-exam forward, {kind} + recenter (simulated):", sc)
        print(f"  {kind}: overall non-exam median {medians[kind]:.1f} px  fit {stats['fit_s'][kind]:.1f} s  "
              f"ConvergenceWarnings {stats['conv'][kind]}  recenter fallbacks {stats['fallbacks'][kind]}")
    ok_wire = abs(medians["ridge"] - S024_RIDGE_NONEXAM) <= 0.1
    print(f"wiring proof: ridge {medians['ridge']:.2f} vs S024 B {S024_RIDGE_NONEXAM} (tol 0.1): {ok_wire}")
    if not ok_wire:
        print("STOP: ridge does not reproduce S024; phase 2 not run")
        return 2
    chosen = select_model(medians, gates["selection_rule"])
    print(f"selection rule {gates['selection_rule']}")
    print(f"chosen kind: {chosen}")

    # ---- phase 2: exams, chosen kind only, once ----
    rows = []
    for sp in splits:
        if not sp.test_is_exam:
            continue
        test = sessions[sp.test_index]
        print(f"  exam {sp.test_index:2d} train {list(sp.train_indexes)}  exams in train: "
              f"{[j for j in sp.train_indexes if j % 5 == 0]}")
        for _, e in fit_and_score(chosen, sp, sessions, frac, screen, target, stats):
            rows.append(evaluate.DotRow(PERSON, sp.test_index, "sim", test["cell"], True, e, "replay"))
    report, interval, sha = evaluate.evaluate_v2(rows)
    print(f"--- phase 2: exam report, {chosen} + recenter ---  gates sha {sha[:12]}")
    _table("Exam cells (simulated):", report.cells)
    if report.status == "MEASURED":
        print(f"{report.label}: {report.v2:.3f}  worst {report.worst_cell}  best {report.best_cell}  "
              f"interval [{interval[0]:.3f}, {interval[1]:.3f}]")
    else:
        print(f"{report.label}: UNMEASURED ({report.n_measured}/9 cells measured; "
              f"v2_partial {report.v2_partial}); worst {report.worst_cell}; interval None")
    print(f"overall exam median {report.overall_median_px:.1f} px  reasons {report.reasons}  "
          f"gate_ok {report.gate_ok}  product_pass {report.product_pass}")
    if chosen == "ridge":
        print(f"ridge exam check vs S024 {S024_RIDGE_EXAM}: {abs(report.overall_median_px - S024_RIDGE_EXAM) <= 0.1}")
    sha_after = hashlib.sha256(evaluate.GATES_PATH.read_bytes()).hexdigest()
    print(f"gates sha before {sha_before[:12]} after {sha_after[:12]} equal {sha_before == sha_after}")
    print(f"runtime {time.perf_counter() - t_all:.1f} s")
    print(f"S025 part 4: non-exam ridge {medians['ridge']:.1f}, krr {medians['kernel_ridge']:.1f}, "
          f"mlp {medians['mlp']:.1f} px; chosen {chosen}; exam overall {report.overall_median_px:.1f} px (simulated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
