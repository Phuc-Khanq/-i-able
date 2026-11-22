"""S047b: score one person's stored sessions with the frozen pipeline (pre-registered, DECISIONS 2026-11-19):
kernel_ridge, one-look recenter, no augmentation, gates from evaluate.load_gates(). No selection, no tuning.

Phase 1: non-exam forward splits -> per-cell table. Phase 2: exam splits, scored once -> evaluate_v2.
Writes only <data root>/results/<code>_scores.json and the dashboard. Nothing when no session is found.
Real sessions use the protocol "session" grid; --source sim uses the simulator grid (S025 wiring proof).

Usage: python eye-tracker/tools/run_real_scores.py --person P001 [--source real|sim] [--data-root R] [--no-dashboard]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import dashboard, evaluate, metrics  # noqa: E402
from iable_eye import session_store as store  # noqa: E402
from iable_eye.candidates import CandidateError, fit_candidate, predict_candidate, recenter_candidate  # noqa: E402
from iable_eye.dots import dot_prediction  # noqa: E402
from iable_eye.paths import DATA_ROOT, ROOT  # noqa: E402
from iable_eye.quality import REAL_GATE_V1, GateConfig  # noqa: E402
from iable_eye.real_scoring import (CHOSEN_KIND, SCORES_VERSION, ScoreError, load_store_session,  # noqa: E402
                                    scores_path, write_scores)
from iable_eye.sim_core import SimScreen  # noqa: E402
from iable_eye.sim_protocol import N_DOTS, grid_dot_px  # noqa: E402
from iable_eye.splits import assert_no_leak, forward_splits  # noqa: E402
from tools.run_candidates import train_rows_dots  # noqa: E402
from tools.run_gaze_baseline import ALPHA, MIN_TRAIN, ONSETS, _table  # noqa: E402
from tools.run_recal import MIN_FRAMES, look_indexes  # noqa: E402


def sim_grid() -> tuple[list[float], np.ndarray]:
    """The simulator grid: S022 onsets and grid_dot_px positions (32 dots)."""
    return list(ONSETS), np.array([grid_dot_px(d, SimScreen()) for d in range(N_DOTS)], float)


def score_truth(c, sess, frac, shift=None) -> list[tuple[int, float | None]]:
    """Like run_candidates.score_candidate, but the truth is sess["dot_px"][d]."""
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
        out.append((d, float(metrics.dot_errors_px(np.array([p]), np.array([sess["dot_px"][d]]))[0])))
    return out


def fit_and_score_truth(kind, sp, sessions, frac, target, stats):
    """Like run_candidates.fit_and_score, but calls score_truth."""
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
    return score_truth(c, test, frac, shift)


def _num(x):
    return None if x is None or x != x else float(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--source", choices=store.SOURCES, default="real")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--gate", choices=("auto", "sim", "real"), default="auto",
                    help="auto: REAL_GATE_V1 for --source real, GateConfig() for sim (S059, DECISIONS 2026-11-22)")
    a = ap.parse_args(argv)
    root_d, person, source = Path(a.data_root), a.person, a.source
    tag = " (simulated)" if source == "sim" else ""
    t_all = time.perf_counter()
    gates = evaluate.load_gates()
    window_ms, frac = tuple(gates["settled_window_ms"]), float(gates["lost_if_ok_frac_below"])
    grid = sim_grid() if source == "sim" else None
    use_real = a.gate == "real" or (a.gate == "auto" and source == "real")
    gate_config = REAL_GATE_V1 if use_real else GateConfig()
    print(f"gate: blink_score {gate_config.blink_score} ({'REAL_GATE_V1' if use_real else 'GateConfig()'})")
    sessions, skipped = {}, []
    for k in store.list_sessions(person, source, data_root=root_d, root=ROOT):
        s = load_store_session(person, k, source, window_ms, grid=grid, data_root=root_d, root=ROOT,
                               gate_config=gate_config)
        if s is None:
            form = store.read_session(person, k, source, data_root=root_d, root=ROOT).meta.fingerprint["proto_form"]
            skipped.append((k, form))
        else:
            sessions[k] = s
    if skipped:
        print(f"skipped (form is not 'session'): {[f'k {k} form {f}' for k, f in skipped]}")
    if not sessions:
        print(f"no session of {person} in {root_d}: nothing scored")
        return 2
    screens = {s["screen"] for s in sessions.values()}
    if len(screens) != 1:
        raise ScoreError("screen", f"sessions of one run must share one screen, got {sorted(screens)}")
    w, h = screens.pop()
    target = np.array([w / 2.0, h / 2.0])
    s25 = SimScreen()
    print(f"loaded {len(sessions)} sessions, screen {w} x {h}, target {(float(target[0]), float(target[1]))}; "
          f"S025 target {(s25.width_px / 2.0, s25.height_px / 2.0)}; kind {CHOSEN_KIND}{tag}")
    splits = forward_splits([s["info"] for s in sessions.values()], min_train=MIN_TRAIN)
    assert_no_leak(splits)
    stats = {f: {CHOSEN_KIND: 0} for f in ("fit_s", "conv", "fallbacks")}
    stats["exam_scored"] = 0
    # ---- phase 1: non-exam only ----
    ne = {c: {"e": [], "lost": 0} for c in metrics.CELLS}
    for sp in [x for x in splits if not x.test_is_exam]:
        for _, e in fit_and_score_truth(CHOSEN_KIND, sp, sessions, frac, target, stats):
            cell = ne[sessions[sp.test_index]["cell"]]
            if e is None:
                cell["lost"] += 1
            else:
                cell["e"].append(e)
    assert stats["exam_scored"] == 0, "phase 1 scored an exam"
    alle = [e for v in ne.values() for e in v["e"]]
    nonexam_med = float(np.median(alle)) if alle else None
    sc = [metrics.cell_score(c, np.asarray(v["e"]), v["lost"]) for c, v in ne.items() if v["e"] or v["lost"]]
    _table(f"Non-exam forward, {CHOSEN_KIND} + recenter{tag}:", sc)
    print(f"  overall non-exam median {nonexam_med if nonexam_med is None else round(nonexam_med, 1)} px  "
          f"fallbacks {stats['fallbacks'][CHOSEN_KIND]}{tag}")
    # ---- phase 2: exams, once ----
    rows = []
    for sp in [x for x in splits if x.test_is_exam]:
        test = sessions[sp.test_index]
        print(f"  exam {sp.test_index:2d} train {list(sp.train_indexes)}  exams in train "
              f"{[j for j in sp.train_indexes if store.exam_flag(j)]}")
        for _, e in fit_and_score_truth(CHOSEN_KIND, sp, sessions, frac, target, stats):
            rows.append(evaluate.DotRow(person, sp.test_index, source, test["cell"], True, e, "replay"))
    if not rows:
        print(f"no exam split yet (need session 5 and {MIN_TRAIN} earlier non-exam): V2 UNMEASURED, nothing written")
        return 3
    report, interval, sha = evaluate.evaluate_v2(rows)
    _table(f"Exam cells{tag}:", report.cells)
    if report.status == "MEASURED":
        print(f"{report.label}: {report.v2:.3f}  interval [{interval[0]:.3f}, {interval[1]:.3f}]{tag}")
    else:
        print(f"{report.label}: UNMEASURED ({report.n_measured}/9 cells measured; "
              f"v2_partial {report.v2_partial}){tag}")
    print(f"overall exam median {report.overall_median_px:.1f} px  reasons {report.reasons}{tag}")
    obj = {
        "cells": [{"cell": c.cell, "n": int(c.n), "lost": int(c.lost), "median_px": _num(c.median_px)}
                  for c in report.cells],
        "chosen_kind": CHOSEN_KIND, "exam_rows": len(rows), "gate_blink_score": gate_config.blink_score,
        "gates_sha": sha,
        "interval": None if interval is None else [float(interval[0]), float(interval[1])],
        "nonexam_median_px": nonexam_med, "overall_exam_median_px": _num(report.overall_median_px),
        "person": person, "scores_version": SCORES_VERSION, "sessions_scored": len(sessions),
        "sessions_skipped": len(skipped), "source": source, "status": report.status,
        "v2": _num(report.v2), "v2_partial": _num(report.v2_partial),
    }
    path = scores_path(person, data_root=root_d, root=ROOT)
    write_scores(obj, path)
    print(f"scores: {path}")
    if not a.no_dashboard:
        iv = None if interval is None else (float(interval[0]), float(interval[1]))
        page = dashboard.render_dashboard([dashboard.DashEntry(person, report, iv, sha)],
                                          generated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"dashboard: {dashboard.write_dashboard(page, data_root=root_d, root=ROOT)}")
    print(f"runtime {time.perf_counter() - t_all:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
