"""S064 rule D1 (pre-registered, DECISIONS 2026-11-22): do the demo calibration recordings make the gaze
model better? Time order only: recording j is scored by models that saw only earlier data.

    M0 = kernel_ridge on all non-exam protocol sessions (never an exam)
    M1 = M0's rows + the "calib" rows of the demo recordings before j
    score: recording j's calib dots except the centre dot, after a centre-dot recenter
    useful  <=>  >= 2 recordings scored  AND  median(M1) <= median(M0) - 10 px

Writes only <data root>/results/<person>_demo_check.json (train_demo_model.py --demo auto reads it).

Usage: python eye-tracker/tools/check_demo_data.py --person P001 [--data-root R]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye.candidates import CandidateError, fit_candidate, predict_candidate, recenter_candidate  # noqa: E402
from iable_eye.demo_record import LABEL_KINDS, label_rows, list_demos, read_demo  # noqa: E402
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path  # noqa: E402
from iable_eye.real_scoring import CHOSEN_KIND  # noqa: E402
from tools.run_gaze_baseline import ALPHA  # noqa: E402
from tools.train_demo_model import protocol_rows  # noqa: E402

MARGIN_PX, MIN_SCORED, MIN_DOTS, MIN_FRAMES = 10.0, 2, 6, 5
CALIB = LABEL_KINDS.index("calib")
DEMO_GROUP = 1000


def usable(demo) -> list:
    """Calib label indexes with >= MIN_FRAMES ok frames."""
    X, _, D = label_rows(demo)
    return [int(j) for j in np.unique(D) if (D == j).sum() >= MIN_FRAMES] if len(D) else []


def demo_train_rows(demos) -> tuple:
    Xs, Ys, Gs, Ds = [], [], [], []
    for i, d in enumerate(demos):
        X, Y, D = label_rows(d)
        Xs.append(X)
        Ys.append(Y)
        Gs.append(np.full(len(X), DEMO_GROUP + i, dtype=np.int64))
        Ds.append(D + 100000 * (i + 1))
    return Xs, Ys, Gs, Ds


def score_demo(c, demo) -> list:
    """Per calib dot (not the centre one): px error of the median prediction after the centre recenter."""
    X, Y, D = label_rows(demo)
    calib_ix = [j for j, row in enumerate(demo["labels"]) if int(row[0]) == CALIB]
    if not calib_ix:
        return []
    centre = calib_ix[0]
    m = D == centre
    try:
        shift = recenter_candidate(c, X[m], demo["labels"][centre][3:5], min_frames=MIN_FRAMES)
    except CandidateError:
        return []
    out = []
    for j in calib_ix[1:]:
        m = D == j
        if m.sum() < MIN_FRAMES:
            continue
        p = np.median(predict_candidate(c, X[m], shift), axis=0)
        out.append(float(np.hypot(*(p - demo["labels"][j][3:5]))))
    return out


def check(person, data_root=DATA_ROOT, log=print) -> dict:
    Xs, Ys, Gs, Ds, used, exams = protocol_rows(person, "real", data_root)
    demos, names = [], []
    for p in list_demos(person, data_root, ROOT):
        d = read_demo(p)
        if len(usable(d)) >= MIN_DOTS:
            demos.append(d)
            names.append(p.name)
    log(f"protocol sessions {used} (exams {exams} not used); demo recordings with >= {MIN_DOTS} calib dots: {len(demos)}")

    def fit(extra):
        ex = demo_train_rows(extra)
        X = np.vstack(Xs + ex[0])
        return fit_candidate(CHOSEN_KIND, X, np.vstack(Ys + ex[1]), np.concatenate(Gs + ex[2]),
                             np.concatenate(Ds + ex[3]), alpha=ALPHA)
    m0 = fit([])
    e0, e1, rows = [], [], []
    for j in range(1, len(demos)):
        a, b = score_demo(m0, demos[j]), score_demo(fit(demos[:j]), demos[j])
        e0 += a
        e1 += b
        rows.append({"demo": names[j], "trained_on_demos": names[:j], "m0_median": float(np.median(a)) if a else None,
                     "m1_median": float(np.median(b)) if b else None, "dots": len(a)})
        log(f"  {names[j]}: M0 {rows[-1]['m0_median']} px, M1 {rows[-1]['m1_median']} px ({len(a)} dots)")
    scored = sum(1 for r in rows if r["dots"])
    med0 = float(np.median(e0)) if e0 else None
    med1 = float(np.median(e1)) if e1 else None
    useful = bool(scored >= MIN_SCORED and med0 is not None and med1 is not None and med1 <= med0 - MARGIN_PX)
    log(f"D1: scored {scored} recordings; median M0 {med0}, M1 {med1} px -> useful {useful}"
        + ("" if scored >= MIN_SCORED else f" (need >= {MIN_SCORED} scored recordings: record more fusion runs)"))
    return {"person": person, "rows": rows, "scored": scored, "m0_median": med0, "m1_median": med1,
            "useful": useful, "rule": "D1 v1 (DECISIONS 2026-11-22)", "demos": names}


def check_path(person, data_root=DATA_ROOT) -> Path:
    return assert_data_path(Path(data_root) / "results" / f"{person}_demo_check.json", data_root=data_root, root=ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    a = ap.parse_args(argv)
    obj = check(a.person, Path(a.data_root))
    path = check_path(a.person, Path(a.data_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
