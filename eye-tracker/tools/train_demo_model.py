"""S059/S064: train the demo gaze model for one person and save it to <data root>/models/.

Pipeline = the pre-registered one (DECISIONS 2026-11-19): kernel_ridge, alpha 1.0, settled window
400-1200 ms; gate REAL_GATE_V1 for real data (DECISIONS 2026-11-22). Trained on every NON-EXAM
"session" form recording only: exam sessions (every 5th k) are never trained on (CLAUDE.md rule 7).
S064: --demo auto adds the "calib" rows of the demo recordings only when check_demo_data.py said
"useful" (rule D1, DECISIONS 2026-11-22); --demo on / off forces it.
Prints what it used; writes one .npz file. Numbers only.

Usage: python eye-tracker/tools/train_demo_model.py --person P001 [--source real|sim] [--data-root R]
       [--kind kernel_ridge|ridge] [--demo auto|on|off]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import evaluate  # noqa: E402
from iable_eye import session_store as store  # noqa: E402
from iable_eye.candidates import fit_candidate  # noqa: E402
from iable_eye.model_store import model_path, save_candidate  # noqa: E402
from iable_eye.paths import DATA_ROOT, ROOT  # noqa: E402
from iable_eye.quality import REAL_GATE_V1, GateConfig  # noqa: E402
from iable_eye.real_scoring import CHOSEN_KIND, load_store_session  # noqa: E402
from tools.run_candidates import train_rows_dots  # noqa: E402
from tools.run_gaze_baseline import ALPHA  # noqa: E402


def protocol_rows(person, source="real", data_root=DATA_ROOT, grid=None):
    """Training rows of the non-exam "session" recordings: lists Xs, Ys, Gs, Ds, plus used k, exam k."""
    window_ms = tuple(evaluate.load_gates()["settled_window_ms"])
    gate = REAL_GATE_V1 if source == "real" else GateConfig()
    Xs, Ys, Gs, Ds, used, exams = [], [], [], [], [], []
    for k in store.list_sessions(person, source, data_root=data_root, root=ROOT):
        if store.exam_flag(k):
            exams.append(k)
            continue
        s = load_store_session(person, k, source, window_ms, grid=grid, data_root=data_root, root=ROOT,
                               gate_config=gate)
        if s is None:
            continue
        X, Y, D = train_rows_dots(s)
        if len(X) == 0:
            continue
        Xs.append(X)
        Ys.append(Y)
        Ds.append(D)
        Gs.append(np.full(len(X), k, dtype=np.int64))
        used.append(k)
    return Xs, Ys, Gs, Ds, used, exams


def demo_useful(person, data_root) -> bool:
    from tools.check_demo_data import check_path
    p = check_path(person, data_root)
    return p.is_file() and bool(json.loads(p.read_text(encoding="utf-8")).get("useful"))


def train(person, source="real", data_root=DATA_ROOT, kind=CHOSEN_KIND, grid=None, demo="auto", log=print):
    """-> (Candidate, info dict). Raises ValueError when no training row exists."""
    Xs, Ys, Gs, Ds, used, exams = protocol_rows(person, source, data_root, grid)
    if not used:
        raise ValueError(f"no non-exam training session for {person} ({source}) in {data_root}")
    gate = REAL_GATE_V1 if source == "real" else GateConfig()
    use_demo = demo == "on" or (demo == "auto" and source == "real" and demo_useful(person, data_root))
    demo_names = []
    if use_demo:
        from iable_eye.demo_record import list_demos, read_demo
        from tools.check_demo_data import demo_train_rows
        paths = list_demos(person, data_root, ROOT)
        ex = demo_train_rows([read_demo(p) for p in paths])
        Xs, Ys, Gs, Ds = Xs + ex[0], Ys + ex[1], Gs + ex[2], Ds + ex[3]
        demo_names = [p.name for p in paths]
    c = fit_candidate(kind, np.vstack(Xs), np.vstack(Ys), np.concatenate(Gs), np.concatenate(Ds), alpha=ALPHA)
    info = {"person": person, "source": source, "kind": kind, "sessions": used, "exams_skipped": exams,
            "rows": int(sum(len(x) for x in Xs)), "gate_blink_score": gate.blink_score,
            "gates_sha": evaluate.sha256_file(evaluate.GATES_PATH), "demo_recordings": demo_names}
    log(f"trained {kind} on sessions {used} ({info['rows']} rows); exams not used: {exams}; "
        f"gate blink_score {gate.blink_score}; demo recordings used: {len(demo_names)}"
        + ("" if use_demo or demo == "off" else " (rule D1 has not said 'useful')"))
    return c, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--source", choices=store.SOURCES, default="real")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--kind", choices=("kernel_ridge", "ridge"), default=CHOSEN_KIND)
    ap.add_argument("--demo", choices=("auto", "on", "off"), default="auto")
    a = ap.parse_args(argv)
    root_d = Path(a.data_root)
    grid = None
    if a.source == "sim":
        from tools.run_real_scores import sim_grid
        grid = sim_grid()
    try:
        c, info = train(a.person, a.source, root_d, a.kind, grid=grid, demo=a.demo)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    path = save_candidate(c, model_path(a.person, a.kind, data_root=root_d, root=ROOT), info)
    print(f"model: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
