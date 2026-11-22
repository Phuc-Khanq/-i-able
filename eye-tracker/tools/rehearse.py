"""S045: rehearse the recording morning with FakeCamera, command by command.

    python eye-tracker/tools/rehearse.py          # run the 17 steps in a fresh temp data root, then delete it
    python eye-tracker/tools/rehearse.py --keep   # keep the temp root and print its path

Every step is a real child process (sys.executable + the tool), the same way a person types it.
A step passes only if the exit code matches and the key text is in stdout.
No step opens a camera: only --fake, --dry-run or a refusal.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPUS))

from iable_eye import session_store as store  # noqa: E402
from iable_eye.paths import DATA_ROOT, ROOT  # noqa: E402
from iable_eye.probe import probe_root  # noqa: E402
from iable_eye.protocol import build_timeline  # noqa: E402
from iable_eye.sim_protocol import PROTOCOL_DURATION_S  # noqa: E402
from iable_eye.sim_session import simulate_session  # noqa: E402

NAME = "Rehearsal Person"  # X4: made up, never a real name
PREFIXES = ("iable_rehearsal_", "_rehearsal_")
TIMEOUT_S = 900
# Copied from opus/tools/record.py (a test checks they are equal).
FAKE_PERSON, FAKE_SEED, FAKE_EXTRA_S = "S902", 20261116, 7.0


@dataclass(frozen=True)
class Step:
    label: str
    script: str
    args: tuple
    want_rc: int
    want_text: str  # must be inside stdout


def steps(root: str) -> list[Step]:
    p = ("--data-root", root)
    out = [Step("1 add person", "people_tool.py", (*p, "add", NAME), 0, "P001"),
           Step("2 list", "people_tool.py", (*p, "list"), 0, "P001"),
           Step("3 probe dry-run", "envprobe.py",
                ("--person", "P001", "--light", "bright", "--distance", "near", "--dry-run", *p), 0, "nothing opened"),
           Step("4 probe check refused", "envprobe.py",
                ("--fake", "--check", "--light", "bright", "--distance", "near", *p), 2, "REFUSED")]
    for light in store.LIGHTS:
        for dist in store.DISTANCES:
            out.append(Step(f"{len(out) + 1} probe {light} {dist}", "envprobe.py",
                            ("--fake", "--light", light, "--distance", dist, *p), 0, "written (simulated)"))
    out += [Step("14 record dry-run", "record.py", ("--person", "P001", "--form", "session", "--dry-run", *p),
                 0, "k = 1"),
            Step("15 record short", "record.py", ("--fake", "--form", "short", *p), 0, "written (simulated)"),
            Step("16 record session", "record.py", ("--fake", "--form", "session", *p), 0, "written (simulated)"),
            Step("17 unknown person", "record.py", ("--person", "P999", "--dry-run", *p), 2,
                 "not in the people file")]
    return out


def run_step(step: Step) -> tuple[int, list[str]]:
    r = subprocess.run([sys.executable, str(OPUS / "tools" / step.script), *step.args],
                       capture_output=True, text=True, timeout=TIMEOUT_S)
    return r.returncode, r.stdout.splitlines() + [f"[stderr] {x}" for x in r.stderr.splitlines()]


def step_passes(step: Step, rc: int, lines: list[str]) -> bool:
    out = "\n".join(x for x in lines if not x.startswith("[stderr] "))
    return rc == step.want_rc and step.want_text in out and NAME not in "\n".join(lines)


def count_files(root) -> dict:
    root, pr = Path(root), probe_root(root)
    npz = list(root.rglob("*.npz"))
    probes = sum(1 for f in npz if pr in f.parents)
    return {"probes": probes, "sessions": len(npz) - probes}


def fake_agreement(root) -> dict:
    """Stored protocol labels of session S902/2 vs the simulator's own truth (a finding, not a gate)."""
    s = store.read_session(FAKE_PERSON, 2, "sim", data_root=root, root=ROOT)
    tl = build_timeline("session", 1920, 1080)
    sim = simulate_session(1, FAKE_SEED, person_code=FAKE_PERSON,
                           duration_s=min(tl.duration_s + FAKE_EXTRA_S, PROTOCOL_DURATION_S))
    n = min(s.n_frames, sim.channels["gaze_px"].shape[0])
    a, b = s.channels["gaze_px"][:n], sim.channels["gaze_px"][:n]
    both = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    d = np.abs(a[both] - b[both]).max(axis=1) if both.any() else np.zeros(0)
    dot_a, dot_b = s.channels["dot"][:n], sim.channels["dot"][:n]
    return {"channel": "sim.channels['gaze_px'] vs stored channels['gaze_px']; 'dot' vs 'dot'",
            "frames_stored": int(s.n_frames), "frames_sim": int(sim.channels["gaze_px"].shape[0]),
            "frames_compared": int(n), "both_finite": int(both.sum()),
            "max_diff_px": float(d.max()) if d.size else float("nan"),
            "frames_diff_gt_1px": int((d > 1.0).sum()), "dot_differs": int((dot_a != dot_b).sum())}


def safe_delete(path) -> None:
    """Delete a rehearsal root. Raises (and deletes nothing) unless the name has a rehearsal prefix."""
    p = Path(path).resolve()
    for bad in (Path(ROOT).resolve(), Path(DATA_ROOT).resolve()):
        if p == bad or p in bad.parents:
            raise ValueError(f"refused to delete {p}: it is or holds a protected folder")
    if p.is_relative_to(Path(ROOT).resolve()):
        raise ValueError(f"refused to delete {p}: inside the project folder")
    if not p.name.startswith(PREFIXES):
        raise ValueError(f"refused to delete {p}: name does not start with {PREFIXES}")
    shutil.rmtree(p)


def _count_data_root() -> int:
    return sum(len(f) for _, _, f in os.walk(DATA_ROOT)) if Path(DATA_ROOT).is_dir() else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rehearse the recording morning with FakeCamera")
    ap.add_argument("--keep", action="store_true", help="keep the temp data root and print its path")
    a = ap.parse_args(argv)
    data_before = _count_data_root()
    root = tempfile.mkdtemp(prefix=PREFIXES[0])
    ok, t_all = True, time.perf_counter()
    try:
        print(f"{'#':>2}  rc want  {'first stdout line':<90}  sec  result")
        for i, st in enumerate(steps(root), 1):
            t0 = time.perf_counter()
            rc, lines = run_step(st)
            good = step_passes(st, rc, lines)
            ok &= good
            first = (lines[0] if lines else "")[:90]
            print(f"{i:>2}  {rc:>2} {st.want_rc:>4}  {first:<90}  {time.perf_counter() - t0:5.1f}  "
                  f"{'PASS' if good else 'FAIL'}")
            if not good:
                for x in lines[-8:]:
                    print(f"      | {x[:150]}")
        files = count_files(root)
        files_ok = files == {"probes": 9, "sessions": 2}
        ok &= files_ok
        print(f"files: {files} ({'PASS' if files_ok else 'FAIL'}; want 9 probes + 2 sessions)")
        try:
            print("fake check (simulated):", fake_agreement(root))
        except Exception as e:  # a finding, not a gate
            print(f"fake check: could not run ({type(e).__name__}: {e})")
        print(f"total {time.perf_counter() - t_all:.1f} s; iable_data files before {data_before}, "
              f"after {_count_data_root()}")
    finally:
        if a.keep:
            print(f"kept: {root}")
        else:
            safe_delete(root)
            print(f"deleted temp root; exists: {Path(root).exists()}")
    print("REHEARSAL PASS" if ok else "REHEARSAL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
