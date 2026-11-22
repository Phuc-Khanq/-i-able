"""record: record one dot session (form short, recal or session) for one person.

    python eye-tracker/tools/record.py --person P001 --dry-run
    python eye-tracker/tools/record.py --person P001 --form short
    python eye-tracker/tools/record.py --person P001 --form session
    python eye-tracker/tools/record.py --fake --data-root ~/iable_data/SIM_rec_x --form short

--dry-run  prints k, the exam flag, the cell to set up, the timeline and the target; opens and creates nothing.
--fake     FakeClock + FakeCamera + a simulated session (person S902, source sim); needs --data-root.
Live mode  real camera + MediaPipe + a full-screen dot window; the person must be in people.local.json.
Screen size: give both --screen-w and --screen-h, or neither (then the real DPI-aware size; 1920 x 1080 with --fake).
The cell: --cell bright/normal (default for a real person) or --cell rotation ((k - 1) mod 9; default with --fake).
Numbers only: no frame and no canvas is ever written.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import people  # noqa: E402
from iable_eye import session_store as store  # noqa: E402
from iable_eye.display import FALLBACK_SIZE, DisplayError, detect_screen  # noqa: E402
from iable_eye.instructions import BlockGate, PauseClock, parts_from_timeline  # noqa: E402
from iable_eye.capture import CaptureError, CvCamera, FakeCamera, FakeClock, ReplayTracker  # noqa: E402
from iable_eye.mp_tracker import DEFAULT_MODEL, MpTracker  # noqa: E402
from iable_eye.paths import DATA_ROOT, ROOT, DataPathError  # noqa: E402
from iable_eye.protocol import ProtocolError, build_timeline, describe, state_at  # noqa: E402
from iable_eye.recorder import (FOCUS_NAME, FORM_CODES, RecorderError, ScreenWindow, cell_for,  # noqa: E402
                                parse_cell, record_session, render_state)
from iable_eye.sim_protocol import PROTOCOL_DURATION_S  # noqa: E402
from iable_eye.sim_session import simulate_session  # noqa: E402
from iable_eye.tracker import TrackerError  # noqa: E402

FAKE_PERSON = "S902"
FAKE_SEED = 20261116
FAKE_EXTRA_S = 7.0


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--person")
    ap.add_argument("--form", default="session", choices=sorted(FORM_CODES))
    ap.add_argument("--screen-w", type=int, default=None)
    ap.add_argument("--screen-h", type=int, default=None)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--posture", default="upright", choices=store.POSTURES)
    ap.add_argument("--glasses", default="off", choices=store.GLASSES)
    ap.add_argument("--camera", default="builtin", choices=store.CAMERAS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--data-root")
    ap.add_argument("--cell", default=None)   # S057: "rotation" or "<light>/<distance>"
    return ap


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if (a.screen_w is None) != (a.screen_h is None):
        print("REFUSED: give both --screen-w and --screen-h, or neither")
        return 2
    if a.fake:
        if not a.data_root:
            print("REFUSED: --fake needs --data-root")
            return 2
        if a.person not in (None, FAKE_PERSON):
            print(f"REFUSED: --fake always records the simulated person {FAKE_PERSON}")
            return 2
        person, source = FAKE_PERSON, "sim"
    else:
        if not a.person:
            print("REFUSED: --person is needed (a code like P001)")
            return 2
        person, source = a.person, "real"
    data_root = Path(a.data_root) if a.data_root else DATA_ROOT
    try:
        if source == "real" and person not in people.list_codes(data_root):
            print(f"REFUSED: {person} is not in the people file (add with opus/tools/people_tool.py)")
            return 2
        # S057: the cell is fixed before the size rule; a bad name is refused before any window.
        name = a.cell or ("rotation" if a.fake else FOCUS_NAME)
        a.cell_t = parse_cell(name)
        a.cell_mode = "rotation" if a.cell_t is None else f"focus {name}"
        # S054: the size is fixed (DPI-aware) before any window or timeline is built.
        if a.screen_w is not None:
            a.dpi_mode = "given"
        elif a.fake:
            (a.screen_w, a.screen_h), a.dpi_mode = FALLBACK_SIZE, "fake"
        else:
            a.screen_w, a.screen_h, a.dpi_mode = detect_screen()
        tl = build_timeline(a.form, a.screen_w, a.screen_h)
        if a.dry_run:
            return _dry_run(a, person, source, data_root, tl)
        if a.fake:
            return _fake(a, data_root, tl)
        return _live(a, person, data_root, tl)
    except (store.SessionStoreError, CaptureError, RecorderError, ProtocolError, TrackerError, DataPathError,
            people.PeopleError, DisplayError) as e:
        print(f"REFUSED: {e}")
        return 2


def _dry_run(a, person, source, data_root, tl) -> int:
    k = store.next_free_index(person, source, data_root=data_root, root=ROOT)
    light, distance = cell_for(k, a.cell_t)
    print(f"k = {k}")
    print(f"exam: {'yes' if store.exam_flag(k) else 'no'} (never trained on)")
    print(f"cell to set up: light {light}, distance {distance}; posture {a.posture}, glasses {a.glasses}, "
          f"camera {a.camera}")
    print(f"cell mode: {a.cell_mode}")
    print(f"screen: {a.screen_w} x {a.screen_h} px (dpi {a.dpi_mode})")
    for line in describe(tl):
        print(line)
    print(f"target: {store.session_path(person, k, source, data_root=data_root, root=ROOT)}")
    if a.fake:
        print(f"model: none (fake: simulated session {FAKE_PERSON}, seed {FAKE_SEED})")
    else:
        mp = Path(data_root) / DEFAULT_MODEL
        print(f"model: {mp} ({'found' if mp.is_file() else 'MISSING'})")
    print("nothing opened")
    return 0


def _fake(a, data_root, tl) -> int:
    # S045: the simulator stops at PROTOCOL_DURATION_S; a longer form then ends early (stop "camera").
    sim = simulate_session(1, FAKE_SEED, person_code=FAKE_PERSON,
                           duration_s=min(tl.duration_s + FAKE_EXTRA_S, PROTOCOL_DURATION_S))
    clock = FakeClock(30)
    camera = FakeCamera(int(sim.arrays["t"].shape[0]), fps=30, sleep=clock.sleep)
    path = record_session(camera, ReplayTracker(sim.arrays), person=FAKE_PERSON, source="sim", form=a.form,
                          screen_w=a.screen_w, screen_h=a.screen_h, posture=a.posture, glasses=a.glasses,
                          cam=a.camera, data_root=data_root, root=ROOT, clock=clock.now, now=datetime.now,
                          cell=a.cell_t)
    print(f"written (simulated): {path}")
    return 0


def _hooks(tl, window, base_clock):
    """S056 (pure, tested with a fake window): a page + SPACE before each part; the page time is paused."""
    clock = PauseClock(base_clock)
    gate = BlockGate(parts_from_timeline(tl), clock, window.ask)

    def on_frame(obs, t):
        return gate(t) and window.show(render_state(state_at(tl, t), tl.screen_w_px, tl.screen_h_px))
    return clock, on_frame


def _live(a, person, data_root, tl) -> int:
    tracker = MpTracker(data_root=data_root)
    window = None
    try:
        window = ScreenWindow(a.screen_w, a.screen_h)
        window.open()
        camera = CvCamera(a.camera_index)
        clock, on_frame = _hooks(tl, window, time.perf_counter)   # the part-1 page is the start
        path = record_session(camera, tracker, person=person, source="real", form=a.form,
                              screen_w=a.screen_w, screen_h=a.screen_h, posture=a.posture, glasses=a.glasses,
                              cam=a.camera, data_root=data_root, root=ROOT, clock=clock,
                              now=datetime.now, on_frame=on_frame, cell=a.cell_t)
    finally:
        if window is not None:
            window.close()
        tracker.close()
    if path is None:
        print("stopped with q: nothing written")
    else:
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
