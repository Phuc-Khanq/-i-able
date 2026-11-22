"""record_day: what to set up and what to type next on the recording day (S046).

    python eye-tracker/tools/record_day.py --person P001              # status: the plan
    python eye-tracker/tools/record_day.py --person P001 next         # only the next command(s)
    python eye-tracker/tools/record_day.py --person P001 next --run   # run them (opens the camera!)
    python eye-tracker/tools/record_day.py --fake --data-root ~/iable_data/SIM_day_x next --run

The plan is computed from the files in the store every time; nothing is remembered.
--fake  plans the simulated person S902 (source sim); needs --data-root.
--cell  bright/normal (phase 1, default for a real person) or rotation (default with --fake); passed on to record.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OPUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPUS))

from iable_eye.paths import DATA_ROOT  # noqa: E402
from iable_eye.recorder import FOCUS_NAME, RecorderError, parse_cell  # noqa: E402
from iable_eye.record_day import FAKE_PERSON, DayError, format_plan, plan_day  # noqa: E402
from tools import rehearse  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="recording-day guide")
    ap.add_argument("action", nargs="?", default="status", choices=("status", "next"))
    ap.add_argument("--person")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--data-root")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cell", default=None)   # S057: like record.py (real: bright/normal, fake: rotation)
    return ap


def _plan(a):
    data_root = Path(a.data_root) if a.data_root else DATA_ROOT
    cell = parse_cell(a.cell or ("rotation" if a.fake else FOCUS_NAME))
    if a.fake:
        return plan_day(FAKE_PERSON, "sim", fake=True, data_root=data_root, cell=cell)
    return plan_day(a.person, "real", data_root=data_root, cell=cell)


def _show(cmd) -> str:
    return "python " + " ".join(f'"{x}"' if " " in x else x for x in cmd)


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if a.fake and a.person:
        print("REFUSED: use --person or --fake, not both")
        return 2
    if a.fake and not a.data_root:
        print("REFUSED: --fake needs --data-root")
        return 2
    if not a.fake and not a.person:
        print("REFUSED: give --person P001 or --fake")
        return 2
    if a.run and a.action != "next":
        print("REFUSED: --run works only with next")
        return 2
    try:
        plan = _plan(a)
    except (DayError, RecorderError) as e:
        print(f"REFUSED: {e}")
        return 2
    if a.action == "status":
        print("\n".join(format_plan(plan)))
        return 0
    cmd, n = plan.commands[0], len(plan.commands)   # S055: one command per call
    if not a.run:
        print(_show(cmd))
        if n == 2:
            print("then run next again: 1 more command for this cell")
        return 0
    print(f"running 1 of {n}: {_show(cmd)}")
    step = rehearse.Step(label=Path(cmd[0]).stem, script=Path(cmd[0]).name, args=tuple(cmd[1:]),
                         want_rc=0, want_text="")
    rc, lines = rehearse.run_step(step)
    if rc != 0:
        print(f"rc {rc}: stopped")
        print("\n".join(lines))
        return 1
    print("\n".join(format_plan(_plan(a))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
