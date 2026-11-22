"""S041: people file tool. Prints codes only (names only with `list --names`).

  python eye-tracker/tools/people_tool.py add "<name>"
  python eye-tracker/tools/people_tool.py list [--names]
  --data-root PATH   (default: DATA_ROOT)
Errors: prints `refused: <code>` (never the name) and exits 2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OPUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPUS))

from iable_eye.paths import DATA_ROOT  # noqa: E402
from iable_eye.people import PeopleError, add_person, load_people  # noqa: E402


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="people file (codes only)")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("name")
    ls = sub.add_parser("list")
    ls.add_argument("--names", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "add":
            print(add_person(args.name, data_root=args.data_root))
        else:
            for code, name in load_people(data_root=args.data_root):
                print(f"{code}  {name}" if args.names else code)
    except PeopleError as e:
        print(f"refused: {e.code}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
