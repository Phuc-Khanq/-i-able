"""Where things live, and the check that keeps data out of ROOT and OneDrive.

Standard library only.
"""
from __future__ import annotations

from pathlib import Path

# ROOT = the project folder: .../Project Iable (this file is ROOT/opus/iable_eye/paths.py).
ROOT = Path(__file__).resolve().parents[2]

# DATA_ROOT = the only place for recordings, models and face-movement numbers.
DATA_ROOT = Path.home() / "iable_data"


class DataPathError(ValueError):
    """A path where data must not be written."""


def _is_inside(child: Path, parent: Path) -> bool:
    """True if `child` is `parent` or below it. Case-insensitive (Windows)."""
    c = [s.lower() for s in child.parts]
    p = [s.lower() for s in parent.parts]
    return len(c) >= len(p) and c[: len(p)] == p


def _is_onedrive_part(part: str) -> bool:
    s = part.lower()
    return s == "onedrive" or s.startswith(("onedrive ", "onedrive-"))


def assert_data_path(p, data_root=DATA_ROOT, root=ROOT) -> Path:
    """Return the resolved path if it is a safe place for data; else raise DataPathError.

    Refused: (a) inside `root`, (b) any part is a OneDrive folder, (c) not inside `data_root`.
    Never creates anything.
    """
    rp = Path(p).resolve()
    r_root = Path(root).resolve()
    r_data = Path(data_root).resolve()
    if _is_inside(rp, r_root):
        raise DataPathError(f"inside the project folder: {rp}")
    if any(_is_onedrive_part(s) for s in rp.parts):
        raise DataPathError(f"OneDrive path: {rp}")
    if not _is_inside(rp, r_data):
        raise DataPathError(f"not inside the data folder {r_data}: {rp}")
    return rp
