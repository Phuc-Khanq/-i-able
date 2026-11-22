"""S041: the people file. Codes P001, P002, ... in order; names only in DATA_ROOT.

File (UTF-8): {"version": 1, "next": 3, "people": [{"code": "P001", "name": "An"}, ...]}
`next` only grows, so a deleted line never frees its code.
Never prints or logs a name.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

from iable_eye.paths import DATA_ROOT, ROOT, DataPathError, assert_data_path

PEOPLE_FILE = "people.local.json"
MAX_NAME = 80
_CODE_RE = re.compile(r"^P[0-9]{3}$")


class PeopleError(ValueError):
    """Refused. `.code` is one of: path, file, name, duplicate, full."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def people_path(data_root=DATA_ROOT, root=ROOT) -> Path:
    """Safe path of the people file. Never creates a folder."""
    d = Path(data_root)
    if not d.is_dir():
        raise PeopleError("path")
    try:
        return assert_data_path(d / PEOPLE_FILE, data_root, root)
    except DataPathError:
        raise PeopleError("path") from None


def _norm(name: str) -> str:
    return unicodedata.normalize("NFC", name.strip())


def _read(path: Path) -> dict:
    """Read and check the file. Missing file -> empty data."""
    if not path.exists():
        return {"version": 1, "next": 1, "people": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise PeopleError("file") from None
    if not isinstance(data, dict) or data.get("version") != 1:
        raise PeopleError("file")
    nxt = data.get("next")
    if type(nxt) is not int or not 1 <= nxt <= 1000:
        raise PeopleError("file")
    people = data.get("people")
    if not isinstance(people, list):
        raise PeopleError("file")
    last = 0
    for p in people:
        if not isinstance(p, dict):
            raise PeopleError("file")
        code, name = p.get("code"), p.get("name")
        if not isinstance(code, str) or not _CODE_RE.match(code) or not isinstance(name, str):
            raise PeopleError("file")
        num = int(code[1:])
        if num <= last or num >= nxt:  # rising (so no duplicates) and below next
            raise PeopleError("file")
        last = num
    return data


def _write(path: Path, data: dict) -> None:
    """Atomic write: temp file in the same folder, then os.replace."""
    tmp = path.with_name(PEOPLE_FILE + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_people(data_root=DATA_ROOT, root=ROOT) -> list[tuple[str, str]]:
    data = _read(people_path(data_root, root))
    return [(p["code"], p["name"]) for p in data["people"]]


def add_person(name, data_root=DATA_ROOT, root=ROOT) -> str:
    """Add one person; return the new code."""
    path = people_path(data_root, root)
    if not isinstance(name, str):
        raise PeopleError("name")
    n = _norm(name)
    if not n or len(n) > MAX_NAME or any(unicodedata.category(c).startswith("C") for c in n):
        raise PeopleError("name")
    data = _read(path)
    key = n.casefold()
    if any(_norm(p["name"]).casefold() == key for p in data["people"]):
        raise PeopleError("duplicate")
    if data["next"] >= 1000:
        raise PeopleError("full")
    code = f"P{data['next']:03d}"
    data["people"].append({"code": code, "name": n})
    data["next"] += 1
    _write(path, data)
    return code


def list_codes(data_root=DATA_ROOT, root=ROOT) -> list[str]:
    return [c for c, _ in load_people(data_root, root)]
