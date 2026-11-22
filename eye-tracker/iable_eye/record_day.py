"""S046: the recording-day plan, computed from the files in the store (nothing is remembered).

plan_day() reads the sessions and probes of one person and says: the next k, its cell, whether
the cell still needs a probe, the progress per cell and the next command(s).
format_plan() turns the plan into the exact lines that the tool prints.
This module opens no camera and writes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iable_eye import people, recorder
from iable_eye import session_store as store
from iable_eye.paths import DATA_ROOT, ROOT
from iable_eye.probe import probe_root

NEEDED_PER_CELL = 5
CELL_NAMES = tuple(f"{l}/{d}" for l, d in recorder.CELLS)  # 9 names, CELLS order
SESSION_FORM = float(recorder.FORM_CODES["session"])
FORM_NAMES = {float(v): k for k, v in recorder.FORM_CODES.items()}
# --fake: record.py writes person S902, envprobe.py writes its probes as person S901
# (copied from the two tools; a test checks they are equal).
FAKE_PERSON, FAKE_PROBE_PERSON = "S902", "S901"


class DayError(ValueError):
    """`code` says why: "person" (not in the people file) or "source"."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True)
class DayPlan:
    person: str
    source: str
    fake: bool
    data_root: str
    next_k: int
    next_light: str
    next_distance: str
    next_is_exam: bool
    probes: tuple       # 9 ints, CELLS order: probe sessions per cell
    sessions: tuple     # 9 ints: sessions with proto_form == 2.0 per cell
    other_forms: tuple  # sorted (k, form name) with proto_form != 2.0
    need_probe: bool    # probes of the next cell == 0
    commands: tuple     # tuples of argv after "python"
    # S057 (phase 1): the focus cell and its progress; all stay default when no cell is given
    focus: tuple | None = None
    focus_sessions: int = 0   # form-session sessions in the focus cell
    focus_exams: int = 0      # of those, exams (store.exam_flag(k))
    outside_focus: int = 0    # form-session sessions in another cell


def _cell_counts(person, source, data_root, root, want=None) -> tuple[list, list]:
    """Count per cell (cell read from the meta). want(fp) -> True counts the session; else it goes to other."""
    counts, other = [0] * len(recorder.CELLS), []
    for k in store.list_sessions(person, source, data_root=data_root, root=root):
        meta = store.read_session(person, k, source, data_root=data_root, root=root).meta
        if want is None or want(meta.fingerprint):
            counts[recorder.CELLS.index((meta.light, meta.distance))] += 1
        else:
            other.append((k, FORM_NAMES.get(meta.fingerprint.get("proto_form"), "unknown")))
    return counts, sorted(other)


def _focus_counts(person, source, data_root, root, cell) -> tuple[int, int, int]:
    """(in cell, exams in cell, outside cell) over the form-session sessions."""
    n = e = m = 0
    for k in store.list_sessions(person, source, data_root=data_root, root=root):
        meta = store.read_session(person, k, source, data_root=data_root, root=root).meta
        if meta.fingerprint.get("proto_form") != SESSION_FORM:
            continue
        if (meta.light, meta.distance) == tuple(cell):
            n, e = n + 1, e + int(store.exam_flag(k))
        else:
            m += 1
    return n, e, m


def plan_day(person, source, fake=False, data_root=DATA_ROOT, root=ROOT, cell=None) -> DayPlan:
    if fake:
        if person != FAKE_PERSON:
            raise DayError("person", f"--fake always plans the simulated person {FAKE_PERSON}, got {person!r}")
        if source != "sim":
            raise DayError("source", f"--fake uses source sim, got {source!r}")
        probe_person = FAKE_PROBE_PERSON
    else:
        if source != "real":
            raise DayError("source", f"a real person uses source real, got {source!r}")
        if person not in people.list_codes(data_root, root):
            raise DayError("person", f"{person!r} is not in the people file")
        probe_person = person
    sessions, other = _cell_counts(person, source, data_root, root,
                                   want=lambda fp: fp.get("proto_form") == SESSION_FORM)
    probes, _ = _cell_counts(probe_person, source, probe_root(data_root), root)
    k = store.next_free_index(person, source, data_root=data_root, root=root)
    light, dist = recorder.cell_for(k, cell)
    need_probe = probes[recorder.CELLS.index((light, dist))] == 0
    r = str(data_root)
    name = "rotation" if cell is None else f"{cell[0]}/{cell[1]}"
    cell_arg = ("--cell", name) if (cell is not None or not fake) else ()   # real: always given
    if fake:
        cmds = [("opus/tools/envprobe.py", "--fake", "--light", light, "--distance", dist, "--data-root", r),
                ("opus/tools/record.py", "--fake", "--form", "session", *cell_arg, "--data-root", r)]
    else:
        tail = () if Path(data_root) == Path(DATA_ROOT) else ("--data-root", r)
        cmds = [("opus/tools/envprobe.py", "--person", person, "--light", light, "--distance", dist, *tail),
                ("opus/tools/record.py", "--person", person, "--form", "session", *cell_arg,
                 *tail)]
    if not need_probe:
        cmds = cmds[1:]
    fn, fe, fm = (0, 0, 0) if cell is None else _focus_counts(person, source, data_root, root, cell)
    return DayPlan(person=person, source=source, fake=bool(fake), data_root=r, next_k=k, next_light=light,
                   next_distance=dist, next_is_exam=store.exam_flag(k), probes=tuple(probes),
                   sessions=tuple(sessions), other_forms=tuple(other), need_probe=need_probe,
                   commands=tuple(cmds), focus=None if cell is None else tuple(cell), focus_sessions=fn,
                   focus_exams=fe, outside_focus=fm)


def _arg(a: str) -> str:
    return f'"{a}"' if " " in a else a


def format_plan(p: DayPlan) -> list[str]:
    exam = " (exam: never trained on)" if p.next_is_exam else ""
    probe_n = p.probes[recorder.CELLS.index((p.next_light, p.next_distance))]
    out = [f"person {p.person} ({p.source})   next session k = {p.next_k}{exam}",
           f"cell to set up: light {p.next_light}, distance {p.next_distance}"]
    if p.focus is not None:
        out.append(f"phase 1: focus cell {p.focus[0]}/{p.focus[1]} only; other cells wait")
    out += [f"probe for this cell: needed ({probe_n} so far)" if p.need_probe
            else f"probe for this cell: done ({probe_n})",
            f"sessions per cell (form session, need {NEEDED_PER_CELL} each):"]
    for row in range(0, len(CELL_NAMES), 3):
        out.append("  " + "  ".join(f"{CELL_NAMES[i]} {p.sessions[i]}/{NEEDED_PER_CELL}"
                                    for i in range(row, row + 3)))
    total = sum(min(n, NEEDED_PER_CELL) for n in p.sessions)
    need = NEEDED_PER_CELL * len(CELL_NAMES)
    done = all(n >= NEEDED_PER_CELL for n in p.sessions)
    out.append(f"total {total}/{need}" + (f": all {len(CELL_NAMES)} cells have {NEEDED_PER_CELL} sessions"
                                          if done else ""))
    if p.focus is not None:
        reached = (p.focus_sessions >= recorder.FOCUS_TARGET_SESSIONS
                   and p.focus_exams >= recorder.FOCUS_TARGET_EXAMS)
        out.append(f"focus {p.focus[0]}/{p.focus[1]}: {p.focus_sessions}/{recorder.FOCUS_TARGET_SESSIONS} sessions, "
                   f"{p.focus_exams}/{recorder.FOCUS_TARGET_EXAMS} exams"
                   + (": phase 1 target reached" if reached else ""))
        if p.outside_focus > 0:
            out.append(f"note: {p.outside_focus} session(s) are outside the focus cell; "
                       "exams still follow k (5, 10, ...)")
    out += [f"note: k = {k} is form {name}; it used a rotation slot" for k, name in p.other_forms]
    out.append("next command(s):")
    out += ["  python " + " ".join(_arg(a) for a in cmd) for cmd in p.commands]
    return out
