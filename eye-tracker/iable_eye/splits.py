"""Per-person forward-chaining splits (hard rule 7).

For one person and a test session k, the training sessions are the earlier
sessions j < k that are not exams. An exam (every 5th session) can be a test,
but it is never in a training tuple.

Pure functions on small records: no file is opened, the clock is never read.
"""
from __future__ import annotations

from dataclasses import dataclass

from iable_eye.session_store import (SessionMeta, SessionStoreError, _check_person,
                                     _check_recorded_at, exam_flag)


class SplitError(ValueError):
    """A split request was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True)
class SessionInfo:
    person: str
    session_index: int
    recorded_at: str


@dataclass(frozen=True)
class Split:
    person: str
    test_index: int
    train_indexes: tuple
    test_is_exam: bool


def info_from_meta(meta) -> SessionInfo:
    """The three fields a split needs, from a SessionMeta."""
    if not isinstance(meta, SessionMeta):
        raise SplitError("infos", f"expected a SessionMeta, got {type(meta).__name__}")
    return SessionInfo(meta.person, meta.session_index, meta.recorded_at)


def _is_exam(code: str, k) -> bool:
    """exam_flag, with its error turned into a SplitError."""
    try:
        return exam_flag(k)
    except SessionStoreError as e:
        raise SplitError(code, str(e)) from None


def check_infos(infos) -> dict:
    """Check every record. Returns person -> list of SessionInfo sorted by index."""
    if not isinstance(infos, (list, tuple)) or len(infos) == 0:
        raise SplitError("infos", "infos must be a non-empty list or tuple of SessionInfo")
    by_person: dict = {}
    for info in infos:
        if not isinstance(info, SessionInfo):
            raise SplitError("infos", f"every item must be a SessionInfo, got {type(info).__name__}")
        try:
            _check_person(info.person)
        except SessionStoreError as e:
            raise SplitError("person", str(e)) from None
        _is_exam("index", info.session_index)
        try:
            _check_recorded_at(info.recorded_at)
        except SessionStoreError as e:
            raise SplitError("recorded_at", str(e)) from None
        by_person.setdefault(info.person, []).append(info)
    out = {}
    for person in sorted(by_person):
        rows = sorted(by_person[person], key=lambda r: r.session_index)
        for a, b in zip(rows, rows[1:]):
            if a.session_index == b.session_index:
                raise SplitError("duplicate", f"{person} session {a.session_index} appears twice")
            # recorded_at passed the fixed-width check (YYYY-MM-DDTHH:MM:SS, a real date),
            # so string order is time order.
            if not a.recorded_at < b.recorded_at:
                raise SplitError("time_order",
                                 f"{person}: session {b.session_index} ({b.recorded_at}) is not later "
                                 f"than session {a.session_index} ({a.recorded_at})")
        out[person] = rows
    return out


def forward_splits(infos, min_train=1) -> list:
    """For each person (sorted), each index k (rising): train on j < k that are not exams."""
    if not isinstance(min_train, int) or isinstance(min_train, bool) or min_train < 1:
        raise SplitError("min_train", f"min_train must be an integer >= 1, got {min_train!r}")
    splits = []
    for person, rows in check_infos(infos).items():
        indexes = [r.session_index for r in rows]
        for k in indexes:
            train = tuple(j for j in indexes if j < k and not exam_flag(j))
            if len(train) >= min_train:
                splits.append(Split(person, k, train, exam_flag(k)))
    return splits


def tuning_indexes(infos, person) -> tuple:
    """That person's non-exam indexes, rising (for tuning; exams stay untouched)."""
    groups = check_infos(infos)
    if person not in groups:
        raise SplitError("person", f"person {person!r} is not in the infos")
    return tuple(r.session_index for r in groups[person] if not exam_flag(r.session_index))


def assert_no_leak(splits) -> None:
    """Raise SplitError if any split trains on its test, a later session, or an exam."""
    if not isinstance(splits, (list, tuple)) or not all(isinstance(s, Split) for s in splits):
        raise SplitError("infos", "splits must be a list or tuple of Split")
    for s in splits:
        train = s.train_indexes
        for j in train:
            if j >= s.test_index:
                raise SplitError("leak", f"{s.person} test {s.test_index}: train index {j} is not earlier")
        for j in train:
            if _is_exam("leak", j):
                raise SplitError("exam_in_train", f"{s.person} test {s.test_index}: exam {j} in train")
        for a, b in zip(train, train[1:]):
            if not a < b:
                raise SplitError("leak", f"{s.person} test {s.test_index}: train not strictly rising "
                                         f"({a}, {b})")
