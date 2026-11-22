"""Instruction pages before each part of a recording (S056). Pure: no window, no camera, no time.

Before part i starts, a page says what to do and how long, then "press SPACE to start, q to quit".
While the page is up, `PauseClock` is frozen, so the reading time never enters the stored `t`.
`BlockGate(t)` is used inside `on_frame(obs, t)`: it asks once per part, the first time `t` is inside it.
"""
from __future__ import annotations

import math

SPACE_LINE = "press SPACE to start, q to quit"
PAGE_CUES = {"range": "turn your head as the text says: left, right, up, down"}


class InstructionError(ValueError):
    """`code` is "value" (a bad argument) or "state" (pause/resume in the wrong order)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_pos_real(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and x > 0)


def page_lines(part: int, n_parts: int, cue: str, seconds: float) -> list:
    if not (_is_int(part) and _is_int(n_parts) and 1 <= part <= n_parts):
        raise InstructionError("value", f"need ints 1 <= part <= n_parts, got {part!r}, {n_parts!r}")
    if not isinstance(cue, str) or not cue.strip():
        raise InstructionError("value", f"cue must be a non-empty string, got {cue!r}")
    if not _is_pos_real(seconds):
        raise InstructionError("value", f"seconds must be a finite number > 0, got {seconds!r}")
    return [f"part {part} of {n_parts}", cue, f"takes about {round(seconds)} s", SPACE_LINE]


class PauseClock:
    """clock() = base() - paused_s. While paused it returns the value it had at pause()."""

    def __init__(self, base):
        self._base = base
        self._paused_s = 0.0
        self._pause_at = None     # base() value at pause(), or None when running

    def __call__(self) -> float:
        if self._pause_at is not None:
            return self._pause_at - self._paused_s
        return self._base() - self._paused_s

    def pause(self) -> None:
        if self._pause_at is not None:
            raise InstructionError("state", "pause() while already paused")
        self._pause_at = self._base()

    def resume(self) -> None:
        if self._pause_at is None:
            raise InstructionError("state", "resume() without pause()")
        self._paused_s += self._base() - self._pause_at
        self._pause_at = None

    @property
    def paused_s(self) -> float:
        return self._paused_s


class BlockGate:
    """gate(t) -> bool. Shows the page of part i once (clock paused meanwhile); False when ask said q."""

    def __init__(self, parts, clock: PauseClock, ask):
        parts = [tuple(p) for p in parts]
        prev_end = -math.inf
        for p in parts:
            if len(p) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                      and math.isfinite(v) for v in p[:2]):
                raise InstructionError("value", f"a part is (start_s, end_s, cue), got {p!r}")
            if not (p[0] < p[1]) or p[0] < prev_end:
                raise InstructionError("value", f"parts must be sorted, start < end, no overlap: {p!r}")
            prev_end = p[1]
        self._parts = parts
        self._clock = clock
        self._ask = ask
        self._shown = []

    def __call__(self, t: float) -> bool:
        i = next((k for k, (s0, s1, _) in enumerate(self._parts) if s0 <= t < s1), None)
        if i is None or i in self._shown:
            return True
        self._shown.append(i)
        s0, s1, cue = self._parts[i]
        self._clock.pause()
        try:
            ok = bool(self._ask(page_lines(i + 1, len(self._parts), cue, s1 - s0)))
        finally:
            self._clock.resume()
        return ok

    @property
    def shown(self) -> tuple:
        return tuple(self._shown)


def parts_from_timeline(tl) -> list:
    return [(b.start_s, b.end_s, PAGE_CUES.get(b.name, b.cue)) for b in tl.blocks]


def parts_from_probe(blocks) -> list:
    return [(s0, s1, cue) for s0, s1, _name, cue in blocks]
