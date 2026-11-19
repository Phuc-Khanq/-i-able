"""The user's own fixes: words Whisper keeps getting wrong for this voice, and what they should be ("waiter" -> "water").

Route: voice
Publishes: nothing (the mode manager records fixes when a fixed card is typed, and applies them to new dictation)
APIs: pathlib

Wispr Flow's idea ("a product that never makes the same mistake twice"), for our users: a voice that a speech
recogniser keeps mishearing the same way (dysarthria, an accent) gets that mishearing fixed by hand once or twice,
then automatically. When the user fixes a word on the card by voice ("change waiter to water") and then types the
card, the fix is counted. After MIN_TIMES of the same fix, new dictation gets it before it shows on the card, where
the user still sees it and can change it back. Changing it back ("change water to waiter") forgets the fix.

The file lives in data/voice/personal/, so "delete my voice data" removes it with the word list and the recordings.
It holds only word pairs the user made: no audio, no transcripts.

    python -m engine.voice.personal_fixes                      # show the fixes
    python -m engine.voice.personal_fixes forget waiter        # forget one
"""
from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXES_FILE = REPO_ROOT / "data" / "voice" / "personal" / "fixes.txt"
MIN_TIMES = 2             # the same fix this many times, then it's made automatically
MAX_FIXES = 300
MAX_WORDS = 4             # a fix covers at most this many words on each side

_HEADER = ("# Your fixes: what the speech recogniser heard, a tab, what you meant, a tab, how many times you fixed it.\n"
           "# Made automatically once you've fixed the same words twice. \"Delete my voice data\" deletes this file.\n")


def key(text: str) -> tuple[str, ...]:
    """Words to compare: lowercase, no punctuation ("Waiter," -> ("waiter",))."""
    return tuple(re.sub(r"[^\w\s']", " ", str(text).lower().replace("’", "'")).split())


@dataclass
class Fix:
    heard: str
    meant: str
    times: int = 1


class PersonalFixes:
    """The fix list on disk. Thread safe; reads are cached until the file changes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else FIXES_FILE
        self._lock = threading.Lock()
        self._cache: list[Fix] = []
        self._stamp: tuple[float, int] | None = None

    def fixes(self) -> list[Fix]:
        """All fixes, oldest first. [] if there is no file or it can't be read (a broken list never stops dictation)."""
        with self._lock:
            try:
                st = self.path.stat()
            except OSError:
                self._cache, self._stamp = [], None
                return []
            stamp = (st.st_mtime, st.st_size)
            if stamp != self._stamp:
                out: dict[tuple[str, ...], Fix] = {}
                try:
                    lines = self.path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    lines = []
                for line in lines:
                    if line.lstrip().startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    heard, meant = " ".join(parts[0].split()), " ".join(parts[1].split())
                    try:
                        times = max(1, int(parts[2])) if len(parts) > 2 else 1
                    except ValueError:
                        times = 1
                    if _ok(heard, meant):
                        out[key(heard)] = Fix(heard, meant, times)
                self._cache, self._stamp = list(out.values())[-MAX_FIXES:], stamp
            return [Fix(f.heard, f.meant, f.times) for f in self._cache]

    def record(self, heard: str, meant: str) -> Fix | None:
        """The user changed `heard` to `meant` and typed it: count it. Changing a fix back forgets it. Returns the
        fix as saved, or None if it was forgotten or isn't a usable pair."""
        heard, meant = " ".join(str(heard).split()), " ".join(str(meant).split())
        if not _ok(heard, meant):
            return None
        fixes = self.fixes()
        back = [f for f in fixes if key(f.heard) == key(meant) and key(f.meant) == key(heard)]
        if back:
            self._write([f for f in fixes if f not in back])
            return None
        old = next((f for f in fixes if key(f.heard) == key(heard)), None)
        if old is not None and key(old.meant) == key(meant):
            new = Fix(heard, meant, old.times + 1)
        else:
            new = Fix(heard, meant, 1)                 # a different fix for the same words starts again
        self._write([f for f in fixes if key(f.heard) != key(heard)] + [new])
        return new

    def forget(self, heard: str) -> bool:
        fixes = self.fixes()
        kept = [f for f in fixes if key(f.heard) != key(heard)]
        if len(kept) == len(fixes):
            return False
        self._write(kept)
        return True

    def apply(self, words: list[dict]) -> tuple[list[dict], list[Fix]]:
        """The card's words ({"text", "conf"}) with every learned fix made (fixes made MIN_TIMES or more), and the
        fixes used. Longer fixes first; each word is changed at most once. Punctuation after the last word and a
        sentence's capital letter stay; the new words get conf 1.0."""
        ready = sorted((f for f in self.fixes() if f.times >= MIN_TIMES), key=lambda f: -len(key(f.heard)))
        if not ready or not words:
            return [dict(w) for w in words], []
        out, used, i = [], [], 0
        keys = [key(w["text"]) for w in words]
        while i < len(words):
            hit = None
            for f in ready:
                k = key(f.heard)
                span = _span(keys, i, len(k))
                if span and tuple(x for ks in keys[i:span] for x in ks) == k:
                    hit = (f, span)
                    break
            if hit is None:
                out.append(dict(words[i]))
                i += 1
                continue
            f, end = hit
            first, last = words[i]["text"], words[end - 1]["text"]
            meant = f.meant
            if first[:1].isupper() and meant[:1].islower():
                meant = meant[0].upper() + meant[1:]
            tail = re.search(r"[^\w']+$", last)
            if tail and not re.search(r"[^\w'\s]$", meant):
                meant += tail.group(0)
            out.extend({"text": t, "conf": 1.0} for t in meant.split())
            used.append(f)
            i = end
        return out, used

    def _write(self, fixes: list[Fix]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        body = "".join(f"{f.heard}\t{f.meant}\t{f.times}\n" for f in fixes[-MAX_FIXES:])
        tmp.write_text(_HEADER + body, encoding="utf-8")
        tmp.replace(self.path)                        # never a half-written list
        with self._lock:
            self._stamp = None


def _ok(heard: str, meant: str) -> bool:
    kh, km = key(heard), key(meant)
    return bool(kh) and bool(km) and kh != km and len(kh) <= MAX_WORDS and len(km) <= MAX_WORDS


def _span(keys: list[tuple[str, ...]], start: int, n: int) -> int | None:
    """The end index of the fewest card words from `start` that hold n compared words, or None."""
    count, j = 0, start
    while j < len(keys) and count < n:
        count += len(keys[j])
        j += 1
    return j if count == n else None


def main(argv: list[str]) -> int:
    pf = PersonalFixes()
    if len(argv) >= 2 and argv[0] == "forget":
        print("Forgotten." if pf.forget(" ".join(argv[1:])) else "No such fix.")
        return 0
    fixes = pf.fixes()
    print(f"{len(fixes)} fixes in {pf.path} (made automatically from {MIN_TIMES} times)")
    for f in fixes:
        print(f"  {f.heard} -> {f.meant}  ({f.times}x{', automatic' if f.times >= MIN_TIMES else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
