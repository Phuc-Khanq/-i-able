"""The user's own word list: names and terms Whisper should lean toward (Hathaway, prednisone, AFO).

Route: voice
Publishes: nothing (the mode manager adds and removes words; Dictation reads them before each utterance)
APIs: pathlib

Words are added by voice ("add Hathaway to my words", "add that to my words" after spelling one) or by editing the
file: one word or phrase per line, lines starting with # are notes. The file lives in data/voice/personal/, so
"delete my voice data" removes it with the recordings. It holds only words the user chose: no audio, no transcripts.

Whisper gets the list as `hotwords` (faster-whisper puts it in front of each utterance as earlier text, cut at about
220 tokens), so only the first MAX_HINT_CHARS characters are used, newest words first.

    python -m engine.voice.personal_words                  # show the list
    python -m engine.voice.personal_words add Hathaway     # add a word
    python -m engine.voice.personal_words remove Hathaway  # remove one
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORDS_FILE = REPO_ROOT / "data" / "voice" / "personal" / "words.txt"
MAX_WORDS = 200          # the file can hold this many...
MAX_WORD_CHARS = 40      # ...each at most this long
MAX_HINT_CHARS = 600     # ...and Whisper gets the newest that fit in this (about 150 tokens)

_HEADER = ("# Your words: names and terms the speech recogniser should lean toward. One per line, newest last.\n"
           "# Say \"add <word> to my words\" to add one. \"Delete my voice data\" deletes this file.\n")


def _clean(word: str) -> str:
    return " ".join(str(word).replace("’", "'").strip(" \t.,!?;:\"").split())[:MAX_WORD_CHARS]


class PersonalWords:
    """The word list on disk. Safe to use from the audio and camera threads at once; reads are cached until the
    file changes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else WORDS_FILE
        self._lock = threading.Lock()
        self._cache: list[str] = []
        self._stamp: tuple[float, int] | None = None

    def words(self) -> list[str]:
        """The words, oldest first. [] if there is no file or it can't be read (a broken list never stops
        dictation)."""
        with self._lock:
            try:
                st = self.path.stat()
            except OSError:
                self._cache, self._stamp = [], None
                return []
            stamp = (st.st_mtime, st.st_size)
            if stamp != self._stamp:
                try:
                    lines = self.path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    lines = []
                seen, out = set(), []
                for line in lines:
                    w = _clean(line) if not line.lstrip().startswith("#") else ""
                    if w and w.lower() not in seen:
                        seen.add(w.lower())
                        out.append(w)
                self._cache, self._stamp = out[-MAX_WORDS:], stamp
            return list(self._cache)

    def hint(self) -> str:
        """The words for Whisper's hotwords, newest first, as long as they fit in MAX_HINT_CHARS: "" if none."""
        out, size = [], 0
        for w in reversed(self.words()):
            if size + len(w) + 2 > MAX_HINT_CHARS:
                break
            out.append(w)
            size += len(w) + 2
        return ", ".join(out)

    def add(self, word: str) -> str | None:
        """Add a word (moved to the end if it's already there). The word as saved, or None if it's empty."""
        w = _clean(word)
        if not w:
            return None
        words = [x for x in self.words() if x.lower() != w.lower()] + [w]
        self._write(words[-MAX_WORDS:])
        return w

    def touch(self, text: str) -> list[str]:
        """The user just typed `text`: words of the list that are in it move to the newest end, so the ones in use
        always fit in Whisper's hint (Wispr Flow ranks its dictionary by use). Returns the words moved; the file is
        only written when the order changes."""
        words = self.words()
        if not words:
            return []
        low = " " + " ".join(str(text).lower().replace("’", "'").split()) + " "
        low = "".join(ch if ch.isalnum() or ch in " '" else " " for ch in low)
        used = [w for w in words if f" {w.lower()} " in low]
        if not used:
            return []
        new = [w for w in words if w not in used] + used
        if new != words:
            self._write(new)
        return used

    def remove(self, word: str) -> bool:
        """Remove a word (any capitals). False if it wasn't on the list."""
        w = _clean(word).lower()
        words = self.words()
        kept = [x for x in words if x.lower() != w]
        if len(kept) == len(words):
            return False
        self._write(kept)
        return True

    def _write(self, words: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(_HEADER + "".join(f"{w}\n" for w in words), encoding="utf-8")
        tmp.replace(self.path)                        # never a half-written list
        with self._lock:
            self._stamp = None                        # read it again next time


# Words never worth a place on the list: Whisper knows them, and they are the usual "their/there" kind of fix.
_NOT_LEARNED = set("""
a an the and or but if so as at by for from in into of off on onto out over to too two up with about after before
i i'm i'll i've i'd me my we our you your you're yours he him his she her they them their they're there here hear
it it's its this that these those what which witch who whom whose where were we're when why how than then
is are was be been am do does did done have has had will would wood can could should shall may might must
no know now new knew not right write rite one won four fore see sea buy bye yes ok okay mr mrs ms dr
""".split())


def worth_learning(word: str, spelled: bool = False) -> bool:
    """Should a word the user put on the card by hand join their list on its own (Wispr Flow's auto-add)?
    A spelled word, or a name, acronym or term (a capital letter or a digit in it), unless it is a common word."""
    w = _clean(word)
    if len("".join(ch for ch in w if ch.isalnum())) < 2 or w.lower() in _NOT_LEARNED:
        return False
    return spelled or any(ch.isupper() or ch.isdigit() for ch in w)


def combine(*hints: str) -> str | None:
    """Join hotword strings, skipping empty ones: None if nothing is left (Whisper's "no hotwords")."""
    joined = ", ".join(h.strip() for h in hints if h and h.strip())
    return joined or None


def main(argv: list[str]) -> int:
    pw = PersonalWords()
    if len(argv) >= 2 and argv[0] in ("add", "remove"):
        word = " ".join(argv[1:])
        if argv[0] == "add":
            print(f"Added: {pw.add(word)}")
        else:
            print("Removed." if pw.remove(word) else "Not on the list.")
        return 0
    words = pw.words()
    print(f"{len(words)} words in {pw.path}")
    for w in words:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
