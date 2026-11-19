"""The user's snippets: a short spoken cue for text they would otherwise dictate again and again (Wispr Flow's snippets).

Route: voice
Publishes: nothing (the mode manager saves snippets and puts them on the card)
APIs: pathlib

For someone who can't type, an address, an email, a phone number or a sign-off is slow to dictate and easy to get
misheard. Made by voice: dictate the text, fix it on the card, then say "save this as my address". After that,
"insert my address" (or "my address" said alone while dictating) puts the saved text on the card, exactly as saved;
"yes" types it as usual. "Delete the snippet my address" removes it.

The file lives in data/voice/personal/, so "delete my voice data" removes it with the word list and the fixes. It
holds only text the user chose to save. One snippet per line: the cue, a tab, the text.

    python -m engine.voice.snippets                 # show the snippets
"""
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNIPPETS_FILE = REPO_ROOT / "data" / "voice" / "personal" / "snippets.txt"
MAX_SNIPPETS = 100
MAX_CUE_WORDS = 5
MAX_TEXT_CHARS = 1000

_HEADER = ("# Your snippets: a cue, a tab, the text it puts on the card. Say \"insert <cue>\".\n"
           "# \"Delete my voice data\" deletes this file.\n")


def cue_key(text: str) -> str:
    """A cue for comparing: lowercase words, no punctuation ("My address." -> "my address")."""
    return " ".join(re.sub(r"[^\w\s']", " ", str(text).lower().replace("’", "'")).split())


class Snippets:
    """The snippet list on disk. Thread safe; reads are cached until the file changes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else SNIPPETS_FILE
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[str, str]] = {}
        self._stamp: tuple[float, int] | None = None

    def all(self) -> dict[str, tuple[str, str]]:
        """{cue key: (cue as said, text)}. {} if there is no file or it can't be read."""
        with self._lock:
            try:
                st = self.path.stat()
            except OSError:
                self._cache, self._stamp = {}, None
                return {}
            stamp = (st.st_mtime, st.st_size)
            if stamp != self._stamp:
                out: dict[str, tuple[str, str]] = {}
                try:
                    lines = self.path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    lines = []
                for line in lines:
                    if line.lstrip().startswith("#") or "\t" not in line:
                        continue
                    cue, text = line.split("\t", 1)
                    k, text = cue_key(cue), " ".join(text.split())
                    if k and text and len(k.split()) <= MAX_CUE_WORDS:
                        out[k] = (" ".join(cue.split()), text[:MAX_TEXT_CHARS])
                self._cache, self._stamp = out, stamp
            return dict(self._cache)

    def get(self, cue: str) -> str | None:
        hit = self.all().get(cue_key(cue))
        return hit[1] if hit else None

    def save(self, cue: str, text: str) -> str | None:
        """Save (or replace) a snippet. The cue as saved, or None if the cue or text is unusable."""
        k, text = cue_key(cue), " ".join(str(text).split())[:MAX_TEXT_CHARS]
        if not k or not text or len(k.split()) > MAX_CUE_WORDS:
            return None
        items = {key: v for key, v in self.all().items() if key != k}
        items[k] = (k, text)
        self._write(list(items.values())[-MAX_SNIPPETS:])
        return k

    def remove(self, cue: str) -> bool:
        items = self.all()
        if cue_key(cue) not in items:
            return False
        del items[cue_key(cue)]
        self._write(list(items.values()))
        return True

    def _write(self, items: list[tuple[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(_HEADER + "".join(f"{cue}\t{text}\n" for cue, text in items), encoding="utf-8")
        tmp.replace(self.path)                        # never a half-written list
        with self._lock:
            self._stamp = None


def main(argv: list[str]) -> int:
    items = Snippets().all()
    print(f"{len(items)} snippets in {SNIPPETS_FILE}")
    for cue, text in items.values():
        print(f"  {cue}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
