"""Turns a clean sentence into what the speech recognizer might write for a speaker who stutters or slurs.

Every function takes a random.Random, so a build with the same seed gives the same file. Names, numbers and
negations are never touched: the model must learn to keep those as heard, never that they can drift.
"""

from __future__ import annotations

import random
import re

FILLERS = ["um", "uh", "um", "uh", "like", "you know", "I mean", "uhh", "hmm"]
FUNCTION_WORDS = {"the", "a", "an", "to", "is", "am", "are", "my", "for", "of", "at", "will", "do", "does", "be",
                  "it", "that", "some", "your", "can", "please"}
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
                "first", "second", "third", "fifth", "twentieth", "noon", "midnight"}
NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "nor"}
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*|\d[\d:.,]*")

# One sound-based misspelling, as the recognizer writes a word it half heard. (pattern, replacement)
_EDITS = [
    (r"([bcdfgklmnprstz])\1", r"\1"),       # doubled letter -> single (coffee -> cofee)
    (r"ph", "f"), (r"ck", "k"), (r"tion\b", "shun"), (r"ee", "ea"), (r"ea", "ee"),
    (r"er\b", "a"), (r"ou", "ow"), (r"\bc(?=[aou])", "k"), (r"([^aeiou])e\b", r"\1"), (r"y\b", "ee"),
    (r"(?<=[a-z])i(?=[a-z])", "e"), (r"(?<=[a-z])a(?=[a-z])", "e"), (r"th", "d"), (r"ight\b", "ite"),
    (r"(?<=[a-z])s\b", "z"), (r"(?<=[a-z])ed\b", "d"),
]


def words(text: str) -> list[str]:
    return text.split()


def _core(token: str) -> str:
    return re.sub(r"[^\w'’]", "", token)


def protected(token: str, index: int) -> bool:
    """Names, numbers, negations, "I" and contractions with n't are never changed."""
    core = _core(token)
    low = core.lower()
    if not core or any(c.isdigit() for c in core) or low in NUMBER_WORDS or low in NEGATIONS:
        return True
    if low.endswith("n't") or low.endswith("n’t") or low == "i" or "'" in core:
        return True
    return index > 0 and core[0].isupper()


def misspell(word: str, rng: random.Random, blocked: set[str] = frozenset()) -> str | None:
    """A sound-alike misspelling of one word (lower-case letters only), or None when no edit applies."""
    if len(word) < 4 or not word.isalpha() or not word.islower() or word in blocked:
        return None
    edits = list(_EDITS)
    rng.shuffle(edits)
    for pattern, repl in edits:
        new = re.sub(pattern, repl, word, count=1)
        if new != word and len(new) >= 3:
            return new
    return None


def stutter(token: str, rng: random.Random) -> str:
    """"can" -> "c- can", "ca-can" or "can, can"; "I" -> "I, I"."""
    core = _core(token)
    if len(core) <= 2 or rng.random() < 0.4:
        return f"{core}, {token}" if rng.random() < 0.5 else f"{core} {token}"
    cut = core[: rng.choice([1, 2])]
    return f"{cut}- {token}" if rng.random() < 0.5 else f"{cut}-{token}"


def add_stutters(toks: list[str], rng: random.Random, n: int = 1) -> list[str]:
    toks = list(toks)
    for _ in range(n):
        i = rng.randrange(min(len(toks), 4)) if rng.random() < 0.7 else rng.randrange(len(toks))  # mostly early
        toks[i] = stutter(toks[i], rng)
    return toks


def add_fillers(toks: list[str], rng: random.Random, n: int = 1) -> list[str]:
    toks = list(toks)
    for _ in range(n):
        i = rng.randrange(0, max(1, len(toks)))
        toks.insert(i, rng.choice(FILLERS) + ("," if rng.random() < 0.3 else ""))
    return toks


def add_restart(toks: list[str], rng: random.Random) -> list[str]:
    """"Can you call me" -> "Can you, can you call me"."""
    if len(toks) < 4:
        return toks
    k = rng.choice([1, 2, 2, 3])
    head = [_core(t) for t in toks[:k]]
    return head[:-1] + [head[-1] + ","] + [toks[0].lower() if toks[0] != "I" else "I"] + toks[1:]


# Careful noise (big2 on): a dropped first word can turn a question into a statement ("Can you call me" ->
# "you call me"), and "will" or "can" after "my", "a" or "the" is a noun ("my will", "a can of soup"). The
# answer can't be recovered from what's left, so these are never dropped.
QUESTION_STARTS = {"can", "will", "do", "does", "is", "am", "are"}
DETERMINERS = {"my", "a", "an", "the", "your", "his", "her", "our", "their", "that", "this"}


def must_stay(toks: list[str], i: int) -> bool:
    low = _core(toks[i]).lower()
    if i == 0 and low in QUESTION_STARTS:
        return True
    return low in {"will", "can"} and i > 0 and _core(toks[i - 1]).lower() in DETERMINERS


def drop_small_words(toks: list[str], rng: random.Random, p: float = 0.6, careful: bool = False) -> list[str]:
    """Drops small words (as a slurred or broken sentence loses them). careful=False is big1's noise; True keeps
    a question's first word and a noun "will" or "can". The random draws are the same either way."""
    kept = [t for i, t in enumerate(toks) if _core(t).lower() not in FUNCTION_WORDS or rng.random() > p
            or (careful and must_stay(toks, i))]
    return kept if len(kept) >= 2 else toks


def add_misspellings(toks: list[str], rng: random.Random, n: int, blocked: set[str]) -> tuple[list[str], int]:
    """Misspell up to n unprotected content words. Returns the tokens and how many were changed."""
    toks = list(toks)
    spots = [i for i, t in enumerate(toks) if not protected(t, i) and _core(t).lower() not in FUNCTION_WORDS]
    rng.shuffle(spots)
    done = 0
    for i in spots:
        if done >= n:
            break
        core = _core(toks[i])
        new = misspell(core.lower(), rng, blocked)
        if new is None or core != core.lower():
            continue
        toks[i] = toks[i].replace(core, new)
        done += 1
    return toks, done


def recognizer_style(text: str, rng: random.Random) -> str:
    """How the recognizer formats it: as is, without the last full stop, or all lower case without punctuation."""
    r = rng.random()
    if r < 0.45:
        return text
    if r < 0.8:
        return re.sub(r"[.?!]+$", "", text)
    return re.sub(r"[.?!,;]", "", text).lower().replace(" i ", " I ").replace("i'm", "I'm").replace("i'll", "I'll")


def messy(text: str, rng: random.Random, blocked: set[str] = frozenset(), broken: bool = False,
          careful: bool = False) -> tuple[str, int]:
    """One messy hearing of a clean sentence: 1 to 3 kinds of noise. Returns (heard, misspelled word count).
    careful: see drop_small_words."""
    toks = words(text)
    kinds = rng.sample(["stutter", "filler", "restart", "misspell"], k=rng.choice([1, 2, 2, 3]))
    misspelled = 0
    if broken:
        toks = drop_small_words(toks, rng, careful=careful)
    if "misspell" in kinds:
        toks, misspelled = add_misspellings(toks, rng, rng.choice([1, 1, 2]), blocked)
    if "restart" in kinds:
        toks = add_restart(toks, rng)
    if "stutter" in kinds:
        toks = add_stutters(toks, rng, rng.choice([1, 1, 2]))
    if "filler" in kinds:
        toks = add_fillers(toks, rng, rng.choice([1, 1, 2]))
    heard = recognizer_style(" ".join(toks), rng)
    if heard == text:                               # nothing applied (very short sentence): force a stutter
        heard = " ".join(add_stutters(words(text), rng))
    return heard, misspelled
