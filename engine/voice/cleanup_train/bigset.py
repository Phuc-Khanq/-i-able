"""Builds the big clean-up data set from the field banks (fields.py): 7,500 training examples and a 2,500-case
test set, split by sentence so no test sentence is ever trained on.

    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.bigset --name big1

Writes (git-ignored) data/cleanup_train/<name>_train.jsonl (one chat per line, like build.py),
<name>_train_preview.csv, <name>_test.csv (the eval's test-set layout; score it with
`python -m eval.dictation_cleanup.run --cases data/cleanup_train/<name>_test.csv`) and <name>_info.json.

The split:
- A field sentence goes to train or test by a hash of its text. A sentence with one of its field's terms goes
  where that term goes, so a misheard term in the test set is never trained on.
- The new pair and change-of-mind templates go by frame: a test frame is never trained on.
- The older seeds (seeds.py, seeds_extra.py, templates.py) are training only: full1 was trained on them.
- Last, a training example whose heard or meant text overlaps a test text (token overlap 0.6 or more) is dropped,
  and build.py's leak check against eval/dictation_cleanup/testset.csv still applies to training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from engine.voice.cleanup_train import build, noise, templates
from engine.voice.cleanup_train.fields import Field, all_fields

OUT = build.OUT
TEST_SHARE = 0.30                  # of sentences and frames; the test set is then sampled down to TEST_SIZE
TRAIN_SIZE, TEST_SIZE = 7500, 2500
MIX = {"clean_control": 0.15, "ambiguous": 0.15, "mispronounced": 0.25, "broken_structure": 0.30, "mixed": 0.15}
TEST_COLUMNS = ["id", "split", "category", "app", "context", "asr_best", "asr_alt1", "asr_alt2", "personal_words",
                "intended", "must_keep", "expected_behavior", "notes"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SMALL_NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "twenty", "thirty", "forty", "fifty"]
TEMPLATE_APPS = ["chat", "email", "notes", "Messages"]

# New template frames, not in templates.py (full1 was trained on those). Split by frame.
NEGATION_FRAMES = [("We will {v}", "We won't {v}"), ("She can {v}", "She can't {v}"),
                   ("He should {v}", "He shouldn't {v}"), ("They could {v}", "They couldn't {v}"),
                   ("I would {v}", "I wouldn't {v}"), ("You should {v}", "You shouldn't {v}"),
                   ("We do need to {v}", "We don't need to {v}"), ("I'm going to {v}", "I'm not going to {v}"),
                   ("She will {v}", "She won't {v}"), ("They can {v}", "They can't {v}"),
                   ("I did want to {v}", "I didn't want to {v}"), ("We have to {v}", "We don't have to {v}"),
                   ("He can {v}", "He can't {v}"), ("I'll {v}", "I won't {v}"),
                   ("You can {v} today", "You can't {v} today"), ("We should {v} this week", "We shouldn't {v} this week")]
VERB_PHRASES = ["sign the lease", "bring the wheelchair", "pick up the prescription", "call the insurance company",
                "move the appointment", "come to the meeting", "pay the invoice", "drive to the clinic",
                "cancel the order", "fix the ramp", "send the report", "visit Grandma", "book the flight",
                "open the package", "take the train", "renew the permit", "join the video call", "use the lift",
                "eat the leftovers", "water the garden", "go to the game", "update the app", "wait for the van",
                "post the photos", "finish the homework", "return the shoes", "charge the battery", "share the file",
                "stay for dinner", "walk the dog"]
NUMBER_FRAMES = ["The room number is {n}", "We have {n} people on the list", "The bill came to {n} dollars",
                 "Turn to page {n}", "My grandson is {n}", "The bus leaves in {n} minutes", "We need {n} copies",
                 "The meeting is on floor {n}", "It weighs {n} pounds", "Order {n} name tags",
                 "There are {n} steps to the front door", "The class has {n} students",
                 "Set the oven timer for {n} minutes", "The ticket costs {n} dollars", "Gate {n} is near the elevator",
                 "I've lived here for {n} years"]
NAME_PAIRS = [("Dan", "Ann"), ("Mark", "Mike"), ("Jon", "Joan"), ("Lynn", "Glen"), ("Pam", "Pat"), ("Ben", "Jen"),
              ("Kim", "Tim"), ("Rose", "Ross"), ("Eric", "Erin"), ("Carla", "Karl"), ("Nina", "Gina"), ("Leo", "Theo"),
              ("Maya", "Mia"), ("Ray", "Jay"), ("Beth", "Bess"), ("Joel", "Noel")]
NAME_FRAMES = ["Tell {a} I'm on my way", "Can {a} pick me up after work?", "Send the photos to {a}",
               "{a} is coming to dinner on Friday", "Ask {a} if the store is open", "I'll call {a} after lunch",
               "Please give the keys to {a}", "Text {a} the address", "Invite {a} to the party",
               "{a} said the van is late", "Remind {a} about the meeting", "Let {a} know I'm home"]
_TIMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "noon", "six thirty", "seven thirty"]
CORRECTION_FRAMES = [
    ("Book the table for {a}, {m}, {b} people", "Book the table for {b} people.", ["two", "three", "four", "five", "six"]),
    ("The meeting is in room {a}, {m}, room {b}", "The meeting is in room {b}.", ["four", "six", "nine", "twelve"]),
    ("I'll be there on {a}, {m}, on {b}", "I'll be there on {b}.", DAYS),
    ("Send it to {a}, {m}, to {b}", "Send it to {b}.", ["Mom", "Dad", "my aunt", "the office", "my teacher", "the pharmacy"]),
    ("We need {a}, {m}, {b} more chairs", "We need {b} more chairs.", ["two", "three", "four", "five", "ten"]),
    ("Turn {a}, {m}, turn {b} at the light", "Turn {b} at the light.", ["left", "right"]),
    ("The game starts at {a}, {m}, at {b}", "The game starts at {b}.", _TIMES),
    ("Put the {a}, {m}, the {b} on the list", "Put the {b} on the list.", ["milk", "eggs", "bread", "rice", "apples", "coffee"]),
    ("Meet us at the {a}, {m}, the {b} entrance", "Meet us at the {b} entrance.", ["north", "south", "east", "west", "main", "side"]),
    ("Her birthday is in {a}, {m}, in {b}", "Her birthday is in {b}.", ["March", "April", "May", "June", "July", "August"]),
    ("The appointment is with Dr. {a}, {m}, Dr. {b}", "The appointment is with Dr. {b}.", ["Lee", "Patel", "Garcia", "Okafor", "Nguyen", "Smith"]),
    ("Set an alarm for {a}, {m}, for {b}", "Set an alarm for {b}.", _TIMES),
    ("We're leaving in {a}, {m}, {b} minutes", "We're leaving in {b} minutes.", ["five", "ten", "fifteen", "twenty", "thirty"]),
    ("I'll take the {a}, {m}, the {b} train", "I'll take the {b} train.", ["early", "late", "eight o'clock", "noon"]),
]


# ------------------------------------------------------------------ helpers
def side(key: str, share: float = TEST_SHARE) -> str:
    """"test" or "train", from a stable hash of the key (the same on every run and machine)."""
    h = int(hashlib.sha1(key.lower().encode("utf-8")).hexdigest()[:8], 16) / 2**32
    return "test" if h < share else "train"


def _term_in(term: str, text: str) -> re.Match | None:
    return re.search(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", text, re.IGNORECASE)


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def mishearings(term: str, forms: list[str]) -> list[str]:
    """The real mishearings: a form that only changes spacing or hyphens ("car pool") isn't one."""
    return [f for f in forms if _letters(f) != _letters(term)]


def keep_items(text: str, extra: list[str] = ()) -> str:
    """must_keep for a test case: names (capitalized, not starting a sentence), number words and digits, plus
    `extra` (a fixed term). Joined with "|" as in testset.csv."""
    items, start = list(extra), True
    for tok in text.split():
        core = re.sub(r"^[^\w']+|[^\w']+$", "", tok)
        low = core.lower()
        if core and (low in noise.NUMBER_WORDS or any(c.isdigit() for c in core)):
            items.append(core)
        elif core and not start and core[0].isupper() and low not in {"i", "i'm", "i'll", "i've", "i'd", "dr", "mr", "mrs", "ms"}:
            items.append(core)
        start = bool(re.search(r"[.!?]$", tok)) and low not in {"dr", "mr", "mrs", "ms"}
    seen: list[str] = []
    for i in items:
        if i.lower() not in (s.lower() for s in seen):
            seen.append(i)
    return "|".join(seen)


def example(kind: str, category: str, expected: str, app: str, before: str, heard: str, target: dict,
            other: str = "", words: list[str] = (), field: str = "general", where: str = "train", keep: str = "",
            intended: str | None = None) -> dict:
    c = build.case(kind, app, before, heard, target, other, list(words))
    c.update({"category": category, "expected": expected, "field": field, "side": where, "must_keep": keep,
              "intended": intended if intended is not None else target["text"]})
    return c


def _with(text: str, old: re.Match, new: str) -> str:
    return text[:old.start()] + new + text[old.end():]


# ------------------------------------------------------------------ field examples
def sentence_side(f: Field, text: str) -> str:
    for term in f.terms:
        if _term_in(term, text):
            return side(f"term:{term}")             # not per field: "potluck" is in two fields, one side
    return side(f"sentence:{text}")


# Words a finished sentence doesn't end on: a cut after one of them is clearly cut off.
DANGLING = {"the", "a", "an", "to", "for", "with", "and", "or", "but", "my", "your", "our", "his", "her", "their",
            "at", "of", "on", "in", "from", "about", "that", "is", "are", "was", "can", "could", "will", "would",
            "should", "if", "because", "when", "so", "this", "some", "into", "before", "after", "than", "by"}


def cut_off(text: str, rng: random.Random) -> str | None:
    """The sentence stopped after a word it can't end on ("Can you send the"), or None if there's no such spot."""
    words_ = text.split()
    spots = [k for k in range(3, len(words_) - 1) if re.sub(r"[^\w']", "", words_[k - 1]).lower() in DANGLING
             and not re.search(r"[,.;!?]$", words_[k - 1])]
    return " ".join(words_[: rng.choice(spots)]) if spots else None


def field_examples(f: Field, rng: random.Random, blocked: set[str]) -> list[dict]:
    out: list[dict] = []
    by_side: dict[str, list[str]] = defaultdict(list)
    for text in f.sentences:
        where = sentence_side(f, text)
        by_side[where].append(text)
        app = rng.choice(f.apps)

        def before() -> str:
            return rng.choice(f.befores) if rng.random() < 0.6 else ""

        keep = keep_items(text)
        out.append(example("clean", "clean_control", "unchanged", app, before(), noise.recognizer_style(text, rng),
                           build.answer(text), field=f.name, where=where))
        for broken in (False, False, True) + ((False,) if where == "train" else ()):   # one more noisy copy to learn from
            heard, miss = noise.messy(text, rng, blocked, broken=broken)
            cat = ("mixed" if miss else "broken_structure") if broken else ("mispronounced" if miss else "broken_structure")
            other = re.sub(r"[.?!]+$", "", text) if miss and rng.random() < 0.3 else ""
            out.append(example("broken" if broken else "messy", cat, "rewrite", app, before(), heard, build.answer(text),
                               other, field=f.name, where=where, keep=keep))
        out += term_fixes(f, text, rng, where, app)
        out += corrections_in(f, text, rng, where, app, before())
        cut = cut_off(text, rng) if rng.random() < 0.6 else None
        if cut:
            out.append(example("cut_off", "ambiguous", "keep_original_and_flag", app, before(), cut,
                               build.answer(cut, [], True), field=f.name, where=where))
    for heard, other in f.ambiguous:
        where = side(f"ambiguous:{heard}")
        out.append(example("ambiguous", "ambiguous", "show_both", rng.choice(f.apps), "", heard,
                           build.answer(build._sentence(heard), [build._sentence(other)], True), other, field=f.name,
                           where=where, intended=f"{build._sentence(heard)} || {build._sentence(other)}"))
    for where, texts in by_side.items():
        out += passages(f, texts, rng, where, blocked)
    return out


def term_fixes(f: Field, text: str, rng: random.Random, where: str, app: str) -> list[dict]:
    """The field term misheard: the right word comes from the word list, the line before, another hearing, or
    only from the sentence itself."""
    out = []
    for term, forms in f.terms.items():
        m = _term_in(term, text)
        if not m:
            continue
        for form in mishearings(term, forms):
            heard = _with(text, m, form)
            other, words_, before = "", [], ""
            how = rng.choices(["words", "before", "other", "sentence"], [0.3, 0.3, 0.2, 0.2])[0]
            if how == "words":
                words_ = [term] + rng.sample([t for t in f.terms if t != term], k=min(2, len(f.terms) - 1))
                rng.shuffle(words_)
            elif how == "before":
                before = rng.choice(f.befores)
            elif how == "other":
                other = re.sub(r"[.?!]+$", "", text)
            cat = "mispronounced"
            if rng.random() < 0.35:
                heard = " ".join(noise.add_fillers(noise.add_stutters(heard.split(), rng), rng))
                cat = "mixed"
            heard = noise.recognizer_style(heard, rng)
            out.append(example(f"term_fix:{how}", cat, "rewrite", app, before, heard, build.answer(text), other,
                               words_, f.name, where, keep_items(text, [term])))
        break                                       # the first term only: its split decided the sentence's
    return out


def corrections_in(f: Field, text: str, rng: random.Random, where: str, app: str, before: str) -> list[dict]:
    """A change of mind on a day or a number already in the sentence: "on Monday, no wait, Tuesday"."""
    for pool in (DAYS, SMALL_NUMBERS):
        m = re.search(r"\b(" + "|".join(pool) + r")\b", text)
        if m:
            wrong = rng.choice([w for w in pool if w.lower() != m.group(1).lower()])
            heard = _with(text, m, f"{wrong}, {rng.choice(templates.MARKERS)} {m.group(1)}")
            heard = noise.recognizer_style(heard, rng)
            return [example("correction", "broken_structure", "rewrite", app, before, heard, build.answer(text),
                            field=f.name, where=where, keep=keep_items(text, [m.group(1)]))]
    return []


def passages(f: Field, texts: list[str], rng: random.Random, where: str, blocked: set[str]) -> list[dict]:
    """Two or three sentences of one field dictated in one go, each with its own noise."""
    texts = list(texts)
    rng.shuffle(texts)
    out, i = [], 0
    while i + 1 < len(texts):
        group = texts[i: i + rng.choice([2, 3])]
        i += len(group)
        parts, misses = [], 0
        for t in group:
            if rng.random() < 0.7:
                heard, miss = noise.messy(t, rng, blocked)
                misses += miss
            else:
                heard = noise.recognizer_style(t, rng)
            parts.append(heard)
        meant = " ".join(group)
        out.append(example("passage", "mixed" if misses else "broken_structure", "rewrite", rng.choice(f.apps),
                           rng.choice(f.befores) if rng.random() < 0.5 else "", " ".join(parts),
                           build.answer(meant), field=f.name, where=where, keep=keep_items(meant)))
    return out


# ------------------------------------------------------------------ template examples
def template_examples(rng: random.Random) -> list[dict]:
    out = []

    def pair(kind, a, b, where, words_=()):
        heard, other = (a, b) if rng.random() < 0.5 else (b, a)
        h, o = build._sentence(heard), build._sentence(other)
        return example(kind, "ambiguous", "show_both", rng.choice(TEMPLATE_APPS), "", heard,
                       build.answer(h, [o], True), other, list(words_), where=where, intended=f"{h} || {o}")

    for pos, neg in NEGATION_FRAMES:
        where = side(f"negation:{pos}")
        for v in rng.sample(VERB_PHRASES, 14):
            out.append(pair("negation_pair", pos.format(v=v), neg.format(v=v), where))
    for frame in NUMBER_FRAMES:
        where = side(f"number:{frame}")
        for a, b in templates.TEEN_TY:
            out.append(pair("number_pair", frame.format(n=a), frame.format(n=b), where))
    for frame in NAME_FRAMES:
        where = side(f"name:{frame}")
        for a, b in rng.sample(NAME_PAIRS, 10):
            out.append(pair("name_pair", frame.format(a=a), frame.format(a=b), where, [a, b]))
    for heard_f, answer_f, choices in CORRECTION_FRAMES:
        where = side(f"correction:{heard_f}")
        for _ in range(10):
            a, b = rng.sample(choices, 2)
            heard = noise.recognizer_style(heard_f.format(a=a, b=b, m=rng.choice(templates.MARKERS)), rng)
            meant = answer_f.format(b=b)
            out.append(example("template_correction", "broken_structure", "rewrite", rng.choice(TEMPLATE_APPS), "",
                               heard, build.answer(meant), where=where, keep=keep_items(meant, [b])))
    return out


# ------------------------------------------------------------------ split, sample, write
def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9']+", text.lower()))


def drop_close(train: list[dict], test: list[dict], similar: float = build.SIMILAR) -> tuple[list[dict], int]:
    """Training examples whose heard or meant text overlaps a test text by `similar` or more (token overlap)."""
    test_sets = list({_tokens(t) for c in test for t in [c["asr_best"], c["target"]["text"],
                                                         *c["target"]["alternatives"]] if t.strip()})
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(test_sets):
        for w in s:
            index[w].append(i)
    kept, dropped = [], 0
    for c in train:
        close = False
        for t in (c["asr_best"], c["target"]["text"]):
            s = _tokens(t)
            if not s:
                continue
            shared = Counter(i for w in s for i in index.get(w, ()))
            if any(n / len(s | test_sets[i]) >= similar for i, n in shared.items() if n >= similar * len(s)):
                close = True
                break
        if close:
            dropped += 1
        else:
            kept.append(c)
    return kept, dropped


def sample(pool: list[dict], size: int, rng: random.Random) -> list[dict]:
    """`size` examples in the MIX proportions; within a category, kinds take turns so no kind crowds out the rest.
    A category short of its share gives the rest to the others."""
    by_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in pool:
        by_cat[c["category"]][c["kind"].split(":")[0]].append(c)
    for kinds in by_cat.values():
        for lst in kinds.values():
            rng.shuffle(lst)
    chosen: list[dict] = []
    want = {cat: round(size * share) for cat, share in MIX.items()}
    for cat, n in want.items():
        kinds = by_cat.get(cat, {})
        order = sorted(kinds)
        while n > 0 and any(kinds[k] for k in order):
            for k in order:
                if n > 0 and kinds[k]:
                    chosen.append(kinds[k].pop())
                    n -= 1
    rest = [c for kinds in by_cat.values() for lst in kinds.values() for c in lst]
    rng.shuffle(rest)
    chosen += rest[: max(0, size - len(chosen))]
    rng.shuffle(chosen)
    return chosen[:size]


def _dedupe(cases: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cases:
        key = (c["asr_best"], c["asr_alt1"], c["context"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def make(seed: int = 11, train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE) -> tuple[list[dict], list[dict], dict]:
    """(train examples, test cases, notes on what was dropped)."""
    rng = random.Random(seed)
    leak = build.LeakCheck()
    pool = [c for f in all_fields() for c in field_examples(f, rng, leak.swap_words)] + template_examples(rng)
    test_pool = _dedupe([c for c in pool if c["side"] == "test"])
    train_new = _dedupe([c for c in pool if c["side"] == "train"])
    test = sample(test_pool, test_size, rng)
    seeds, _ = build.examples(rng, leak, variants=2, extra=True, generated=build.FULL_GENERATED)
    for c in seeds:
        c.update({"kind": f"seed_{c['kind']}", "field": "seeds", "side": "train"})
    # The old test set's leak check (build.py), then the new test set's.
    train_new = [c for c in train_new if not leak.too_close(c["asr_best"], c["target"]["text"])
                 and not (c["kind"].startswith("term_fix") and leak.uses_test_pair(c["asr_best"], c["target"]["text"]))]
    seeds, seeds_dropped = drop_close(seeds, test)
    train_new, new_dropped = drop_close(train_new, test)
    room = train_size - len(seeds)
    train = seeds + sample(train_new, room, rng) if room < len(train_new) else seeds + train_new
    rng.shuffle(train)
    notes = {"test_pool": len(test_pool), "train_pool_new": len(train_new) + new_dropped, "seed_examples": len(seeds),
             "dropped_close_to_test": {"seeds": seeds_dropped, "new": new_dropped}}
    return train[:train_size], test, notes


def _count(cases: list[dict], key) -> dict:
    return dict(sorted(Counter(key(c) for c in cases).items()))


def write(name: str, train: list[dict], test: list[dict], notes: dict, seed: int, out_dir: Path = OUT,
          prompt_name: str = build.PROMPT) -> dict:
    from eval.dictation_cleanup.prompt_files import load_prompt

    prompt = load_prompt(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_train.jsonl").open("w", encoding="utf-8") as f:
        for c in train:
            f.write(json.dumps(build.to_chat(c, prompt), ensure_ascii=False) + "\n")
    with (out_dir / f"{name}_train_preview.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "field", "app", "before", "heard", "other_hearing", "word_list", "answer_text",
                    "alternatives", "uncertain"])
        for c in train:
            t = c["target"]
            w.writerow([c["kind"], c.get("field", ""), c["app"], c["context"], c["asr_best"], c["asr_alt1"],
                        c["personal_words"], t["text"], " || ".join(t["alternatives"]), t["uncertain"]])
    test_path = out_dir / f"{name}_test.csv"
    with test_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEST_COLUMNS)
        w.writeheader()
        for i, c in enumerate(test, 1):
            w.writerow({"id": f"B{i:04d}", "split": "test", "category": c["category"], "app": c["app"],
                        "context": c["context"], "asr_best": c["asr_best"], "asr_alt1": c["asr_alt1"], "asr_alt2": "",
                        "personal_words": c["personal_words"], "intended": c["intended"], "must_keep": c["must_keep"],
                        "expected_behavior": c["expected"], "notes": f"{c['kind']}; field {c['field']}"})
    info = {"prompt": prompt.name, "prompt_sha256": prompt.sha256, "seed": seed, "train": len(train), "test": len(test),
            "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()[:16], **notes,
            "test_by_category": _count(test, lambda c: c["category"]),
            "test_by_kind": _count(test, lambda c: c["kind"].split(":")[0]),
            "test_by_field": _count(test, lambda c: c["field"]),
            "train_by_kind": _count(train, lambda c: c["kind"].split(":")[0]),
            "train_by_field": _count(train, lambda c: c.get("field", "")),
            "fields": len(all_fields()), "field_sentences": sum(len(f.sentences) for f in all_fields())}
    (out_dir / f"{name}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the big clean-up training file and test set from the field banks.")
    p.add_argument("--name", default="big1")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    p.add_argument("--test-size", type=int, default=TEST_SIZE)
    args = p.parse_args(argv)
    train, test, notes = make(args.seed, args.train_size, args.test_size)
    info = write(args.name, train, test, notes, args.seed)
    print(json.dumps({k: v for k, v in info.items() if not k.endswith("_field")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
