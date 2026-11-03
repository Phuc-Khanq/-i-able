"""Builds the clean-up training file from seeds.py and noise.py.

    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.build                 # pilot, about 250 examples
    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.build --variants 3    # more noisy copies per sentence
    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.build --name full1 --full  # every seed and template

Writes data/cleanup_train/<name>.jsonl (git-ignored; one chat per line: the eval prompt's system and user
messages, and the answer as the model should give it) and <name>_preview.csv to read the examples in Excel.

Leak check: drops any example whose heard or meant text is close to a test case, and any misheard word pair
that a test case uses, so eval/dictation_cleanup/ stays a fair test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from engine.voice.cleanup_train import noise, seeds

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "cleanup_train"
TESTSET = ROOT / "eval" / "dictation_cleanup" / "testset.csv"
PROMPT = "t1_short"         # short instructions: training puts the know-how in the weights, and every token
                            # of prompt costs training time (v4_full is about 1,100 tokens, t1_short about 120)
SIMILAR = 0.6            # token overlap with a test case at or above this counts as a leak


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


class LeakCheck:
    """What the training data may not contain: test sentences, and the words test cases mishear."""

    def __init__(self, path: Path = TESTSET):
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.sentences = [_tokens(t) for r in rows for t in (r["asr_best"], r["intended"].split("||")[0])]
        swap: set[str] = set()
        for r in rows:
            heard = _tokens(r["asr_best"])
            for other in [r["asr_alt1"], r["asr_alt2"], *r["intended"].split("||")]:
                if other.strip():                   # every word that differs between two readings of a case
                    swap |= heard ^ _tokens(other)
        # Negations and numbers are whole categories the model must learn (can/can't, thirteen/thirty), not a
        # word pair to keep out; too_close() still keeps the test sentences themselves out.
        generic = noise.FUNCTION_WORDS | noise.NUMBER_WORDS | noise.NEGATIONS
        self.swap_words = {w for w in swap if w not in generic and len(w) > 2 and not w.endswith("n't")}

    def too_close(self, *texts: str) -> bool:
        return any(_overlap(_tokens(t), s) >= SIMILAR for t in texts if t for s in self.sentences)

    def uses_test_pair(self, a: str, b: str) -> bool:
        """Whether the words that differ between two readings include a word a test case mixes up."""
        return bool((_tokens(a) ^ _tokens(b)) & self.swap_words)


def answer(text: str, alternatives: list[str] | None = None, uncertain: bool = False) -> dict:
    return {"text": text, "alternatives": alternatives or [], "uncertain": uncertain}


def case(kind: str, app: str, before: str, heard: str, target: dict, other: str = "",
         words: list[str] | None = None) -> dict:
    return {"kind": kind, "app": app, "context": before, "asr_best": heard, "asr_alt1": other, "asr_alt2": "",
            "personal_words": "|".join(words or []), "target": target}


def examples(rng: random.Random, leak: LeakCheck, variants: int = 2, extra: bool = False,
             generated: dict | None = None) -> tuple[list[dict], list[str]]:
    """Every training example, and a note for each one the leak check dropped.

    extra=True adds seeds_extra.py; generated={"corrections": n, "negations": n, "numbers": n} adds that many
    template cases (templates.py). The pilot used neither.
    """
    mods = [seeds]
    if extra:
        from engine.voice.cleanup_train import seeds_extra

        mods.append(seeds_extra)

    def rows(name: str) -> list:
        return [row for m in mods for row in getattr(m, name, [])]

    self_correct, ambiguous = rows("SELF_CORRECT"), rows("AMBIGUOUS")
    if generated:
        from engine.voice.cleanup_train import templates

        self_correct += templates.corrections(rng, generated.get("corrections", 0))
        ambiguous += templates.negation_pairs(rng, generated.get("negations", 0))
        ambiguous += templates.number_pairs(rng, generated.get("numbers", 0))

    out: list[dict] = []
    for app, before, text in rows("PLAIN"):
        if rng.random() < 0.25:                     # some come as they are: "already fine, keep it"
            out.append(case("clean", app, before, text, answer(text)))
        for v in range(variants):
            broken = v == 1 and rng.random() < 0.5
            heard, misspelled = noise.messy(text, rng, leak.swap_words, broken=broken)
            other = re.sub(r"[.?!]+$", "", text) if misspelled and rng.random() < 0.4 else ""
            out.append(case("broken" if broken else "messy", app, before, heard, answer(text), other))
    for app, before, text in rows("CASUAL"):
        out.append(case("clean", app, before, text, answer(text)))
    for app, before, heard, meant in rows("SENTENCE_FIX"):
        out.append(case("sentence_fix", app, before, heard, answer(meant)))
    for app, before, heard, other, meant in rows("CONTEXT_FIX"):
        out.append(case("context_fix", app, before, heard, answer(meant), other))
    for app, before, heard, other, meant in rows("CONTEXT_KEEP"):
        out.append(case("context_keep", app, before, heard, answer(meant), other))
    for app, before, heard, other, words, meant in rows("NAME_FIX"):
        out.append(case("name_fix", app, before, heard, answer(meant), other, words))
    for app, before, heard, meant in self_correct:
        out.append(case("self_correct", app, before, heard, answer(meant)))
    for app, before, heard, meant in rows("BROKEN"):
        out.append(case("broken", app, before, heard, answer(meant)))
        messy_heard, _ = noise.messy(heard, rng, leak.swap_words)
        out.append(case("broken", app, before, messy_heard, answer(meant)))
    for app, before, heard, other, words in ambiguous:
        out.append(case("ambiguous", app, before, heard, answer(_sentence(heard), [_sentence(other)], True), other, words))
    for app, before, heard, text, readings in rows("CONTEXT_AMBIGUOUS"):
        if readings:
            out.append(case("context_ambiguous", app, before, heard, answer(text, readings, True)))
        else:                                       # nothing to choose between: clear words, keep as said
            out.append(case("context_keep", app, before, heard, answer(text)))
    for app, before, heard in rows("CUT_OFF"):
        out.append(case("cut_off", app, before, heard, answer(heard, [], True)))
    for app, before, heard, meant in rows("NOT_FOR_YOU"):
        out.append(case("not_for_you", app, before, heard, answer(meant)))

    kept, dropped, seen = [], [], set()
    for c in out:
        meant = c["target"]["text"]
        key = (c["asr_best"], c["asr_alt1"], c["context"])
        if key in seen:
            continue                                # the same line twice (seeds and templates can overlap)
        seen.add(key)
        if leak.too_close(c["asr_best"], meant):
            dropped.append(f"{c['kind']}: too close to a test case: {c['asr_best']!r}")
        elif _mixes_test_words(c, leak):
            dropped.append(f"{c['kind']}: uses a word a test case mishears: {c['asr_best']!r}")
        else:
            kept.append(c)
    return kept, dropped


def _mixes_test_words(c: dict, leak: LeakCheck) -> bool:
    """A fix or an ambiguous pair that turns on a word the test set mixes up."""
    meant, heard, other = c["target"]["text"], c["asr_best"], c["asr_alt1"]
    if c["kind"] in ("sentence_fix", "context_fix", "name_fix"):
        return leak.uses_test_pair(heard, meant)
    if c["kind"] in ("ambiguous", "context_keep") and other:
        return leak.uses_test_pair(heard, other)
    return False


def _sentence(text: str) -> str:
    """Capital first letter and an end mark, for answers built from a heard line."""
    text = text.strip()
    if not text:
        return text
    question = re.match(r"(?i)(can|could|is|are|do|does|did|will|would|what|where|when|who|how|why)\b", text)
    end = "" if text[-1] in ".?!" else ("?" if question else ".")
    return text[0].upper() + text[1:] + end


def to_chat(c: dict, prompt) -> dict:
    """The eval prompt's messages plus the answer, so the trained model sees exactly what the app will send."""
    msgs = prompt.messages(c) + [{"role": "assistant", "content": json.dumps(c["target"], ensure_ascii=False)}]
    return {"kind": c["kind"], "messages": msgs}


FULL_GENERATED = {"corrections": 90, "negations": 48, "numbers": 24}


def build(name: str = "pilot", variants: int = 2, seed: int = 7, out_dir: Path = OUT,
          prompt_name: str = PROMPT, full: bool = False) -> tuple[Path, dict, list[str]]:
    """full=False is the pilot (seeds.py only); full=True adds seeds_extra.py and the template cases."""
    from eval.dictation_cleanup.prompt_files import load_prompt

    rng = random.Random(seed)
    kept, dropped = examples(rng, LeakCheck(), variants, extra=full, generated=FULL_GENERATED if full else None)
    rng.shuffle(kept)
    prompt = load_prompt(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(to_chat(c, prompt), ensure_ascii=False) + "\n")
    with (out_dir / f"{name}_preview.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "app", "before", "heard", "other_hearing", "word_list", "answer_text", "alternatives", "uncertain"])
        for c in kept:
            t = c["target"]
            w.writerow([c["kind"], c["app"], c["context"], c["asr_best"], c["asr_alt1"], c["personal_words"],
                        t["text"], " || ".join(t["alternatives"]), t["uncertain"]])
    counts: dict[str, int] = {}
    for c in kept:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    (out_dir / f"{name}_info.json").write_text(json.dumps(
        {"prompt": prompt.name, "prompt_sha256": prompt.sha256, "seed": seed, "variants": variants, "full": full,
         "examples": len(kept), "by_kind": counts, "dropped": dropped}, indent=2), encoding="utf-8")
    return path, counts, dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the clean-up training file.")
    p.add_argument("--name", default="pilot")
    p.add_argument("--variants", type=int, default=2, help="noisy copies per plain sentence")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prompt", default=PROMPT, help="prompt version in eval/dictation_cleanup/prompts/")
    p.add_argument("--full", action="store_true", help="add seeds_extra.py and the template cases (not the pilot)")
    args = p.parse_args(argv)
    path, counts, dropped = build(args.name, args.variants, args.seed, prompt_name=args.prompt, full=args.full)
    print(f"{sum(counts.values())} examples -> {path}")
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n}")
    print(f"Dropped by the leak check: {len(dropped)}")
    for d in dropped:
        print(f"  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
