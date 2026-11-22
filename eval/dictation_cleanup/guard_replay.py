"""Replay the clean-up guard on answers that were already collected (no GPU, no model): what would a stricter guard
have shown, and was that better?

Why: the trained model sometimes changes text that was fine, or adds words to a short phrase ("Refresh." ->
"Please refresh the page."), and check_cleanup lets it through (STATUS Next 5c). This tool tries guard variants on the
saved answers and counts the good and the bad:

  - eval runs (results.jsonl of eval.dictation_cleanup.run: the dev cases and the 2,500-case test): how many answers
    equal what was meant, per category, and how many clean controls changed;
  - Whisper rows (checkpoints/hearings/*.cleanup-greedy.jsonl, from engine/voice/finetune/hearing_eval.py): words
    wrong against the reference, on LibriSpeech (clear book sentences), Khang's recordings and TORGO dev sentences
    (dysarthric speech, where the clean-up helps).

A blocked answer falls back the way the app does it (Rewriter._fallback: the rule-based light clean, else the
user's own words). A variant only ever blocks more than check_cleanup does.

    .\\.venv\\Scripts\\python.exe -m eval.dictation_cleanup.guard_replay
    .\\.venv\\Scripts\\python.exe -m eval.dictation_cleanup.guard_replay --run-dev <dir> --run-test <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
HEARINGS = ROOT / "checkpoints" / "hearings"
BIG1_DEV = "2026-11-03_big1-q8_0-greedy_cleanup_t1_short"
BIG1_TEST = "2026-11-03_big1-q8_0-greedy_cleanup_t1_short_big1_test"
WHISPER_SETS = {"libri-dev": "libri-dev.cleanup-greedy.jsonl", "personal-test": "personal-test.cleanup-greedy.jsonl",
                "torgo-dev": "torgo-dev.cleanup-greedy.jsonl"}



@dataclass(frozen=True)
class Variant:
    """Block an answer that adds `min_new` or more content words nobody said, when the input has at most
    `max_input` words. max_input 0: no extra rule (check_cleanup alone)."""

    name: str
    max_input: int = 0
    min_new: int = 1
    drop_run: int = 0          # also block when this many content words in a row were dropped and not repeated


VARIANTS = [Variant("check_cleanup before the short-phrase rule"),
            Variant("+ no new word, input up to 3 words", 3, 1),
            Variant("+ no new word, input up to 6 words", 6, 1),
            Variant("+ no new word, any length", 1000, 1),
            Variant("up to 6 words, + no dropped run of 3 content words", 6, 1, 3),
            Variant("up to 6 words, + no dropped run of 2 content words", 6, 1, 2)]


def dropped_runs(original: str, clean: str) -> list[list[str]]:
    """Runs of content words the answer removed, that it doesn't say elsewhere (a repeat or restart removed,
    "go to the go to the store", leaves its words behind). Fillers and "you know" are not content."""
    from engine.voice.rewrite import FILLERS, _content, normalize_words

    a, b = normalize_words(original), normalize_words(clean)
    kept = set(b)
    runs = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            run = [w for w in a[i1:i2] if w in _content(a[i1:i2]) and w not in FILLERS and w not in kept]
            if run:
                runs.append(run)
    return runs


def new_words(original: str, clean: str, hearings=(), context: str = "", words=()) -> list[str]:
    """Content words `clean` gained that nothing said or gave (engine.voice.rewrite.gained_content_words, the rule
    now in check_cleanup): not in the heard text, a hearing, the text typed before or the word list, and more of
    them than the words dropped (a swap is a fix)."""
    from engine.voice.rewrite import gained_content_words

    return gained_content_words(original, clean, [*hearings, context, *words])


def blocked(variant: Variant, original: str, clean: str, hearings=(), context: str = "", words=()) -> str | None:
    """Why this guard blocks the answer, or None. check_cleanup without its short-phrase rule comes first (the guard
    before 2026-11-22); the variant "input up to 6 words" is what check_cleanup does now."""
    from engine.voice.rewrite import check_cleanup, normalize_words

    from engine.voice.rewrite import CHANGE_OF_MIND

    why = check_cleanup(original, clean, list(words), context, hearings, short_input=0)   # the guard before 2026-11-22
    if why or not variant.max_input:
        return why
    if len(normalize_words(original)) <= variant.max_input:
        added = new_words(original, clean, hearings, context, words)
        if len(added) >= variant.min_new:
            return f"new words: {' '.join(added)}"
    if variant.drop_run and not CHANGE_OF_MIND.search(original):
        for run in dropped_runs(original, clean):
            if len(run) >= variant.drop_run:
                return f"dropped: {' '.join(run)}"
    return None


def shown(variant: Variant, original: str, clean: str, hearings=(), context: str = "", words=()) -> str:
    """The text the card would show: the answer, or the app's fallback when the guard blocks it."""
    from engine.voice.rewrite import Rewriter

    why = blocked(variant, original, clean, hearings, context, words)
    return clean if why is None else Rewriter._fallback(original, why, 0.0).clean


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


# ------------------------------------------------------------------ eval runs (dev, test)
def eval_rows(run: Path, cases_path: Path | None, split: str | None) -> list[tuple[dict, dict]]:
    """[(case, saved result row)] of one eval run."""
    from eval.dictation_cleanup import scoring

    if cases_path is not None:
        cases = scoring.load_case_file(cases_path)
    else:
        cases = scoring.load_cases(split=split)
    by_id = {c["id"]: c for c in cases}
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(by_id[r["id"]], r) for r in rows if r.get("json_ok") and r["id"] in by_id]


def score_eval(variant: Variant, pairs: list[tuple[dict, dict]]) -> dict:
    from eval.dictation_cleanup import scoring

    out = {"n": len(pairs), "blocked": 0, "exact": 0, "exact_lost": 0, "exact_gained": 0, "auto_pass": 0,
           "controls": 0, "controls_changed": 0}
    for case, row in pairs:
        words = [w.strip() for w in case.get("personal_words", "").split("|") if w.strip()]
        hearings = [h for h in (case.get("asr_alt1", ""), case.get("asr_alt2", "")) if h.strip()]
        text = shown(variant, case["asr_best"], row["text"], hearings, case.get("context", ""), words)
        out["blocked"] += text != row["text"]
        ans = scoring.Answer(text=text, alternatives=row["alternatives"], uncertain=row["uncertain"], json_ok=True)
        out["auto_pass"] += scoring.check_case(case, ans)["auto_pass"]
        if case["expected_behavior"] in ("rewrite", "unchanged"):
            want = _norm(case["intended"])
            was, now = _norm(row["text"]) == want, _norm(text) == want
            out["exact"] += now
            out["exact_lost"] += was and not now
            out["exact_gained"] += now and not was
        if case["category"] == "clean_control":
            out["controls"] += 1
            out["controls_changed"] += _norm(text) != _norm(case["asr_best"])
    return out


# ------------------------------------------------------------------ Whisper rows (words wrong)
def score_whisper(variant: Variant, rows: list[dict]) -> dict:
    from engine.voice.finetune.hearing_eval import errors

    out = {"rows": len(rows), "words": 0, "wrong": 0, "wrong_before": 0, "blocked": 0, "better": 0, "worse": 0}
    for r in rows:
        text = shown(variant, r["text"], r["clean_h"], r.get("hearings") or ())
        wrong = errors(r["ref"], text)[0]
        out["words"] += r["words"]
        out["wrong"] += wrong
        out["wrong_before"] += r["err_clean_h"]
        out["blocked"] += _norm(text) != _norm(r["clean_h"])
        out["better"] += wrong < r["err_clean_h"]
        out["worse"] += wrong > r["err_clean_h"]
    return out


def whisper_rows(name: str) -> list[dict]:
    path = HEARINGS / WHISPER_SETS[name]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.2f}%" if b else "-"


def report(dev: Path, test: Path) -> str:
    lines = []
    sets = {"dev (67)": eval_rows(dev, None, "dev"),
            "test (2,500)": eval_rows(test, ROOT / "data" / "cleanup_train" / "big1_test.csv", None)}
    for name, pairs in sets.items():
        lines += [f"\n### Eval answers, {name}: answers equal to what was meant ('rewrite' and 'unchanged' cases)\n",
                  "| guard | blocked | equal to meant | lost | gained | clean controls changed | auto pass |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for v in VARIANTS:
            s = score_eval(v, pairs)
            lines.append(f"| {v.name} | {s['blocked']} | {s['exact']} | {s['exact_lost']} | {s['exact_gained']} | "
                         f"{s['controls_changed']} of {s['controls']} | {s['auto_pass']} of {s['n']} |")
    for name in WHISPER_SETS:
        rows = whisper_rows(name)
        lines += [f"\n### Whisper rows, {name}: words wrong after the clean-up (with other hearings)\n",
                  "| guard | blocked | words wrong | rows better | rows worse |", "| --- | --- | --- | --- | --- |"]
        for v in VARIANTS:
            s = score_whisper(v, rows)
            lines.append(f"| {v.name} | {s['blocked']} of {s['rows']} | {pct(s['wrong'], s['words'])} "
                         f"({s['wrong']} of {s['words']}) | {s['better']} | {s['worse']} |")
        base = sum(r["err_text"] for r in rows)
        lines.append(f"\nWhisper alone: {pct(base, sum(r['words'] for r in rows))} ({base}).")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dev", type=Path, default=RUNS / BIG1_DEV)
    ap.add_argument("--run-test", type=Path, default=RUNS / BIG1_TEST)
    args = ap.parse_args(argv)
    print(report(args.run_dev, args.run_test))
    return 0


if __name__ == "__main__":
    sys.exit(main())
