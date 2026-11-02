"""The automatic checks for the dictation clean-up test set (testset.csv), and the summary of a run.

A case passes the automatic checks when the answer is usable (valid JSON for the new models), every
must_keep item is there, and the behaviour is right for expected_behavior. "Possible invented facts"
(a number, name or negation the person never said) are flagged separately: a person decides on the
review sheet. Meaning kept and nothing invented are human checks (README.md, "How to score a case").

Reads testset.csv only; never changes it.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from engine.voice.rewrite import NEGATIONS, NUMBER_WORDS
from engine.voice.voice_check import normalize_words, percentile

HERE = Path(__file__).resolve().parent
TESTSET = HERE / "testset.csv"
SPLITS = ("dev", "holdout")
CATEGORIES = ("mispronounced", "broken_structure", "mixed", "ambiguous", "clean_control")


def load_cases(path: Path = TESTSET, split: str = "dev") -> list[dict]:
    """The cases of one split. The file starts with a BOM on purpose (for Excel), hence utf-8-sig."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


# ------------------------------------------------------------------ the answer
@dataclass
class Answer:
    """What a backend answered. json_ok: True/False for JSON answers, None for a backend that answers
    in plain text (the app's current clean-up)."""

    text: str
    alternatives: list[str] = field(default_factory=list)
    uncertain: bool = False
    json_ok: bool | None = None
    raw: str = ""
    error: str | None = None


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_answer(raw: str) -> Answer:
    """The model's JSON answer {"text", "alternatives", "uncertain"}. Anything else is a failure
    (json_ok False, empty text), never retried. A ``` fence around the whole object is allowed."""
    if "<think>" in raw or "</think>" in raw:
        return Answer("", json_ok=False, raw=raw, error="thinking output")
    body = raw.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1)
    try:
        data = json.loads(body)
    except ValueError as exc:
        return Answer("", json_ok=False, raw=raw, error=f"not JSON: {exc}")
    if not isinstance(data, dict) or not {"text", "alternatives", "uncertain"} <= set(data):
        return Answer("", json_ok=False, raw=raw, error="missing keys")
    text, alts, unsure = data["text"], data["alternatives"], data["uncertain"]
    if not isinstance(text, str) or not isinstance(unsure, bool) or not isinstance(alts, list) \
            or not all(isinstance(a, str) for a in alts):
        return Answer("", json_ok=False, raw=raw, error="wrong types")
    return Answer(text, [a for a in alts if a.strip()], unsure, True, raw)


# ------------------------------------------------------------------ the checks
def normalize(text: str) -> str:
    """Lowercase, no punctuation, single spaces: what "unchanged" compares."""
    t = text.replace("’", "'").replace("‘", "'").lower()
    return " ".join(re.sub(r"[^\w\s]", "", t).split())


def must_keep_missing(case: dict, text: str) -> list[str]:
    """The must_keep items not in `text` (case-insensitive, whole words; digits are not number words)."""
    t = text.replace("’", "'")
    missing = []
    for item in (i.strip() for i in case["must_keep"].split("|")):
        if not item:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in item.split()) + r"(?!\w)"
        if not re.search(pattern, t, re.IGNORECASE):
            missing.append(item)
    return missing


def options(answer: Answer) -> list[str]:
    """The distinct readings the answer offers (text plus alternatives), normalized."""
    seen: list[str] = []
    for t in [answer.text, *answer.alternatives]:
        n = normalize(t)
        if n and n not in seen:
            seen.append(n)
    return seen


_TITLES = {"dr", "mr", "mrs", "ms", "st", "prof", "jr", "sr"}     # "Dr." doesn't end a sentence


def _word(token: str) -> str:
    return re.sub(r"'s$", "", re.sub(r"^[^\w']+|[^\w']+$", "", token.replace("’", "'")).lower())


def _names(text: str) -> set[str]:
    """Capitalized words that don't start a sentence: probably names."""
    names, start = set(), True
    for token in text.split():
        core = re.sub(r"^[^\w']+|[^\w']+$", "", token.replace("’", "'"))
        word = _word(token)
        if core and not start and core[0].isupper() and word not in {"i", "i'm", "i'll", "i've", "i'd"}:
            names.add(word)
        start = bool(re.search(r"[.!?][\"')\]]*$", token)) and word not in _TITLES
    return names


def _negations(text: str) -> set[str]:
    """Negation words, with "don't", "can't", "cannot" all as "not"."""
    found = set()
    for w in normalize_words(text):
        if w.endswith("n't") or w in {"not", "cannot"}:
            found.add("not")
        elif w in NEGATIONS:
            found.add(w)
    return found


def invented_facts(case: dict, answer: Answer) -> list[str]:
    """Numbers, names and negations in the answer (text and alternatives) that aren't in the input,
    the alternative transcripts, the context or the user's word list (a name from their own list isn't
    invented: case A08). Flags for a person to look at, not proof."""
    sources = [case.get(k, "") for k in ("asr_best", "asr_alt1", "asr_alt2", "context")]
    sources += [w for w in case.get("personal_words", "").split("|") if w.strip()]
    src_words = {w for s in sources for w in normalize_words(s)}
    src_tokens = {_word(t) for s in sources for t in s.split()} | src_words
    src_neg = set().union(*(_negations(s) for s in sources))
    flags: list[str] = []
    for out in [answer.text, *answer.alternatives]:
        for w in normalize_words(out):
            if w in NUMBER_WORDS and w not in src_words and f"number: {w}" not in flags:
                flags.append(f"number: {w}")
        for n in sorted(_names(out)):
            if n not in src_tokens and f"name: {n}" not in flags:
                flags.append(f"name: {n}")
        for n in sorted(_negations(out) - src_neg):
            if f"negation: {n}" not in flags:
                flags.append(f"negation: {n}")
    return flags


def check_case(case: dict, answer: Answer) -> dict:
    """The automatic checks for one case: auto_pass, the problems, possible invented facts, and whether
    the answer changed the text or silently picked one reading of an ambiguous case."""
    if answer.json_ok is False:
        return {"auto_pass": False, "problems": ["broken JSON"], "invented": [], "changed": None,
                "silent_guess": None, "options": 0}
    problems = [f"must-keep missing: {m}" for m in must_keep_missing(case, answer.text)]
    expected = case["expected_behavior"]
    changed = normalize(answer.text) != normalize(case["asr_best"])
    opts = options(answer)
    if expected in ("unchanged", "keep_original_and_flag") and changed:
        problems.append("changed")
    if expected == "show_both" and not (answer.uncertain or len(opts) >= 2):
        problems.append("picked one reading silently")
    if expected == "keep_original_and_flag" and not answer.uncertain:
        problems.append("not marked uncertain")
    silent = None
    if case["category"] == "ambiguous":
        silent = not answer.uncertain and len(opts) < 2 and (expected == "show_both" or changed)
    return {"auto_pass": not problems, "problems": problems, "invented": invented_facts(case, answer),
            "changed": changed, "silent_guess": silent, "options": len(opts)}


def result_row(case: dict, answer: Answer, ms: float, tokens: int | None = None, extra: dict | None = None) -> dict:
    """One line of results.jsonl: the case, the answer, the checks and the timing."""
    row = {k: case[k] for k in ("id", "category", "expected_behavior", "asr_best", "intended", "must_keep")}
    row.update({"text": answer.text, "alternatives": answer.alternatives, "uncertain": answer.uncertain,
                "json_ok": answer.json_ok, "json_error": answer.error, "raw": answer.raw, "ms": ms,
                "tokens": tokens})
    row.update(check_case(case, answer))
    row.update(extra or {})
    return row


def error_row(case: dict, exc: Exception, ms: float) -> dict:
    """The backend crashed on this case: a failure, recorded, and the run goes on."""
    row = result_row(case, Answer("", raw=""), ms)
    row.update({"auto_pass": False, "problems": [f"error: {exc}"], "changed": None, "silent_guess": None})
    return row


# ------------------------------------------------------------------ the summary
def summarize(results: list[dict]) -> dict:
    """Pass rates and safety rates. Pairs are [count, out of]. Over-edit and silent-guess only count
    usable answers (a broken JSON answer is already counted under json_failures)."""
    def pair(rows, key):
        usable = [r for r in rows if r.get(key) is not None]
        return [sum(1 for r in usable if r[key]), len(usable)]

    by_cat = {}
    for cat in CATEGORIES:
        rows = [r for r in results if r["category"] == cat]
        if rows:
            by_cat[cat] = [sum(1 for r in rows if r["auto_pass"]), len(rows)]
    ms = [r["ms"] for r in results]
    tokens = [r["tokens"] for r in results if r.get("tokens") is not None]
    paths: dict[str, int] = {}
    for r in results:
        if r.get("path"):
            paths[r["path"]] = paths.get(r["path"], 0) + 1
    return {
        "n": len(results),
        "auto_pass": sum(1 for r in results if r["auto_pass"]),
        "by_category": by_cat,
        "over_edit": pair([r for r in results if r["category"] == "clean_control"], "changed"),
        "silent_guess": pair([r for r in results if r["category"] == "ambiguous"], "silent_guess"),
        "invented_cases": sum(1 for r in results if r["invented"]),
        "json_failures": sum(1 for r in results if r.get("json_ok") is False),
        "errors": sum(1 for r in results if any(p.startswith("error:") for p in r["problems"])),
        "latency_ms": {q: (percentile(ms, v) if ms else None) for q, v in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))},
        "tokens": {"total": sum(tokens), "mean": round(sum(tokens) / len(tokens), 1) if tokens else None},
        "paths": paths,
    }


def _rate(pair) -> str:
    k, n = pair
    return f"{k}/{n} ({k / n:.0%})" if n else "n/a"


def render_summary(s: dict, results: list[dict], meta: dict) -> str:
    """summary.md for one run."""
    lat = s["latency_ms"]
    lines = [f"# Clean-up run: {meta.get('model', '?')} · {meta.get('prompt', '?')} · {meta.get('split', '?')}", "",
             f"- Backend: {meta.get('backend')} · cases: {s['n']} · date: {meta.get('date')}",
             f"- Machine: {meta.get('machine', '?')}", "",
             "## Automatic checks", "", "| Category | Pass |", "| --- | --- |"]
    lines += [f"| {cat} | {_rate(p)} |" for cat, p in s["by_category"].items()]
    lines += [f"| **all** | **{_rate([s['auto_pass'], s['n']])}** |", "",
              "## Safety", "",
              f"- Over-edit (clean controls changed): {_rate(s['over_edit'])}",
              f"- Silent guess (ambiguous cases resolved without showing both): {_rate(s['silent_guess'])}",
              f"- Cases with a possible invented fact: {s['invented_cases']}",
              f"- Broken JSON answers: {s['json_failures']} · backend errors: {s['errors']}", "",
              "## Latency and size", "",
              f"- Latency per case (ms, wall time, after one warm-up): p50 {lat['p50']} · p90 {lat['p90']} · "
              f"p99 {lat['p99']}" + (" (p99 is about the slowest case with this few cases)" if s["n"] < 100 else ""),
              f"- Tokens generated: total {s['tokens']['total']} · mean per case {s['tokens']['mean']}",
              f"- Peak GPU memory (PyTorch, weights included): {meta.get('peak_gpu_gb', 'n/a')} GB"]
    if s["paths"]:
        lines += ["", "## What the app's guard did", "",
                  "- " + " · ".join(f"{k}: {v}" for k, v in sorted(s["paths"].items()))]
        reasons: dict[str, int] = {}
        for r in results:
            if r.get("guard_reason"):
                key = re.sub(r":.*", "", r["guard_reason"].replace("light clean: model ", ""))
                reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            lines.append("- Why the model's text was rejected: " +
                         " · ".join(f"{k}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    fails = [r for r in results if not r["auto_pass"]]
    flagged = [r for r in results if r["invented"]]
    lines += ["", f"## Failed the automatic checks ({len(fails)})", ""]
    lines += [f"- {r['id']} ({r['expected_behavior']}): {'; '.join(r['problems'])}" for r in fails] or ["- none"]
    lines += ["", f"## Possible invented facts ({len(flagged)})", ""]
    lines += [f"- {r['id']}: {', '.join(r['invented'])}" for r in flagged] or ["- none"]
    return "\n".join(lines) + "\n"


REVIEW_COLUMNS = ["id", "category", "expected_behavior", "input", "intended", "output", "alternatives", "uncertain",
                  "auto_pass", "auto_problems", "possible_invented", "meaning_kept_Khang", "nothing_invented_Khang",
                  "meaning_kept_Khoi", "nothing_invented_Khoi", "notes"]


def write_review(path: Path, results: list[dict]) -> None:
    """The sheet for the human checks, with a BOM so Excel opens it right. Reviewers fill yes/no."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in results:
            w.writerow({"id": r["id"], "category": r["category"], "expected_behavior": r["expected_behavior"],
                        "input": r["asr_best"], "intended": r["intended"],
                        "output": r["text"] if r.get("json_ok") is not False else f"[broken JSON] {r['raw']}",
                        "alternatives": " || ".join(r["alternatives"]), "uncertain": "yes" if r["uncertain"] else "",
                        "auto_pass": "yes" if r["auto_pass"] else "no", "auto_problems": "; ".join(r["problems"]),
                        "possible_invented": ", ".join(r["invented"])})
