"""Does the clean-up model's own confidence flag its wrong answers?

llama-server can report each generated token's probability (n_probs, from the full distribution before
sampling). This measures whether a low probability inside the answer's "text" marks the answers the scorer
calls wrong, on the same greedy Q8_0 answers the scorer ran.

    python -m eval.dictation_cleanup.confidence collect --model checkpoints/cleanup/big1/gguf/big1-q8_0.gguf --label big1 --cases test
    python -m eval.dictation_cleanup.confidence second  --model ... --label big1 --cases test
    python -m eval.dictation_cleanup.confidence report  --label big1 --cases test --run 2026-11-03_big1-q8_0-greedy_cleanup_t1_short_big1_test

collect: the answer and, per token, its log-probability and the top 5 candidates -> runs/confidence/<label>_<cases>.jsonl
second:  for answers whose least-sure text token is under 0.95: put the runner-up token there, let the model
         finish (greedy), and score that second reading with the scorer's rules -> <label>_<cases>_second.jsonl
report:  joins with the scored run (right/wrong per case) and prints what each flag would catch.

Signals, all from the answer's "text" field (a case is flagged when the probability is below a threshold):
  changed  the least-sure token inside a word Whisper didn't hear (not in asr_best)
  any      the least-sure token anywhere in the text, punctuation included
  content  the least-sure token inside a word that isn't a small word (SMALL); punctuation-only tokens skipped
  p_unsure the model's own probability of writing "uncertain": true
Outputs stay on this PC (runs/ is git-ignored). Nothing leaves the machine: llama-server listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from engine.voice.cleanup_model import SMALL_WORDS as SMALL, text_span, weakest_word   # what the app ships

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "dictation_cleanup" / "runs" / "confidence"
TEST_FILE = ROOT / "data" / "cleanup_train" / "big1_test.csv"
WORD = re.compile(r"[\w'’]+")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def norm(word: str) -> str:
    return word.lower().replace("’", "'").strip("'")


def signals(toks: list[dict], asr_best: str) -> dict:
    """Log-probability signals for one answer. toks: [{"id", "b" (bytes as ints), "lp", "top"}] in order."""
    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t["b"])
    raw = b"".join(bytes(t["b"]) for t in toks)
    out = {"raw": raw.decode("utf-8", "replace"), "changed": 0.0, "any": 0.0, "content": 0.0,
           "weak": None, "cweak": None, "p_unsure": None}
    span = text_span(raw)
    if span is not None:
        s, e = span
        in_text = [i for i, t in enumerate(toks) if starts[i] < e and starts[i] + len(t["b"]) > s]
        out["any"] = min((toks[i]["lp"] for i in in_text), default=0.0)
        text = raw[s:e].decode("utf-8", "replace")
        char_b = [0]
        for ch in text:
            char_b.append(char_b[-1] + len(ch.encode("utf-8")))
        heard = {norm(w) for w in WORD.findall(asr_best)}
        for m in WORD.finditer(text):
            if norm(m.group()) in heard:
                continue
            b0, b1 = s + char_b[m.start()], s + char_b[m.end()]
            for i in in_text:
                if starts[i] < b1 and starts[i] + len(toks[i]["b"]) > b0 and toks[i]["lp"] < out["changed"]:
                    out["changed"], out["weak"] = toks[i]["lp"], m.group()
        weak = weakest_word([{"bytes": t["b"], "logprob": t["lp"]} for t in toks])   # the app's own function
        if weak is not None and weak[1] < 1.0:
            out["content"], out["cweak"] = math.log(weak[1]), weak[0]
    key = b'"uncertain": '
    k = raw.find(key)
    if k >= 0:
        at = k + len(key)
        for i, t in enumerate(toks):
            if starts[i] <= at < starts[i] + len(t["b"]):
                out["p_unsure"] = sum(math.exp(lp) for _, b, lp in t["top"]
                                      if bytes(b).decode("utf-8", "replace").strip().startswith("t"))
                break
    return out


def weakest_text_token(toks: list[dict]) -> int | None:
    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t["b"])
    span = text_span(b"".join(bytes(t["b"]) for t in toks))
    if span is None:
        return None
    s, e = span
    idx = [i for i, t in enumerate(toks) if starts[i] < e and starts[i] + len(t["b"]) > s]
    return min(idx, key=lambda i: toks[i]["lp"]) if idx else None


# ------------------------------------------------------------------ the model runs

def _cases(which: str) -> list[dict]:
    from eval.dictation_cleanup import scoring

    return scoring.load_case_file(TEST_FILE) if which == "test" else scoring.load_cases(split="dev")


def _backend(model: str):
    from engine.voice.cleanup_train.export import LLAMA_CPP
    from eval.dictation_cleanup import backends

    b = backends.LlamaCppBackend(str(ROOT / model), str(LLAMA_CPP / "bin" / "llama-server.exe"),
                                 sampling=backends.LlamaCppBackend.GREEDY)
    b.load()
    return b


def _prompt_ids(b, prompt, case: dict) -> list[int]:
    text = b._template.apply_chat_template(prompt.messages(case), add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
    return b._tok(text, add_special_tokens=False)["input_ids"]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(model: str, label: str, which: str) -> Path:
    from eval.dictation_cleanup.prompt_files import load_prompt

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}_{which}.jsonl"
    done = {r["id"] for r in _read_jsonl(path)} if path.exists() else set()
    prompt, cases = load_prompt("cleanup_t1_short"), _cases(which)
    b, t0 = _backend(model), time.time()
    try:
        with path.open("a", encoding="utf-8") as f:
            for n, case in enumerate(cases):
                if case["id"] in done:
                    continue
                r = b._post("/completion", {"prompt": _prompt_ids(b, prompt, case), "n_predict": 256, "seed": 1234,
                                            "cache_prompt": True, "stream": False, "n_probs": 5, **b.sampling})
                toks = [{"id": t["id"], "b": t["bytes"], "lp": t["logprob"],
                         "top": [[c["id"], c["bytes"], c["logprob"]] for c in t["top_logprobs"]]}
                        for t in r["completion_probabilities"]]
                f.write(json.dumps({"id": case["id"], "raw": r["content"], "toks": toks}, ensure_ascii=False) + "\n")
                if n % 100 == 0:
                    f.flush()
                    print(f"{label} {which} {n + 1}/{len(cases)} {time.time() - t0:.0f}s", flush=True)
    finally:
        b.close()
    print(f"done {label} {which} in {time.time() - t0:.0f}s", flush=True)
    return path


def second(model: str, label: str, which: str, below: float = 0.95) -> Path:
    from eval.dictation_cleanup import scoring
    from eval.dictation_cleanup.prompt_files import load_prompt

    cases = {c["id"]: c for c in _cases(which)}
    todo = []
    for r in _read_jsonl(OUT / f"{label}_{which}.jsonl"):
        i = weakest_text_token(r["toks"])
        if i is not None and r["toks"][i]["lp"] < math.log(below) and len(r["toks"][i]["top"]) > 1:
            todo.append((r, i))
    print(f"{label} {which}: {len(todo)} answers have a text token under {below}", flush=True)
    prompt, path = load_prompt("cleanup_t1_short"), OUT / f"{label}_{which}_second.jsonl"
    b, t0 = _backend(model), time.time()
    try:
        with path.open("w", encoding="utf-8") as f:
            for r, i in todo:
                case, chosen = cases[r["id"]], r["toks"][i]["id"]
                alt = next(c for c in r["toks"][i]["top"] if c[0] != chosen)
                prefix = [t["id"] for t in r["toks"][:i]] + [alt[0]]
                res = b._post("/completion", {"prompt": _prompt_ids(b, prompt, case) + prefix, "n_predict": 256,
                                              "seed": 1234, "cache_prompt": True, "stream": False, **b.sampling})
                raw2 = (b"".join(bytes(t["b"]) for t in r["toks"][:i]) + bytes(alt[1])).decode("utf-8", "replace") \
                    + res["content"]
                ans = scoring.parse_answer(raw2)
                f.write(json.dumps({"id": r["id"], "p_first": math.exp(r["toks"][i]["lp"]), "p_second": math.exp(alt[2]),
                                    "first_tok": bytes(r["toks"][i]["b"]).decode("utf-8", "replace"),
                                    "second_tok": bytes(alt[1]).decode("utf-8", "replace"), "text2": ans.text,
                                    "json_ok": ans.json_ok, "pass2": scoring.check_case(case, ans)["auto_pass"]},
                                   ensure_ascii=False) + "\n")
    finally:
        b.close()
    print(f"done {label} {which} second readings in {time.time() - t0:.0f}s", flush=True)
    return path


# ------------------------------------------------------------------ the report

def new_wrong_words(row: dict) -> list[str]:
    """Content words in the answer that neither Whisper's words nor the intended text has."""
    known = {norm(w) for w in WORD.findall(row["asr_best"] + " " + row["intended"])}
    return [w for w in WORD.findall(row["text"]) if norm(w) not in known and norm(w) not in SMALL]


def load(label: str, which: str, run: str) -> list[dict]:
    recs = {r["id"]: r for r in _read_jsonl(OUT / f"{label}_{which}.jsonl")}
    sec_path = OUT / f"{label}_{which}_second.jsonl"
    sec = {r["id"]: r for r in _read_jsonl(sec_path)} if sec_path.exists() else {}
    data = []
    for row in _read_jsonl(ROOT / "eval" / "dictation_cleanup" / "runs" / run / "results.jsonl"):
        if row["id"] not in recs:
            continue
        sig = signals(recs[row["id"]]["toks"], row["asr_best"])
        data.append({**row, **sig, "same_answer": recs[row["id"]]["raw"].strip() == (row.get("raw") or "").strip(),
                     "sure": not row["uncertain"] and row["options"] <= 1, "wrong": not row["auto_pass"],
                     "new": new_wrong_words(row) if not row["auto_pass"] else [], "second": sec.get(row["id"])})
    return data


def report(data: list[dict], label: str, which: str, show: float | None = None) -> None:
    sure = [d for d in data if d["sure"]]
    wrong = [d for d in sure if d["wrong"]]
    right = [d for d in sure if not d["wrong"]]
    clean = [d for d in right if d["category"] == "clean_control"]
    new = [d for d in wrong if d["new"]]
    print(f"{label} {which}: {len(data)} cases, answers identical to the scored run: {sum(d['same_answer'] for d in data)}")
    print(f"answers given as sure: {len(sure)}; wrong {len(wrong)} (with a new wrong word {len(new)}), right {len(right)}"
          f" (clean sentences {len(clean)})")
    for key in ("changed", "any", "content"):
        print(f"\nflag on '{key}': p under | wrong caught | new-word wrong caught | right flagged | clean flagged"
              + (" | 2nd reading right" if key == "any" else ""))
        for t in THRESHOLDS:
            lt = math.log(t)
            f = [d for d in wrong if d[key] < lt]
            line = (f"  {t:>5} | {len(f):>3}/{len(wrong)} | {sum(d[key] < lt for d in new):>3}/{len(new)}"
                    f" | {sum(d[key] < lt for d in right):>4}/{len(right)} ({100 * sum(d[key] < lt for d in right) / max(1, len(right)):.1f}%)"
                    f" | {sum(d[key] < lt for d in clean):>3}/{len(clean)}")
            if key == "any":
                line += f" | {sum(1 for d in f if (d['second'] or {}).get('pass2'))}/{len(f)}"
            print(line)
    hi = [d for d in sure if d["p_unsure"] is not None and d["p_unsure"] > 0.01]
    print(f"\nthe model's own 'uncertain: true' probability above 1%: {sum(d['wrong'] for d in hi)} wrong, "
          f"{sum(not d['wrong'] for d in hi)} right")
    if show is not None:
        print(f"\nwrong answers (content p; N = has a new wrong word; * = flagged at {show}):")
        for d in sorted(wrong, key=lambda d: d["content"]):
            mark = "*" if d["content"] < math.log(show) else " "
            print(f"  {mark} {d['id']} {'N' if d['new'] else ' '} p={math.exp(d['content']):.2f} {d['cweak']!r}: "
                  f"{d['text']!r} (meant {d['intended']!r})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="The clean-up model's own confidence as an 'unsure' flag.")
    p.add_argument("command", choices=["collect", "second", "report"])
    p.add_argument("--model", help="the GGUF, relative to the repo (collect, second)")
    p.add_argument("--label", required=True)
    p.add_argument("--cases", choices=["test", "dev"], default="test")
    p.add_argument("--run", help="the scored run folder under runs/ (report)")
    p.add_argument("--show", type=float, help="report: list the wrong answers, marking those flagged at this p")
    args = p.parse_args(argv)
    if args.command == "collect":
        collect(args.model, args.label, args.cases)
    elif args.command == "second":
        second(args.model, args.label, args.cases)
    else:
        report(load(args.label, args.cases, args.run), args.label, args.cases, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
