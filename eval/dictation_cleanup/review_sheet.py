"""Blind review sheet for the human checks: meaning kept and nothing invented (README.md, "How to score a case").

    python -m eval.dictation_cleanup.review_sheet build runs/<run A> runs/<run B> --name 2026-11-02_dev
    python -m eval.dictation_cleanup.review_sheet score reviews/2026-11-02_dev.csv

build: one row per output, grouped by case. Each case's outputs get the labels A, B ... in a random order, so a
reviewer can't tell which run wrote which. Writes reviews/<name>.csv (for Khang and Khoi, with a BOM for Excel)
and reviews/<name>_key.csv (which run each output came from, and its automatic checks: don't open it while
reviewing).
score: reads the filled sheet and its key. Per run and reviewer: meaning kept, nothing invented, and the full
pass (both human checks plus the automatic ones), plus where the two reviewers disagree. Writes <name>_score.md.

Answers: y or n (yes/no/1/0 also work). An empty cell is "not reviewed" and is left out of the counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from eval.dictation_cleanup import scoring

REVIEWS = scoring.HERE / "reviews"
REVIEWERS = ("Khang", "Khoi")
CHECKS = ("meaning_kept", "nothing_invented")
SHEET_COLUMNS = (["case", "category", "expected_behavior", "heard", "other_hearings", "before", "word_list", "intended",
                  "output_id", "output"] + [f"{c}_{r}" for r in REVIEWERS for c in CHECKS] + ["case_looks_wrong", "notes"])
KEY_COLUMNS = ["output_id", "run", "auto_pass", "auto_problems", "possible_invented"]


def _results(run_dir: Path) -> dict[str, dict]:
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f)}


def show(row: dict) -> str:
    """The output as a reviewer sees it: the text, any other readings, and the unsure flag."""
    if row.get("json_ok") is False:
        return f"[broken answer] {row.get('raw', '')}"
    out = row["text"]
    if row.get("alternatives"):
        out += "   [or: " + " | ".join(row["alternatives"]) + "]"
    if row.get("uncertain"):
        out += "   [marked unsure]"
    return out


def build(run_dirs: list[Path], name: str, out_dir: Path = REVIEWS, seed: int = 0, split: str = "dev") -> Path:
    runs = [(d.name, _results(d)) for d in run_dirs]
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet, key = out_dir / f"{name}.csv", out_dir / f"{name}_key.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fs, key.open("w", encoding="utf-8", newline="") as fk:
        ws, wk = csv.DictWriter(fs, fieldnames=SHEET_COLUMNS), csv.DictWriter(fk, fieldnames=KEY_COLUMNS)
        ws.writeheader()
        wk.writeheader()
        for case in scoring.load_cases(split=split):
            order = [(run, res[case["id"]]) for run, res in runs if case["id"] in res]
            rng.shuffle(order)
            for label, (run, row) in zip("ABCDEFGH", order):
                oid = f"{case['id']}-{label}"
                ws.writerow({"case": case["id"], "category": case["category"],
                             "expected_behavior": case["expected_behavior"], "heard": case["asr_best"],
                             "other_hearings": " | ".join(a for a in (case["asr_alt1"], case["asr_alt2"]) if a),
                             "before": case["context"], "word_list": case["personal_words"].replace("|", ", "),
                             "intended": case["intended"], "output_id": oid, "output": show(row)})
                wk.writerow({"output_id": oid, "run": run, "auto_pass": "yes" if row["auto_pass"] else "no",
                             "auto_problems": "; ".join(row["problems"]),
                             "possible_invented": ", ".join(row["invented"])})
    return sheet


def _answer(cell: str) -> bool | None:
    c = (cell or "").strip().lower()
    if c in ("y", "yes", "1", "true"):
        return True
    if c in ("n", "no", "0", "false"):
        return False
    return None


def score(sheet: Path) -> dict:
    """Per run: for each reviewer, [yes, reviewed] for each check and for the full pass; and disagreements."""
    key_path = sheet.with_name(sheet.stem + "_key.csv")
    with key_path.open(encoding="utf-8", newline="") as f:
        key = {r["output_id"]: r for r in csv.DictReader(f)}
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict = {}
    for row in rows:
        k = key[row["output_id"]]
        run = out.setdefault(k["run"], {"outputs": 0, "disagree": [], **{r: {c: [0, 0] for c in (*CHECKS, "full_pass")}
                                                                          for r in REVIEWERS}})
        run["outputs"] += 1
        for r in REVIEWERS:
            answers = [_answer(row[f"{c}_{r}"]) for c in CHECKS]
            for c, a in zip(CHECKS, answers):
                if a is not None:
                    run[r][c][1] += 1
                    run[r][c][0] += a
            if None not in answers:
                run[r]["full_pass"][1] += 1
                run[r]["full_pass"][0] += all(answers) and k["auto_pass"] == "yes"
        for c in CHECKS:
            a, b = (_answer(row[f"{c}_{r}"]) for r in REVIEWERS)
            if a is not None and b is not None and a != b:
                run["disagree"].append(f"{row['output_id']} {c}")
    return out


def render_score(result: dict) -> str:
    rate = lambda p: f"{p[0]}/{p[1]}" + (f" ({p[0] / p[1]:.0%})" if p[1] else "")
    lines = ["# Human review score", "", "| Run | Reviewer | Meaning kept | Nothing invented | Full pass |",
             "| --- | --- | --- | --- | --- |"]
    for run, s in result.items():
        for r in REVIEWERS:
            lines.append(f"| {run} | {r} | {rate(s[r]['meaning_kept'])} | {rate(s[r]['nothing_invented'])} | "
                         f"{rate(s[r]['full_pass'])} |")
    lines += ["", "Full pass = both human checks yes and the automatic checks passed.", "", "## Disagreements", ""]
    for run, s in result.items():
        lines.append(f"- {run}: " + (", ".join(s["disagree"]) if s["disagree"] else "none"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blind review sheets for the clean-up runs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("runs", nargs="+", type=Path, help="run folders (eval/dictation_cleanup/runs/...)")
    b.add_argument("--name", required=True, help="sheet name, e.g. 2026-11-02_dev")
    b.add_argument("--split", choices=list(scoring.SPLITS), default="dev")
    b.add_argument("--out", type=Path, default=REVIEWS)
    s = sub.add_parser("score")
    s.add_argument("sheet", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "build":
        print(f"Review sheet: {build(args.runs, args.name, args.out, split=args.split)}")
    else:
        text = render_score(score(args.sheet))
        args.sheet.with_name(args.sheet.stem + "_score.md").write_text(text, encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
