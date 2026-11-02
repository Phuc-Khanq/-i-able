"""Runs one clean-up backend over the test set and writes a run folder: the evaluation runner.

    python -m eval.dictation_cleanup.run --backend current                      # today's clean-up, the baseline
    python -m eval.dictation_cleanup.run --backend transformers --model Qwen/Qwen3.5-4B --prompt v1_plain
    python -m eval.dictation_cleanup.run --backend transformers --model Qwen/Qwen3.5-4B --prompt v1_full
    python -m eval.dictation_cleanup.run --backend fake --limit 5               # no GPU: checks the runner

Writes eval/dictation_cleanup/runs/<date>_<model>_<prompt>/ (git-ignored): results.jsonl (one line per case,
written as it goes), review.csv (the human checks), summary.md and summary.json, settings.json.

Dev split by default. --split holdout prints a warning: run it only to report, after the pass bar is set.
Close the engine before a GPU run: the 16 GB card fits one heavy job at a time.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

from eval.dictation_cleanup import scoring
from eval.dictation_cleanup.prompt_files import load_prompt

RUNS = scoring.HERE / "runs"


def machine() -> str:
    parts = [f"{platform.system()} {platform.release()}", f"Python {platform.python_version()}"]
    torch = sys.modules.get("torch")
    if torch is not None:
        parts.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            parts.append(torch.cuda.get_device_name(0))
    transformers = sys.modules.get("transformers")
    if transformers is not None:
        parts.append(f"transformers {transformers.__version__}")
    return ", ".join(parts)


def run_folder(out_root: Path, model_label: str, prompt_label: str) -> Path:
    """A new folder <date>_<model>_<prompt>, with _2, _3 ... so a run never overwrites another."""
    short = re.sub(r"[^\w.-]+", "-", model_label.split("/")[-1])
    base = f"{time.strftime('%Y-%m-%d')}_{short}_{prompt_label}"
    folder, n = out_root / base, 2
    while folder.exists():
        folder, n = out_root / f"{base}_{n}", n + 1
    folder.mkdir(parents=True)
    return folder


def run(backend, cases: list[dict], prompt, out_root: Path = RUNS, model_label: str | None = None,
        split: str = "dev", log=print) -> Path:
    """Load, warm up once (not timed), run every case, write the run folder. Returns the folder."""
    t0 = time.perf_counter()
    backend.load()
    load_s = round(time.perf_counter() - t0, 1)
    backend.warm_up()
    label = model_label or backend.model_label
    folder = run_folder(out_root, label, prompt.name if prompt is not None else "app")
    results: list[dict] = []
    with (folder / "results.jsonl").open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            t = time.perf_counter()
            try:
                reply = backend.clean(case, prompt)
                backend.sync()
                ms = round((time.perf_counter() - t) * 1000, 1)
                row = scoring.result_row(case, reply.answer, ms, reply.tokens, reply.extra)
            except Exception as exc:                      # recorded as a failure; never retried
                row = scoring.error_row(case, exc, round((time.perf_counter() - t) * 1000, 1))
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            log(f"[{i}/{len(cases)}] {case['id']} {'pass' if row['auto_pass'] else 'FAIL'} {row['ms']:.0f} ms")
    summary = scoring.summarize(results)
    meta = {"model": label, "backend": backend.name, "prompt": prompt.name if prompt is not None else "app",
            "prompt_sha256": prompt.sha256 if prompt is not None else None, "split": split,
            "date": time.strftime("%Y-%m-%d %H:%M"), "machine": machine(), "peak_gpu_gb": backend.peak_gpu_gb(),
            "load_s": load_s}
    (folder / "summary.json").write_text(json.dumps({**summary, "meta": meta}, indent=2), encoding="utf-8")
    (folder / "summary.md").write_text(scoring.render_summary(summary, results, meta), encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps({**backend.settings(), **meta}, indent=2), encoding="utf-8")
    scoring.write_review(folder / "review.csv", results)
    return folder


def make_backend(args):
    from eval.dictation_cleanup import backends

    if args.backend == "current":
        return backends.CurrentBackend(model=args.model, device=args.device)
    if args.backend == "transformers":
        if not args.model:
            raise SystemExit("--model is required for the transformers backend")
        return backends.TransformersBackend(args.model, device=args.device or "cuda", seed=args.seed,
                                            max_new_tokens=args.max_new_tokens)
    return backends.FakeBackend()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a dictation clean-up backend on eval/dictation_cleanup/testset.csv.")
    p.add_argument("--backend", choices=["current", "transformers", "fake"], required=True)
    p.add_argument("--model", help="model name (transformers), or an override of rewrite_model (current)")
    p.add_argument("--prompt", default="v1_plain", help="prompt version in prompts/ (not used by current)")
    p.add_argument("--split", choices=list(scoring.SPLITS), default="dev")
    p.add_argument("--device", help="cuda (default) or cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="only the first N cases")
    p.add_argument("--out", type=Path, default=RUNS, help="where run folders go")
    args = p.parse_args(argv)
    if args.split == "holdout":
        print("WARNING: running the HOLDOUT split. Use it only to report results after the pass bar is set, "
              "never to tune prompts or train.")
    cases = scoring.load_cases(split=args.split)
    if args.limit:
        cases = cases[:args.limit]
    prompt = None if args.backend == "current" else load_prompt(args.prompt)
    folder = run(make_backend(args), cases, prompt, args.out, split=args.split)
    print(f"Done: {folder / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
