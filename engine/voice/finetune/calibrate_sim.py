"""How hard is the simulated dysarthric speech for stock Whisper? Word error rate per severity, all in memory.

    python -m engine.voice.finetune.calibrate_sim                    # 120 clips, clean and each severity
    python -m engine.voice.finetune.calibrate_sim --only slur        # one effect alone, at each severity

The same LibriSpeech dev-clean clips go through every severity, so the numbers differ only by the simulation.
Decoding is the app's (evaluate.CT2Backend: faster-whisper, greedy). Nothing is written but the printed table.

Why: stock Whisper missed 2.7% of words on the first simulation and about 41% on TORGO's dysarthric test speakers
(2026-11-11): speech it already hears can't teach it much. The aim is roughly 5 to 15% (mild), 15 to 35% (moderate)
and 35 to 65% (severe).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from engine.voice.finetune import common, simulate


def _job(job: tuple[str, str, int, str | None]):
    """(flac, severity, seed, only) -> (simulated audio, effects). Runs in a worker process."""
    import soundfile as sf

    flac, severity, seed, only = job
    audio, _ = sf.read(flac, dtype="float32")
    return simulate.simulate(audio, severity, np.random.default_rng(seed), only=only)


def pick(source: Path, n: int, seed: int, min_s: float = 3.0, max_s: float = 12.0) -> list[tuple[Path, str]]:
    import soundfile as sf

    clips = simulate.librispeech_clips(source)
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(clips)):
        flac, text = clips[int(i)]
        if min_s <= sf.info(str(flac)).duration <= max_s:
            out.append((flac, text))
        if len(out) >= n:
            break
    return out


def run(source: Path, n: int, seed: int, only: str | None, workers: int, model: str, log=print) -> dict:
    from engine.voice.finetune.evaluate import CT2Backend

    clips = pick(source, n, seed)
    backend = CT2Backend(model, "cuda", "float16")
    texts = [t for _, t in clips]
    result = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for severity in ("clean",) + simulate.SEVERITIES:
            t0 = time.perf_counter()
            if severity == "clean":
                audios = [common.load_audio(f) for f, _ in clips]
                effects = [[] for _ in clips]
            else:
                jobs = [(str(f), severity, seed * 7919 + k, only) for k, (f, _) in enumerate(clips)]
                made = list(pool.map(_job, jobs, chunksize=4))
                audios, effects = [a for a, _ in made], [u for _, u in made]
            hyps = []
            for k in range(0, len(audios), 16):
                hyps += backend.transcribe_many(audios[k:k + 16])
            table = common.wer_table(list(zip(texts, hyps)))
            secs = sum(len(a) for a in audios) / common.SAMPLE_RATE
            per_clip = [common.wer_table([(texts[i], hyps[i])])["wer"] or 0.0 for i in range(len(texts))]
            worst = sorted(range(len(texts)), key=lambda i: -per_clip[i])[:5]
            result[severity] = {**table, "minutes": round(secs / 60, 1),
                                "clean_clips": sum(1 for w in per_clip if w == 0),
                                "examples": [(texts[i], hyps[i], "; ".join(effects[i]), per_clip[i]) for i in worst]}
            log(f"{severity:9} WER {table['wer']:6.2f}%  substituted {table['sub']}, missed {table['del']}, "
                f"added {table['ins']} of {table['words']} words; {result[severity]['clean_clips']}/{len(texts)} "
                f"clips perfect ({secs / 60:.1f} min of audio, {time.perf_counter() - t0:.0f} s)")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(p)
    p.add_argument("--clips", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", default=None, help=f"one effect alone: {', '.join(simulate.EFFECTS)}")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=common.BASE_CT2)
    p.add_argument("--examples", action="store_true", help="print the 5 worst transcripts per severity")
    a = p.parse_args(argv)
    if a.only and a.only not in simulate.EFFECTS:
        raise SystemExit(f"--only must be one of {', '.join(simulate.EFFECTS)}")
    result = run(Path(a.data_dir) / "librispeech" / "dev-clean", a.clips, a.seed, a.only, a.workers, a.model)
    if a.examples:
        for sev, r in result.items():
            for ref, hyp, eff, wer in r["examples"]:
                print(f"[{sev}] {wer:.0f}% wrong | {eff}\n  said:  {ref.lower()}\n  heard: {hyp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
