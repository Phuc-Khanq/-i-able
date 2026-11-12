"""Build the simulated-dysarthria experiment: simulated audio and data/voice/manifests/sim.csv.

    python -m engine.voice.finetune.get_data --train-clean-100     # once: LibriSpeech train-clean-100 (6.3 GB)
    python -m engine.voice.finetune.prepare_sim --hours 20         # simulate 20 h of it, write sim.csv

The question (docs/DECISIONS.md, 2026-11-10): does Whisper hear dysarthric speech better after training on clear
speech made to sound dysarthric (simulate.py)? Two runs answer it, on the same sentences:
  control   main_sources sim_clean   the LibriSpeech clips as they are
  sim       main_sources sim         the same clips, simulated
Both mix in LibriSpeech dev-clean (the train half, as prepare.py does) and pick checkpoints on simulated dev
clips. TORGO is ONLY in the test split here (F04, M01): nothing trains or picks checkpoints on it.

sim.csv rows:
  train  sim        train-clean-100 clips, simulated (data/voice/simulated/v2/train-clean-100/...)
  train  sim_clean  the same clips, clean
  train  librispeech  dev-clean, train half (the typical-speech mix)
  dev    sim        dev-clean, dev half, simulated
  dev    librispeech  dev-clean, dev half, clean
  test   torgo      F04 and M01 (dys_unseen / dys_seen / dys_words, as in all.csv)
  test   librispeech  test-clean
  test   personal   the personal pack's test sentences, if there is a pack
Simulation is fixed per clip (seeded by its ID), so a rerun skips clips already made and gives the same data.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from engine.voice.finetune import common, prepare
from engine.voice.finetune.common import Clip

SOURCE_SUBSET = "train-clean-100"
MIN_SOURCE_S = 1.0
MAX_SOURCE_S = 14.0          # a severe clip comes out up to about 2.7x as long; past Whisper's 30 s it is made milder


def sim_dir(data_dir: Path) -> Path:
    """data/voice/simulated/v<version>: a new simulator version never reuses an older version's clips."""
    from engine.voice.finetune import simulate

    return Path(data_dir) / "simulated" / f"v{simulate.VERSION}"


def choose(by_speaker: dict, hours: float, seed: int) -> list[tuple[str, Path, str, float]]:
    """About `hours` of clips (each 1 to 14 s), taken in turn from every speaker so no voice dominates:
    [(speaker, flac, text, seconds)]."""
    rng = random.Random(seed)
    queues = {}
    for spk in sorted(by_speaker):
        items = list(by_speaker[spk])
        rng.shuffle(items)
        queues[spk] = items
    budget, total, picked = hours * 3600, 0.0, []
    while total < budget and any(queues.values()):
        for spk in sorted(queues):
            while queues[spk]:
                flac, text = queues[spk].pop()
                secs = prepare._seconds(flac)
                if secs is not None and MIN_SOURCE_S <= secs <= MAX_SOURCE_S:
                    picked.append((spk, flac, text, round(secs, 2)))
                    total += secs
                    break
            if total >= budget:
                break
    return picked


def _simulate_one(job: tuple[str, str, int]) -> tuple[str, str, str, float | None]:
    """(source flac, target flac, seed) -> (target, severity, effects, seconds or None if too long). A clip that
    comes out over Whisper's 30 s is made again one severity milder (severe -> moderate -> mild), so severe lands on
    the shorter sentences, as it does for people who speak with severe dysarthria. Runs in a worker process."""
    import numpy as np
    import soundfile as sf

    from engine.voice.finetune import simulate

    src, dst, seed = job
    rng = np.random.default_rng(seed)
    severity = str(rng.choice(simulate.SEVERITIES))
    audio = common.load_audio(Path(src))
    while True:
        out, used = simulate.simulate(audio, severity, rng)
        secs = len(out) / common.SAMPLE_RATE
        if secs <= common.MAX_CLIP_S:
            break
        milder = simulate.SEVERITIES.index(severity) - 1
        if milder < 0:
            return dst, severity, "; ".join(used), None
        severity = simulate.SEVERITIES[milder]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(dst).with_suffix(".tmp.flac")
    sf.write(str(tmp), out, common.SAMPLE_RATE)
    tmp.replace(dst)                                  # a stopped run never leaves a half-written clip
    return dst, severity, "; ".join(used), round(secs, 2)


def clip_seed(seed: int, clip_id: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(clip_id.encode("utf-8"))) % (2 ** 32)


def simulate_all(jobs: list[tuple[Path, Path]], seed: int, workers: int, log) -> dict[str, tuple[str, str, float]]:
    """Make every missing simulated clip; return {target path: (severity, effects, seconds)} for the ones that
    exist. Effects are kept in effects.csv next to the clips."""
    if not jobs:
        return {}
    table = sim_dir_of(jobs[0][1]) / "effects.csv"
    known: dict[str, tuple[str, str, float]] = {}
    if table.exists():
        with open(table, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                known[r["path"]] = (r["severity"], r["effects"], float(r["seconds"]))
    todo = [(str(src), str(dst), clip_seed(seed, dst.stem)) for src, dst in jobs
            if not (dst.exists() and str(dst) in known)]
    log(f"Simulating {len(todo)} clips ({len(jobs) - len(todo)} already made) with {workers} workers")
    dropped = 0
    if todo:
        done = 0
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        results = pool.map(_simulate_one, todo, chunksize=8) if pool else map(_simulate_one, todo)
        try:
            for dst, severity, effects, secs in results:
                done += 1
                if secs is None:
                    dropped += 1
                else:
                    known[dst] = (severity, effects, secs)
                if done % 500 == 0:
                    log(f"  {done}/{len(todo)}")
                    _write_effects(table, known)
        finally:
            if pool:
                pool.shutdown()
        _write_effects(table, known)
    if dropped:
        log(f"Dropped {dropped} clips that came out over {common.MAX_CLIP_S:.0f} s")
    return {p: v for p, v in known.items() if Path(p).exists()}


def sim_dir_of(path: Path) -> Path:
    """data/voice/simulated/v<version>, from a clip path under it."""
    for parent in path.parents:
        if parent.parent.name == "simulated":
            return parent
    return path.parent


def _write_effects(table: Path, known: dict) -> None:
    table.parent.mkdir(parents=True, exist_ok=True)
    tmp = table.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "severity", "effects", "seconds"])
        for p, (sev, eff, secs) in sorted(known.items()):
            w.writerow([p, sev, eff, secs])
    tmp.replace(table)


def build(data_dir: Path, hours: float, seed: int, workers: int, log) -> list[Clip]:
    data_dir = Path(data_dir).resolve()
    split = common.load_split()
    stats: Counter = Counter()
    libri = data_dir / "librispeech"
    source = prepare._libri_subset(libri, SOURCE_SUBSET)
    if not source:
        raise SystemExit(f"No LibriSpeech {SOURCE_SUBSET} in {libri}. "
                         "Run: python -m engine.voice.finetune.get_data --train-clean-100")
    base = prepare.scan_librispeech(data_dir, split, stats)         # test-clean, dev-clean halves
    picks = choose(source, hours, seed)
    out_root = sim_dir(data_dir)

    def target(flac: Path, subset: str) -> Path:
        return out_root / subset / flac.relative_to(libri / subset)

    dev_clean = [c for c in base if c.split == "dev"]
    jobs = [(flac, target(flac, SOURCE_SUBSET)) for _, flac, _, _ in picks]
    jobs += [(c.audio_path(data_dir), target(c.audio_path(data_dir), "dev-clean")) for c in dev_clean]
    made = simulate_all(jobs, seed, workers, log)

    clips: list[Clip] = list(base)
    for spk, flac, text, secs in picks:
        sim = made.get(str(target(flac, SOURCE_SUBSET)))
        if sim:                       # only pairs: both runs train on exactly the same sentences
            clips.append(Clip(prepare._rel(flac, data_dir), text, spk, "sim_clean", "train", "sim_clean", secs))
            clips.append(Clip(prepare._rel(target(flac, SOURCE_SUBSET), data_dir), text, spk, "sim", "train", "sim",
                              sim[2]))
    for c in dev_clean:
        sim = made.get(str(target(c.audio_path(data_dir), "dev-clean")))
        if sim:
            clips.append(Clip(prepare._rel(target(c.audio_path(data_dir), "dev-clean"), data_dir), c.text,
                              c.speaker, "sim", "dev", "sim", sim[2]))
    clips += [c for c in prepare.scan_torgo(data_dir, split, stats) if c.split == "test"]
    clips += [c for c in prepare.scan_personal(data_dir, split, stats) if c.split == "test"]
    check_sim(clips)
    for k, v in sorted(stats.items()):
        log(f"{k}: {v}")
    return clips


def check_sim(clips: list[Clip]) -> None:
    """TORGO only in test; no speaker in two splits (also across sources: a train-clean-100 voice is never in the
    dev or test speech)."""
    bad = [c for c in clips if c.source == "torgo" and c.split != "test"]
    if bad:
        raise SystemExit(f"TORGO outside the test split: {bad[0].path}")
    prepare.check_disjoint(clips)
    libri_like = ("librispeech", "sim", "sim_clean")
    where: dict[str, str] = {}
    for c in clips:
        if c.source in libri_like:
            if where.setdefault(c.speaker, c.split) != c.split:
                raise SystemExit(f"Split leak: LibriSpeech speaker {c.speaker} is in {where[c.speaker]} and {c.split}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(p)
    p.add_argument("--hours", type=float, default=20.0, help="hours of train-clean-100 to simulate (default 20)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    a = p.parse_args(argv)
    log = common.Log()
    clips = build(a.data_dir, a.hours, a.seed, a.workers, log)
    out = common.manifest_path(Path(a.data_dir), name="sim")
    common.write_manifest(out, clips)
    prepare.summarize(clips, log)
    log(f"Wrote {len(clips)} clips to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
