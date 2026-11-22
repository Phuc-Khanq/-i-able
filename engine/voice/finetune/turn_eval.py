"""Can a smart end-of-turn model end utterances sooner without cutting dysarthric sentences? (Smart Turn v3)

    python -m engine.voice.finetune.turn_eval --set torgo-dev
    python -m engine.voice.finetune.turn_eval --set torgo-all
    python -m engine.voice.finetune.turn_eval --rows checkpoints/turn/torgo-all.jsonl     # the numbers again

The app ends an utterance after `pause_seconds` of silence (Silero VAD in audio_in.py). 1.0 s (Khang's PC) cut 22 of
TORGO dev's 108 sentences into pieces; 2.5 s cuts none, but then every command and every "yes" waits 2.5 s before
anything happens. Smart Turn v3 (Pipecat; BSD-2-Clause on its model card; an 8 MB ONNX model on the CPU, a Whisper Tiny
encoder with a classifier) says from the audio alone whether the speaker has finished.

For every clip this runs Silero over it the way audio_in does (512-sample blocks, start at the threshold, silence below
it minus 0.15) and, at every pause inside the speech and at the end, asks Smart Turn "finished?" about the audio so far
plus QUERY_AFTER_S of the silence (Pipecat asks after 0.2 s). Then it simulates end rules on those answers: a fixed
pause (today), and "end after a short pause if Smart Turn says finished, otherwise after the long pause". CPU only.
Rows (numbers and reference text, no audio) to checkpoints/turn/<set>.jsonl. Model file: downloads/smart-turn/
smart-turn-v3.2-cpu.onnx (git-ignored; https://huggingface.co/pipecat-ai/smart-turn-v3).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from engine.voice.finetune import common

SETS = {
    "torgo-dev": ("dev", "torgo"),
    "torgo-test": ("test", "torgo"),
    "torgo-all": (None, "torgo"),        # stock models never saw any TORGO speaker; only scored here
    "libri-dev": ("dev", "librispeech"),
    "personal-all": (None, "personal"),  # Khang's voice, his own mic: commands and sentences
}
SMART_TURN = common.REPO_ROOT / "downloads" / "smart-turn" / "smart-turn-v3.2-cpu.onnx"
RATE = 16000
BLOCK = 512                      # audio_in.BLOCK: 32 ms
QUERY_AFTER_S = 0.2              # Smart Turn hears this much of the pause (Pipecat's VAD stop_secs)
FIXED = (0.5, 0.7, 1.0, 1.5, 2.0, 2.5)
SHORT = (0.3, 0.5, 0.7)          # smart rule: end this soon when Smart Turn says finished...
LONG = (1.5, 2.5, 3.0)           # ...else at this pause
CUTOFFS = (0.3, 0.5, 0.7, 0.9)   # Smart Turn's "finished" probability needed


class SmartTurn:
    """Smart Turn v3's "finished?" probability for audio so far, exactly as its inference.py prepares it: the last
    8 s (zeros added at the start when shorter), Whisper features with the waveform normalized."""

    def __init__(self, path: Path = SMART_TURN) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        if not Path(path).exists():
            raise SystemExit(f"No Smart Turn model at {path}. Download smart-turn-v3.2-cpu.onnx from "
                             "https://huggingface.co/pipecat-ai/smart-turn-v3 into downloads/smart-turn/")
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=8)
        self.ms: list[float] = []

    def __call__(self, audio) -> float:
        import numpy as np

        n = 8 * RATE
        audio = np.asarray(audio, dtype=np.float32)
        audio = audio[-n:] if len(audio) > n else np.pad(audio, (n - len(audio), 0))
        t0 = time.perf_counter()
        x = self.features(audio, sampling_rate=RATE, return_tensors="np", padding="max_length", max_length=n,
                          truncation=True, do_normalize=True).input_features.astype(np.float32)
        p = float(self.session.run(None, {"input_features": x})[0].reshape(-1)[0])
        self.ms.append(1000 * (time.perf_counter() - t0))
        return p


def speech_probs(audio) -> list[float]:
    """Silero's speech probability for each 512-sample block, as audio_in feeds it."""
    import numpy as np

    from engine.voice.audio_in import SileroOnnx

    vad = SileroOnnx()
    audio = np.asarray(audio, dtype=np.float32)
    return [vad(audio[i:i + BLOCK]) for i in range(0, len(audio) - BLOCK + 1, BLOCK)]


def pauses(probs: list[float], threshold: float = 0.5) -> tuple[int | None, list[tuple[int, int]], int | None]:
    """(first speech block, [(start block, blocks)] of each pause between speech, start block of the silence that runs
    to the end or None). audio_in's rules: speech from a block at the threshold, a pause from a block under the
    threshold minus 0.15 until the next block at the threshold."""
    first, inside, quiet = None, [], None
    for i, p in enumerate(probs):
        if p >= threshold:
            if first is None:
                first = i
            if quiet is not None:
                inside.append((quiet, i - quiet))
                quiet = None
        elif p < threshold - 0.15 and first is not None and quiet is None:
            quiet = i
    return first, inside, quiet


def clip_row(c, audio, turn: SmartTurn) -> dict | None:
    import numpy as np

    probs = speech_probs(audio)
    first, inside, end = pauses(probs)
    if first is None:
        return None                                   # Silero never heard speech: nothing to end
    start = max(0, first * BLOCK - int(0.3 * RATE))   # audio_in's pre-roll
    after = int(QUERY_AFTER_S * RATE)

    def ask(silence_block: int) -> float:
        stop = silence_block * BLOCK + after
        piece = audio[start:stop]
        if stop > len(audio):                         # the clip ends sooner: the rest of the pause is silence
            piece = np.concatenate([piece, np.zeros(stop - len(audio), np.float32)])
        return round(turn(piece), 4)

    rows = [[round(n * BLOCK / RATE, 3), ask(b)] for b, n in inside if n * BLOCK / RATE >= QUERY_AFTER_S]
    end_block = end if end is not None else len(probs)
    return {"path": c.path, "speaker": c.speaker, "group": c.group, "ref": c.text,
            "seconds": round(len(audio) / RATE, 2), "pauses": rows, "p_end": ask(end_block),
            "longest_pause": max((p for p, _ in rows), default=0.0)}


def summarize(rows: list[dict], log, name: str) -> dict:
    """Cut sentences and the wait at the true end, per rule."""
    out = {"clips": len(rows)}
    if not rows:
        return out
    ends = [r["p_end"] for r in rows]
    out["p_end_median"] = round(statistics.median(ends), 3)
    inside = [p for r in rows for _, p in r["pauses"]]
    out["pauses"] = len(inside)
    out["p_pause_median"] = round(statistics.median(inside), 3) if inside else None
    log(f"--- {name}: {len(rows)} clips, {len(inside)} pauses inside the speech; Smart Turn's 'finished' median "
        f"{out['p_pause_median']} at pauses, {out['p_end_median']} at the end")
    rules = {}
    for f in FIXED:
        cut = sum(1 for r in rows if any(n >= f for n, _ in r["pauses"]))
        rules[f"fixed {f:g} s"] = {"cut": cut, "wait_mean": f, "fast_ends": None}
    for s in SHORT:
        for long_ in LONG:
            for c in CUTOFFS:
                cut = sum(1 for r in rows if any((n >= s and p >= c) or n >= long_ for n, p in r["pauses"]))
                fast = sum(1 for r in rows if r["p_end"] >= c)
                wait = (fast * s + (len(rows) - fast) * long_) / len(rows)
                rules[f"smart {s:g}/{long_:g} s at {c:g}"] = {"cut": cut, "wait_mean": round(wait, 2),
                                                               "fast_ends": fast}
    for k, v in rules.items():
        fast = "" if v["fast_ends"] is None else f", ended fast {pct(v['fast_ends'], len(rows))}"
        log(f"   {k:24s} cut {pct(v['cut'], len(rows))}, mean wait at the end {v['wait_mean']:.2f} s{fast}")
    out["rules"] = rules
    return out


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


def report(rows: list[dict], log) -> dict:
    out = {}
    groups = {}
    for r in rows:
        key = ("sentences" if r["group"] in ("dys_seen", "dys_unseen") else
               "single words" if r["group"] == "dys_words" else r["group"])
        groups.setdefault(key, []).append(r)
    for key, sel in sorted(groups.items()):
        out[key] = summarize(sel, log, key)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(p)
    p.add_argument("--set", default="torgo-dev", choices=sorted(SETS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rows", type=Path, default=None, help="instead: the numbers again from saved rows")
    a = p.parse_args(argv)
    log = common.Log()
    if a.rows:
        rows = [json.loads(line) for line in open(a.rows, encoding="utf-8")]
        common.save_json(a.rows.with_suffix(".summary.json"), report(rows, log))
        return 0
    split, source = SETS[a.set]
    clips = sorted((c for c in common.read_manifest(common.manifest_path(Path(a.data_dir)))
                    if split in (None, c.split) and c.source == source), key=lambda c: c.path)
    clips = clips[:a.limit] if a.limit else clips
    turn = SmartTurn()
    dest = common.REPO_ROOT / "checkpoints" / "turn" / f"{a.set}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows, silent = [], 0
    log(f"{len(clips)} clips from {a.set}; rows to {dest}")
    with open(dest, "w", encoding="utf-8") as f:
        for i, c in enumerate(clips):
            row = clip_row(c, common.load_audio(c.audio_path(Path(a.data_dir))), turn)
            if row is None:
                silent += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(clips)}")
    log(f"Silero heard no speech in {silent} clips; Smart Turn p50 {statistics.median(turn.ms):.1f} ms per question "
        f"on the CPU ({len(turn.ms)} questions)")
    summary = report(rows, log)
    summary["smart_turn_ms_p50"] = round(statistics.median(turn.ms), 1)
    common.save_json(dest.with_suffix(".summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
