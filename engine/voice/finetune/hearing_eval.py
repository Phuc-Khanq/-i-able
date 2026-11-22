"""What Whisper's other readings and the text before the cursor could add (hearings plan step 0; Wispr Flow review).

    python -m engine.voice.finetune.hearing_eval --set torgo-dev     # TORGO dev speaker M05 (470 clips)
    python -m engine.voice.finetune.hearing_eval --set libri-dev     # LibriSpeech dev half (298 clips)
    python -m engine.voice.finetune.hearing_eval --set torgo-test    # F04 and M01 (801 clips): report only

For every clip, with stock large-v3-turbo (or --model) on the GPU, the same audio is read four ways:
  greedy   the app's pass today (Dictation.transcribe: beam 1, word confidences)
  beam     beam 5, best reading                     } one pass: Dictation.hearings(), the top 3 of a beam of 5
  hearings beam 5, readings 2 and 3                 }
  prompt   greedy, with the previous clip's right text as Whisper's prompt ("the text before the cursor").
           LibriSpeech: the previous sentence of the same chapter (text that leads in). TORGO: the previous prompt
           of the same session, which is unrelated (does a wrong-topic prompt hurt?).
It prints WER for each, the oracle (the best of greedy and the 3 readings, per clip), how often greedy is wrong and a
reading is exactly right, the time of the extra pass, and what a confidence cut-off would catch. One row per clip
(text only, no audio) goes to checkpoints/hearings/<set>.jsonl, which git ignores. TORGO is only scored here, never
trained on (docs/DECISIONS.md, 2026-11-10).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from engine.voice.finetune import common

SETS = {
    "torgo-dev": ("dev", "torgo"),
    "torgo-test": ("test", "torgo"),
    "libri-dev": ("dev", "librispeech"),
    "libri-test": ("test", "librispeech"),
    "personal-test": ("test", "personal"),                     # the personal pack (Khang's voice, his own mic)
    "torgo-all": (None, "torgo"),        # every TORGO speaker (stock Whisper never saw any): --closed-test only
}
CUTOFFS = (0.3, 0.5, 0.7, 0.8, 0.9)
PROMPT_CHARS = 300          # the text before the cursor: the last sentences, cut at a word


def pick(clips, split: str, source: str, limit: int | None):
    rows = sorted((c for c in clips if split in (None, c.split) and c.source == source), key=lambda c: c.path)
    return rows[:limit] if limit else rows


def context_key(clip) -> str:
    """Clips that follow each other: a LibriSpeech chapter, a TORGO session (and microphone) folder."""
    return str(Path(clip.path).parent)


def errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word errors, reference words) after Whisper's normalizer."""
    import jiwer

    r, h = common.normalize(ref), common.normalize(hyp)
    if not r.strip():
        return 0, 0
    out = jiwer.process_words(r, h)
    return out.substitutions + out.deletions + out.insertions, out.hits + out.substitutions + out.deletions


def wer(rows, key: str) -> float | None:
    e = sum(r[key + "_err"] for r in rows)
    n = sum(r["words"] for r in rows)
    return round(100.0 * e / n, 2) if n else None


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


def summarize(rows: list[dict], log) -> dict:
    rows = [r for r in rows if r["words"]]
    out = {"clips": len(rows), "words": sum(r["words"] for r in rows)}
    for key in ("greedy", "raw", "b5", "b5raw", "beam", "oracle", "prompt"):
        if all(key + "_err" in r for r in rows):
            out[f"wer_{key}"] = wer(rows, key)
    if rows and "raw" in rows[0]:
        out["greedy_empty"] = sum(1 for r in rows if not r["greedy"].strip())
        out["raw_empty"] = sum(1 for r in rows if not r["raw"].strip())
        out["exact_greedy"] = sum(1 for r in rows if r["greedy_err"] == 0)
        out["exact_raw"] = sum(1 for r in rows if r["raw_err"] == 0)
        out["exact_beam"] = sum(1 for r in rows if r["beam_err"] == 0)
        log(f"empty results: greedy {out['greedy_empty']}, without the noise filters {out['raw_empty']}; exactly "
            f"right: greedy {out['exact_greedy']}, without filters {out['exact_raw']}, beam-5 {out['exact_beam']}")
    wrong = [r for r in rows if r["greedy_err"]]
    fixable = [r for r in wrong if r["oracle_err"] == 0]
    better = [r for r in wrong if r["oracle_err"] < r["greedy_err"]]
    out.update(greedy_wrong=len(wrong), reading_exactly_right=len(fixable), reading_fewer_errors=len(better))
    extra = [r["hearings_ms"] for r in rows]
    base = [r["greedy_ms"] for r in rows]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    out.update(greedy_ms_p50=q(base, 0.5), hearings_ms_p50=q(extra, 0.5), hearings_ms_p90=q(extra, 0.9))
    out["cutoffs"] = {}
    for c in CUTOFFS:
        hit = [r for r in rows if r["min_conf"] < c]
        out["cutoffs"][str(c)] = {"runs": len(hit), "catches_better": sum(1 for r in better if r["min_conf"] < c),
                                  "greedy_right_but_run": sum(1 for r in hit if not r["greedy_err"])}
    log(f"clips {out['clips']}, words {out['words']}")
    if rows and "b5_ms" in rows[0]:
        b5 = [r["b5_ms"] for r in rows]
        out.update(b5_ms_p50=q(b5, 0.5), b5_ms_p90=q(b5, 0.9),
                   exact_b5=sum(1 for r in rows if r["b5_err"] == 0),
                   exact_b5raw=sum(1 for r in rows if r["b5raw_err"] == 0))
        log(f"app pass with beam 5: WER {out.get('wer_b5')} (without filters {out.get('wer_b5raw')}), exactly right "
            f"{out['exact_b5']} ({out['exact_b5raw']} without filters), time p50 {out['b5_ms_p50']} ms, "
            f"p90 {out['b5_ms_p90']} ms (greedy p50 {out['greedy_ms_p50']} ms)")
    log(f"WER greedy (app today) {out['wer_greedy']}  without noise filters {out.get('wer_raw')}  "
        f"beam-5 best {out['wer_beam']}  "
        f"oracle of greedy + 3 readings {out['wer_oracle']}  greedy + previous-text prompt {out['wer_prompt']}")
    log(f"greedy wrong on {pct(len(wrong), len(rows))} clips; a reading is exactly right on {pct(len(fixable), len(wrong))}"
        f" of those, and has fewer errors on {pct(len(better), len(wrong))}")
    log(f"time: greedy p50 {out['greedy_ms_p50']} ms; extra pass for 3 readings p50 {out['hearings_ms_p50']} ms, "
        f"p90 {out['hearings_ms_p90']} ms")
    for c, v in out["cutoffs"].items():
        log(f"  run the extra pass when a word is under {c}: on {pct(v['runs'], len(rows))} clips, catching "
            f"{pct(v['catches_better'], len(better))} of the clips a reading improves "
            f"({v['greedy_right_but_run']} runs where greedy was already right)")
    return out


HOT_TEST_WORDS = ("Hathaway, prednisone, AFO, Khoa Nhi, Gainesville, Alvarez, gabapentin, baclofen, Medicaid, "
                  "occupational therapy, Hoyer lift, BiPAP, Shands, Riluzole, Edaravone, Tobii, Venus, Discord, "
                  "Spotify, Nguyen, Truong, Kowalski, Okonkwo, Siobhan, Zoom, Teams, Gmail, Outlook, PowerPoint, "
                  "Excel")


def hotwords_test(clips, data_dir: Path, model, log, words: str = HOT_TEST_WORDS) -> dict:
    """Does a word list the user isn't saying cost accuracy? The app's pass with and without these hotwords (a
    plausible personal list of names and terms, none of them in the test sentences)."""
    import tempfile

    from engine.voice.dictation import Dictation, words_to_text
    from engine.voice.personal_words import PersonalWords

    tmp = tempfile.TemporaryDirectory()
    pw = PersonalWords(Path(tmp.name) / "words.txt")
    for w in words.split(","):
        pw.add(w)
    plain = Dictation(model=model)
    app = Dictation(model=model, personal_words=pw)              # the app's path for the user's word list
    rows = []
    for c in clips:
        audio = common.load_audio(c.audio_path(data_dir))
        rows.append((c, *(words_to_text(d.transcribe(audio)) for d in (plain, app))))
    tmp.cleanup()
    listed = {w.strip().lower() for w in words.split(",")}
    out = {}
    for g in sorted({r[0].group for r in rows}) + ["all"]:
        sel = [r for r in rows if g == "all" or r[0].group == g]
        out[g] = {"clips": len(sel)}
        parts = []
        for i, name in ((1, "no list"), (2, "with the list")):
            w = common.wer_table([(r[0].text, r[i]) for r in sel])["wer"]
            leaked = sum(1 for r in sel for x in common.words(r[i].lower()) if x in listed)
            out[g][name] = {"wer": w, "list_words_written": leaked}
            parts.append(f"{name} {w} (list words {leaked})")
        log(f"{g} ({len(sel)} clips): " + "; ".join(parts))
    return out


VARIANTS = {
    "app": {},                                                              # the app's pass as it is now (DECODE)
    "fallback": {"log_prob_threshold": -1.0},                               # before 2026-11-19: re-decode when unsure
    "t0": {"temperature": 0.0},                                             # no fallback
    "t0_notok": {"temperature": 0.0, "without_timestamps": True},           # ...and no timestamp tokens
    "b5_t0_notok": {"beam_size": 5, "temperature": 0.0, "without_timestamps": True},
    "b5_notok": {"beam_size": 5, "without_timestamps": True},               # beam 5, fallback kept
    "b5_t0_notok_keep": {"beam_size": 5, "temperature": 0.0, "without_timestamps": True, "keep_short": True},
    # without_timestamps inside transcribe() repeats text (TORGO dev sentences 26% -> 64%); these keep timestamps:
    "t0_keep": {"temperature": 0.0, "keep_short": True},
    "b5_t0": {"beam_size": 5, "temperature": 0.0},
    "b5_t0_keep": {"beam_size": 5, "temperature": 0.0, "keep_short": True},
    # the fallback only for repetition loops (compression ratio), not for "unsure" (log probability)
    "nolp": {"log_prob_threshold": None},
    "nolp_keep": {"log_prob_threshold": None, "keep_short": True},
    "b5_nolp": {"beam_size": 5, "log_prob_threshold": None},
    "b5_nolp_keep": {"beam_size": 5, "log_prob_threshold": None, "keep_short": True},
}


def variants_test(clips, data_dir: Path, model, log, names=tuple(VARIANTS)) -> dict:
    """The app's pass with different decoding options (all with word timestamps, as the mouth gate needs them):
    WER per group, exactly right clips, empty results and time."""
    from engine.voice.dictation import Dictation, words_to_text

    d = Dictation(model=model)
    rows = []
    for c in clips:
        audio = common.load_audio(c.audio_path(data_dir))
        row = {"group": c.group, "ref": c.text}
        for name in names:
            t0 = time.perf_counter()
            row[name] = words_to_text(d.transcribe(audio, **VARIANTS[name]))
            row[name + "_ms"] = round(1000 * (time.perf_counter() - t0))
        rows.append(row)
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["all"]:
        sel = [r for r in rows if g == "all" or r["group"] == g]
        out[g] = {}
        parts = []
        for name in names:
            w = common.wer_table([(r["ref"], r[name]) for r in sel])["wer"]
            exact = sum(1 for r in sel if common.normalize(r[name]) == common.normalize(r["ref"]))
            empty = sum(1 for r in sel if not r[name].strip())
            ms = sorted(r[name + "_ms"] for r in sel)
            out[g][name] = {"wer": w, "exact": exact, "empty": empty, "ms_p50": ms[len(ms) // 2],
                            "ms_p90": ms[min(len(ms) - 1, int(0.9 * len(ms)))]}
            parts.append(f"{name} {w} (exact {exact}, empty {empty}, p50 {ms[len(ms) // 2]} ms)")
        log(f"{g} ({len(sel)} clips): " + "; ".join(parts))
    return out


def context_test(clips, data_dir: Path, model, log, max_words: int = 40) -> dict:
    """Would names on screen help (Wispr Flow's context awareness)? For each clip, the rare words (4+ letters, more
    than one Whisper token) of the OTHER clips of the same chapter or session go to Whisper as hotwords, like names
    read off the screen. Reports WER without and with them, and how often this clip's own rare words that were in the
    list come out right."""
    import re
    from collections import defaultdict

    from engine.voice.dictation import Dictation, words_to_text

    tok = model.hf_tokenizer

    def rare(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']+", common.normalize(text)) if len(w) >= 4
                and len(tok.encode(" " + w).ids) > 1}

    by_key = defaultdict(list)
    for c in clips:
        by_key[context_key(c)].append(c)
    plain = Dictation(model=model)
    rows = []
    for c in clips:
        others = set().union(*(rare(o.text) for o in by_key[context_key(c)] if o is not c))
        words = sorted(others)[:max_words]
        audio = common.load_audio(c.audio_path(data_dir))
        a = words_to_text(plain.transcribe(audio))
        b = words_to_text(Dictation(model=model, hotwords=", ".join(words)).transcribe(audio)) if words else a
        target = rare(c.text) & set(words)
        rows.append((c, a, b, target, len(words)))
    out = {}
    for g in sorted({r[0].group for r in rows}) + ["all"]:
        sel = [r for r in rows if g == "all" or r[0].group == g]
        wa = common.wer_table([(r[0].text, r[1]) for r in sel])["wer"]
        wb = common.wer_table([(r[0].text, r[2]) for r in sel])["wer"]
        n = sum(len(r[3]) for r in sel)
        hit = lambda i: sum(1 for r in sel for w in r[3] if w in common.normalize(r[i]).split())
        out[g] = {"clips": len(sel), "wer_plain": wa, "wer_context": wb, "context_words_said": n,
                  "right_plain": hit(1), "right_context": hit(2)}
        log(f"{g} ({len(sel)} clips): WER without context words {wa}, with {wb}; this clip's own rare words that "
            f"were in the list: {n}, right {hit(1)} without, {hit(2)} with")
    return out


def vad_test(clips, data_dir: Path, model, log, pause: float | None = None, threshold: float | None = None,
             min_speech: float | None = None, max_gain: float | None = None) -> dict:
    """Does the app's front door (audio_in.py: auto gain, Silero VAD, pre-roll, pads) lose speech before Whisper?
    Each clip goes through AudioIn as if from the mic (clips of one session in a row, so the gain adapts), with
    silence after it to end the utterance. Compares the app's pass on what AudioIn cut against the whole clip.
    `pause`: the user's pause setting in seconds (default: settings.pause_seconds); `threshold`: Silero's speech
    threshold (default audio_in.VAD_THRESHOLD); `min_speech`: the shortest utterance kept (audio_in.MIN_SPEECH_S)."""
    import copy
    import itertools

    import numpy as np

    from engine.config import settings as base

    settings = copy.copy(base)
    if pause:
        settings.pause_seconds = pause
    from engine.voice import audio_in as _ain

    old_min, old_gain = _ain.MIN_SPEECH_S, _ain.MAX_GAIN
    if min_speech is not None:
        _ain.MIN_SPEECH_S = min_speech
    if max_gain is not None:
        _ain.MAX_GAIN = max_gain
    thr = threshold or _ain.VAD_THRESHOLD
    log(f"pause_seconds {settings.pause_seconds}, VAD threshold {thr}, shortest utterance {_ain.MIN_SPEECH_S} s, "
        f"most gain {_ain.MAX_GAIN}x")
    from engine.core import topics
    from engine.voice import audio_in
    from engine.voice.dictation import Dictation, words_to_text

    class Bus:
        def __init__(self):
            self.utterances = []

        def subscribe(self, topic, handler):
            pass

        def publish(self, topic, payload):
            if topic == topics.UTTERANCE:
                self.utterances.append(payload)

    d = Dictation(model=model)
    tail = np.zeros(int((float(settings.pause_seconds) + 1.0) * common.SAMPLE_RATE), np.float32)
    rows = []
    for _, session in itertools.groupby(clips, context_key):
        bus = Bus()
        vad = audio_in.VadIterator(audio_in.SileroOnnx(), threshold=thr,
                                   min_silence_duration_ms=int(float(settings.pause_seconds) * 1000),
                                   speech_pad_ms=audio_in.SPEECH_PAD_MS)
        ain = audio_in.AudioIn(bus, settings, vad=vad)
        for c in session:
            audio = common.load_audio(c.audio_path(data_dir))
            bus.utterances.clear()
            feed = np.concatenate([audio.astype(np.float32), tail])
            for i in range(0, len(feed), 2048):
                ain.process(feed[i:i + 2048])
            cut = " ".join(words_to_text(d.transcribe(u)) for u in bus.utterances).strip()
            full = words_to_text(d.transcribe(audio))
            gap = np.zeros(int(0.3 * common.SAMPLE_RATE), np.float32)       # "re-hear": the pieces as one clip
            joined = (words_to_text(d.transcribe(np.concatenate([x for u in bus.utterances for x in (u, gap)])))
                      if len(bus.utterances) > 1 else cut)
            rows.append({"group": c.group, "ref": c.text, "full": full, "cut": cut, "joined": joined,
                         "pieces": len(bus.utterances),
                         "cut_s": round(sum(len(u) for u in bus.utterances) / common.SAMPLE_RATE, 2),
                         "clip_s": round(len(audio) / common.SAMPLE_RATE, 2)})
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["all"]:
        sel = [r for r in rows if g == "all" or r["group"] == g]
        a = common.wer_table([(r["ref"], r["full"]) for r in sel])
        b = common.wer_table([(r["ref"], r["cut"]) for r in sel])
        j = common.wer_table([(r["ref"], r["joined"]) for r in sel])
        missed = sum(1 for r in sel if r["pieces"] == 0)
        split = sum(1 for r in sel if r["pieces"] > 1)
        out[g] = {"clips": len(sel), "wer_whole_clip": a["wer"], "wer_after_audio_in": b["wer"],
                  "wer_pieces_reheard": j["wer"], "missed": missed, "split": split}
        log(f"{g}: {len(sel)} clips, WER on the whole clip {a['wer']}, on what audio_in cut {b['wer']}, "
            f"pieces heard again as one {j['wer']}; no utterance at all {missed}, cut in pieces {split}")
    _ain.MIN_SPEECH_S, _ain.MAX_GAIN = old_min, old_gain
    dest = (common.REPO_ROOT / "checkpoints" / "hearings"
            / f"vad_rows_{settings.pause_seconds}_{thr}_{min_speech if min_speech is not None else old_min}"
              f"_{max_gain or old_gain}.jsonl")
    with open(dest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


# Closed-set listening, for commands. TORGO's "short words" are like voice commands: yes, no, the arrows, digits and
# the radio alphabet (44 words, as TORGO writes them). Instead of Whisper writing down anything it likes, it can score
# each allowed phrase against the audio (CTranslate2's align(): the probability of a given text, token by token) and
# pick the best, rejecting it when even the best is far less likely than Whisper's own free reading.
CLOSED_SET = ("yes no up down left right forward back select menu zero one two three four five six seven eight nine "
              "alpha charlie delta echo foxtrot golf hotel juliet kilo lima mike november oscar papa quebec romeo "
              "sierra tango uniform victor whisky xray yankee zulu").split()
_DIGITS = "zero one two three four five six seven eight nine".split()
_SPELLINGS = {"xray": ("X-ray", "Xray", "X-Ray"), "whisky": ("Whisky", "Whiskey")}
_SAME = {"x-ray": "xray", "whiskey": "whisky", "juliette": "juliet", **{str(i): w for i, w in enumerate(_DIGITS)}}
TEST_SPEAKERS = ("F04", "M01")       # TORGO's test speakers: report only, never used to pick a threshold


def closed_forms(word: str) -> list[str]:
    """How Whisper may write a closed-set word: "Yes." or "Yes", "Seven." or "7."."""
    base = list(_SPELLINGS.get(word, (word.capitalize(),)))
    if word in _DIGITS:
        base.append(str(_DIGITS.index(word)))
    return [b + end for b in base for end in (".", "")]


def closed_word(text: str) -> str | None:
    """Whisper's free text as one closed-set word, or None: "Seven." and "7" -> seven; "Yes, sir." -> None."""
    import re

    words = [w for w in re.sub(r"[^\w\s'-]", " ", str(text).lower()).split() if w.strip("-'")]
    if len(words) != 1:
        return None
    w = _SAME.get(words[0], words[0])
    return w if w in CLOSED_SET else None


class TextScorer:
    """log P(text | audio) from Whisper's decoder for texts we choose (teacher-forced), one encoder pass per clip:
    CTranslate2's align() returns each given token's probability. It gives the end token no probability (it scores
    text tokens only), so a text's score doesn't say whether more speech follows it: compare texts of similar length."""

    def __init__(self, model):
        import importlib

        self.m = model
        tok_mod = importlib.import_module("faster_whisper.tokenizer")
        self.tok = tok_mod.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe",
                                     language="en")
        self.start = list(self.tok.sot_sequence) + [self.tok.no_timestamps]
        self._tokens: dict[str, list[int]] = {}

    def encode(self, audio):
        import numpy as np

        features = self.m.feature_extractor(np.asarray(audio, dtype=np.float32))
        frames = self.m.feature_extractor.nb_max_frames
        n = min(features.shape[-1], frames)
        features = features[:, :frames]
        if features.shape[-1] < frames:
            features = np.pad(features, ((0, 0), (0, frames - features.shape[-1])))
        return self.m.encode(features), n

    def tokens(self, text: str) -> list[int]:
        if text not in self._tokens:
            self._tokens[text] = self.tok.encode(" " + text.strip())
        return self._tokens[text]

    def score(self, encoded, n_frames: int, texts: list[str]) -> list[float]:
        import math

        import ctranslate2
        import torch

        enc = torch.as_tensor(encoded, device="cuda")
        batch = enc.expand(len(texts), -1, -1).contiguous()
        res = self.m.model.align(ctranslate2.StorageView.from_array(batch), self.start,
                                 [self.tokens(t) for t in texts], [n_frames] * len(texts))
        return [round(sum(math.log(max(p, 1e-12)) for p in r.text_token_probs), 3) for r in res]


def closed_set_test(clips, data_dir: Path, model, log) -> dict:
    """Every clip (in the closed set or not), free reading vs closed-set scoring. Rows to <set>.closed.jsonl; the
    numbers come from closed_set_summary(), which can be rerun on the rows."""
    from engine.voice.dictation import NO_SPEECH_SKIP, Dictation, short_unsure, words_to_text

    d = Dictation(model=model)
    scorer = TextScorer(model)
    forms = [(w, f) for w in CLOSED_SET for f in closed_forms(w)]
    dest = common.REPO_ROOT / "checkpoints" / "hearings" / "closed_rows.jsonl"
    rows = []
    log(f"{len(clips)} clips; {len(CLOSED_SET)} words, {len(forms)} written forms; rows to {dest}")
    with open(dest, "w", encoding="utf-8") as f:
        for i, c in enumerate(clips):
            audio = common.load_audio(c.audio_path(data_dir))
            t0 = time.perf_counter()
            raw = d.transcribe(audio, keep_unsure=True)
            t1 = time.perf_counter()
            free = words_to_text(raw)
            app = "" if (d.last_no_speech > NO_SPEECH_SKIP or short_unsure(raw)) else free
            encoded, n = scorer.encode(audio)
            texts = [t for _, t in forms] + ([free] if free.strip() else [])
            scores = scorer.score(encoded, n, texts)
            t2 = time.perf_counter()
            best: dict[str, tuple[float, int]] = {}
            for (w, t), s in zip(forms, scores):
                if s > best.get(w, (-1e9, 0))[0]:
                    best[w] = (s, len(scorer.tokens(t)))
            ranked = sorted(best.items(), key=lambda x: -x[1][0])
            ref = closed_word(c.text)
            row = {"path": c.path, "speaker": c.speaker, "group": c.group, "ref": c.text, "in_set": ref is not None,
                   "word": ref, "free": free, "app": app, "free_word": closed_word(free), "app_word": closed_word(app),
                   "free_score": scores[-1] if free.strip() else None,
                   "free_tokens": len(scorer.tokens(free)) if free.strip() else 0,
                   "free_words": len(free.split()),
                   "top": [[w, s, n] for w, (s, n) in ranked[:3]],
                   "ref_rank": [w for w, _ in ranked].index(ref) + 1 if ref else None,
                   "free_ms": round(1000 * (t1 - t0)), "score_ms": round(1000 * (t2 - t1))}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if (i + 1) % 200 == 0:
                log(f"  {i + 1}/{len(clips)}")
    return closed_set_summary(rows, log)


CLOSED_SHORT = 3          # only a free reading this short (words) can be a closed-set word; longer is never scored
CLOSED_EMPTY_FLOOR = -3.0  # Whisper wrote nothing: the best word needs at least this score (minus the margin)


def closed_pick(r: dict, margin: float, gap: float = 0.0) -> str | None:
    """The closed-set word for a row, or None: the best-scoring word when the free reading is short, the best word
    scores at least the free reading's score minus `margin`, and beats the runner-up by `gap`."""
    if r["free_words"] > CLOSED_SHORT:
        return None
    (word, score, *_), runner = r["top"][0], (r["top"][1][1] if len(r["top"]) > 1 else -1e9)
    if score - runner < gap:
        return None
    floor = CLOSED_EMPTY_FLOOR if r["free_score"] is None else r["free_score"]
    return word if score >= floor - margin else None


def closed_set_summary(rows: list[dict], log, margins=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0), gaps=(0.0, 1.0, 2.0)) -> dict:
    """Per speaker group (tuning speakers, test speakers): in-set clips right / wrong / nothing done, and how often a
    clip that isn't a closed-set word would trigger one, for the free reading and for closed-set scoring (closed_pick)
    alone and after the free reading (hybrid: a free reading that is a closed-set word is taken as it is)."""
    import statistics

    out = {}
    for name, keep in (("tune", lambda r: r["speaker"] not in TEST_SPEAKERS),
                       ("test", lambda r: r["speaker"] in TEST_SPEAKERS)):
        sel = [r for r in rows if keep(r)]
        ins = [r for r in sel if r["in_set"]]
        outs = [r for r in sel if not r["in_set"]]
        words = [r for r in outs if "words" in r["group"]]
        res = {"in_set": len(ins), "other_words": len(words), "sentences": len(outs) - len(words)}

        def tally(pick):
            right = sum(1 for r in ins if pick(r) == r["word"])
            wrong = sum(1 for r in ins if pick(r) not in (None, r["word"]))
            fa_w = sum(1 for r in words if pick(r) is not None)
            fa_s = sum(1 for r in outs if "words" not in r["group"] and pick(r) is not None)
            return {"right": right, "wrong": wrong, "nothing": len(ins) - right - wrong,
                    "false_other_words": fa_w, "false_sentences": fa_s}

        res["free"] = tally(lambda r: r["free_word"])
        res["app"] = tally(lambda r: r["app_word"])
        res["closed_always"] = tally(lambda r: r["top"][0][0])
        for m in margins:
            for g in gaps:
                res[f"closed_m{m:g}_g{g:g}"] = tally(lambda r, m=m, g=g: closed_pick(r, m, g))
                res[f"hybrid_m{m:g}_g{g:g}"] = tally(lambda r, m=m, g=g: r["free_word"] or closed_pick(r, m, g))
        res["ref_in_top1"] = sum(1 for r in ins if r["ref_rank"] == 1)
        res["ref_in_top3"] = sum(1 for r in ins if r["ref_rank"] and r["ref_rank"] <= 3)
        res["score_ms_p50"] = statistics.median(r["score_ms"] for r in sel) if sel else None
        res["free_ms_p50"] = statistics.median(r["free_ms"] for r in sel) if sel else None
        out[name] = res
        log(f"--- {name} speakers: {len(ins)} closed-set clips, {len(words)} other single words, "
            f"{res['sentences']} sentences; scoring p50 {res['score_ms_p50']} ms (free reading {res['free_ms_p50']} ms)")
        log(f"   the right word scores best: {pct(res['ref_in_top1'], len(ins))}; in the top 3: "
            f"{pct(res['ref_in_top3'], len(ins))}")
        for k, v in res.items():
            if isinstance(v, dict):
                log(f"   {k:16s} right {pct(v['right'], len(ins))}, wrong {v['wrong']}, nothing {v['nothing']}; "
                    f"triggered by other words {pct(v['false_other_words'], len(words))}, "
                    f"by sentences {pct(v['false_sentences'], res['sentences'])}")
    return out


def fix_simulation(rows_path: Path, text_key: str, log, mic: str = "wav_headMic") -> dict:
    """Would learning from fixes (engine/voice/personal_fixes.py) pay off for a dysarthric speaker? Replays a set's
    rows in recording order (one microphone, so no utterance counts twice): each clip first gets the fixes learned
    so far (made MIN_TIMES or more), then the user is assumed to fix every word still wrong by voice, which teaches
    or forgets fixes exactly as PersonalFixes does. Counts words wrong before and after, and the learned fixes that
    were right and wrong."""
    import tempfile

    import jiwer

    from engine.voice.personal_fixes import PersonalFixes

    rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
    rows = sorted((r for r in rows if f"/{mic}/" in r["path"]), key=lambda r: r["path"])
    tmp = tempfile.TemporaryDirectory()
    fixes = PersonalFixes(Path(tmp.name) / "fixes.txt")
    before = after = n_words = good = bad = 0
    for r in rows:
        ref = common.normalize(r["ref"]).split()
        heard = common.normalize(r[text_key]).split()
        if not ref:
            continue
        fixed_words, used = fixes.apply([{"text": w, "conf": 0.5} for w in heard])
        fixed = [w["text"].lower() for w in fixed_words]
        e0 = jiwer.process_words(" ".join(ref), " ".join(heard) or "<none>")
        e1 = jiwer.process_words(" ".join(ref), " ".join(fixed) or "<none>")
        before += e0.substitutions + e0.deletions + e0.insertions
        after += e1.substitutions + e1.deletions + e1.insertions
        n_words += len(ref)
        refset = set(ref)
        for f in used:
            good += f.meant.lower() in refset
            bad += f.meant.lower() not in refset
        for chunk in e1.alignments[0]:                 # the user fixes what is still wrong, word by word
            rs, re_, hs, he = chunk.ref_start_idx, chunk.ref_end_idx, chunk.hyp_start_idx, chunk.hyp_end_idx
            if chunk.type == "substitute" and re_ - rs == he - hs:
                for i, j in zip(range(rs, re_), range(hs, he)):
                    if j < len(fixed):                 # an empty answer was scored as "<none>": nothing to fix
                        fixes.record(fixed[j], ref[i])
        fixes.kept(" ".join(ref))                      # typed as fixed: heard words left in it were right
    tmp.cleanup()
    out = {"clips": len(rows), "words": n_words, "wer_before": round(100.0 * before / n_words, 2),
           "wer_after": round(100.0 * after / n_words, 2), "learned_fixes_used_right": good,
           "learned_fixes_used_wrong": bad}
    log(f"{len(rows)} clips ({mic}), {n_words} words: WER {out['wer_before']} -> {out['wer_after']} with learned "
        f"fixes; learned fixes made {good} times rightly, {bad} times wrongly")
    return out


def cleanup_compare(rows_path: Path, text_key: str, log,
                    groups=("dys_seen", "dys_unseen", "librispeech", "personal")) -> dict:
    """The hearings plan's last check: the trained clean-up model (llama-server, settings.rewrite_gguf) on each
    sentence's Whisper text, without and with Whisper's other readings as "Other hearings". Scores both against the
    right text. Text only; nothing is written but the summary next to the rows."""
    import copy

    from engine.config import settings as base
    from engine.voice.rewrite import Rewriter

    rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
    rows = [r for r in rows if r["group"] in groups and r.get(text_key, "").strip()]
    s = copy.copy(base)
    s.rewrite_backend = "llamacpp"
    rw = Rewriter(s, log=lambda *a: None)
    rw.load()
    out_rows, n_words, fell_back = [], 0, 0
    try:
        for r in rows:
            text = r[text_key]
            hears = [t for t, _ in r["readings"] if common.normalize(t) != common.normalize(text)][:2]
            plain = rw.rewrite_full(text)
            with_h = rw.rewrite_full(text, hearings=hears) if hears else plain
            fell_back += plain.reason is not None and str(plain.reason).startswith(("error", "light clean: model error"))
            e_raw, n = errors(r["ref"], text)
            out_rows.append({"ref": r["ref"], "text": text, "hearings": hears, "clean": plain.clean,
                             "clean_h": with_h.clean, "unsure": plain.uncertain, "unsure_h": with_h.uncertain,
                             "reason": plain.reason, "err_text": e_raw, "err_clean": errors(r["ref"], plain.clean)[0],
                             "err_clean_h": errors(r["ref"], with_h.clean)[0], "words": n})
            n_words += n
    finally:
        rw.close()                                     # one llama-server at a time: a second one fights for the GPU
    if fell_back:
        log(f"WARNING: the clean-up model failed on {fell_back} of {len(out_rows)} rows (fell back to the rule-based "
            "clean-up); don't use these numbers")
    w = lambda k: round(100.0 * sum(x[k] for x in out_rows) / n_words, 2) if n_words else None
    with_any = [x for x in out_rows if x["hearings"]]
    summary = {"rows": len(out_rows), "words": n_words, "text_key": text_key,
               "wer_whisper": w("err_text"), "wer_cleanup": w("err_clean"), "wer_cleanup_hearings": w("err_clean_h"),
               "with_hearings": len(with_any),
               "hearings_better": sum(1 for x in with_any if x["err_clean_h"] < x["err_clean"]),
               "hearings_worse": sum(1 for x in with_any if x["err_clean_h"] > x["err_clean"]),
               "unsure": sum(1 for x in out_rows if x["unsure"]), "unsure_h": sum(1 for x in out_rows if x["unsure_h"])}
    log(f"{summary['rows']} sentences ({n_words} words), Whisper text '{text_key}': WER Whisper {summary['wer_whisper']}, "
        f"after the clean-up {summary['wer_cleanup']}, with other hearings {summary['wer_cleanup_hearings']}")
    log(f"hearings sent on {summary['with_hearings']}: better {summary['hearings_better']}, worse "
        f"{summary['hearings_worse']}; marked unsure {summary['unsure']} without, {summary['unsure_h']} with")
    tag = "" if set(groups) >= {"dys_seen", "librispeech"} else "-" + "-".join(groups)   # other groups: own file
    dest = rows_path.with_name(rows_path.stem + f".cleanup-{text_key}{tag}.jsonl")
    with open(dest, "w", encoding="utf-8") as f:
        for x in out_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    common.save_json(dest.with_suffix(".summary.json"), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(p)
    p.add_argument("--set", default="torgo-dev", choices=sorted(SETS))
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--limit", type=int, default=None, help="only the first N clips (a quick check)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cleanup", type=Path, default=None,
                   help="instead: run the trained clean-up on these rows, without and with the other hearings")
    p.add_argument("--text", default="greedy", help="--cleanup: which Whisper text to clean (greedy, b5)")
    p.add_argument("--fix-sim", type=Path, default=None,
                   help="instead: replay these rows as if the user fixed every wrong word (learning from fixes)")
    p.add_argument("--hot-test", action="store_true",
                   help="instead: the app's pass with and without a word list the speaker isn't saying")
    p.add_argument("--vad-test", action="store_true",
                   help="instead: the app's pass on what audio_in.py cuts from each clip, against the whole clip")
    p.add_argument("--pause", type=float, default=None, help="--vad-test: the pause setting to try, in seconds")
    p.add_argument("--vad-threshold", type=float, default=None, help="--vad-test: Silero's speech threshold")
    p.add_argument("--min-speech", type=float, default=None, help="--vad-test: the shortest utterance kept, s")
    p.add_argument("--max-gain", type=float, default=None, help="--vad-test: the auto gain's ceiling (times)")
    p.add_argument("--context-test", action="store_true",
                   help="instead: rare words of the same chapter or session as hotwords (names on screen)")
    p.add_argument("--variants", default=None, const=",".join(VARIANTS), nargs="?",
                   help="instead: the app's pass with different decoding options (VARIANTS; a comma list picks some)")
    p.add_argument("--closed-test", action="store_true",
                   help="instead: closed-set listening (commands) on TORGO's short words; use --set torgo-all")
    p.add_argument("--closed-summary", type=Path, default=None,
                   help="instead: the closed-set numbers again from saved rows (no GPU)")
    a = p.parse_args(argv)
    log = common.Log()
    if a.closed_summary:
        rows = [json.loads(line) for line in open(a.closed_summary, encoding="utf-8")]
        common.save_json(a.closed_summary.with_suffix(".summary.json"), closed_set_summary(rows, log))
        return 0
    if a.cleanup:
        cleanup_compare(a.cleanup, a.text, log)
        return 0
    if a.fix_sim:
        for mic in ("wav_headMic", "wav_arrayMic"):
            fix_simulation(a.fix_sim, a.text, log, mic=mic)
        return 0
    split, source = SETS[a.set]
    clips = pick(common.read_manifest(common.manifest_path(Path(a.data_dir))), split, source, a.limit)
    if not clips:
        raise SystemExit(f"No {source} clips in the {split} split of {common.manifest_path(Path(a.data_dir))}")

    common.preload_cuda12_libs()
    from engine.voice.dictation import Dictation, load_torch_first, resolve_model, words_to_text

    load_torch_first("cuda")
    from engine.voice.whisper_import import faster_whisper

    model = faster_whisper().WhisperModel(resolve_model(a.model), device="cuda", compute_type="float16")
    if a.hot_test or a.vad_test or a.variants or a.context_test or a.closed_test:
        if a.closed_test:
            test, name = closed_set_test, "closed"
        elif a.context_test:
            test, name = context_test, "context"
        elif a.variants:
            names = tuple(n.strip() for n in a.variants.split(",") if n.strip() in VARIANTS)
            test, name = (lambda *args: variants_test(*args, names=names)), "variants"
        elif a.hot_test:
            test, name = hotwords_test, "hotwords"
        else:
            test = lambda *args: vad_test(*args, pause=a.pause, threshold=a.vad_threshold,  # noqa: E731
                                          min_speech=a.min_speech, max_gain=a.max_gain)
            name = f"vad{a.pause or ''}_{a.vad_threshold or ''}_{a.min_speech if a.min_speech is not None else ''}"
        summary = test(clips, Path(a.data_dir), model, log)
        common.save_json(common.REPO_ROOT / "checkpoints" / "hearings" / f"{a.set}.{name}.json", summary)
        return 0
    d = Dictation(model=model)
    d.transcribe(common.load_audio(clips[0].audio_path(Path(a.data_dir))))     # warm up CUDA
    d.hearings(common.load_audio(clips[0].audio_path(Path(a.data_dir))))

    out_path = a.out or common.REPO_ROOT / "checkpoints" / "hearings" / f"{a.set}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, previous = [], {}
    log(f"{len(clips)} clips from {a.set}, model {a.model}; rows to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(clips):
            audio = common.load_audio(c.audio_path(Path(a.data_dir)))
            t0 = time.perf_counter()
            words = d.transcribe(audio)
            t1 = time.perf_counter()
            readings = d.hearings(audio)
            t2 = time.perf_counter()
            raw_words = d.transcribe(audio, keep_unsure=True)          # the same pass without the noise filters
            no_speech = round(d.last_no_speech, 3)
            t3 = time.perf_counter()
            b5 = d.transcribe(audio, beam_size=5)                      # the app's pass with a beam of 5
            t4 = time.perf_counter()
            b5_raw = d.transcribe(audio, beam_size=5, keep_unsure=True)
            prompt = previous.get(context_key(c))
            prompted = words_to_text(d.transcribe(audio, prompt=prompt)) if prompt else words_to_text(words)
            before = (previous.get(context_key(c), "") + " " + common.whisper_style(c.text)).strip()
            previous[context_key(c)] = before[-PROMPT_CHARS:].split(" ", 1)[-1] if len(before) > PROMPT_CHARS else before
            greedy = words_to_text(words)
            texts = [t for t, _ in readings]
            g_err, n = errors(c.text, greedy)
            b_err = errors(c.text, texts[0])[0] if texts else n
            o_err = min([g_err] + [errors(c.text, t)[0] for t in texts])
            row = {"path": c.path, "group": c.group, "speaker": c.speaker, "ref": c.text, "greedy": greedy,
                   "readings": readings, "prompt_used": bool(prompt), "prompted": prompted,
                   "words_conf": [[w["text"], w["conf"]] for w in words],
                   "min_conf": min((w["conf"] for w in words), default=0.0),
                   "words": n, "greedy_err": g_err, "beam_err": b_err, "oracle_err": o_err,
                   "prompt_err": errors(c.text, prompted)[0],
                   "raw": words_to_text(raw_words), "raw_err": errors(c.text, words_to_text(raw_words))[0],
                   "raw_conf": [[w["text"], w["conf"]] for w in raw_words], "no_speech": no_speech,
                   "b5": words_to_text(b5), "b5_err": errors(c.text, words_to_text(b5))[0],
                   "b5raw": words_to_text(b5_raw), "b5raw_err": errors(c.text, words_to_text(b5_raw))[0],
                   "b5_conf": [[w["text"], w["conf"]] for w in b5], "b5_ms": round(1000 * (t4 - t3)),
                   "greedy_ms": round(1000 * (t1 - t0)), "hearings_ms": round(1000 * (t2 - t1))}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(clips)}")
    summary = {"set": a.set, "model": a.model, "gpu": common.gpu_name(), "all": summarize(rows, log)}
    for g in sorted({r["group"] for r in rows}):
        log(f"--- {g}")
        summary[g] = summarize([r for r in rows if r["group"] == g], log)
    common.save_json(out_path.with_suffix(".summary.json"), summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
