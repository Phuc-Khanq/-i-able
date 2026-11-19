"""Transcribes each utterance with per-word confidence.

Route: Khoi and Khoa (Khoi leads)
Publishes: topics.TRANSCRIPT {"text", "words": [{"text", "conf"}], "ms"}
Subscribes: topics.UTTERANCE (float32 audio from audio_in) and topics.SPEECH_SPAN (when it was
            spoken, for the mouth gate: only the user's voice counts; see DictationWorker), and
            topics.PARTIAL_AUDIO (the utterance so far, for the live preview)
Also sends: a dictation_live UI message (what Whisper hears so far) while the user is still speaking
APIs: faster-whisper WhisperModel("large-v3-turbo", device="cuda", compute_type="float16"),
      transcribe(word_timestamps=True, hotwords=...)

The mode manager decides whether a transcript is a command or dictation, and only
dictation becomes a dictation_preview, so "send it" never flashes up as typed text.

Whisper runs in its own worker thread (CONTRIBUTING rule 9): on_utterance only
puts audio on a queue, so audio_in's thread is never blocked.
"""

from __future__ import annotations

import importlib
import queue
import sys
import threading
import time
from pathlib import Path

from engine.core import topics

REPO_ROOT = Path(__file__).resolve().parents[2]

BEAM_SIZE = 1             # greedy: fastest; short utterances barely gain from beam search
# Other faster-whisper options for the app's pass. log_prob_threshold None: when Whisper is unsure (a low average log
# probability, which is most of the time for a voice it finds hard), keep its answer instead of decoding again at
# random (the temperature fallback); the fallback still catches repetition loops (compression ratio). TORGO dev (M05,
# 470 clips): 55.4% -> 51.5% of words wrong, sentences unchanged (26.0%), single words 121.6% -> 108.7%; LibriSpeech
# dev unchanged (1.50%); slightly faster (hearing_eval.py --variants, 2026-11-19).
DECODE: dict = {"log_prob_threshold": None}
NO_SPEECH_SKIP = 0.6      # drop a segment Whisper thinks is silence
HALLUCINATION_CONF = 0.35  # a 1-3 word result this unsure is usually noise ("Thank you.")
MAX_QUEUED = 3            # if Whisper falls behind, drop the oldest utterance
HEARINGS_N = 3            # Whisper's other readings for the clean-up (docs/cleanup_hearings_plan.md): best + 2
HEARINGS_BEAM = 5
HEARINGS_BELOW = 0.5      # ...only when a word is under this: on TORGO dev (M05) that ran on 70% of clips and caught
                          # 88% of those a reading improves (hearing_eval.py, 2026-11-19)
MAX_HEARINGS = 2


def resolve_model(name: str) -> str:
    """A folder under the repo root (e.g. "models/whisper-elder-ct2") becomes its full path;
    anything else ("large-v3-turbo", "small") is a model name faster-whisper downloads."""
    path = Path(name)
    if not path.is_absolute() and (REPO_ROOT / path).is_dir():
        return str(REPO_ROOT / path)
    return name


def load_torch_first(device: str, platform: str | None = None, import_module=importlib.import_module) -> bool:
    """Windows + CUDA: import torch before faster-whisper. UI-Venus (torch) and Whisper (CTranslate2)
    each load the CUDA libraries (cuBLAS); when CTranslate2 loaded its copy first, torch's load on the
    UI-Venus thread could fail (the cuBLAS load race on Khang's PC). With torch imported first, its DLLs
    are in place and CTranslate2 uses them. No torch (a CPU-only install) or not CUDA on Windows: skip.
    Returns whether torch was imported here."""
    if (platform or sys.platform) != "win32" or not str(device).startswith("cuda"):
        return False
    try:
        import_module("torch")
    except ImportError:
        return False
    return True


class Dictation:
    """Wraps faster-whisper. Pass `model` in tests to skip loading Whisper."""

    def __init__(self, settings=None, model=None, hotwords: str | None = None, personal_words=None) -> None:
        self.settings = settings
        explicit = hotwords is not None
        if hotwords is None:      # settings.whisper_hotwords; "" = none
            hotwords = str(getattr(settings, "whisper_hotwords", "") or "").strip() or None
        self.hotwords = hotwords  # e.g. names the user says often: "Khoa Nhi Gmail"
        # The user's word list (engine/voice/personal_words.py), read before each utterance so a word added by voice
        # helps at once. Only for the real model from settings: tests and the fine-tune evaluation (model=...) and an
        # explicit hotwords= don't get it unless they pass one.
        if personal_words is None and settings is not None and model is None and not explicit:
            from engine.voice.personal_words import PersonalWords
            personal_words = PersonalWords()
        self.personal_words = personal_words
        self._model = model

    def current_hotwords(self) -> str | None:
        """What Whisper leans toward for the next utterance: the user's words (newest first), then
        settings.whisper_hotwords. A list that can't be read is skipped, never an error.

        Safe only without the temperature fallback's random re-decoding (DECODE): with it, a 30-word list was
        written into TORGO dev's single words 112 times (WER 118.5% -> 141.0%); without it 6 times, and the list
        helps (108.7% -> 102.0%; sentences 26.0% -> 25.9%; LibriSpeech 1.50% -> 1.62%; hearing_eval.py --hot-test)."""
        if self.personal_words is None:
            return self.hotwords
        from engine.voice.personal_words import combine
        try:
            hint = self.personal_words.hint()
        except Exception:
            hint = ""
        return combine(hint, self.hotwords or "")

    def load(self) -> None:
        if self._model is None:
            s = self.settings
            if s is None:
                from engine.config import settings as s
            load_torch_first(s.whisper_device)
            from engine.voice.whisper_import import faster_whisper   # heavy: import only when used (rule 6)

            WhisperModel = faster_whisper().WhisperModel     # without PyAV's GPL codecs (whisper_import.py)
            name = resolve_model(s.whisper_model)
            kwargs = dict(device=s.whisper_device, compute_type=s.whisper_compute_type)
            try:
                # already downloaded: load offline (skips a ~2 s online check, nothing leaves the laptop)
                self._model = WhisperModel(name, local_files_only=True, **kwargs)
            except Exception:
                self._model = WhisperModel(name, **kwargs)   # first run: download it once

    def warm_up(self) -> None:
        """First CUDA call is slow; run one second of silence at startup."""
        import numpy as np

        self.load()
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio, prompt: str | None = None, keep_unsure: bool = False,
                   beam_size: int | None = None, keep_short: bool = False, **decode) -> list[dict]:
        """Return [{"text": str, "conf": float, "start": s, "end": s}, ...].

        start / end are seconds from the start of `audio` (Whisper's word timestamps). The
        DictationWorker uses them to line words up with the camera, then drops them: the
        transcript and the preview bubble only get "text" and "conf".

        `prompt`: text the user wrote just before this (Whisper's initial_prompt), so it continues it.
        `keep_unsure`: keep what the noise filters drop (a segment Whisper thinks is silence, a short unsure result).
        `keep_short`: keep only a short unsure result (the caller decides, see DictationWorker.show_unsure_short).
        `decode`: other faster-whisper options for measurements (temperature, without_timestamps); the app uses
        DECODE.
        """
        self.load()
        extra = {"initial_prompt": prompt} if prompt and prompt.strip() else {}
        extra.update(DECODE)
        extra.update(decode)
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=beam_size or BEAM_SIZE,
            word_timestamps=True,
            vad_filter=False,                # audio_in already cut the utterance with Silero VAD
            condition_on_previous_text=False,
            hotwords=self.current_hotwords(),
            **extra,
        )
        words: list[dict] = []
        self.last_no_speech = 0.0
        for seg in segments:                 # segments is lazy: this loop runs the model
            self.last_no_speech = max(self.last_no_speech, float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
            if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_SKIP and not keep_unsure:
                continue
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    start = float(getattr(w, "start", None) or 0.0)
                    end = max(start, float(getattr(w, "end", None) or 0.0))
                    words.append({"text": text, "conf": round(float(w.probability), 3),
                                  "start": round(start, 3), "end": round(end, 3)})
        if short_unsure(words) and not (keep_unsure or keep_short):
            return []
        return words

    def hearings(self, audio, n: int = HEARINGS_N, beam_size: int = HEARINGS_BEAM,
                 prompt: str | None = None) -> list[tuple[str, float]]:
        """Whisper's top `n` readings of one utterance, best first: [(text, score)].

        A beam search without timestamps, with the same hotwords (and prompt) as transcribe(). faster-whisper's
        transcribe() only returns its best reading, so this calls its parts directly: features, encoder, prompt,
        then CTranslate2's generate with num_hypotheses. Utterances over 30 s get [] (Whisper hears 30 s at once).
        score is the hypothesis's length-normalized log probability (higher is surer).
        """
        import numpy as np

        self.load()
        m = self._model
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) > 30 * 16000 or not len(audio):
            return []
        tok_mod = importlib.import_module("faster_whisper.tokenizer")
        fw = importlib.import_module("faster_whisper.transcribe")
        features = m.feature_extractor(audio)
        frames = m.feature_extractor.nb_max_frames
        features = features[:, :frames]
        if features.shape[-1] < frames:
            features = np.pad(features, ((0, 0), (0, frames - features.shape[-1])))
        encoded = m.encode(features)
        tokenizer = tok_mod.Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language="en")
        previous = tokenizer.encode(" " + prompt.strip()) if prompt and prompt.strip() else []
        start = m.get_prompt(tokenizer, previous, without_timestamps=True, hotwords=self.current_hotwords())
        result = m.model.generate(encoded, [start], beam_size=max(beam_size, n), num_hypotheses=n,
                                  return_scores=True, max_length=m.max_length, suppress_blank=True,
                                  suppress_tokens=fw.get_suppressed_tokens(tokenizer, [-1]))[0]
        out = []
        for ids, score in zip(result.sequences_ids, result.scores):
            text = tokenizer.decode([t for t in ids if t < tokenizer.eot]).strip()
            if text:
                out.append((text, round(float(score), 4)))
        return out


def short_unsure(words: list[dict]) -> bool:
    """1 to 3 words, none of them surer than HALLUCINATION_CONF: usually noise ("Thank you."), but for a voice
    Whisper finds hard it is often the right word (TORGO dev single words: about half the right ones look like this)."""
    return bool(words) and len(words) <= 3 and max(w["conf"] for w in words) < HALLUCINATION_CONF


def _same(text: str) -> str:
    """Text for "says the same": lowercase words without punctuation."""
    import re

    return " ".join(re.sub(r"[^\w\s']", " ", str(text).lower()).split())


def words_to_text(words: list[dict]) -> str:
    # Whisper keeps punctuation on the word ("hello,"), so joining with spaces is enough.
    return " ".join(w["text"] for w in words)


def _seconds(audio) -> float:
    """How long a clip is, in seconds (0 for anything that isn't an audio array: tests pass text)."""
    try:
        return len(audio) / 16000.0 if not isinstance(audio, str) else 0.0
    except TypeError:
        return 0.0


def public_words(words: list[dict]) -> list[dict]:
    """Words as the transcript and the UI get them: text and confidence, no timing."""
    return [{"text": w["text"], "conf": w["conf"]} for w in words]


class DictationWorker:
    """Listens for utterances on the bus and transcribes them on its own thread.

    Only the user's voice counts (the mouth gate). With teammates talking near the mic, Whisper
    hears them too. audio_in publishes topics.SPEECH_SPAN (when the clip was spoken) right before
    each UTTERANCE; with `mouth` (engine.eyes.mouth.MouthTracker) the utterance is kept only if
    the camera saw the user's mouth moving while it was said (mouth_gate_mode "utterance"):
        f = mouth.moving_fraction(span start + first word start - pad, span start + last word end + pad)
        keep every word if f is None or f >= mouth_min_moving, else drop them all
    Per utterance, not per word: a word with little jaw movement ("search" in "Click search bar")
    scored low on its own and was dropped, so UI-Venus clicked "Bar". mouth_gate_mode "word" is
    the old gate (the same test on each word's own window), kept as a fallback.
    MouthTracker's contract (confirmed with Khang): it returns None when the face was seen in
    under half of the window's expected frames ("can't tell", so we keep the word); otherwise
    the share of moving frames among the frames where the face was seen. It is thread-safe
    (it locks and copies its history), so calling it from this Whisper thread while the
    camera thread adds frames is fine. If it raises, the word is kept (fail open: a broken
    camera must not mute the user). No mouth, mouth_gate off, or an utterance with no span:
    every word is kept, as before.
    """

    def __init__(self, bus, dictation: Dictation, mouth=None, settings=None,
                 clock=time.monotonic, log=lambda text: print(text, flush=True)) -> None:
        if settings is None:
            from engine.config import settings
        self.bus = bus
        self.dictation = dictation
        self.mouth = mouth
        self.gate = bool(getattr(settings, "mouth_gate", True))
        self.gate_mode = str(getattr(settings, "mouth_gate_mode", "utterance"))
        self.min_moving = float(getattr(settings, "mouth_min_moving", 0.3))
        self.pad_ms = float(getattr(settings, "mouth_pad_ms", 150.0))
        self.status_every_s = float(getattr(settings, "mouth_status_every_s", 10.0))
        self._clock = clock
        self._log = log                       # console lines for tuning; never audio
        self._last_mode_note = float("-inf")  # clock time of the last "live preview is off in this mode" line
        self._span: dict | None = None        # the SPEECH_SPAN for the next UTTERANCE
        self._span_lock = threading.Lock()
        self._last_ignored = float("-inf")    # clock time of the last "Ignored speech" status
        self._last_warning = float("-inf")    # ... and of the last "tracker failed" warning
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUED)
        self._mode = ""                       # the mode manager's mode, from the UI messages
        # live preview: the newest audio-so-far (older ones are skipped), what the card shows now, and a counter
        # that a finished utterance bumps so a partial transcribed meanwhile is thrown away
        self.live = bool(getattr(settings, "live_preview", False))
        self._partial: tuple | None = None
        self._partial_lock = threading.Lock()
        self._partial_gen = 0
        self._live_text = ""
        # Whisper's other readings for the clean-up ("hearings"): only with the real Whisper (tests pass a model), and
        # only on the GPU (about 120 ms there; on a laptop's CPU a second beam-5 pass would take seconds)
        self.send_hearings = (bool(getattr(settings, "whisper_hearings", True))
                              and getattr(dictation, "_model", None) is None
                              and str(getattr(settings, "whisper_device", "cuda")).startswith("cuda"))
        # Short unsure results ("Up." at 0.2) go to a dictation card, marked unsure, when the mouth gate saw the user
        # say them (TORGO dev single words: twice as many right words; see docs/dictation_learning_from_wispr.md)
        self.show_unsure_short = (bool(getattr(settings, "show_unsure_short", True))
                                  and getattr(dictation, "_model", None) is None)
        self._hearings_failed = False
        bus.subscribe(topics.TO_UI, self.on_ui)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        bus.subscribe(topics.SPEECH_SPAN, self.on_speech_span)
        bus.subscribe(topics.UTTERANCE, self.on_utterance)
        bus.subscribe(topics.PARTIAL_AUDIO, self.on_partial)

    def start(self, warm_up: bool = True) -> None:
        self._thread = threading.Thread(target=self._run, args=(warm_up,),
                                        name="whisper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def on_speech_span(self, span: dict) -> None:
        """Called on audio_in's thread right before the UTTERANCE it belongs to."""
        with self._span_lock:
            self._span = span

    def on_partial(self, payload: dict) -> None:
        """Called on audio_in's thread: keep the newest audio-so-far for the live preview (Dictation mode only)."""
        if not self.live:
            return
        if self._mode != "dictation":
            now = self._clock()
            if now - self._last_mode_note >= 15:
                self._last_mode_note = now
                self._log(f"[live] not showing: the mode is '{self._mode or 'unknown'}', live text is for Dictation mode")
            return
        with self._partial_lock:
            self._partial = (payload["audio"], {"start_ms": payload["start_ms"], "end_ms": payload["end_ms"]},
                             self._partial_gen)

    def on_utterance(self, audio) -> None:
        """Called on audio_in's thread: queue the audio with its span and return immediately."""
        with self._partial_lock:
            self._partial, self._partial_gen = None, self._partial_gen + 1   # the finished words are coming
        with self._span_lock:
            item, self._span = (audio, self._span), None
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()       # drop the oldest (its span goes with it)
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def _run(self, warm_up: bool) -> None:
        if warm_up:
            try:
                self.dictation.warm_up()
            except Exception as exc:
                self._error(f"Whisper could not load: {exc}")
                return
        while not self._stop.is_set():
            try:
                audio, span = self._queue.get(timeout=0.05 if self.live else 0.2)
            except queue.Empty:
                self._live_step()
                continue
            try:
                self.process(audio, span)
            except Exception as exc:           # one bad utterance must never end the thread: before this,
                try:                           # a word the Windows console couldn't print killed dictation
                    self._error(f"Dictation hit an error and skipped one utterance: {type(exc).__name__}: {exc}")
                except Exception:
                    pass

    def process(self, audio, span: dict | None = None) -> None:
        """Transcribe one utterance and publish it. Tests call this directly."""
        try:
            self._process(audio, span)
        finally:
            self._set_live("")                 # the finished words (or nothing) replace the live text

    # ---------------------------------------------------------- live preview
    def _live_step(self) -> None:
        """Idle: transcribe the newest audio-so-far and show it. It never delays a finished utterance by
        more than one partial (a Whisper call can't be interrupted), and a partial that was overtaken by
        the finished utterance, or by leaving Dictation mode, is dropped, never shown."""
        with self._partial_lock:
            item, self._partial = self._partial, None
        if item is None:
            return
        audio, span, gen = item
        t0 = time.perf_counter()
        try:
            words = self.dictation.transcribe(audio)
        except Exception as exc:
            self._log(f"[live] Whisper failed on a partial ({type(exc).__name__}: {exc})")
            return                             # the live text is a courtesy: the final transcription reports errors
        ms = (time.perf_counter() - t0) * 1000
        with self._partial_lock:
            stale = gen != self._partial_gen
        head = f"[live] {_seconds(audio):.1f} s of audio, Whisper {ms:.0f} ms"
        if stale or not self._queue.empty() or self._mode != "dictation":
            self._log(f"{head}: too late, the finished words got there first (not shown)")
            return
        if words and self.mouth is not None and self.gate and not self._partial_mouth_ok(words, span):
            self._log(f"{head}: hidden by the mouth gate (your mouth wasn't seen moving): {words_to_text(words)}")
            words = []
        elif words:
            self._log(f"{head}: {words_to_text(words)}")
        else:
            self._log(f"{head}: no words")
        self._set_live(words_to_text(words) if words else "")

    def _partial_mouth_ok(self, words: list[dict], span: dict) -> bool:
        """The mouth gate for live text (quiet: it only decides whether to show it)."""
        start = float(span["start_ms"])
        f = self._fraction(start + words[0]["start"] * 1000 - self.pad_ms,
                           start + words[-1]["end"] * 1000 + self.pad_ms)
        return f is None or f >= self.min_moving

    def _set_live(self, text: str) -> None:
        text = text.strip()
        if text == self._live_text:
            return
        self._live_text = text
        try:
            from engine.core import messages

            self.bus.publish(topics.TO_UI, messages.make(messages.DICTATION_LIVE, text=text))
        except Exception:
            pass

    def _process(self, audio, span: dict | None = None) -> None:
        t0 = time.perf_counter()
        # A short unsure result is noise for most voices but often the right word for ours: keep it when the camera
        # can say whether the user's mouth moved (the mouth gate below), else drop it as before
        keep_short = self.show_unsure_short and self.mouth is not None and self.gate and span is not None
        try:
            words = (self.dictation.transcribe(audio, keep_short=True) if keep_short
                     else self.dictation.transcribe(audio))
        except Exception as exc:               # one bad clip must not kill the demo
            self._error(f"Transcription failed: {exc}")
            return
        unsure = keep_short and short_unsure(words)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if self.live:                          # timing for tuning: where a slow dictation spends its time
            self._log(f"[whisper] {_seconds(audio):.1f} s of audio, {len(words)} words in {ms:.0f} ms")
        heard = len(words)
        if (words and span is not None and self.mouth is not None and self.gate
                and not self._resume_while_paused(words)):
            words = self._mouth_gate(words, span)
            if not words:
                self._ignored()
                return
        if not words:
            return
        payload = {"text": words_to_text(words), "words": public_words(words), "ms": ms}
        if unsure:
            if len(words) < heard:
                return                         # part of it wasn't the user's mouth: drop it as before
            payload["unsure"] = True           # the mode manager only puts it on a dictation card, never a command
            self._log(f"[whisper] unsure, kept because your mouth moved: {payload['text']}")
        hearings = self._hearings(audio, words) if len(words) == heard else []   # gated words: they don't line up
        if hearings:
            payload["hearings"] = hearings
        self.bus.publish(topics.TRANSCRIPT, payload)

    def _hearings(self, audio, words: list[dict]) -> list[str]:
        """Whisper's other readings of this utterance for the clean-up model (docs/cleanup_hearings_plan.md): a beam
        search's top 3, run only when a word is under HEARINGS_BELOW, without the ones that say the same as the
        best. None for clips over 30 s, or if the extra pass fails (dictation goes on without them)."""
        if not self.send_hearings or min(w["conf"] for w in words) >= HEARINGS_BELOW or _seconds(audio) > 30:
            return []
        t0 = time.perf_counter()
        try:
            readings = self.dictation.hearings(audio)
        except Exception as exc:
            if not self._hearings_failed:
                self._hearings_failed = True
                self._log(f"[whisper] other hearings skipped: {exc}")
            return []
        seen = {_same(words_to_text(words))}
        out = []
        for text, _score in readings:
            if _same(text) and _same(text) not in seen:
                seen.add(_same(text))
                out.append(text)
        if self.live:
            self._log(f"[whisper] {len(out)} other hearings in {1000 * (time.perf_counter() - t0):.0f} ms")
        return out[:MAX_HEARINGS]

    # ----------------------------------------------------------- mouth gate
    def on_ui(self, msg: dict) -> None:
        if msg.get("type") == "mode":
            self._mode = str(msg.get("mode", ""))
            if self._mode != "dictation":
                with self._partial_lock:
                    self._partial = None
                self._set_live("")

    def _resume_while_paused(self, words: list[dict]) -> bool:
        """While paused, "resume" always gets through: everything else is ignored in that mode anyway, so
        the worst case is someone else waking it. Without this, a quiet "resume" (mouth score 0.00 to 0.29
        in Khang's live test, Oct 24) was dropped four times and he couldn't get out of Paused."""
        if self._mode != "paused":
            return False
        from engine.voice import commands
        cmd = commands.parse(words_to_text(words), allow_targets=False)
        if cmd is not None and cmd.name == "resume":
            self._log(f"[mouth] paused: \"{words_to_text(words)}\" always counts")
            return True
        return False

    def _fraction(self, start_ms: float, end_ms: float) -> float | None:
        """mouth.moving_fraction, or None (keep) if the tracker fails."""
        try:
            return self.mouth.moving_fraction(start_ms, end_ms)
        except Exception as exc:
            now = self._clock()
            if now - self._last_warning >= 60:
                self._last_warning = now
                self._log(f"[mouth] the mouth tracker failed ({exc}); keeping the words")
            return None

    def _mouth_gate(self, words: list[dict], span: dict) -> list[dict]:
        """The words said while the user's mouth moved (see the class docstring)."""
        start = float(span["start_ms"])
        per_word = [(w, self._fraction(start + w["start"] * 1000 - self.pad_ms,
                                       start + w["end"] * 1000 + self.pad_ms)) for w in words]

        def show(pairs):
            return " ".join(f"{w['text']}({'-' if f is None else f'{f:.2f}'})" for w, f in pairs)

        if self.gate_mode != "word":
            f = self._fraction(start + words[0]["start"] * 1000 - self.pad_ms,
                               start + words[-1]["end"] * 1000 + self.pad_ms)
            keep = f is None or f >= self.min_moving
            number = "-" if f is None else f"{f:.2f} {'>=' if keep else '<'} {self.min_moving:.2f}"
            self._log(f"[mouth] utterance {number}: {'kept' if keep else 'dropped'} "
                      f"{len(words)}/{len(words)}: {show(per_word)}")
            return list(words) if keep else []

        kept = [(w, f) for w, f in per_word if f is None or f >= self.min_moving]
        dropped = [(w, f) for w, f in per_word if not (f is None or f >= self.min_moving)]
        line = f"[mouth] kept {len(kept)}/{len(words)}: {show(kept)}"
        self._log(line + (f"  dropped: {show(dropped)}" if dropped else ""))
        kept_words = [w for w, _ in kept]
        if (dropped and kept_words and len(kept_words) <= 3
                and max(w["conf"] for w in kept_words) < HALLUCINATION_CONF):
            return []                           # what's left is a few unsure words: noise
        return kept_words

    def _ignored(self) -> None:
        now = self._clock()
        if now - self._last_ignored >= self.status_every_s:
            self._last_ignored = now
            from engine.core import messages

            self.bus.publish(topics.TO_UI, messages.make(
                messages.STATUS, level="info", text="Ignored speech while your mouth wasn't moving"))

    def _error(self, text: str) -> None:
        from engine.core import messages

        self.bus.publish(topics.TO_UI, messages.make(messages.STATUS, level="error", text=text))
