"""Make clear speech sound dysarthric, to train Whisper for our users without dysarthric recordings.

Route: voice
Publishes: nothing (a training tool)
APIs: numpy, scipy, librosa, soundfile (all in requirements.txt)

Our own code, built from what is known about dysarthric speech, not from any dysarthric recordings, so what it makes
from openly licensed speech (LibriSpeech, CC BY 4.0) can train a model we sell. docs/DECISIONS.md, 2026-11-10.

What it changes, each at a random strength for the severity:
  slow       speaking rate 0.4x to 0.9x (dysarthric speech is slow)
  rhythm     uneven rate inside the sentence, and extra pauses
  slur       imprecise articulation: vowels drift toward each other and sounds run together (the vocal-tract shape
             smoothed and shifted; the pitch stays)
  consonants s, sh, f, t, k weakened or lost
  nasal      a hypernasal voice
  blur       consonants made less crisp (the high frequencies of s, f, t softened)
  breathy    air in the voice, following the speech's loudness
  tremor     a 4 to 7 Hz wobble in loudness and pitch
  harsh      a rough, strained voice (fast random pitch jitter)
  fading     loudness falling across the sentence (as in Parkinson's)
  room       a laptop microphone in a room: echo, a narrower band, background noise
Not modelled: a flat (monotone) pitch; it needs a vocoder.

Version 2 (2026-11-11) adds slur, consonants and nasal. Version 1 was too easy: stock Whisper missed 2.3% (mild),
2.7% (moderate) and 6.7% (severe) of words, against 2.2% on the clean clips. Version 2: 4.5%, 20.0% and 48.4% (120
LibriSpeech dev-clean clips; 4.8%, 17.2% and 50.1% on 200 others), measured with calibrate_sim.py on the RTX 5080.
Slurring is what Whisper notices; every other effect alone changes its error rate by under half a point.

    python -m engine.voice.finetune.simulate --samples 10        # listening samples from LibriSpeech dev-clean
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16_000
SEVERITIES = ("mild", "moderate", "severe")
VERSION = 2                            # prepare_sim keeps each version's clips in their own folder


@dataclass(frozen=True)
class Ranges:
    rate: tuple[float, float]          # overall speed (1 = as said)
    rhythm: float                      # how unevenly each chunk is stretched (+- this fraction)
    pauses: tuple[int, int]            # how many extra pauses
    pause_s: tuple[float, float]       # each pause, seconds
    blur_hz: tuple[float, float]       # low-pass cutoff: lower = blurrier consonants
    breath_db: tuple[float, float]     # breath noise below the speech (lower = breathier)
    tremor: tuple[float, float]        # loudness wobble depth (0 to 1)
    jitter_ms: tuple[float, float]     # pitch roughness, as a delay wobble in ms
    fade: tuple[float, float]          # how much quieter the end is (0 to 1)
    slur_hz: tuple[float, float]       # how far the vocal-tract shape is smoothed across frequency (Hz)
    slur_ms: tuple[float, float]       # ... and across time (ms): sounds run into each other
    warp: float                        # vowels drift: formants shifted by up to +- this fraction, changing slowly
    drop_db: tuple[float, float]       # how much a weakened consonant loses
    p_drop: float                      # chance that each consonant is weakened
    p_rhythm: float                    # chance of each effect
    p_slur: float
    p_consonants: float
    p_nasal: float
    p_blur: float
    p_voice: float                     # breathy, tremor or harsh (one of them)
    p_fade: float


RANGES = {
    "mild": Ranges(rate=(0.70, 0.90), rhythm=0.15, pauses=(0, 2), pause_s=(0.10, 0.30), blur_hz=(4000, 5500),
                   breath_db=(24, 30), tremor=(0.05, 0.15), jitter_ms=(0.05, 0.15), fade=(0.1, 0.3),
                   slur_hz=(900, 1200), slur_ms=(80, 120), warp=0.12, drop_db=(10, 20), p_drop=0.35,
                   p_rhythm=0.5, p_slur=0.9, p_consonants=0.6, p_nasal=0.2, p_blur=0.5, p_voice=0.5, p_fade=0.2),
    "moderate": Ranges(rate=(0.55, 0.80), rhythm=0.25, pauses=(1, 4), pause_s=(0.15, 0.50), blur_hz=(2800, 4000),
                       breath_db=(16, 24), tremor=(0.10, 0.30), jitter_ms=(0.10, 0.30), fade=(0.2, 0.5),
                       slur_hz=(1100, 1500), slur_ms=(100, 150), warp=0.17, drop_db=(15, 25), p_drop=0.5,
                       p_rhythm=0.8, p_slur=1.0, p_consonants=0.85, p_nasal=0.3, p_blur=0.8, p_voice=0.8,
                       p_fade=0.3),
    "severe": Ranges(rate=(0.40, 0.65), rhythm=0.35, pauses=(2, 6), pause_s=(0.25, 0.80), blur_hz=(2000, 3000),
                     breath_db=(10, 18), tremor=(0.20, 0.45), jitter_ms=(0.20, 0.50), fade=(0.3, 0.6),
                     slur_hz=(1300, 1800), slur_ms=(120, 180), warp=0.20, drop_db=(18, 30), p_drop=0.6,
                     p_rhythm=0.9, p_slur=1.0, p_consonants=1.0, p_nasal=0.4, p_blur=0.9, p_voice=0.9, p_fade=0.4),
}


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


# ------------------------------------------------------------------ the effects (mono float32, SR Hz)
def slow_down(audio: np.ndarray, rate: float) -> np.ndarray:
    """Slower (rate < 1) without changing the pitch."""
    import librosa

    if len(audio) < 512 or abs(rate - 1) < 1e-3:
        return audio
    return librosa.effects.time_stretch(audio, rate=rate, n_fft=512, hop_length=128).astype(np.float32)


def _quiet_points(audio: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """n sample positions in the quietest 30% of 20 ms frames, at least 0.15 s from the ends: where a pause or a
    change of pace falls naturally (between words)."""
    frame = SR // 50
    if n <= 0 or len(audio) < SR // 2:
        return []
    frames = len(audio) // frame
    energy = np.array([float(np.mean(audio[i * frame:(i + 1) * frame] ** 2)) for i in range(frames)])
    margin = int(0.15 * SR) // frame
    idx = np.arange(margin, frames - margin)
    if len(idx) == 0:
        return []
    quiet = idx[energy[idx] <= np.quantile(energy[idx], 0.3)]
    if len(quiet) == 0:
        return []
    picks = rng.choice(quiet, size=min(n, len(quiet)), replace=False)
    return sorted(int(p) * frame + frame // 2 for p in picks)


def _join(chunks: list[np.ndarray], fade_s: float = 0.01) -> np.ndarray:
    """Concatenate with a short crossfade, so no clicks at the joins."""
    n = int(fade_s * SR)
    out = chunks[0]
    for c in chunks[1:]:
        if len(out) > n and len(c) > n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            mid = out[-n:] * (1 - ramp) + c[:n] * ramp
            out = np.concatenate([out[:-n], mid, c[n:]])
        else:
            out = np.concatenate([out, c])
    return out


def uneven_rhythm(audio: np.ndarray, rng: np.random.Generator, spread: float, pauses: int,
                  pause_s: tuple[float, float]) -> np.ndarray:
    """Each stretch between natural breaks gets its own pace (+- spread), and `pauses` silences go in at breaks."""
    cuts = _quiet_points(audio, max(3, pauses + 2), rng)
    if not cuts:
        return audio
    bounds = [0] + cuts + [len(audio)]
    pause_at = set(rng.choice(len(cuts), size=min(pauses, len(cuts)), replace=False).tolist()) if pauses else set()
    chunks = []
    for i in range(len(bounds) - 1):
        seg = audio[bounds[i]:bounds[i + 1]]
        if len(seg) >= SR // 4:
            seg = slow_down(seg, float(np.clip(1 + rng.uniform(-spread, spread), 0.5, 1.5)))
        chunks.append(seg)
        if i in pause_at:
            chunks.append(np.zeros(int(_u(rng, pause_s) * SR), dtype=np.float32))
    return _join(chunks)


def blur_consonants(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Soften everything above cutoff_hz, where the hiss and burst of s, f, t, k live."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, min(cutoff_hz, SR / 2 - 100), btype="low", fs=SR, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


N_FFT, HOP = 512, 128                  # 31.25 Hz per bin, 8 ms per frame


def _stft(audio: np.ndarray):
    import librosa

    s = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)
    return np.abs(s), np.angle(s)


def _istft(mag: np.ndarray, phase: np.ndarray, length: int) -> np.ndarray:
    import librosa

    return librosa.istft(mag * np.exp(1j * phase), hop_length=HOP, length=length).astype(np.float32)


def _slow_wander(rng: np.random.Generator, frames: int, every_s: float = 0.3) -> np.ndarray:
    """A random value in -1..1 per frame that changes smoothly, about every `every_s` seconds."""
    step = max(1, int(every_s * SR / HOP))
    knots = rng.uniform(-1, 1, frames // step + 2)
    return np.interp(np.arange(frames), np.arange(len(knots)) * step, knots)


def slur(audio: np.ndarray, rng: np.random.Generator, freq_hz: float, time_ms: float, warp: float) -> np.ndarray:
    """Imprecise articulation. The spectrum is split into the vocal-tract shape (a smooth envelope) and the voice
    (pitch harmonics); only the shape is changed, so the pitch stays. The shape is smoothed across frequency (vowels
    sound alike), across time (sounds run into each other), and its formants drift by up to +-warp, slowly."""
    from scipy.ndimage import uniform_filter1d

    mag, phase = _stft(audio)
    log = np.log(mag + 1e-7)
    env = uniform_filter1d(log, 13, axis=0, mode="nearest")         # ~400 Hz: the vocal-tract shape
    fine = log - env                                                # the harmonics
    env = uniform_filter1d(env, max(1, int(freq_hz / 31.25)), axis=0, mode="nearest")
    env = uniform_filter1d(env, max(1, int(time_ms / 8)), axis=1, mode="nearest")
    if warp:
        bins = np.arange(env.shape[0], dtype=np.float64)
        factor = 1 + warp * _slow_wander(rng, env.shape[1])
        env = np.stack([np.interp(bins / f, bins, env[:, t]) for t, f in enumerate(factor)], axis=1)
    return _istft(np.exp(env + fine), phase, len(audio))


def weak_consonants(audio: np.ndarray, rng: np.random.Generator, drop_db: tuple[float, float],
                    p_drop: float) -> tuple[np.ndarray, int]:
    """Consonants weakened or lost: stretches whose energy is mostly above 2.5 kHz (s, sh, f, t, k bursts) each
    lose drop_db with chance p_drop. Returns the audio and how many were weakened."""
    from scipy.ndimage import uniform_filter1d

    mag, phase = _stft(audio)
    power = mag ** 2
    high = power[80:].sum(axis=0)                                   # bin 80 = 2.5 kHz
    total = power.sum(axis=0) + 1e-12
    loud = total > np.quantile(total, 0.2)
    hiss = (high / total > 0.5) & loud
    gain_db = np.zeros(mag.shape[1])
    weakened, t = 0, 0
    while t < len(hiss):
        if not hiss[t]:
            t += 1
            continue
        end = t
        while end < len(hiss) and hiss[end]:
            end += 1
        if rng.random() < p_drop:
            gain_db[max(0, t - 2):end + 2] = -_u(rng, drop_db)
            weakened += 1
        t = end
    gain = 10 ** (uniform_filter1d(gain_db, 3, mode="nearest") / 20)
    return _istft(mag * gain[None, :], phase, len(audio)), weakened


def nasal(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A hypernasal voice: an extra resonance near 250 Hz and a dip near 1 kHz (air escaping through the nose)."""
    from scipy.signal import filtfilt, iirnotch, iirpeak

    b, a = iirpeak(rng.uniform(220, 320), 2.0, fs=SR)
    x = audio + rng.uniform(0.8, 1.6) * filtfilt(b, a, audio)
    b, a = iirnotch(rng.uniform(800, 1200), 1.5, fs=SR)
    return filtfilt(b, a, x).astype(np.float32)


def _envelope(audio: np.ndarray, smooth_s: float = 0.02) -> np.ndarray:
    n = max(1, int(smooth_s * SR))
    return np.convolve(np.abs(audio), np.ones(n, dtype=np.float32) / n, mode="same").astype(np.float32)


def breathy(audio: np.ndarray, rng: np.random.Generator, below_db: float) -> np.ndarray:
    """Air in the voice: high-passed noise that rises and falls with the speech, below_db under it."""
    from scipy.signal import butter, sosfilt

    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise = sosfilt(butter(2, 1000, btype="high", fs=SR, output="sos"), noise).astype(np.float32)
    shaped = noise * _envelope(audio)
    speech_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
    shaped_rms = float(np.sqrt(np.mean(shaped ** 2))) + 1e-9
    return audio + shaped * (speech_rms / shaped_rms) * 10 ** (-below_db / 20)


def _delay_wobble(audio: np.ndarray, delay_s: np.ndarray) -> np.ndarray:
    """Read the audio through a varying delay: a changing delay bends the pitch."""
    t = np.arange(len(audio), dtype=np.float64) - delay_s * SR
    return np.interp(t, np.arange(len(audio)), audio, left=0.0, right=0.0).astype(np.float32)


def tremor(audio: np.ndarray, rng: np.random.Generator, depth: float) -> np.ndarray:
    """A 4 to 7 Hz shake in loudness and pitch."""
    rate = rng.uniform(4.0, 7.0)
    t = np.arange(len(audio)) / SR
    phase = rng.uniform(0, 2 * np.pi)
    wobble = np.sin(2 * np.pi * rate * t + phase)
    out = _delay_wobble(audio, 0.0006 * depth * (1 + wobble))
    return (out * (1 - depth / 2 + depth / 2 * wobble)).astype(np.float32)


def harsh(audio: np.ndarray, rng: np.random.Generator, jitter_ms: float) -> np.ndarray:
    """A rough, strained voice: the pitch jitters a little at random, many times a second."""
    from scipy.signal import butter, sosfiltfilt

    noise = rng.standard_normal(len(audio))
    noise = sosfiltfilt(butter(2, 80, btype="low", fs=SR, output="sos"), noise)
    noise /= (np.max(np.abs(noise)) + 1e-9)
    return _delay_wobble(audio, (noise + 1) * jitter_ms / 1000)


def fading(audio: np.ndarray, amount: float) -> np.ndarray:
    """Loudness falls across the sentence: the end is `amount` quieter (0.5 = half as loud)."""
    return (audio * np.linspace(1.0, 1.0 - amount, len(audio), dtype=np.float32)).astype(np.float32)


def room(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A laptop microphone in a room: a short echo (0.15 to 0.5 s), 120 Hz to 7 kHz, background noise 18 to 35 dB
    below the speech."""
    from scipy.signal import butter, fftconvolve, sosfilt

    rt60 = rng.uniform(0.15, 0.5)
    n = int(rt60 * SR)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    ir *= rng.uniform(0.15, 0.4) / (np.sqrt(np.sum(ir[1:] ** 2)) + 1e-9) if n > 1 else 1.0
    ir[0] = 1.0
    wet = fftconvolve(audio, ir)[:len(audio)]
    wet = sosfilt(butter(2, [120, 7000], btype="band", fs=SR, output="sos"), wet)
    noise = np.cumsum(rng.standard_normal(len(wet)))          # brown noise: hum and rumble, like a room
    noise -= np.convolve(noise, np.ones(400) / 400, mode="same")
    speech_rms = float(np.sqrt(np.mean(wet ** 2))) + 1e-9
    noise_rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-9
    wet = wet + noise * (speech_rms / noise_rms) * 10 ** (-rng.uniform(18, 35) / 20)
    return wet.astype(np.float32)


# ------------------------------------------------------------------ the whole simulation
EFFECTS = ("rhythm", "slow", "slur", "consonants", "nasal", "blur", "voice", "fade", "room")


def simulate(audio: np.ndarray, severity: str, rng: np.random.Generator,
             only: str | None = None) -> tuple[np.ndarray, list[str]]:
    """The clip made to sound dysarthric at `severity`, and the effects used (for the manifest), e.g.
    ["slow 0.78", "rhythm 2 pauses", "blur 3800 Hz", "tremor 0.18", "room"]. `only` names one effect to apply
    alone, always (calibrate_sim.py measures each effect this way)."""
    r = RANGES[severity]
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < SR // 10:
        return x.copy(), []                                    # too short to be speech: left as it is
    x = x - float(np.mean(x))
    used = []

    def on(name: str, p: float) -> bool:
        return name == only if only else bool(p) and rng.random() < p

    if on("rhythm", r.p_rhythm):
        pauses = int(rng.integers(r.pauses[0], r.pauses[1] + 1))
        x = uneven_rhythm(x, rng, r.rhythm, pauses, r.pause_s)
        used.append(f"rhythm {pauses} pauses")
    if on("slow", 1.0):
        rate = _u(rng, r.rate)
        x = slow_down(x, rate)
        used.insert(0, f"slow {rate:.2f}")
    if on("slur", r.p_slur):
        hz, ms = _u(rng, r.slur_hz), _u(rng, r.slur_ms)
        x = slur(x, rng, hz, ms, r.warp)
        used.append(f"slur {hz:.0f} Hz {ms:.0f} ms")
    if on("consonants", r.p_consonants):
        x, n = weak_consonants(x, rng, r.drop_db, r.p_drop)
        used.append(f"consonants {n} weakened")
    if on("nasal", r.p_nasal):
        x = nasal(x, rng)
        used.append("nasal")
    if on("blur", r.p_blur):
        hz = _u(rng, r.blur_hz)
        x = blur_consonants(x, hz)
        used.append(f"blur {hz:.0f} Hz")
    if on("voice", r.p_voice):
        kind = rng.choice(["breathy", "tremor", "harsh"])
        if kind == "breathy":
            db = _u(rng, r.breath_db)
            x = breathy(x, rng, db)
            used.append(f"breathy {db:.0f} dB")
        elif kind == "tremor":
            d = _u(rng, r.tremor)
            x = tremor(x, rng, d)
            used.append(f"tremor {d:.2f}")
        else:
            j = _u(rng, r.jitter_ms)
            x = harsh(x, rng, j)
            used.append(f"harsh {j:.2f} ms")
    if on("fade", r.p_fade):
        a = _u(rng, r.fade)
        x = fading(x, a)
        used.append(f"fading {a:.2f}")
    if on("room", 0.7):
        x = room(x, rng)
        used.append("room")
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0:
        x = x * (0.9 / peak)                                   # same loudness as the training clips expect
    return np.clip(x, -1.0, 1.0).astype(np.float32), used


# ------------------------------------------------------------------ listening samples
def librispeech_clips(root: Path) -> list[tuple[Path, str]]:
    """(flac, transcript) for every utterance under a LibriSpeech folder (speaker/chapter/*.trans.txt)."""
    out = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            flac = trans.parent / f"{uid}.flac"
            if flac.exists() and text:
                out.append((flac, text.strip()))
    return out


def write_samples(source: Path, out_dir: Path, n: int, seed: int = 1234, min_s: float = 4.0,
                  max_s: float = 10.0) -> list[dict]:
    """n clips from `source` (LibriSpeech), each as said and simulated, severities in turn, plus samples.csv."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    clips = librispeech_clips(source)
    if not clips:
        raise SystemExit(f"No LibriSpeech clips under {source}")
    order = rng.permutation(len(clips))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in order:
        if len(rows) >= n:
            break
        flac, text = clips[int(i)]
        audio, sr = sf.read(flac, dtype="float32")
        if sr != SR or audio.ndim != 1 or not min_s <= len(audio) / SR <= max_s:
            continue
        severity = SEVERITIES[len(rows) % len(SEVERITIES)]
        sim, used = simulate(audio, severity, rng)
        stem = f"{len(rows) + 1:02d}_{severity}"
        sf.write(out_dir / f"{stem}_a_original.wav", audio, SR)
        sf.write(out_dir / f"{stem}_b_simulated.wav", sim, SR)
        rows.append({"file": stem, "severity": severity, "seconds_before": round(len(audio) / SR, 2),
                     "seconds_after": round(len(sim) / SR, 2), "effects": "; ".join(used), "text": text,
                     "source": flac.relative_to(source).as_posix()})
    with open(out_dir / "samples.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    from engine.voice.finetune.common import DEFAULT_DATA_DIR

    p = argparse.ArgumentParser(description="Make listening samples of simulated dysarthric speech.")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--source", type=Path, default=DEFAULT_DATA_DIR / "librispeech" / "dev-clean")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "simulated" / "samples")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args(argv)
    rows = write_samples(a.source, a.out, a.samples, a.seed)
    for r in rows:
        print(f"{r['file']:14} {r['seconds_before']:5.1f}s -> {r['seconds_after']:5.1f}s  {r['effects']}")
    print(f"{len(rows)} pairs in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
