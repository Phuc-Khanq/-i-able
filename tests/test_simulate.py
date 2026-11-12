"""Run with: python -m unittest tests.test_simulate

The dysarthric speech simulator (engine/voice/finetune/simulate.py) on made-up audio: syllable-like tone bursts with
gaps, so no recordings are needed.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.voice.finetune import simulate as sim

SR = sim.SR


def speechlike(seconds=3.0, seed=0):
    """Bursts of a buzzy 140 Hz tone with hiss on top (vowel + consonant), 0.2 s on, 0.1 s off."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
    hiss = rng.standard_normal(len(t)) * 0.3
    gate = ((t % 0.3) < 0.2).astype(float)
    return ((voice + hiss) * gate * 0.2).astype(np.float32)


def high_share(x, above=4000):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spec[freqs > above].sum() / spec.sum())


class TestEffects(unittest.TestCase):
    def test_slow_down_lengthens(self):
        x = speechlike()
        self.assertAlmostEqual(len(sim.slow_down(x, 0.75)) / len(x), 1 / 0.75, delta=0.03)

    def test_blur_takes_away_the_highs(self):
        x = speechlike()
        self.assertLess(high_share(sim.blur_consonants(x, 2500)), high_share(x) / 5)

    def test_fading_ends_quieter(self):
        x = np.ones(SR, dtype=np.float32)
        y = sim.fading(x, 0.5)
        self.assertAlmostEqual(float(y[0]), 1.0)
        self.assertAlmostEqual(float(y[-1]), 0.5, places=3)

    def test_rhythm_adds_pauses(self):
        x = speechlike(4.0)
        y = sim.uneven_rhythm(x, np.random.default_rng(1), spread=0.0, pauses=3, pause_s=(0.3, 0.3))
        self.assertGreater(len(y), len(x) + 0.5 * SR)

    def test_voice_effects_keep_the_length(self):
        x = speechlike()
        rng = np.random.default_rng(2)
        for y in [sim.breathy(x, rng, 15), sim.tremor(x, rng, 0.3), sim.harsh(x, rng, 0.3), sim.room(x, rng)]:
            self.assertEqual(len(y), len(x))
            self.assertTrue(np.all(np.isfinite(y)))


class TestArticulation(unittest.TestCase):
    """Version 2's effects: slurring, weak consonants, a nasal voice."""

    def test_keep_length_and_stay_finite(self):
        x = speechlike()
        rng = np.random.default_rng(4)
        out = [sim.slur(x, rng, 1500, 150, 0.2), sim.weak_consonants(x, rng, (20, 30), 1.0)[0], sim.nasal(x, rng)]
        for y in out:
            self.assertEqual(len(y), len(x))
            self.assertTrue(np.all(np.isfinite(y)))

    def test_slur_changes_the_sound_but_keeps_its_loudness_roughly(self):
        x = speechlike()
        y = sim.slur(x, np.random.default_rng(5), 1500, 150, 0.2)
        self.assertGreater(float(np.mean((y - x) ** 2)), 0.01 * float(np.mean(x ** 2)))
        ratio = float(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2)))
        self.assertTrue(0.3 < ratio < 3.0, ratio)

    def test_weak_consonants_quiets_hiss_and_leaves_hum(self):
        t = np.arange(SR) / SR
        hum = 0.3 * np.sin(2 * np.pi * 200 * t)                        # a vowel-like first half
        hiss = np.random.default_rng(0).standard_normal(SR) * 0.3        # an s-like second half
        from scipy.signal import butter, sosfilt
        hiss = sosfilt(butter(4, 4000, btype="high", fs=SR, output="sos"), hiss)
        x = np.concatenate([hum, hiss]).astype(np.float32)
        y, n = sim.weak_consonants(x, np.random.default_rng(1), (20, 20), 1.0)
        self.assertGreaterEqual(n, 1)
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        self.assertLess(rms(y[SR + SR // 4:]), rms(x[SR + SR // 4:]) * 0.3)    # the hiss lost about 20 dB
        self.assertGreater(rms(y[SR // 4:SR // 2]), rms(x[SR // 4:SR // 2]) * 0.8)   # the hum kept

    def test_only_applies_just_that_effect(self):
        for name in sim.EFFECTS:
            _, used = sim.simulate(speechlike(), "moderate", np.random.default_rng(2), only=name)
            self.assertLessEqual(len(used), 1, (name, used))
        _, used = sim.simulate(speechlike(), "severe", np.random.default_rng(2), only="slur")
        self.assertEqual(len(used), 1)
        self.assertTrue(used[0].startswith("slur"))


class TestSimulate(unittest.TestCase):
    def test_output_is_clean_audio(self):
        for severity in sim.SEVERITIES:
            y, used = sim.simulate(speechlike(), severity, np.random.default_rng(3))
            self.assertEqual(y.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertLessEqual(float(np.max(np.abs(y))), 1.0)
            self.assertTrue(used[0].startswith("slow"))
        _, used = sim.simulate(speechlike(), "severe", np.random.default_rng(3))
        self.assertTrue(any(u.startswith("slur") for u in used))          # severe always slurs

    def test_same_seed_same_result(self):
        a, ua = sim.simulate(speechlike(), "moderate", np.random.default_rng(7))
        b, ub = sim.simulate(speechlike(), "moderate", np.random.default_rng(7))
        self.assertEqual(ua, ub)
        np.testing.assert_array_equal(a, b)

    def test_severe_is_slower_than_mild(self):
        x = speechlike()
        def mean_len(sev):
            return np.mean([len(sim.simulate(x, sev, np.random.default_rng(s))[0]) for s in range(6)])
        self.assertGreater(mean_len("severe"), mean_len("mild") * 1.15)

    def test_tiny_and_empty_clips_dont_crash(self):
        for n in (0, 100, SR // 4):
            y, _ = sim.simulate(np.zeros(n, dtype=np.float32), "severe", np.random.default_rng(0))
            self.assertTrue(np.all(np.isfinite(y)))


class TestSamples(unittest.TestCase):
    def test_writes_pairs_and_a_csv(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "dev-clean" / "84" / "121123"
            src.mkdir(parents=True)
            lines = []
            for i in range(4):
                uid = f"84-121123-{i:04d}"
                sf.write(src / f"{uid}.flac", speechlike(5.0, seed=i), SR)
                lines.append(f"{uid} HELLO WORLD NUMBER {i}")
            (src / "84-121123.trans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            out = Path(d) / "out"
            rows = sim.write_samples(Path(d) / "dev-clean", out, n=3)
            self.assertEqual([r["severity"] for r in rows], ["mild", "moderate", "severe"])
            self.assertEqual(len(list(out.glob("*_b_simulated.wav"))), 3)
            self.assertTrue((out / "samples.csv").exists())


if __name__ == "__main__":
    unittest.main()
