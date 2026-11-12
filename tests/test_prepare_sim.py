"""Run with: python -m unittest tests.test_prepare_sim

The simulated-dysarthria experiment's data (engine/voice/finetune/prepare_sim.py) on a tiny made-up data folder:
TORGO must only ever be in the test split, and the control and simulated runs must train on the same sentences.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.voice.finetune import common, prepare_sim, simulate
from engine.voice.finetune.train import Trainer, TrainConfig
from tests.test_simulate import speechlike


def write_flac(path: Path, seconds: float, seed: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), speechlike(seconds, seed), 16000)


def libri(root: Path, subset: str, speakers: list[str], per_speaker: int, seconds: float) -> None:
    for s, spk in enumerate(speakers):
        chapter = root / "librispeech" / subset / spk / "100"
        lines = []
        for i in range(per_speaker):
            uid = f"{spk}-100-{i:04d}"
            write_flac(chapter / f"{uid}.flac", seconds, seed=s * 10 + i)
            lines.append(f"{uid} SENTENCE {i} FROM SPEAKER {spk}")
        (chapter / f"{spk}-100.trans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def torgo(root: Path, spk: str, prompts: list[str]) -> None:
    from tests.test_finetune import write_wav

    session = root / "torgo" / spk / "Session1"
    for i, text in enumerate(prompts):
        p = session / "prompts" / f"{i:04d}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
        write_wav(session / "wav_arrayMic" / f"{i:04d}.wav")


class TestPrepareSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name)
        libri(cls.data, "train-clean-100", ["19", "26", "27"], 3, 2.0)
        libri(cls.data, "dev-clean", ["84", "174", "251", "422"], 2, 2.0)
        libri(cls.data, "test-clean", ["1089"], 2, 2.0)
        torgo(cls.data, "F01", ["The quick brown fox jumps over the lazy dog.", "yes"])     # a train speaker
        torgo(cls.data, "M05", ["A dev sentence for the dev speaker."])                     # the dev speaker
        torgo(cls.data, "F04", ["The quick brown fox jumps over the lazy dog.", "go home"])  # a test speaker
        cls.clips = prepare_sim.build(cls.data, hours=0.01, seed=1, workers=1, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, **kw):
        return [c for c in self.clips if all(getattr(c, k) == v for k, v in kw.items())]

    def test_torgo_is_test_only(self):
        torgo_rows = self.rows(source="torgo")
        self.assertTrue(torgo_rows)
        self.assertEqual({c.split for c in torgo_rows}, {"test"})
        self.assertEqual({c.speaker for c in torgo_rows}, {"F04"})

    def test_control_and_sim_train_on_the_same_sentences(self):
        clean = self.rows(source="sim_clean", split="train")
        sim = self.rows(source="sim", split="train")
        self.assertTrue(clean)
        self.assertEqual(sorted(c.text for c in clean), sorted(c.text for c in sim))
        for c in sim:
            self.assertTrue(c.path.startswith(f"simulated/v{simulate.VERSION}/train-clean-100/"))
            self.assertTrue((self.data / c.path).exists())

    def test_simulated_dev_and_clean_test(self):
        self.assertTrue(self.rows(source="sim", split="dev"))
        self.assertTrue(self.rows(source="librispeech", split="test"))
        self.assertTrue((prepare_sim.sim_dir(self.data) / "effects.csv").exists())

    def test_a_rerun_makes_nothing_new(self):
        logs = []
        again = prepare_sim.build(self.data, hours=0.01, seed=1, workers=1, log=logs.append)
        self.assertEqual(len(again), len(self.clips))
        self.assertTrue(any("Simulating 0 clips" in m for m in logs))

    def test_the_trainer_picks_the_right_clips(self):
        common.write_manifest(common.manifest_path(self.data, name="sim"), self.clips)
        with tempfile.TemporaryDirectory() as ck:
            for name, main_source in (("sim-control", "sim_clean"), ("sim-dys", "sim")):
                cfg = TrainConfig(name=name, manifest="sim", main_sources=[main_source], dev_main_groups=["sim"])
                main, typical, dev_main, dev_typ = Trainer(cfg, self.data, Path(ck), log=lambda *a: None)._clips()
                self.assertEqual({c.source for c in main}, {main_source})
                self.assertEqual({c.source for c in typical}, {"librispeech"})
                self.assertEqual({c.source for c in dev_main}, {"sim"})
                self.assertFalse(any(c.source == "torgo" for c in main + typical + dev_main + dev_typ))

    def test_default_trainer_settings_are_unchanged(self):
        cfg = TrainConfig()
        self.assertEqual((cfg.manifest, cfg.main_sources, cfg.typical_sources), ("all", ["torgo", "personal"],
                                                                                  ["librispeech"]))
        self.assertEqual(cfg.dev_main_groups, list(common.DYS_GROUPS))


class TestLeaks(unittest.TestCase):
    def test_torgo_outside_test_is_refused(self):
        bad = [common.Clip("t.wav", "Hi.", "F01", "dys_words", "train", "torgo", 1.0)]
        with self.assertRaises(SystemExit):
            prepare_sim.check_sim(bad)

    def test_a_librispeech_voice_in_train_and_test_is_refused(self):
        a = common.Clip("a.flac", "Hi.", "19", "sim", "train", "sim", 1.0)
        b = common.Clip("b.flac", "Hi.", "19", "librispeech", "test", "librispeech", 1.0)
        with self.assertRaises(SystemExit):
            prepare_sim.check_sim([a, b])


class TestTooLong(unittest.TestCase):
    def test_a_long_clip_is_made_milder_not_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "long.flac"
            write_flac(src, 14.0, seed=3)
            results = [prepare_sim._simulate_one((str(src), str(Path(d) / f"out{s}.flac"), s)) for s in range(6)]
            for _, severity, effects, secs in results:
                self.assertIsNotNone(secs, effects)
                self.assertLessEqual(secs, common.MAX_CLIP_S)
            self.assertNotEqual({r[1] for r in results}, {"severe"})


class TestChoose(unittest.TestCase):
    def test_takes_from_every_speaker_in_turn(self):
        with tempfile.TemporaryDirectory() as d:
            libri(Path(d), "train-clean-100", ["1", "2"], 4, 2.0)
            from engine.voice.finetune import prepare
            src = prepare._libri_subset(Path(d) / "librispeech", "train-clean-100")
            picked = prepare_sim.choose(src, hours=4 * 2.0 / 3600, seed=0)
            self.assertEqual(sorted(p[0] for p in picked), ["1", "1", "2", "2"])
            self.assertTrue(all(np.isclose(p[3], 2.0) for p in picked))


if __name__ == "__main__":
    unittest.main()
