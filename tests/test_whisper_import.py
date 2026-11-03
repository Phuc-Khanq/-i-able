"""Run with: python -m unittest tests.test_whisper_import  (the engine never loads PyAV: its GPL codecs stay out;
docs/licences/2026-11-03_python_packages.md)"""
import subprocess
import sys
import types
import unittest
from pathlib import Path

from engine.voice import whisper_import

ROOT = Path(__file__).resolve().parents[1]
# A fresh Python where the real PyAV can't be imported, as on a PC installed without it
NO_AV = ("import importlib.abc, sys\n"
         "class NoAV(importlib.abc.MetaPathFinder):\n"
         "    def find_spec(self, name, path=None, target=None):\n"
         "        if name == 'av' or name.startswith('av.'): raise ImportError('not installed')\n"
         "sys.meta_path.insert(0, NoAV())\n")


def fresh(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", NO_AV + code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                          env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"})


class TestStandIn(unittest.TestCase):
    def test_any_use_says_why(self):
        av = whisper_import._stand_in()
        with self.assertRaisesRegex(ImportError, "GPL codecs"):
            av.open("clip.wav")
        with self.assertRaises(AttributeError):              # dunders stay plain, so tools that probe modules work
            av.__file__

    def test_a_real_one_already_loaded_is_left_alone(self):
        real = types.ModuleType("av")
        saved = sys.modules.get("av")
        sys.modules["av"] = real
        try:
            whisper_import.block_av()
            self.assertIs(sys.modules["av"], real)
            self.assertTrue(whisper_import.real_av_loaded())
        finally:
            if saved is None:
                sys.modules.pop("av", None)
            else:
                sys.modules["av"] = saved


class TestWithoutPyAV(unittest.TestCase):
    def test_faster_whisper_imports_and_a_file_gives_a_clear_error(self):
        r = fresh("from engine.voice.whisper_import import faster_whisper\n"
                  "fw = faster_whisper()\n"
                  "print('model class:', fw.WhisperModel.__name__)\n"
                  "try:\n    fw.decode_audio('clip.wav')\nexcept ImportError as e:\n    print('file:', e)\n")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("model class: WhisperModel", r.stdout)
        self.assertIn("GPL codecs", r.stdout)

    def test_the_engine_blocks_it_before_anything_heavy(self):
        r = fresh("import engine.main, torchvision\n"
                  "from engine.voice.whisper_import import real_av_loaded\n"
                  "print('stand-in:', getattr(sys.modules['av'], '__eyeable_stand_in__', False), real_av_loaded())\n")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("stand-in: True False", r.stdout)


if __name__ == "__main__":
    unittest.main()
