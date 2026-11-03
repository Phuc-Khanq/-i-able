"""Imports faster-whisper without loading PyAV (the `av` package).

Route: Khang and Khoi
PyAV ships libx264 and libx265 (GPL-2.0) with its FFmpeg, and [eye]able will be sold, so the engine never loads it
(docs/licences/2026-11-03_python_packages.md). faster-whisper imports `av` at the top of faster_whisper/audio.py,
but only uses it to decode audio *files*; the engine always hands Whisper numpy samples (the mic, or --wav read with
scipy). So a stand-in for `av` goes into sys.modules before faster-whisper is imported: the import works, and if
anything ever asks Whisper to decode a file, it fails with a clear message instead of loading GPL code.
Training scripts (engine/voice/finetune/) run in their own processes and may still use the real package.
"""

from __future__ import annotations

import sys
import types

WHY = ("PyAV is left out of [eye]able on purpose (GPL codecs: docs/licences/2026-11-03_python_packages.md). "
       "Read the audio file yourself (scipy or soundfile) and pass Whisper the samples.")


def _stand_in() -> types.ModuleType:
    module = types.ModuleType("av", "Stand-in for PyAV: [eye]able doesn't load it (engine/voice/whisper_import.py).")

    def __getattr__(name: str):                     # any use: av.open, av.audio, av.error ...
        if name.startswith("__"):
            raise AttributeError(name)
        raise ImportError(f"av.{name}: {WHY}")

    module.__getattr__ = __getattr__
    module.__eyeable_stand_in__ = True
    return module


def block_av() -> None:
    """Put the stand-in in place, unless the real PyAV is already loaded. engine/main.py calls it first thing:
    torchvision (which transformers may import) also imports `av` when it loads, and treats a failing one as
    "not installed"."""
    if "av" not in sys.modules:
        sys.modules["av"] = _stand_in()


def faster_whisper():
    """The faster_whisper package, imported with the stand-in for `av` (unless the real one is already loaded)."""
    block_av()
    import faster_whisper as fw

    return fw


def real_av_loaded() -> bool:
    """True if the real PyAV is in this process (the check in tests/test_whisper_import.py)."""
    av = sys.modules.get("av")
    return av is not None and not getattr(av, "__eyeable_stand_in__", False)
