"""Run with: python -m unittest tests.test_spoken_punctuation

Spoken punctuation while dictating (Wispr Flow's smart formatting): only the clear cases become marks.
"""
import unittest

from engine.core import messages
from engine.core.mode_manager import ModeManager
from engine.voice.commands import spoken_punctuation
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


def run(text):
    return " ".join(w["text"] for w in spoken_punctuation([{"text": t, "conf": 0.9} for t in text.split()]))


class TestSpokenPunctuation(unittest.TestCase):
    def test_marks(self):
        self.assertEqual(run("see you then comma Mom question mark"), "see you then, Mom?")
        self.assertEqual(run("I love it exclamation point see you"), "I love it! See you")
        self.assertEqual(run("Call me tomorrow period"), "Call me tomorrow.")
        self.assertEqual(run("that is all full stop thanks"), "that is all. Thanks")

    def test_whisper_punctuation_around_it_is_dropped(self):
        self.assertEqual(run("Hi, comma, how are you? Question mark."), "Hi, how are you?")
        self.assertEqual(run("it ended period. Then we left"), "it ended. Then we left")

    def test_words_that_are_just_words(self):
        for text in ["The trial period ends soon", "a period of time", "put a comma here", "comma first",
                     "What is the question mark for", "my period started"]:
            self.assertEqual(run(text), text)

    def test_on_the_card(self):
        bus = FakeBus()
        worker = DictationWorker(bus, Dictation(model=FakeWhisperModel()), settings=None)
        ModeManager(bus, log=lambda *_: None)
        worker.process("start dictation")
        worker.process("are you coming comma Mom question mark")
        words = bus.ui(messages.DICTATION_PREVIEW)[-1]["words"]
        self.assertEqual(" ".join(w["text"] for w in words), "are you coming, Mom?")


if __name__ == "__main__":
    unittest.main()
