"""Run with: python -m unittest tests.test_personal_words

The user's word list: saved by voice ("add Hathaway to my words"), read by Whisper as hotwords before each utterance.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine.core import messages
from engine.core.mode_manager import ModeManager
from engine.voice import commands, personal_words
from engine.voice.commands import WordListChange, parse_word_list
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.personal_words import PersonalWords, combine
from tests.helpers import FakeBus, FakeWhisperModel


class TempList(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "personal" / "words.txt"
        self.pw = PersonalWords(self.path)

    def tearDown(self):
        self.dir.cleanup()


class TestPersonalWords(TempList):
    def test_no_file_is_an_empty_list(self):
        self.assertEqual(self.pw.words(), [])
        self.assertEqual(self.pw.hint(), "")

    def test_add_remove_and_no_doubles(self):
        self.assertEqual(self.pw.add(" Hathaway. "), "Hathaway")
        self.pw.add("prednisone")
        self.pw.add("hathaway")                      # again, other capitals: moves to the end, one copy
        self.assertEqual(self.pw.words(), ["prednisone", "hathaway"])
        self.assertTrue(self.pw.remove("PREDNISONE"))
        self.assertFalse(self.pw.remove("AFO"))
        self.assertEqual(self.pw.words(), ["hathaway"])

    def test_the_file_is_plain_text_you_can_edit(self):
        self.pw.add("AFO")
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#"))
        self.path.write_text(text + "Khoa Nhi\n# a note\n\n", encoding="utf-8")
        self.assertEqual(PersonalWords(self.path).words(), ["AFO", "Khoa Nhi"])

    def test_hint_is_newest_first_and_capped(self):
        for i in range(300):
            self.pw.add(f"word{i:03d}")
        self.assertEqual(len(self.pw.words()), personal_words.MAX_WORDS)
        hint = self.pw.hint()
        self.assertTrue(hint.startswith("word299, word298"))
        self.assertLessEqual(len(hint), personal_words.MAX_HINT_CHARS)

    def test_a_broken_file_never_stops_dictation(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"\xff\xfe\x00bad")
        self.assertEqual(self.pw.words(), [])

    def test_combine(self):
        self.assertIsNone(combine("", "  "))
        self.assertEqual(combine("Hathaway, AFO", "pause yes no"), "Hathaway, AFO, pause yes no")


class RecordingModel(FakeWhisperModel):
    def __init__(self):
        self.hotwords = []

    def transcribe(self, audio, **kwargs):
        self.hotwords.append(kwargs.get("hotwords"))
        return super().transcribe(audio, **kwargs)


class TestWhisperGetsTheWords(TempList):
    def test_each_utterance_reads_the_list(self):
        model = RecordingModel()
        d = Dictation(SimpleNamespace(whisper_hotwords="pause"), model=model, personal_words=self.pw)
        d.transcribe("hello")
        self.pw.add("Hathaway")
        d.transcribe("hello")
        self.assertEqual(model.hotwords, ["pause", "Hathaway, pause"])

    def test_tests_and_evaluation_never_read_the_real_list(self):
        self.assertIsNone(Dictation(model=FakeWhisperModel()).personal_words)
        self.assertIsNone(Dictation(SimpleNamespace(whisper_hotwords=""), model=FakeWhisperModel()).personal_words)
        self.assertIsNone(Dictation(SimpleNamespace(whisper_hotwords=""), hotwords="x").personal_words)

    def test_the_real_list_lives_with_the_voice_data(self):
        from engine.voice import voice_data
        self.assertEqual(personal_words.WORDS_FILE.parent, voice_data.PERSONAL_DIR)   # "delete my voice data" removes it


class TestParseWordList(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_word_list("Add Hathaway to my words."), WordListChange("add", "Hathaway"))
        self.assertEqual(parse_word_list("add the word prednisone to my dictionary please"),
                         WordListChange("add", "prednisone"))
        self.assertEqual(parse_word_list("Add H-A-T-H-A-W-A-Y to my word list."), WordListChange("add", "hathaway"))
        self.assertEqual(parse_word_list("add that to my words"), WordListChange("add", ""))
        self.assertEqual(parse_word_list("Remove Hathaway from my words."), WordListChange("remove", "Hathaway"))
        self.assertEqual(parse_word_list("add Khoa Nhi to my vocabulary"), WordListChange("add", "Khoa Nhi"))

    def test_not_a_word_list_change(self):
        for text in ["add milk to my list", "Add it to my cart.", "add to my words", "add words",
                     "add a lot of very long words here to my words"]:
            self.assertIsNone(parse_word_list(text), text)


class TestByVoice(TempList):
    def setUp(self):
        super().setUp()
        self.bus = FakeBus()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, log=lambda *_: None)
        self.mm.personal_words = self.pw

    def say(self, text):
        self.worker.process(text)

    def statuses(self):
        return [m["text"] for m in self.bus.ui(messages.STATUS)]

    def test_add_and_remove_by_voice(self):
        self.say("add Hathaway to my words")
        self.assertEqual(self.pw.words(), ["Hathaway"])
        self.assertIn("Added to your words", [m["label"] for m in self.bus.ui(messages.COMMAND)])
        self.say("remove Hathaway from my words")
        self.assertEqual(self.pw.words(), [])

    def test_add_that_after_spelling(self):
        self.say("start dictation")
        self.say("Call Doctor")
        self.say("spell capital H A T H A W A Y")
        self.say("add that to my words")
        self.assertEqual(self.pw.words(), ["Hathaway"])
        previews = self.bus.ui(messages.DICTATION_PREVIEW)
        self.assertEqual(" ".join(w["text"] for w in previews[-1]["words"]), "Call Doctor Hathaway")  # card unchanged

    def test_add_that_after_a_change(self):
        self.say("start dictation")
        self.say("Take the prednizone tonight.")
        self.say("change prednizone to P R E D N I S O N E")
        self.say("add that to my words")
        self.assertEqual(self.pw.words(), ["prednisone"])

    def test_add_that_with_nothing_spelled_asks_for_a_spelling(self):
        self.say("add that to my words")
        self.assertEqual(self.pw.words(), [])
        self.assertTrue(any("Spell the word first" in s for s in self.statuses()))

    def test_ordinary_dictation_is_not_a_word_list_change(self):
        self.say("start dictation")
        self.say("Add milk to my list.")
        self.assertEqual(self.pw.words(), [])
        previews = self.bus.ui(messages.DICTATION_PREVIEW)
        self.assertEqual(" ".join(w["text"] for w in previews[-1]["words"]), "Add milk to my list.")


if __name__ == "__main__":
    unittest.main()
