"""Run with: python -m unittest tests.test_snippets

Snippets (Wispr Flow's idea, made by voice): "save this as a snippet called my address" saves the card's text,
"insert my address" (or the cue said alone while dictating) puts it back on the card.
"""
import tempfile
import unittest
from pathlib import Path

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.voice import snippets
from engine.voice.commands import SnippetRequest, parse_snippet
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.snippets import Snippets
from tests.helpers import FakeBus, FakeWhisperModel


class TestParse(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_snippet("Save this as a snippet called my address."),
                         SnippetRequest("save", "my address"))
        self.assertEqual(parse_snippet("make that a shortcut named sign off"), SnippetRequest("save", "sign off"))
        self.assertEqual(parse_snippet("save a snippet called email"), SnippetRequest("save", "email"))
        self.assertEqual(parse_snippet("Insert my address."), SnippetRequest("insert", "my address"))
        self.assertEqual(parse_snippet("insert the snippet my address"), SnippetRequest("insert", "my address"))
        self.assertEqual(parse_snippet("Delete the snippet my address."), SnippetRequest("delete", "my address"))
        self.assertEqual(parse_snippet("forget snippet home"), SnippetRequest("delete", "home"))

    def test_not_snippets(self):
        for text in ["Save this as a PDF.", "save it", "insert it", "paste", "delete the last word",
                     "Delete Hawthorne.", "save this as a snippet called one two three four five six"]:
            self.assertIsNone(parse_snippet(text), text)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.s = Snippets(Path(self.dir.name) / "personal" / "snippets.txt")

    def tearDown(self):
        self.dir.cleanup()

    def test_save_get_replace_remove(self):
        self.assertEqual(self.s.save("My Address.", "12 Oak Street, Gainesville"), "my address")
        self.assertEqual(self.s.get("my address"), "12 Oak Street, Gainesville")
        self.s.save("my address", "14 Elm Road")
        self.assertEqual(len(self.s.all()), 1)
        self.assertEqual(self.s.get("MY ADDRESS"), "14 Elm Road")
        self.assertTrue(self.s.remove("my address"))
        self.assertFalse(self.s.remove("my address"))
        self.assertIsNone(self.s.save("", "x"))

    def test_the_file_lives_with_the_voice_data(self):
        from engine.voice import voice_data
        self.assertEqual(snippets.SNIPPETS_FILE.parent, voice_data.PERSONAL_DIR)


class TestByVoice(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.bus = FakeBus()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, log=lambda *_: None)
        self.mm.snippets = Snippets(Path(self.dir.name) / "snippets.txt")
        self.say("start dictation")

    def tearDown(self):
        self.dir.cleanup()

    def say(self, text):
        self.worker.process(text)

    def shown(self):
        return " ".join(w["text"] for w in self.bus.ui(messages.DICTATION_PREVIEW)[-1]["words"])

    def test_save_then_insert_then_type(self):
        self.say("12 Oak Street, Gainesville, Florida")
        self.say("save this as a snippet called my address")
        self.assertEqual(self.mm.snippets.get("my address"), "12 Oak Street, Gainesville, Florida")
        self.say("yes")
        self.say("Deliver it to")
        self.say("insert my address")
        self.assertEqual(self.shown(), "Deliver it to 12 Oak Street, Gainesville, Florida")
        self.say("yes")
        self.assertEqual(self.bus.of(topics.OS_TYPE)[-1]["text"].strip(),
                         "Deliver it to 12 Oak Street, Gainesville, Florida")

    def test_the_cue_alone_while_dictating(self):
        self.mm.snippets.save("sign off", "Best wishes, Khang")
        self.say("sign off")
        self.assertEqual(self.shown(), "Best wishes, Khang")

    def test_ordinary_dictation_is_left_alone(self):
        self.say("Insert the key into the lock.")
        self.assertEqual(self.shown(), "Insert the key into the lock.")
        self.say("Save this as a PDF.")
        self.assertEqual(self.shown(), "Insert the key into the lock. Save this as a PDF.")

    def test_a_command_cant_be_a_snippet_name(self):
        self.say("See you soon")
        self.say("save this as a snippet called send it")
        self.assertEqual(self.mm.snippets.all(), {})
        self.assertTrue(any("is a command" in m["text"] for m in self.bus.ui(messages.STATUS)))

    def test_save_with_nothing_on_the_card(self):
        self.say("save this as a snippet called home")
        self.assertEqual(self.mm.snippets.all(), {})
        self.assertTrue(any("Dictate the text first" in m["text"] for m in self.bus.ui(messages.STATUS)))


if __name__ == "__main__":
    unittest.main()
