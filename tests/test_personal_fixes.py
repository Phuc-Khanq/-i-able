"""Run with: python -m unittest tests.test_personal_fixes

Learning from the user's fixes (Wispr Flow's "never make the same mistake twice"): a fix made by voice and typed is
counted, made automatically from the second time, and forgotten when the user changes it back; spelled words and names
put in by a fix join the word list.
"""
import tempfile
import unittest
from pathlib import Path

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.voice import personal_fixes
from engine.voice.dictation import Dictation, DictationWorker
from engine.voice.personal_fixes import PersonalFixes
from engine.voice.personal_words import PersonalWords, worth_learning
from tests.helpers import FakeBus, FakeWhisperModel


def card(text):
    return [{"text": t, "conf": 0.9} for t in text.split()]


def text_of(words):
    return " ".join(w["text"] for w in words)


class Temp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.fixes = PersonalFixes(Path(self.dir.name) / "personal" / "fixes.txt")
        self.words = PersonalWords(Path(self.dir.name) / "personal" / "words.txt")

    def tearDown(self):
        self.dir.cleanup()


class TestPersonalFixes(Temp):
    def test_counted_then_automatic(self):
        self.assertEqual(self.fixes.record("waiter", "water").times, 1)
        self.assertEqual(self.fixes.apply(card("A glass of waiter."))[1], [])        # once: not yet
        self.assertEqual(self.fixes.record("Waiter", "water").times, 2)
        words, used = self.fixes.apply(card("Waiter please, a waiter."))
        self.assertEqual(text_of(words), "Water please, a water.")                   # capital and punctuation kept
        self.assertEqual([f.meant for f in used], ["water", "water"])
        self.assertEqual(words[0]["conf"], 1.0)

    def test_changing_it_back_forgets_it(self):
        self.fixes.record("waiter", "water")
        self.fixes.record("waiter", "water")
        self.assertIsNone(self.fixes.record("water", "waiter"))
        self.assertEqual(self.fixes.fixes(), [])

    def test_a_different_fix_for_the_same_words_starts_again(self):
        self.fixes.record("bank", "bench")
        self.fixes.record("bank", "bench")
        self.assertEqual(self.fixes.record("bank", "bang").times, 1)
        self.assertEqual(self.fixes.apply(card("the bank"))[1], [])

    def test_several_words(self):
        for _ in range(2):
            self.fixes.record("hat away", "Hathaway")
        words, _ = self.fixes.apply(card("Call doctor hat away."))
        self.assertEqual(text_of(words), "Call doctor Hathaway.")

    def test_unusable_pairs_and_forget(self):
        self.assertIsNone(self.fixes.record("same", "Same."))
        self.assertIsNone(self.fixes.record("", "x"))
        self.assertIsNone(self.fixes.record("one two three four five", "x"))
        self.fixes.record("waiter", "water")
        self.assertTrue(self.fixes.forget("WAITER"))
        self.assertFalse(self.fixes.forget("waiter"))

    def test_a_broken_file_never_stops_dictation(self):
        self.fixes.path.parent.mkdir(parents=True)
        self.fixes.path.write_bytes(b"\xff\xfe\x00bad")
        self.assertEqual(self.fixes.apply(card("hello")), (card("hello"), []))

    def test_the_file_lives_with_the_voice_data(self):
        from engine.voice import voice_data
        self.assertEqual(personal_fixes.FIXES_FILE.parent, voice_data.PERSONAL_DIR)


class TestWorthLearning(unittest.TestCase):
    def test_names_terms_and_spelled_words(self):
        self.assertTrue(worth_learning("Hathaway"))
        self.assertTrue(worth_learning("AFO"))
        self.assertTrue(worth_learning("B12"))
        self.assertTrue(worth_learning("prednisone", spelled=True))
        self.assertFalse(worth_learning("prednisone"))          # said, lowercase: Whisper wrote it already
        self.assertFalse(worth_learning("there", spelled=True))  # a their/there fix
        self.assertFalse(worth_learning("I"))
        self.assertFalse(worth_learning("x", spelled=True))


class TestByVoice(Temp):
    def setUp(self):
        super().setUp()
        self.bus = FakeBus()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, log=lambda *_: None)
        self.mm.personal_words, self.mm.personal_fixes, self.mm.learning = self.words, self.fixes, True
        self.say("start dictation")

    def say(self, text):
        self.worker.process(text)

    def shown(self):
        return text_of(self.bus.ui(messages.DICTATION_PREVIEW)[-1]["words"])

    def typed(self):
        return [p["text"] for p in self.bus.of(topics.OS_TYPE)]

    def statuses(self):
        return [m["text"] for m in self.bus.ui(messages.STATUS)]

    def fix_and_type(self):
        self.say("Bring me a waiter please.")
        self.say("change waiter to water")
        self.say("yes")

    def test_the_second_fix_is_learned_and_used(self):
        self.fix_and_type()
        self.assertEqual(self.typed()[-1].strip(), "Bring me a water please.")
        self.assertEqual(self.fixes.fixes()[0].times, 1)
        self.fix_and_type()
        self.assertTrue(any("waiter → water from now on" in s for s in self.statuses()))
        self.say("A glass of waiter.")
        self.assertEqual(self.shown(), "A glass of water.")
        self.assertTrue(any("Used your fix" in s for s in self.statuses()))

    def test_a_dropped_card_teaches_nothing(self):
        self.say("Bring me a waiter please.")
        self.say("change waiter to water")
        self.say("no")
        self.assertEqual(self.fixes.fixes(), [])

    def test_a_spelled_word_joins_the_word_list_when_typed(self):
        self.say("Call Doctor")
        self.say("spell capital H A T H A W A Y")
        self.assertEqual(self.words.words(), [])                      # not until it's typed
        self.say("yes")
        self.assertEqual(self.words.words(), ["Hathaway"])
        self.assertTrue(any('"Hathaway" in your words' in s for s in self.statuses()))

    def test_a_name_put_in_by_a_fix_joins_the_word_list(self):
        self.say("Call doctor Hawthorne.")
        self.say("change Hawthorne to Hathaway")
        self.say("yes")
        self.assertEqual(self.words.words(), ["Hathaway"])

    def test_off_unless_the_app_turns_it_on(self):
        mm = ModeManager(FakeBus(), log=lambda *_: None)
        self.assertFalse(mm.learning)


if __name__ == "__main__":
    unittest.main()
