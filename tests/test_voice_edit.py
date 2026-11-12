"""Run with: python -m unittest tests.test_voice_edit

Fixing words on the dictation card by voice: "change Hawthorne to Hathaway", "delete Hawthorne",
"delete the last word", and "scratch that" taking back one step at a time.
"""
import unittest

from engine.core import messages, topics
from engine.core.mode_manager import ModeManager
from engine.voice import commands
from engine.voice.commands import Edit, apply_edit, drop_last_word, find_span, parse_edit
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


def W(text):
    return [{"text": t, "conf": 0.95} for t in text.split()]


def T(words):
    return " ".join(w["text"] for w in words)


class TestParseEdit(unittest.TestCase):
    def test_change_and_replace(self):
        self.assertEqual(parse_edit("Change Hawthorne to Hathaway."), Edit("replace", "hawthorne", "Hathaway"))
        self.assertEqual(parse_edit("replace the word their with there"), Edit("replace", "their", "there"))
        self.assertEqual(parse_edit("Correct fifteen to fifty."), Edit("replace", "fifteen", "fifty"))
        self.assertEqual(parse_edit("change new york into New Jersey"), Edit("replace", "new york", "New Jersey"))

    def test_fillers_and_please(self):
        self.assertEqual(parse_edit("Um, can you change Monday to Tuesday, please?"),
                         Edit("replace", "monday", "Tuesday"))

    def test_delete(self):
        self.assertEqual(parse_edit("Delete Hawthorne."), Edit("delete", "hawthorne"))
        self.assertEqual(parse_edit("remove the word really"), Edit("delete", "really"))

    def test_not_an_edit(self):
        for text in ["delete that", "Remove it.", "change", "change to", "Change the oil. Then call me.",
                     "change my flight to a much later one on Sunday evening please if possible",
                     "fix the bug with the login", "I'll change it to Friday", "send it"]:
            self.assertIsNone(parse_edit(text), text)


class TestFindAndApply(unittest.TestCase):
    def test_fuzzy_finds_a_misheard_word(self):
        self.assertEqual(find_span(["Call", "Mr.", "Hawthorn", "tomorrow."], "hawthorne"), (2, 3))

    def test_short_words_match_exactly(self):
        self.assertIsNone(find_span(["They", "went", "home."], "the"))
        self.assertEqual(find_span(["See", "the", "doctor."], "the"), (1, 2))

    def test_the_last_one_wins(self):
        words = W("the cat sat and the cat ran")
        self.assertEqual(T(apply_edit(words, Edit("replace", "cat", "dog"))), "the cat sat and the dog ran")

    def test_two_words_against_one(self):
        self.assertEqual(find_span("my health care plan".split(), "healthcare"), (1, 3))
        self.assertEqual(T(apply_edit(W("my health care plan"), Edit("replace", "healthcare", "dental"))),
                         "my dental plan")

    def test_punctuation_and_capitals_stay(self):
        self.assertEqual(T(apply_edit(W("See you Monday."), Edit("replace", "monday", "Tuesday"))), "See you Tuesday.")
        self.assertEqual(T(apply_edit(W("Hawthorne is here."), Edit("replace", "hawthorne", "hathaway"))),
                         "Hathaway is here.")
        new = apply_edit(W("See you Monday."), Edit("replace", "monday", "Tuesday"))
        self.assertEqual(new[-1]["conf"], 1.0)

    def test_delete_keeps_the_sentence_tidy(self):
        self.assertEqual(T(apply_edit(W("Hello Hawthorne, how are you."), Edit("delete", "hawthorne"))),
                         "Hello, how are you.")
        self.assertEqual(T(apply_edit(W("Hawthorne said hi."), Edit("delete", "hawthorne"))), "Said hi.")
        self.assertEqual(T(apply_edit(W("I will see you Monday."), Edit("delete", "monday"))), "I will see you.")

    def test_not_there(self):
        self.assertIsNone(apply_edit(W("I need a new car."), Edit("replace", "oil", "synthetic")))

    def test_drop_last_word(self):
        self.assertEqual(T(drop_last_word(W("See you Monday."))), "See you.")
        self.assertEqual(drop_last_word(W("Hi")), [])


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    WAIT = None          # the clean-up off: the card shows the user's own words

    def setUp(self):
        self.bus, self.clock = FakeBus(), Clock()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), settings=None)
        self.mm = ModeManager(self.bus, clock=self.clock, rewrite_wait_s=self.WAIT, log=lambda *_: None)
        self.say("start dictation")

    def say(self, text):
        self.worker.process(text)

    def shown(self):
        previews = self.bus.ui(messages.DICTATION_PREVIEW)
        return T(previews[-1]["words"]) if previews else ""

    def typed(self):
        return [p["text"] for p in self.bus.of(topics.OS_TYPE)]

    def flashes(self):
        return [m["label"] for m in self.bus.ui(messages.COMMAND)]


class TestEditInDictation(Base):
    def test_change_a_word_then_type_it(self):
        self.say("Call Mr. Hawthorne tomorrow.")
        self.say("Change Hawthorne to Hathaway.")
        self.assertEqual(self.shown(), "Call Mr. Hathaway tomorrow.")
        self.assertIn("Changed", self.flashes())
        self.say("yes")
        self.assertEqual(self.typed(), ["Call Mr. Hathaway tomorrow. "])

    def test_delete_a_word(self):
        self.say("I really really want to go.")
        self.say("Delete really.")
        self.assertEqual(self.shown(), "I really want to go.")
        self.assertIn("Deleted", self.flashes())

    def test_old_words_not_on_the_card_is_dictation(self):
        self.say("I need a new car.")
        self.say("Change the oil to synthetic.")
        self.assertEqual(self.shown(), "I need a new car. Change the oil to synthetic.")

    def test_trigger_word_always_edits(self):
        self.say("I need a new car.")
        self.say("Venus, change dog to cat.")
        self.assertEqual(self.shown(), "I need a new car.")
        self.assertTrue(any("Couldn't find" in m["text"] for m in self.bus.ui(messages.STATUS)))
        self.say("Venus, change car to bike.")
        self.assertEqual(self.shown(), "I need a new bike.")

    def test_scratch_that_takes_back_one_step_at_a_time(self):
        self.say("Meet me at noon.")
        self.say("Bring the blue folder.")
        self.say("Change blue to red.")
        self.assertEqual(self.shown(), "Meet me at noon. Bring the red folder.")
        self.say("scratch that")
        self.assertEqual(self.shown(), "Meet me at noon. Bring the blue folder.")
        self.say("scratch that")
        self.assertEqual(self.shown(), "Meet me at noon.")
        self.say("scratch that")
        self.assertEqual(self.mm.preview, [])
        self.assertEqual(self.bus.of(topics.OS_UNDO), [])     # never Ctrl+Z in the app while a preview waited

    def test_no_still_drops_everything(self):
        self.say("Meet me at noon.")
        self.say("Bring the blue folder.")
        self.say("no")
        self.assertEqual(self.mm.preview, [])
        self.assertEqual(self.typed(), [])

    def test_backspace_and_delete_last_word_fix_the_preview(self):
        self.say("See you on Monday.")
        self.say("backspace")
        self.assertEqual(self.shown(), "See you on.")
        self.say("delete the last word")
        self.assertEqual(self.shown(), "See you.")
        self.assertEqual(self.bus.of(topics.OS_KEY), [])      # nothing pressed in the app behind the card

    def test_delete_last_word_with_no_preview_presses_ctrl_backspace(self):
        self.say("delete the last word")
        self.assertEqual(self.bus.of(topics.OS_KEY), [{"key": "ctrl+backspace"}])

    def test_fixed_commands_still_win(self):
        self.say("My voice data is private.")
        self.say("delete my voice data")
        self.assertEqual(self.shown(), "My voice data is private.")
        self.assertIsNotNone(self.mm.question)

    def test_no_preview_no_edit(self):
        self.say("Change Hawthorne to Hathaway.")
        self.assertEqual(self.shown(), "Change Hawthorne to Hathaway.")   # dictated as it was said


class TestEditWithCleanUp(Base):
    WAIT = 1.5

    def requests(self):
        return self.bus.of(topics.REWRITE_REQUEST)

    def answer(self, clean):
        req = self.requests()[-1]
        self.bus.publish(topics.REWRITE_RESULT, {"id": req["id"], "clean": clean, "used": True, "ms": 300.0})

    def test_edit_the_clean_text_and_it_stays_fixed(self):
        self.say("um call mister hawthorne tomorrow")
        self.answer("Call Mr. Hawthorne tomorrow.")
        self.assertEqual(self.shown(), "Call Mr. Hawthorne tomorrow.")
        n = len(self.requests())
        self.say("Change Hawthorne to Hathaway.")
        self.assertEqual(self.shown(), "Call Mr. Hathaway tomorrow.")
        self.assertEqual(len(self.requests()), n)            # a fix is final: no new clean-up
        self.say("yes")
        self.assertEqual(self.typed(), ["Call Mr. Hathaway tomorrow. "])

    def test_scratch_that_after_a_fix_brings_back_the_clean_up(self):
        self.say("um call mister hawthorne tomorrow")
        self.answer("Call Mr. Hawthorne tomorrow.")
        self.say("Change Hawthorne to Hathaway.")
        n = len(self.requests())
        self.say("scratch that")
        self.assertEqual(len(self.requests()), n + 1)        # the user's own words are cleaned up again
        self.answer("Call Mr. Hawthorne tomorrow.")
        self.assertEqual(self.shown(), "Call Mr. Hawthorne tomorrow.")


class TestSpelled(unittest.TestCase):
    def test_ways_whisper_writes_spelling(self):
        for said in ["H A T H A W A Y", "H-A-T-H-A-W-A-Y.", "H. A. T. H. A. W. A. Y.", "HATHAWAY", "h, a, t, h, a, w, a, y"]:
            self.assertEqual(commands.spelled(said.split()), "hathaway", said)

    def test_letter_names_capitals_and_doubles(self):
        self.assertEqual(commands.spelled("capital M c capital D o n a l d".split()), "McDonald")
        self.assertEqual(commands.spelled("bee ee double ell".split()), "bell")
        self.assertEqual(commands.spelled("double you H Y".split()), "why")
        self.assertEqual(commands.spelled("E dash M A I L".split()), "e-mail")
        self.assertEqual(commands.spelled("P R E D N I S O N E".split()), "prednisone")

    def test_not_spelling(self):
        for said in ["check", "it out", "the word", "you are a star"]:
            self.assertIsNone(commands.spelled(said.split()), said)

    def test_strict_needs_single_letters(self):
        self.assertEqual(commands.spelled("T H E R E.".split(), strict=True), "there")
        self.assertEqual(commands.spelled(["T-H-E-R-E"], strict=True), "there")
        for said in ["you are", "U.S.", "A", "a car", "HATHAWAY", "bee ee"]:
            self.assertIsNone(commands.spelled(said.split(), strict=True), said)


class TestParseSpell(unittest.TestCase):
    def test_spell(self):
        self.assertEqual(commands.parse_spell("Spell H-A-T-H-A-W-A-Y."), "hathaway")
        self.assertEqual(commands.parse_spell("um, spell out A F O please"), "afo")
        self.assertEqual(commands.parse_spell("Spell Hathaway, H-A-T-H-A-W-A-Y."), "hathaway")   # the letters win

    def test_not_spell(self):
        for text in ["spell", "Spell check is on.", "spell check", "I can't spell it", "spelling bee tonight"]:
            self.assertIsNone(commands.parse_spell(text), text)

    def test_change_to_spelled(self):
        self.assertEqual(parse_edit("Change their to T H E R E."), Edit("replace", "their", "there"))
        self.assertEqual(parse_edit("change Hawthorn to spell H A T H A W A Y"), Edit("replace", "hawthorn", "hathaway"))
        self.assertEqual(parse_edit("change pregnazone to P-R-E-D-N-I-S-O-N-E"),
                         Edit("replace", "pregnazone", "prednisone"))
        self.assertEqual(parse_edit("change it to you are"), Edit("replace", "it", "you are"))   # words, not letters
        self.assertIsNone(parse_edit("change Hawthorn to spell the new name"))


class TestSpellInDictation(Base):
    def test_spell_adds_the_word(self):
        self.say("Please call Doctor")
        self.say("Spell H-A-T-H-A-W-A-Y.")
        self.assertEqual(self.shown(), "Please call Doctor hathaway")
        self.assertIn("Spelled", self.flashes())
        self.say("scratch that")
        self.assertEqual(self.shown(), "Please call Doctor")

    def test_capital_and_sentence_start(self):
        self.say("Spell all caps A F O.")
        self.assertEqual(self.shown(), "AFO")
        self.say("I need a new one.")
        self.say("spell A F O")
        self.assertEqual(self.shown(), "AFO I need a new one. Afo")

    def test_change_to_spelled_keeps_the_capital(self):
        self.say("Call Mr. Hawthorn tomorrow.")
        self.say("Change Hawthorn to H A T H A W A Y.")
        self.assertEqual(self.shown(), "Call Mr. Hathaway tomorrow.")
        self.say("yes")
        self.assertEqual(self.typed(), ["Call Mr. Hathaway tomorrow. "])

    def test_spell_check_is_dictation(self):
        self.say("Spell check is on.")
        self.assertEqual(self.shown(), "Spell check is on.")

    def test_spell_in_pointer_mode_starts_a_preview(self):
        self.say("stop dictation")
        self.say("spell H A T H A W A Y")
        self.assertEqual(self.shown(), "Hathaway")
        self.say("yes")
        self.assertEqual(self.typed(), ["Hathaway "])


class TestCommandList(unittest.TestCase):
    def test_new_phrases_are_one_command(self):
        for p in ["delete last word", "delete the last word", "remove the last word"]:
            self.assertEqual(commands.parse(p).name, "delete_last_word")
        self.assertEqual(commands.parse("backspace").name, "key_backspace")


if __name__ == "__main__":
    unittest.main()
