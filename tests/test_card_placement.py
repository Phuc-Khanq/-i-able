"""Run with: python -m unittest tests.test_card_placement

Khang, 2026-11-22: a blink never types the dictation card (it only clicks); "yes" with no text box focused keeps
the card and asks the user to click one, then types once the click put the focus in a text box; a card waits for
"yes" or "no" however long the user is quiet, with one reminder after CARD_REMIND_S.
"""
import unittest

from engine.config import Settings
from engine.core import messages, topics
from engine.core.mode_manager import CARD_REMIND_S, PLACE_CHECK_S, ModeManager
from engine.core.types import GazePoint, Gesture
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Focus:
    """text_focus stand-in: what the focus is now, and how often it was asked."""

    def __init__(self, value):
        self.value, self.asked = value, 0

    def __call__(self):
        self.asked += 1
        return self.value


class Base(unittest.TestCase):
    def setUp(self):
        self.bus, self.clock = FakeBus(), Clock()
        self.worker = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()))
        self.mm = ModeManager(self.bus, clock=self.clock)
        self.bus.publish(topics.GAZE, GazePoint(t_ms=0, x=500, y=300, confidence=0.9))

    def say(self, text):
        self.worker.process(text)

    def gesture(self, kind):
        self.bus.publish(topics.GESTURE, Gesture(t_ms=0, type=kind))

    def typed(self):
        return [p["text"] for t, p in self.bus.log if t == topics.OS_TYPE]

    def clicks(self):
        return [p for t, p in self.bus.log if t == topics.OS_CLICK]

    def statuses(self):
        return [m["text"] for m in self.bus.ui("status")]

    def later(self, seconds):
        self.clock.t += seconds
        self.mm.tick()


class TestBlink(Base):
    def test_a_blink_clicks_and_never_types_the_card(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.gesture("blink")
        self.assertEqual(self.typed(), [])
        self.assertEqual(self.clicks(), [{"x": 500, "y": 300}])
        self.assertEqual(self.mm._preview_text(), "see you at noon")

    def test_a_nod_still_types_it(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.gesture("nod")
        self.assertEqual(self.typed(), ["see you at noon "])


class TestWhereToType(Base):
    def test_yes_types_at_once_in_a_text_box(self):
        self.mm.text_focus = Focus(True)
        self.say("start dictation")
        self.say("see you at noon")
        self.say("yes")
        self.assertEqual(self.typed(), ["see you at noon "])

    def test_no_text_box_asks_and_the_blink_click_on_one_types_it(self):
        focus = self.mm.text_focus = Focus(False)
        self.say("start dictation")
        self.say("see you at noon")
        self.say("yes")
        self.assertEqual(self.typed(), [])                         # nothing typed into nowhere
        self.assertTrue(any("Where should the text go" in s for s in self.statuses()))
        self.assertEqual(self.mm._preview_text(), "see you at noon")
        self.gesture("blink")                                     # clicks somewhere that still isn't a text box
        self.later(PLACE_CHECK_S)
        self.assertEqual(self.typed(), [])
        self.assertTrue(any("isn't a text box" in s for s in self.statuses()))
        focus.value = True
        self.gesture("blink")                                     # now onto the text box
        self.assertEqual(self.typed(), [])                         # not before the app moved its focus
        self.later(PLACE_CHECK_S)
        self.assertEqual(self.typed(), ["see you at noon "])
        self.assertEqual(len(self.clicks()), 2)

    def test_yes_again_after_clicking_some_other_way(self):
        focus = self.mm.text_focus = Focus(False)
        self.say("start dictation")
        self.say("see you at noon")
        self.say("yes")
        focus.value = True                                        # e.g. a dwell or voice click
        self.say("yes")
        self.assertEqual(self.typed(), ["see you at noon "])

    def test_no_drops_it_while_waiting(self):
        self.mm.text_focus = Focus(False)
        self.say("start dictation")
        self.say("see you at noon")
        self.say("yes")
        self.say("no")
        self.assertEqual(self.mm._preview_text(), "")
        self.mm.text_focus.value = True
        self.gesture("blink")
        self.later(PLACE_CHECK_S)
        self.assertEqual(self.typed(), [])

    def test_cant_tell_types_as_before(self):
        self.mm.text_focus = Focus(None)
        self.say("start dictation")
        self.say("see you at noon")
        self.say("yes")
        self.assertEqual(self.typed(), ["see you at noon "])

    def test_a_failing_check_types_as_before(self):
        def broken():
            raise OSError("UI Automation is not available")
        self.mm.text_focus = broken
        self.say("start dictation")
        self.say("see you at noon")
        self.say("yes")
        self.assertEqual(self.typed(), ["see you at noon "])


class TestPatience(Base):
    def test_the_card_waits_however_long_with_one_reminder(self):
        self.say("start dictation")
        self.say("see you at noon")
        self.later(CARD_REMIND_S - 1)
        self.assertFalse(any("still on the card" in s for s in self.statuses()))
        self.later(2)
        self.later(600)                                           # ten more minutes
        self.assertEqual(sum("still on the card" in s for s in self.statuses()), 1)
        self.assertEqual(self.typed(), [])
        self.assertEqual(self.mm._preview_text(), "see you at noon")
        self.say("and bring water")                               # still collecting after a long quiet
        self.say("yes")
        self.assertEqual(self.typed(), ["see you at noon and bring water "])

    def test_the_default_pause_is_patient(self):
        self.assertEqual(Settings().pause_seconds, 2.5)
        self.assertTrue(Settings().ask_where_to_type)


if __name__ == "__main__":
    unittest.main()
